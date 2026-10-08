// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include "sim/coralnpu_m3_user_decoder.h"
#include "sim/coralnpu_v2_state.h"
#include "sim/hw_sim/coralnpu_simulator.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace {

const uint32_t kAddrMailbox = 0x401fc000;  // user-configurable

// The mcause of the CoralNPU usage fault (custom cause 24 + 1), which the RTL
// raises on ebreak.
constexpr uint32_t kMcauseUsageFault = 25;

// DMA Controller Constants
constexpr uint32_t kDmaBase = 0x40050000;
constexpr uint32_t kDmaRegsSize = 0x14;  // 5 registers, 4 bytes each

// Register Offsets
constexpr uint32_t kDmaCtrlOffset = 0x00;
constexpr uint32_t kDmaStatusOffset = 0x04;
constexpr uint32_t kDmaDescAddrOffset = 0x08;
constexpr uint32_t kDmaCurDescOffset = 0x0c;
constexpr uint32_t kDmaXferRemainOffset = 0x10;

// Control Register Bits
constexpr uint32_t kDmaCtrlEnable = 0x1;
constexpr uint32_t kDmaCtrlStart = 0x2;
[[maybe_unused]] constexpr uint32_t kDmaCtrlAbort = 0x4;

// Status Register Bits
constexpr uint32_t kDmaStatusBusy = 0x1;
constexpr uint32_t kDmaStatusDone = 0x2;
[[maybe_unused]] constexpr uint32_t kDmaStatusError = 0x4;

// The callbacks set with SetTraceCallback and SetMemoryAccessCallback.
struct ObserverState {
  TraceCallback trace_callback;
  bool trace_disasm = false;
  MemoryAccessCallback memory_access_callback;
  // True while the core runs, so that host ReadMem/WriteMem aren't reported.
  bool core_running = false;
  // True while the decoder fetches an instruction.
  bool decoding = false;

  // Returns whether to report data accesses: only those of the running core
  // (not instruction fetches), and only if there is a callback.
  bool ReportsMemory() const {
    return core_running && !decoding && memory_access_callback != nullptr;
  }
};

class DmaMemoryWrapper : public ::mpact::sim::util::MemoryInterface {
 public:
  DmaMemoryWrapper(::mpact::sim::util::MemoryInterface* parent,
                   const ObserverState* observer_state)
      : parent_(parent), observer_state_(observer_state) {}

  ~DmaMemoryWrapper() override = default;

  void Load(uint64_t address, ::mpact::sim::generic::DataBuffer* db,
            ::mpact::sim::generic::Instruction* inst,
            ::mpact::sim::generic::ReferenceCount* context) override {
    Report(address, db->size<uint8_t>(), /*is_store=*/false);
    if (IsDmaAddress(address)) {
      HandleDmaRead(address, db);
      if (inst != nullptr) {
        inst->Execute(context);
      }
      return;
    }
    parent_->Load(address, db, inst, context);
  }

  void Store(uint64_t address, ::mpact::sim::generic::DataBuffer* db) override {
    Report(address, db->size<uint8_t>(), /*is_store=*/true);
    if (IsDmaAddress(address)) {
      HandleDmaWrite(address, db);
      return;
    }
    parent_->Store(address, db);
  }

  void Load(::mpact::sim::generic::DataBuffer* address_db,
            ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
            ::mpact::sim::generic::DataBuffer* db,
            ::mpact::sim::generic::Instruction* inst,
            ::mpact::sim::generic::ReferenceCount* context) override {
    ReportVector(address_db, mask_db, el_size, /*is_store=*/false);
    parent_->Load(address_db, mask_db, el_size, db, inst, context);
  }

  void Store(::mpact::sim::generic::DataBuffer* address_db,
             ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
             ::mpact::sim::generic::DataBuffer* db) override {
    ReportVector(address_db, mask_db, el_size, /*is_store=*/true);
    parent_->Store(address_db, mask_db, el_size, db);
  }

 private:
  // Reports a data access of the core to the memory access callback.
  void Report(uint64_t address, int size, bool is_store) {
    // Report only the core's own accesses, and only if there is a callback.
    if (!observer_state_->ReportsMemory()) return;
    observer_state_->memory_access_callback(
        static_cast<uint32_t>(address), static_cast<uint32_t>(size), is_store);
  }

  // Reports each active element of a vector access as a data access.
  void ReportVector(::mpact::sim::generic::DataBuffer* address_db,
                    ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
                    bool is_store) {
    // Report only the core's own accesses, and only if there is a callback.
    if (!observer_state_->ReportsMemory()) return;
    int num_elements = address_db->size<uint64_t>();
    for (int i = 0; i < num_elements; ++i) {
      // Inactive elements don't access memory.
      if (mask_db != nullptr && !mask_db->Get<bool>(i)) continue;
      Report(address_db->Get<uint64_t>(i), el_size, is_store);
    }
  }

  bool IsDmaAddress(uint64_t addr) {
    return addr >= kDmaBase && addr < kDmaBase + kDmaRegsSize;
  }

  void HandleDmaRead(uint64_t address, ::mpact::sim::generic::DataBuffer* db) {
    uint32_t offset = address - kDmaBase;
    uint32_t val = 0;
    switch (offset) {
      case kDmaCtrlOffset:
        val = dma_ctrl_;
        break;
      case kDmaStatusOffset:
        val = dma_status_;
        break;
      case kDmaDescAddrOffset:
        val = dma_desc_addr_;
        break;
      case kDmaCurDescOffset:
        val = dma_cur_desc_;
        break;
      case kDmaXferRemainOffset:
        val = dma_xfer_remain_;
        break;
    }
    if (db->size<uint8_t>() == 4) {
      db->Set<uint32_t>(0, val);
    }
  }

  void HandleDmaWrite(uint64_t address, ::mpact::sim::generic::DataBuffer* db) {
    uint32_t offset = address - kDmaBase;
    uint32_t val = 0;
    if (db->size<uint8_t>() == 4) {
      val = db->Get<uint32_t>(0);
    } else {
      return;
    }

    switch (offset) {
      case kDmaCtrlOffset:
        dma_ctrl_ = val;
        if ((dma_ctrl_ & (kDmaCtrlEnable | kDmaCtrlStart)) ==
            (kDmaCtrlEnable | kDmaCtrlStart)) {
          RunDma();
        }
        break;
      case kDmaDescAddrOffset:
        dma_desc_addr_ = val;
        break;
    }
  }

  void RunDma() {
    dma_status_ |= kDmaStatusBusy;
    dma_status_ &= ~kDmaStatusDone;

    uint32_t desc_addr = dma_desc_addr_;
    ::mpact::sim::generic::DataBufferFactory db_factory;

    while (desc_addr != 0) {
      dma_cur_desc_ = desc_addr;

      // Read descriptor (32 bytes)
      auto* desc_db = db_factory.Allocate(32);
      parent_->Load(desc_addr, desc_db, nullptr, nullptr);

      uint32_t src_addr = desc_db->Get<uint32_t>(0);
      uint32_t dst_addr = desc_db->Get<uint32_t>(1);
      uint32_t len_flags = desc_db->Get<uint32_t>(2);
      uint32_t next_desc = desc_db->Get<uint32_t>(3);

      uint32_t xfer_len = len_flags & 0x00FFFFFFu;

      if (xfer_len > 0) {
        auto* data_db = db_factory.Allocate(xfer_len);
        parent_->Load(src_addr, data_db, nullptr, nullptr);
        parent_->Store(dst_addr, data_db);
        data_db->DecRef();
      }

      desc_db->DecRef();
      desc_addr = next_desc;
    }

    dma_status_ &= ~kDmaStatusBusy;
    dma_status_ |= kDmaStatusDone;
  }

  ::mpact::sim::util::MemoryInterface* parent_;
  const ObserverState* observer_state_;
  uint32_t dma_ctrl_ = 0;
  uint32_t dma_status_ = 0;
  uint32_t dma_desc_addr_ = 0;
  uint32_t dma_cur_desc_ = 0;
  uint32_t dma_xfer_remain_ = 0;
};

// Wraps a decoded instruction to call the trace callback before executing it.
// Instruction has no getter for its semantic function, so the wrapper owns the
// decoded instruction and executes it from its own semantic function.
class ObservedInstruction : public ::mpact::sim::generic::Instruction {
 public:
  ObservedInstruction(::mpact::sim::generic::Instruction* inst,
                      uint32_t encoding, const ObserverState* observer_state)
      : Instruction(inst->address(), inst->state()),
        inst_(inst),
        encoding_(encoding),
        observer_state_(observer_state) {
    set_opcode(inst->opcode());
    set_size(inst->size());
    set_semantic_function(
        [this](::mpact::sim::generic::Instruction*) { ExecuteObserved(); });
  }

  ~ObservedInstruction() override { inst_->DecRef(); }

  std::string AsString() const override { return inst_->AsString(); }

 private:
  void ExecuteObserved() {
    // The callback may have been removed since the instruction was decoded.
    if (observer_state_->trace_callback) {
      std::string disassembly;
      if (observer_state_->trace_disasm) disassembly = inst_->AsString();
      observer_state_->trace_callback(static_cast<uint32_t>(address()),
                                      encoding_, disassembly);
    }
    inst_->Execute(context());
  }

  ::mpact::sim::generic::Instruction* inst_;
  uint32_t encoding_;
  const ObserverState* observer_state_;
};

// Forwards to the CoralNPU decoder. Marks instruction fetches, so that they
// aren't reported as data accesses, and wraps the decoded instructions in
// ObservedInstruction while a trace callback is set.
class ObservingDecoder : public ::mpact::sim::generic::DecoderInterface {
 public:
  ObservingDecoder(::mpact::sim::generic::DecoderInterface* decoder,
                   ::mpact::sim::util::MemoryInterface* memory,
                   ObserverState* observer_state)
      : decoder_(decoder),
        memory_(memory),
        observer_state_(observer_state),
        encoding_db_(db_factory_.Allocate<uint32_t>(1)) {}

  ~ObservingDecoder() override { encoding_db_->DecRef(); }

  ::mpact::sim::generic::Instruction* DecodeInstruction(
      uint64_t address) override {
    observer_state_->decoding = true;
    ::mpact::sim::generic::Instruction* inst =
        decoder_->DecodeInstruction(address);
    observer_state_->decoding = false;
    // Without a trace callback, the instruction runs without the wrapper.
    if (inst == nullptr || !observer_state_->trace_callback) return inst;
    memory_->Load(address, encoding_db_, nullptr, nullptr);
    uint32_t encoding = encoding_db_->Get<uint32_t>(0);
    // Compressed instructions are 16 bits wide.
    if (inst->size() == 2) encoding &= 0xffff;
    return new ObservedInstruction(inst, encoding, observer_state_);
  }

  int GetNumOpcodes() const override { return decoder_->GetNumOpcodes(); }

  const char* GetOpcodeName(int index) const override {
    return decoder_->GetOpcodeName(index);
  }

 private:
  ::mpact::sim::generic::DecoderInterface* decoder_;
  ::mpact::sim::util::MemoryInterface* memory_;
  ObserverState* observer_state_;
  ::mpact::sim::generic::DataBufferFactory db_factory_;
  ::mpact::sim::generic::DataBuffer* encoding_db_;
};

class MpactSimulator final : public CoralNPUSimulator {
 public:
  MpactSimulator()
      : memory_(),
        dma_memory_(&memory_, &observer_state_),
        rv_state_("RiscV32GV", mpact::sim::riscv::RiscVXlen::RV32,
                  &dma_memory_),
        rv_fp_state_(rv_state_.csr_set(), &rv_state_),
        rvv_state_(
            &rv_state_,
            /*byte_length=*/::coralnpu::sim::kCoralNPUV2VectorByteLength),
        rv_decoder_(&rv_state_, &dma_memory_),
        observing_decoder_(&rv_decoder_, &memory_, &observer_state_),
        rv_top_("CoralNPUPlaceholder", &rv_state_, &observing_decoder_) {
    // Make sure the architectural and abi register aliases are added.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, i);
      (void)rv_state_.AddRegister<::mpact::sim::riscv::RV32Register>(reg_name);
      (void)rv_state_.AddRegisterAlias<::mpact::sim::riscv::RV32Register>(
          reg_name, mpact::sim::riscv::kXRegisterAliases[i]);

      reg_name = absl::StrCat(mpact::sim::riscv::RiscVState::kFregPrefix, i);
      (void)rv_state_.AddRegister<::mpact::sim::riscv::RVFpRegister>(reg_name);
      (void)rv_state_.AddRegisterAlias<::mpact::sim::riscv::RVFpRegister>(
          reg_name, mpact::sim::riscv::kFRegisterAliases[i]);

      reg_name = absl::StrCat(mpact::sim::riscv::RiscVState::kVregPrefix, i);
      (void)rv_state_.AddRegister<::mpact::sim::riscv::RVVectorRegister>(
          reg_name, /*width=*/::coralnpu::sim::kCoralNPUV2VectorByteLength);
    }
    rv_state_.set_rv_fp(&rv_fp_state_);
    rv_state_.set_rv_vector(&rvv_state_);

    // Configure the MISA (Machine ISA Register) with 0x40201120:
    // - Bit 30 (MXLEN = 32): 32-bit register width.
    // - Bit 21 (V): Enable RISC-V Vector extension.
    // - Bit 12 (M): Enable Integer Multiply/Divide extension.
    // - Bit 8  (I): Enable Base Integer ISA.
    // - Bit 5  (F): Enable Single-Precision Floating-Point extension.
    rv_state_.misa()->Set(static_cast<uint32_t>(0x40201120));

    // Register a full-range memory region with Read/Write/Execute permissions.
    // The custom CoralNPUV2State enforces memory permission checks for all
    // decoded load/store instructions. Since this simulator wrapper maps memory
    // dynamically without registering explicit regions (using
    // FlatDemandMemory), we must allow all accesses to prevent permission fault
    // traps on execution.
    rv_state_.AddMemoryRegion(
        0, 0xFFFFFFFF, ::coralnpu::sim::MemoryPermission::kReadWriteExecute);

    // Register handler for the custom 'mpause' instruction.
    // When the firmware terminates successfully, it executes 'mpause'. The
    // handler intercepts this and requests the simulator core to halt.
    rv_state_.AddMpauseHandler([this](const ::mpact::sim::generic::Instruction*
                                          inst) {
      halted_ = true;
      rv_top_.RequestHalt(
          ::mpact::sim::generic::CoreDebugInterface::HaltReason::kUserRequest,
          inst);
      return true;
    });

    // Register handler for 'ebreak' instructions.
    // When the firmware encounters an assertion failure or crash, it executes
    // 'ebreak'. Halting the core here prevents the simulator from hanging in
    // the failure idle loop.
    rv_state_.AddEbreakHandler([this](const ::mpact::sim::generic::Instruction*
                                          inst) {
      uint32_t mcause = rv_state_.mcause() ? rv_state_.mcause()->AsUint32() : 0;
      uint32_t mepc = rv_state_.mepc() ? rv_state_.mepc()->AsUint32() : 0;
      uint32_t mtval = rv_state_.mtval() ? rv_state_.mtval()->AsUint32() : 0;
      uint32_t vtype = 0;
      auto vtype_csr = rv_state_.csr_set()->GetCsr("vtype");
      if (vtype_csr.ok()) {
        vtype = vtype_csr.value()->AsUint32();
      }
      uint32_t vl = 0;
      auto vl_csr = rv_state_.csr_set()->GetCsr("vl");
      if (vl_csr.ok()) {
        vl = vl_csr.value()->AsUint32();
      }
      uint32_t frm = 0;
      auto frm_csr = rv_state_.csr_set()->GetCsr("frm");
      if (frm_csr.ok()) {
        frm = frm_csr.value()->AsUint32();
      }
      std::string mepc_disasm = "unknown";
      auto mepc_inst = rv_top_.GetInstruction(mepc);
      if (mepc_inst.ok()) {
        mepc_disasm = mepc_inst.value()->AsString();
        mepc_inst.value()->DecRef();
      }
      LOG(INFO) << "Simulator: ebreak hit at 0x" << std::hex << inst->address()
                << ", mcause=0x" << mcause << ", mepc=0x" << mepc << " ("
                << mepc_disasm << ")"
                << ", mtval=0x" << mtval << ", vtype=0x" << vtype << ", vl=0x"
                << vl << ", frm=0x" << frm << std::dec;
      // As on the RTL, ebreak is a usage fault: it sets mcause and mtval (not
      // mepc) and halts the core with the fault bit set.
      if (rv_state_.mcause()) rv_state_.mcause()->Set(kMcauseUsageFault);
      if (rv_state_.mtval()) {
        rv_state_.mtval()->Set(static_cast<uint32_t>(inst->address()));
      }
      halted_ = true;
      fault_ = true;
      rv_top_.RequestHalt(
          ::mpact::sim::generic::CoreDebugInterface::HaltReason::kUserRequest,
          inst);
      return true;
    });

    // Register handler for WFI (Wait For Interrupt) instructions.
    // The original simulator loop intercepted WFI (0x10500073) to halt.
    // RiscVState provides a built-in 'on_wfi' callback when executing WFI.
    // The simulation stops, but as on the RTL, the core isn't halted: it waits
    // for an interrupt.
    rv_state_.set_on_wfi([this](
                             const ::mpact::sim::generic::Instruction* inst) {
      rv_top_.RequestHalt(
          ::mpact::sim::generic::CoreDebugInterface::HaltReason::kUserRequest,
          inst);
      return true;
    });
  }
  ~MpactSimulator() final = default;

  void ReadMem(uint32_t addr, size_t size, char* data) final;
  const CoralNPUMailbox& ReadMailbox() final;
  void WriteMem(uint32_t addr, size_t size, const char* data) final;
  void WriteMailbox(const CoralNPUMailbox& mailbox) final;
  void Run(uint32_t start_addr) final;
  bool WaitForTermination(int timeout) final;
  uint64_t GetCycleCount() const final;
  bool SetTraceCallback(TraceCallback callback, bool disasm) final;
  bool SetMemoryAccessCallback(MemoryAccessCallback callback) final;
  bool ReadCoreState(CoralNPUCoreState* state) final;

 private:
  CoralNPUMailbox mailbox_;
  ObserverState observer_state_;
  ::mpact::sim::util::FlatDemandMemory memory_;
  DmaMemoryWrapper dma_memory_;
  ::coralnpu::sim::CoralNPUV2State rv_state_;
  ::mpact::sim::riscv::RiscVFPState rv_fp_state_;
  ::mpact::sim::riscv::RiscVVectorState rvv_state_;
  ::coralnpu::sim::CoralNPUM3UserDecoder rv_decoder_;
  ObservingDecoder observing_decoder_;
  ::mpact::sim::riscv::RiscVTop rv_top_;
  // The status of the core, as in the status register of the RTL: mpause
  // halts the core, ebreak halts it on a fault and wfi doesn't halt it.
  bool halted_ = false;
  bool fault_ = false;
  // Whether the last run failed, so that the state of the core is unknown.
  bool run_failed_ = false;
};

void MpactSimulator::ReadMem(uint32_t addr, size_t size, char* data) {
  auto result = rv_top_.ReadMemory(addr, data, size);
  if (!result.ok()) {
    LOG(ERROR) << "Error: " << result.status();
  }
  assert(result.ok());
}

const CoralNPUMailbox& MpactSimulator::ReadMailbox() {
  auto result = rv_top_.ReadMemory(
      kAddrMailbox, reinterpret_cast<char*>(mailbox_.message), 16);
  if (!result.ok()) {
    LOG(ERROR) << "Error: " << result.status();
  }
  assert(result.ok());
  return mailbox_;
}

void MpactSimulator::WriteMem(uint32_t addr, size_t size, const char* data) {
  auto result = rv_top_.WriteMemory(addr, data, size);
  if (!result.ok()) {
    std::cerr << "Error: " << result.status() << std::endl;
  }
  assert(result.ok());
}

void MpactSimulator::WriteMailbox(const CoralNPUMailbox& mailbox) {
  for (int i = 0; i < 4; i++) {
    mailbox_.message[i] = mailbox.message[i];
  }

  this->WriteMem(kAddrMailbox, 16,
                 reinterpret_cast<const char*>(mailbox.message));
}

void MpactSimulator::Run(uint32_t start_addr) {
  // The core status of the previous run doesn't apply to this one.
  halted_ = false;
  fault_ = false;
  run_failed_ = false;
  absl::Status pc_write = rv_top_.WriteRegister("pc", start_addr);
  assert(pc_write.ok());
}

bool MpactSimulator::WaitForTermination(int timeout) {
  observer_state_.core_running = true;
  auto status = rv_top_.Run();
  if (!status.ok()) {
    observer_state_.core_running = false;
    run_failed_ = true;
    LOG(ERROR) << "Simulator run failed: " << status.message();
    return false;
  }

  status = rv_top_.Wait();
  observer_state_.core_running = false;
  if (!status.ok()) {
    run_failed_ = true;
    LOG(ERROR) << "Simulator wait failed: " << status.message();
    return false;
  }

  this->ReadMem(kAddrMailbox, 16, reinterpret_cast<char*>(mailbox_.message));

  return true;
}

uint64_t MpactSimulator::GetCycleCount() const {
  return const_cast<::mpact::sim::riscv::RiscVTop&>(rv_top_)
      .counter_num_cycles()
      ->GetValue();
}

bool MpactSimulator::SetTraceCallback(TraceCallback callback, bool disasm) {
  observer_state_.trace_callback = std::move(callback);
  observer_state_.trace_disasm = disasm;
  return true;
}

bool MpactSimulator::SetMemoryAccessCallback(MemoryAccessCallback callback) {
  observer_state_.memory_access_callback = std::move(callback);
  return true;
}

bool MpactSimulator::ReadCoreState(CoralNPUCoreState* state) {
  // After a failed run, the state of the core is unknown.
  if (run_failed_) return false;
  auto pc = rv_top_.ReadRegister("pc");
  auto minstret = rv_state_.csr_set()->GetCsr("minstret");
  auto minstreth = rv_state_.csr_set()->GetCsr("minstreth");
  // Without the pc and minstret, the state is incomplete.
  if (!pc.ok() || !minstret.ok() || !minstreth.ok()) return false;
  state->halted = halted_;
  state->fault = fault_;
  state->pc = static_cast<uint32_t>(*pc);
  state->mepc = rv_state_.mepc() ? rv_state_.mepc()->AsUint32() : 0;
  state->mtval = rv_state_.mtval() ? rv_state_.mtval()->AsUint32() : 0;
  state->mcause = rv_state_.mcause() ? rv_state_.mcause()->AsUint32() : 0;
  // The CSR, as the firmware reads it: unlike the instruction counter of the
  // simulator, it counts from the last write by the firmware.
  state->minstret = (static_cast<uint64_t>((*minstreth)->AsUint32()) << 32) |
                    (*minstret)->AsUint32();
  return true;
}

}  // namespace

extern "C" CoralNPUSimulator* coralnpu_simulator_mpact_create(void) {
  return new MpactSimulator();
}
