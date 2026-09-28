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
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace {

const uint32_t kAddrMailbox = 0x401fc000;  // user-configurable

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

class DmaMemoryWrapper : public ::mpact::sim::util::MemoryInterface {
 public:
  explicit DmaMemoryWrapper(::mpact::sim::util::MemoryInterface* parent)
      : parent_(parent) {}

  ~DmaMemoryWrapper() override = default;

  void Load(uint64_t address, ::mpact::sim::generic::DataBuffer* db,
            ::mpact::sim::generic::Instruction* inst,
            ::mpact::sim::generic::ReferenceCount* context) override {
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
    parent_->Load(address_db, mask_db, el_size, db, inst, context);
  }

  void Store(::mpact::sim::generic::DataBuffer* address_db,
             ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
             ::mpact::sim::generic::DataBuffer* db) override {
    parent_->Store(address_db, mask_db, el_size, db);
  }

 private:
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
  uint32_t dma_ctrl_ = 0;
  uint32_t dma_status_ = 0;
  uint32_t dma_desc_addr_ = 0;
  uint32_t dma_cur_desc_ = 0;
  uint32_t dma_xfer_remain_ = 0;
};

class MpactSimulator final : public CoralNPUSimulator {
 public:
  MpactSimulator()
      : memory_(),
        dma_memory_(&memory_),
        rv_state_("RiscV32GV", mpact::sim::riscv::RiscVXlen::RV32,
                  &dma_memory_),
        rv_fp_state_(rv_state_.csr_set(), &rv_state_),
        rvv_state_(
            &rv_state_,
            /*byte_length=*/::coralnpu::sim::kCoralNPUV2VectorByteLength),
        rv_decoder_(&rv_state_, &dma_memory_),
        rv_top_("CoralNPUPlaceholder", &rv_state_, &rv_decoder_) {
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
      rv_top_.RequestHalt(
          ::mpact::sim::generic::CoreDebugInterface::HaltReason::kUserRequest,
          inst);
      return true;
    });

    // Register handler for WFI (Wait For Interrupt) instructions.
    // The original simulator loop intercepted WFI (0x10500073) to halt.
    // RiscVState provides a built-in 'on_wfi' callback when executing WFI.
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
  void SetTraceCallback(TraceCallback callback, bool disasm) final;

 private:
  CoralNPUMailbox mailbox_;
  ::mpact::sim::util::FlatDemandMemory memory_;
  DmaMemoryWrapper dma_memory_;
  ::coralnpu::sim::CoralNPUV2State rv_state_;
  ::mpact::sim::riscv::RiscVFPState rv_fp_state_;
  ::mpact::sim::riscv::RiscVVectorState rvv_state_;
  ::coralnpu::sim::CoralNPUM3UserDecoder rv_decoder_;
  ::mpact::sim::riscv::RiscVTop rv_top_;
  TraceCallback trace_callback_;
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
  absl::Status pc_write = rv_top_.WriteRegister("pc", start_addr);
  assert(pc_write.ok());
}

bool MpactSimulator::WaitForTermination(int timeout) {
  auto status = rv_top_.Run();
  if (!status.ok()) {
    LOG(ERROR) << "Simulator run failed: " << status.message();
    return false;
  }

  status = rv_top_.Wait();
  if (!status.ok()) {
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

void MpactSimulator::SetTraceCallback(TraceCallback callback, bool disasm) {
  trace_callback_ = std::move(callback);
}

}  // namespace

extern "C" CoralNPUSimulator* coralnpu_simulator_mpact_create(void) {
  return new MpactSimulator();
}
