#include "sim/kelvin_top.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>  // NOLINT(build/c++11): built with c++17
#include <utility>

#include "sim/decoder.h"
#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "absl/flags/flag.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_breakpoint.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decode_cache.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

ABSL_FLAG(bool, use_semihost, false, "Use semihost in the simulation");

namespace kelvin::sim {

constexpr char kKelvinName[] = "Kelvin";

// Local helper function used to execute instructions.
static inline bool ExecuteInstruction(mpact::sim::util::Instruction *inst) {
  for (auto *resource : inst->ResourceHold()) {
    if (!resource->IsFree()) {
      return false;
    }
  }
  for (auto *resource : inst->ResourceAcquire()) {
    resource->Acquire();
  }
  // Comment out instruction logging during execution.
  // LOG(INFO) << "[" << std::hex << inst->address() << "] " <<
  // inst->AsString();

  inst->Execute(nullptr);
  return true;
}

KelvinTop::KelvinTop(std::string name)
    : Component{std::move(name)},
      counter_num_instructions_{"num_instructions", 0},
      counter_num_cycles_{"num_cycles", 0} {
  // Using a single flat memory for this core.
  memory_ = new mpact::sim::util::FlatDemandMemory(0);
  Initialize();
}

KelvinTop::~KelvinTop() {
  // If the simulator is still running, request a halt (set halted_ to true),
  // and wait until the simulator finishes before continuing the destructor.
  if (run_status_ == RunStatus::kRunning) {
    run_halted_->WaitForNotification();
    delete run_halted_;
  }

  delete rv_bp_manager_;
  delete decode_cache_;
  delete kelvin_decoder_;
  delete state_;
  delete fp_state_;
  delete watcher_;
  delete memory_;
  delete semihost_;
}

void KelvinTop::Initialize() {
  // Create the simulation state
  state_ = new sim::KelvinState(kKelvinName, mpact::sim::riscv::RiscVXlen::RV32,
                                memory_);
  state_->set_max_physical_address(kKelvinMaxMemoryAddress);
  fp_state_ = new mpact::sim::riscv::RiscVFPState(state_);
  state_->set_rv_fp(fp_state_);
  pc_ = state_->registers()->at(sim::KelvinState::kPcName);
  // Set up the decoder and decode cache.
  kelvin_decoder_ = new sim::KelvinDecoder(state_, memory_);
  for (int i = 0; i < static_cast<int>(isa32::OpcodeEnum::kPastMaxValue); i++) {
    counter_opcode_[i].Initialize(absl::StrCat("num_", isa32::kOpcodeNames[i]),
                                  0);
    CHECK_OK(AddCounter(&counter_opcode_[i]));
  }
  decode_cache_ =
      mpact::sim::generic::DecodeCache::Create({16 * 1024, 2}, kelvin_decoder_);
  CHECK(decode_cache_) << "Failed to create decode cache";
  // Register instruction counter.
  CHECK_OK(AddCounter(&counter_num_instructions_))
      << "Failed to register counter";

  // Always return 4-byte breakpoint instruction size
  rv_bp_manager_ = new mpact::sim::riscv::RiscVBreakpointManager(
      memory_,
      absl::bind_front(&mpact::sim::generic::DecodeCache::Invalidate,
                       decode_cache_),
      [](uint64_t, uint32_t) -> int { return 4; });
  // Make sure the architectural and abi register aliases are added.
  std::string reg_name;
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(sim::KelvinState::kXregPrefix, i);
    (void)state_->AddRegister<mpact::sim::riscv::RV32Register>(reg_name);
    (void)state_->AddRegisterAlias<mpact::sim::riscv::RV32Register>(
        reg_name, mpact::sim::riscv::kXRegisterAliases[i]);
  }

  semihost_ = new mpact::sim::riscv::RiscVArmSemihost(
      mpact::sim::riscv::RiscVArmSemihost::BitWidth::kWord32, memory_, memory_);
  // Set the software breakpoint callback.
  state_->AddEbreakHandler(
      [this](const mpact::sim::generic::Instruction *inst) -> bool {
        if (inst != nullptr) {
          if (absl::GetFlag(FLAGS_use_semihost) &&
              semihost_->IsSemihostingCall(inst)) {
            semihost_->OnEBreak(inst);
          } else if (rv_bp_manager_->HasBreakpoint(
                         inst->address())) {  // Software breakpoint.
            RequestHalt(HaltReason::kSoftwareBreakpoint, inst);
          } else {  // The default Kelvin simulation mode.
            std::cout << "Program exits with fault" << std::endl;
            RequestHalt(HaltReason::kUserRequest, inst);
          }
          return true;
        }
        return false;
      });

  state_->AddMpauseHandler(
      [this](const mpact::sim::generic::Instruction *inst) -> bool {
        if (inst != nullptr) {
          std::cout << "Program exits properly" << std::endl;
          RequestHalt(HaltReason::kUserRequest, inst);
          return true;
        }
        return false;
      });

  // Set trap callbacks.
  state_->set_on_trap([this](bool is_interrupt, uint64_t trap_value,
                             uint64_t exception_code, uint64_t epc,
                             const Instruction *inst) -> bool {
    auto code = static_cast<mpact::sim::riscv::ExceptionCode>(exception_code);
    bool result = false;
    switch (code) {
      case mpact::sim::riscv::ExceptionCode::kIllegalInstruction: {
        std::cerr << "Illegal instruction at 0x" << std::hex << epc
                  << std::endl;
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      case mpact::sim::riscv::ExceptionCode::kLoadAccessFault: {
        std::cerr << "Memory load access fault at 0x" << std::hex << epc
                  << " as: " << inst->AsString() << std::endl;
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      case mpact::sim::riscv::ExceptionCode::kStoreAccessFault: {
        std::cerr << "Memory store access fault at 0x" << std::hex << epc
                  << " as: " << inst->AsString() << std::endl;
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      default:
        break;
    }
    return result;
  });

  semihost_->set_exit_callback(
      [this]() { RequestHalt(HaltReason::kSemihostHaltRequest, nullptr); });
}

absl::Status KelvinTop::Halt() {
  // If it is already halted, just return.
  if (run_status_ == RunStatus::kHalted) {
    return absl::OkStatus();
  }
  // If it is not running, then there's an error.
  if (run_status_ != RunStatus::kRunning) {
    return absl::FailedPreconditionError(
        "KelvinTop::Halt: Core is not running");
  }
  halt_reason_ = HaltReason::kUserRequest;
  halted_ = true;
  return absl::OkStatus();
}

absl::Status KelvinTop::StepPastBreakpoint() {
  uint64_t pc = state_->pc_operand()->AsUint64(0);
  uint64_t bpt_pc = pc;
  // Disable the breakpoint. Status will show error if there is no breakpoint.
  auto status = rv_bp_manager_->DisableBreakpoint(pc);
  // Execute the real instruction.
  auto real_inst = decode_cache_->GetDecodedInstruction(pc);
  real_inst->IncRef();
  auto next_seq_pc = pc + real_inst->size();
  SetPc(next_seq_pc);
  bool executed = false;
  do {
    executed = ExecuteInstruction(real_inst);
    counter_num_cycles_.Increment(1);
    state_->AdvanceDelayLines();
  } while (!executed);
  // Increment counter.
  counter_opcode_[real_inst->opcode()].Increment(1);
  counter_num_instructions_.Increment(1);
  real_inst->DecRef();
  // Re-enable the breakpoint.
  if (status.ok()) {
    status = rv_bp_manager_->EnableBreakpoint(bpt_pc);
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::StatusOr<int> KelvinTop::Step(int num) {
  if (num <= 0) {
    return absl::InvalidArgumentError("Step count must be > 0");
  }
  // If the simulator is running, return with an error.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "KelvinTop::Step: Core must be halted");
  }
  run_status_ = RunStatus::kSingleStep;
  int count = 0;
  halted_ = false;
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (halt_reason_ == HaltReason::kSoftwareBreakpoint) {
    halt_reason_ = HaltReason::kNone;
    auto status = StepPastBreakpoint();
    if (!status.ok()) return status;
    count++;
  }

  // Step the simulator forward until the number of steps have been achieved, or
  // there is a halt request.
  auto pc_operand = state_->pc_operand();
  // This holds the value of the current pc, and post-loop, the address of
  // the most recently executed instruction.
  uint64_t pc;
  // At the top of the loop this holds the address of the instruction to be
  // executed next. Post-loop it holds the address of the next instruction to
  // be executed.
  uint64_t next_pc = pc_operand->AsUint64(0);
  uint64_t next_seq_pc;
  while (!halted_ && (count < num)) {
    pc = next_pc;
    auto *inst = decode_cache_->GetDecodedInstruction(pc);
    next_seq_pc = pc + inst->size();
    // Set the PC destination operand to next_seq_pc. Any branch that is
    // executed will overwrite this.
    SetPc(next_seq_pc);
    bool executed = false;
    do {
      executed = ExecuteInstruction(inst);
      counter_num_cycles_.Increment(1);
      state_->AdvanceDelayLines();
    } while (!executed);
    count++;
    // Update counters.
    counter_opcode_[inst->opcode()].Increment(1);
    counter_num_instructions_.Increment(1);
    // Get the next pc value.
    next_pc = pc_operand->AsUint64(0);
  }
  // Update the pc register, now that it can be read.
  if (halt_reason_ == HaltReason::kSoftwareBreakpoint) {
    // If at a breakpoint, keep the pc at the current value.
    SetPc(pc);
  } else {
    // Otherwise set it to point to the next instruction.
    SetPc(next_pc);
  }
  // If there is no halt request, there is no specific halt reason.
  if (!halted_) {
    halt_reason_ = HaltReason::kNone;
  }
  run_status_ = RunStatus::kHalted;
  return count;
}

absl::Status KelvinTop::Run() {
  // Verify that the core isn't running already.
  if (run_status_ == RunStatus::kRunning) {
    return absl::FailedPreconditionError(
        "KelvinTop::Run: core is already running");
  }
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (halt_reason_ == HaltReason::kSoftwareBreakpoint) {
    halt_reason_ = HaltReason::kNone;
    auto status = StepPastBreakpoint();
    if (!status.ok()) return status;
  }
  run_status_ = RunStatus::kRunning;
  halted_ = false;

  // The simulator is now run in a separate thread so as to allow a user
  // interface to continue operating. Allocate a new run_halted_ Notification
  // object, as they are single used only.
  run_halted_ = new absl::Notification();
  // The thread is detached so it executes without having to be joined.
  std::thread([this]() {
    auto pc_operand = state_->pc_operand();
    // This holds the value of the current pc, and post-loop, the address of
    // the most recently executed instruction.
    uint64_t pc;
    // At the top of the loop this holds the address of the instruction to be
    // executed next. Post-loop it holds the address of the next instruction to
    // be executed.
    uint64_t next_pc = pc_operand->AsUint64(0);
    uint64_t next_seq_pc;
    while (!halted_) {
      pc = next_pc;
      auto *inst = decode_cache_->GetDecodedInstruction(pc);
      next_seq_pc = pc + inst->size();
      // Set the PC destination operand to next_seq_pc. Any branch that is
      // executed will overwrite this.
      SetPc(next_seq_pc);
      bool executed = false;
      do {
        executed = ExecuteInstruction(inst);
        counter_num_cycles_.Increment(1);
        state_->AdvanceDelayLines();
      } while (!executed);
      // Update counters.
      counter_opcode_[inst->opcode()].Increment(1);
      counter_num_instructions_.Increment(1);
      // Get the next pc value.
      next_pc = pc_operand->AsUint64(0);
    }
    // Update the pc register, now that it can be read (since we are not
    // running).
    if (halt_reason_ == HaltReason::kSoftwareBreakpoint) {
      // If at a breakpoint, keep the pc at the current value.
      SetPc(pc);
    } else {
      // Otherwise set it to point to the next instruction.
      SetPc(next_pc);
    }
    run_status_ = RunStatus::kHalted;
    // Notify that the run has completed.
    run_halted_->Notify();
  }).detach();
  return absl::OkStatus();
}

absl::Status KelvinTop::Wait() {
  // If the simulator isn't running, then just return.
  if (run_status_ != RunStatus::kRunning) return absl::OkStatus();

  // Wait for the simulator to finish (i.e., a value is available on the
  // channel).
  run_halted_->WaitForNotification();
  delete run_halted_;
  run_halted_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<KelvinTop::RunStatus> KelvinTop::GetRunStatus() {
  return run_status_;
}

absl::StatusOr<KelvinTop::HaltReason> KelvinTop::GetLastHaltReason() {
  return halt_reason_;
}

absl::StatusOr<uint64_t> KelvinTop::ReadRegister(const std::string &name) {
  // The registers aren't protected by a mutex, so let's not read them while
  // the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("ReadRegister: Core must be halted");
  }
  auto iter = state_->registers()->find(name);

  // Was the register found? If not try CSRs.
  if (iter == state_->registers()->end()) {
    auto result = state_->csr_set()->GetCsr(name);
    if (!result.ok()) {
      return absl::NotFoundError(
          absl::StrCat("Register '", name, "' not found"));
    }
    auto *csr = *result;
    return csr->GetUint32();
  }

  auto *db = (iter->second)->data_buffer();
  uint64_t value;
  switch (db->size<uint8_t>()) {
    case 1:
      value = static_cast<uint64_t>(db->Get<uint8_t>(0));
      break;
    case 2:
      value = static_cast<uint64_t>(db->Get<uint16_t>(0));
      break;
    case 4:
      value = static_cast<uint64_t>(db->Get<uint32_t>(0));
      break;
    case 8:
      value = static_cast<uint64_t>(db->Get<uint64_t>(0));
      break;
    default:
      return absl::InternalError("Register size is not 1, 2, 4, or 8 bytes");
  }
  return value;
}

absl::Status KelvinTop::WriteRegister(const std::string &name, uint64_t value) {
  // The registers aren't protected by a mutex, so let's not write them while
  // the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("WriteRegister: Core must be halted");
  }
  auto iter = state_->registers()->find(name);
  // Was the register found? If not try CSRs.
  if (iter == state_->registers()->end()) {
    auto result = state_->csr_set()->GetCsr(name);
    if (!result.ok()) {
      return absl::NotFoundError(
          absl::StrCat("Register '", name, "' not found"));
    }
    auto *csr = *result;
    csr->Set(static_cast<uint32_t>(value));
    return absl::OkStatus();
  }

  // If stopped at a software breakpoint and the pc is changed, change the
  // halt reason, since the next instruction won't be where we stopped.
  if ((name == "pc") && (halt_reason_ == HaltReason::kSoftwareBreakpoint)) {
    halt_reason_ = HaltReason::kNone;
  }

  auto *db = (iter->second)->data_buffer();
  switch (db->size<uint8_t>()) {
    case 1:
      db->Set<uint8_t>(0, static_cast<uint8_t>(value));
      break;
    case 2:
      db->Set<uint16_t>(0, static_cast<uint16_t>(value));
      break;
    case 4:
      db->Set<uint32_t>(0, static_cast<uint32_t>(value));
      break;
    case 8:
      db->Set<uint64_t>(0, static_cast<uint64_t>(value));
      break;
    default:
      return absl::InternalError("Register size is not 1, 2, 4, or 8 bytes");
  }
  return absl::OkStatus();
}

absl::StatusOr<DataBuffer *> KelvinTop::GetRegisterDataBuffer(
    const std::string &name) {
  // The registers aren't protected by a mutex, so let's not access them while
  // the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "GetRegisterDataBuffer: Core must be halted");
  }
  auto iter = state_->registers()->find(name);
  if (iter == state_->registers()->end()) {
    return absl::NotFoundError(absl::StrCat("Register '", name, "' not found"));
  }
  return iter->second->data_buffer();
}

absl::StatusOr<size_t> KelvinTop::ReadMemory(uint64_t address, void *buffer,
                                             size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("ReadMemory: Core must be halted");
  }
  auto *db = db_factory_.Allocate(length);
  // Load bypassing any watch points/semihosting.
  state_->memory()->Load(address, db, nullptr, nullptr);
  std::memcpy(buffer, db->raw_ptr(), length);
  db->DecRef();
  return length;
}

absl::StatusOr<size_t> KelvinTop::WriteMemory(uint64_t address,
                                              const void *buffer,
                                              size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("WriteMemory: Core must be halted");
  }
  auto *db = db_factory_.Allocate(length);
  std::memcpy(db->raw_ptr(), buffer, length);
  // Store bypassing any watch points/semihosting.
  state_->memory()->Store(address, db);
  db->DecRef();
  return length;
}

bool KelvinTop::HasBreakpoint(uint64_t address) {
  return rv_bp_manager_->HasBreakpoint(address);
}

absl::Status KelvinTop::SetSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "SetSwBreakpoint: Core must be halted");
  }
  // If there is no breakpoint manager, return an error.
  if (rv_bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  // Try setting the breakpoint.
  return rv_bp_manager_->SetBreakpoint(address);
}

absl::Status KelvinTop::ClearSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "ClearSwBreakpoing: Core must be halted");
  }
  if (rv_bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  return rv_bp_manager_->ClearBreakpoint(address);
}

absl::Status KelvinTop::ClearAllSwBreakpoints() {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "ClearAllSwBreakpoints: Core must be halted");
  }
  if (rv_bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  rv_bp_manager_->ClearAllBreakpoints();
  return absl::OkStatus();
}

absl::StatusOr<mpact::sim::generic::Instruction *> KelvinTop::GetInstruction(
    uint64_t address) {
  auto inst = decode_cache_->GetDecodedInstruction(address);
  return inst;
}

absl::StatusOr<std::string> KelvinTop::GetDisassembly(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("GetDissasembly: Core must be halted");
  }

  mpact::sim::generic::Instruction *inst = nullptr;
  // If requesting the disassembly for an instruction at a breakpoint, return
  // that of the original instruction instead.
  if (rv_bp_manager_->IsBreakpoint(address)) {
    auto bp_pc = address;
    // Disable the breakpoint.
    auto status = rv_bp_manager_->DisableBreakpoint(bp_pc);
    if (!status.ok()) return status;
    // Get the real instruction.
    inst = decode_cache_->GetDecodedInstruction(bp_pc);
    auto disasm = inst != nullptr ? inst->AsString() : "Invalid instruction";
    // Re-enable the breakpoint.
    status = rv_bp_manager_->EnableBreakpoint(bp_pc);
    if (!status.ok()) return status;
    return disasm;
  }

  // If not at the breakpoint, or requesting a different instruction,
  inst = decode_cache_->GetDecodedInstruction(address);
  auto disasm = inst != nullptr ? inst->AsString() : "Invalid instruction";
  return disasm;
}

void KelvinTop::RequestHalt(HaltReason halt_reason,
                            const mpact::sim::generic::Instruction *inst) {
  // First set the halt_reason_, then the half flag.
  halt_reason_ = halt_reason;
  halted_ = true;
}

void KelvinTop::SetPc(uint64_t value) {
  if (pc_->data_buffer()->size<uint8_t>() == 4) {
    pc_->data_buffer()->Set<uint32_t>(0, static_cast<uint32_t>(value));
  } else {
    pc_->data_buffer()->Set<uint64_t>(0, value);
  }
}

}  // namespace kelvin::sim
