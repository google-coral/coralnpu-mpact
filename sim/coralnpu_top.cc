// Copyright 2023 Google LLC
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

#include "sim/coralnpu_top.h"

#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <thread>  // NOLINT(build/c++11): built with c++17
#include <utility>

#include "sim/coralnpu_action_point_memory_interface.h"
#include "sim/coralnpu_enums.h"
#include "sim/coralnpu_state.h"
#include "sim/decoder.h"
#include "sim/proto/coralnpu_trace.pb.h"
#include "sim/renode/coralnpu_renode_memory.h"
#include "absl/flags/flag.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "riscv/riscv_arm_semihost.h"
#include "riscv/riscv_counter_csr.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/action_point_manager_base.h"
#include "mpact/sim/generic/breakpoint_manager.h"
#include "mpact/sim/generic/component.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decode_cache.h"
#include "mpact/sim/generic/resource_operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

ABSL_FLAG(bool, use_semihost, false, "Use semihost in the simulation");
ABSL_FLAG(bool, trace, false,
          "Dump executed instruction trace as Google Protobuf binary file. The "
          "output can be decoded with protoc");
ABSL_FLAG(bool, trace_disasm, false,
          "Dump the disassembled opcode string with the trace");
ABSL_FLAG(std::string, trace_path, "/tmp/coralnpu_trace.pb",
          "Path to save trace");

namespace coralnpu::sim {

using ::mpact::sim::generic::ActionPointManagerBase;
using ::mpact::sim::generic::BreakpointManager;
using ::mpact::sim::generic::DecodeCache;
using ::mpact::sim::riscv::RiscVCounterCsr;
using ::mpact::sim::riscv::RiscVCounterCsrHigh;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

constexpr char kCoralnpuName[] = "CoralNPU";

// Local helper function used to execute instructions.
static inline bool ExecuteInstruction(mpact::sim::util::Instruction* inst) {
  for (auto* resource : inst->ResourceHold()) {
    if (!resource->IsFree()) {
      return false;
    }
  }
  for (auto* resource : inst->ResourceAcquire()) {
    resource->Acquire();
  }
  inst->Execute(nullptr);
  return true;
}

CoralNPUTop::CoralNPUTop(std::string name)
    : Component{std::move(name)},
      counter_num_instructions_{"num_instructions", 0},
      counter_num_cycles_{"num_cycles", 0} {
  // Using a single flat memory for this core.
  memory_ = new mpact::sim::util::FlatDemandMemory(0);
  Initialize();
}

CoralNPUTop::CoralNPUTop(std::string name, uint64_t memory_block_size_bytes,
                         uint64_t memory_size_bytes,
                         uint8_t** memory_block_ptr_list)
    : Component(std::move(name)),
      counter_num_instructions_{"num_instructions", 0},
      counter_num_cycles_{"num_cycles", 0} {
  // Use CoralNPU renode memory for this core.
  memory_ = new renode::CoralNPURenodeMemory(
      memory_block_size_bytes, memory_size_bytes, memory_block_ptr_list);
  Initialize();
}

CoralNPUTop::~CoralNPUTop() {
  // If the simulator is still running, request a halt (set halted_ to true),
  // and wait until the simulator finishes before continuing the destructor.
  if (run_status_ == RunStatus::kRunning) {
    run_halted_->WaitForNotification();
    delete run_halted_;
  }

  delete bp_manager_;
  delete ap_manager_;
  delete coralnpu_ap_memory_interface_;
  delete decode_cache_;
  delete coralnpu_decoder_;
  delete state_;
  delete fp_state_;
  delete watcher_;
  delete memory_;
  delete semihost_;
}

void CoralNPUTop::Initialize() {
  // Create the simulation state
  state_ = new sim::CoralNPUState(kCoralnpuName,
                                  mpact::sim::riscv::RiscVXlen::RV32, memory_);
  state_->set_max_physical_address(kCoralnpuMaxMemoryAddress);
  fp_state_ = new mpact::sim::riscv::RiscVFPState(state_->csr_set(), state_);
  state_->set_rv_fp(fp_state_);
  pc_ = state_->registers()->at(sim::CoralNPUState::kPcName);
  // Set up the decoder and decode cache.
  coralnpu_decoder_ = new sim::CoralNPUDecoder(state_, memory_);
  for (int i = 0; i < static_cast<int>(isa32::OpcodeEnum::kPastMaxValue); i++) {
    counter_opcode_[i].Initialize(absl::StrCat("num_", isa32::kOpcodeNames[i]),
                                  0);
    CHECK_OK(AddCounter(&counter_opcode_[i]));
  }
  decode_cache_ = mpact::sim::generic::DecodeCache::Create({16 * 1024, 2},
                                                           coralnpu_decoder_);
  CHECK(decode_cache_) << "Failed to create decode cache";
  // Register instruction counter.
  CHECK_OK(AddCounter(&counter_num_instructions_))
      << "Failed to register counter";

  // Set up break and action points.
  coralnpu_ap_memory_interface_ = new CoralNPUActionPointMemoryInterface(
      state_->memory(),
      absl::bind_front(&DecodeCache::Invalidate, decode_cache_));
  ap_manager_ = new ActionPointManagerBase(coralnpu_ap_memory_interface_);
  bp_manager_ = new BreakpointManager(ap_manager_, [this]() {
    RequestHalt(HaltReason::kSoftwareBreakpoint, nullptr);
  });

  // Make sure the architectural and abi register aliases are added.
  std::string reg_name;
  for (int i = 0; i < 32; i++) {
    reg_name = absl::StrCat(sim::CoralNPUState::kXregPrefix, i);
    (void)state_->AddRegister<mpact::sim::riscv::RV32Register>(reg_name);
    (void)state_->AddRegisterAlias<mpact::sim::riscv::RV32Register>(
        reg_name, mpact::sim::riscv::kXRegisterAliases[i]);
  }

  semihost_ = new mpact::sim::riscv::RiscVArmSemihost(
      mpact::sim::riscv::RiscVArmSemihost::BitWidth::kWord32, memory_, memory_);
  // Set the software breakpoint callback.
  state_->AddEbreakHandler(
      [this](const mpact::sim::generic::Instruction* inst) -> bool {
        if (inst != nullptr) {
          if (absl::GetFlag(FLAGS_use_semihost) &&
              semihost_->IsSemihostingCall(inst)) {
            semihost_->OnEBreak(inst);
          } else if (ap_manager_->IsActionPointActive(inst->address())) {
            // Possible software breakpoint.
            RequestHalt(HaltReason::kActionPoint, inst);
            ap_manager_->PerformActions(inst->address());
          } else {  // The default CoralNPU simulation mode.
            std::cout << "Program exits with fault" << '\n';
            RequestHalt(kHaltAbort, inst);
          }
          return true;
        }
        return false;
      });

  state_->AddMpauseHandler(
      [this](const mpact::sim::generic::Instruction* inst) -> bool {
        if (inst != nullptr) {
          std::cout << "Program exits properly" << '\n';
          RequestHalt(HaltReason::kUserRequest, inst);
          return true;
        }
        return false;
      });

  // Set trap callbacks.
  state_->set_on_trap([this](bool is_interrupt, uint64_t trap_value,
                             uint64_t exception_code, uint64_t epc,
                             const Instruction* inst) -> bool {
    auto code = static_cast<mpact::sim::riscv::ExceptionCode>(exception_code);
    bool result = false;
    switch (code) {
      case mpact::sim::riscv::ExceptionCode::kIllegalInstruction: {
        std::cerr << "Illegal instruction at 0x" << std::hex << epc << '\n';
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      case mpact::sim::riscv::ExceptionCode::kLoadAccessFault: {
        std::cerr << "Memory load access fault at 0x" << std::hex << epc
                  << " as: " << inst->AsString() << '\n';
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      case mpact::sim::riscv::ExceptionCode::kStoreAccessFault: {
        std::cerr << "Memory store access fault at 0x" << std::hex << epc
                  << " as: " << inst->AsString() << '\n';
        RequestHalt(HaltReason::kUserRequest, nullptr);
        result = true;
      } break;
      default:
        break;
    }
    return result;
  });

  // Connect counters to instret(h) and mcycle(h) CSRs.
  auto csr_res = state_->csr_set()->GetCsr("minstret");
  // Minstret/minstreth.
  auto* minstret = reinterpret_cast<RiscVCounterCsr<uint32_t, CoralNPUState>*>(
      csr_res.value());
  minstret->set_counter(&counter_num_instructions_);
  csr_res = state_->csr_set()->GetCsr("minstreth");
  CHECK_OK(csr_res.status()) << "Failed to get minstret CSR";
  auto* minstreth =
      reinterpret_cast<RiscVCounterCsrHigh<CoralNPUState>*>(csr_res.value());
  minstreth->set_counter(&counter_num_instructions_);
  // Mcycle/mcycleh.
  csr_res = state_->csr_set()->GetCsr("mcycle");
  CHECK_OK(csr_res.status()) << "Failed to get mcycle CSR";
  auto* mcycle = reinterpret_cast<RiscVCounterCsr<uint32_t, CoralNPUState>*>(
      csr_res.value());
  mcycle->set_counter(&counter_num_cycles_);
  csr_res = state_->csr_set()->GetCsr("mcycleh");
  CHECK_OK(csr_res.status()) << "Failed to get mcycleh CSR";
  auto* mcycleh =
      reinterpret_cast<RiscVCounterCsrHigh<CoralNPUState>*>(csr_res.value());
  mcycleh->set_counter(&counter_num_cycles_);

  semihost_->set_exit_callback(
      [this]() { RequestHalt(HaltReason::kSemihostHaltRequest, nullptr); });
}

absl::Status CoralNPUTop::Halt() {
  // If it is already halted, just return.
  if (run_status_ == RunStatus::kHalted) {
    return absl::OkStatus();
  }
  // If it is not running, then there's an error.
  if (run_status_ != RunStatus::kRunning) {
    return absl::FailedPreconditionError(
        "CoralNPUTop::Halt: Core is not running");
  }
  halt_reason_ = *HaltReason::kUserRequest;
  halted_ = true;
  return absl::OkStatus();
}

absl::Status CoralNPUTop::Halt(HaltReason halt_reason) {
  RequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::Status CoralNPUTop::Halt(HaltReasonValueType halt_reason) {
  RequestHalt(halt_reason, nullptr);
  return absl::OkStatus();
}

absl::Status CoralNPUTop::StepPastBreakpoint() {
  uint64_t pc = state_->pc_operand()->AsUint64(0);
  // Disable the breakpoint. Status will show error if there is no breakpoint.
  (void)ap_manager_->ap_memory_interface()->WriteOriginalInstruction(pc);
  // Execute the real instruction.
  auto real_inst = decode_cache_->GetDecodedInstruction(pc);
  real_inst->IncRef();
  auto next_seq_pc = pc + real_inst->size();
  SetPc(next_seq_pc);
  bool executed = false;
  do {
    executed = ExecuteInstruction(real_inst);
    IncrementCycleCount(1);
    state_->AdvanceDelayLines();
  } while (!executed);
  // Increment counter.
  counter_opcode_[real_inst->opcode()].Increment(1);
  IncrementInstructionCount(1);
  real_inst->DecRef();
  // Re-enable the breakpoint.
  (void)ap_manager_->ap_memory_interface()->WriteBreakpointInstruction(pc);
  return absl::OkStatus();
}

absl::StatusOr<int> CoralNPUTop::Step(int num) {
  if (num <= 0) {
    return absl::InvalidArgumentError("Step count must be > 0");
  }
  // If the simulator is running, return with an error.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "CoralNPUTop::Step: Core must be halted");
  }
  run_status_ = RunStatus::kSingleStep;
  int count = 0;
  halted_ = false;
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (need_to_step_over_) {
    need_to_step_over_ = false;
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
    auto* inst = decode_cache_->GetDecodedInstruction(pc);
    next_seq_pc = pc + inst->size();
    // Set the PC destination operand to next_seq_pc. Any branch that is
    // executed will overwrite this.
    SetPc(next_seq_pc);
    bool executed = false;
    do {
      executed = ExecuteInstruction(inst);
      IncrementCycleCount(1);
      state_->AdvanceDelayLines();
    } while (!executed);
    count++;
    // Update counters.
    counter_opcode_[inst->opcode()].Increment(1);
    IncrementInstructionCount(1);
    // Get the next pc value.
    next_pc = pc_operand->AsUint64(0);
    if (!halted_) continue;
    // If it's an action point, just step over and continue.
    if (halt_reason_ == *HaltReason::kActionPoint) {
      auto status = StepPastBreakpoint();
      if (!status.ok()) return status;
      // Reset the halt reason and continue;
      halted_ = false;
      halt_reason_ = *HaltReason::kNone;
      need_to_step_over_ = false;
      continue;
    }
    break;
  }
  // Update the pc register, now that it can be read.
  if (halt_reason_ == *HaltReason::kSoftwareBreakpoint) {
    // If at a breakpoint, keep the pc at the current value.
    SetPc(pc);
  } else {
    // Otherwise set it to point to the next instruction.
    SetPc(next_pc);
  }
  // If there is no halt request, there is no specific halt reason.
  if (!halted_) {
    halt_reason_ = *HaltReason::kNone;
  }
  run_status_ = RunStatus::kHalted;
  return count;
}

absl::Status CoralNPUTop::Run() {
  // Verify that the core isn't running already.
  if (run_status_ == RunStatus::kRunning) {
    return absl::FailedPreconditionError(
        "CoralNPUTop::Run: core is already running");
  }
  // First check to see if the previous halt was due to a breakpoint. If so,
  // need to step over the breakpoint.
  if (need_to_step_over_) {
    need_to_step_over_ = false;
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

    std::fstream trace_file;
    coralnpu::sim::proto::TraceData trace_data;
    auto* inst_db = db_factory_.Allocate<uint32_t>(1);
    if (absl::GetFlag(FLAGS_trace)) {
      std::string trace_path = absl::GetFlag(FLAGS_trace_path);
      std::string trace_dir =
          trace_path.substr(0, trace_path.find_last_of('/'));
      int res = mkdir(trace_dir.c_str(), 0777);
      if (res == 0 || errno == EEXIST) {
        trace_file.open(absl::GetFlag(FLAGS_trace_path),
                        std::ios_base::out | std::ios_base::binary);
        std::cout << "Dump trace file at " << absl::GetFlag(FLAGS_trace_path)
                  << '\n';
      } else {
        std::cerr << "Failed to create " << trace_dir << '\n';
      }
    }

    while (!halted_) {
      pc = next_pc;
      auto* inst = decode_cache_->GetDecodedInstruction(pc);
      next_seq_pc = pc + inst->size();
      // Set the PC destination operand to next_seq_pc. Any branch that is
      // executed will overwrite this.
      SetPc(next_seq_pc);
      bool executed = false;
      do {
        executed = ExecuteInstruction(inst);
        if (trace_file.is_open()) {
          // Set trace entry {address, instruction}
          memory_->Load(pc, inst_db, nullptr, nullptr);
          auto inst_word = inst_db->Get<uint32_t>(0);
          coralnpu::sim::proto::TraceEntry* trace_entry =
              trace_data.add_entry();
          trace_entry->set_address(pc);
          trace_entry->set_opcode(inst_word);
          if (absl::GetFlag(FLAGS_trace_disasm)) {
            trace_entry->set_disasm(inst->AsString());
          }
        }
        IncrementCycleCount(1);
        state_->AdvanceDelayLines();
      } while (!executed);
      // Update counters.
      counter_opcode_[inst->opcode()].Increment(1);
      IncrementInstructionCount(1);
      // Get the next pc value.
      next_pc = pc_operand->AsUint64(0);
      if (!halted_) continue;
      // If it's an action point, just step over and continue executing, as
      // this is not a full breakpoint.
      if (halt_reason_ == *HaltReason::kActionPoint) {
        auto status = StepPastBreakpoint();
        if (!status.ok()) {
          // If there is an error, signal a simulator error.
          halt_reason_ = *HaltReason::kSimulatorError;
          break;
        };
        // Reset the halt reason and continue;
        halted_ = false;
        halt_reason_ = *HaltReason::kNone;
        continue;
      }
      break;
    }
    // Update the pc register, now that it can be read (since we are not
    // running).
    if (halt_reason_ == *HaltReason::kSoftwareBreakpoint) {
      // If at a breakpoint, keep the pc at the current value.
      SetPc(pc);
    } else {
      // Otherwise set it to point to the next instruction.
      SetPc(next_pc);
    }
    run_status_ = RunStatus::kHalted;

    inst_db->DecRef();
    if (trace_file.is_open()) {
      trace_data.SerializeToOstream(&trace_file);
      trace_file.close();
    }
    // Notify that the run has completed.
    run_halted_->Notify();
  }).detach();
  return absl::OkStatus();
}

absl::Status CoralNPUTop::Wait() {
  // If the simulator isn't running, then just return.
  if (run_status_ != RunStatus::kRunning) return absl::OkStatus();

  // Wait for the simulator to finish (i.e., a value is available on the
  // channel).
  run_halted_->WaitForNotification();
  delete run_halted_;
  run_halted_ = nullptr;
  return absl::OkStatus();
}

absl::StatusOr<CoralNPUTop::RunStatus> CoralNPUTop::GetRunStatus() {
  return run_status_;
}

absl::StatusOr<CoralNPUTop::HaltReasonValueType>
CoralNPUTop::GetLastHaltReason() {
  return halt_reason_;
}

absl::StatusOr<uint64_t> CoralNPUTop::ReadRegister(const std::string& name) {
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
    auto* csr = *result;
    return csr->GetUint32();
  }

  auto* db = (iter->second)->data_buffer();
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

absl::Status CoralNPUTop::WriteRegister(const std::string& name,
                                        uint64_t value) {
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
    auto* csr = *result;
    csr->Set(static_cast<uint32_t>(value));
    return absl::OkStatus();
  }

  // If stopped at a software breakpoint and the pc is changed, change the
  // halt reason, since the next instruction won't be where we stopped.
  if ((name == "pc") && (halt_reason_ == *HaltReason::kSoftwareBreakpoint)) {
    halt_reason_ = *HaltReason::kNone;
  }

  auto* db = (iter->second)->data_buffer();
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

absl::StatusOr<DataBuffer*> CoralNPUTop::GetRegisterDataBuffer(
    const std::string& name) {
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

absl::StatusOr<size_t> CoralNPUTop::ReadMemory(uint64_t address, void* buffer,
                                               size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("ReadMemory: Core must be halted");
  }
  if (address > state_->max_physical_address()) {
    return absl::InvalidArgumentError("Memory address invalid");
  }
  length =
      std::min<size_t>(length, state_->max_physical_address() - address + 1);
  auto* db = db_factory_.Allocate(length);
  // Load bypassing any watch points/semihosting.
  state_->memory()->Load(address, db, nullptr, nullptr);
  std::memcpy(buffer, db->raw_ptr(), length);
  db->DecRef();
  return length;
}

absl::StatusOr<size_t> CoralNPUTop::WriteMemory(uint64_t address,
                                                const void* buffer,
                                                size_t length) {
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("WriteMemory: Core must be halted");
  }
  if (address > state_->max_physical_address()) {
    return absl::InvalidArgumentError("Memory address invalid");
  }
  length =
      std::min<size_t>(length, state_->max_physical_address() - address + 1);
  auto* db = db_factory_.Allocate(length);
  std::memcpy(db->raw_ptr(), buffer, length);
  // Store bypassing any watch points/semihosting.
  state_->memory()->Store(address, db);
  db->DecRef();
  return length;
}

bool CoralNPUTop::HasBreakpoint(uint64_t address) {
  return bp_manager_->HasBreakpoint(address);
}

absl::Status CoralNPUTop::SetSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "SetSwBreakpoint: Core must be halted");
  }
  // If there is no breakpoint manager, return an error.
  if (bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  // Try setting the breakpoint.
  return bp_manager_->SetBreakpoint(address);
}

absl::Status CoralNPUTop::ClearSwBreakpoint(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "ClearSwBreakpoing: Core must be halted");
  }
  if (bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  return bp_manager_->ClearBreakpoint(address);
}

absl::Status CoralNPUTop::ClearAllSwBreakpoints() {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError(
        "ClearAllSwBreakpoints: Core must be halted");
  }
  if (bp_manager_ == nullptr) {
    return absl::InternalError("Breakpoints are not enabled");
  }
  bp_manager_->ClearAllBreakpoints();
  return absl::OkStatus();
}

absl::StatusOr<mpact::sim::generic::Instruction*> CoralNPUTop::GetInstruction(
    uint64_t address) {
  auto inst = decode_cache_->GetDecodedInstruction(address);
  return inst;
}

absl::StatusOr<std::string> CoralNPUTop::GetDisassembly(uint64_t address) {
  // Don't try if the simulator is running.
  if (run_status_ != RunStatus::kHalted) {
    return absl::FailedPreconditionError("GetDissasembly: Core must be halted");
  }

  mpact::sim::generic::Instruction* inst = nullptr;
  // If requesting the disassembly for an instruction at a breakpoint, return
  // that of the original instruction instead.
  // If requesting the disassembly for an instruction at an action point, return
  // that of the original instruction instead.
  if (ap_manager_->IsActionPointActive(address)) {
    // Write the original instruction back to memory.
    (void)ap_manager_->ap_memory_interface()->WriteOriginalInstruction(address);
    // Get the real instruction.
    inst = decode_cache_->GetDecodedInstruction(address);
    auto disasm = inst != nullptr ? inst->AsString() : "Invalid instruction";
    // Restore the breakpoint instruction.
    (void)ap_manager_->ap_memory_interface()->WriteBreakpointInstruction(
        address);
    return disasm;
  }

  // If not at the breakpoint, or requesting a different instruction,
  inst = decode_cache_->GetDecodedInstruction(address);
  auto disasm = inst != nullptr ? inst->AsString() : "Invalid instruction";
  return disasm;
}

absl::Status CoralNPUTop::LoadImage(const std::string& image_path,
                                    uint64_t start_address) {
  std::ifstream image_file;
  constexpr size_t kBufferSize = 4096;
  image_file.open(image_path, std::ios::in | std::ios::binary);
  char buffer[kBufferSize];
  size_t gcount = 0;
  uint64_t load_address = start_address;
  if (!image_file.good()) {
    return absl::Status(absl::StatusCode::kInternal, "Failed to open file");
  }
  do {
    // Fill buffer.
    image_file.read(buffer, kBufferSize);
    // Get the number of bytes that was read.
    gcount = image_file.gcount();
    if (gcount == 0) break;
    // Write to the simulator memory.
    auto res = WriteMemory(load_address, buffer, gcount);
    // Check that the write succeeded, increment address if it did.
    if (!res.ok()) {
      return absl::InternalError("Memory write failed");
    }
    if (res.value() != gcount) {
      return absl::InternalError("Failed to write all the bytes");
    }
    load_address += gcount;
  } while (image_file.good());
  image_file.close();
  return absl::OkStatus();
}

void CoralNPUTop::RequestHalt(HaltReasonValueType halt_reason,
                              const mpact::sim::generic::Instruction* inst) {
  // First set the halt_reason_, then the halt flag.
  halt_reason_ = halt_reason;
  halted_ = true;
  // If the halt reason is either sw breakpoint or action point, set
  // need_to_step_over to true.
  if ((halt_reason_ == *HaltReason::kSoftwareBreakpoint) ||
      (halt_reason_ == *HaltReason::kActionPoint)) {
    need_to_step_over_ = true;
  }
}

void CoralNPUTop::RequestHalt(HaltReason halt_reason,
                              const mpact::sim::generic::Instruction* inst) {
  RequestHalt(*halt_reason, inst);
}

void CoralNPUTop::SetPc(uint64_t value) {
  if (pc_->data_buffer()->size<uint8_t>() == 4) {
    pc_->data_buffer()->Set<uint32_t>(0, static_cast<uint32_t>(value));
  } else {
    pc_->data_buffer()->Set<uint64_t>(0, value);
  }
}

void CoralNPUTop::IncrementCycleCount(uint64_t value) {
  counter_num_cycles_.Increment(value);
}

void CoralNPUTop::IncrementInstructionCount(uint64_t value) {
  counter_num_instructions_.Increment(value);
}

}  // namespace coralnpu::sim
