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
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>

#include "sim/hw_sim/coralnpu_simulator.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv32g_vec_decoder.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/single_initiator_router.h"
#include "mpact/sim/util/memory/memory_interface.h"


class CoralMemoryTargetAdapter final : public ::mpact::sim::util::MemoryInterface {
 public:
  explicit CoralMemoryTargetAdapter(CoralMemoryTarget* target) : target_(target) {}

  void set_target(CoralMemoryTarget* target) { target_ = target; }

  void Load(uint64_t address, ::mpact::sim::generic::DataBuffer* db,
            ::mpact::sim::generic::Instruction* inst,
            ::mpact::sim::generic::ReferenceCount* context) override {
    target_->Load(address, static_cast<uint8_t*>(db->raw_ptr()),
                  db->size<uint8_t>());
    if (inst != nullptr) {
      inst->Execute(context);
    }
  }

  void Load(::mpact::sim::generic::DataBuffer* address_db,
            ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
            ::mpact::sim::generic::DataBuffer* db,
            ::mpact::sim::generic::Instruction* inst,
            ::mpact::sim::generic::ReferenceCount* context) override {
    auto addresses = address_db->Get<uint32_t>();
    auto mask = mask_db->Get<bool>();
    auto* data = static_cast<uint8_t*>(db->raw_ptr());
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i]) continue;
      target_->Load(addresses[i], data + i * el_size, el_size);
    }
    if (inst != nullptr) {
      inst->Execute(context);
    }
  }

  void Store(uint64_t address, ::mpact::sim::generic::DataBuffer* db) override {
    target_->Store(address, static_cast<const uint8_t*>(db->raw_ptr()),
                   db->size<uint8_t>());
  }

  void Store(::mpact::sim::generic::DataBuffer* address_db,
             ::mpact::sim::generic::DataBuffer* mask_db, int el_size,
             ::mpact::sim::generic::DataBuffer* db) override {
    auto addresses = address_db->Get<uint32_t>();
    auto mask = mask_db->Get<bool>();
    auto* data = static_cast<const uint8_t*>(db->raw_ptr());
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i]) continue;
      target_->Store(addresses[i], data + i * el_size, el_size);
    }
  }

 private:
  CoralMemoryTarget* target_;
};

class MpactSimulator final : public CoralNPUSimulator {
 public:
  MpactSimulator()
      : router_("coralnpu-router"),
        rv_state_("RiscV32GV", mpact::sim::riscv::RiscVXlen::RV32, &router_),
        rv_fp_state_(rv_state_.csr_set(), &rv_state_),
        rvv_state_(&rv_state_, /*vlenb*/ 16),
        rv_decoder_(&rv_state_, &router_),
        rv_top_("CoralNPUPlaceholder", &rv_state_, &rv_decoder_) {
    // Make sure the architectural and abi register aliases are added.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, i);
      (void)rv_state_.AddRegister<::mpact::sim::riscv::RV32Register>(reg_name);
      (void)rv_state_.AddRegisterAlias<::mpact::sim::riscv::RV32Register>(
          reg_name, mpact::sim::riscv::kXRegisterAliases[i]);
    }
  }
  ~MpactSimulator() final = default;

  void RegisterMemoryTarget(uint64_t address, uint64_t size,
                            CoralMemoryTarget* target) final;
  void RegisterOOBMemoryTarget(CoralMemoryTarget* target) final;

  void Run(uint32_t start_addr) final;
  bool WaitForTermination(int timeout) final;
  void SetTraceCallback(TraceCallback callback, bool disasm) final;

 private:  
  bool tracing_disasm_ = false;
  ::mpact::sim::util::SingleInitiatorRouter router_;  
  std::vector<std::unique_ptr<CoralMemoryTargetAdapter>>
      registered_targets_;
  ::mpact::sim::riscv::RiscVState rv_state_;
  ::mpact::sim::riscv::RiscVFPState rv_fp_state_;
  ::mpact::sim::riscv::RiscVVectorState rvv_state_;
  ::mpact::sim::riscv::RiscV32GVecDecoder rv_decoder_;
  ::mpact::sim::riscv::RiscVTop rv_top_;
  TraceCallback trace_callback_;
};

void MpactSimulator::RegisterMemoryTarget(uint64_t address, uint64_t size,
                                          CoralMemoryTarget* target) {
  
  const uint64_t top = address + size - 1;  

  auto adapter = std::make_unique<CoralMemoryTargetAdapter>(target);
  absl::Status status =
      router_.AddTarget<::mpact::sim::util::MemoryInterface>(adapter.get(),
                                                              address, top);
  if (!status.ok()) {
    std::cerr << "Failed to register target at 0x" << std::hex << address
              << " size 0x" << size << ": " << status << std::endl;
  }
  assert(status.ok());
  registered_targets_.push_back(std::move(adapter));
}

void MpactSimulator::RegisterOOBMemoryTarget(CoralMemoryTarget* target) {
  auto adapter = std::make_unique<CoralMemoryTargetAdapter>(target);
  absl::Status status =
      router_.AddDefaultTarget<::mpact::sim::util::MemoryInterface>(adapter.get());
  if (!status.ok()) {
    std::cerr << "Failed to register OOB target: " << status << std::endl;
  }
  assert(status.ok());
  registered_targets_.push_back(std::move(adapter));
}

void MpactSimulator::SetTraceCallback(TraceCallback callback, bool disasm) {   
  trace_callback_ = std::move(callback);
  tracing_disasm_ = disasm;
}

void MpactSimulator::Run(uint32_t start_addr) {
  absl::Status pc_write = rv_top_.WriteRegister("pc", start_addr);
  assert(pc_write.ok());
}

bool MpactSimulator::WaitForTermination(int timeout) {
  const uint32_t halt = 0x08000073;
  const uint32_t wfi = 0x10500073;

  uint32_t inst = 0;

  auto* inst_db = rv_state_.db_factory()->Allocate<uint32_t>(1);

  while (true) {
    auto status = rv_top_.Step(1);
    if (!status.ok()) {
      inst_db->DecRef();
      return false;
    }

    uint32_t pc = rv_top_.ReadRegister("pc").value();

    router_.Load(pc, inst_db, nullptr, nullptr);
    inst = inst_db->Get<uint32_t>(0);

    if (trace_callback_) {
      std::string disassembly;
      if (tracing_disasm_) {
        if (inst == halt) {
          disassembly = "HALT";
        } else if (inst == wfi) {
          disassembly = "WFI";
        } else {
          auto disasm_result = rv_top_.GetDisassembly(pc);
          if (disasm_result.ok()) {
            disassembly = disasm_result.value();
          } else {
            disassembly = "Disassembly error: " + disasm_result.status().ToString();
          }
        }
      }
      trace_callback_(pc, inst, disassembly);
    }
    
    if (pc > 0x1FFF || inst == halt || inst == wfi) {
      break;
    }
  }

  inst_db->DecRef();

  return true;
}

// static
CoralNPUSimulator* CoralNPUSimulator::Create() { return new MpactSimulator(); }
