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
#include "mpact/sim/util/memory/flat_demand_memory.h"

const uint32_t addr_mailbox = 0x401fc000;  // user-configurable

class MpactSimulator final : public CoralNPUSimulator {
 public:
  MpactSimulator()
      : rv_state_("RiscV32GV", mpact::sim::riscv::RiscVXlen::RV32, &memory_),
        rv_fp_state_(rv_state_.csr_set(), &rv_state_),
        rvv_state_(&rv_state_, /*vlenb*/ 16),
        rv_decoder_(&rv_state_, &memory_),
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

  void ReadTCM(uint32_t addr, size_t size, char* data) final;
  const CoralNPUMailbox& ReadMailbox() final;
  void WriteTCM(uint32_t addr, size_t size, const char* data) final;
  void WriteMailbox(const CoralNPUMailbox& mailbox) final;
  void Run(uint32_t start_addr) final;
  bool WaitForTermination(int timeout) final;

 private:
  CoralNPUMailbox mailbox_;
  ::mpact::sim::util::FlatDemandMemory memory_;
  ::mpact::sim::riscv::RiscVState rv_state_;
  ::mpact::sim::riscv::RiscVFPState rv_fp_state_;
  ::mpact::sim::riscv::RiscVVectorState rvv_state_;
  ::mpact::sim::riscv::RiscV32GVecDecoder rv_decoder_;
  ::mpact::sim::riscv::RiscVTop rv_top_;
};

void MpactSimulator::ReadTCM(uint32_t addr, size_t size, char* data) {
  auto result = rv_top_.ReadMemory(addr, data, size);
  if (!result.ok()) {
    std::cerr << "Error: " << result.status() << std::endl;
  }
  assert(result.ok());
}

const CoralNPUMailbox& MpactSimulator::ReadMailbox() {
  auto result = rv_top_.ReadMemory(
      addr_mailbox, reinterpret_cast<char*>(mailbox_.message), 16);
  if (!result.ok()) {
    std::cerr << "Error: " << result.status() << std::endl;
  }
  assert(result.ok());
  return mailbox_;
}

void MpactSimulator::WriteTCM(uint32_t addr, size_t size, const char* data) {
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

  this->WriteTCM(addr_mailbox, 16,
                 reinterpret_cast<const char*>(mailbox.message));
}

void MpactSimulator::Run(uint32_t start_addr) {
  absl::Status pc_write = rv_top_.WriteRegister("pc", start_addr);
  assert(pc_write.ok());
}

bool MpactSimulator::WaitForTermination(int timeout) {
  const uint32_t halt = 0x08000073;
  const uint32_t wfi = 0x10500073;

  while (true) {
    auto status = rv_top_.Step(1);
    if (!status.ok()) {
      return false;
    }

    uint32_t pc = rv_top_.ReadRegister("pc").value();
    uint32_t inst = 0;
    this->ReadTCM(pc, 4, reinterpret_cast<char*>(&inst));
    if (pc > 0x1FFF || inst == halt || inst == wfi) {
      break;
    }
  }

  this->ReadTCM(addr_mailbox, 16, reinterpret_cast<char*>(mailbox_.message));

  return true;
}

// static
CoralNPUSimulator* CoralNPUSimulator::Create() { return new MpactSimulator(); }
