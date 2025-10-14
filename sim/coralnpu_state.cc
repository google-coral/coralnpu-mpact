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

#include "sim/coralnpu_state.h"

#include <any>
#include <cstdint>
#include <iostream>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim {

using ::mpact::sim::riscv::RiscVCsrEnum;
using ::mpact::sim::riscv::RiscVCsrInterface;

// The misa implementation uses only the 64-bit variant.
constexpr uint64_t kCoralnpuMisaVal = 0x4000000000801100;

enum class CoralNPUCsrEnum {
  kKIsa = 0xFC0,
};

constexpr uint32_t kVectorRegisterWidth = 32;

CoralNPUState::CoralNPUState(
    absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
    mpact::sim::util::MemoryInterface* memory,
    mpact::sim::util::AtomicMemoryOpInterface* atomic_memory)
    : mpact::sim::riscv::RiscVState(id, xlen, memory, atomic_memory),
      kisa_("kisa", static_cast<RiscVCsrEnum>(CoralNPUCsrEnum::kKIsa), this) {
  auto res = csr_set()->GetCsr("minstret");
  if (!res.ok()) {
    LOG(FATAL) << "Failed to get minstret";
  }
  minstret_ = res.value();
  res = csr_set()->GetCsr("minstreth");
  if (!res.ok()) {
    LOG(FATAL) << "Failed to get minstret";
  }
  minstreth_ = res.value();
  set_vector_register_width(kVectorRegisterWidth);
  for (int i = 0; i < acc_register_.size(); ++i) {
    acc_register_[i].fill(0);
  }
  if (!csr_set()->AddCsr(&kisa_).ok()) {
    LOG(FATAL) << "Failed to register kisa";
  }

  absl::StatusOr<RiscVCsrInterface*> result = csr_set()->GetCsr("misa");
  if (!result.ok()) {
    LOG(FATAL) << "Failed to get misa";
  }
  auto* misa = *result;
  misa->Set(kCoralnpuMisaVal);
}

CoralNPUState::CoralNPUState(absl::string_view id,
                             mpact::sim::riscv::RiscVXlen xlen,
                             mpact::sim::util::MemoryInterface* memory)
    : CoralNPUState(id, xlen, memory, nullptr) {}

void CoralNPUState::MPause(const Instruction* inst) {
  for (auto& handler : on_mpause_) {
    bool res = handler(inst);
    if (res) return;
  }
  // Set the return address to the current instruction.
  auto epc = (inst != nullptr) ? inst->address() : 0;
  Trap(/*is_interrupt=*/false, 0, 3, epc, inst);
}

// Print the logging message based on log_args_.
void CoralNPUState::PrintLog(absl::string_view format_string) {
  char* print_ptr = const_cast<char*>(format_string.data());
  std::string log_string = "";
  while (*print_ptr) {
    if (*print_ptr == '%') {
      CHECK_GT(log_args_.size(), 0)
          << "Invalid program with insufficient log argurments";
      if (log_args_[0].type() == typeid(uint32_t)) {
        switch (print_ptr[1]) {
          case 'u':
            log_string +=
                absl::StrFormat("%u", std::any_cast<uint32_t>(log_args_[0]));
            break;
          case 'd':
            log_string += absl::StrFormat(
                "%d",
                static_cast<int32_t>(std::any_cast<uint32_t>(log_args_[0])));
            break;
          case 'x':
            log_string +=
                absl::StrFormat("%x", std::any_cast<uint32_t>(log_args_[0]));
            break;
          default:
            std::cerr << "incorrect format" << '\n';
            break;
        }
      }
      if (log_args_[0].type() == typeid(std::string)) {
        if (print_ptr[1] == 's') {
          log_string += std::any_cast<std::string>(log_args_[0]);
        } else {
          std::cerr << "incorrect format" << '\n';
        }
      }
      log_args_.erase(log_args_.begin());
      print_ptr += 2;  // skip the format specifier too.
    } else {  // Default. Just append the character from the format string.
      log_string += *print_ptr++;
    }
  }
  std::cout << log_string;
  // Flush log_args_
  log_args_.clear();
}

}  // namespace coralnpu::sim
