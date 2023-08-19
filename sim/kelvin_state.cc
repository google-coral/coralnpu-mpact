
#include "sim/kelvin_state.h"

#include <any>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace kelvin::sim {

constexpr uint32_t kVectorRegisterWidth = 32;

KelvinState::KelvinState(
    absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
    mpact::sim::util::MemoryInterface *memory,
    mpact::sim::util::AtomicMemoryOpInterface *atomic_memory)
    : mpact::sim::riscv::RiscVState(id, xlen, memory, atomic_memory) {
  set_vector_register_width(kVectorRegisterWidth);
  for (int i = 0; i < acc_register_.size(); ++i) {
    acc_register_.at(i).fill(0);
  }
}

KelvinState::KelvinState(absl::string_view id,
                         mpact::sim::riscv::RiscVXlen xlen,
                         mpact::sim::util::MemoryInterface *memory)
    : KelvinState(id, xlen, memory, nullptr) {}

KelvinState::KelvinState(absl::string_view id,
                         mpact::sim::riscv::RiscVXlen xlen)
    : KelvinState(id, xlen, nullptr, nullptr) {}

void KelvinState::MPause(const Instruction *inst) {
  for (auto &handler : on_mpause_) {
    bool res = handler(inst);
    if (res) return;
  }
  // Set the return address to the current instruction.
  auto epc = (inst != nullptr) ? inst->address() : 0;
  Trap(/*is_interrupt=*/false, 0, 3, epc, inst);
}

// Print the logging message based on log_args_.
void KelvinState::PrintLog(absl::string_view format_string) {
  char *print_ptr = const_cast<char *>(format_string.data());
  while (*print_ptr) {
    if (*print_ptr == '%') {
      CHECK_GT(log_args_.size(), 0)
          << "Invalid program with insufficient log argurments";
      if (log_args_[0].type() == typeid(uint32_t)) {
        switch (print_ptr[1]) {
          case 'u':
            std::cout << std::dec << std::any_cast<uint32_t>(log_args_[0]);
            break;
          case 'd':
            std::cout << std::dec
                      << static_cast<int32_t>(
                             std::any_cast<uint32_t>(log_args_[0]));
            break;
          case 'x':
            std::cout << std::hex << std::any_cast<uint32_t>(log_args_[0]);
            break;
          default:
            std::cerr << "incorrect format" << std::endl;
            break;
        }
      }
      if (log_args_[0].type() == typeid(std::string)) {
        if (print_ptr[1] == 's') {
          std::cout << std::any_cast<std::string>(log_args_[0]);
        } else {
          std::cerr << "incorrect format" << std::endl;
        }
      }
      log_args_.erase(log_args_.begin());
      print_ptr += 2;  // skip the format specifier too.
    } else {
      std::cout << *print_ptr++;
    }
  }
  // Flush log_args_
  log_args_.clear();
}

}  // namespace kelvin::sim
