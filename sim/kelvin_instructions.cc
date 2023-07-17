#include "sim/kelvin_instructions.h"

#include <cstdint>
#include <string>

#include "sim/kelvin_state.h"

namespace kelvin::sim {

void KelvinIllegalInstruction(mpact::sim::generic::Instruction *inst) {
  auto *state = static_cast<KelvinState *>(inst->state());
  state->Trap(/*is_interrupt*/ false, /*trap_value*/ 0,
              *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
              /*epc*/ inst->address(), inst);
}

void KelvinNopInstruction(mpact::sim::generic::Instruction *inst) {}

void KelvinIMpause(const mpact::sim::generic::Instruction *inst) {
  auto *state = static_cast<KelvinState *>(inst->state());
  state->MPause(inst);
}

// A helper function to determine if there is a \0 in a char[4] stored in
// uint32_t
bool WordHasZero(uint32_t data) {
  return (((data >> 24) & 0xff) == 0) || (((data >> 16) & 0xff) == 0) ||
         (((data >> 8) & 0xff) == 0) || ((data & 0xff) == 0);
}

// A helper function to load a string from the memory address by detecting the
// '\0' terminator
void KelvinStringLoadHelper(const mpact::sim::generic::Instruction *inst,
                            std::string *out_string) {
  auto *state = static_cast<KelvinState *>(inst->state());
  auto addr = mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
  uint32_t data;
  auto *db = state->db_factory()->Allocate<uint32_t>(1);
  do {
    state->LoadMemory(inst, addr, db, nullptr, nullptr);
    data = db->Get<uint32_t>(0);
    *out_string +=
        std::string(reinterpret_cast<char *>(&data), sizeof(uint32_t));
    addr += 4;
  } while (!WordHasZero(data) && addr < state->max_physical_address());
  // Trim the string properly.
  out_string->resize(out_string->find('\0'));
  db->DecRef();
}

// Handle FLOG, SLOG, CLOG, and KLOG instructions
void KelvinLogInstruction(int log_mode,
                          mpact::sim::generic::Instruction *inst) {
  auto *state = static_cast<KelvinState *>(inst->state());
  switch (log_mode) {
    case 0: {  // Format log op to set the format of the printout and print it.
      std::string format_string;
      KelvinStringLoadHelper(inst, &format_string);
      state->PrintLog(format_string);
      break;
    }
    case 1: {  // Scalar log op to load an integer argument.
      auto data =
          mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
      state->SetLogArgs(data);
      break;
    }
    case 2: {  // Character log op to load a group of char[4] as an argument.
      auto data =
          mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0, 0);
      auto *clog_string = state->clog_string();
      // CLOG can break a long character array as multiple CLOG calls, and they
      // need to be combined as a single string argument.
      *clog_string +=
          std::string(reinterpret_cast<char *>(&data), sizeof(uint32_t));
      if (WordHasZero(data)) {
        // Trim the string properly.
        clog_string->resize(clog_string->find('\0'));
        state->SetLogArgs(*clog_string);
        clog_string->clear();
      }
      break;
    }
    case 3: {  // String log to op load a string argument.
      std::string str_arg;
      KelvinStringLoadHelper(inst, &str_arg);
      state->SetLogArgs(str_arg);
      break;
    }
    default:
      break;
  }
}

}  // namespace kelvin::sim
