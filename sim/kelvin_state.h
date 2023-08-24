#ifndef SIM_KELVIN_STATE_H_
#define SIM_KELVIN_STATE_H_

#include <any>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "riscv/riscv_state.h"

namespace kelvin::sim {

using Instruction = ::mpact::sim::generic::Instruction;

// Default to 256 to match
// https://spacebeaker.googlesource.com/shodan/hw/kelvin/+/refs/heads/master/hdl/chisel/src/kelvin/Parameters.scala#13.
inline constexpr uint32_t kVectorLengthInBits = 256;

constexpr uint64_t kKelvinMaxMemoryAddress = 0x3f'ffffULL;  // 4MB

template <typename T>
using AccArrayTemplate = std::array<T, kVectorLengthInBits / 32>;

using AccArrayType = AccArrayTemplate<uint32_t>;

class KelvinState : public mpact::sim::riscv::RiscVState {
 public:
  KelvinState(absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
              mpact::sim::util::MemoryInterface *memory,
              mpact::sim::util::AtomicMemoryOpInterface *atomic_memory);
  KelvinState(absl::string_view id, mpact::sim::riscv::RiscVXlen xlen,
              mpact::sim::util::MemoryInterface *memory);
  KelvinState(absl::string_view id, mpact::sim::riscv::RiscVXlen xlen);
  ~KelvinState() override = default;

  // Deleted Constructors and operators.

  KelvinState(const KelvinState &) = delete;
  KelvinState(KelvinState &&) = delete;
  KelvinState &operator=(const KelvinState &) = delete;
  KelvinState &operator=(KelvinState &&) = delete;

  void set_vector_length(uint32_t length) { vector_length_ = length; }
  uint32_t vector_length() const { return vector_length_; }

  AccArrayType *acc_vec(int index) { return &(acc_register_[index]); }
  AccArrayTemplate<AccArrayType> acc_register() const { return acc_register_; }

  void SetLogArgs(std::any data) { log_args_.emplace_back(std::move(data)); }
  std::string *clog_string() { return &clog_string_; }
  void PrintLog(absl::string_view format_string);

  // Extra Kelvin terminating state.
  void MPause(const Instruction *inst);

  // Add terminating state handler.
  void AddMpauseHandler(absl::AnyInvocable<bool(const Instruction *)> handler) {
    on_mpause_.emplace_back(std::move(handler));
  }

 private:
  uint32_t vector_length_{kVectorLengthInBits};

  // Variables to store the log arguments.
  std::vector<std::any> log_args_;
  std::string clog_string_;
  // Extra state handlers
  std::vector<absl::AnyInvocable<bool(const Instruction *)>> on_mpause_;

  // Convolution accumulation register, set to be uint32[VLENW][VLENW].
  AccArrayTemplate<AccArrayType> acc_register_;
};

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_STATE_H_
