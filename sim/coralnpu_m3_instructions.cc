// Copyright 2026 Google LLC
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

// Implementation of instruction semantic functions for the CoralNPU M3 ISA.
// Instruction semantic functions specific to the CoralNPU M3 architecture are
// added here.

#include "sim/coralnpu_m3_instructions.h"

#include <cstdint>

#include "absl/base/casts.h"
#include "riscv/riscv_fp_info.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/type_helpers.h"

namespace coralnpu::sim {
using ::mpact::sim::generic::FPTypeInfo;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::generic::RegisterDestinationOperand;
using ::mpact::sim::riscv::FPExceptions;
using ::mpact::sim::riscv::FPRoundingMode;
using ::mpact::sim::riscv::GetNaNBoxedSource;
using ::mpact::sim::riscv::RiscVFPState;
namespace {

// Internal layout parameters, bitmasks, and predicate functions for bfloat16
// format.
struct BF16Info {
  static constexpr int kBitSize = 16;
  static constexpr int kExpSize = FPTypeInfo<float>::kExpSize;
  static constexpr int kSigSize = kBitSize - kExpSize - 1;
  static constexpr int kF32ToBF16Shift = FPTypeInfo<float>::kSigSize - kSigSize;

  static constexpr uint16_t kCanonicalNaN = 0x7FC0;
  static constexpr uint16_t kExponentMask = ((1U << kExpSize) - 1) << kSigSize;
  static constexpr uint16_t kFractionMask = (1U << kSigSize) - 1;
  static constexpr uint16_t kQuietBit = 1U << (kSigSize - 1);

  static bool IsNaN(uint16_t val) {
    return (val & kExponentMask) == kExponentMask && (val & kFractionMask) != 0;
  }
  static bool IsSNaN(uint16_t val) {
    return IsNaN(val) && (val & kQuietBit) == 0;
  }
};

// Mask used to NaN-box a 16-bit bfloat16 value in a 64-bit floating-point
// register.
static constexpr uint64_t kBoxedBf16Mask = ~uint64_t{0} << 16;

// Rounds a 32-bit float bit pattern down to 16-bit bfloat16 precision based on
// specified rounding mode and updates exception flags (fflags).
uint16_t RoundBFloat16(uint32_t bits, uint32_t inst_rm, RiscVFPState* fp_state,
                       uint32_t* fflags) {
  [[maybe_unused]] bool sign =
      FPTypeInfo<float>::SignBit(absl::bit_cast<float>(bits));

  // Extract rounding bits relative to bfloat16 mantissa boundary.
  bool lsb = (bits >> BF16Info::kF32ToBF16Shift) & 1;
  bool guard_bit = (bits >> (BF16Info::kF32ToBF16Shift - 1)) & 1;
  constexpr uint32_t kStickyMask = (1U << (BF16Info::kF32ToBF16Shift - 1)) - 1;
  bool sticky_bit = (bits & kStickyMask) != 0;

  bool increment = false;
  if (guard_bit || sticky_bit) {
    *fflags |= *FPExceptions::kInexact;
    // Determine effective rounding mode (resolving dynamic mode from FCSR if
    // requested).
    auto rm = (inst_rm == *FPRoundingMode::kDynamic)
                  ? (fp_state != nullptr ? fp_state->GetRoundingMode()
                                         : FPRoundingMode::kRoundToNearest)
                  : static_cast<FPRoundingMode>(inst_rm);
    switch (rm) {
      case FPRoundingMode::kRoundToNearest:
        increment = guard_bit && (lsb || sticky_bit);
        break;
      case FPRoundingMode::kRoundUp:
        increment = !sign && (guard_bit || sticky_bit);
        break;
      case FPRoundingMode::kRoundDown:
        increment = sign && (guard_bit || sticky_bit);
        break;
      case FPRoundingMode::kRoundToNearestTiesToMax:
        increment = guard_bit;
        break;
      default:
        break;
    }
  }

  uint32_t exp =
      (bits & FPTypeInfo<float>::kExpMask) >> FPTypeInfo<float>::kSigSize;
  constexpr uint32_t kMaxExp = (1U << FPTypeInfo<float>::kExpSize) - 1;
  uint32_t res_bits = (bits >> BF16Info::kF32ToBF16Shift);
  if (increment) {
    res_bits += 1;
    // Check for rounding overflow beyond representable normal range.
    if ((res_bits & BF16Info::kExponentMask) == BF16Info::kExponentMask &&
        exp != kMaxExp) {
      *fflags |= *FPExceptions::kOverflow | *FPExceptions::kInexact;
    }
  }

  // Check for underflow signal when result is subnormal and inexact.
  if ((guard_bit || sticky_bit) && (res_bits & BF16Info::kExponentMask) == 0) {
    *fflags |= *FPExceptions::kUnderflow;
  }
  return static_cast<uint16_t>(res_bits);
}

// Converts single-precision float bit pattern to bfloat16, handling NaN
// payloads.
uint16_t ConvertF32ToBF16(uint32_t bits, uint32_t rm_val,
                          ::mpact::sim::riscv::RiscVState* state,
                          uint32_t* fflags) {
  float f32_f = absl::bit_cast<float>(bits);
  if (FPTypeInfo<float>::IsNaN(f32_f)) {
    if (FPTypeInfo<float>::IsSNaN(f32_f)) {
      *fflags |= *FPExceptions::kInvalidOp;
    }
    return BF16Info::kCanonicalNaN;
  }
  return RoundBFloat16(bits, rm_val, state->rv_fp(), fflags);
}
}  // namespace

void CoralNPUFcvtBf16S(Instruction* inst) {
  auto state = static_cast<::mpact::sim::riscv::RiscVState*>(inst->state());

  // Check floating-point unit enable bit in mstatus.
  if (state->mstatus()->fs() == 0) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }
  // Validate rounding mode specified in instruction or FCSR.
  uint32_t rm_val = inst->Source(1)->AsUint32(0);
  if (rm_val > *FPRoundingMode::kRoundToNearestTiesToMax &&
      rm_val != *FPRoundingMode::kDynamic) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }
  if (rm_val == *FPRoundingMode::kDynamic &&
      !state->rv_fp()->rounding_mode_valid()) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }
  uint32_t rm = (rm_val == *FPRoundingMode::kDynamic)
                    ? *state->rv_fp()->GetRoundingMode()
                    : rm_val;

  // Fetch NaN-boxed float input from fs1.
  float f32_val = GetNaNBoxedSource<uint64_t, float>(inst, 0);
  uint32_t f32 = absl::bit_cast<uint32_t>(f32_val);

  uint32_t fflags = 0;
  uint16_t bf16 = ConvertF32ToBF16(f32, rm, state, &fflags);

  // Write NaN-boxed bfloat16 output to destination register fd.
  auto* rd_op =
      static_cast<RegisterDestinationOperand<uint64_t>*>(inst->Destination(0));
  auto* rd_reg = rd_op->GetRegister();
  rd_reg->data_buffer()->Set<uint64_t>(
      0, static_cast<uint64_t>(bf16) | kBoxedBf16Mask);

  // Accumulate exception flags into fflags/fcsr destination operand.
  auto* fflags_db = inst->Destination(1)->AllocateDataBuffer();
  fflags_db->Set<uint32_t>(0, fflags);
  fflags_db->Submit();
}

}  // namespace coralnpu::sim
