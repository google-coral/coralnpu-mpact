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

#include "sim/kelvin_vector_instructions.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>

#include "sim/kelvin_state.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

namespace kelvin::sim {

using ::mpact::sim::generic::operator*;
using mpact::sim::generic::DataBuffer;
using mpact::sim::generic::GetInstructionSource;
using mpact::sim::riscv::RV32VectorDestinationOperand;

template <typename Vd, typename Vs1, typename Vs2>
Vd BinaryOpInvoke(std::function<Vd(Vs1, Vs2)> op, Vd vd, Vs1 vs1, Vs2 vs2) {
  return op(vs1, vs2);
}
template <typename Vd, typename Vs1, typename Vs2>
Vd BinaryOpInvoke(std::function<Vd(Vd, Vs1, Vs2)> op, Vd vd, Vs1 vs1, Vs2 vs2) {
  return op(vd, vs1, vs2);
}

template <typename Vd, typename Vs1, typename Vs2>
Vs1 CommonBinaryOpGetArg1(const Instruction *inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs1);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs1);
  if (sizeof(Vd) == sizeof(Vs1) && sizeof(Vs1) == 2 * sizeof(Vs2)) {
    // special case for VAcc instructions, which uses double the amount
    // of registers for Vs1, because it's 2x the size of Vs2.
    src_element_index += num_ops * elts_per_register * dst_reg_index;
  } else {
    src_element_index += dst_reg_index;
  }
  return GetInstructionSource<Vs1>(inst, 0, src_element_index);
}

template <typename Vd, typename Vs1, typename Vs2>
Vs2 CommonBinaryOpGetArg2(const Instruction *inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs2);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs2) +
                           dst_reg_index;
  return GetInstructionSource<Vs2>(inst, 1, scalar ? 0 : src_element_index);
}

template <typename T, typename Vd, typename Vs1, typename Vs2>
using SourceArgGetter =
    std::function<T(const Instruction *inst, bool scalar, int num_ops,
                    int op_index, int dst_element_index, int dst_reg_index)>;

template <bool halftype = false, bool widen_dst = false, typename Vd,
          typename Vs1, typename Vs2, typename... VDArgs>
void KelvinBinaryVectorOp(const Instruction *inst, bool scalar, bool strip_mine,
                          std::function<Vd(VDArgs..., Vs1, Vs2)> op,
                          SourceArgGetter<Vs1, Vd, Vs1, Vs2> arg1_getter =
                              CommonBinaryOpGetArg1<Vd, Vs1, Vs2>,
                          SourceArgGetter<Vs2, Vd, Vs1, Vs2> arg2_getter =
                              CommonBinaryOpGetArg2<Vd, Vs1, Vs2>) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_dest_register = vector_size_in_bytes / sizeof(Vd);

  // For kelvin, stripmining issues 4 contiguous vector ops.
  auto num_ops = strip_mine ? 4 : 1;
  constexpr bool is_widen_op =
      (sizeof(Vd) > sizeof(Vs2) && !halftype) || widen_dst;
  // Widening requires 2 destination regs per op.
  constexpr size_t dest_regs_per_op = is_widen_op ? 2 : 1;
  // Special case for VADD3 op which is adding dest value to vs1 + vs2.
  constexpr bool is_reading_dest = sizeof...(VDArgs) == 1;
  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));

  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer *dest_db[dest_regs_per_op];
    absl::Span<Vd> dest_span[dest_regs_per_op];

    for (int i = 0; i < dest_regs_per_op; ++i) {
      dest_db[i] = is_reading_dest
                       ? vd->CopyDataBuffer(op_index + i * num_ops)
                       : vd->AllocateDataBuffer(op_index + i * num_ops);
      dest_span[i] = dest_db[i]->template Get<Vd>();
    }

    for (int dst_element_index = 0; dst_element_index < elts_per_dest_register;
         ++dst_element_index) {
      for (int dst_reg_index = 0; dst_reg_index < dest_regs_per_op;
           ++dst_reg_index) {
        auto arg1 = arg1_getter(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index);
        auto arg2 = arg2_getter(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index);
        dest_span[dst_reg_index][dst_element_index] = BinaryOpInvoke(
            op, dest_span[dst_reg_index][dst_element_index], arg1, arg2);
      }
    }

    for (int i = 0; i < dest_regs_per_op; ++i) {
      dest_db[i]->Submit();
    }
  }
}

template <typename Vd, typename Vs>
void KelvinUnaryVectorOp(const Instruction *inst, bool strip_mine,
                         std::function<Vd(Vs)> op,
                         SourceArgGetter<Vs, Vd, Vs, Vs> arg_getter =
                             CommonBinaryOpGetArg1<Vd, Vs, Vs>) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_dest_register = vector_size_in_bytes / sizeof(Vd);

  // For kelvin, stripmining issues 4 contiguous vector ops.
  auto num_ops = strip_mine ? 4 : 1;
  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));

  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer *dest_db = vd->AllocateDataBuffer(op_index);
    absl::Span<Vd> dest_span = dest_db->template Get<Vd>();

    for (int dst_element_index = 0; dst_element_index < elts_per_dest_register;
         ++dst_element_index) {
      auto arg = arg_getter(inst, false /* scalar */, num_ops, op_index,
                            dst_element_index, 0 /* dst_reg_index */);
      dest_span[dst_element_index] = op(arg);
    }

    dest_db->Submit();
  }
}

template <typename T>
void KelvinVAdd(bool scalar, bool strip_mine, Instruction *inst) {
  // Return vs1 + vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         using UT = typename std::make_unsigned<T>::type;
                         // Cast to unsigned type before the operation to avoid
                         // undefined overflow behavior in intx_t.
                         UT uvs1 = static_cast<UT>(vs1);
                         UT uvs2 = static_cast<UT>(vs2);
                         return static_cast<T>(uvs1 + uvs2);
                       }));
}
template void KelvinVAdd<int8_t>(bool, bool, Instruction *);
template void KelvinVAdd<int16_t>(bool, bool, Instruction *);
template void KelvinVAdd<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVSub(bool scalar, bool strip_mine, Instruction *inst) {
  // Return vs1 - vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         using UT = typename std::make_unsigned<T>::type;
                         // Cast to unsigned type before the operation to avoid
                         // undefined overflow behavior in intx_t.
                         UT uvs1 = static_cast<UT>(vs1);
                         UT uvs2 = static_cast<UT>(vs2);
                         return static_cast<T>(uvs1 - uvs2);
                       }));
}
template void KelvinVSub<int8_t>(bool, bool, Instruction *);
template void KelvinVSub<int16_t>(bool, bool, Instruction *);
template void KelvinVSub<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVRSub(bool scalar, bool strip_mine, Instruction *inst) {
  // Return vs2 - vs1.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         using UT = typename std::make_unsigned<T>::type;
                         // Cast to unsigned type before the operation to avoid
                         // undefined overflow behavior in intx_t.
                         UT uvs1 = static_cast<UT>(vs1);
                         UT uvs2 = static_cast<UT>(vs2);
                         return static_cast<T>(uvs2 - uvs1);
                       }));
}
template void KelvinVRSub<int8_t>(bool, bool, Instruction *);
template void KelvinVRSub<int16_t>(bool, bool, Instruction *);
template void KelvinVRSub<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVEq(bool scalar, bool strip_mine, Instruction *inst) {
  // Return 1 if vs1 and vs2 are equal, else returns 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 == vs2; }));
}
template void KelvinVEq<int8_t>(bool, bool, Instruction *);
template void KelvinVEq<int16_t>(bool, bool, Instruction *);
template void KelvinVEq<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVNe(bool scalar, bool strip_mine, Instruction *inst) {
  // Return 1 if vs1 and vs2 are not equal, else return 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 != vs2; }));
}
template void KelvinVNe<int8_t>(bool, bool, Instruction *);
template void KelvinVNe<int16_t>(bool, bool, Instruction *);
template void KelvinVNe<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVLt(bool scalar, bool strip_mine, Instruction *inst) {
  // Returns 1 if vs1 < vs2, else return 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 < vs2; }));
}
template void KelvinVLt<int8_t>(bool, bool, Instruction *);
template void KelvinVLt<int16_t>(bool, bool, Instruction *);
template void KelvinVLt<int32_t>(bool, bool, Instruction *);
template void KelvinVLt<uint8_t>(bool, bool, Instruction *);
template void KelvinVLt<uint16_t>(bool, bool, Instruction *);
template void KelvinVLt<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVLe(bool scalar, bool strip_mine, Instruction *inst) {
  // Returns 1 if vs1 <= vs2, else return 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 <= vs2; }));
}
template void KelvinVLe<int8_t>(bool, bool, Instruction *);
template void KelvinVLe<int16_t>(bool, bool, Instruction *);
template void KelvinVLe<int32_t>(bool, bool, Instruction *);
template void KelvinVLe<uint8_t>(bool, bool, Instruction *);
template void KelvinVLe<uint16_t>(bool, bool, Instruction *);
template void KelvinVLe<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVGt(bool scalar, bool strip_mine, Instruction *inst) {
  // Returns 1 if vs1 > vs2, else return 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 > vs2; }));
}
template void KelvinVGt<int8_t>(bool, bool, Instruction *);
template void KelvinVGt<int16_t>(bool, bool, Instruction *);
template void KelvinVGt<int32_t>(bool, bool, Instruction *);
template void KelvinVGt<uint8_t>(bool, bool, Instruction *);
template void KelvinVGt<uint16_t>(bool, bool, Instruction *);
template void KelvinVGt<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVGe(bool scalar, bool strip_mine, Instruction *inst) {
  // Returns 1 if vs1 >= vs2, else return 0.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 >= vs2; }));
}
template void KelvinVGe<int8_t>(bool, bool, Instruction *);
template void KelvinVGe<int16_t>(bool, bool, Instruction *);
template void KelvinVGe<int32_t>(bool, bool, Instruction *);
template void KelvinVGe<uint8_t>(bool, bool, Instruction *);
template void KelvinVGe<uint16_t>(bool, bool, Instruction *);
template void KelvinVGe<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVAbsd(bool scalar, bool strip_mine, Instruction *inst) {
  // Returns the absolute difference between vs1 and vs2.
  // Note: for signed(INTx_MAX - INTx_MIN) the result will be UINTx_MAX.
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */,
                       typename std::make_unsigned<T>::type, T, T>(
      inst, scalar, strip_mine,
      std::function<typename std::make_unsigned<T>::type(T, T)>(
          [](T vs1, T vs2) -> typename std::make_unsigned<T>::type {
            using UT = typename std::make_unsigned<T>::type;
            // Cast to unsigned type before the operation to avoid undefined
            // overflow behavior in intx_t.
            UT uvs1 = static_cast<UT>(vs1);
            UT uvs2 = static_cast<UT>(vs2);
            return vs1 > vs2 ? uvs1 - uvs2 : uvs2 - uvs1;
          }));
}
template void KelvinVAbsd<int8_t>(bool, bool, Instruction *);
template void KelvinVAbsd<int16_t>(bool, bool, Instruction *);
template void KelvinVAbsd<int32_t>(bool, bool, Instruction *);
template void KelvinVAbsd<uint8_t>(bool, bool, Instruction *);
template void KelvinVAbsd<uint16_t>(bool, bool, Instruction *);
template void KelvinVAbsd<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVMax(bool scalar, bool strip_mine, Instruction *inst) {
  // Return the max of vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         return std::max(vs1, vs2);
                       }));
}
template void KelvinVMax<int8_t>(bool, bool, Instruction *);
template void KelvinVMax<int16_t>(bool, bool, Instruction *);
template void KelvinVMax<int32_t>(bool, bool, Instruction *);
template void KelvinVMax<uint8_t>(bool, bool, Instruction *);
template void KelvinVMax<uint16_t>(bool, bool, Instruction *);
template void KelvinVMax<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVMin(bool scalar, bool strip_mine, Instruction *inst) {
  // Return the min of vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         return std::min(vs1, vs2);
                       }));
}
template void KelvinVMin<int8_t>(bool, bool, Instruction *);
template void KelvinVMin<int16_t>(bool, bool, Instruction *);
template void KelvinVMin<int32_t>(bool, bool, Instruction *);
template void KelvinVMin<uint8_t>(bool, bool, Instruction *);
template void KelvinVMin<uint16_t>(bool, bool, Instruction *);
template void KelvinVMin<uint32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVAdd3(bool scalar, bool strip_mine, Instruction *inst) {
  // Return the summation of vd, vs1, and vs2.
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using UT = typename std::make_unsigned<T>::type;
        UT uvs1 = static_cast<UT>(vs1);
        UT uvs2 = static_cast<UT>(vs2);
        UT uvd = static_cast<UT>(vd);
        return static_cast<T>(uvd + uvs1 + uvs2);
      }));
}
template void KelvinVAdd3<int8_t>(bool, bool, Instruction *);
template void KelvinVAdd3<int16_t>(bool, bool, Instruction *);
template void KelvinVAdd3<int32_t>(bool, bool, Instruction *);

// Helper function for Vadds (saturated signed addition).
// Uses unsigned arithmetic for the addition to avoid signed overflow, which,
// when compiled with --config=asan, will trigger an exception.
template <typename T>
inline T VAddsHelper(T vs1, T vs2) {
  using UT = typename std::make_unsigned<T>::type;
  UT uvs1 = static_cast<UT>(vs1);
  UT uvs2 = static_cast<UT>(vs2);
  UT usum = uvs1 + uvs2;
  T sum = static_cast<T>(usum);
  if (((vs1 ^ vs2) >= 0) && ((sum ^ vs1) < 0)) {
    return vs1 > 0 ? std::numeric_limits<T>::max()
                   : std::numeric_limits<T>::min();
  }
  return sum;
}

template <typename T>
void KelvinVAdds(bool scalar, bool strip_mine, Instruction *inst) {
  // Return saturated sum of vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>(VAddsHelper<T>));
}
template void KelvinVAdds<int8_t>(bool, bool, Instruction *);
template void KelvinVAdds<int16_t>(bool, bool, Instruction *);
template void KelvinVAdds<int32_t>(bool, bool, Instruction *);

// Helper function for Vaddsu (saturated unsigned addition).
template <typename T>
inline T VAddsuHelper(T vs1, T vs2) {
  T sum = vs1 + vs2;
  if (sum < vs1) {
    sum = std::numeric_limits<T>::max();
  }
  return sum;
}

template <typename T>
void KelvinVAddsu(bool scalar, bool strip_mine, Instruction *inst) {
  // Return saturated sum of unsigned vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>(VAddsuHelper<T>));
}
template void KelvinVAddsu<uint8_t>(bool, bool, Instruction *);
template void KelvinVAddsu<uint16_t>(bool, bool, Instruction *);
template void KelvinVAddsu<uint32_t>(bool, bool, Instruction *);

// Helper function for Vsubs (saturated signed subtraction).
template <typename T>
inline T VSubsHelper(T vs1, T vs2) {
  using UT = typename std::make_unsigned<T>::type;
  UT uvs1 = static_cast<UT>(vs1);
  UT uvs2 = static_cast<UT>(vs2);
  UT usub = uvs1 - uvs2;
  T sub = static_cast<T>(usub);
  if (((vs1 ^ vs2) < 0) && ((sub ^ vs2) >= 0)) {
    return vs2 < 0 ? std::numeric_limits<T>::max()
                   : std::numeric_limits<T>::min();
  }
  return sub;
}

template <typename T>
void KelvinVSubs(bool scalar, bool strip_mine, Instruction *inst) {
  // Return saturated sub of vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>(VSubsHelper<T>));
}
template void KelvinVSubs<int8_t>(bool, bool, Instruction *);
template void KelvinVSubs<int16_t>(bool, bool, Instruction *);
template void KelvinVSubs<int32_t>(bool, bool, Instruction *);

template <typename T>
void KelvinVSubsu(bool scalar, bool strip_mine, Instruction *inst) {
  // Return saturated sub of unsigned vs1 and vs2.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         return vs1 < vs2 ? 0 : vs1 - vs2;
                       }));
}
template void KelvinVSubsu<uint8_t>(bool, bool, Instruction *);
template void KelvinVSubsu<uint16_t>(bool, bool, Instruction *);
template void KelvinVSubsu<uint32_t>(bool, bool, Instruction *);

template <typename Td, typename Ts>
void KelvinVAddw(bool scalar, bool strip_mine, Instruction *inst) {
  // Adds operands with widening.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                         return static_cast<Td>(vs1) + static_cast<Td>(vs2);
                       }));
}
template void KelvinVAddw<int16_t, int8_t>(bool, bool, Instruction *);
template void KelvinVAddw<int32_t, int16_t>(bool, bool, Instruction *);
template void KelvinVAddw<uint16_t, uint8_t>(bool, bool, Instruction *);
template void KelvinVAddw<uint32_t, uint16_t>(bool, bool, Instruction *);

template <typename Td, typename Ts>
void KelvinVSubw(bool scalar, bool strip_mine, Instruction *inst) {
  // Subtracts operands with widening.
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                         return static_cast<Td>(vs1) - static_cast<Td>(vs2);
                       }));
}
template void KelvinVSubw<int16_t, int8_t>(bool, bool, Instruction *);
template void KelvinVSubw<int32_t, int16_t>(bool, bool, Instruction *);
template void KelvinVSubw<uint16_t, uint8_t>(bool, bool, Instruction *);
template void KelvinVSubw<uint32_t, uint16_t>(bool, bool, Instruction *);

template <typename Td, typename Ts2>
void KelvinVAcc(bool scalar, bool strip_mine, Instruction *inst) {
  // Accumulates operands with widening.
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<Td(Td, Ts2)>([](Td vs1, Ts2 vs2) -> Td {
        using UTd = typename std::make_unsigned<Td>::type;
        return static_cast<Td>(static_cast<UTd>(vs1) + static_cast<UTd>(vs2));
      }));
}
template void KelvinVAcc<int16_t, int8_t>(bool, bool, Instruction *);
template void KelvinVAcc<int32_t, int16_t>(bool, bool, Instruction *);
template void KelvinVAcc<uint16_t, uint8_t>(bool, bool, Instruction *);
template void KelvinVAcc<uint32_t, uint16_t>(bool, bool, Instruction *);

template <typename Vd, typename Vs1, typename Vs2>
Vs1 PackedBinaryOpGetArg1(const Instruction *inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs1);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs1);
  return GetInstructionSource<Vs1>(inst, 0, src_element_index);
}

template <typename Vd, typename Vs1, typename Vs2>
Vs2 PackedBinaryOpGetArg2(const Instruction *inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs2);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs2) + 1;
  return GetInstructionSource<Vs2>(inst, 0, src_element_index);
}

template <typename Td, typename Ts>
void KelvinVPadd(bool strip_mine, Instruction *inst) {
  // Adds lane pairs.
  KelvinBinaryVectorOp<true /* halftype */, false /* widen_dst */, Td, Ts, Ts>(
      inst, false /* scalar */, strip_mine,
      std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
        return static_cast<Td>(vs1) + static_cast<Td>(vs2);
      }),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg1<Td, Ts, Ts>),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg2<Td, Ts, Ts>));
}
template void KelvinVPadd<int16_t, int8_t>(bool, Instruction *);
template void KelvinVPadd<int32_t, int16_t>(bool, Instruction *);
template void KelvinVPadd<uint16_t, uint8_t>(bool, Instruction *);
template void KelvinVPadd<uint32_t, uint16_t>(bool, Instruction *);

template <typename Td, typename Ts>
void KelvinVPsub(bool strip_mine, Instruction *inst) {
  // Subtracts lane pairs.
  KelvinBinaryVectorOp<true /* halftype */, false /* widen_dst */, Td, Ts, Ts>(
      inst, false /* scalar */, strip_mine,
      std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
        return static_cast<Td>(vs1) - static_cast<Td>(vs2);
      }),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg1<Td, Ts, Ts>),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg2<Td, Ts, Ts>));
}
template void KelvinVPsub<int16_t, int8_t>(bool, Instruction *);
template void KelvinVPsub<int32_t, int16_t>(bool, Instruction *);
template void KelvinVPsub<uint16_t, uint8_t>(bool, Instruction *);
template void KelvinVPsub<uint32_t, uint16_t>(bool, Instruction *);

// Halving addition with optional rounding bit.
template <typename T>
T KelvinVHaddHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  return static_cast<T>(
      (static_cast<WT>(vs1) + static_cast<WT>(vs2) + (round ? 1 : 0)) >> 1);
}

template <typename T>
void KelvinVHadd(bool scalar, bool strip_mine, bool round, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&KelvinVHaddHelper<T>, round)));
}
template void KelvinVHadd<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVHadd<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVHadd<int32_t>(bool, bool, bool, Instruction *);
template void KelvinVHadd<uint8_t>(bool, bool, bool, Instruction *);
template void KelvinVHadd<uint16_t>(bool, bool, bool, Instruction *);
template void KelvinVHadd<uint32_t>(bool, bool, bool, Instruction *);

// Halving subtraction with optional rounding bit.
template <typename T>
T KelvinVHsubHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  return static_cast<T>(
      (static_cast<WT>(vs1) - static_cast<WT>(vs2) + (round ? 1 : 0)) >> 1);
}

template <typename T>
void KelvinVHsub(bool scalar, bool strip_mine, bool round, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&KelvinVHsubHelper<T>, round)));
}
template void KelvinVHsub<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVHsub<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVHsub<int32_t>(bool, bool, bool, Instruction *);
template void KelvinVHsub<uint8_t>(bool, bool, bool, Instruction *);
template void KelvinVHsub<uint16_t>(bool, bool, bool, Instruction *);
template void KelvinVHsub<uint32_t>(bool, bool, bool, Instruction *);

// Bitwise and.
template <typename T>
void KelvinVAnd(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 & vs2; }));
}
template void KelvinVAnd<uint8_t>(bool, bool, Instruction *);
template void KelvinVAnd<uint16_t>(bool, bool, Instruction *);
template void KelvinVAnd<uint32_t>(bool, bool, Instruction *);

// Bitwise or.
template <typename T>
void KelvinVOr(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 | vs2; }));
}
template void KelvinVOr<uint8_t>(bool, bool, Instruction *);
template void KelvinVOr<uint16_t>(bool, bool, Instruction *);
template void KelvinVOr<uint32_t>(bool, bool, Instruction *);

// Bitwise xor.
template <typename T>
void KelvinVXor(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 ^ vs2; }));
}
template void KelvinVXor<uint8_t>(bool, bool, Instruction *);
template void KelvinVXor<uint16_t>(bool, bool, Instruction *);
template void KelvinVXor<uint32_t>(bool, bool, Instruction *);

// Generalized reverse using bit ladder.
template <typename T>
void KelvinVRev(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine, std::function<T(T, T)>([](T vs1, T vs2) -> T {
        T r = vs1;
        T count = vs2 & 0b11111;
        if (count & 1) r = ((r & 0x55555555) << 1) | ((r & 0xAAAAAAAA) >> 1);
        if (count & 2) r = ((r & 0x33333333) << 2) | ((r & 0xCCCCCCCC) >> 2);
        if (count & 4) r = ((r & 0x0F0F0F0F) << 4) | ((r & 0xF0F0F0F0) >> 4);
        if (sizeof(T) == 1) return r;
        if (count & 8) r = ((r & 0x00FF00FF) << 8) | ((r & 0xFF00FF00) >> 8);
        if (sizeof(T) == 2) return r;
        if (count & 16) r = ((r & 0x0000FFFF) << 16) | ((r & 0xFFFF0000) >> 16);
        return r;
      }));
}
template void KelvinVRev<uint8_t>(bool, bool, Instruction *);
template void KelvinVRev<uint16_t>(bool, bool, Instruction *);
template void KelvinVRev<uint32_t>(bool, bool, Instruction *);

// Cyclic rotation right using a bit ladder.
template <typename T>
void KelvinVRor(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         T r = vs1;
                         T count = vs2 & static_cast<T>(sizeof(T) * 8 - 1);
                         for (auto shift : {1, 2, 4, 8, 16}) {
                           if (count & shift)
                             r = (r >> shift) | (r << (sizeof(T) * 8 - shift));
                         }
                         return r;
                       }));
}
template void KelvinVRor<uint8_t>(bool, bool, Instruction *);
template void KelvinVRor<uint16_t>(bool, bool, Instruction *);
template void KelvinVRor<uint32_t>(bool, bool, Instruction *);

// Returns Arg1 as either vs1 or vs2 based on dst_reg_index.
template <typename Vd, typename Vs1, typename Vs2>
Vs1 VMvpOpGetArg1(const Instruction *inst, bool scalar, int num_ops,
                  int op_index, int dst_element_index, int dst_reg_index) {
  return dst_reg_index == 0
             ? CommonBinaryOpGetArg1<Vd, Vs1, Vs2>(
                   inst, scalar, num_ops, op_index, dst_element_index, 0)
             : CommonBinaryOpGetArg2<Vd, Vs1, Vs2>(
                   inst, scalar, num_ops, op_index, dst_element_index, 0);
}

// Copies a pair of registers.
template <typename T>
void KelvinVMvp(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VMvpOpGetArg1<T, T, T>),
      // Arg2 isn't used. We provide a custom getter here because the default
      // getter expects extra source registers for widening ops.
      SourceArgGetter<T, T, T, T>(VMvpOpGetArg1<T, T, T>));
}
template void KelvinVMvp<uint8_t>(bool, bool, Instruction *);
template void KelvinVMvp<uint16_t>(bool, bool, Instruction *);
template void KelvinVMvp<uint32_t>(bool, bool, Instruction *);

// Logical shift left.
template <typename T>
void KelvinVSll(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         size_t shift = vs2 & (sizeof(T) * 8 - 1);
                         return vs1 << shift;
                       }));
}
template void KelvinVSll<uint8_t>(bool, bool, Instruction *);
template void KelvinVSll<uint16_t>(bool, bool, Instruction *);
template void KelvinVSll<uint32_t>(bool, bool, Instruction *);

// Arithmetic shift right.
template <typename T>
void KelvinVSra(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         size_t shift = vs2 & (sizeof(T) * 8 - 1);
                         return vs1 >> shift;
                       }));
}
template void KelvinVSra<int8_t>(bool, bool, Instruction *);
template void KelvinVSra<int16_t>(bool, bool, Instruction *);
template void KelvinVSra<int32_t>(bool, bool, Instruction *);

// Logical shift right.
template <typename T>
void KelvinVSrl(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>([](T vs1, T vs2) -> T {
                         size_t shift = vs2 & (sizeof(T) * 8 - 1);
                         return vs1 >> shift;
                       }));
}
template void KelvinVSrl<uint8_t>(bool, bool, Instruction *);
template void KelvinVSrl<uint16_t>(bool, bool, Instruction *);
template void KelvinVSrl<uint32_t>(bool, bool, Instruction *);

// Logical and arithmetic left/right shift with saturating shift amount and
// result.
template <typename T>
T KelvinVShiftHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  if (std::is_signed<T>::value == true) {
    constexpr int kMaxShiftBit = sizeof(T) * 8;
    int shamt = vs2;
    WT shift = vs1;
    if (!vs1) {
      return 0;
    } else if (vs1 < 0 && shamt >= kMaxShiftBit) {
      shift = -1 + round;
    } else if (vs1 > 0 && shamt >= kMaxShiftBit) {
      shift = 0;
    } else if (shamt > 0) {
      shift = (static_cast<WT>(vs1) +
               (round ? static_cast<WT>(1ll << (shamt - 1)) : 0)) >>
              shamt;
    } else {  // shamt < 0
      using UT = typename std::make_unsigned<T>::type;
      UT ushamt =
          static_cast<UT>(-shamt <= kMaxShiftBit ? -shamt : kMaxShiftBit);
      CHECK_LE(ushamt, kMaxShiftBit);
      CHECK_GE(ushamt, 0);
      // Use unsigned WideType to prevent undefined negative shift.
      using UWT = typename mpact::sim::generic::WideType<UT>::type;
      shift = static_cast<WT>(static_cast<UWT>(vs1) << ushamt);
    }
    T neg_max = std::numeric_limits<T>::min();
    T pos_max = std::numeric_limits<T>::max();
    bool neg_sat = vs1 < 0 && (shamt <= -kMaxShiftBit || shift < neg_max);
    bool pos_sat = vs1 > 0 && (shamt <= -kMaxShiftBit || shift > pos_max);
    if (neg_sat) return neg_max;
    if (pos_sat) return pos_max;
    return shift;
  }
  // unsigned.
  constexpr int kMaxShiftBit = sizeof(T) * 8;
  // Shift can be positive/negative.
  int shamt = static_cast<typename std::make_signed<T>::type>(vs2);
  WT shift = vs1;
  if (!vs1) {
    return 0;
  } else if (shamt > kMaxShiftBit) {
    shift = 0;
  } else if (shamt > 0) {
    shift = (static_cast<WT>(vs1) +
             (round ? static_cast<WT>(1ull << (shamt - 1)) : 0)) >>
            shamt;
  } else {
    using UT = typename std::make_unsigned<T>::type;
    UT ushamt = static_cast<UT>(-shamt <= kMaxShiftBit ? -shamt : kMaxShiftBit);
    shift = static_cast<WT>(vs1) << (ushamt);
  }
  T pos_max = std::numeric_limits<T>::max();
  bool pos_sat = vs1 && (shamt < -kMaxShiftBit || shift > pos_max);
  if (pos_sat) return pos_max;
  return shift;
}

template <typename T>
void KelvinVShift(bool round, bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&KelvinVShiftHelper<T>, round)));
}
template void KelvinVShift<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVShift<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVShift<int32_t>(bool, bool, bool, Instruction *);
template void KelvinVShift<uint8_t>(bool, bool, bool, Instruction *);
template void KelvinVShift<uint16_t>(bool, bool, bool, Instruction *);
template void KelvinVShift<uint32_t>(bool, bool, bool, Instruction *);

// Bitwise not.
template <typename T>
void KelvinVNot(bool strip_mine, Instruction *inst) {
  KelvinUnaryVectorOp(inst, strip_mine,
                      std::function<T(T)>([](T vs) -> T { return ~vs; }));
}
template void KelvinVNot<int32_t>(bool, Instruction *);

// Count the leading bits.
template <typename T>
void KelvinVClb(bool strip_mine, Instruction *inst) {
  KelvinUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                        return (vs & (1u << (sizeof(T) * 8 - 1)))
                                   ? absl::countl_one(vs)
                                   : absl::countl_zero(vs);
                      }));
}
template void KelvinVClb<uint8_t>(bool, Instruction *);
template void KelvinVClb<uint16_t>(bool, Instruction *);
template void KelvinVClb<uint32_t>(bool, Instruction *);

// Count the leading zeros.
template <typename T>
void KelvinVClz(bool strip_mine, Instruction *inst) {
  KelvinUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                        return absl::countl_zero(vs);
                      }));
}
template void KelvinVClz<uint8_t>(bool, Instruction *);
template void KelvinVClz<uint16_t>(bool, Instruction *);
template void KelvinVClz<uint32_t>(bool, Instruction *);

// Count the set bits.
template <typename T>
void KelvinVCpop(bool strip_mine, Instruction *inst) {
  KelvinUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                        return absl::popcount(vs);
                      }));
}
template void KelvinVCpop<uint8_t>(bool, Instruction *);
template void KelvinVCpop<uint16_t>(bool, Instruction *);
template void KelvinVCpop<uint32_t>(bool, Instruction *);

// Move a register.
template <typename T>
void KelvinVMv(bool strip_mine, Instruction *inst) {
  KelvinUnaryVectorOp(inst, strip_mine,
                      std::function<T(T)>([](T vs) -> T { return vs; }));
}
template void KelvinVMv<int32_t>(bool, Instruction *);

// Alternates Vs1 register used for odd/even destination indices.
template <typename Vd, typename Vs1, typename Vs2>
Vs1 VSransOpGetArg1(const Instruction *inst, bool scalar, int num_ops,
                    int op_index, int dst_element_index, int dst_reg_index) {
  static_assert(2 * sizeof(Vd) == sizeof(Vs1) || 4 * sizeof(Vd) == sizeof(Vs1));
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs1);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs1);

  if (sizeof(Vs1) / sizeof(Vd) == 2) {
    src_element_index +=
        dst_element_index & 1 ? num_ops * elts_per_register : 0;
  } else {  // sizeof(Vs1) / sizeof(Vd) == 4
    const int interleave[4] = {0, 2, 1, 3};
    src_element_index +=
        interleave[dst_element_index & 3] * num_ops * elts_per_register;
  }

  return GetInstructionSource<Vs1>(inst, 0, src_element_index);
}

// Arithmetic right shift with rounding and signed/unsigned saturation.
// Narrowing x2 or x4.
template <typename Td, typename Ts>
Td KelvinVSransHelper(bool round, Ts vs1, Td vs2) {
  static_assert(2 * sizeof(Td) == sizeof(Ts) || 4 * sizeof(Td) == sizeof(Ts));
  constexpr int src_bits = sizeof(Ts) * 8;
  vs2 &= (src_bits - 1);
  using WTs = typename mpact::sim::generic::WideType<Ts>::type;
  WTs res = (static_cast<WTs>(vs1) +
             (vs2 && round ? static_cast<WTs>(1ll << (vs2 - 1)) : 0)) >>
            vs2;

  bool neg_sat = res < std::numeric_limits<Td>::min();
  bool pos_sat = res > std::numeric_limits<Td>::max();
  bool zero = !vs1;
  if (neg_sat) return std::numeric_limits<Td>::min();
  if (pos_sat) return std::numeric_limits<Td>::max();
  if (zero) return 0;
  return res;
}

template <typename Td, typename Ts>
void KelvinVSrans(bool round, bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<Td(Ts, Td)>(
          absl::bind_front(&KelvinVSransHelper<Td, Ts>, round)),
      SourceArgGetter<Ts, Td, Ts, Td>(VSransOpGetArg1<Td, Ts, Td>));
}
template void KelvinVSrans<int8_t, int16_t>(bool, bool, bool, Instruction *);
template void KelvinVSrans<int16_t, int32_t>(bool, bool, bool, Instruction *);
template void KelvinVSrans<uint8_t, uint16_t>(bool, bool, bool, Instruction *);
template void KelvinVSrans<uint16_t, uint32_t>(bool, bool, bool, Instruction *);
template void KelvinVSrans<int8_t, int32_t>(bool, bool, bool, Instruction *);
template void KelvinVSrans<uint8_t, uint32_t>(bool, bool, bool, Instruction *);

// Multiplication of vector elements.
template <typename T>
void KelvinVMul(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine, std::function<T(T, T)>([](T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;

        return static_cast<T>(static_cast<WT>(vs1) * static_cast<WT>(vs2));
      }));
}
template void KelvinVMul<int8_t>(bool, bool, Instruction *);
template void KelvinVMul<int16_t>(bool, bool, Instruction *);
template void KelvinVMul<int32_t>(bool, bool, Instruction *);

// Multiplication of vector elements with saturation.
template <typename T>
void KelvinVMuls(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine, std::function<T(T, T)>([](T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;
        WT result = static_cast<WT>(vs1) * static_cast<WT>(vs2);
        if (std::is_signed<T>::value) {
          result = std::max(
              static_cast<WT>(std::numeric_limits<T>::min()),
              std::min(static_cast<WT>(std::numeric_limits<T>::max()), result));
          return result;
        }

        result =
            std::min(static_cast<WT>(std::numeric_limits<T>::max()), result);
        return result;
      }));
}
template void KelvinVMuls<int8_t>(bool, bool, Instruction *);
template void KelvinVMuls<int16_t>(bool, bool, Instruction *);
template void KelvinVMuls<int32_t>(bool, bool, Instruction *);
template void KelvinVMuls<uint8_t>(bool, bool, Instruction *);
template void KelvinVMuls<uint16_t>(bool, bool, Instruction *);
template void KelvinVMuls<uint32_t>(bool, bool, Instruction *);

// Multiplication of vector elements with widening.
template <typename Td, typename Ts>
void KelvinVMulw(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                         return static_cast<Td>(vs1) * static_cast<Td>(vs2);
                       }));
}
template void KelvinVMulw<int16_t, int8_t>(bool, bool, Instruction *);
template void KelvinVMulw<int32_t, int16_t>(bool, bool, Instruction *);
template void KelvinVMulw<uint16_t, uint8_t>(bool, bool, Instruction *);
template void KelvinVMulw<uint32_t, uint16_t>(bool, bool, Instruction *);

// Multiplication of vector elements with widening and optional rounding.
// Returns high half.
template <typename T>
T KelvinVMulhHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  constexpr int n = sizeof(T) * 8;

  WT result = static_cast<WT>(vs1) * static_cast<WT>(vs2);
  result += round ? static_cast<WT>(1ll << (n - 1)) : 0;
  return result >> n;
}

template <typename T>
void KelvinVMulh(bool scalar, bool strip_mine, bool round, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&KelvinVMulhHelper<T>, round)));
}
template void KelvinVMulh<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVMulh<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVMulh<int32_t>(bool, bool, bool, Instruction *);
template void KelvinVMulh<uint8_t>(bool, bool, bool, Instruction *);
template void KelvinVMulh<uint16_t>(bool, bool, bool, Instruction *);
template void KelvinVMulh<uint32_t>(bool, bool, bool, Instruction *);

// Saturating signed doubling multiply returning high half with optional
// rounding.
template <typename T>
T KelvinVDmulhHelper(bool round, bool round_neg, T vs1, T vs2) {
  constexpr int n = sizeof(T) * 8;
  using WT = typename mpact::sim::generic::WideType<T>::type;
  WT result = static_cast<WT>(vs1) * static_cast<WT>(vs2);
  if (round) {
    WT rnd = static_cast<WT>(0x40000000ll >> (32 - n));
    if (result < 0 && round_neg) {
      rnd = static_cast<WT>((-0x40000000ll) >> (32 - n));
    }
    result += rnd;
  }
  result >>= (n - 1);
  if (vs1 == std::numeric_limits<T>::min() &&
      vs2 == std::numeric_limits<T>::min()) {
    result = std::numeric_limits<T>::max();
  }
  return result;
}
template <typename T>
void KelvinVDmulh(bool scalar, bool strip_mine, bool round, bool round_neg,
                  Instruction *inst) {
  KelvinBinaryVectorOp(inst, scalar, strip_mine,
                       std::function<T(T, T)>(absl::bind_front(
                           &KelvinVDmulhHelper<T>, round, round_neg)));
}
template void KelvinVDmulh<int8_t>(bool, bool, bool, bool, Instruction *);
template void KelvinVDmulh<int16_t>(bool, bool, bool, bool, Instruction *);
template void KelvinVDmulh<int32_t>(bool, bool, bool, bool, Instruction *);

// Multiply accumulate.
template <typename T>
void KelvinVMacc(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;
        return static_cast<WT>(vd) +
               static_cast<WT>(vs1) * static_cast<WT>(vs2);
      }));
}
template void KelvinVMacc<int8_t>(bool, bool, Instruction *);
template void KelvinVMacc<int16_t>(bool, bool, Instruction *);
template void KelvinVMacc<int32_t>(bool, bool, Instruction *);

// Multiply add.
template <typename T>
void KelvinVMadd(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;
        return static_cast<WT>(vs1) +
               static_cast<WT>(vd) * static_cast<WT>(vs2);
      }));
}
template void KelvinVMadd<int8_t>(bool, bool, Instruction *);
template void KelvinVMadd<int16_t>(bool, bool, Instruction *);
template void KelvinVMadd<int32_t>(bool, bool, Instruction *);

// Computes slide index for next register and takes result from either vs1 or
// vs2.
template <typename T>
T VSlidenOpGetArg1(bool horizontal, int index, const Instruction *inst,
                   bool scalar, int num_ops, int op_index,
                   int dst_element_index, int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(T);

  using Interleave = struct {
    int register_num;
    int source_arg;
  };
  const Interleave interleave_start[2][4] = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                                             {{0, 0}, {1, 0}, {2, 0}, {3, 0}}};
  const Interleave interleave_end[2][4] = {{{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                                           {{1, 0}, {2, 0}, {3, 0}, {0, 1}}};
  // Get the elements from the right up to `index`.
  // For the horizontal mode, it treats the stripmine `vm` register based on
  // `vs1` as a contiguous block, and only the first `index` elements from `vs2`
  // will be used.
  //
  // For the vertical mode, each stripmine vector register `op_index` is mapped
  // separatedly. it mimics the imaging tiling process shift of
  //   |--------|--------|
  //   | 4xVLEN | 4xVLEN |
  //   |  (vs1) |  (vs2) |
  //   |--------|--------|
  // The vertical mode can also support the non-stripmine version to handle
  // the last columns of the image.
  if (dst_element_index + index < elts_per_register) {
    auto src_element_index =
        interleave_start[horizontal][op_index].register_num *
            elts_per_register +
        dst_element_index + index;
    return GetInstructionSource<T>(
        inst, interleave_start[horizontal][op_index].source_arg,
        src_element_index);
  }

  auto src_element_index =
      interleave_end[horizontal][op_index].register_num * elts_per_register +
      dst_element_index + index - elts_per_register;
  return GetInstructionSource<T>(
      inst, interleave_end[horizontal][op_index].source_arg, src_element_index);
}

// Slide next register vertically by index.
template <typename T>
void KelvinVSlidevn(int index, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, false /* scalar */, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(absl::bind_front(
          VSlidenOpGetArg1<T>, false /* horizontal */, index)));
}
template void KelvinVSlidevn<int8_t>(int, bool, Instruction *);
template void KelvinVSlidevn<int16_t>(int, bool, Instruction *);
template void KelvinVSlidevn<int32_t>(int, bool, Instruction *);

// Slide next register horizontally by index.
template <typename T>
void KelvinVSlidehn(int index, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, false /* scalar */, true /* strip_mine */,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(
          absl::bind_front(VSlidenOpGetArg1<T>, true /* horizontal */, index)));
}
template void KelvinVSlidehn<int8_t>(int, Instruction *);
template void KelvinVSlidehn<int16_t>(int, Instruction *);
template void KelvinVSlidehn<int32_t>(int, Instruction *);

// Computes slide index for previous register and takes result from either vs1
// or vs2.
template <typename T>
T VSlidepOpGetArg1(bool horizontal, int index, const Instruction *inst,
                   bool scalar, int num_ops, int op_index,
                   int dst_element_index, int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(T);

  using Interleave = struct {
    int register_num;
    int source_arg;
  };
  const Interleave interleave_start[2][4] = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                                             {{3, 0}, {0, 1}, {1, 1}, {2, 1}}};
  const Interleave interleave_end[2][4] = {{{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                                           {{0, 1}, {1, 1}, {2, 1}, {3, 1}}};
  // Get the elements from the left up to `index`.
  // For the horizontal mode, it treats the stripmine `vm` register based on
  // `vs2` as a contiguous block, and only the LAST `index` elements from
  // stripmine vm register based on `vs1` will be used AT THE BEGINNING.
  //
  // For the vertical mode, each stripmine vector register `op_index` is mapped
  // separatedly. it mimics the imaging tiling process shift of
  //   |--------|--------|
  //   | 4xVLEN | 4xVLEN |
  //   |  (vs1) |  (vs2) |
  //   |--------|--------|
  // The vertical mode can also support the non-stripmine version to handle
  // the last columns of the image.
  if (dst_element_index < index) {
    auto src_element_index =
        interleave_start[horizontal][op_index].register_num *
            elts_per_register +
        dst_element_index - index + elts_per_register;
    return GetInstructionSource<T>(
        inst, interleave_start[horizontal][op_index].source_arg,
        src_element_index);
  }

  auto src_element_index =
      interleave_end[horizontal][op_index].register_num * elts_per_register +
      dst_element_index - index;
  return GetInstructionSource<T>(
      inst, interleave_end[horizontal][op_index].source_arg, src_element_index);
}

// Slide previous register vertically by index.
template <typename T>
void KelvinVSlidevp(int index, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, false /* scalar */, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(absl::bind_front(
          VSlidepOpGetArg1<T>, false /* horizontal */, index)));
}
template void KelvinVSlidevp<int8_t>(int, bool, Instruction *);
template void KelvinVSlidevp<int16_t>(int, bool, Instruction *);
template void KelvinVSlidevp<int32_t>(int, bool, Instruction *);

// Slide previous register horizontally by index.
template <typename T>
void KelvinVSlidehp(int index, Instruction *inst) {
  KelvinBinaryVectorOp(
      inst, false /* scalar */, true /* strip_mine */,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(
          absl::bind_front(VSlidepOpGetArg1<T>, true /* horizontal */, index)));
}
template void KelvinVSlidehp<int8_t>(int, Instruction *);
template void KelvinVSlidehp<int16_t>(int, Instruction *);
template void KelvinVSlidehp<int32_t>(int, Instruction *);

template <typename T>
void KelvinVSel(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSel(bool scalar, bool strip_mine, Instruction *inst) {
  // Select lanes from two operands with vector selection boolean.
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>(
          [](T vd, T vs1, T vs2) -> T { return vs1 & 1 ? vd : vs2; }));
}
template void KelvinVSel<int8_t>(bool, bool, Instruction *);
template void KelvinVSel<int16_t>(bool, bool, Instruction *);
template void KelvinVSel<int32_t>(bool, bool, Instruction *);

// Returns even elements of concatenated registers.
template <typename T>
T VEvnOpGetArg1(const Instruction *inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const int elts_per_register = vector_size_in_bytes / sizeof(T);

  auto src_element_index =
      op_index * elts_per_register * 2 + dst_element_index * 2;
  const int elts_per_src = elts_per_register * num_ops;

  if (src_element_index < elts_per_src) {
    return GetInstructionSource<T>(inst, 0, src_element_index);
  }

  return GetInstructionSource<T>(inst, 1,
                                 scalar ? 0 : src_element_index - elts_per_src);
}

template <typename T>
void KelvinVEvn(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VEvnOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VEvnOpGetArg1<T>));
}
template void KelvinVEvn<int8_t>(bool, bool, Instruction *);
template void KelvinVEvn<int16_t>(bool, bool, Instruction *);
template void KelvinVEvn<int32_t>(bool, bool, Instruction *);

// Returns odd elements of concatenated registers.
template <typename T>
T VOddOpGetArg1(const Instruction *inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const int elts_per_register = vector_size_in_bytes / sizeof(T);

  auto src_element_index =
      op_index * elts_per_register * 2 + dst_element_index * 2 + 1;
  const int elts_per_src = elts_per_register * num_ops;

  if (src_element_index < elts_per_src) {
    return GetInstructionSource<T>(inst, 0, src_element_index);
  }

  return GetInstructionSource<T>(inst, 1,
                                 scalar ? 0 : src_element_index - elts_per_src);
}

template <typename T>
void KelvinVOdd(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VOddOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VOddOpGetArg1<T>));
}
template void KelvinVOdd<int8_t>(bool, bool, Instruction *);
template void KelvinVOdd<int16_t>(bool, bool, Instruction *);
template void KelvinVOdd<int32_t>(bool, bool, Instruction *);

// Returns evn/odd elements of concatenated registers based on dst_reg_index.
template <typename T>
T VEvnoddOpGetArg1(const Instruction *inst, bool scalar, int num_ops,
                   int op_index, int dst_element_index, int dst_reg_index) {
  return dst_reg_index == 0
             ? VEvnOpGetArg1<T>(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index)
             : VOddOpGetArg1<T>(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index);
}

template <typename T>
void KelvinVEvnodd(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VEvnoddOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VEvnoddOpGetArg1<T>));
}
template void KelvinVEvnodd<int8_t>(bool, bool, Instruction *);
template void KelvinVEvnodd<int16_t>(bool, bool, Instruction *);
template void KelvinVEvnodd<int32_t>(bool, bool, Instruction *);

// Interleave even/odd lanes of two operands.
// Returns even elements of concatenated registers.
template <typename T>
T VZipOpGetArg1(const Instruction *inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const int elts_per_register = vector_size_in_bytes / sizeof(T);
  const int half_elts_per_register = elts_per_register / 2;

  // Only takes the even elements. For the stripmine version, the offset are
  // counted as half of the register size.
  auto src_element_index = op_index * half_elts_per_register +
                           dst_element_index / 2 +
                           dst_reg_index * half_elts_per_register * num_ops;

  if (dst_element_index & 1) {
    return GetInstructionSource<T>(inst, 1, scalar ? 0 : src_element_index);
  }

  return GetInstructionSource<T>(inst, 0, src_element_index);
}

template <typename T>
void KelvinVZip(bool scalar, bool strip_mine, Instruction *inst) {
  KelvinBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VZipOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VZipOpGetArg1<T>));
}
template void KelvinVZip<int8_t>(bool, bool, Instruction *);
template void KelvinVZip<int16_t>(bool, bool, Instruction *);
template void KelvinVZip<int32_t>(bool, bool, Instruction *);
}  // namespace kelvin::sim
