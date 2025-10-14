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

#include "sim/coralnpu_vector_instructions.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>

#include "sim/coralnpu_state.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"

namespace coralnpu::sim {

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
Vs1 CommonBinaryOpGetArg1(const Instruction* inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
Vs2 CommonBinaryOpGetArg2(const Instruction* inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs2);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs2) +
                           dst_reg_index;
  return GetInstructionSource<Vs2>(inst, 1, scalar ? 0 : src_element_index);
}

template <typename T, typename Vd, typename Vs1, typename Vs2>
using SourceArgGetter =
    std::function<T(const Instruction* inst, bool scalar, int num_ops,
                    int op_index, int dst_element_index, int dst_reg_index)>;

template <bool halftype = false, bool widen_dst = false, typename Vd,
          typename Vs1, typename Vs2, typename... VDArgs>
void CoralNPUBinaryVectorOp(const Instruction* inst, bool scalar,
                            bool strip_mine,
                            std::function<Vd(VDArgs..., Vs1, Vs2)> op,
                            SourceArgGetter<Vs1, Vd, Vs1, Vs2> arg1_getter =
                                CommonBinaryOpGetArg1<Vd, Vs1, Vs2>,
                            SourceArgGetter<Vs2, Vd, Vs1, Vs2> arg2_getter =
                                CommonBinaryOpGetArg2<Vd, Vs1, Vs2>) {
  auto state = static_cast<CoralNPUState*>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_dest_register = vector_size_in_bytes / sizeof(Vd);

  // For coralnpu, stripmining issues 4 contiguous vector ops.
  auto num_ops = strip_mine ? 4 : 1;
  constexpr bool is_widen_op =
      (sizeof(Vd) > sizeof(Vs2) && !halftype) || widen_dst;
  // Widening requires 2 destination regs per op.
  constexpr size_t dest_regs_per_op = is_widen_op ? 2 : 1;
  // Special case for VADD3 op which is adding dest value to vs1 + vs2.
  constexpr bool is_reading_dest = sizeof...(VDArgs) == 1;
  auto vd = static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));

  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer* dest_db[dest_regs_per_op];
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
void CoralNPUUnaryVectorOp(const Instruction* inst, bool strip_mine,
                           std::function<Vd(Vs)> op,
                           SourceArgGetter<Vs, Vd, Vs, Vs> arg_getter =
                               CommonBinaryOpGetArg1<Vd, Vs, Vs>) {
  auto state = static_cast<CoralNPUState*>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_dest_register = vector_size_in_bytes / sizeof(Vd);

  // For coralnpu, stripmining issues 4 contiguous vector ops.
  auto num_ops = strip_mine ? 4 : 1;
  auto vd = static_cast<RV32VectorDestinationOperand*>(inst->Destination(0));

  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer* dest_db = vd->AllocateDataBuffer(op_index);
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
void CoralNPUVAdd(bool scalar, bool strip_mine, Instruction* inst) {
  // Return vs1 + vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           using UT = typename std::make_unsigned<T>::type;
                           // Cast to unsigned type before the operation to
                           // avoid undefined overflow behavior in intx_t.
                           UT uvs1 = static_cast<UT>(vs1);
                           UT uvs2 = static_cast<UT>(vs2);
                           return static_cast<T>(uvs1 + uvs2);
                         }));
}
template void CoralNPUVAdd<int8_t>(bool, bool, Instruction*);
template void CoralNPUVAdd<int16_t>(bool, bool, Instruction*);
template void CoralNPUVAdd<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVSub(bool scalar, bool strip_mine, Instruction* inst) {
  // Return vs1 - vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           using UT = typename std::make_unsigned<T>::type;
                           // Cast to unsigned type before the operation to
                           // avoid undefined overflow behavior in intx_t.
                           UT uvs1 = static_cast<UT>(vs1);
                           UT uvs2 = static_cast<UT>(vs2);
                           return static_cast<T>(uvs1 - uvs2);
                         }));
}
template void CoralNPUVSub<int8_t>(bool, bool, Instruction*);
template void CoralNPUVSub<int16_t>(bool, bool, Instruction*);
template void CoralNPUVSub<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVRSub(bool scalar, bool strip_mine, Instruction* inst) {
  // Return vs2 - vs1.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           using UT = typename std::make_unsigned<T>::type;
                           // Cast to unsigned type before the operation to
                           // avoid undefined overflow behavior in intx_t.
                           UT uvs1 = static_cast<UT>(vs1);
                           UT uvs2 = static_cast<UT>(vs2);
                           return static_cast<T>(uvs2 - uvs1);
                         }));
}
template void CoralNPUVRSub<int8_t>(bool, bool, Instruction*);
template void CoralNPUVRSub<int16_t>(bool, bool, Instruction*);
template void CoralNPUVRSub<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVEq(bool scalar, bool strip_mine, Instruction* inst) {
  // Return 1 if vs1 and vs2 are equal, else returns 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 == vs2; }));
}
template void CoralNPUVEq<int8_t>(bool, bool, Instruction*);
template void CoralNPUVEq<int16_t>(bool, bool, Instruction*);
template void CoralNPUVEq<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVNe(bool scalar, bool strip_mine, Instruction* inst) {
  // Return 1 if vs1 and vs2 are not equal, else return 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 != vs2; }));
}
template void CoralNPUVNe<int8_t>(bool, bool, Instruction*);
template void CoralNPUVNe<int16_t>(bool, bool, Instruction*);
template void CoralNPUVNe<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVLt(bool scalar, bool strip_mine, Instruction* inst) {
  // Returns 1 if vs1 < vs2, else return 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 < vs2; }));
}
template void CoralNPUVLt<int8_t>(bool, bool, Instruction*);
template void CoralNPUVLt<int16_t>(bool, bool, Instruction*);
template void CoralNPUVLt<int32_t>(bool, bool, Instruction*);
template void CoralNPUVLt<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVLt<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVLt<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVLe(bool scalar, bool strip_mine, Instruction* inst) {
  // Returns 1 if vs1 <= vs2, else return 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 <= vs2; }));
}
template void CoralNPUVLe<int8_t>(bool, bool, Instruction*);
template void CoralNPUVLe<int16_t>(bool, bool, Instruction*);
template void CoralNPUVLe<int32_t>(bool, bool, Instruction*);
template void CoralNPUVLe<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVLe<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVLe<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVGt(bool scalar, bool strip_mine, Instruction* inst) {
  // Returns 1 if vs1 > vs2, else return 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 > vs2; }));
}
template void CoralNPUVGt<int8_t>(bool, bool, Instruction*);
template void CoralNPUVGt<int16_t>(bool, bool, Instruction*);
template void CoralNPUVGt<int32_t>(bool, bool, Instruction*);
template void CoralNPUVGt<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVGt<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVGt<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVGe(bool scalar, bool strip_mine, Instruction* inst) {
  // Returns 1 if vs1 >= vs2, else return 0.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 >= vs2; }));
}
template void CoralNPUVGe<int8_t>(bool, bool, Instruction*);
template void CoralNPUVGe<int16_t>(bool, bool, Instruction*);
template void CoralNPUVGe<int32_t>(bool, bool, Instruction*);
template void CoralNPUVGe<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVGe<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVGe<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVAbsd(bool scalar, bool strip_mine, Instruction* inst) {
  // Returns the absolute difference between vs1 and vs2.
  // Note: for signed(INTx_MAX - INTx_MIN) the result will be UINTx_MAX.
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */,
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
template void CoralNPUVAbsd<int8_t>(bool, bool, Instruction*);
template void CoralNPUVAbsd<int16_t>(bool, bool, Instruction*);
template void CoralNPUVAbsd<int32_t>(bool, bool, Instruction*);
template void CoralNPUVAbsd<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVAbsd<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVAbsd<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVMax(bool scalar, bool strip_mine, Instruction* inst) {
  // Return the max of vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           return std::max(vs1, vs2);
                         }));
}
template void CoralNPUVMax<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMax<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMax<int32_t>(bool, bool, Instruction*);
template void CoralNPUVMax<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVMax<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVMax<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVMin(bool scalar, bool strip_mine, Instruction* inst) {
  // Return the min of vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           return std::min(vs1, vs2);
                         }));
}
template void CoralNPUVMin<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMin<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMin<int32_t>(bool, bool, Instruction*);
template void CoralNPUVMin<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVMin<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVMin<uint32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVAdd3(bool scalar, bool strip_mine, Instruction* inst) {
  // Return the summation of vd, vs1, and vs2.
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T,
                         T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using UT = typename std::make_unsigned<T>::type;
        UT uvs1 = static_cast<UT>(vs1);
        UT uvs2 = static_cast<UT>(vs2);
        UT uvd = static_cast<UT>(vd);
        return static_cast<T>(uvd + uvs1 + uvs2);
      }));
}
template void CoralNPUVAdd3<int8_t>(bool, bool, Instruction*);
template void CoralNPUVAdd3<int16_t>(bool, bool, Instruction*);
template void CoralNPUVAdd3<int32_t>(bool, bool, Instruction*);

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
void CoralNPUVAdds(bool scalar, bool strip_mine, Instruction* inst) {
  // Return saturated sum of vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>(VAddsHelper<T>));
}
template void CoralNPUVAdds<int8_t>(bool, bool, Instruction*);
template void CoralNPUVAdds<int16_t>(bool, bool, Instruction*);
template void CoralNPUVAdds<int32_t>(bool, bool, Instruction*);

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
void CoralNPUVAddsu(bool scalar, bool strip_mine, Instruction* inst) {
  // Return saturated sum of unsigned vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>(VAddsuHelper<T>));
}
template void CoralNPUVAddsu<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVAddsu<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVAddsu<uint32_t>(bool, bool, Instruction*);

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
void CoralNPUVSubs(bool scalar, bool strip_mine, Instruction* inst) {
  // Return saturated sub of vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>(VSubsHelper<T>));
}
template void CoralNPUVSubs<int8_t>(bool, bool, Instruction*);
template void CoralNPUVSubs<int16_t>(bool, bool, Instruction*);
template void CoralNPUVSubs<int32_t>(bool, bool, Instruction*);

template <typename T>
void CoralNPUVSubsu(bool scalar, bool strip_mine, Instruction* inst) {
  // Return saturated sub of unsigned vs1 and vs2.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           return vs1 < vs2 ? 0 : vs1 - vs2;
                         }));
}
template void CoralNPUVSubsu<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVSubsu<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVSubsu<uint32_t>(bool, bool, Instruction*);

template <typename Td, typename Ts>
void CoralNPUVAddw(bool scalar, bool strip_mine, Instruction* inst) {
  // Adds operands with widening.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                           return static_cast<Td>(vs1) + static_cast<Td>(vs2);
                         }));
}
template void CoralNPUVAddw<int16_t, int8_t>(bool, bool, Instruction*);
template void CoralNPUVAddw<int32_t, int16_t>(bool, bool, Instruction*);
template void CoralNPUVAddw<uint16_t, uint8_t>(bool, bool, Instruction*);
template void CoralNPUVAddw<uint32_t, uint16_t>(bool, bool, Instruction*);

template <typename Td, typename Ts>
void CoralNPUVSubw(bool scalar, bool strip_mine, Instruction* inst) {
  // Subtracts operands with widening.
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                           return static_cast<Td>(vs1) - static_cast<Td>(vs2);
                         }));
}
template void CoralNPUVSubw<int16_t, int8_t>(bool, bool, Instruction*);
template void CoralNPUVSubw<int32_t, int16_t>(bool, bool, Instruction*);
template void CoralNPUVSubw<uint16_t, uint8_t>(bool, bool, Instruction*);
template void CoralNPUVSubw<uint32_t, uint16_t>(bool, bool, Instruction*);

template <typename Td, typename Ts2>
void CoralNPUVAcc(bool scalar, bool strip_mine, Instruction* inst) {
  // Accumulates operands with widening.
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<Td(Td, Ts2)>([](Td vs1, Ts2 vs2) -> Td {
        using UTd = typename std::make_unsigned<Td>::type;
        return static_cast<Td>(static_cast<UTd>(vs1) + static_cast<UTd>(vs2));
      }));
}
template void CoralNPUVAcc<int16_t, int8_t>(bool, bool, Instruction*);
template void CoralNPUVAcc<int32_t, int16_t>(bool, bool, Instruction*);
template void CoralNPUVAcc<uint16_t, uint8_t>(bool, bool, Instruction*);
template void CoralNPUVAcc<uint32_t, uint16_t>(bool, bool, Instruction*);

template <typename Vd, typename Vs1, typename Vs2>
Vs1 PackedBinaryOpGetArg1(const Instruction* inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs1);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs1);
  return GetInstructionSource<Vs1>(inst, 0, src_element_index);
}

template <typename Vd, typename Vs1, typename Vs2>
Vs2 PackedBinaryOpGetArg2(const Instruction* inst, bool scalar, int num_ops,
                          int op_index, int dst_element_index,
                          int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  auto elts_per_register = vector_size_in_bytes / sizeof(Vs2);
  auto src_element_index = op_index * elts_per_register +
                           dst_element_index * sizeof(Vd) / sizeof(Vs2) + 1;
  return GetInstructionSource<Vs2>(inst, 0, src_element_index);
}

template <typename Td, typename Ts>
void CoralNPUVPadd(bool strip_mine, Instruction* inst) {
  // Adds lane pairs.
  CoralNPUBinaryVectorOp<true /* halftype */, false /* widen_dst */, Td, Ts,
                         Ts>(
      inst, false /* scalar */, strip_mine,
      std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
        return static_cast<Td>(vs1) + static_cast<Td>(vs2);
      }),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg1<Td, Ts, Ts>),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg2<Td, Ts, Ts>));
}
template void CoralNPUVPadd<int16_t, int8_t>(bool, Instruction*);
template void CoralNPUVPadd<int32_t, int16_t>(bool, Instruction*);
template void CoralNPUVPadd<uint16_t, uint8_t>(bool, Instruction*);
template void CoralNPUVPadd<uint32_t, uint16_t>(bool, Instruction*);

template <typename Td, typename Ts>
void CoralNPUVPsub(bool strip_mine, Instruction* inst) {
  // Subtracts lane pairs.
  CoralNPUBinaryVectorOp<true /* halftype */, false /* widen_dst */, Td, Ts,
                         Ts>(
      inst, false /* scalar */, strip_mine,
      std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
        return static_cast<Td>(vs1) - static_cast<Td>(vs2);
      }),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg1<Td, Ts, Ts>),
      SourceArgGetter<Ts, Td, Ts, Ts>(PackedBinaryOpGetArg2<Td, Ts, Ts>));
}
template void CoralNPUVPsub<int16_t, int8_t>(bool, Instruction*);
template void CoralNPUVPsub<int32_t, int16_t>(bool, Instruction*);
template void CoralNPUVPsub<uint16_t, uint8_t>(bool, Instruction*);
template void CoralNPUVPsub<uint32_t, uint16_t>(bool, Instruction*);

// Halving addition with optional rounding bit.
template <typename T>
T CoralNPUVHaddHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  return static_cast<T>(
      (static_cast<WT>(vs1) + static_cast<WT>(vs2) + (round ? 1 : 0)) >> 1);
}

template <typename T>
void CoralNPUVHadd(bool scalar, bool strip_mine, bool round,
                   Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&CoralNPUVHaddHelper<T>, round)));
}
template void CoralNPUVHadd<int8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHadd<int16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHadd<int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHadd<uint8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHadd<uint16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHadd<uint32_t>(bool, bool, bool, Instruction*);

// Halving subtraction with optional rounding bit.
template <typename T>
T CoralNPUVHsubHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  return static_cast<T>(
      (static_cast<WT>(vs1) - static_cast<WT>(vs2) + (round ? 1 : 0)) >> 1);
}

template <typename T>
void CoralNPUVHsub(bool scalar, bool strip_mine, bool round,
                   Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&CoralNPUVHsubHelper<T>, round)));
}
template void CoralNPUVHsub<int8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHsub<int16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHsub<int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHsub<uint8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHsub<uint16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVHsub<uint32_t>(bool, bool, bool, Instruction*);

// Bitwise and.
template <typename T>
void CoralNPUVAnd(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 & vs2; }));
}
template void CoralNPUVAnd<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVAnd<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVAnd<uint32_t>(bool, bool, Instruction*);

// Bitwise or.
template <typename T>
void CoralNPUVOr(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 | vs2; }));
}
template void CoralNPUVOr<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVOr<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVOr<uint32_t>(bool, bool, Instruction*);

// Bitwise xor.
template <typename T>
void CoralNPUVXor(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1 ^ vs2; }));
}
template void CoralNPUVXor<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVXor<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVXor<uint32_t>(bool, bool, Instruction*);

// Generalized reverse using bit ladder.
template <typename T>
void CoralNPUVRev(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
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
template void CoralNPUVRev<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVRev<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVRev<uint32_t>(bool, bool, Instruction*);

// Cyclic rotation right using a bit ladder.
template <typename T>
void CoralNPUVRor(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine, std::function<T(T, T)>([](T vs1, T vs2) -> T {
        T r = vs1;
        T count = vs2 & static_cast<T>(sizeof(T) * 8 - 1);
        for (auto shift : {1, 2, 4, 8, 16}) {
          if (count & shift) r = (r >> shift) | (r << (sizeof(T) * 8 - shift));
        }
        return r;
      }));
}
template void CoralNPUVRor<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVRor<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVRor<uint32_t>(bool, bool, Instruction*);

// Returns Arg1 as either vs1 or vs2 based on dst_reg_index.
template <typename Vd, typename Vs1, typename Vs2>
Vs1 VMvpOpGetArg1(const Instruction* inst, bool scalar, int num_ops,
                  int op_index, int dst_element_index, int dst_reg_index) {
  return dst_reg_index == 0
             ? CommonBinaryOpGetArg1<Vd, Vs1, Vs2>(
                   inst, scalar, num_ops, op_index, dst_element_index, 0)
             : CommonBinaryOpGetArg2<Vd, Vs1, Vs2>(
                   inst, scalar, num_ops, op_index, dst_element_index, 0);
}

// Copies a pair of registers.
template <typename T>
void CoralNPUVMvp(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VMvpOpGetArg1<T, T, T>),
      // Arg2 isn't used. We provide a custom getter here because the default
      // getter expects extra source registers for widening ops.
      SourceArgGetter<T, T, T, T>(VMvpOpGetArg1<T, T, T>));
}
template void CoralNPUVMvp<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVMvp<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVMvp<uint32_t>(bool, bool, Instruction*);

// Logical shift left.
template <typename T>
void CoralNPUVSll(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           size_t shift = vs2 & (sizeof(T) * 8 - 1);
                           return vs1 << shift;
                         }));
}
template void CoralNPUVSll<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVSll<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVSll<uint32_t>(bool, bool, Instruction*);

// Arithmetic shift right.
template <typename T>
void CoralNPUVSra(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           size_t shift = vs2 & (sizeof(T) * 8 - 1);
                           return vs1 >> shift;
                         }));
}
template void CoralNPUVSra<int8_t>(bool, bool, Instruction*);
template void CoralNPUVSra<int16_t>(bool, bool, Instruction*);
template void CoralNPUVSra<int32_t>(bool, bool, Instruction*);

// Logical shift right.
template <typename T>
void CoralNPUVSrl(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>([](T vs1, T vs2) -> T {
                           size_t shift = vs2 & (sizeof(T) * 8 - 1);
                           return vs1 >> shift;
                         }));
}
template void CoralNPUVSrl<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVSrl<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVSrl<uint32_t>(bool, bool, Instruction*);

// Logical and arithmetic left/right shift with saturating shift amount and
// result.
template <typename T>
T CoralNPUVShiftHelper(bool round, T vs1, T vs2) {
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
void CoralNPUVShift(bool round, bool scalar, bool strip_mine,
                    Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>(absl::bind_front(
                             &CoralNPUVShiftHelper<T>, round)));
}
template void CoralNPUVShift<int8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVShift<int16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVShift<int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVShift<uint8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVShift<uint16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVShift<uint32_t>(bool, bool, bool, Instruction*);

// Bitwise not.
template <typename T>
void CoralNPUVNot(bool strip_mine, Instruction* inst) {
  CoralNPUUnaryVectorOp(inst, strip_mine,
                        std::function<T(T)>([](T vs) -> T { return ~vs; }));
}
template void CoralNPUVNot<int32_t>(bool, Instruction*);

// Count the leading bits.
template <typename T>
void CoralNPUVClb(bool strip_mine, Instruction* inst) {
  CoralNPUUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                          return (vs & (1u << (sizeof(T) * 8 - 1)))
                                     ? absl::countl_one(vs)
                                     : absl::countl_zero(vs);
                        }));
}
template void CoralNPUVClb<uint8_t>(bool, Instruction*);
template void CoralNPUVClb<uint16_t>(bool, Instruction*);
template void CoralNPUVClb<uint32_t>(bool, Instruction*);

// Count the leading zeros.
template <typename T>
void CoralNPUVClz(bool strip_mine, Instruction* inst) {
  CoralNPUUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                          return absl::countl_zero(vs);
                        }));
}
template void CoralNPUVClz<uint8_t>(bool, Instruction*);
template void CoralNPUVClz<uint16_t>(bool, Instruction*);
template void CoralNPUVClz<uint32_t>(bool, Instruction*);

// Count the set bits.
template <typename T>
void CoralNPUVCpop(bool strip_mine, Instruction* inst) {
  CoralNPUUnaryVectorOp(inst, strip_mine, std::function<T(T)>([](T vs) -> T {
                          return absl::popcount(vs);
                        }));
}
template void CoralNPUVCpop<uint8_t>(bool, Instruction*);
template void CoralNPUVCpop<uint16_t>(bool, Instruction*);
template void CoralNPUVCpop<uint32_t>(bool, Instruction*);

// Move a register.
template <typename T>
void CoralNPUVMv(bool strip_mine, Instruction* inst) {
  CoralNPUUnaryVectorOp(inst, strip_mine,
                        std::function<T(T)>([](T vs) -> T { return vs; }));
}
template void CoralNPUVMv<int32_t>(bool, Instruction*);

// Alternates Vs1 register used for odd/even destination indices.
template <typename Vd, typename Vs1, typename Vs2>
Vs1 VSransOpGetArg1(const Instruction* inst, bool scalar, int num_ops,
                    int op_index, int dst_element_index, int dst_reg_index) {
  static_assert(2 * sizeof(Vd) == sizeof(Vs1) || 4 * sizeof(Vd) == sizeof(Vs1));
  auto state = static_cast<CoralNPUState*>(inst->state());
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
Td CoralNPUVSransHelper(bool round, Ts vs1, Td vs2) {
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
void CoralNPUVSrans(bool round, bool scalar, bool strip_mine,
                    Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<Td(Ts, Td)>(
          absl::bind_front(&CoralNPUVSransHelper<Td, Ts>, round)),
      SourceArgGetter<Ts, Td, Ts, Td>(VSransOpGetArg1<Td, Ts, Td>));
}
template void CoralNPUVSrans<int8_t, int16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVSrans<int16_t, int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVSrans<uint8_t, uint16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVSrans<uint16_t, uint32_t>(bool, bool, bool,
                                                 Instruction*);
template void CoralNPUVSrans<int8_t, int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVSrans<uint8_t, uint32_t>(bool, bool, bool, Instruction*);

// Multiplication of vector elements.
template <typename T>
void CoralNPUVMul(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine, std::function<T(T, T)>([](T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;

        return static_cast<T>(static_cast<WT>(vs1) * static_cast<WT>(vs2));
      }));
}
template void CoralNPUVMul<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMul<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMul<int32_t>(bool, bool, Instruction*);

// Multiplication of vector elements with saturation.
template <typename T>
void CoralNPUVMuls(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
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
template void CoralNPUVMuls<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMuls<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMuls<int32_t>(bool, bool, Instruction*);
template void CoralNPUVMuls<uint8_t>(bool, bool, Instruction*);
template void CoralNPUVMuls<uint16_t>(bool, bool, Instruction*);
template void CoralNPUVMuls<uint32_t>(bool, bool, Instruction*);

// Multiplication of vector elements with widening.
template <typename Td, typename Ts>
void CoralNPUVMulw(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<Td(Ts, Ts)>([](Ts vs1, Ts vs2) -> Td {
                           return static_cast<Td>(vs1) * static_cast<Td>(vs2);
                         }));
}
template void CoralNPUVMulw<int16_t, int8_t>(bool, bool, Instruction*);
template void CoralNPUVMulw<int32_t, int16_t>(bool, bool, Instruction*);
template void CoralNPUVMulw<uint16_t, uint8_t>(bool, bool, Instruction*);
template void CoralNPUVMulw<uint32_t, uint16_t>(bool, bool, Instruction*);

// Multiplication of vector elements with widening and optional rounding.
// Returns high half.
template <typename T>
T CoralNPUVMulhHelper(bool round, T vs1, T vs2) {
  using WT = typename mpact::sim::generic::WideType<T>::type;
  constexpr int n = sizeof(T) * 8;

  WT result = static_cast<WT>(vs1) * static_cast<WT>(vs2);
  result += round ? static_cast<WT>(1ll << (n - 1)) : 0;
  return result >> n;
}

template <typename T>
void CoralNPUVMulh(bool scalar, bool strip_mine, bool round,
                   Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, scalar, strip_mine,
      std::function<T(T, T)>(absl::bind_front(&CoralNPUVMulhHelper<T>, round)));
}
template void CoralNPUVMulh<int8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVMulh<int16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVMulh<int32_t>(bool, bool, bool, Instruction*);
template void CoralNPUVMulh<uint8_t>(bool, bool, bool, Instruction*);
template void CoralNPUVMulh<uint16_t>(bool, bool, bool, Instruction*);
template void CoralNPUVMulh<uint32_t>(bool, bool, bool, Instruction*);

// Saturating signed doubling multiply returning high half with optional
// rounding.
template <typename T>
T CoralNPUVDmulhHelper(bool round, bool round_neg, T vs1, T vs2) {
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
void CoralNPUVDmulh(bool scalar, bool strip_mine, bool round, bool round_neg,
                    Instruction* inst) {
  CoralNPUBinaryVectorOp(inst, scalar, strip_mine,
                         std::function<T(T, T)>(absl::bind_front(
                             &CoralNPUVDmulhHelper<T>, round, round_neg)));
}
template void CoralNPUVDmulh<int8_t>(bool, bool, bool, bool, Instruction*);
template void CoralNPUVDmulh<int16_t>(bool, bool, bool, bool, Instruction*);
template void CoralNPUVDmulh<int32_t>(bool, bool, bool, bool, Instruction*);

// Multiply accumulate.
template <typename T>
void CoralNPUVMacc(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T,
                         T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;
        return static_cast<WT>(vd) +
               static_cast<WT>(vs1) * static_cast<WT>(vs2);
      }));
}
template void CoralNPUVMacc<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMacc<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMacc<int32_t>(bool, bool, Instruction*);

// Multiply add.
template <typename T>
void CoralNPUVMadd(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T,
                         T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>([](T vd, T vs1, T vs2) -> T {
        using WT = typename mpact::sim::generic::WideType<T>::type;
        return static_cast<WT>(vs1) +
               static_cast<WT>(vd) * static_cast<WT>(vs2);
      }));
}
template void CoralNPUVMadd<int8_t>(bool, bool, Instruction*);
template void CoralNPUVMadd<int16_t>(bool, bool, Instruction*);
template void CoralNPUVMadd<int32_t>(bool, bool, Instruction*);

// Computes slide index for next register and takes result from either vs1 or
// vs2.
template <typename T>
T VSlidenOpGetArg1(bool horizontal, int index, const Instruction* inst,
                   bool scalar, int num_ops, int op_index,
                   int dst_element_index, int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
void CoralNPUVSlidevn(int index, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, false /* scalar */, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(absl::bind_front(
          VSlidenOpGetArg1<T>, false /* horizontal */, index)));
}
template void CoralNPUVSlidevn<int8_t>(int, bool, Instruction*);
template void CoralNPUVSlidevn<int16_t>(int, bool, Instruction*);
template void CoralNPUVSlidevn<int32_t>(int, bool, Instruction*);

// Slide next register horizontally by index.
template <typename T>
void CoralNPUVSlidehn(int index, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, false /* scalar */, true /* strip_mine */,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(
          absl::bind_front(VSlidenOpGetArg1<T>, true /* horizontal */, index)));
}
template void CoralNPUVSlidehn<int8_t>(int, Instruction*);
template void CoralNPUVSlidehn<int16_t>(int, Instruction*);
template void CoralNPUVSlidehn<int32_t>(int, Instruction*);

// Computes slide index for previous register and takes result from either vs1
// or vs2.
template <typename T>
T VSlidepOpGetArg1(bool horizontal, int index, const Instruction* inst,
                   bool scalar, int num_ops, int op_index,
                   int dst_element_index, int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
void CoralNPUVSlidevp(int index, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, false /* scalar */, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(absl::bind_front(
          VSlidepOpGetArg1<T>, false /* horizontal */, index)));
}
template void CoralNPUVSlidevp<int8_t>(int, bool, Instruction*);
template void CoralNPUVSlidevp<int16_t>(int, bool, Instruction*);
template void CoralNPUVSlidevp<int32_t>(int, bool, Instruction*);

// Slide previous register horizontally by index.
template <typename T>
void CoralNPUVSlidehp(int index, Instruction* inst) {
  CoralNPUBinaryVectorOp(
      inst, false /* scalar */, true /* strip_mine */,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(
          absl::bind_front(VSlidepOpGetArg1<T>, true /* horizontal */, index)));
}
template void CoralNPUVSlidehp<int8_t>(int, Instruction*);
template void CoralNPUVSlidehp<int16_t>(int, Instruction*);
template void CoralNPUVSlidehp<int32_t>(int, Instruction*);

template <typename T>
void CoralNPUVSel(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSel(bool scalar, bool strip_mine, Instruction* inst) {
  // Select lanes from two operands with vector selection boolean.
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T,
                         T>(
      inst, scalar, strip_mine,
      std::function<T(T, T, T)>(
          [](T vd, T vs1, T vs2) -> T { return vs1 & 1 ? vd : vs2; }));
}
template void CoralNPUVSel<int8_t>(bool, bool, Instruction*);
template void CoralNPUVSel<int16_t>(bool, bool, Instruction*);
template void CoralNPUVSel<int32_t>(bool, bool, Instruction*);

// Returns even elements of concatenated registers.
template <typename T>
T VEvnOpGetArg1(const Instruction* inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
void CoralNPUVEvn(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VEvnOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VEvnOpGetArg1<T>));
}
template void CoralNPUVEvn<int8_t>(bool, bool, Instruction*);
template void CoralNPUVEvn<int16_t>(bool, bool, Instruction*);
template void CoralNPUVEvn<int32_t>(bool, bool, Instruction*);

// Returns odd elements of concatenated registers.
template <typename T>
T VOddOpGetArg1(const Instruction* inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
void CoralNPUVOdd(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, false /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VOddOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VOddOpGetArg1<T>));
}
template void CoralNPUVOdd<int8_t>(bool, bool, Instruction*);
template void CoralNPUVOdd<int16_t>(bool, bool, Instruction*);
template void CoralNPUVOdd<int32_t>(bool, bool, Instruction*);

// Returns evn/odd elements of concatenated registers based on dst_reg_index.
template <typename T>
T VEvnoddOpGetArg1(const Instruction* inst, bool scalar, int num_ops,
                   int op_index, int dst_element_index, int dst_reg_index) {
  return dst_reg_index == 0
             ? VEvnOpGetArg1<T>(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index)
             : VOddOpGetArg1<T>(inst, scalar, num_ops, op_index,
                                dst_element_index, dst_reg_index);
}

template <typename T>
void CoralNPUVEvnodd(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VEvnoddOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VEvnoddOpGetArg1<T>));
}
template void CoralNPUVEvnodd<int8_t>(bool, bool, Instruction*);
template void CoralNPUVEvnodd<int16_t>(bool, bool, Instruction*);
template void CoralNPUVEvnodd<int32_t>(bool, bool, Instruction*);

// Interleave even/odd lanes of two operands.
// Returns even elements of concatenated registers.
template <typename T>
T VZipOpGetArg1(const Instruction* inst, bool scalar, int num_ops, int op_index,
                int dst_element_index, int dst_reg_index) {
  auto state = static_cast<CoralNPUState*>(inst->state());
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
void CoralNPUVZip(bool scalar, bool strip_mine, Instruction* inst) {
  CoralNPUBinaryVectorOp<false /* halftype */, true /* widen_dst */, T, T, T>(
      inst, scalar, strip_mine,
      std::function<T(T, T)>([](T vs1, T vs2) -> T { return vs1; }),
      SourceArgGetter<T, T, T, T>(VZipOpGetArg1<T>),
      SourceArgGetter<T, T, T, T>(VZipOpGetArg1<T>));
}
template void CoralNPUVZip<int8_t>(bool, bool, Instruction*);
template void CoralNPUVZip<int16_t>(bool, bool, Instruction*);
template void CoralNPUVZip<int32_t>(bool, bool, Instruction*);
}  // namespace coralnpu::sim
