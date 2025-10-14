/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SIM_CORALNPU_VECTOR_INSTRUCTIONS_H_
#define SIM_CORALNPU_VECTOR_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace coralnpu::sim {

using mpact::sim::generic::Instruction;

// Vector 2-arg .vv, .vx arithmetic operations.
template <typename T>
void CoralNPUVAdd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSub(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVRSub(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVEq(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVNe(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVLt(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVLe(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVGt(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVGe(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVAbsd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMax(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMin(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVAdd3(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVAdds(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVAddsu(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSubs(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSubsu(bool scalar, bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVAddw(bool scalar, bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVSubw(bool scalar, bool strip_mine, Instruction* inst);

template <typename Td, typename Ts2>
void CoralNPUVAcc(bool scalar, bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVPadd(bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVPsub(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVHadd(bool scalar, bool strip_mine, bool round, Instruction* inst);

template <typename T>
void CoralNPUVHsub(bool scalar, bool strip_mine, bool round, Instruction* inst);

template <typename T>
void CoralNPUVAnd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVOr(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVXor(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVRev(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVRor(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMvp(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSll(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSra(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSrl(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVShift(bool round, bool scalar, bool strip_mine,
                    Instruction* inst);

template <typename T>
void CoralNPUVNot(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVClb(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVClz(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVCpop(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMv(bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVSrans(bool round, bool scalar, bool strip_mine,
                    Instruction* inst);

template <typename T>
void CoralNPUVMul(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMuls(bool scalar, bool strip_mine, Instruction* inst);

template <typename Td, typename Ts>
void CoralNPUVMulw(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMulh(bool scalar, bool strip_mine, bool round, Instruction* inst);

template <typename T>
void CoralNPUVDmulh(bool scalar, bool strip_mine, bool round, bool round_neg,
                    Instruction* inst);

template <typename T>
void CoralNPUVMacc(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVMadd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSlidevn(int index, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSlidehn(int index, Instruction* inst);

template <typename T>
void CoralNPUVSlidevp(int index, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSlidehp(int index, Instruction* inst);

template <typename T>
void CoralNPUVSel(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVEvn(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVOdd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVEvnodd(bool scalar, bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVZip(bool scalar, bool strip_mine, Instruction* inst);
}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_VECTOR_INSTRUCTIONS_H_
