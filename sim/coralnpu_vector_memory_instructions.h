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

#ifndef SIM_CORALNPU_VECTOR_MEMORY_INSTRUCTIONS_H_
#define SIM_CORALNPU_VECTOR_MEMORY_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace coralnpu::sim {

using mpact::sim::generic::Instruction;

template <typename T>
void CoralNPUVLd(bool has_length, bool has_stride, bool strip_mine,
                 Instruction* inst);

template <typename T>
void CoralNPUVLdRegWrite(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVSt(bool has_length, bool has_stride, bool strip_mine,
                 Instruction* inst);

template <typename T>
void CoralNPUVDup(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUVStQ(bool strip_mine, Instruction* inst);

template <typename T>
void CoralNPUGetVl(bool strip_mine, bool is_rs1, bool is_rs2,
                   const mpact::sim::generic::Instruction* inst);

void CoralNPUVcGet(const mpact::sim::generic::Instruction* inst);

void CoralNPUAcSet(bool is_transpose,
                   const mpact::sim::generic::Instruction* inst);

void CoralNPUADwInit(const mpact::sim::generic::Instruction* inst);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_VECTOR_MEMORY_INSTRUCTIONS_H_
