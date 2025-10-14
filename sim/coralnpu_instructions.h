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

#ifndef SIM_CORALNPU_INSTRUCTIONS_H_
#define SIM_CORALNPU_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// We define this empty namespace and using it so that coralnpu_encoder
// can successfully resolve definitions for generic RiscV semfuncs.
namespace mpact::sim::riscv {}
using namespace mpact::sim::riscv;  // NOLINT

namespace coralnpu::sim {

void CoralNPUIllegalInstruction(mpact::sim::generic::Instruction* inst);

void CoralNPUNopInstruction(mpact::sim::generic::Instruction* inst);

void CoralNPUIMpause(const mpact::sim::generic::Instruction* inst);

void CoralNPULogInstruction(int log_mode,
                            mpact::sim::generic::Instruction* inst);

template <typename T>
void CoralNPUIStore(mpact::sim::generic::Instruction* inst);

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_INSTRUCTIONS_H_
