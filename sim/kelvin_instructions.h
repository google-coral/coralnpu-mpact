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

#ifndef SIM_KELVIN_INSTRUCTIONS_H_
#define SIM_KELVIN_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

// We define this empty namespace and using it so that kelvin_encoder
// can successfully resolve definitions for generic RiscV semfuncs.
namespace mpact::sim::riscv {}
using namespace mpact::sim::riscv;  // NOLINT

namespace kelvin::sim {

void KelvinIllegalInstruction(mpact::sim::generic::Instruction* inst);

void KelvinNopInstruction(mpact::sim::generic::Instruction* inst);

void KelvinIMpause(const mpact::sim::generic::Instruction* inst);

void KelvinLogInstruction(int log_mode, mpact::sim::generic::Instruction* inst);

template <typename T>
void KelvinIStore(mpact::sim::generic::Instruction* inst);

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_INSTRUCTIONS_H_
