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

#ifndef SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_
#define SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_

#include <cstdint>

#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

using mpact::sim::generic::Instruction;

// Command structure for the convolution instruction.
typedef struct KelvinVConvCmd {
  uint32_t mode : 2;    // 31:30
  uint32_t start : 5;   // 29:25
  uint32_t stop : 5;    // 24:20
  uint32_t sbias1 : 9;  // 19:11
  uint32_t sdata1 : 1;  // 10
  uint32_t sbias2 : 9;  // 9:1
  uint32_t sdata2 : 1;  // 0
} vconv_cmd_t;

// Command structure for the depthwise convolution instruction.
typedef struct KelvinVDwconvCmd {
  uint32_t mode : 2;      // 1:0
  uint32_t sparsity : 2;  // 3:2
  uint32_t regbase : 4;   // 7:4
  uint32_t rsvd : 4;      // 11:8
  int32_t sbias1 : 9;     // 20:12
  uint32_t sdata1 : 1;    // 21
  int32_t sbias2 : 9;     // 30:22
  uint32_t sdata2 : 1;    // 31
} vdwconv_u8_t;

void KelvinVConv(Instruction *inst);

void KelvinVDwconv(bool write_acc, Instruction *inst);

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_
