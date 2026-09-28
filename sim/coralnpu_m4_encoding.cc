// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sim/coralnpu_m4_encoding.h"

#include <cstdint>

#include "sim/coralnpu_m4_bin_decoder.h"
#include "sim/coralnpu_v2_encoding_template.h"

namespace coralnpu::sim {

void CoralNPUM4Encoding::ParseInstruction(uint32_t inst_word) {
  CoralNPUV2EncodingTemplate::ParseInstruction(
      inst_word, ::coralnpu::sim::encoding_m4::DecodeCoralNPUM4Inst32);
}

}  // namespace coralnpu::sim
