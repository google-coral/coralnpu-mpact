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

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_DEBUG_INTERFACE_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_DEBUG_INTERFACE_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mpact/sim/generic/core_debug_interface.h"

namespace kelvin::sim::renode {

// This structure mirrors the one defined in renode to provide information
// about the target registers. Do not change, as it maps to the marshaling
// structure used by renode.
struct RenodeCpuRegister {
  int index;
  int width;
  bool is_general;
  bool is_read_only;
};

class RenodeDebugInterface : public mpact::sim::generic::CoreDebugInterface {
 public:
  // These using declarations are required to tell the compiler that these
  // methods are not overridden nor hidden by the two virtual methods declared
  // in this class.
  using mpact::sim::generic::CoreDebugInterface::ReadRegister;
  using mpact::sim::generic::CoreDebugInterface::WriteRegister;
  // Read/write the numeric id registers.
  virtual absl::StatusOr<uint64_t> ReadRegister(uint32_t reg_id) = 0;
  virtual absl::Status WriteRegister(uint32_t reg_id, uint64_t value) = 0;
  // Get register information.
  virtual int32_t GetRenodeRegisterInfoSize() const = 0;
  virtual absl::Status GetRenodeRegisterInfo(int32_t index, int32_t max_len,
                                             char *name,
                                             RenodeCpuRegister &info) = 0;

  virtual absl::Status LoadImage(const std::string &image_path,
                                 uint64_t start_address) = 0;
};

}  // namespace kelvin::sim::renode

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_RENODE_DEBUG_INTERFACE_H_
