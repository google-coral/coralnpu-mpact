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

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_REGISTER_INFO_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_REGISTER_INFO_H_

#include <vector>

#include "sim/renode/renode_debug_interface.h"

namespace coralnpu::sim {

// This file defines a class that is used to store the register information
// that needs to be provided to renode for the CoralNPU registers.

class CoralNPURenodeRegisterInfo {
 public:
  using RenodeRegisterInfo = std::vector<renode::RenodeCpuRegister>;
  static const RenodeRegisterInfo& GetRenodeRegisterInfo();

 private:
  CoralNPURenodeRegisterInfo();
  static CoralNPURenodeRegisterInfo* Instance();
  void InitializeRenodeRegisterInfo();
  const RenodeRegisterInfo& GetRenodeRegisterInfoPrivate();

  static CoralNPURenodeRegisterInfo* instance_;
  RenodeRegisterInfo renode_register_info_;
};

}  // namespace coralnpu::sim

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_REGISTER_INFO_H_
