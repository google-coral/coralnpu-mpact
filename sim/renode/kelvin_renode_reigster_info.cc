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

#include "sim/renode/kelvin_renode_register_info.h"
#include "riscv/riscv_debug_info.h"
#include "mpact/sim/generic/type_helpers.h"

namespace kelvin::sim {

using ::mpact::sim::generic::operator*;  // NOLINT: used below (clange error).

KelvinRenodeRegisterInfo *KelvinRenodeRegisterInfo::instance_ = nullptr;

void KelvinRenodeRegisterInfo::InitializeRenodeRegisterInfo() {
  using DbgReg = mpact::sim::riscv::DebugRegisterEnum;

  renode_register_info_ = {
      {*DbgReg::kPc, 32, true, false},  {*DbgReg::kX0, 32, true, true},
      {*DbgReg::kX1, 32, true, false},  {*DbgReg::kX2, 32, true, false},
      {*DbgReg::kX3, 32, true, false},  {*DbgReg::kX4, 32, true, false},
      {*DbgReg::kX5, 32, true, false},  {*DbgReg::kX6, 32, true, false},
      {*DbgReg::kX7, 32, true, false},  {*DbgReg::kX8, 32, true, false},
      {*DbgReg::kX9, 32, true, false},  {*DbgReg::kX10, 32, true, false},
      {*DbgReg::kX11, 32, true, false}, {*DbgReg::kX12, 32, true, false},
      {*DbgReg::kX13, 32, true, false}, {*DbgReg::kX14, 32, true, false},
      {*DbgReg::kX15, 32, true, false}, {*DbgReg::kX16, 32, true, false},
      {*DbgReg::kX17, 32, true, false}, {*DbgReg::kX18, 32, true, false},
      {*DbgReg::kX19, 32, true, false}, {*DbgReg::kX20, 32, true, false},
      {*DbgReg::kX21, 32, true, false}, {*DbgReg::kX22, 32, true, false},
      {*DbgReg::kX23, 32, true, false}, {*DbgReg::kX24, 32, true, false},
      {*DbgReg::kX25, 32, true, false}, {*DbgReg::kX26, 32, true, false},
      {*DbgReg::kX27, 32, true, false}, {*DbgReg::kX28, 32, true, false},
      {*DbgReg::kX29, 32, true, false}, {*DbgReg::kX30, 32, true, false},
      {*DbgReg::kX31, 32, true, false},
  };
}

KelvinRenodeRegisterInfo::KelvinRenodeRegisterInfo() {
  InitializeRenodeRegisterInfo();
}

const KelvinRenodeRegisterInfo::RenodeRegisterInfo &
KelvinRenodeRegisterInfo::GetRenodeRegisterInfo() {
  return Instance()->GetRenodeRegisterInfoPrivate();
}

KelvinRenodeRegisterInfo *KelvinRenodeRegisterInfo::Instance() {
  if (instance_ == nullptr) {
    instance_ = new KelvinRenodeRegisterInfo();
  }
  return instance_;
}

const KelvinRenodeRegisterInfo::RenodeRegisterInfo &
KelvinRenodeRegisterInfo::GetRenodeRegisterInfoPrivate() {
  return renode_register_info_;
}

}  // namespace kelvin::sim
