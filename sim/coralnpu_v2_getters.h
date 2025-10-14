// Copyright 2025 Google LLC
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

#ifndef SIM_CORALNPU_V2_GETTERS_H_
#define SIM_CORALNPU_V2_GETTERS_H_

#include <cstdint>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_encoding_common.h"
#include "riscv/riscv_getter_helpers.h"
#include "riscv/riscv_getters_vector.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"

namespace coralnpu::sim {
using ::mpact::sim::generic::DestinationOperandInterface;
using ::mpact::sim::generic::ImmediateOperand;
using ::mpact::sim::generic::IntLiteralOperand;
using ::mpact::sim::generic::operator*;  // NOLINT
using ::mpact::sim::generic::SourceOperandInterface;
using ::mpact::sim::riscv::GetVectorRegisterSourceOp;
using ::mpact::sim::riscv::Insert;
using ::mpact::sim::riscv::kFRegisterAliases;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscVCsrInterface;
using ::mpact::sim::riscv::RiscVEncodingCommon;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorTrueOperand;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::RVVectorRegister;

using SourceOpGetterMap =
    absl::flat_hash_map<int, absl::AnyInvocable<SourceOperandInterface*()>>;
using DestOpGetterMap =
    absl::flat_hash_map<int,
                        absl::AnyInvocable<DestinationOperandInterface*(int)>>;

template <typename SourceOpEnum, typename Extractors>
void AddCoralNPUV2SourceGetters(SourceOpGetterMap& getter_map,
                                RiscVEncodingCommon* /*absl_nonnull*/ common) {
  // Source operand getters.
  Insert(getter_map, *SourceOpEnum::kBImm12,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::Inst32Format::ExtractBImm(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kCSRUimm5,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<uint32_t>(
               Extractors::Inst32Format::ExtractIUimm5(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kConst1, []() -> SourceOperandInterface* {
    return new ImmediateOperand<uint32_t>(1);
  });
  Insert(getter_map, *SourceOpEnum::kConst2, []() -> SourceOperandInterface* {
    return new ImmediateOperand<uint32_t>(2);
  });
  Insert(getter_map, *SourceOpEnum::kConst4, []() -> SourceOperandInterface* {
    return new ImmediateOperand<uint32_t>(4);
  });
  Insert(getter_map, *SourceOpEnum::kCsr,
         [common]() -> SourceOperandInterface* {
           uint16_t csr_index =
               Extractors::Inst32Format::ExtractUImm12(common->inst_word());
           absl::StatusOr<RiscVCsrInterface*> csr_status =
               common->state()->csr_set()->GetCsr(csr_index);
           if (!csr_status.ok()) {
             return new ImmediateOperand<uint32_t>(csr_index);
           } else {
             return new ImmediateOperand<uint32_t>(csr_index,
                                                   (*csr_status)->name());
           }
         });
  Insert(getter_map, *SourceOpEnum::kFrs1,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::RType::ExtractRs1(common->inst_word());
           return GetRegisterSourceOp<RVFpRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               kFRegisterAliases[num]);
         });
  Insert(getter_map, *SourceOpEnum::kFrs2,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::RType::ExtractRs2(common->inst_word());
           return GetRegisterSourceOp<RVFpRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               kFRegisterAliases[num]);
         });
  Insert(getter_map, *SourceOpEnum::kFrs3,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::R4Type::ExtractRs3(common->inst_word());
           return GetRegisterSourceOp<RVFpRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               kFRegisterAliases[num]);
         });
  Insert(getter_map, *SourceOpEnum::kIImm12,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::IType::ExtractImm12(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kIUimm5,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<uint32_t>(
               Extractors::IType::ExtractIUimm5(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kJImm12,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::Inst32Format::ExtractImm12(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kJImm20,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::JType::ExtractJImm(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kNf, [common]() -> SourceOperandInterface* {
    int num_fields = Extractors::VMem::ExtractNf(common->inst_word());
    return new ImmediateOperand<uint8_t>(num_fields,
                                         absl::StrCat(num_fields + 1));
  });
  Insert(getter_map, *SourceOpEnum::kPred,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<uint32_t>(
               Extractors::Fence::ExtractPred(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kRUimm5,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<uint32_t>(
               Extractors::RType::ExtractRUimm5(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kRm, [common]() -> SourceOperandInterface* {
    int rm = Extractors::RType::ExtractFunc3(common->inst_word());
    switch (rm) {
      case 0:
        return new IntLiteralOperand<0>();
      case 1:
        return new IntLiteralOperand<1>();
      case 2:
        return new IntLiteralOperand<2>();
      case 3:
        return new IntLiteralOperand<3>();
      case 4:
        return new IntLiteralOperand<4>();
      case 5:
        return new IntLiteralOperand<5>();
      case 6:
        return new IntLiteralOperand<6>();
      case 7:
        return new IntLiteralOperand<7>();
      default:
        return nullptr;
    }
  });
  Insert(getter_map, *SourceOpEnum::kRs1,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::RType::ExtractRs1(common->inst_word());
           if (num == 0) return new IntLiteralOperand<0>({1});
           return GetRegisterSourceOp<RV32Register>(
               common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
               kXRegisterAliases[num]);
         });
  Insert(getter_map, *SourceOpEnum::kRs2,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::RType::ExtractRs2(common->inst_word());
           if (num == 0) return new IntLiteralOperand<0>({1});
           return GetRegisterSourceOp<RV32Register>(
               common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
               kXRegisterAliases[num]);
         });
  Insert(getter_map, *SourceOpEnum::kSImm12,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::SType::ExtractSImm(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kSimm5,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::VArith::ExtractSimm5(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kSucc,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<uint32_t>(
               Extractors::Fence::ExtractSucc(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kUImm20,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::UType::ExtractUImm(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kUimm5,
         [common]() -> SourceOperandInterface* {
           return new ImmediateOperand<int32_t>(
               Extractors::VArith::ExtractUimm5(common->inst_word()));
         });
  Insert(getter_map, *SourceOpEnum::kVd, [common]() -> SourceOperandInterface* {
    return GetVectorRegisterSourceOp<RVVectorRegister>(
        common->state(), Extractors::VArith::ExtractVd(common->inst_word()));
  });
  Insert(getter_map, *SourceOpEnum::kVm, [common]() -> SourceOperandInterface* {
    int vm = Extractors::VArith::ExtractVm(common->inst_word());
    return new ImmediateOperand<bool>(vm, absl::StrCat("vm.", vm ? "t" : "f"));
  });
  Insert(getter_map, *SourceOpEnum::kVmask,
         [common]() -> SourceOperandInterface* {
           int vm = Extractors::VArith::ExtractVm(common->inst_word());
           if (vm) {
             // Unmasked, return the True mask.
             return new RV32VectorTrueOperand(common->state());
           }
           // Masked. Return the mask register.
           return GetVectorMaskRegisterSourceOp<RVVectorRegister>(
               common->state(), 0);
         });
  Insert(getter_map, *SourceOpEnum::kVmaskTrue,
         [common]() -> SourceOperandInterface* {
           return new RV32VectorTrueOperand(common->state());
         });
  Insert(getter_map, *SourceOpEnum::kVs1,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VArith::ExtractVs1(common->inst_word());
           return GetVectorRegisterSourceOp<RVVectorRegister>(common->state(),
                                                              num);
         });
  Insert(getter_map, *SourceOpEnum::kVs2,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VArith::ExtractVs2(common->inst_word());
           return GetVectorRegisterSourceOp<RVVectorRegister>(common->state(),
                                                              num);
         });
  Insert(getter_map, *SourceOpEnum::kVs3,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VMem::ExtractVs3(common->inst_word());
           return GetVectorRegisterSourceOp<RVVectorRegister>(common->state(),
                                                              num);
         });
  Insert(getter_map, *SourceOpEnum::kZimm10,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VConfig::ExtractZimm10(common->inst_word());
           return new ImmediateOperand<int32_t>(num);
         });
  Insert(getter_map, *SourceOpEnum::kZimm11,
         [common]() -> SourceOperandInterface* {
           int num = Extractors::VConfig::ExtractZimm11(common->inst_word());
           return new ImmediateOperand<int32_t>(num);
         });
}

template <typename DestOpEnum, typename Extractors>
void AddCoralNPUV2DestGetters(DestOpGetterMap& getter_map,
                              RiscVEncodingCommon* /*absl_nonnull*/ common) {
  // Destination operand getters.
  Insert(getter_map, *DestOpEnum::kCsr,
         [common](int latency) -> DestinationOperandInterface* {
           return GetRegisterDestinationOp<RV32Register>(
               common->state(), RiscVState::kCsrName, latency);
         });
  Insert(getter_map, *DestOpEnum::kFflags,
         [common](int latency) -> DestinationOperandInterface* {
           return GetCSRSetBitsDestinationOp<uint32_t>(common->state(),
                                                       "fflags", latency, "");
         });
  Insert(getter_map, *DestOpEnum::kFrd,
         [common](int latency) -> DestinationOperandInterface* {
           int num = Extractors::RType::ExtractRd(common->inst_word());
           return GetRegisterDestinationOp<RVFpRegister>(
               common->state(), absl::StrCat(RiscVState::kFregPrefix, num),
               latency, kFRegisterAliases[num]);
         });
  Insert(getter_map, *DestOpEnum::kNextPc,
         [common](int latency) -> DestinationOperandInterface* {
           return GetRegisterDestinationOp<RV32Register>(
               common->state(), RiscVState::kPcName, latency);
         });
  Insert(getter_map, *DestOpEnum::kRd,
         [common](int latency) -> DestinationOperandInterface* {
           int num = Extractors::RType::ExtractRd(common->inst_word());
           if (num == 0) {
             return GetRegisterDestinationOp<RV32Register>(common->state(),
                                                           "X0Dest", 0);
           } else {
             return GetRegisterDestinationOp<RV32Register>(
                 common->state(), absl::StrCat(RiscVState::kXregPrefix, num),
                 latency, kXRegisterAliases[num]);
           }
         });
  Insert(getter_map, *DestOpEnum::kVd,
         [common](int latency) -> DestinationOperandInterface* {
           int num = Extractors::VArith::ExtractVd(common->inst_word());
           return GetVectorRegisterDestinationOp<RVVectorRegister>(
               common->state(), latency, num);
         });
}

}  // namespace coralnpu::sim

#endif  // SIM_CORALNPU_V2_GETTERS_H_
