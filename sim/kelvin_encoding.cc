#include "sim/kelvin_encoding.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sim/kelvin_bin_decoder.h"
#include "sim/kelvin_decoder.h"
#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/immediate_operand.h"
#include "mpact/sim/generic/literal_operand.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/simple_resource.h"

namespace kelvin::sim::isa32 {

template <typename RegType>
inline void GetVRegGroup(
    KelvinState *state, int reg_num, bool strip_mine, int widen_factor,
    std::vector<mpact::sim::generic::RegisterBase *> *vreg_group) {
  auto regs_count = (strip_mine ? 4 : 1) * widen_factor;
  for (int i = 0; i < regs_count; ++i) {
    auto vreg_name =
        absl::StrCat(mpact::sim::riscv::RiscVState::kVregPrefix, reg_num + i);
    vreg_group->push_back(state->GetRegister<RegType>(vreg_name).first);
  }
}

template <typename RegType>
inline SourceOperandInterface *GetVectorRegisterSourceOp(KelvinState *state,
                                                         int reg_num,
                                                         bool strip_mine,
                                                         int widen_factor) {
  std::vector<mpact::sim::generic::RegisterBase *> vreg_group;
  GetVRegGroup<RegType>(state, reg_num, strip_mine, widen_factor, &vreg_group);
  auto *v_src_op = new mpact::sim::riscv::RV32VectorSourceOperand(
      absl::Span<mpact::sim::generic::RegisterBase *>(vreg_group),
      absl::StrCat(mpact::sim::riscv::RiscVState::kVregPrefix, reg_num));
  return v_src_op;
}

template <typename RegType>
inline DestinationOperandInterface *GetVectorRegisterDestinationOp(
    KelvinState *state, int reg_num, bool strip_mine, bool widening,
    int latency) {
  std::vector<mpact::sim::generic::RegisterBase *> vreg_group;
  GetVRegGroup<RegType>(state, reg_num, strip_mine, widening ? 2 : 1,
                        &vreg_group);
  auto *v_dst_op = new mpact::sim::riscv::RV32VectorDestinationOperand(
      absl::Span<mpact::sim::generic::RegisterBase *>(vreg_group), latency,
      absl::StrCat(mpact::sim::riscv::RiscVState::kVregPrefix, reg_num));
  return v_dst_op;
}

// Generic helper functions to create register operands.
template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(KelvinState *state,
                                                             std::string name,
                                                             int latency) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency);
}

template <typename RegType>
inline DestinationOperandInterface *GetRegisterDestinationOp(
    KelvinState *state, std::string name, int latency, std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  return reg->CreateDestinationOperand(latency, op_name);
}

template <typename T>
inline DestinationOperandInterface *GetCSRSetBitsDestinationOp(
    KelvinState *state, std::string name, int latency, std::string op_name) {
  auto result = state->csr_set()->GetCsr(name);
  if (!result.ok()) {
    LOG(ERROR) << "No such CSR '" << name << "'";
    return nullptr;
  }
  auto *csr = result.value();
  auto *op = csr->CreateSetDestinationOperand(latency, op_name);
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(KelvinState *state,
                                                   std::string name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand();
  return op;
}

template <typename RegType>
inline SourceOperandInterface *GetRegisterSourceOp(KelvinState *state,
                                                   std::string name,
                                                   std::string op_name) {
  auto *reg = state->GetRegister<RegType>(name).first;
  auto *op = reg->CreateSourceOperand(op_name);
  return op;
}

KelvinEncoding::KelvinEncoding(KelvinState *state) : state_(state) {
  InitializeSourceOperandGetters();
  InitializeDestinationOperandGetters();
  resource_pool_ = new mpact::sim::generic::SimpleResourcePool("Kelvin", 128);
}

KelvinEncoding::~KelvinEncoding() { delete resource_pool_; }

void KelvinEncoding::InitializeSourceOperandGetters() {
  // Source operand getters.
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kBImm12), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractBImm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCSRUimm5), [this]() {
        return new mpact::sim::generic::ImmediateOperand<uint32_t>(
            encoding::inst32_format::ExtractIUimm5(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kCsr), [this]() {
        auto csr_indx = encoding::i_type::ExtractUImm12(inst_word_);
        auto res = state_->csr_set()->GetCsr(csr_indx);
        if (!res.ok()) {
          return new mpact::sim::generic::ImmediateOperand<uint32_t>(csr_indx);
        }
        auto *csr = res.value();
        return new mpact::sim::generic::ImmediateOperand<uint32_t>(csr_indx,
                                                                   csr->name());
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIImm12), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractImm12(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kIUimm5), [this]() {
        return new mpact::sim::generic::ImmediateOperand<uint32_t>(
            encoding::inst32_format::ExtractRUimm5(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kJImm12), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractImm12(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kJImm20), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractJImm(inst_word_));
      }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kRs1),
      [this]() -> SourceOperandInterface * {
        int num = encoding::r_type::ExtractRs1(inst_word_);
        if (num == 0)
          return new mpact::sim::generic::IntLiteralOperand<0>({1},
                                                               xreg_alias_[0]);
        return GetRegisterSourceOp<mpact::sim::riscv::RV32Register>(
            state_,
            absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kRs2),
      [this]() -> SourceOperandInterface * {
        int num = encoding::r_type::ExtractRs2(inst_word_);
        if (num == 0)
          return new mpact::sim::generic::IntLiteralOperand<0>({1},
                                                               xreg_alias_[0]);
        return GetRegisterSourceOp<mpact::sim::riscv::RV32Register>(
            state_,
            absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, num),
            xreg_alias_[num]);
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kSImm12), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractSImm(inst_word_));
      }));
  source_op_getters_.insert(
      std::make_pair(static_cast<int>(SourceOpEnum::kUImm20), [this]() {
        return new mpact::sim::generic::ImmediateOperand<int32_t>(
            encoding::inst32_format::ExtractUImm(inst_word_));
      }));
  source_op_getters_.emplace(
      static_cast<int>(SourceOpEnum::kVs1),
      [this]() -> SourceOperandInterface * {
        auto reg_num = encoding::kelvin_v2_args_type::ExtractVs1(inst_word_);
        bool strip_mine = encoding::kelvin_v2_args_type::ExtractM(inst_word_);
        auto form = encoding::kelvin_v2_args_type::ExtractForm(inst_word_);
        // .xx form uses scalar xs1.
        if (form == 3) {
          if (reg_num == 0) {
            return new mpact::sim::generic::IntLiteralOperand<0>(
                {1}, xreg_alias_[0]);
          }
          // `vs1` is stored in bit[19:14], but scalar xs1 is in bit[19:15]
          // (same as the regular riscv32 encoding)
          reg_num >>= 1;
          return GetRegisterSourceOp<mpact::sim::riscv::RV32Register>(
              state_,
              absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, reg_num),
              xreg_alias_[reg_num]);
        }
        return GetVectorRegisterSourceOp<mpact::sim::riscv::RVVectorRegister>(
            state_, reg_num, strip_mine, GetSrc1WidenFactor());
      });
  source_op_getters_.emplace(
      static_cast<int>(SourceOpEnum::kVs2),
      [this]() -> SourceOperandInterface * {
        auto reg_num = encoding::kelvin_v2_args_type::ExtractVs2(inst_word_);
        bool strip_mine = encoding::kelvin_v2_args_type::ExtractM(inst_word_);
        auto form = encoding::kelvin_v2_args_type::ExtractForm(inst_word_);
        // .vx or .xx forms are using scalar xs2.
        if (form == 2 || form == 3) {
          if (reg_num == 0) {
            return new mpact::sim::generic::IntLiteralOperand<0>(
                {1}, xreg_alias_[0]);
          }
          // `vs2` is stored in bit[26:20], but scalar xs2 is in bit[25:20]
          // (same as in the regular riscv32 encoding)
          reg_num = reg_num & 0x1F;
          return GetRegisterSourceOp<mpact::sim::riscv::RV32Register>(
              state_,
              absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, reg_num),
              xreg_alias_[reg_num]);
        }
        return GetVectorRegisterSourceOp<mpact::sim::riscv::RVVectorRegister>(
            state_, reg_num, strip_mine, 1 /* widen_factor */);
      });
  source_op_getters_.emplace(
      // vst and vstq use `vd` field as the source for the vector store.
      static_cast<int>(SourceOpEnum::kVd),
      [this]() -> SourceOperandInterface * {
        auto reg_num = encoding::kelvin_v2_args_type::ExtractVd(inst_word_);
        bool strip_mine = encoding::kelvin_v2_args_type::ExtractM(inst_word_);
        if (opcode_ < OpcodeEnum::kVstBLXx || opcode_ > OpcodeEnum::kVstqWSpXxM)
          return nullptr;
        return GetVectorRegisterSourceOp<mpact::sim::riscv::RVVectorRegister>(
            state_, reg_num, strip_mine, 1 /* widen_factor */);
      });
  source_op_getters_.insert(std::make_pair(
      static_cast<int>(SourceOpEnum::kNone), []() { return nullptr; }));
}

void KelvinEncoding::InitializeDestinationOperandGetters() {
  // Destination operand getters.
  dest_op_getters_.insert(
      std::make_pair(static_cast<int>(DestOpEnum::kCsr), [this](int latency) {
        return GetRegisterDestinationOp<mpact::sim::riscv::RV32Register>(
            state_, KelvinState::kCsrName, latency);
      }));
  dest_op_getters_.insert(std::make_pair(
      static_cast<int>(DestOpEnum::kNextPc), [this](int latency) {
        return GetRegisterDestinationOp<mpact::sim::riscv::RV32Register>(
            state_, KelvinState::kPcName, latency);
      }));
  dest_op_getters_.insert(std::make_pair(
      static_cast<int>(DestOpEnum::kRd),
      [this](int latency) -> DestinationOperandInterface * {
        int num = encoding::r_type::ExtractRd(inst_word_);
        if (num == 0) {
          return GetRegisterDestinationOp<mpact::sim::riscv::RV32Register>(
              state_, "X0Dest", 0, xreg_alias_[0]);
        } else {
          return GetRegisterDestinationOp<mpact::sim::riscv::RVFpRegister>(
              state_, absl::StrCat(KelvinState::kXregPrefix, num), latency,
              xreg_alias_[num]);
        }
      }));
  dest_op_getters_.emplace(
      static_cast<int>(DestOpEnum::kVd),
      [this](int latency) -> DestinationOperandInterface * {
        auto reg_num = encoding::kelvin_v2_args_type::ExtractVd(inst_word_);
        bool strip_mine = encoding::kelvin_v2_args_type::ExtractM(inst_word_);
        return GetVectorRegisterDestinationOp<
            mpact::sim::riscv::RVVectorRegister>(state_, reg_num, strip_mine,
                                                 IsWidenDestinationRegisterOp(),
                                                 latency);
      });
  dest_op_getters_.insert(std::make_pair(
      static_cast<int>(DestOpEnum::kVs1),
      [this](int latency) -> DestinationOperandInterface * {
        auto reg_num = encoding::kelvin_v2_args_type::ExtractVs1(inst_word_);
        // Only vld.*p/vst.*p instructions are writing post incremented address
        // to "vs1" register. And it has to be a scalar register in that case.
        if (reg_num == 0) {
          return GetRegisterDestinationOp<mpact::sim::riscv::RV32Register>(
              state_, "X0Dest", 0, xreg_alias_[0]);
        } else {
          // `vs1` is stored in bit[19:14], but scalar xs1 is in bit[19:15]
          // (same as the regular riscv32 encoding)
          reg_num >>= 1;
          return GetRegisterDestinationOp<mpact::sim::riscv::RVFpRegister>(
              state_, absl::StrCat(KelvinState::kXregPrefix, reg_num), latency,
              xreg_alias_[reg_num]);
        }
      }));
  dest_op_getters_.insert(std::make_pair(static_cast<int>(DestOpEnum::kNone),
                                         [](int latency) { return nullptr; }));
}

// Parse the instruction word to determine the opcode.
void KelvinEncoding::ParseInstruction(uint32_t inst_word) {
  inst_word_ = inst_word;
  opcode_ = encoding::DecodeKelvinInst(inst_word_);
  if (opcode_ == OpcodeEnum::kNone)
    opcode_ = encoding::DecodeKelvinVectorInst(inst_word_);
}

DestinationOperandInterface *KelvinEncoding::GetDestination(SlotEnum, int,
                                                            OpcodeEnum opcode,
                                                            DestOpEnum dest_op,
                                                            int, int latency) {
  int index = static_cast<int>(dest_op);
  auto iter = dest_op_getters_.find(index);
  if (iter == dest_op_getters_.end()) {
    LOG(ERROR) << absl::StrCat("No getter for destination op enum value ",
                               index, "for instruction ",
                               kOpcodeNames[static_cast<int>(opcode)]);
    return nullptr;
  }
  return (iter->second)(latency);
}

SourceOperandInterface *KelvinEncoding::GetSource(SlotEnum, int,
                                                  OpcodeEnum opcode,
                                                  SourceOpEnum source_op,
                                                  int source_no) {
  int index = static_cast<int>(source_op);
  auto iter = source_op_getters_.find(index);
  if (iter == source_op_getters_.end()) {
    LOG(ERROR) << absl::StrCat("No getter for source op enum value ", index,
                               " for instruction ",
                               kOpcodeNames[static_cast<int>(opcode)]);
    return nullptr;
  }
  return (iter->second)();
}

bool KelvinEncoding::IsWidenDestinationRegisterOp() const {
  auto func1 = encoding::kelvin_v2_args_type::ExtractFunc1(inst_word_);
  auto func2 = encoding::kelvin_v2_args_type::ExtractFunc2(inst_word_);
  auto func2_ignore_unsigned = func2 & (~(1u << 0));
  // Func1 0b100 VAddw[u] and VSubw[u] need 2x destination registers.
  if ((func1 == 0b100) &&
      (func2_ignore_unsigned == 0b100 || func2_ignore_unsigned == 0b110)) {
    return true;
  }

  // Func1 0b001 VMvp also needs 2x destination registers.
  if ((func1 == 0b001) && (func2 == 0b1101)) {
    return true;
  }

  // Func1 0b011 VMulw[u] needs 2x destination registers.
  if ((func1 == 0b011) && (func2_ignore_unsigned == 0b0100)) {
    return true;
  }

  // Func1 0b110 VEvnodd and VZip needs 2x destination registers.
  if ((func1 == 0b110) && (func2 == 0b011010 || func2 == 0b011100)) {
    return true;
  }

  return false;
}

int KelvinEncoding::GetSrc1WidenFactor() const {
  auto func1 = encoding::kelvin_v2_args_type::ExtractFunc1(inst_word_);
  auto func2 = encoding::kelvin_v2_args_type::ExtractFunc2(inst_word_);
  auto sz = encoding::kelvin_v2_args_type::ExtractSz(inst_word_);
  auto func2_ignore_unsigned = func2 & (~(1u << 0));

  // Func1 0b100 VAcc[u] needs 2x src1 registers.
  if ((func1 == 0b100) && (func2_ignore_unsigned == 0b1010)) {
    return 2;
  }

  // Func1 0b010 VSrans[u][.r] also needs 2x src1 registers.
  if ((func1 == 0b010) && (sz == 0) &&
      (func2_ignore_unsigned == 0b010000 ||
       func2_ignore_unsigned == 0b010010)) {
    return 2;
  }

  // Func1 0b010 VSraqs[u][.r] needs 4x src1 registers.
  if ((func1 == 0b010) && (sz == 0) &&
      (func2_ignore_unsigned == 0b011000 ||
       func2_ignore_unsigned == 0b011010)) {
    return 4;
  }

  return 1;
}

}  // namespace kelvin::sim::isa32
