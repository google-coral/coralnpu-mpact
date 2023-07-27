#ifndef SIM_TEST_KELVIN_VECTOR_INSTRUCTIONS_TEST_BASE_H_
#define SIM_TEST_KELVIN_VECTOR_INSTRUCTIONS_TEST_BASE_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sim/kelvin_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"
#include "mpact/sim/generic/state_item.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace kelvin::sim::test {

using absl::Span;
using mpact::sim::generic::Instruction;
using mpact::sim::generic::RegisterBase;
using mpact::sim::riscv::RV32Register;
using mpact::sim::riscv::RV32VectorDestinationOperand;
using mpact::sim::riscv::RV32VectorSourceOperand;
using mpact::sim::riscv::RVFpRegister;
using mpact::sim::riscv::RVVectorRegister;
using mpact::sim::util::FlatDemandMemory;

// Constants used in the tests.
constexpr uint32_t kInstAddress = 0x1000;
constexpr uint32_t kDataLoadAddress = 0x1'0000;
constexpr uint32_t kNumVectorRegister = 64;
constexpr char kRs1Name[] = "x1";
constexpr char kRs2Name[] = "x2";
constexpr int kRs1 = 1;
constexpr int kVd = 32;
constexpr int kVs1 = 8;
constexpr int kVs2 = 24;

// This is the base class for vector instruction test fixtures. It implements
// generic methods for testing and supporting testing of the RiscV vector
// instructions.
class KelvinVectorInstructionsTestBase : public testing::Test {
 public:
  KelvinVectorInstructionsTestBase() {
    memory_ = new FlatDemandMemory(0);
    state_ =
        new KelvinState("test", mpact::sim::riscv::RiscVXlen::RV32, memory_);
    // Initialize a portion of memory with a known pattern.
    auto *db = state_->db_factory()->Allocate(8192);
    auto span = db->Get<uint8_t>();
    for (int i = 0; i < 8192; i++) {
      span[i] = i & 0xff;
    }
    memory_->Store(kDataLoadAddress - 4096, db);
    db->DecRef();
    for (int i = 1; i < 32; i++) {
      xreg_[i] = state_->GetRegister<RV32Register>(absl::StrCat("x", i)).first;
    }
    for (int i = 1; i < kNumVectorRegister; i++) {
      vreg_[i] =
          state_->GetRegister<RVVectorRegister>(absl::StrCat("v", i)).first;
    }
  }

  template <typename T>
  absl::string_view KelvinTestTypeSuffix() {
    absl::string_view type_suffix = "Unknown";
    switch (sizeof(T)) {
      case 4:
        type_suffix = "W";
        break;
      case 2:
        type_suffix = "H";
        break;
      case 1:
        type_suffix = "B";
        break;
    }
    return type_suffix;
  }

  template <typename Vd, typename Vs1, typename Vs2>
  static std::pair<Vs1, Vs2> CommonBinaryOpArgsGetter(
      int num_ops, int op_num, int dest_reg_sub_index, int element_index,
      int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
      const std::vector<Vs1> &vs1_value, int vs2_size, bool s2_scalar,
      const std::vector<Vs2> &vs2_value, Vs2 rs2_value, bool halftype_op,
      bool vmvp_op) {
    auto src1_element_index =
        op_num * vs1_size + element_index * sizeof(Vd) / sizeof(Vs1);
    if (!vmvp_op) {
      if (widen_dst) {
        src1_element_index += (src1_widen_factor > 1 ? num_ops * vs1_size : 1) *
                              dest_reg_sub_index;
      } else if (src1_widen_factor == 2) {
        src1_element_index += element_index & 1 ? num_ops * vs1_size : 0;
      } else if (src1_widen_factor == 4) {
        const int interleave[4] = {0, 2, 1, 3};
        src1_element_index +=
            interleave[element_index & 3] * num_ops * vs1_size;
      }
    }

    auto src2_element_index = op_num * vs2_size +
                              element_index * (widen_dst && !vmvp_op ? 2 : 1) +
                              (vmvp_op ? 0 : 1) * dest_reg_sub_index;
    Vs1 arg1 = vs1_value[src1_element_index];
    Vs2 arg2 = halftype_op ? vs1_value[src1_element_index + 1]
                           : vs2_value[src2_element_index];
    arg2 = s2_scalar ? rs2_value : arg2;
    if (vmvp_op && dest_reg_sub_index == 1) {
      arg1 = arg2;
    }

    return {arg1, arg2};
  }

  template <typename Vd, typename Vs1, typename Vs2>
  using BinaryOpsArgsGetter =
      std::function<decltype(CommonBinaryOpArgsGetter<Vd, Vs1, Vs2>)>;

  // Helper function for testing vector-vector instructions.
  template <typename Vd, typename Vs1, typename Ts2, typename... VDArgs>
  void BinaryOpTestHelper(Instruction::SemanticFunction fcn,
                          absl::string_view name, bool s2_scalar,
                          bool strip_mine,
                          std::function<Vd(VDArgs..., Vs1, Ts2)> operation,
                          BinaryOpsArgsGetter<Vd, Vs1, Ts2> args_getter,
                          bool halftype_op, bool vmvp_op, bool widen_dst) {
    auto instruction = CreateInstruction();
    instruction->set_semantic_function(fcn);

    const uint32_t num_ops = strip_mine ? 4 : 1;
    constexpr int src1_widen_factor = sizeof(Vs1) / sizeof(Ts2);
    static_assert(src1_widen_factor == 1 || src1_widen_factor == 2 ||
                  src1_widen_factor == 4);

    // Half type ops don't use s2, so s2_scalar should be false.
    if (halftype_op && s2_scalar) {
      GTEST_FAIL();
    }

    if (s2_scalar) {
      AppendVectorRegisterOperands(instruction.get(), num_ops,
                                   src1_widen_factor, kVs1, {}, widen_dst,
                                   {kVd});
      AppendRegisterOperands(instruction.get(), {kRs1Name}, {});
    } else if (halftype_op) {
      AppendVectorRegisterOperands(instruction.get(), num_ops,
                                   src1_widen_factor, kVs1, {}, widen_dst,
                                   {kVd});
    } else {
      AppendVectorRegisterOperands(instruction.get(), num_ops,
                                   src1_widen_factor, kVs1, {kVs2}, widen_dst,
                                   {kVd});
    }

    // Initialize input values.
    const auto vector_length_in_bytes = state_->vector_length() / 8;
    int vs1_size = vector_length_in_bytes / sizeof(Vs1);
    const size_t vs1_regs_count = num_ops * src1_widen_factor;
    std::vector<Vs1> vs1_value(vs1_size * vs1_regs_count);
    // Use the first 4 elements to check the min/max boundary behavior
    vs1_value[0] = std::numeric_limits<Vs1>::lowest();
    vs1_value[1] = std::numeric_limits<Vs1>::lowest();
    vs1_value[2] = std::numeric_limits<Vs1>::max();
    vs1_value[3] = std::numeric_limits<Vs1>::max();
    auto vs1_span = absl::Span<Vs1>(vs1_value);
    FillArrayWithRandomValues<Vs1>(vs1_span.subspan(4, vs1_span.size() - 4));
    for (int i = 0; i < vs1_regs_count; i++) {
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      SetVectorRegisterValues<Vs1>(
          {{vs1_name, vs1_span.subspan(vs1_size * i, vs1_size)}});
    }

    int vs2_size = vector_length_in_bytes / sizeof(Ts2);
    std::vector<Ts2> vs2_value(vs2_size * num_ops);
    Ts2 rs2_value = 0;

    if (s2_scalar) {
      // Generate a new rs2 value.
      RV32Register::ValueType rs2_reg_value =
          RandomValue<RV32Register::ValueType>();
      SetRegisterValues<RV32Register::ValueType>({{kRs1Name, rs2_reg_value}});
      // Cast the value to the appropriate width, sign-extending if needed.
      rs2_value = static_cast<Ts2>(
          static_cast<typename mpact::sim::riscv::SameSignedType<
              RV32Register::ValueType, Ts2>::type>(rs2_reg_value));
    } else if (!halftype_op) {
      // Use the value slightly greater than min so VShift won't complain
      // -shamt.
      vs2_value[0] = std::numeric_limits<Ts2>::lowest() + 1;
      vs2_value[1] = std::numeric_limits<Ts2>::max();
      vs2_value[2] = std::numeric_limits<Ts2>::lowest() + 1;
      vs2_value[3] = std::numeric_limits<Ts2>::max();
      auto vs2_span = absl::Span<Ts2>(vs2_value);
      FillArrayWithRandomValues<Ts2>(vs2_span.subspan(4, vs2_span.size() - 4));
      for (int i = 0; i < num_ops; i++) {
        auto vs2_name = absl::StrCat("v", kVs2 + i);
        SetVectorRegisterValues<Ts2>(
            {{vs2_name, vs2_span.subspan(vs2_size * i, vs2_size)}});
      }
    }

    const size_t dest_regs_per_op = widen_dst ? 2 : 1;
    const size_t vd_size = vector_length_in_bytes / sizeof(Vd);
    const size_t dest_regs_count = num_ops * dest_regs_per_op;
    std::vector<Vd> vd_value(vd_size * dest_regs_count);
    vd_value[0] = std::numeric_limits<Vd>::lowest();
    vd_value[1] = std::numeric_limits<Vd>::max();
    vd_value[2] = std::numeric_limits<Vd>::lowest();
    vd_value[3] = std::numeric_limits<Vd>::max();
    auto vd_span = absl::Span<Vd>(vd_value);
    FillArrayWithRandomValues<Vd>(vd_span.subspan(4, vd_span.size() - 4));
    for (int i = 0; i < dest_regs_count; i++) {
      auto vd_name = absl::StrCat("v", kVd + i);
      SetVectorRegisterValues<Vd>(
          {{vd_name, vd_span.subspan(vd_size * i, vd_size)}});
    }

    // Executing instruction.
    instruction->Execute();

    // Check if ops gives the same result as vd.
    for (int op_num = 0; op_num < num_ops; op_num++) {
      for (int dest_reg_sub_index = 0; dest_reg_sub_index < dest_regs_per_op;
           dest_reg_sub_index++) {
        auto dest_reg_index = dest_reg_sub_index * num_ops + op_num;
        auto dest_vreg_num = kVd + dest_reg_index;
        auto dest_reg = vreg_[dest_vreg_num];
        auto dest_span = dest_reg->data_buffer()->Get<Vd>();

        for (int element_index = 0; element_index < vd_size; element_index++) {
          auto args = args_getter(
              num_ops, op_num, dest_reg_sub_index, element_index, vd_size,
              widen_dst, src1_widen_factor, vs1_size, vs1_value, vs2_size,
              s2_scalar, vs2_value, rs2_value, halftype_op, vmvp_op);

          auto dst_element_index = dest_reg_index * vd_size + element_index;
          auto expected_value = BinaryOpInvoke(
              operation, vd_value[dst_element_index], args.first, args.second);
          EXPECT_EQ(expected_value, dest_span[element_index])
              << absl::StrCat(name, "[", dst_element_index, "] != reg[",
                              dest_vreg_num, "*", element_index, "]");
        }
      }
    }
  }

  template <typename Vd, typename Vs1, typename Ts2, typename... VDArgs>
  void BinaryOpTestHelper(Instruction::SemanticFunction fcn,
                          absl::string_view name, bool s2_scalar,
                          bool strip_mine,
                          std::function<Vd(VDArgs..., Vs1, Ts2)> operation,
                          bool halftype_op = false, bool vmvp_op = false) {
    const bool widen_dst =
        (sizeof(Vd) > sizeof(Ts2) && !halftype_op) || vmvp_op;
    BinaryOpTestHelper<Vd, Vs1, Ts2, VDArgs...>(
        fcn, name, s2_scalar, strip_mine, operation,
        CommonBinaryOpArgsGetter<Vd, Vs1, Ts2>, halftype_op, vmvp_op,
        widen_dst);
  }

  template <typename Vd, typename Vs1, typename Ts2>
  void BinaryOpTestHelper(Instruction::SemanticFunction fcn,
                          absl::string_view name, bool s2_scalar,
                          bool strip_mine,
                          std::function<Vd(Vd, Vs1, Ts2)> operation) {
    BinaryOpTestHelper<Vd, Vs1, Ts2, Vd>(fcn, name, s2_scalar, strip_mine,
                                         operation);
  }

  // Helper function for testing single vector argument instructions.
  template <typename Vd, typename Vs>
  void UnaryOpTestHelper(Instruction::SemanticFunction fcn,
                         absl::string_view name, bool strip_mine,
                         std::function<Vd(Vs)> operation) {
    auto instruction = CreateInstruction();
    instruction->set_semantic_function(fcn);

    const uint32_t num_ops = strip_mine ? 4 : 1;

    AppendVectorRegisterOperands(instruction.get(), num_ops,
                                 1 /* src1_widen_factor */, kVs1, {},
                                 false /* widen_dst */, {kVd});

    // Initialize input values.
    const auto vector_length_in_bytes = state_->vector_length() / 8;
    int vs_size = vector_length_in_bytes / sizeof(Vs);
    const size_t vs_regs_count = num_ops;
    std::vector<Vs> vs_value(vs_size * vs_regs_count);
    auto vs_span = absl::Span<Vs>(vs_value);
    FillArrayWithRandomValues<Vs>(vs_span);
    for (int i = 0; i < vs_regs_count; i++) {
      auto vs1_name = absl::StrCat("v", kVs1 + i);
      SetVectorRegisterValues<Vs>(
          {{vs1_name, vs_span.subspan(vs_size * i, vs_size)}});
    }

    const size_t vd_size = vector_length_in_bytes / sizeof(Vd);
    const size_t dest_regs_count = num_ops;
    std::vector<Vd> vd_value(vd_size * dest_regs_count);
    auto vd_span = absl::Span<Vd>(vd_value);
    FillArrayWithRandomValues<Vd>(vd_span);
    for (int i = 0; i < dest_regs_count; i++) {
      auto vd_name = absl::StrCat("v", kVd + i);
      SetVectorRegisterValues<Vd>(
          {{vd_name, vd_span.subspan(vd_size * i, vd_size)}});
    }

    // Executing instruction.
    instruction->Execute();

    // Check if ops gives the same result as vd.
    for (int op_num = 0; op_num < num_ops; op_num++) {
      auto dest_reg_index = op_num;
      auto dest_vreg_num = kVd + dest_reg_index;
      auto dest_reg = vreg_[dest_vreg_num];
      auto dest_span = dest_reg->data_buffer()->Get<Vd>();

      for (int element_index = 0; element_index < vd_size; element_index++) {
        auto dst_element_index = dest_reg_index * vd_size + element_index;
        auto src1_element_index =
            op_num * vs_size + element_index * sizeof(Vd) / sizeof(Vs);

        Vs arg = vs_value[src1_element_index];
        auto expected_value = operation(arg);
        EXPECT_EQ(expected_value, dest_span[element_index])
            << absl::StrCat(name, "[", dst_element_index, "] != reg[",
                            dest_vreg_num, "*", element_index, "]");
      }
    }
  }

  ~KelvinVectorInstructionsTestBase() override {
    delete state_;
    delete memory_;
  }

 protected:
  // Helper function invoking vector operations which aren't reading Vd
  template <typename Vd, typename Vs1, typename Vs2>
  Vd BinaryOpInvoke(std::function<Vd(Vs1, Vs2)> op, Vd vd, Vs1 vs1, Vs2 vs2) {
    return op(vs1, vs2);
  }

  // Overloaded version which for operations reading Vd
  template <typename Vd, typename Vs1, typename Vs2>
  Vd BinaryOpInvoke(std::function<Vd(Vd, Vs1, Vs2)> op, Vd vd, Vs1 vs1,
                    Vs2 vs2) {
    return op(vd, vs1, vs2);
  }

  // Create a random value in the valid range for the type.
  template <typename T>
  T RandomValue() {
    return absl::Uniform(absl::IntervalClosed, bitgen_,
                         std::numeric_limits<T>::lowest(),
                         std::numeric_limits<T>::max());
  }

  // Fill the span with random values.
  template <typename T>
  void FillArrayWithRandomValues(absl::Span<T> span) {
    for (auto &val : span) {
      val = RandomValue<T>();
    }
  }

  // Set a vector register value. Takes a vector of tuples of register names and
  // spans of values, fetches each register and sets it to the corresponding
  // value.
  template <typename T>
  void SetVectorRegisterValues(
      const std::vector<std::tuple<std::string, Span<const T>>> &values) {
    for (auto &[vreg_name, span] : values) {
      auto *vreg = state_->GetRegister<RVVectorRegister>(vreg_name).first;
      auto *db = state_->db_factory()->MakeCopyOf(vreg->data_buffer());
      db->template Set<T>(span);
      vreg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Set the named registers to their corresponding value.
  template <typename T, typename RegisterType = RV32Register>
  void SetRegisterValues(
      const std::vector<std::tuple<std::string, const T>> &values) {
    for (auto &[reg_name, value] : values) {
      auto *reg = state_->GetRegister<RegisterType>(reg_name).first;
      auto *db =
          state_->db_factory()->Allocate<typename RegisterType::ValueType>(1);
      db->template Set<T>(0, value);
      reg->SetDataBuffer(db);
      db->DecRef();
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and appends them to the given instruction.
  void AppendRegisterOperands(Instruction *inst,
                              const std::vector<std::string> &sources,
                              const std::vector<std::string> &destinations) {
    for (auto &reg_name : sources) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendSource(reg->CreateSourceOperand());
    }
    for (auto &reg_name : destinations) {
      auto *reg = state_->GetRegister<RV32Register>(reg_name).first;
      inst->AppendDestination(reg->CreateDestinationOperand(0));
    }
  }

  // Creates source and destination scalar register operands for the registers
  // named in the two vectors and appends them to the given instruction.
  void AppendVectorRegisterOperands(Instruction *inst, const uint32_t num_ops,
                                    int src1_widen_factor, int src1_reg,
                                    const std::vector<int> &other_sources,
                                    bool widen_dst,
                                    const std::vector<int> &destinations) {
    {
      std::vector<RegisterBase *> reg_vec;
      auto regs_count = src1_widen_factor * num_ops;
      for (int i = 0; (i < regs_count) && (i + src1_reg < kNumVectorRegister);
           i++) {
        std::string reg_name = absl::StrCat("v", i + src1_reg);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto *op = new RV32VectorSourceOperand(
          absl::Span<RegisterBase *>(reg_vec), absl::StrCat("v", src1_reg));
      inst->AppendSource(op);
    }
    for (auto &reg_no : other_sources) {
      std::vector<RegisterBase *> reg_vec;
      for (int i = 0; (i < num_ops) && (i + reg_no < kNumVectorRegister); i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto *op = new RV32VectorSourceOperand(
          absl::Span<RegisterBase *>(reg_vec), absl::StrCat("v", reg_no));
      inst->AppendSource(op);
    }
    for (auto &reg_no : destinations) {
      std::vector<RegisterBase *> reg_vec;
      auto regs_count = widen_dst ? num_ops * 2 : num_ops;
      for (int i = 0; (i < regs_count) && (i + reg_no < kNumVectorRegister);
           i++) {
        std::string reg_name = absl::StrCat("v", i + reg_no);
        reg_vec.push_back(
            state_->GetRegister<RVVectorRegister>(reg_name).first);
      }
      auto *op = new RV32VectorDestinationOperand(
          absl::Span<RegisterBase *>(reg_vec), 0, absl::StrCat("v", reg_no));
      inst->AppendDestination(op);
    }
  }

  using InstructionPtr = std::unique_ptr<Instruction, void (*)(Instruction *)>;
  InstructionPtr CreateInstruction() {
    InstructionPtr inst(new Instruction(next_instruction_address_, state_),
                        [](Instruction *inst) { inst->DecRef(); });
    inst->set_size(4);
    next_instruction_address_ += 4;
    return inst;
  }

  RVVectorRegister *vreg_[kNumVectorRegister];
  RV32Register *xreg_[32];
  KelvinState *state_;
  FlatDemandMemory *memory_;
  absl::BitGen bitgen_;
  uint32_t next_instruction_address_ = kInstAddress;
};
}  // namespace kelvin::sim::test
#endif  // SIM_TEST_KELVIN_VECTOR_INSTRUCTIONS_TEST_BASE_H_
