#include "sim/kelvin_vector_memory_instructions.h"

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "sim/test/kelvin_vector_instructions_test_base.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/instruction.h"

// This file contains the tests for testing kelvin vector memory instructions.

namespace {

using mpact::sim::generic::Instruction;

// Semantic functions.
using kelvin::sim::KelvinAcSet;
using kelvin::sim::KelvinADwInit;
using kelvin::sim::KelvinGetVl;
using kelvin::sim::KelvinVcGet;
using kelvin::sim::KelvinVLd;
using kelvin::sim::KelvinVLdRegWrite;
using kelvin::sim::KelvinVSt;
using kelvin::sim::KelvinVStQ;

class KelvinVectorMemoryInstructionsTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 public:
  template <typename T>
  void MemoryLoadStoreOpTestHelper(absl::string_view name, bool has_length,
                                   bool has_stride, bool strip_mine,
                                   bool post_increment, bool x_variant,
                                   bool is_load, bool is_quad) {
    InstructionPtr child_instruction(
        new Instruction(next_instruction_address_, state_),
        [](Instruction *inst) { inst->DecRef(); });
    child_instruction->set_size(4);
    auto instruction = CreateInstruction();

    if (is_load) {
      child_instruction->set_semantic_function(
          absl::bind_front(&KelvinVLdRegWrite<T>, strip_mine));
      instruction->set_semantic_function(
          absl::bind_front(&KelvinVLd<T>, has_length, has_stride, strip_mine));
      instruction->AppendChild(child_instruction.get());
    } else {
      if (is_quad) {
        instruction->set_semantic_function(
            absl::bind_front(&KelvinVStQ<T>, strip_mine));
      } else {
        instruction->set_semantic_function(absl::bind_front(
            &KelvinVSt<T>, has_length, has_stride, strip_mine));
      }
    }

    // Setup source and child instruction operands.
    const uint32_t num_ops = strip_mine ? 4 : 1;
    if (is_load) {
      AppendVectorRegisterOperands(
          child_instruction.get(), num_ops, 1 /* src1_widen_factor */, {}, {},
          false /* widen_dst */, {kelvin::sim::test::kVd});
    } else {  // Store
      AppendVectorRegisterOperands(
          instruction.get(), num_ops, 1 /* src1_widen_factor */,
          kelvin::sim::test::kVd, {}, false /* widen_dst */, {});
    }
    AppendRegisterOperands(instruction.get(), {kelvin::sim::test::kRs1Name},
                           {});
    if (!x_variant) {
      AppendRegisterOperands(instruction.get(), {kelvin::sim::test::kRs2Name},
                             {});
    }

    if (post_increment) {
      AppendRegisterOperands(instruction.get(), {},
                             {kelvin::sim::test::kRs1Name});
    }

    // x variant can't have length or stride fields.
    if (x_variant && (has_length || has_stride)) {
      GTEST_FAIL();
    }

    // xx variant can't have no length, no stride, and no post_increment
    // encoding
    if (!x_variant && !has_length && !has_stride && !post_increment) {
      GTEST_FAIL();
    }

    // length and stride fields can't coexist without post_increment
    if (has_length && has_stride && !post_increment) {
      GTEST_FAIL();
    }

    // Quad store need to have stride specified and no length
    if (is_quad && is_load) {
      GTEST_FAIL();
    }
    if ((is_quad && has_length) || (is_quad && !has_stride)) {
      GTEST_FAIL();
    }
    const uint32_t vector_length_in_bytes = state_->vector_length() / 8;
    const uint32_t vd_size = vector_length_in_bytes / sizeof(T);
    const uint32_t len_or_strides[] = {0,       1,           vd_size - 1,
                                       vd_size, 2 * vd_size, 4 * vd_size};

    // Check with different values for length and stride if applicable.
    for (int test = 0;
         test < (has_length || has_stride ? std::size(len_or_strides) : 1);
         test++) {
      // Store stride can't be smaller than vd_size
      if ((is_quad && len_or_strides[test] < vd_size / 4) ||
          (!is_load && has_stride && len_or_strides[test] < vd_size)) {
        continue;
      }
      // Set input register values.
      SetRegisterValues<uint32_t>(
          {{kelvin::sim::test::kRs1Name, kelvin::sim::test::kDataLoadAddress}});

      if (!x_variant) {
        SetRegisterValues<uint32_t>(
            {{kelvin::sim::test::kRs2Name, len_or_strides[test]}});
      }

      // Fill vector register(s) with random values.
      std::vector<T> vd_value(vector_length_in_bytes / sizeof(T) * num_ops);
      auto vd_span = absl::Span<T>(vd_value);
      FillArrayWithRandomValues<T>(vd_span);
      for (int i = 0; i < num_ops; i++) {
        auto vd_name = absl::StrCat("v", kelvin::sim::test::kVd + i);
        SetVectorRegisterValues<T>(
            {{vd_name, vd_span.subspan(vd_size * i, vd_size)}});
      }

      // Execute instruction.
      instruction->Execute();

      // Compute memory values. For load test it is the expected output; for
      // store test it is the actual output.
      std::vector<T> memory_values(vd_size * num_ops);
      uint32_t addr = kelvin::sim::test::kDataLoadAddress;
      uint32_t rs2_value = len_or_strides[test];
      uint32_t count = vd_size * num_ops;
      if (has_length) {
        count = std::min(count, rs2_value);
      }
      uint32_t left = count;
      for (int op_num = 0; op_num < num_ops; op_num++) {
        const int n = std::min(vd_size, left);
        if (is_quad) {
          const uint32_t quad_size = vd_size / 4;
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < quad_size; ++j) {
              memory_values[op_num * vd_size + i * quad_size + j] =
                  GetSavedMemoryValue<T>(addr +
                                         (i * quad_size + j) * sizeof(T));
            }
            // Stride increase per quad_size.
            addr += rs2_value * sizeof(T);
          }
        } else {
          for (int i = 0; i < vd_size; ++i) {
            if (is_load) {
              memory_values[op_num * vd_size + i] =
                  i < n ? GetDefaultMemoryValue<T>(addr + i * sizeof(T)) : 0;
            } else {
              memory_values[op_num * vd_size + i] =
                  i < n ? GetSavedMemoryValue<T>(addr + i * sizeof(T)) : 0;
            }
          }
          left -= n;
          if (has_stride) {
            addr += rs2_value * sizeof(T);
          } else {
            addr += n * sizeof(T);
          }
        }
      }

      uint32_t expected_rs1_value = kelvin::sim::test::kDataLoadAddress;
      if (post_increment && count) {
        if (has_length && has_stride) {  // .tp
          expected_rs1_value += vd_size * sizeof(T);
        } else if (!has_length && !has_stride && x_variant) {  // .p.x
          expected_rs1_value += vd_size * sizeof(T) * num_ops;
        } else if (has_length) {  // .lp
          expected_rs1_value += count * sizeof(T);
        } else if (has_stride) {  // .sp
          const uint32_t quad_scale = is_quad ? 4 : 1;
          expected_rs1_value += rs2_value * sizeof(T) * num_ops * quad_scale;
        } else {  // .p.xx
          expected_rs1_value += rs2_value * sizeof(T);
        }
      }

      // Check result
      left = count;
      for (int op_num = 0; op_num < num_ops; op_num++) {
        auto vreg_num = kelvin::sim::test::kVd + op_num;
        auto test_vreg = vreg_[vreg_num];
        auto vreg_span = test_vreg->data_buffer()->Get<T>();
        if (is_load) {
          for (int element_index = 0; element_index < vd_size;
               element_index++) {
            auto vreg_element_index = op_num * vd_size + element_index;
            EXPECT_EQ(memory_values[vreg_element_index],
                      vreg_span[element_index])
                << absl::StrCat(name, "[", vreg_element_index, "] != reg[",
                                vreg_num, "*", element_index, "]");
          }
        } else {  // Store
          const int n = std::min(vd_size, left);
          for (int element_index = 0;
               element_index < vd_size && element_index < n; element_index++) {
            auto total_element_index = op_num * vd_size + element_index;
            EXPECT_EQ(memory_values[total_element_index],
                      vreg_span[element_index])
                << absl::StrCat(name, " mem at ", total_element_index,
                                " != vreg[", vreg_num, "][", element_index,
                                "]");
          }
          left -= n;
        }
      }

      if (post_increment) {
        // Check rs1 value.
        auto *reg = state_
                        ->GetRegister<kelvin::sim::test::RV32Register>(
                            kelvin::sim::test::kRs1Name)
                        .first;
        EXPECT_EQ(expected_rs1_value, reg->data_buffer()->Get<uint32_t>()[0])
            << absl::StrCat(name, " post incremented rs1 is incorrect.");
      }
    }
  }

  template <typename T>
  void MemoryLoadStoreOpTestHelper(absl::string_view name, bool is_load) {
    constexpr bool kNoLength = false;
    constexpr bool kLength = true;
    constexpr bool kNoStride = false;
    constexpr bool kStride = true;
    constexpr bool kPostIncrement = true;
    constexpr bool kXVariant = true;
    constexpr bool kNotXVariant = false;
    constexpr bool kNotQuad = false;

    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<T>());

    for (auto strip_mine : {false, true}) {
      for (auto post_increment : {false, true}) {
        // .x variants.
        auto subname = absl::StrCat(name_with_type, post_increment ? "P" : "",
                                    "X", strip_mine ? "M" : "");
        MemoryLoadStoreOpTestHelper<T>(subname, kNoLength, kNoStride,
                                       strip_mine, post_increment, kXVariant,
                                       is_load, kNotQuad);
      }
      // .xx variants
      for (auto len_stride_post :
           {std::tuple(false, false, true), std::tuple(false, true, false),
            std::tuple(false, true, true), std::tuple(true, false, false),
            std::tuple(true, false, true)}) {
        auto has_length = std::get<0>(len_stride_post);
        auto has_stride = std::get<1>(len_stride_post);
        auto post_increment = std::get<2>(len_stride_post);
        auto subname = absl::StrCat(name_with_type,
                                    has_length   ? "L"
                                    : has_stride ? "S"
                                                 : "",
                                    post_increment ? "P" : "", "XX",
                                    strip_mine ? "M" : "");
        MemoryLoadStoreOpTestHelper<T>(subname, has_length, has_stride,
                                       strip_mine, post_increment, kNotXVariant,
                                       is_load, kNotQuad);
      }

      // .tp variants.
      auto subname =
          absl::StrCat(name_with_type, "TP", "XX", strip_mine ? "M" : "");
      MemoryLoadStoreOpTestHelper<T>(subname, kLength, kStride, strip_mine,
                                     kPostIncrement, kNotXVariant, is_load,
                                     kNotQuad);
    }
  }

  template <typename T>
  void StoreQuadOpTestHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<T>());
    constexpr bool kNotLength = false;
    constexpr bool kStride = true;
    constexpr bool kNotXVariant = false;
    constexpr bool kNotLoad = false;
    constexpr bool kIsQuad = true;
    for (auto strip_mine : {false, true}) {
      for (auto post_increment : {false, true}) {
        auto subname =
            absl::StrCat(name_with_type, "S", post_increment ? "P" : "", "XX",
                         strip_mine ? "M" : "");
        MemoryLoadStoreOpTestHelper<T>(subname, kNotLength, kStride, strip_mine,
                                       post_increment, kNotXVariant, kNotLoad,
                                       kIsQuad);
      }
    }
  }

  template <typename T1, typename TNext1, typename... TNext>
  void MemoryLoadStoreOpTestHelper(absl::string_view name, bool is_load) {
    MemoryLoadStoreOpTestHelper<T1>(name, is_load);
    MemoryLoadStoreOpTestHelper<TNext1, TNext...>(name, is_load);
  }

  template <typename T1, typename TNext1, typename... TNext>
  void StoreQuadOpTestHelper(absl::string_view name) {
    StoreQuadOpTestHelper<T1>(name);
    StoreQuadOpTestHelper<TNext1, TNext...>(name);
  }

 protected:
  template <typename T>
  T GetDefaultMemoryValue(int address) {
    T value = 0;
    uint8_t *ptr = reinterpret_cast<uint8_t *>(&value);
    for (int j = 0; j < sizeof(T); j++) {
      ptr[j] = (address + j) & 0xff;
    }
    return value;
  }

  template <typename T>
  T GetSavedMemoryValue(int address) {
    auto *db = state_->db_factory()->Allocate<T>(1);
    memory_->Load(address, db, nullptr, nullptr);
    T data = db->template Get<T>(0);
    db->DecRef();
    return data;
  }
};

TEST_F(KelvinVectorMemoryInstructionsTest, VLd) {
  MemoryLoadStoreOpTestHelper<int8_t, int16_t, int32_t>("VLd",
                                                        /*is_load=*/true);
}

TEST_F(KelvinVectorMemoryInstructionsTest, VSt) {
  MemoryLoadStoreOpTestHelper<int8_t, int16_t, int32_t>("VSt",
                                                        /*is_load=*/false);
}

TEST_F(KelvinVectorMemoryInstructionsTest, VStQ) {
  StoreQuadOpTestHelper<int8_t, int16_t, int32_t>("VStQ");
}

class KelvinGetVlInstructionTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 public:
  template <typename T>
  void GetVlTestHelper() {
    constexpr char kRdName[] = "x8";
    constexpr uint32_t kMaxVlenInBytes = kelvin::sim::kVectorLengthInBits / 8;
    auto instruction = CreateInstruction();
    AppendRegisterOperands(
        instruction.get(),
        {kelvin::sim::test::kRs1Name, kelvin::sim::test::kRs2Name}, {kRdName});
    for (auto strip_mine : {false, true}) {
      for (auto is_rs1 : {false, true}) {
        for (auto is_rs2 : {false, true}) {
          uint32_t rs1_value = RandomValue();
          uint32_t rs2_value = RandomValue();
          SetRegisterValues<uint32_t>({{kelvin::sim::test::kRs1Name, rs1_value},
                                       {kelvin::sim::test::kRs2Name, rs2_value},
                                       {kRdName, UINT32_MAX}});
          instruction->set_semantic_function(
              absl::bind_front(&KelvinGetVl<T>, strip_mine, is_rs1, is_rs2));
          uint32_t expected_vlen =
              kMaxVlenInBytes / sizeof(T) * (strip_mine ? 4 : 1);
          if (is_rs1) {
            expected_vlen = std::min(expected_vlen, rs1_value);
          }
          if (is_rs2) {
            expected_vlen = std::min(expected_vlen, rs2_value);
          }
          // Execute instruction.
          instruction->Execute(nullptr);
          EXPECT_EQ(xreg_[8]->data_buffer()->Get<uint32_t>(0), expected_vlen)
              << "Test failed with type "
              << (sizeof(T) == 4 ? "W" : (sizeof(T) == 2 ? "H" : "B"))
              << ", strip_mine: " << strip_mine << ", rs1_set: " << is_rs1
              << ", rs2_set: " << is_rs2;
        }
      }
    }
  }

  template <typename T1, typename TNext1, typename... TNext>
  void GetVlTestHelper() {
    GetVlTestHelper<T1>();
    GetVlTestHelper<TNext1, TNext...>();
  }

 protected:
  // Create a random value in the valid range for the type.
  uint32_t RandomValue() {
    return absl::Uniform(absl::IntervalClosed, bitgen_,
                         std::numeric_limits<uint32_t>::lowest(),
                         std::numeric_limits<uint32_t>::max());
  }
};

TEST_F(KelvinGetVlInstructionTest, GetVl) {
  GetVlTestHelper<int8_t, int16_t, int32_t>();
}

class KelvinAccumulateInstructionTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 public:
  void VcGetTestHelper() {
    constexpr int kVd = 48;
    const uint32_t kVLenInWord = state_->vector_length() / 32;
    // Set v48..55 with random values.
    std::vector<uint32_t> vd_value(kVLenInWord * kVLenInWord);
    auto vd_span = absl::Span<uint32_t>(vd_value);
    FillArrayWithRandomValues<uint32_t>(vd_span);
    for (int i = 0; i < kVLenInWord; ++i) {
      auto vd_name = absl::StrCat("v", kVd + i);
      SetVectorRegisterValues<uint32_t>(
          {{vd_name, vd_span.subspan(kVLenInWord * i, kVLenInWord)}});
    }
    auto instruction = CreateInstruction();
    AppendVectorRegisterOperands(instruction.get(), kVLenInWord,
                                 1 /* src1_widen_factor */, {}, {},
                                 false /* widen_dst */, {kVd});
    instruction->set_semantic_function(&KelvinVcGet);
    instruction->Execute();
    // Resulting v48..55 should all have 0 values
    for (int i = 0; i < kVLenInWord; ++i) {
      auto vreg_num = kVd + i;
      auto test_vreg = vreg_[vreg_num];
      auto vreg_span = test_vreg->data_buffer()->Get<uint32_t>();
      for (int element_index = 0; element_index < kVLenInWord;
           element_index++) {
        EXPECT_EQ(vreg_span[element_index], 0)
            << absl::StrCat("vreg[", vreg_num, "][", element_index, "] != 0");
      }
    }
  }
  void AcSetTestHelper(bool is_transpose) {
    constexpr int kVd = 48;
    constexpr int kVs = 16;
    const uint32_t kVLenInWord = state_->vector_length() / 32;
    // Set v24..31, 48..55 with random values.
    std::vector<uint32_t> vd_value(kVLenInWord * kVLenInWord);
    auto vd_span = absl::Span<uint32_t>(vd_value);
    FillArrayWithRandomValues<uint32_t>(vd_span);
    for (int i = 0; i < kVLenInWord; ++i) {
      auto vd_name = absl::StrCat("v", kVd + i);
      auto vs_name = absl::StrCat("v", kVs + i);
      SetVectorRegisterValues<uint32_t>(
          {{vd_name, vd_span.subspan(kVLenInWord * i, kVLenInWord)}});
      SetVectorRegisterValues<uint32_t>(
          {{vs_name, vd_span.subspan(kVLenInWord * i, kVLenInWord)}});
    }
    auto instruction = CreateInstruction();
    AppendVectorRegisterOperands(instruction.get(), kVLenInWord,
                                 1 /* src1_widen_factor */, kVs, {},
                                 false /* widen_dst */, {kVd});
    instruction->set_semantic_function(
        absl::bind_front(&KelvinAcSet, is_transpose));
    instruction->Execute();
    // Resulting acc_register_ should match `vs` content
    for (int i = 0; i < kVLenInWord; ++i) {
      auto vreg_num = kVs + i;
      auto test_vreg = vreg_[vreg_num];
      auto vreg_span = test_vreg->data_buffer()->Get<uint32_t>();
      for (int element_index = 0; element_index < kVLenInWord;
           element_index++) {
        if (is_transpose) {
          auto *acc_vec = state_->acc_vec(element_index);
          EXPECT_EQ(vreg_span[element_index], acc_vec->at(i))
              << absl::StrCat("vreg[", vreg_num, "][", element_index,
                              "] != acc[", element_index, "][", i, "]");
        } else {
          auto *acc_vec = state_->acc_vec(i);
          EXPECT_EQ(vreg_span[element_index], acc_vec->at(element_index))
              << absl::StrCat("vreg[", vreg_num, "][", element_index,
                              "] != acc[", i, "][", element_index, "]");
        }
      }
    }
  }
};

TEST_F(KelvinAccumulateInstructionTest, VcGet) { VcGetTestHelper(); }

TEST_F(KelvinAccumulateInstructionTest, AcSet) {
  AcSetTestHelper(/*is_transpose=*/false);
}

TEST_F(KelvinAccumulateInstructionTest, AcTr) {
  AcSetTestHelper(/*is_transpose=*/true);
}

TEST_F(KelvinAccumulateInstructionTest, ADwInit) {
  constexpr int kVd = 16;
  constexpr int kVs = 32;
  const uint32_t kVLenInByte = state_->vector_length() / 8;
  constexpr int kInitLength = 4;
  // Set vs and vd with random values.
  std::vector<uint8_t> vs_value(kVLenInByte * kInitLength);
  auto vs_span = absl::Span<uint8_t>(vs_value);
  FillArrayWithRandomValues<uint8_t>(vs_span);
  std::vector<uint8_t> vd_value(kVLenInByte * kInitLength);
  auto vd_span = absl::Span<uint8_t>(vd_value);
  FillArrayWithRandomValues<uint8_t>(vd_span);
  for (int i = 0; i < kInitLength; ++i) {
    auto vd_name = absl::StrCat("v", kVd + i);
    auto vs_name = absl::StrCat("v", kVs + i);
    SetVectorRegisterValues<uint8_t>(
        {{vs_name, vs_span.subspan(kVLenInByte * i, kVLenInByte)},
         {vd_name, vd_span.subspan(kVLenInByte * i, kVLenInByte)}});
  }
  auto instruction = CreateInstruction();
  AppendVectorRegisterOperands(instruction.get(), kVLenInByte,
                               1 /* src1_widen_factor */, kVs, {},
                               false /* widen_dst */, {kVd});
  instruction->set_semantic_function(&KelvinADwInit);
  instruction->Execute();
  // Resulting `vd` should match `vs` in the first quarter of each vector
  for (int i = 0; i < kInitLength; ++i) {
    auto vreg_num = kVd + i;
    auto test_vreg = vreg_[vreg_num];
    auto vreg_span = test_vreg->data_buffer()->Get<uint8_t>();
    auto vref_num = kVs + i;
    auto ref_vreg = vreg_[vref_num];
    auto ref_span = ref_vreg->data_buffer()->Get<uint8_t>();
    for (int element_index = 0; element_index < ref_span.size() / 4;
         element_index++) {
      EXPECT_EQ(vreg_span[element_index], ref_span[element_index])
          << absl::StrCat("vreg[", vreg_num, "][", element_index, "] != ref[",
                          vref_num, "][", element_index, "]");
    }
  }
}

}  // namespace
