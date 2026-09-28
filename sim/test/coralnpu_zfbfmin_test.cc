// Copyright 2026 Google LLC
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

#include <cinttypes>
#include <cstdint>
#include <memory>
#include <vector>

#include "sim/coralnpu_architecture.h"
#include "sim/coralnpu_simulator.h"
#include "sim/coralnpu_v2_state.h"
#include "sim/test/coralnpu_zfbfmin_generated.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/util/memory/memory_interface.h"

/* start - new test macros */
#define ABSL_EXPECT_OK(expression) \
  EXPECT_THAT(expression, ::absl_testing::IsOk())
#define ABSL_ASSERT_OK(expression) \
  ASSERT_THAT(expression, ::absl_testing::IsOk())
/* end - new test macros */

namespace {

using ::coralnpu::sim::Architecture;
using ::coralnpu::sim::CoralNPUSimulator;
using ::coralnpu::sim::CoralNPUSimulatorOptions;
using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::MemoryPermission;
using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::DecoderInterface;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::util::MemoryInterface;

constexpr uint32_t kMemoryStart = 0x0;
constexpr uint32_t kMemorySize = 0x1000;
constexpr uint32_t kRoundingModeShift = 5;
constexpr uint64_t kBoxedBf16Mask = ~uint64_t{0} << 16;

class CoralNPUZfbfminTest : public ::testing::Test {
 public:
  void SetUp() override {
    CoralNPUSimulatorOptions options;
    options.architecture = Architecture::kM3;
    options.memory_regions.push_back(
        {kMemoryStart, kMemorySize, MemoryPermission::kReadExecute});
    simulator_ = std::make_unique<CoralNPUSimulator>(options);

    state_ = simulator_->state();
    memory_ = simulator_->memory();
    decoder_ = simulator_->decoder();
    rv_fp_state_ = state_->rv_fp();
  }

 protected:
  std::unique_ptr<CoralNPUSimulator> simulator_;
  // Non-owning view pointers to objects managed by simulator_.
  CoralNPUV2State* state_;
  MemoryInterface* memory_;
  DecoderInterface* decoder_;
  RiscVFPState* rv_fp_state_;
};

TEST_F(CoralNPUZfbfminTest, ExecuteFcvtBf16S) {
  // fcvt.bf16.s f2, f1, RNE (rm=0)
  const uint32_t instruction_word =
      ::coralnpu::sim::test_data::coralnpu_zfbfmin::GetInstructions()[0]
          .instruction;
  constexpr uint64_t kTestAddress = 0x0;

  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(0, instruction_word);
  memory_->Store(kTestAddress, inst_db);

  struct TestCase {
    uint64_t f64_in;
    uint16_t bf16_expected;
    uint32_t fflags_expected;
    uint32_t rm;
  } cases[] = {
      {0xFFFFFFFF00000000ULL | 0x3F800000ULL, 0x3F80, 0x0,
       0},  // 1.0 -> 1.0 (RNE)
      {0xFFFFFFFF00000000ULL | 0x402df854ULL, 0x402e, 0x1,
       0},  // e -> 2.71875 (RNE), NX
      {0xFFFFFFFF00000000ULL | 0x7F800000ULL, 0x7F80, 0x0, 0},  // +Inf -> +Inf
      {0xFFFFFFFF00000000ULL | 0xFF800000ULL, 0xFF80, 0x0, 0},  // -Inf -> -Inf
      {0xFFFFFFFF00000000ULL | 0x7FC00000ULL, 0x7FC0, 0x0, 0},  // qNaN -> qNaN
      {0xFFFFFFFF00000000ULL | 0x7F800001ULL, 0x7FC0, 0x10,
       0},  // sNaN -> qNaN, NV
      {0xFFFFFFFF00000000ULL | 0x00000001ULL, 0x0000, 0x03,
       0},  // tiny subnormal -> 0.0, UF | NX
      {0xFFFFFFFF00000000ULL | 0x7F7FFFFFULL, 0x7F80, 0x05,
       0},  // MaxFloat -> +Inf (Overflow 0x04 | Inexact 0x01 = 0x05)
      {0xFFFFFFFF00000000ULL | 0x402C8000ULL, 0x402C, 0x01,
       0},  // guard=1, lsb=0, sticky=0 -> tie to even (0x402C), NX
      {0xFFFFFFFF00000000ULL | 0x402D8000ULL, 0x402E, 0x01,
       0},  // guard=1, lsb=1, sticky=0 -> round up (0x402E), NX
      // Additional rounding modes
      {0xFFFFFFFF00000000ULL | 0x402df854ULL, 0x402d, 0x1,
       1},  // e -> 2.7109... (RTZ), NX
      {0xFFFFFFFF00000000ULL | 0x402df854ULL, 0x402d, 0x1,
       2},  // e -> 2.7109... (RDN), NX
      {0xFFFFFFFF00000000ULL | 0x402df854ULL, 0x402e, 0x1,
       3},  // e -> 2.71875 (RUP), NX
      {0xFFFFFFFF00000000ULL | 0x402df854ULL, 0x402e, 0x1,
       4},  // e -> 2.71875 (RMM), NX
      {0xFFFFFFFF00000000ULL | 0x3F800000ULL, 0x3F80, 0x0,
       3},  // exact 1.0 under RUP -> 1.0 (no increment when guard=0, sticky=0)
      {0xFFFFFFFF00000000ULL | 0xC02df854ULL, 0xC02d, 0x1,
       3},  // -e under RUP -> -2.7109... (truncate away from zero), NX
      {0xFFFFFFFF00000000ULL | 0xC02df854ULL, 0xC02e, 0x1,
       2},  // -e under RDN -> -2.71875 (round down towards -Inf), NX
      {0x000000003F800000ULL, 0x7FC0, 0x0, 0},  // invalid NaN box -> qNaN
  };

  for (const auto& c : cases) {
    ABSL_ASSERT_OK(simulator_->WriteRegister("pc", kTestAddress));

    auto [f1_reg, f1_unused] =
        state_->GetRegister<::mpact::sim::riscv::RVFpRegister>("f1");
    auto* f1_db = f1_reg->data_buffer();
    auto [f2_reg, f2_unused] =
        state_->GetRegister<::mpact::sim::riscv::RVFpRegister>("f2");
    auto* f2_db = f2_reg->data_buffer();
    rv_fp_state_->fcsr()->Write(
        static_cast<uint32_t>(c.rm << kRoundingModeShift));

    f1_db->Set<uint64_t>(0, c.f64_in);
    f2_db->Set<uint64_t>(0, 0);

    ABSL_EXPECT_OK(simulator_->Step(1));

    auto [f2_reg_post, f2_unused_post] =
        state_->GetRegister<::mpact::sim::riscv::RVFpRegister>("f2");
    uint64_t f2_val = f2_reg_post->data_buffer()->Get<uint64_t>(0);
    uint32_t fflags_val = rv_fp_state_->fflags()->GetUint32();

    EXPECT_EQ(f2_val, c.bf16_expected | kBoxedBf16Mask)
        << absl::StrFormat("f64: 0x%" PRIx64 ", rm: %d", c.f64_in, c.rm);
    EXPECT_EQ(fflags_val, c.fflags_expected)
        << absl::StrFormat("f64: 0x%" PRIx64 ", rm: %d", c.f64_in, c.rm);
    uint32_t fcsr_val = rv_fp_state_->fcsr()->AsUint32();
    EXPECT_EQ(fcsr_val, (c.rm << kRoundingModeShift) | c.fflags_expected)
        << "fcsr rm cleared or stale";
  }
  inst_db->DecRef();
}

TEST_F(CoralNPUZfbfminTest, TrapWhenFsDisabled) {
  constexpr uint64_t kTestAddress = 0x0;
  // fcvt.bf16.s f2, f1, RNE
  const uint32_t instruction_word =
      ::coralnpu::sim::test_data::coralnpu_zfbfmin::GetInstructions()[1]
          .instruction;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(0, instruction_word);
  memory_->Store(kTestAddress, inst_db);

  state_->mstatus()->ClearBits(0x6000);
  ABSL_ASSERT_OK(simulator_->WriteRegister("pc", kTestAddress));

  bool trap_called = false;
  state_->set_on_trap(
      [&trap_called](bool, uint64_t, uint64_t code, uint64_t,
                     const ::mpact::sim::generic::Instruction*) -> bool {
        trap_called = true;
        EXPECT_EQ(code, 2);  // IllegalInstruction
        return true;
      });

  ABSL_EXPECT_OK(simulator_->Step(1));
  EXPECT_TRUE(trap_called);
  inst_db->DecRef();
}

TEST_F(CoralNPUZfbfminTest, TrapWhenInvalidDynamicRoundingMode) {
  constexpr uint64_t kTestAddress = 0x0;
  // fcvt.bf16.s f2, f1, dyn (rm=7)
  const uint32_t instruction_word =
      ::coralnpu::sim::test_data::coralnpu_zfbfmin::GetInstructions()[0]
          .instruction;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(0, instruction_word);
  memory_->Store(kTestAddress, inst_db);

  // Set invalid rounding mode (frm = 5) in FCSR.
  rv_fp_state_->fcsr()->Write(static_cast<uint32_t>(5 << kRoundingModeShift));
  ABSL_ASSERT_OK(simulator_->WriteRegister("pc", kTestAddress));

  bool trap_called = false;
  state_->set_on_trap(
      [&trap_called](bool, uint64_t, uint64_t code, uint64_t,
                     const ::mpact::sim::generic::Instruction*) -> bool {
        trap_called = true;
        EXPECT_EQ(code, 2);  // IllegalInstruction
        return true;
      });

  ABSL_EXPECT_OK(simulator_->Step(1));
  EXPECT_TRUE(trap_called);
  inst_db->DecRef();
}

TEST_F(CoralNPUZfbfminTest, TrapWhenInvalidStaticRoundingMode) {
  constexpr uint64_t kTestAddress = 0x0;
  // fcvt.bf16.s f2, f1, 5 (rm=5 is invalid)
  const uint32_t instruction_word =
      ::coralnpu::sim::test_data::coralnpu_zfbfmin::GetInstructions()[2]
          .instruction;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(0, instruction_word);
  memory_->Store(kTestAddress, inst_db);

  ABSL_ASSERT_OK(simulator_->WriteRegister("pc", kTestAddress));

  bool trap_called = false;
  state_->set_on_trap(
      [&trap_called](bool, uint64_t, uint64_t code, uint64_t,
                     const ::mpact::sim::generic::Instruction*) -> bool {
        trap_called = true;
        EXPECT_EQ(code, 2);  // IllegalInstruction
        return true;
      });

  ABSL_EXPECT_OK(simulator_->Step(1));
  EXPECT_TRUE(trap_called);
  inst_db->DecRef();
}

TEST_F(CoralNPUZfbfminTest, StaticInstructionRmOverridesFcsr) {
  constexpr uint64_t kTestAddress = 0x0;
  // fcvt.bf16.s f2, f1, rup (static rm=3 encoded in instruction word)
  const uint32_t instruction_word =
      ::coralnpu::sim::test_data::coralnpu_zfbfmin::GetInstructions()[3]
          .instruction;
  DataBuffer* inst_db = state_->db_factory()->Allocate<uint32_t>(1);
  inst_db->Set<uint32_t>(0, instruction_word);
  memory_->Store(kTestAddress, inst_db);

  // Set FCSR frm to RTZ (1).
  rv_fp_state_->fcsr()->Write(static_cast<uint32_t>(1 << kRoundingModeShift));
  ABSL_ASSERT_OK(simulator_->WriteRegister("pc", kTestAddress));

  auto [f1_reg, f1_unused] =
      state_->GetRegister<::mpact::sim::riscv::RVFpRegister>("f1");
  f1_reg->data_buffer()->Set<uint64_t>(
      0, 0xFFFFFFFF00000000ULL |
             0x402df854ULL);  // 32-bit float bit representation of e
                              // (2.7182818...), NaN-boxed.

  ABSL_EXPECT_OK(simulator_->Step(1));

  auto [f2_reg_post, f2_unused_post] =
      state_->GetRegister<::mpact::sim::riscv::RVFpRegister>("f2");
  uint64_t f2_val = f2_reg_post->data_buffer()->Get<uint64_t>(0);

  // Must output 0x402e (RUP), NOT 0x402d (RTZ from FCSR).
  EXPECT_EQ(f2_val, 0x402e | kBoxedBf16Mask);
  inst_db->DecRef();
}

}  // namespace
