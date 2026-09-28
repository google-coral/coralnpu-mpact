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

// Execution tests for the M4 matrix configuration instructions: decode a real
// instruction word, execute it, and check the resulting mtype CSR / vl state.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_m4_user_decoder.h"
#include "sim/coralnpu_v2_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_state.h"
#include "riscv/riscv_zvt_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace {

using ::coralnpu::sim::CoralNPUM4State;
using ::coralnpu::sim::CoralNPUM4UserDecoder;
using ::coralnpu::sim::MemoryPermission;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::kRiscVMtypeCsrIndex;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RiscVZvtMatrixState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::util::FlatDemandMemory;

constexpr uint32_t kOpcodeOpV = 0b101'0111;
constexpr uint32_t kInstAddress = 0x1000;

uint32_t MakeVConfig(uint32_t func7, uint32_t rs2, uint32_t rs1, uint32_t func3,
                     uint32_t rd) {
  return (func7 << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) |
         kOpcodeOpV;
}

class CoralNPUM4ConfigTest : public ::testing::Test {
 protected:
  CoralNPUM4ConfigTest()
      : memory_(std::make_unique<FlatDemandMemory>()),
        state_(::coralnpu::sim::CreateCoralNPUM4State("test", RiscVXlen::RV32,
                                                      memory_.get())) {
    for (int i = 0; i < 32; i++) {
      state_->AddRegister<RV32Register>(absl::StrCat("x", i));
    }
    rvv_ = std::make_unique<RiscVVectorState>(
        state_.get(), ::coralnpu::sim::kCoralNPUV2VectorByteLength);
    auto matrix_status = RiscVZvtMatrixState::Create(state_.get());
    CHECK_OK(matrix_status);
    matrix_ = std::move(*matrix_status);
    decoder_ =
        std::make_unique<CoralNPUM4UserDecoder>(state_.get(), memory_.get());
    // Allow execute permission at the instruction address.
    state_->AddMemoryRegion(kInstAddress, 0x100,
                            MemoryPermission::kReadWriteExecute);
  }

  void SetXReg(int index, uint32_t value) {
    auto [reg, unused] =
        state_->GetRegister<RV32Register>(absl::StrCat("x", index));
    reg->data_buffer()->Set<uint32_t>(0, value);
  }

  uint32_t GetXReg(int index) {
    auto [reg, unused] =
        state_->GetRegister<RV32Register>(absl::StrCat("x", index));
    return reg->data_buffer()->Get<uint32_t>(0);
  }

  uint32_t Mtype() {
    auto status_or_csr = state_->csr_set()->GetCsr(kRiscVMtypeCsrIndex);
    CHECK_OK(status_or_csr);
    return (*status_or_csr)->AsUint32();
  }

  // Writes `inst_word` to memory and decodes + executes it.
  void Execute(uint32_t inst_word) {
    auto db = state_->db_factory()->Allocate<uint32_t>(1);
    db->Set<uint32_t>(0, inst_word);
    memory_->Store(kInstAddress, db);
    db->DecRef();
    Instruction* inst = decoder_->DecodeInstruction(kInstAddress);
    inst->Execute(nullptr);
    inst->DecRef();
    // Commit register writes scheduled on the data-buffer delay line (the
    // RiscVTop does this each step in the full simulator).
    state_->AdvanceDelayLines();
  }

  std::unique_ptr<FlatDemandMemory> memory_;
  std::unique_ptr<CoralNPUM4State> state_;
  std::unique_ptr<RiscVVectorState> rvv_;
  std::unique_ptr<RiscVZvtMatrixState> matrix_;
  std::unique_ptr<CoralNPUM4UserDecoder> decoder_;
};

// msetmtype with an int8 configuration (mtwiden=3 => TWIDEN=4,
// TILE_ELEMENT_WIDTH=32).
TEST_F(CoralNPUM4ConfigTest, MsetMtypeInt8ConfigClampsTkTm) {
  // mtype: mtwiden=3 (bits[1:0]), tk=4 (bits[7:5]), tm=8 (bits[23:10]).
  const uint32_t kMtype = 0x3 | (4u << 5) | (8u << 10);
  SetXReg(1, kMtype);  // rs1 = mtype
  SetXReg(2, 0x0);     // rs2 = vtype: SEW=8, LMUL=1.

  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  // vl must be 0 after msetmtype.
  EXPECT_EQ(rvv_->vector_length(), 0);
  // mtwiden, tk (clamped to KMAX=4) and tm (clamped to min(8,16,32)=8)
  // preserved.
  EXPECT_EQ(matrix_->mtwiden(), 3);
  EXPECT_EQ(matrix_->tile_k(), 4);
  EXPECT_EQ(matrix_->tile_m(), 8);
  // The mtype CSR reads back the live value.
  EXPECT_EQ(Mtype(), matrix_->mtype());
}

// When mtwiden == 0 the matrix unit is unconfigured and mtype becomes 0.
TEST_F(CoralNPUM4ConfigTest, MsetMtypeUnconfiguredZeroesMtype) {
  SetXReg(1, 0x0);  // mtwiden = 0.
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));
  EXPECT_EQ(matrix_->mtype(), 0u);
  EXPECT_FALSE(matrix_->configured());
}

// msettn sets tn (== vl) bounded by the matrix config and writes rd.
TEST_F(CoralNPUM4ConfigTest, MsetTnSetsVlAndWritesRd) {
  // Configure int8 first.
  const uint32_t kMtype = 0x3 | (4u << 5) | (8u << 10);
  SetXReg(1, kMtype);
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  // msettn x3, x4 with x4 = 10. Bound = min(10, VLMAX=16,
  // EFFECTIVE_TILE_EDGE=32) = 10.
  SetXReg(4, 10);
  Execute(MakeVConfig(0b100'0010, /*rs2=*/0, /*rs1=*/4, 0b111, /*rd=*/3));
  EXPECT_EQ(rvv_->vector_length(), 10u);
  EXPECT_EQ(GetXReg(3), 10u);
}

// msettk clamps to KMAX (4 for SEW=8) and writes rd.
TEST_F(CoralNPUM4ConfigTest, MsetTkClampsToKmax) {
  const uint32_t kMtype = 0x3 | (1u << 5) | (8u << 10);  // tk starts at 1.
  SetXReg(1, kMtype);
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  // msettk x3, x4 with x4 = 7 -> clamped to KMAX = 4.
  SetXReg(4, 7);
  Execute(MakeVConfig(0b100'0010, /*rs2=*/2, /*rs1=*/4, 0b111, /*rd=*/3));
  EXPECT_EQ(matrix_->tile_k(), 4);
  EXPECT_EQ(GetXReg(3), 4u);
}

TEST_F(CoralNPUM4ConfigTest, MsetTnWhenUnconfigured) {
  SetXReg(1, 0x0);  // mtwiden = 0
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  SetXReg(4, 10);
  Execute(MakeVConfig(0b100'0010, /*rs2=*/0, /*rs1=*/4, 0b111,
                      /*rd=*/3));  // msettn
  EXPECT_EQ(rvv_->vector_length(), 10u);
  EXPECT_EQ(GetXReg(3), 10u);
}

TEST_F(CoralNPUM4ConfigTest, MsetTmWhenUnconfigured) {
  SetXReg(1, 0x0);  // mtwiden = 0
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  SetXReg(4, 10);
  Execute(MakeVConfig(0b100'0010, /*rs2=*/1, /*rs1=*/4, 0b111,
                      /*rd=*/3));  // msettm
  EXPECT_EQ(matrix_->tile_m(), 0u);
  EXPECT_EQ(GetXReg(3), 0u);
}

TEST_F(CoralNPUM4ConfigTest, MsetTkWhenUnconfigured) {
  SetXReg(1, 0x0);  // mtwiden = 0
  SetXReg(2, 0x0);
  Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));

  SetXReg(4, 10);
  Execute(MakeVConfig(0b100'0010, /*rs2=*/2, /*rs1=*/4, 0b111,
                      /*rd=*/3));  // msettk
  EXPECT_EQ(matrix_->tile_k(), 0u);
  EXPECT_EQ(GetXReg(3), 0u);
}

}  // namespace
