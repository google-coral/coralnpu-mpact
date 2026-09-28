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

// Execution tests for the M4 matrix-multiply and tile move/load/store
// instructions: decode real instruction words, execute, and check tile state.

#include <cstdint>
#include <memory>
#include <utility>

#include "sim/coralnpu_m4_state.h"
#include "sim/coralnpu_m4_user_decoder.h"
#include "sim/coralnpu_v2_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
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
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RiscVZvtMatrixState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::FlatDemandMemory;

constexpr uint32_t kInstAddress = 0x1000;
constexpr uint32_t kDataAddress = 0x10000;

// Builds an OP-V configuration word (vset-family layout).
uint32_t MakeVConfig(uint32_t func7, uint32_t rs2, uint32_t rs1, uint32_t func3,
                     uint32_t rd) {
  return (func7 << 25) | (rs2 << 20) | (rs1 << 15) | (func3 << 12) | (rd << 7) |
         0b101'0111u;
}

// Builds an OP-VE matrix-multiply word.
uint32_t MakeVMatrix(uint32_t func6, uint32_t vs2, uint32_t vs1, uint32_t func3,
                     uint32_t mtd) {
  return (func6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15) |
         (func3 << 12) | (mtd << 7) | 0b111'0111u;
}

// Builds an OP-V VArith-layout word (used by the tile move instructions).
uint32_t MakeVArith(uint32_t func6, uint32_t vs2, uint32_t vs1, uint32_t func3,
                    uint32_t vd) {
  return (func6 << 26) | (1u << 25) | (vs2 << 20) | (vs1 << 15) |
         (func3 << 12) | (vd << 7) | 0b101'0111u;
}

// Builds a tile load/store (VMem) word: mew=1, mop=0, vm=1, width=0b111.
uint32_t MakeVtMem(uint32_t nf, uint32_t rs2, uint32_t rs1, uint32_t opcode) {
  return (nf << 29) | (1u << 28) | (1u << 25) | (rs2 << 20) | (rs1 << 15) |
         (0b111u << 12) | opcode;
}

// TSS (tile subset specifier): tile[30:27], pattern[26:24], index[23:0].
uint32_t MakeTss(int tile, int pattern, uint32_t index) {
  return (static_cast<uint32_t>(tile) << 27) |
         (static_cast<uint32_t>(pattern) << 24) | (index & 0xFFFFFF);
}

float Bf16Bits(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  return static_cast<float>(u >> 16);  // high 16 bits as the bf16 pattern.
}

class CoralNPUM4OpsTest : public ::testing::Test {
 protected:
  CoralNPUM4OpsTest()
      : memory_(std::make_unique<FlatDemandMemory>()),
        state_(coralnpu::sim::CreateCoralNPUM4State("test", RiscVXlen::RV32,
                                                    memory_.get())) {
    for (int i = 0; i < 32; i++) {
      state_->AddRegister<RV32Register>(absl::StrCat("x", i));
      state_->AddRegister<RVVectorRegister>(
          absl::StrCat("v", i), coralnpu::sim::kCoralNPUV2VectorByteLength);
    }
    rvv_ = std::make_unique<RiscVVectorState>(
        state_.get(), coralnpu::sim::kCoralNPUV2VectorByteLength);
    auto matrix_status = RiscVZvtMatrixState::Create(state_.get());
    CHECK_OK(matrix_status);
    matrix_ = std::move(*matrix_status);
    decoder_ =
        std::make_unique<CoralNPUM4UserDecoder>(state_.get(), memory_.get());
    state_->AddMemoryRegion(kInstAddress, 0x100,
                            MemoryPermission::kReadWriteExecute);
    state_->AddMemoryRegion(kDataAddress, 0x1000, MemoryPermission::kReadWrite);
    state_->set_on_trap(
        [this](bool, uint64_t, uint64_t, uint64_t,
               const ::mpact::sim::generic::Instruction*) -> bool {
          was_trap_handler_called_ = true;
          return true;
        });
  }

  void SetXReg(int index, uint32_t value) {
    auto [reg, unused] =
        state_->GetRegister<RV32Register>(absl::StrCat("x", index));
    reg->data_buffer()->Set<uint32_t>(0, value);
  }

  template <typename T>
  void SetVreg(int v, absl::Span<const T> values) {
    auto [reg, unused] =
        state_->GetRegister<RVVectorRegister>(absl::StrCat("v", v));
    auto* db = reg->data_buffer();
    for (int i = 0; i < static_cast<int>(values.size()); i++) {
      db->Set<T>(i, values[i]);
    }
  }

  void Execute(uint32_t inst_word) {
    auto db = state_->db_factory()->Allocate<uint32_t>(1);
    db->Set<uint32_t>(0, inst_word);
    memory_->Store(kInstAddress, db);
    db->DecRef();
    Instruction* inst = decoder_->DecodeInstruction(kInstAddress);
    inst->Execute(nullptr);
    inst->DecRef();
    state_->AdvanceDelayLines();
  }

  // Configures the matrix unit with the given mtype, vtype and tn (== vl).
  void Configure(uint32_t mtype, uint32_t vtype, int tn) {
    SetXReg(1, mtype);
    SetXReg(2, vtype);
    Execute(MakeVConfig(0b100'0001, /*rs2=*/2, /*rs1=*/1, 0b111, /*rd=*/0));
    rvv_->set_vector_length(tn);
  }

  std::unique_ptr<FlatDemandMemory> memory_;
  std::unique_ptr<CoralNPUM4State> state_;
  std::unique_ptr<RiscVVectorState> rvv_;
  std::unique_ptr<RiscVZvtMatrixState> matrix_;
  std::unique_ptr<CoralNPUM4UserDecoder> decoder_;
  bool was_trap_handler_called_ = false;
};

// int8 x int8 -> int32, C = A^T * B for 2x2 matrices, K=2.
TEST_F(CoralNPUM4OpsTest, Vtmms_Int8Matmul) {
  // mtype: mtwiden=3 (TWIDEN=4), tk=2, tm=2. vtype: SEW=8, LMUL=1.
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  ASSERT_EQ(matrix_->tile_k(), 2);
  ASSERT_EQ(matrix_->tile_m(), 2);

  // A rows in v8 (k=0) and v10 (k=1) -- stride 8/KMAX = 2 registers.
  SetVreg<int8_t>(8, {1, 2});
  SetVreg<int8_t>(10, {3, 4});
  // B rows in v16 (k=0) and v18 (k=1).
  SetVreg<int8_t>(16, {5, 6});
  SetVreg<int8_t>(18, {7, 8});

  // vtmms.tvv mt0, v8, v16 : func6=111100, func3=000, alt=1 (mtd=1).
  Execute(MakeVMatrix(0b111'100, /*vs2=*/8, /*vs1=*/16, 0b000, /*mtd=*/1));

  // A^T*B = [[1,3],[2,4]] * [[5,6],[7,8]] = [[26,30],[38,44]].
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 0, 0, 32)), 26);
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 0, 1, 32)), 30);
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, 0, 32)), 38);
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, 1, 32)), 44);
}

// fp32 x fp32 -> fp32, K=1 (outer product).
TEST_F(CoralNPUM4OpsTest, Vtfmm_Fp32Matmul) {
  // mtype: mtwiden=1 (TWIDEN=1), tk=1, tm=2. vtype: SEW=32 (vsew=010).
  Configure(/*mtype=*/0x1 | (1u << 5) | (2u << 10), /*vtype=*/(0b010u << 3),
            /*tn=*/2);
  ASSERT_EQ(matrix_->tile_k(), 1);

  SetVreg<float>(8, {1.0f, 2.0f});   // A row 0.
  SetVreg<float>(16, {3.0f, 4.0f});  // B row 0.

  // vtfmm.tvv mt0, v8, v16 : func6=111100, func3=001, alt=0 (mtd=0).
  Execute(MakeVMatrix(0b111'100, /*vs2=*/8, /*vs1=*/16, 0b001, /*mtd=*/0));

  // Outer product [1,2]^T * [3,4] = [[3,4],[6,8]].
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 0, 0, 32)), 3.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 0, 1, 32)), 4.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 1, 0, 32)), 6.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 1, 1, 32)), 8.0f);
}

// bf16 x bf16 -> fp32, K=2.
TEST_F(CoralNPUM4OpsTest, VtfmmAlt_Bf16Matmul) {
  // mtype: mtwiden=2 (TWIDEN=2), tk=2, tm=2. vtype: SEW=16 (vsew=001).
  Configure(/*mtype=*/0x2 | (2u << 5) | (2u << 10), /*vtype=*/(0b001u << 3),
            /*tn=*/2);
  ASSERT_EQ(matrix_->tile_k(), 2);

  // bf16 values stored as the high 16 bits of the fp32 pattern. Rows at
  // v8 (k=0), v12 (k=1) -- stride 8/KMAX = 4 registers.
  SetVreg<uint16_t>(8, {static_cast<uint16_t>(Bf16Bits(1.0f)),
                        static_cast<uint16_t>(Bf16Bits(2.0f))});
  SetVreg<uint16_t>(12, {static_cast<uint16_t>(Bf16Bits(3.0f)),
                         static_cast<uint16_t>(Bf16Bits(4.0f))});
  SetVreg<uint16_t>(16, {static_cast<uint16_t>(Bf16Bits(5.0f)),
                         static_cast<uint16_t>(Bf16Bits(6.0f))});
  SetVreg<uint16_t>(20, {static_cast<uint16_t>(Bf16Bits(7.0f)),
                         static_cast<uint16_t>(Bf16Bits(8.0f))});

  // vtfmm.alt.tvv mt0, v8, v16 : func6=111100, func3=001, alt=1 (mtd=1).
  Execute(MakeVMatrix(0b111'100, /*vs2=*/8, /*vs1=*/16, 0b001, /*mtd=*/1));

  // Same operands as the int8 case -> [[26,30],[38,44]].
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 0, 0, 32)), 26.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 0, 1, 32)), 30.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 1, 0, 32)), 38.0f);
  EXPECT_FLOAT_EQ((matrix_->GetElem<float>(0, 1, 1, 32)), 44.0f);
}

// vtmms accumulates into the existing tile (C += A^T*B): running it twice
// doubles the result.
TEST_F(CoralNPUM4OpsTest, VtmmsAccumulates) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  SetVreg<int8_t>(8, {1, 2});
  SetVreg<int8_t>(10, {3, 4});
  SetVreg<int8_t>(16, {5, 6});
  SetVreg<int8_t>(18, {7, 8});
  uint32_t word = MakeVMatrix(0b111'100, 8, 16, 0b000, 1);
  Execute(word);
  Execute(word);
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 0, 0, 32)), 52);  // 2 * 26.
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, 1, 32)), 88);  // 2 * 44.
}

// vtzero clears the tm x tn submatrix.
TEST_F(CoralNPUM4OpsTest, VtzeroClearsSubmatrix) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++) matrix_->SetElem<int32_t>(0, r, c, 32, 99);
  // vtzero mt0: func6=010000, vs2=11110, vs1=0, func3=110, rd=tttt0 -> tile 0.
  Execute((0b010'000u << 26) | (1u << 25) | (0b11110u << 20) | (0b110u << 12) |
          0b101'0111u);
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 2; c++)
      EXPECT_EQ((matrix_->GetElem<int32_t>(0, r, c, 32)), 0);
}

// vtmv.t.v then vtmv.v.t round-trips a vector through a tile row
// (TILE_ELEMENT_WIDTH = SEW).
TEST_F(CoralNPUM4OpsTest, VtmvRoundTrip) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/4);
  SetVreg<int8_t>(8, {10, 20, 30, 40});
  SetXReg(3, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));  // row 0.

  // vtmv.t.v: v8 -> tile0 row0. func6=010111, vs2=8, vs1(rs1)=3, func3=110.
  Execute(MakeVArith(0b010'111, /*vs2=*/8, /*vs1=*/3, 0b110, /*vd=*/0));
  EXPECT_EQ((matrix_->GetElem<uint8_t>(0, 0, 0, 8)), 10);
  EXPECT_EQ((matrix_->GetElem<uint8_t>(0, 0, 3, 8)), 40);

  // vtmv.v.t: tile0 row0 -> v24. func6=010000, vs2=11111, vs1(rs1)=3, vd=24.
  Execute(MakeVArith(0b010'000, /*vs2=*/0b11111, /*vs1=*/3, 0b110, /*vd=*/24));
  auto [reg, unused] = state_->GetRegister<RVVectorRegister>("v24");
  auto* v24 = reg->data_buffer();
  EXPECT_EQ(v24->Get<int8_t>(0), 10);
  EXPECT_EQ(v24->Get<int8_t>(1), 20);
  EXPECT_EQ(v24->Get<int8_t>(2), 30);
  EXPECT_EQ(v24->Get<int8_t>(3), 40);
}

// vtse32 then vtle32 round-trips a tile row through memory.
TEST_F(CoralNPUM4OpsTest, VtseVtleRoundTrip) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  matrix_->SetElem<int32_t>(0, 0, 0, 32, 0x11111111);
  matrix_->SetElem<int32_t>(0, 0, 1, 32, 0x22222222);

  SetXReg(5, kDataAddress);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));  // row 0.
  // vtse32 x4, (x5): nf=010 (eew 32), store-fp opcode.
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));

  // Memory now holds the two words.
  auto db = state_->db_factory()->Allocate<uint32_t>(2);
  memory_->Load(kDataAddress, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint32_t>(0), 0x11111111u);
  EXPECT_EQ(db->Get<uint32_t>(1), 0x22222222u);
  db->DecRef();

  // vtle32 x4, (x5) into tile0 row 1.
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/1));  // row 1.
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b000'0111));
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, 0, 32)),
            static_cast<int32_t>(0x11111111));
  EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, 1, 32)),
            static_cast<int32_t>(0x22222222));
}

// vtse32 then vtle32 round-trips a full tile row through memory.
TEST_F(CoralNPUM4OpsTest, VtseVtleRoundTrip_FullRow) {
  // mtype=0x3 (eew=32), vtype=0, tn=16 to cover a larger row that exceeds te/2.
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/16);
  for (int i = 0; i < 16; i++) {
    matrix_->SetElem<int32_t>(0, 0, i, 32, 0x11111110 + i);
  }

  SetXReg(5, kDataAddress);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));  // row 0.
  // vtse32 x4, (x5)
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));

  // vtle32 x4, (x5) into tile0 row 1.
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/1));  // row 1.
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b000'0111));

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ((matrix_->GetElem<int32_t>(0, 1, i, 32)),
              static_cast<int32_t>(0x11111110 + i));
  }
}

TEST_F(CoralNPUM4OpsTest, DecoderInfo) {
  EXPECT_GT(decoder_->GetNumOpcodes(), 0);
  EXPECT_NE(decoder_->GetOpcodeName(0), nullptr);
}

TEST_F(CoralNPUM4OpsTest, DecoderInvalidAddress) {
  // Decode at address without execute permission
  Instruction* inst = decoder_->DecodeInstruction(0xDeadBeef);
  EXPECT_EQ(inst->size(), 1);
  inst->Execute(nullptr);
  EXPECT_TRUE(was_trap_handler_called_);
  inst->DecRef();
}

TEST_F(CoralNPUM4OpsTest, AsUint64) {
  auto result = state_->csr_set()->GetCsr(0xC23);  // kRiscVMtypeCsrIndex
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)->AsUint64(), 0);
}

TEST_F(CoralNPUM4OpsTest, MemoryAccessFault) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  SetXReg(5, 0xDeadBeef);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));
  // vtse32 should trigger access fault
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));
  EXPECT_TRUE(was_trap_handler_called_);
  was_trap_handler_called_ = false;

  // vtle32 should trigger access fault
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b000'0111));
  EXPECT_TRUE(was_trap_handler_called_);
}

TEST_F(CoralNPUM4OpsTest, MemoryAccessFaultBoundary) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);

  // Vector length is 2. For EEW=32, count=2, esize=4. Total bytes = 8.
  // Valid range is 0x10000 to 0x10FFF.
  // Address 0x10FF8 + 8 bytes = 0x11000 (just past end). So 0x10FF8 is the last
  // valid start address.

  // Test that 0x10FF8 succeeds (will fail if esize > 4).
  SetXReg(5, 0x10FF8);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));
  was_trap_handler_called_ = false;
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));  // vtse32
  EXPECT_FALSE(was_trap_handler_called_);

  was_trap_handler_called_ = false;
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b000'0111));  // vtle32
  EXPECT_FALSE(was_trap_handler_called_);

  // Test that 0x10FF9 fails (will succeed if esize < 4).
  SetXReg(5, 0x10FF9);
  was_trap_handler_called_ = false;
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));  // vtse32
  EXPECT_TRUE(was_trap_handler_called_);

  was_trap_handler_called_ = false;
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b000'0111));  // vtle32
  EXPECT_TRUE(was_trap_handler_called_);
}

TEST_F(CoralNPUM4OpsTest, MemoryAccessFault_Vtse32_Mutant) {
  // te = 32 by default.
  // Original logic: tile_element_width=32, effective_tile_edge = te = 32.
  // Mutated logic: tile_element_width=32, effective_tile_edge = te / 2 = 16.
  // tn=32. count = min(tn, effective_tile_edge). Original: count=32. Mutated:
  // count=16.
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/32);

  // Valid memory range is 0x10000 to 0x10FFF.
  // We want to test accessing 32 elements (128 bytes).
  // If base = 0x11000 - 64 = 0x10FC0.
  // 32 elements = 128 bytes. End = 0x11040 (Faults).
  // 16 elements = 64 bytes. End = 0x11000 (Does not fault).
  SetXReg(5, 0x10FC0);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));

  was_trap_handler_called_ = false;
  Execute(MakeVtMem(/*nf=*/0b010, /*rs2=*/4, /*rs1=*/5, 0b010'0111));  // vtse32
  EXPECT_TRUE(was_trap_handler_called_);  // Original must trap. Mutant won't.
}

// vtse64 then vtle64 round-trips a tile row through memory.
TEST_F(CoralNPUM4OpsTest, VtseVtle64RoundTrip) {
  Configure(/*mtype=*/0x3 | (2u << 5) | (2u << 10), /*vtype=*/0, /*tn=*/2);
  matrix_->SetElem<int64_t>(0, 0, 0, 64, 0x1111111122222222ll);
  matrix_->SetElem<int64_t>(0, 0, 1, 64, 0x3333333344444444ll);

  SetXReg(5, kDataAddress);
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/0));  // row 0.
  // vtse64 x4, (x5): nf=011 (eew 64), store-fp opcode.
  Execute(MakeVtMem(/*nf=*/0b011, /*rs2=*/4, /*rs1=*/5, 0b010'0111));

  // Memory now holds the two double words.
  auto db = state_->db_factory()->Allocate<uint64_t>(2);
  memory_->Load(kDataAddress, db, nullptr, nullptr);
  EXPECT_EQ(db->Get<uint64_t>(0), 0x1111111122222222ll);
  EXPECT_EQ(db->Get<uint64_t>(1), 0x3333333344444444ll);
  db->DecRef();

  // vtle64 x4, (x5) into tile0 row 1.
  SetXReg(4, MakeTss(/*tile=*/0, /*pattern=*/0, /*index=*/1));  // row 1.
  Execute(MakeVtMem(/*nf=*/0b011, /*rs2=*/4, /*rs1=*/5, 0b000'0111));
  EXPECT_EQ((matrix_->GetElem<int64_t>(0, 1, 0, 64)),
            static_cast<int64_t>(0x1111111122222222ll));
  EXPECT_EQ((matrix_->GetElem<int64_t>(0, 1, 1, 64)),
            static_cast<int64_t>(0x3333333344444444ll));
}

}  // namespace
