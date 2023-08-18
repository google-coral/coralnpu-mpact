#include "sim/kelvin_encoding.h"

#include <cstdint>

#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_register.h"

namespace {

using kelvin::sim::KelvinState;
using kelvin::sim::isa32::KelvinEncoding;
using SlotEnum = kelvin::sim::isa32::SlotEnum;
using OpcodeEnum = kelvin::sim::isa32::OpcodeEnum;
using SourceOpEnum = kelvin::sim::isa32::SourceOpEnum;
using RV32VectorSourceOperand = mpact::sim::riscv::RV32VectorSourceOperand;

// RV32I
constexpr uint32_t kLui = 0b0000000000000000000000000'0110111;
constexpr uint32_t kAuipc = 0b0000000000000000000000000'0010111;
constexpr uint32_t kJal = 0b00000000000000000000'00000'1101111;
constexpr uint32_t kJalr = 0b00000000000'00000'000'00000'1100111;
constexpr uint32_t kBeq = 0b0000000'00000'00000'000'00000'1100011;
constexpr uint32_t kBne = 0b0000000'00000'00000'001'00000'1100011;
constexpr uint32_t kBlt = 0b0000000'00000'00000'100'00000'1100011;
constexpr uint32_t kBge = 0b0000000'00000'00000'101'00000'1100011;
constexpr uint32_t kBltu = 0b0000000'00000'00000'110'00000'1100011;
constexpr uint32_t kBgeu = 0b0000000'00000'00000'111'00000'1100011;
constexpr uint32_t kLb = 0b000000000000'00000'000'00000'0000011;
constexpr uint32_t kLh = 0b000000000000'00000'001'00000'0000011;
constexpr uint32_t kLw = 0b000000000000'00000'010'00000'0000011;
constexpr uint32_t kLbu = 0b000000000000'00000'100'00000'0000011;
constexpr uint32_t kLhu = 0b000000000000'00000'101'00000'0000011;
constexpr uint32_t kSb = 0b0000000'00000'00000'000'00000'0100011;
constexpr uint32_t kSh = 0b0000000'00000'00000'001'00000'0100011;
constexpr uint32_t kSw = 0b0000000'00000'00000'010'00000'0100011;
constexpr uint32_t kAddi = 0b000000000000'00000'000'00000'0010011;
constexpr uint32_t kSlti = 0b000000000000'00000'010'00000'0010011;
constexpr uint32_t kSltiu = 0b000000000000'00000'011'00000'0010011;
constexpr uint32_t kXori = 0b000000000000'00000'100'00000'0010011;
constexpr uint32_t kOri = 0b000000000000'00000'110'00000'0010011;
constexpr uint32_t kAndi = 0b000000000000'00000'111'00000'0010011;
constexpr uint32_t kSlli = 0b0000000'00000'00000'001'00000'0010011;
constexpr uint32_t kSrli = 0b0000000'00000'00000'101'00000'0010011;
constexpr uint32_t kSrai = 0b0100000'00000'00000'101'00000'0010011;
constexpr uint32_t kAdd = 0b0000000'00000'00000'000'00000'0110011;
constexpr uint32_t kSub = 0b0100000'00000'00000'000'00000'0110011;
constexpr uint32_t kSll = 0b0000000'00000'00000'001'00000'0110011;
constexpr uint32_t kSlt = 0b0000000'00000'00000'010'00000'0110011;
constexpr uint32_t kSltu = 0b0000000'00000'00000'011'00000'0110011;
constexpr uint32_t kXor = 0b0000000'00000'00000'100'00000'0110011;
constexpr uint32_t kSrl = 0b0000000'00000'00000'101'00000'0110011;
constexpr uint32_t kSra = 0b0100000'00000'00000'101'00000'0110011;
constexpr uint32_t kOr = 0b0000000'00000'00000'110'00000'0110011;
constexpr uint32_t kAnd = 0b0000000'00000'00000'111'00000'0110011;
constexpr uint32_t kFence = 0b000000000000'00000'000'00000'0001111;
constexpr uint32_t kEcall = 0b000000000000'00000'000'00000'1110011;
constexpr uint32_t kEbreak = 0b000000000001'00000'000'00000'1110011;
constexpr uint32_t kMpause = 0b000010000000'00000'000'00000'1110011;
// Kelvin Memory ops
constexpr uint32_t kFlushall = 0b001001100000'00000'000'00000'1110111;
constexpr uint32_t kFlushat = 0b001001100000'00000'000'00000'1110111;
// RV32 Zifencei
constexpr uint32_t kFencei = 0b000000000000'00000'001'00000'0001111;
// RV32 Zicsr
constexpr uint32_t kCsrw = 0b000000000000'00000'001'00000'1110011;
constexpr uint32_t kCsrs = 0b000000000000'00000'010'00000'1110011;
constexpr uint32_t kCsrc = 0b000000000000'00000'011'00000'1110011;
constexpr uint32_t kCsrwi = 0b000000000000'00000'101'00000'1110011;
constexpr uint32_t kCsrsi = 0b000000000000'00000'110'00000'1110011;
constexpr uint32_t kCsrci = 0b000000000000'00000'111'00000'1110011;
// RV32M
constexpr uint32_t kMul = 0b0000001'00000'00000'000'00000'0110011;
constexpr uint32_t kMulh = 0b0000001'00000'00000'001'00000'0110011;
constexpr uint32_t kMulhsu = 0b0000001'00000'00000'010'00000'0110011;
constexpr uint32_t kMulhu = 0b0000001'00000'00000'011'00000'0110011;
constexpr uint32_t kDiv = 0b0000001'00000'00000'100'00000'0110011;
constexpr uint32_t kDivu = 0b0000001'00000'00000'101'00000'0110011;
constexpr uint32_t kRem = 0b0000001'00000'00000'110'00000'0110011;
constexpr uint32_t kRemu = 0b0000001'00000'00000'111'00000'0110011;

// Kelvin System Op
constexpr uint32_t kGetMaxVl = 0b0001'0'00'00000'00000'000'00000'111'0111;

// Kelvin Logging Op
constexpr uint32_t kFLog = 0b011'1100'00000'00000'000'00000'111'0111;

// Kelvin VLd
constexpr uint32_t kVld = 0b000000'000000'000000'00'000000'0'111'11;

class KelvinEncodingTest : public testing::Test {
 protected:
  KelvinEncodingTest() {
    state_ = new KelvinState("test", mpact::sim::riscv::RiscVXlen::RV32);
    enc_ = new KelvinEncoding(state_);
  }
  ~KelvinEncodingTest() override {
    delete enc_;
    delete state_;
  }

  RV32VectorSourceOperand *VectorSourceEncodeHelper(
      uint32_t inst_word, OpcodeEnum opcode, SourceOpEnum source_op) const {
    enc_->ParseInstruction(inst_word);
    EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), opcode);
    auto source = enc_->GetSource(SlotEnum::kKelvin, 0, opcode, source_op, 0);
    return reinterpret_cast<mpact::sim::riscv::RV32VectorSourceOperand *>(
        source);
  }

  KelvinState *state_;
  KelvinEncoding *enc_;
};

constexpr int kRdValue = 1;
constexpr int kSuccValue = 0xf;
constexpr int kPredValue = 0xf;

static uint32_t SetRd(uint32_t iword, uint32_t rdval) {
  return (iword | ((rdval & 0x1f) << 7));
}

static uint32_t SetRs1(uint32_t iword, uint32_t rsval) {
  return (iword | ((rsval & 0x1f) << 15));
}

static uint32_t SetRs2(uint32_t iword, uint32_t rsval) {
  return (iword | ((rsval & 0x1f) << 20));
}

static uint32_t SetPred(uint32_t iword, uint32_t pred) {
  return (iword | ((pred & 0xf) << 24));
}

static uint32_t SetSucc(uint32_t iword, uint32_t succ) {
  return (iword | ((succ & 0xf) << 20));
}

TEST_F(KelvinEncodingTest, RV32IOpcodes) {
  enc_->ParseInstruction(SetRd(kLui, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLui);
  enc_->ParseInstruction(SetRd(kAuipc, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kAuipc);
  enc_->ParseInstruction(SetRd(kJal, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kJal);
  enc_->ParseInstruction(SetRd(kJalr, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kJalr);
  enc_->ParseInstruction(kBeq);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBeq);
  enc_->ParseInstruction(kBne);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBne);
  enc_->ParseInstruction(kBlt);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBlt);
  enc_->ParseInstruction(kBge);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBge);
  enc_->ParseInstruction(kBltu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBltu);
  enc_->ParseInstruction(kBgeu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kBgeu);
  enc_->ParseInstruction(SetRd(kLb, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLb);
  enc_->ParseInstruction(SetRd(kLh, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLh);
  enc_->ParseInstruction(SetRd(kLw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLw);
  enc_->ParseInstruction(SetRd(kLbu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLbu);
  enc_->ParseInstruction(SetRd(kLhu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kLhu);
  enc_->ParseInstruction(SetRd(kSb, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSb);
  enc_->ParseInstruction(SetRd(kSh, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSh);
  enc_->ParseInstruction(SetRd(kSw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSw);
  enc_->ParseInstruction(SetRd(kAddi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kAddi);
  enc_->ParseInstruction(SetRd(kSlti, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSlti);
  enc_->ParseInstruction(SetRd(kSltiu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSltiu);
  enc_->ParseInstruction(SetRd(kXori, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kXori);
  enc_->ParseInstruction(SetRd(kOri, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kOri);
  enc_->ParseInstruction(SetRd(kAndi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kAndi);
  enc_->ParseInstruction(SetRd(kSlli, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSlli);
  enc_->ParseInstruction(SetRd(kSrli, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSrli);
  enc_->ParseInstruction(SetRd(kSrai, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSrai);
  enc_->ParseInstruction(SetRd(kAdd, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kAdd);
  enc_->ParseInstruction(SetRd(kSub, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSub);
  enc_->ParseInstruction(SetRd(kSll, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSll);
  enc_->ParseInstruction(SetRd(kSlt, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSlt);
  enc_->ParseInstruction(SetRd(kSltu, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSltu);
  enc_->ParseInstruction(SetRd(kXor, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kXor);
  enc_->ParseInstruction(SetRd(kSrl, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSrl);
  enc_->ParseInstruction(SetRd(kSra, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSra);
  enc_->ParseInstruction(SetRd(kOr, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kOr);
  enc_->ParseInstruction(SetRd(kAnd, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kAnd);
  enc_->ParseInstruction(SetSucc(SetPred(kFence, kPredValue), kSuccValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kFence);
  enc_->ParseInstruction(kEcall);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kEcall);
  enc_->ParseInstruction(kEbreak);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kEbreak);
  enc_->ParseInstruction(kMpause);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kMpause);
}

TEST_F(KelvinEncodingTest, KelvinMemoryOpcodes) {
  enc_->ParseInstruction(kFlushall);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kFlushall);
  enc_->ParseInstruction(SetRs1(kFlushat, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kFlushat);
}

TEST_F(KelvinEncodingTest, KelvinSystemOpcodes) {
  enc_->ParseInstruction(SetRd(kGetMaxVl, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kGetmaxvlB);
  enc_->ParseInstruction(SetRd(SetRs1(kGetMaxVl, kRdValue), kRdValue) |
                         (0b1) << 25);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kGetvlHX);
  enc_->ParseInstruction(
      SetRd(SetRs1(SetRs2(kGetMaxVl, kRdValue), kRdValue), kRdValue) |
      (0b10 << 25));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kGetvlWXx);
}

TEST_F(KelvinEncodingTest, KelvinLogOpcodes) {
  enc_->ParseInstruction(SetRs1(kFLog, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kFlog);
  enc_->ParseInstruction(SetRs1(kFLog, kRdValue) | (0b01 << 12));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kSlog);
  enc_->ParseInstruction(SetRs1(kFLog, kRdValue) | (0b10 << 12));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kClog);
  enc_->ParseInstruction(SetRs1(kFLog, kRdValue) | (0b11 << 12));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kKlog);
}

TEST_F(KelvinEncodingTest, KelvinVldOpcodes) {
  enc_->ParseInstruction(SetRs1(kVld, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kVldBX);
  enc_->ParseInstruction(SetRs1(kVld, kRdValue) | (0b01 << 12 /* sz */));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kVldHX);
  enc_->ParseInstruction(SetRs1(kVld, kRdValue) | (0b10 << 12 /* sz */) |
                         (0b10 << 20 /* xs2 */) | (0b1 << 26 /* length */));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kVldWLXx);
}

TEST_F(KelvinEncodingTest, ZifenceiOpcodes) {
  // RV32 Zifencei
  enc_->ParseInstruction(kFencei);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kFencei);
}

TEST_F(KelvinEncodingTest, ZicsrOpcodes) {
  // RV32 Zicsr
  enc_->ParseInstruction(SetRd(kCsrw, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrw);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrs, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrs);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrc, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrc);
  enc_->ParseInstruction(kCsrw);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrwNr);
  enc_->ParseInstruction(kCsrs);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrsNw);
  enc_->ParseInstruction(kCsrc);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrcNw);
  enc_->ParseInstruction(SetRd(kCsrwi, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrwi);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrsi, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrsi);
  enc_->ParseInstruction(SetRd(SetRs1(kCsrci, kRdValue), kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrci);
  enc_->ParseInstruction(kCsrwi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrwiNr);
  enc_->ParseInstruction(kCsrsi);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrsiNw);
  enc_->ParseInstruction(kCsrci);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kCsrrciNw);
}

TEST_F(KelvinEncodingTest, RV32MOpcodes) {
  // RV32M
  enc_->ParseInstruction(kMul);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kMul);
  enc_->ParseInstruction(kMulh);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kMulh);
  enc_->ParseInstruction(kMulhsu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kMulhsu);
  enc_->ParseInstruction(kMulhu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kMulhu);
  enc_->ParseInstruction(kDiv);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kDiv);
  enc_->ParseInstruction(kDivu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kDivu);
  enc_->ParseInstruction(kRem);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kRem);
  enc_->ParseInstruction(kRemu);
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kRemu);
}

TEST_F(KelvinEncodingTest, VsraxsWideningVs1) {
  constexpr uint32_t kVSransBase = 0b010000'000001'000000'00'001000'0'010'00;
  auto v_src = VectorSourceEncodeHelper(kVSransBase, OpcodeEnum::kVsransBVv,
                                        SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsrans.b.r.vv
  v_src = VectorSourceEncodeHelper(kVSransBase | (1 << 27),
                                   OpcodeEnum::kVsransBRVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsrans.h.vv
  v_src = VectorSourceEncodeHelper(kVSransBase | (1 << 12),
                                   OpcodeEnum::kVsransHVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsraqs.b.vv
  v_src = VectorSourceEncodeHelper(kVSransBase | (1 << 29),
                                   OpcodeEnum::kVsraqsBVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 4);
  delete v_src;

  // Test vsraqs.b.r.vv
  v_src = VectorSourceEncodeHelper(kVSransBase | (1 << 29) | (1 << 27),
                                   OpcodeEnum::kVsraqsBRVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 4);
  delete v_src;

  // Test illegal vsrans (vsrans.w.vv)
  enc_->ParseInstruction(kVSransBase | (2 << 12));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kNone);
}

}  // namespace
