#include "sim/kelvin_encoding.h"

#include <any>
#include <cstdint>
#include <type_traits>

#include "sim/kelvin_enums.h"
#include "sim/kelvin_state.h"
#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_register.h"
#include "mpact/sim/generic/register.h"

namespace {

using kelvin::sim::KelvinState;
using kelvin::sim::isa32::KelvinEncoding;
using SlotEnum = kelvin::sim::isa32::SlotEnum;
using OpcodeEnum = kelvin::sim::isa32::OpcodeEnum;
using SourceOpEnum = kelvin::sim::isa32::SourceOpEnum;
using RV32VectorSourceOperand = mpact::sim::riscv::RV32VectorSourceOperand;
using RV32SourceOperand = mpact::sim::generic::RegisterSourceOperand<uint64_t>;
using DestOpEnum = kelvin::sim::isa32::DestOpEnum;
using RV32VectorDestOperand = mpact::sim::riscv::RV32VectorDestinationOperand;
using RV32DestOperand =
    mpact::sim::generic::RegisterDestinationOperand<uint64_t>;

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

  template <typename T>
  T *EncodeOpHelper(uint32_t inst_word, OpcodeEnum opcode, std::any op) const {
    enc_->ParseInstruction(inst_word);
    EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), opcode);
    if (std::is_same<T, RV32SourceOperand>::value ||
        std::is_same<T, RV32VectorSourceOperand>::value) {
      auto *source = enc_->GetSource(SlotEnum::kKelvin, 0, opcode,
                                     std::any_cast<SourceOpEnum>(op), 0);
      return reinterpret_cast<T *>(source);
    }
    auto *dest = enc_->GetDestination(SlotEnum::kKelvin, 0, opcode,
                                      std::any_cast<DestOpEnum>(op), 0,
                                      /*latency=*/0);
    return reinterpret_cast<T *>(dest);
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

static inline uint32_t SetSz(uint32_t iword, uint32_t sz) {
  return (iword | ((sz & 0x3) << 12));
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

TEST_F(KelvinEncodingTest, KelvinVldEncodeXs1Xs2) {
  enc_->ParseInstruction(SetRs1(kVld, kRdValue));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kVldBX);
  enc_->ParseInstruction(SetSz(SetRs1(kVld, kRdValue), 0b1));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kVldHX);
  // Test vld.w.l.xx
  auto *src = EncodeOpHelper<RV32SourceOperand>(
      SetSz(SetRs1(kVld, kRdValue), 0b10) | (0b10 << 20 /* xs2 */) |
          (0b1 << 26 /* length */),
      OpcodeEnum::kVldWLXx, SourceOpEnum::kVs1);
  EXPECT_EQ(src->AsString(), "ra");
  delete src;
  src = EncodeOpHelper<RV32SourceOperand>(
      SetSz(SetRs1(kVld, kRdValue), 0b10) | (0b10 << 20 /* xs2 */) |
          (0b1 << 26 /* length */),
      OpcodeEnum::kVldWLXx, SourceOpEnum::kVs2);
  EXPECT_EQ(src->AsString(), "sp");
  delete src;
}

TEST_F(KelvinEncodingTest, KelvinVstEncodeXs1Xs2Vd) {
  constexpr uint32_t kVstBase = 0b001000'000000'000000'00'000000'0'111'11;
  // Test vd in vst.b.x as source
  auto *v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      SetRs1(kVstBase, kRdValue), OpcodeEnum::kVstBX, SourceOpEnum::kVd);
  EXPECT_EQ(v_src->AsString(), "v0");
  delete v_src;

  // Test xs1 in vst.w.l.xx as destination
  auto *dest = EncodeOpHelper<RV32DestOperand>(
      SetSz(SetRs1(kVstBase, kRdValue), 0b10) | (0b10 << 20 /* xs2 */) |
          (1 << 26 /* length */),
      OpcodeEnum::kVstWLXx, DestOpEnum::kVs1);
  EXPECT_EQ(dest->AsString(), "ra");
  delete dest;

  // Test xs2 in vstq.b.s.xx as source
  auto *src = EncodeOpHelper<RV32SourceOperand>(
      SetRs1(kVstBase, kRdValue) | (1 << 30 /* vstq */) |
          (0b10 << 20 /* xs2 */) | (1 << 27 /* stride */),
      OpcodeEnum::kVstqBSXx, SourceOpEnum::kVs2);
  EXPECT_EQ(src->AsString(), "sp");
  delete src;
}

TEST_F(KelvinEncodingTest, KelvinWideningVs1) {
  constexpr uint32_t kVSransBase = 0b010000'000001'000000'00'001000'0'010'00;
  auto *v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      kVSransBase, OpcodeEnum::kVsransBVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsrans.b.r.vv
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      kVSransBase | (1 << 27 /* vsrans.r */), OpcodeEnum::kVsransBRVv,
      SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsrans.h.vv
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      SetSz(kVSransBase, 0b1), OpcodeEnum::kVsransHVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vsraqs.b.vv
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      kVSransBase | (1 << 29 /* vsraqs */), OpcodeEnum::kVsraqsBVv,
      SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 4);
  delete v_src;

  // Test vsraqs.b.r.vv
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      kVSransBase | (1 << 29) | (1 << 27), OpcodeEnum::kVsraqsBRVv,
      SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 4);
  delete v_src;

  // Test illegal vsrans (vsrans.w.vv)
  enc_->ParseInstruction(SetSz(kVSransBase, 0b10));
  EXPECT_EQ(enc_->GetOpcode(SlotEnum::kKelvin, 0), OpcodeEnum::kNone);

  // Test vacc.h.vv
  constexpr uint32_t kVAccsBase = 0b001010'000001'000000'00'001000'0'100'00;
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      SetSz(kVAccsBase, 0b1), OpcodeEnum::kVaccHVv, SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;

  // Test vacc.w.u.vv
  v_src = EncodeOpHelper<RV32VectorSourceOperand>(
      SetSz(kVAccsBase, 0b10) | (1 << 26), OpcodeEnum::kVaccWUVv,
      SourceOpEnum::kVs1);
  EXPECT_EQ(v_src->size(), 2);
  delete v_src;
}

TEST_F(KelvinEncodingTest, KelvinWideningVd) {
  // No widening for vld
  auto *v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetRs1(kVld, kRdValue), OpcodeEnum::kVldBX, DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 1);
  delete v_dest;

  // Test vaddw.h.vv
  constexpr uint32_t kVAddwBase = 0b000100'000001'000000'00'001000'0'100'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetSz(kVAddwBase, 0b1), OpcodeEnum::kVaddwHVv, DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vsubw.w.u.vv
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetSz(kVAddwBase, 0b10) | (0b11 << 26 /* vsubw.u */),
      OpcodeEnum::kVsubwWUVv, DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vmvp.vv
  constexpr uint32_t kVMvpBase = 0b001101'000001'000000'00'001000'0'001'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(kVMvpBase, OpcodeEnum::kVmvpVv,
                                                 DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vmulw.h.vv
  constexpr uint32_t kVMulwBase = 0b000100'000001'000000'00'001000'0'011'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetSz(kVMulwBase, 0b1), OpcodeEnum::kVmulwHVv, DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vmulw.w.u.vv
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetSz(kVMulwBase, 0b10) | (1 << 26 /* vmulw.u */), OpcodeEnum::kVmulwWUVv,
      DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vevnodd.b.vv
  constexpr uint32_t kVEvnoddBase = 0b011000'000001'000000'00'001000'0'110'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      kVEvnoddBase | (0b10 << 26 /* vevenodd */), OpcodeEnum::kVevnoddBVv,
      DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vpadd.w.vv
  constexpr uint32_t kVPAdd = 0b001100'000001'000000'10'001000'0'100'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(kVPAdd, OpcodeEnum::kVpaddWVv,
                                                 DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vpsub.h.u.vv
  constexpr uint32_t kVPSub = 0b001111'000001'000000'01'001000'0'100'00;
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(kVPSub, OpcodeEnum::kVpsubHUVv,
                                                 DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;

  // Test vzip.h.vv
  v_dest = EncodeOpHelper<RV32VectorDestOperand>(
      SetSz(kVEvnoddBase, 0b1) | (0b100 << 26 /* vzip */), OpcodeEnum::kVzipHVv,
      DestOpEnum::kVd);
  EXPECT_EQ(v_dest->size(), 2);
  delete v_dest;
}

}  // namespace
