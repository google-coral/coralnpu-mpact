#include "sim/kelvin_vector_instructions.h"

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "sim/test/kelvin_vector_instructions_test_base.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mpact/sim/generic/instruction.h"

// This file contains the tests for testing kelvin vector binary instructions.

namespace {

using mpact::sim::generic::Instruction;

// Semantic functions.
using kelvin::sim::KelvinVAbsd;
using kelvin::sim::KelvinVAcc;
using kelvin::sim::KelvinVAdd;
using kelvin::sim::KelvinVAdd3;
using kelvin::sim::KelvinVAdds;
using kelvin::sim::KelvinVAddsu;
using kelvin::sim::KelvinVAddw;
using kelvin::sim::KelvinVAnd;
using kelvin::sim::KelvinVClb;
using kelvin::sim::KelvinVClz;
using kelvin::sim::KelvinVCpop;
using kelvin::sim::KelvinVDmulh;
using kelvin::sim::KelvinVEq;
using kelvin::sim::KelvinVEvn;
using kelvin::sim::KelvinVEvnodd;
using kelvin::sim::KelvinVGe;
using kelvin::sim::KelvinVGt;
using kelvin::sim::KelvinVHadd;
using kelvin::sim::KelvinVHsub;
using kelvin::sim::KelvinVLe;
using kelvin::sim::KelvinVLt;
using kelvin::sim::KelvinVMacc;
using kelvin::sim::KelvinVMadd;
using kelvin::sim::KelvinVMax;
using kelvin::sim::KelvinVMin;
using kelvin::sim::KelvinVMul;
using kelvin::sim::KelvinVMulh;
using kelvin::sim::KelvinVMuls;
using kelvin::sim::KelvinVMulw;
using kelvin::sim::KelvinVMv;
using kelvin::sim::KelvinVMvp;
using kelvin::sim::KelvinVNe;
using kelvin::sim::KelvinVNot;
using kelvin::sim::KelvinVOdd;
using kelvin::sim::KelvinVOr;
using kelvin::sim::KelvinVPadd;
using kelvin::sim::KelvinVPsub;
using kelvin::sim::KelvinVRev;
using kelvin::sim::KelvinVRor;
using kelvin::sim::KelvinVRSub;
using kelvin::sim::KelvinVSel;
using kelvin::sim::KelvinVShift;
using kelvin::sim::KelvinVSlidehn;
using kelvin::sim::KelvinVSlidehp;
using kelvin::sim::KelvinVSlidevn;
using kelvin::sim::KelvinVSlidevp;
using kelvin::sim::KelvinVSll;
using kelvin::sim::KelvinVSra;
using kelvin::sim::KelvinVSrans;
using kelvin::sim::KelvinVSrl;
using kelvin::sim::KelvinVSub;
using kelvin::sim::KelvinVSubs;
using kelvin::sim::KelvinVSubsu;
using kelvin::sim::KelvinVSubw;
using kelvin::sim::KelvinVXor;
using kelvin::sim::KelvinVZip;

constexpr bool kIsScalar = true;
constexpr bool kNonScalar = false;
constexpr bool kIsStripmine = true;
constexpr bool kNonStripmine = false;
constexpr bool kUnsigned = false;
constexpr bool kHalftypeOp = true;
constexpr bool kNonHalftypeOp = false;
constexpr bool kVmvpOp = true;
constexpr bool kNonVmvpOp = false;
constexpr bool kIsRounding = true;
constexpr bool kNonRounding = false;
constexpr bool kHorizontal = true;
constexpr bool kVertical = false;
constexpr bool kNonWidenDst = false;
constexpr bool kWidenDst = true;

class KelvinVectorInstructionsTest
    : public kelvin::sim::test::KelvinVectorInstructionsTestBase {
 public:
  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2>
  void KelvinVectorBinaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<TD>());

    // Test [VV, VX].{M} variants
    for (auto scalar : {kNonScalar, kIsScalar}) {
      for (auto stripmine : {kNonStripmine, kIsStripmine}) {
        auto op_name = absl::StrCat(name_with_type, "V", scalar ? "X" : "V",
                                    stripmine ? "M" : "");
        BinaryOpTestHelper<TD, TS1, TS2>(
            absl::bind_front(F<TD, TS1, TS2>::KelvinOp, scalar, stripmine),
            op_name, scalar, stripmine, F<TD, TS1, TS2>::Op);
      }
    }
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2, typename TNext1, typename... TNext>
  void KelvinVectorBinaryOpHelper(absl::string_view name) {
    KelvinVectorBinaryOpHelper<F, TD, TS1, TS2>(name);
    KelvinVectorBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename, typename> class F,
            bool is_signed = true>
  void KelvinVectorBinaryOpHelper(absl::string_view name) {
    if (is_signed) {
      KelvinVectorBinaryOpHelper<F, int8_t, int8_t, int8_t, int16_t, int16_t,
                                 int16_t, int32_t, int32_t, int32_t>(name);
    } else {
      KelvinVectorBinaryOpHelper<F, uint8_t, uint8_t, uint8_t, uint16_t,
                                 uint16_t, uint16_t, uint32_t, uint32_t,
                                 uint32_t>(name);
    }
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2>
  void KelvinHalftypeVectorBinaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<TD>());

    // Vector OP single vector.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::KelvinOp, kNonStripmine),
        absl::StrCat(name_with_type, "V"), kNonScalar, kNonStripmine,
        F<TD, TS1, TS2>::Op, kHalftypeOp);

    // Vector OP single vector stripmined.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::KelvinOp, kIsStripmine),
        absl::StrCat(name_with_type, "VM"), kNonScalar, kIsStripmine,
        F<TD, TS1, TS2>::Op, kHalftypeOp);
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2, typename TNext1, typename... TNext>
  void KelvinHalftypeVectorBinaryOpHelper(absl::string_view name) {
    KelvinHalftypeVectorBinaryOpHelper<F, TD, TS1, TS2>(name);
    KelvinHalftypeVectorBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2>
  void KelvinVectorVXBinaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<TD>());

    // Vector OP vector-scalar.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::KelvinOp, kNonStripmine),
        absl::StrCat(name_with_type, "VX"), kIsScalar, kNonStripmine,
        F<TD, TS1, TS2>::Op);

    // Vector OP vector-scalar stripmined.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::KelvinOp, kIsStripmine),
        absl::StrCat(name_with_type, "VXM"), kIsScalar, kIsStripmine,
        F<TD, TS1, TS2>::Op);
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2, typename TNext1, typename... TNext>
  void KelvinVectorVXBinaryOpHelper(absl::string_view name) {
    KelvinVectorVXBinaryOpHelper<F, TD, TS1, TS2>(name);
    KelvinVectorVXBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename, typename> class F, typename T>
  void KelvinVectorShiftBinaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<T>());

    // Test {R}.[VV, VX].{M} variants.
    for (auto rounding : {kNonRounding, kIsRounding}) {
      for (auto scalar : {kNonScalar, kIsScalar}) {
        for (auto stripmine : {kNonStripmine, kIsStripmine}) {
          auto op_name = absl::StrCat(name_with_type, rounding ? "R" : "", "V",
                                      scalar ? "X" : "V", stripmine ? "M" : "");
          BinaryOpTestHelper<T, T, T>(
              absl::bind_front(F<T, T, T>::KelvinOp, rounding, scalar,
                               stripmine),
              op_name, scalar, stripmine,
              absl::bind_front(F<T, T, T>::Op, rounding));
        }
      }
    }
  }

  template <template <typename, typename, typename> class F, typename T,
            typename TNext1, typename... TNext>
  void KelvinVectorShiftBinaryOpHelper(absl::string_view name) {
    KelvinVectorShiftBinaryOpHelper<F, T>(name);
    KelvinVectorShiftBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename> class F, typename TD, typename TS>
  void KelvinVectorUnaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<TD>());

    // Vector OP single vector.
    UnaryOpTestHelper<TD, TS>(
        absl::bind_front(F<TD, TS>::KelvinOp, kNonStripmine),
        absl::StrCat(name_with_type, "V"), kNonStripmine, F<TD, TS>::Op);

    // Vector OP single vector stripmined.
    UnaryOpTestHelper<TD, TS>(
        absl::bind_front(F<TD, TS>::KelvinOp, kIsStripmine),
        absl::StrCat(name_with_type, "VM"), kIsStripmine, F<TD, TS>::Op);
  }

  template <template <typename, typename> class F, typename TD, typename TS,
            typename TNext1, typename... TNext>
  void KelvinVectorUnaryOpHelper(absl::string_view name) {
    KelvinVectorUnaryOpHelper<F, TD, TS>(name);
    KelvinVectorUnaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename> class F, typename T>
  void KelvinSlideOpHelper(absl::string_view name, bool horizontal,
                           bool strip_mine) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<T>());

    for (int i = 1; i < 5; ++i) {
      BinaryOpTestHelper<T, T, T>(
          absl::bind_front(F<T>::KelvinOp, i, strip_mine),
          absl::StrCat(name_with_type, i, "V", strip_mine ? "M" : ""),
          kNonScalar, strip_mine, F<T>::Op,
          absl::bind_front(F<T>::kArgsGetter, horizontal, i), kNonHalftypeOp,
          kNonVmvpOp, kNonWidenDst);
    }
  }

  template <template <typename> class F, typename T, typename TNext1,
            typename... TNext>
  void KelvinSlideOpHelper(absl::string_view name, bool horizontal,
                           bool strip_mine) {
    KelvinSlideOpHelper<F, T>(name, horizontal, strip_mine);
    KelvinSlideOpHelper<F, TNext1, TNext...>(name, horizontal, strip_mine);
  }

  template <template <typename> class F, typename T>
  void KelvinShuffleOpHelper(absl::string_view name, bool widen_dst) {
    const auto name_with_type = absl::StrCat(name, KelvinTestTypeSuffix<T>());

    // Test [VV, VX].{M} variants.
    for (auto scalar : {kNonScalar, kIsScalar}) {
      for (auto stripmine : {kNonStripmine, kIsStripmine}) {
        auto op_name = absl::StrCat(name_with_type, "V", scalar ? "X" : "V",
                                    stripmine ? "M" : "");
        BinaryOpTestHelper<T, T, T>(
            absl::bind_front(F<T>::KelvinOp, scalar, stripmine), op_name,
            scalar, stripmine, F<T>::Op, F<T>::kArgsGetter, kNonHalftypeOp,
            kNonVmvpOp, widen_dst);
      }
    }
  }

  template <template <typename> class F, typename T, typename TNext1,
            typename... TNext>
  void KelvinShuffleOpHelper(absl::string_view name, bool widen_dst = false) {
    KelvinShuffleOpHelper<F, T>(name, widen_dst);
    KelvinShuffleOpHelper<F, TNext1, TNext...>(name, widen_dst);
  }
};

// Vector add.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs1_ext + vs2_ext);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAdd<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VAdd) {
  KelvinVectorBinaryOpHelper<VAddOp>("VAdd");
}

// Vector subtract.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs1_ext - vs2_ext);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSub<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VSub) {
  KelvinVectorBinaryOpHelper<VSubOp>("VSub");
}

// Vector reverse subtract.
template <typename Vd, typename Vs1, typename Vs2>
struct VRSubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs2_ext - vs1_ext);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVRSub<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VRsub) {
  KelvinVectorBinaryOpHelper<VRSubOp>("VRsub");
}

// Vector equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VEqOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 == vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVEq<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VEq) {
  KelvinVectorBinaryOpHelper<VEqOp>("VEq");
}

// Vector not equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VNeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 != vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVNe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VNe) {
  KelvinVectorBinaryOpHelper<VNeOp>("VNe");
}

// Vector less than.
template <typename Vd, typename Vs1, typename Vs2>
struct VLtOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 < vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVLt<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VLt) {
  KelvinVectorBinaryOpHelper<VLtOp>("VLt");
}

// Vector less than unsigned.
TEST_F(KelvinVectorInstructionsTest, VLtu) {
  KelvinVectorBinaryOpHelper<VLtOp, kUnsigned>("VLtu");
}

// Vector less than or equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VLeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 <= vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVLe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VLe) {
  KelvinVectorBinaryOpHelper<VLeOp>("VLe");
}

// Vector less than or equal unsigned.
TEST_F(KelvinVectorInstructionsTest, VLeu) {
  KelvinVectorBinaryOpHelper<VLeOp, kUnsigned>("VLeu");
}

// Vector greater than.
template <typename Vd, typename Vs1, typename Vs2>
struct VGtOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 > vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVGt<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VGt) {
  KelvinVectorBinaryOpHelper<VGtOp>("VGt");
}

// Vector greater than unsigned.
TEST_F(KelvinVectorInstructionsTest, VGtu) {
  KelvinVectorBinaryOpHelper<VGtOp, kUnsigned>("VGtu");
}

// Vector greater than or equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VGeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >= vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVGe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VGe) {
  KelvinVectorBinaryOpHelper<VGeOp>("VGe");
}

// Vector greater than or equal unsigned.
TEST_F(KelvinVectorInstructionsTest, VGeu) {
  KelvinVectorBinaryOpHelper<VGeOp, kUnsigned>("VGeu");
}

// Vector absolute difference.
template <typename Vd, typename Vs1, typename Vs2>
struct VAbsdOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    auto result = vs1_ext > vs2_ext ? vs1_ext - vs2_ext : vs2_ext - vs1_ext;
    return static_cast<Vd>(result);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAbsd<Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAbsd) {
  KelvinVectorBinaryOpHelper<VAbsdOp, uint8_t, int8_t, int8_t, uint16_t,
                             int16_t, int16_t, uint32_t, int32_t, int32_t>(
      "VAbsd");
}

TEST_F(KelvinVectorInstructionsTest, VAbsdu) {
  KelvinVectorBinaryOpHelper<VAbsdOp, kUnsigned>("VAbsdu");
}

// Vector max.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaxOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return std::max(vs1, vs2); }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMax<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VMax) {
  KelvinVectorBinaryOpHelper<VMaxOp>("VMax");
}

// Vector max unsigned.
TEST_F(KelvinVectorInstructionsTest, VMaxu) {
  KelvinVectorBinaryOpHelper<VMaxOp, kUnsigned>("VMaxu");
}

// Vector min.
template <typename Vd, typename Vs1, typename Vs2>
struct VMinOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return std::min(vs1, vs2); }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMin<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VMin) {
  KelvinVectorBinaryOpHelper<VMinOp>("VMin");
}

// Vector min unsigned.
TEST_F(KelvinVectorInstructionsTest, VMinu) {
  KelvinVectorBinaryOpHelper<VMinOp, kUnsigned>("VMinu");
}

// Vector add3.
template <typename Vd, typename Vs1, typename Vs2>
struct VAdd3Op {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    int64_t vd_ext = static_cast<int64_t>(vd);
    return static_cast<Vd>(vd_ext + vs1_ext + vs2_ext);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAdd3<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VAdd3) {
  KelvinVectorBinaryOpHelper<VAdd3Op>("VAdd3");
}

// Vector saturated add.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddsOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    // typenames Vs1 and Vs2 can be up to int32_t. Promoting to int64_t to
    // prevent overflow.
    int64_t sum = static_cast<int64_t>(vs1) + static_cast<int64_t>(vs2);
    return std::min<int64_t>(
        std::max<int64_t>(std::numeric_limits<Vd>::min(), sum),
        std::numeric_limits<Vd>::max());
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAdds<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAdds) {
  KelvinVectorBinaryOpHelper<VAddsOp>("VAdds");
}

// Vector saturated unsigned add.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddsuOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    uint64_t sum = static_cast<uint64_t>(vs1) + static_cast<uint64_t>(vs2);
    return std::min<uint64_t>(std::numeric_limits<Vd>::max(), sum);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAddsu<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAddsu) {
  KelvinVectorBinaryOpHelper<VAddsuOp, kUnsigned>("VAddsu");
}

// Vector saturated sub.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubsOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    // typenames Vs1 and Vs2 can be up to int32_t. Promoting to int64_t to
    // prevent overflow.
    int64_t sub = static_cast<int64_t>(vs1) - static_cast<int64_t>(vs2);
    return std::min<int64_t>(
        std::max<int64_t>(std::numeric_limits<Vd>::min(), sub),
        std::numeric_limits<Vd>::max());
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSubs<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSubs) {
  KelvinVectorBinaryOpHelper<VSubsOp>("VSubs");
}

// Vector saturated unsigned sub.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubsuOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 < vs2 ? 0 : vs1 - vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSubsu<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSubsu) {
  KelvinVectorBinaryOpHelper<VSubsuOp, kUnsigned>("VSubsu");
}

// Vector addition with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) + static_cast<Vd>(vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAddw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAddw) {
  KelvinVectorBinaryOpHelper<VAddwOp, int16_t, int8_t, int8_t, int32_t, int16_t,
                             int16_t>("VAddwOp");
}

TEST_F(KelvinVectorInstructionsTest, VAddwu) {
  KelvinVectorBinaryOpHelper<VAddwOp, uint16_t, uint8_t, uint8_t, uint32_t,
                             uint16_t, uint16_t>("VAddwuOp");
}

// Vector subtraction with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) - static_cast<Vd>(vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSubw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSubw) {
  KelvinVectorBinaryOpHelper<VSubwOp, int16_t, int8_t, int8_t, int32_t, int16_t,
                             int16_t>("VSubwOp");
}

TEST_F(KelvinVectorInstructionsTest, VSubwu) {
  KelvinVectorBinaryOpHelper<VSubwOp, uint16_t, uint8_t, uint8_t, uint32_t,
                             uint16_t, uint16_t>("VSubwuOp");
}

// Vector accumulate with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VAccOp {
  static Vd Op(Vd vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs1_ext + vs2_ext);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAcc<Vd, Vs2>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAcc) {
  KelvinVectorBinaryOpHelper<VAccOp, int16_t, int16_t, int8_t, int32_t, int32_t,
                             int16_t>("VAccOp");
}

TEST_F(KelvinVectorInstructionsTest, VAccu) {
  KelvinVectorBinaryOpHelper<VAccOp, uint16_t, uint16_t, uint8_t, uint32_t,
                             uint32_t, uint16_t>("VAccuOp");
}

// Selects pairs from register
template <typename T>
static std::pair<T, T> PairwiseOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T> &vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T> &vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  int start_index = (op_num * vs1_size) + (2 * element_index);
  if (dest_reg_sub_index == 0) {
    return {vs1_value[start_index], vs1_value[start_index + 1]};
  }

  return {vs2_value[start_index], vs2_value[start_index + 1]};
}

// Vector packed add
template <typename Vd, typename Vs1, typename Vs2>
struct VPaddOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) + static_cast<Vd>(vs2);
  }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVPadd<Vd, Vs2>(strip_mine, inst);
  }
  static constexpr auto kArgsGetter = PairwiseOpArgsGetter<Vs1>;
};

TEST_F(KelvinVectorInstructionsTest, VPadd) {
  KelvinHalftypeVectorBinaryOpHelper<VPaddOp, int16_t, int8_t, int8_t, int32_t,
                                     int16_t, int16_t>("VPaddOp");
}

TEST_F(KelvinVectorInstructionsTest, VPaddu) {
  KelvinHalftypeVectorBinaryOpHelper<VPaddOp, uint16_t, uint8_t, uint8_t,
                                     uint32_t, uint16_t, uint16_t>("VPaddOp");
}

// Vector packed sub
template <typename Vd, typename Vs1, typename Vs2>
struct VPsubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) - static_cast<Vd>(vs2);
  }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVPsub<Vd, Vs2>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VPsub) {
  KelvinHalftypeVectorBinaryOpHelper<VPsubOp, int16_t, int8_t, int8_t, int32_t,
                                     int16_t, int16_t>("VPsubOp");
}

TEST_F(KelvinVectorInstructionsTest, VPsubu) {
  KelvinHalftypeVectorBinaryOpHelper<VPsubOp, uint16_t, uint8_t, uint8_t,
                                     uint32_t, uint16_t, uint16_t>("VPsubOp");
}

// Vector halving addition.
template <typename Vd, typename Vs1, typename Vs2>
struct VHaddOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      return static_cast<Vd>(
          (static_cast<int64_t>(vs1) + static_cast<int64_t>(vs2)) >> 1);
    }

    return static_cast<Vd>(
        (static_cast<uint64_t>(vs1) + static_cast<uint64_t>(vs2)) >> 1);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVHadd<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VHadd) {
  KelvinVectorBinaryOpHelper<VHaddOp>("VHadd");
}

TEST_F(KelvinVectorInstructionsTest, VHaddu) {
  KelvinVectorBinaryOpHelper<VHaddOp, kUnsigned>("VHaddu");
}

// Vector halving addition with rounding.
template <typename Vd, typename Vs1, typename Vs2>
struct VHaddrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      return static_cast<Vd>(
          (static_cast<int64_t>(vs1) + static_cast<int64_t>(vs2) + 1) >> 1);
    }

    return static_cast<Vd>(
        (static_cast<uint64_t>(vs1) + static_cast<uint64_t>(vs2) + 1) >> 1);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVHadd<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VHaddr) {
  KelvinVectorBinaryOpHelper<VHaddrOp>("VHaddr");
}

TEST_F(KelvinVectorInstructionsTest, VHaddur) {
  KelvinVectorBinaryOpHelper<VHaddrOp, kUnsigned>("VHaddur");
}

// Vector halving subtraction.
template <typename Vd, typename Vs1, typename Vs2>
struct VHsubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      return static_cast<Vd>(
          (static_cast<int64_t>(vs1) - static_cast<int64_t>(vs2)) >> 1);
    }

    return static_cast<Vd>(
        (static_cast<uint64_t>(vs1) - static_cast<uint64_t>(vs2)) >> 1);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVHsub<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VHsub) {
  KelvinVectorBinaryOpHelper<VHsubOp>("VHsub");
}

TEST_F(KelvinVectorInstructionsTest, VHsubu) {
  KelvinVectorBinaryOpHelper<VHsubOp, kUnsigned>("VHsubu");
}

// Vector halving subtraction with rounding.
template <typename Vd, typename Vs1, typename Vs2>
struct VHsubrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      return static_cast<Vd>(
          (static_cast<int64_t>(vs1) - static_cast<int64_t>(vs2) + 1) >> 1);
    }

    return static_cast<Vd>(
        (static_cast<uint64_t>(vs1) - static_cast<uint64_t>(vs2) + 1) >> 1);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVHsub<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VHsubr) {
  KelvinVectorBinaryOpHelper<VHsubrOp>("VHsubr");
}

TEST_F(KelvinVectorInstructionsTest, VHsubur) {
  KelvinVectorBinaryOpHelper<VHsubrOp, kUnsigned>("VHsubur");
}

// Vector bitwise and.
template <typename Vd, typename Vs1, typename Vs2>
struct VAndOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 & vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVAnd<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VAnd) {
  KelvinVectorBinaryOpHelper<VAndOp, kUnsigned>("VAnd");
}

// Vector bitwise or.
template <typename Vd, typename Vs1, typename Vs2>
struct VOrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 | vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVOr<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VOr) {
  KelvinVectorBinaryOpHelper<VOrOp, kUnsigned>("VOr");
}

// Vector bitwise xor.
template <typename Vd, typename Vs1, typename Vs2>
struct VXorOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 ^ vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVXor<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VXor) {
  KelvinVectorBinaryOpHelper<VXorOp, kUnsigned>("VXor");
}

// Vector logical shift left.
template <typename Vd, typename Vs1, typename Vs2>
struct VSllOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 << (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSll<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSll) {
  KelvinVectorBinaryOpHelper<VSllOp, kUnsigned>("VSll");
}

// Vector logical shift right.
template <typename Vd, typename Vs1, typename Vs2>
struct VSrlOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >> (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSrl<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSrl) {
  KelvinVectorBinaryOpHelper<VSrlOp, kUnsigned>("VSrl");
}

// Vector arithmetic shift right.
template <typename Vd, typename Vs1, typename Vs2>
struct VSraOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >> (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSra<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSra) {
  KelvinVectorBinaryOpHelper<VSraOp>("VSra");
}

// Vector reverse using bit ladder.
template <typename Vd, typename Vs1, typename Vs2>
struct VRevOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    Vs1 r = vs1;
    Vs2 count = vs2 & 0b11111;
    if (count & 1) r = ((r & 0x55555555) << 1) | ((r & 0xAAAAAAAA) >> 1);
    if (count & 2) r = ((r & 0x33333333) << 2) | ((r & 0xCCCCCCCC) >> 2);
    if (count & 4) r = ((r & 0x0F0F0F0F) << 4) | ((r & 0xF0F0F0F0) >> 4);
    if (sizeof(Vs1) == 1) return r;
    if (count & 8) r = ((r & 0x00FF00FF) << 8) | ((r & 0xFF00FF00) >> 8);
    if (sizeof(Vs1) == 2) return r;
    if (count & 16) r = ((r & 0x0000FFFF) << 16) | ((r & 0xFFFF0000) >> 16);
    return r;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVRev<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VRev) {
  KelvinVectorBinaryOpHelper<VRevOp, uint8_t, uint8_t, uint8_t, uint16_t,
                             uint16_t, uint16_t, uint32_t, uint32_t, uint32_t>(
      "VRevOp");
}

// Cyclic rotation right using a bit ladder.
template <typename Vd, typename Vs1, typename Vs2>
struct VRorOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    Vs1 r = vs1;
    Vd count = vs2 & static_cast<Vd>(sizeof(Vd) * 8 - 1);
    for (auto shift : {1, 2, 4, 8, 16}) {
      if (count & shift) r = (r >> shift) | (r << (sizeof(Vd) * 8 - shift));
    }
    return r;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVRor<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VRor) {
  KelvinVectorBinaryOpHelper<VRorOp, uint8_t, uint8_t, uint8_t, uint16_t,
                             uint16_t, uint16_t, uint32_t, uint32_t, uint32_t>(
      "VRorOp");
}

// Vector move pair.
template <typename T>
struct VMvpOp {
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMvp<T>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VMvp) {
  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::KelvinOp, kNonScalar, kNonStripmine),
      "VMvpVV", kNonScalar, kNonStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::KelvinOp, kNonScalar, kIsStripmine),
      "VMvpVVM", kNonScalar, kIsStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::KelvinOp, kIsScalar, kNonStripmine),
      "VMvpWVX", kIsScalar, kNonStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::KelvinOp, kIsScalar, kIsStripmine),
      "VMvpWVXM", kIsScalar, kIsStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint16_t, uint16_t, uint16_t>(
      absl::bind_front(VMvpOp<uint16_t>::KelvinOp, kIsScalar, kNonStripmine),
      "VMvpHVX", kIsScalar, kNonStripmine, VMvpOp<uint16_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint16_t, uint16_t, uint16_t>(
      absl::bind_front(VMvpOp<uint16_t>::KelvinOp, kIsScalar, kIsStripmine),
      "VMvpHVXM", kIsScalar, kIsStripmine, VMvpOp<uint16_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint8_t, uint8_t, uint8_t>(
      absl::bind_front(VMvpOp<uint8_t>::KelvinOp, kIsScalar, kNonStripmine),
      "VMvpBVX", kIsScalar, kNonStripmine, VMvpOp<uint8_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint8_t, uint8_t, uint8_t>(
      absl::bind_front(VMvpOp<uint8_t>::KelvinOp, kIsScalar, kIsStripmine),
      "VMvpBVXM", kIsScalar, kIsStripmine, VMvpOp<uint8_t>::Op, kNonHalftypeOp,
      kVmvpOp);
}

// Left/right shift with saturating shift amount and result.
template <typename Vd, typename Vs1, typename Vs2>
struct VShiftOp {
  static Vd Op(bool round, Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value == true) {
      constexpr int kMaxShiftBit = sizeof(Vd) * 8;
      int shamt = 0;
      if (sizeof(Vd) == 1) shamt = static_cast<int8_t>(vs2);
      if (sizeof(Vd) == 2) shamt = static_cast<int16_t>(vs2);
      if (sizeof(Vd) == 4) shamt = static_cast<int32_t>(vs2);
      int64_t shift = vs1;
      if (!vs1) {
        return 0;
      } else if (vs1 < 0 && shamt >= kMaxShiftBit) {
        shift = -1 + round;
      } else if (vs1 > 0 && shamt >= kMaxShiftBit) {
        shift = 0;
      } else if (shamt > 0) {
        shift =
            (static_cast<int64_t>(vs1) + (round ? (1ll << (shamt - 1)) : 0)) >>
            shamt;
      } else {  // shamt < 0
        uint32_t ushamt = static_cast<uint32_t>(
            -shamt <= kMaxShiftBit ? -shamt : kMaxShiftBit);
        shift = static_cast<int64_t>(static_cast<uint64_t>(vs1) << ushamt);
      }
      int64_t neg_max = (-1ull) << (kMaxShiftBit - 1);
      int64_t pos_max = (1ll << (kMaxShiftBit - 1)) - 1;
      bool neg_sat = vs1 < 0 && (shamt <= -kMaxShiftBit || shift < neg_max);
      bool pos_sat = vs1 > 0 && (shamt <= -kMaxShiftBit || shift > pos_max);
      if (neg_sat) return neg_max;
      if (pos_sat) return pos_max;
      return shift;
    }

    constexpr int kMaxShiftBit = sizeof(Vd) * 8;
    int shamt = 0;
    if (sizeof(Vd) == 1) shamt = static_cast<int8_t>(vs2);
    if (sizeof(Vd) == 2) shamt = static_cast<int16_t>(vs2);
    if (sizeof(Vd) == 4) shamt = static_cast<int32_t>(vs2);
    uint64_t shift = vs1;
    if (!vs1) {
      return 0;
    } else if (shamt > kMaxShiftBit) {
      shift = 0;
    } else if (shamt > 0) {
      shift =
          (static_cast<uint64_t>(vs1) + (round ? (1ull << (shamt - 1)) : 0)) >>
          shamt;
    } else {
      using UT = typename std::make_unsigned<Vd>::type;
      UT ushamt =
          static_cast<UT>(-shamt <= kMaxShiftBit ? -shamt : kMaxShiftBit);
      shift = static_cast<uint64_t>(vs1) << (ushamt);
    }
    uint64_t pos_max = (1ull << kMaxShiftBit) - 1;
    bool pos_sat =
        vs1 && (shamt < -kMaxShiftBit || shift >= (1ull << kMaxShiftBit));
    if (pos_sat) return pos_max;
    return shift;
  }

  static void KelvinOp(bool round, bool scalar, bool strip_mine,
                       Instruction *inst) {
    KelvinVShift<Vd>(round, scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VShift) {
  KelvinVectorShiftBinaryOpHelper<VShiftOp, int8_t, int16_t, int32_t, uint8_t,
                                  uint16_t, uint32_t>("VShift");
}

// Vector bitwise not.
template <typename Vd, typename Vs>
struct VNotOp {
  static Vd Op(Vs vs) { return ~vs; }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVNot<Vs>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VNot) {
  KelvinVectorUnaryOpHelper<VNotOp, int32_t, int32_t>("VNot");
}

// Count the leading bits.
template <typename Vd, typename Vs>
struct VClbOp {
  static Vd Op(Vs vs) {
    constexpr int n = sizeof(Vs) * 8;
    if (vs & (1u << (n - 1))) {
      vs = ~vs;
    }
    for (int count = 0; count < n; count++) {
      if ((vs << count) >> (n - 1)) {
        return count;
      }
    }
    return n;
  }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVClb<Vs>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VClb) {
  KelvinVectorUnaryOpHelper<VClbOp, uint8_t, uint8_t, uint16_t, uint16_t,
                            uint32_t, uint32_t>("VClb");
}

// Count the leading zeros.
template <typename Vd, typename Vs>
struct VClzOp {
  static Vd Op(Vs vs) {
    constexpr int n = sizeof(Vs) * 8;
    for (int count = 0; count < n; count++) {
      if ((vs << count) >> (n - 1)) {
        return count;
      }
    }
    return n;
  }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVClz<Vs>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VClz) {
  KelvinVectorUnaryOpHelper<VClzOp, uint8_t, uint8_t, uint16_t, uint16_t,
                            uint32_t, uint32_t>("VClz");
}

// Count the set bits.
template <typename Vd, typename Vs>
struct VCpopOp {
  static Vd Op(Vs vs) { return absl::popcount(vs); }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVCpop<Vs>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VCpop) {
  KelvinVectorUnaryOpHelper<VCpopOp, uint8_t, uint8_t, uint16_t, uint16_t,
                            uint32_t, uint32_t>("VCpop");
}

// Count the set bits.
template <typename Vd, typename Vs>
struct VMvOp {
  static Vd Op(Vs vs) { return vs; }
  static void KelvinOp(bool strip_mine, Instruction *inst) {
    KelvinVMv<Vs>(strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMv) {
  KelvinVectorUnaryOpHelper<VMvOp, int32_t, int32_t>("VMv");
}

// Arithmetic right shift without rounding and signed/unsigned saturation.
// Narrowing x2 or x4.
template <typename Vd, typename Vs1, typename Vs2>
struct VSransOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    static_assert(2 * sizeof(Vd) == sizeof(Vs1) ||
                  4 * sizeof(Vd) == sizeof(Vs1));
    constexpr int src_bits = sizeof(Vs1) * 8;
    vs2 &= (src_bits - 1);

    int64_t res = (static_cast<int64_t>(vs1)) >> vs2;

    bool neg_sat = res < std::numeric_limits<Vd>::min();
    bool pos_sat = res > std::numeric_limits<Vd>::max();
    bool zero = !vs1;
    if (neg_sat) return std::numeric_limits<Vd>::min();
    if (pos_sat) return std::numeric_limits<Vd>::max();
    if (zero) return 0;
    return res;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSrans<Vd, Vs1>(kNonRounding, scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSrans) {
  KelvinVectorBinaryOpHelper<VSransOp, int8_t, int16_t, int8_t, int16_t,
                             int32_t, int16_t, uint8_t, uint16_t, uint8_t,
                             uint16_t, uint32_t, uint16_t>("VSrans");
}

// Arithmetic right shift with rounding and signed/unsigned saturation.
// Narrowing x2 or x4.
template <typename Vd, typename Vs1, typename Vs2>
struct VSransrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    static_assert(2 * sizeof(Vd) == sizeof(Vs1) ||
                  4 * sizeof(Vd) == sizeof(Vs1));
    constexpr int src_bits = sizeof(Vs1) * 8;
    vs2 &= (src_bits - 1);

    int64_t res =
        (static_cast<int64_t>(vs1) + (vs2 ? (1ll << (vs2 - 1)) : 0)) >> vs2;

    bool neg_sat = res < std::numeric_limits<Vd>::min();
    bool pos_sat = res > std::numeric_limits<Vd>::max();
    bool zero = !vs1;
    if (neg_sat) return std::numeric_limits<Vd>::min();
    if (pos_sat) return std::numeric_limits<Vd>::max();
    if (zero) return 0;
    return res;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSrans<Vd, Vs1>(kIsRounding, scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSransr) {
  KelvinVectorBinaryOpHelper<VSransrOp, int8_t, int16_t, int8_t, int16_t,
                             int32_t, int16_t, uint8_t, uint16_t, uint8_t,
                             uint16_t, uint32_t, uint16_t>("VSransr");
}

TEST_F(KelvinVectorInstructionsTest, VSraqs) {
  KelvinVectorBinaryOpHelper<VSransOp, int8_t, int32_t, int8_t, uint8_t,
                             uint32_t, uint8_t>("VSraqs");
}

TEST_F(KelvinVectorInstructionsTest, VSraqsr) {
  KelvinVectorBinaryOpHelper<VSransrOp, int8_t, int32_t, int8_t, uint8_t,
                             uint32_t, uint8_t>("VSraqsr");
}

// Vector elements multiplication.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      return static_cast<Vd>(static_cast<int64_t>(vs1) *
                             static_cast<int64_t>(vs2));
    }

    return static_cast<Vd>(static_cast<uint64_t>(vs1) *
                           static_cast<uint64_t>(vs2));
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMul<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VMul) {
  KelvinVectorBinaryOpHelper<VMulOp>("VMul");
}

// Vector elements multiplication with saturation.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulsOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    if (std::is_signed<Vd>::value) {
      int64_t m = static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
      m = std::max(
          static_cast<int64_t>(std::numeric_limits<Vd>::min()),
          std::min(static_cast<int64_t>(std::numeric_limits<Vd>::max()), m));
      return m;
    }

    uint64_t m = static_cast<uint64_t>(vs1) * static_cast<uint64_t>(vs2);
    m = std::min(static_cast<uint64_t>(std::numeric_limits<Vd>::max()), m);
    return m;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMuls<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMuls) {
  KelvinVectorBinaryOpHelper<VMulsOp>("VMuls");
}

TEST_F(KelvinVectorInstructionsTest, VMulsu) {
  KelvinVectorBinaryOpHelper<VMulsOp, kUnsigned>("VMulsu");
}

// Vector elements multiplication with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) * static_cast<Vd>(vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMulw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMulw) {
  KelvinVectorBinaryOpHelper<VMulwOp, int16_t, int8_t, int8_t, int32_t, int16_t,
                             int16_t>("VMulwOp");
}

TEST_F(KelvinVectorInstructionsTest, VMulwu) {
  KelvinVectorBinaryOpHelper<VMulwOp, uint16_t, uint8_t, uint8_t, uint32_t,
                             uint16_t, uint16_t>("VMulwuOp");
}

// Vector elements multiplication with widening. Returns high half.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulhOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    constexpr int n = sizeof(Vd) * 8;
    if (std::is_signed<Vs1>::value) {
      int64_t result = static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
      return static_cast<uint64_t>(result) >> n;
    }

    uint64_t result = static_cast<uint64_t>(vs1) * static_cast<uint64_t>(vs2);
    return result >> n;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMulh<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMulh) {
  KelvinVectorBinaryOpHelper<VMulhOp>("VMulh");
}

TEST_F(KelvinVectorInstructionsTest, VMulhu) {
  KelvinVectorBinaryOpHelper<VMulhOp, kUnsigned>("VMulhu");
}

// Vector elements multiplication with rounding and widening. Returns high
// half.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulhrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    constexpr int n = sizeof(Vd) * 8;
    if (std::is_signed<Vs1>::value) {
      int64_t result = static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
      result += 1ll << (n - 1);
      return static_cast<uint64_t>(result) >> n;
    }

    uint64_t result = static_cast<uint64_t>(vs1) * static_cast<uint64_t>(vs2);
    result += 1ull << (n - 1);
    return result >> n;
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMulh<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMulhr) {
  KelvinVectorBinaryOpHelper<VMulhrOp>("VMulhr");
}

TEST_F(KelvinVectorInstructionsTest, VMulhur) {
  KelvinVectorBinaryOpHelper<VMulhrOp, kUnsigned>("VMulhur");
}

// Saturating signed doubling multiply returning high half with optional
// rounding.
template <typename T>
T KelvinVDmulhHelper(bool round, bool round_neg, T vs1, T vs2) {
  constexpr int n = sizeof(T) * 8;
  int64_t result = static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
  if (round) {
    int64_t rnd = 0x40000000ll >> (32 - n);
    if (result < 0 && round_neg) {
      rnd = (-0x40000000ll) >> (32 - n);
    }
    result += rnd;
  }
  result >>= (n - 1);
  if (vs1 == std::numeric_limits<T>::min() &&
      vs2 == std::numeric_limits<T>::min()) {
    result = std::numeric_limits<T>::max();
  }
  return result;
}

template <typename Vd, typename Vs1, typename Vs2>
struct VDmulhOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return KelvinVDmulhHelper<Vd>(kNonRounding, false /* round_neg*/, vs1, vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVDmulh<Vd>(scalar, strip_mine, kNonRounding, false /* round_neg*/,
                     inst);
  }
};

template <typename Vd, typename Vs1, typename Vs2>
struct VDmulhrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return KelvinVDmulhHelper<Vd>(kIsRounding, false /* round_neg*/, vs1, vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVDmulh<Vd>(scalar, strip_mine, kIsRounding, false /* round_neg*/,
                     inst);
  }
};

template <typename Vd, typename Vs1, typename Vs2>
struct VDmulhrnOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return KelvinVDmulhHelper<Vd>(kIsRounding, true /* round_neg*/, vs1, vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVDmulh<Vd>(scalar, strip_mine, kIsRounding, true /* round_neg*/,
                     inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VDmulh) {
  KelvinVectorBinaryOpHelper<VDmulhOp>("VDmulh");
}

TEST_F(KelvinVectorInstructionsTest, VDmulhr) {
  KelvinVectorBinaryOpHelper<VDmulhrOp>("VDmulhr");
}

TEST_F(KelvinVectorInstructionsTest, VDmulhrn) {
  KelvinVectorBinaryOpHelper<VDmulhrnOp>("VDmulhrn");
}

// Multiply accumulate.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaccOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) {
    return static_cast<int64_t>(vd) +
           static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMacc<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMacc) {
  KelvinVectorBinaryOpHelper<VMaccOp>("VMacc");
}

// Multiply add.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaddOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) {
    return static_cast<int64_t>(vs1) +
           static_cast<int64_t>(vd) * static_cast<int64_t>(vs2);
  }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVMadd<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VMadd) {
  KelvinVectorBinaryOpHelper<VMaddOp>("VMadd");
}

// Slide next register by index.
template <typename T>
static std::pair<T, T> SlidenArgsGetter(
    bool horizontal, int index, int num_ops, int op_num, int dest_reg_sub_index,
    int element_index, int vd_size, bool widen_dst, int src1_widen_factor,
    int vs1_size, const std::vector<T> &vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T> &vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  assert(!s2_scalar && !halftype_op && !vmvp_op && dest_reg_sub_index == 0);

  using Interleave = struct {
    int register_num;
    int source_arg;
  };
  const Interleave interleave_start[2][4] = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                                             {{0, 0}, {1, 0}, {2, 0}, {3, 0}}};
  const Interleave interleave_end[2][4] = {{{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                                           {{1, 0}, {2, 0}, {3, 0}, {0, 1}}};

  T arg1;
  if (element_index + index < vd_size) {
    auto src_element_index =
        interleave_start[horizontal][op_num].register_num * vd_size +
        element_index + index;
    arg1 = interleave_start[horizontal][op_num].source_arg
               ? vs2_value[src_element_index]
               : vs1_value[src_element_index];
  } else {
    auto src_element_index =
        interleave_end[horizontal][op_num].register_num * vd_size +
        element_index + index - vd_size;

    arg1 = interleave_end[horizontal][op_num].source_arg
               ? vs2_value[src_element_index]
               : vs1_value[src_element_index];
  }

  return {arg1, 0};
}

// Slide next register horizontally by index.
template <typename T>
struct VSlidehnOp {
  static constexpr auto kArgsGetter = SlidenArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(int index, bool strip_mine, Instruction *inst) {
    KelvinVSlidehn<T>(index, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSlidehn) {
  KelvinSlideOpHelper<VSlidehnOp, int8_t, int16_t, int32_t>(
      "VSlidehnOp", kHorizontal, true /* strip_mine */);
}

template <typename T>
struct VSlidevnOp {
  static constexpr auto kArgsGetter = SlidenArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(int index, bool strip_mine, Instruction *inst) {
    KelvinVSlidevn<T>(index, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSliden) {
  KelvinSlideOpHelper<VSlidevnOp, int8_t, int16_t, int32_t>(
      "VSlidenOp", kVertical, false /* strip_mine */);
}

TEST_F(KelvinVectorInstructionsTest, VSlidevn) {
  KelvinSlideOpHelper<VSlidevnOp, int8_t, int16_t, int32_t>(
      "VSlidevnOp", kVertical, true /* strip_mine */);
}

// Slide previous register by index.
template <typename T>
static std::pair<T, T> SlidepArgsGetter(
    bool horizontal, int index, int num_ops, int op_num, int dest_reg_sub_index,
    int element_index, int vd_size, bool widen_dst, int src1_widen_factor,
    int vs1_size, const std::vector<T> &vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T> &vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  assert(!s2_scalar && !halftype_op && !vmvp_op && dest_reg_sub_index == 0);

  using Interleave = struct {
    int register_num;
    int source_arg;
  };
  const Interleave interleave_start[2][4] = {{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                                             {{3, 0}, {0, 1}, {1, 1}, {2, 1}}};
  const Interleave interleave_end[2][4] = {{{0, 1}, {1, 1}, {2, 1}, {3, 1}},
                                           {{0, 1}, {1, 1}, {2, 1}, {3, 1}}};

  T arg1;
  if (element_index < index) {
    auto src_element_index =
        interleave_start[horizontal][op_num].register_num * vd_size +
        element_index - index + vd_size;
    arg1 = interleave_start[horizontal][op_num].source_arg
               ? vs2_value[src_element_index]
               : vs1_value[src_element_index];
  } else {
    auto src_element_index =
        interleave_end[horizontal][op_num].register_num * vd_size +
        element_index - index;

    arg1 = interleave_end[horizontal][op_num].source_arg
               ? vs2_value[src_element_index]
               : vs1_value[src_element_index];
  }

  return {arg1, 0};
}

// Slide previous register horizontally by index.
template <typename T>
struct VSlidehpOp {
  static constexpr auto kArgsGetter = SlidepArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(int index, bool strip_mine, Instruction *inst) {
    KelvinVSlidehp<T>(index, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSlidehp) {
  KelvinSlideOpHelper<VSlidehpOp, int8_t, int16_t, int32_t>(
      "VSlidehpOp", kHorizontal, true /* strip_mine */);
}

template <typename T>
struct VSlidevpOp {
  static constexpr auto kArgsGetter = SlidepArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(int index, bool strip_mine, Instruction *inst) {
    KelvinVSlidevp<T>(index, strip_mine, inst);
  }
};

TEST_F(KelvinVectorInstructionsTest, VSlidep) {
  KelvinSlideOpHelper<VSlidevpOp, int8_t, int16_t, int32_t>(
      "VSlidepOp", kVertical, false /* strip_mine */);
}

TEST_F(KelvinVectorInstructionsTest, VSlidevp) {
  KelvinSlideOpHelper<VSlidevpOp, int8_t, int16_t, int32_t>(
      "VSlidevpOp", kVertical, true /* strip_mine */);
}

// Select lanes from two operands with vector selection boolean.
template <typename Vd, typename Vs1, typename Vs2>
struct VSelOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) { return vs1 & 1 ? vd : vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVSel<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VSel) {
  KelvinVectorBinaryOpHelper<VSelOp>("VSel");
}

// Select even/odd elements of concatenated registers.
template <typename T>
static std::pair<T, T> EvnOddOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T> &vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T> &vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  const int combined_element_index = (op_num * vs1_size + element_index) * 2;
  const int elts_per_src = num_ops * vs1_size;
  T even, odd;

  if (combined_element_index < elts_per_src) {
    even = vs1_value[combined_element_index];
    odd = vs1_value[combined_element_index + 1];
  } else {
    even = s2_scalar ? rs2_value
                     : vs2_value[combined_element_index - elts_per_src];
    odd = s2_scalar ? rs2_value
                    : vs2_value[combined_element_index - elts_per_src + 1];
  }

  return {dest_reg_sub_index == 0 ? even : odd, odd};
}

template <typename T>
struct VEvnOp {
  static constexpr auto kArgsGetter = EvnOddOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVEvn<T>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VEvn) {
  KelvinShuffleOpHelper<VEvnOp, int8_t, int16_t, int32_t>("VEvn");
}

template <typename T>
struct VOddOp {
  static constexpr auto kArgsGetter = EvnOddOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs2; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVOdd<T>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VOdd) {
  KelvinShuffleOpHelper<VOddOp, int8_t, int16_t, int32_t>("VOdd");
}

template <typename T>
struct VEvnoddOp {
  static constexpr auto kArgsGetter = EvnOddOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVEvnodd<T>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VEvnodd) {
  KelvinShuffleOpHelper<VEvnoddOp, int8_t, int16_t, int32_t>("VEvnodd",
                                                             kWidenDst);
}

// Select even/odd elements of concatenated registers.
template <typename T>
static std::pair<T, T> ZipOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T> &vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T> &vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  auto src_index =
      op_num * vs1_size + element_index / 2 + dest_reg_sub_index * vs1_size / 2;

  T arg1;
  if (element_index & 1) {
    arg1 = s2_scalar ? rs2_value : vs2_value[src_index];
  } else {
    arg1 = vs1_value[src_index];
  }
  return {arg1, 0};
}

template <typename T>
struct VZipOp {
  static constexpr auto kArgsGetter = ZipOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void KelvinOp(bool scalar, bool strip_mine, Instruction *inst) {
    KelvinVZip<T>(scalar, strip_mine, inst);
  }
};
TEST_F(KelvinVectorInstructionsTest, VZip) {
  KelvinShuffleOpHelper<VZipOp, int8_t, int16_t, int32_t>("VZip", kWidenDst);
}

}  // namespace
