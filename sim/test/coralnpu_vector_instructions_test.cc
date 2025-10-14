// Copyright 2023 Google LLC
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

#include "sim/coralnpu_vector_instructions.h"

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "sim/test/coralnpu_vector_instructions_test_base.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/functional/bind_front.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mpact/sim/generic/instruction.h"

// This file contains the tests for testing coralnpu vector binary instructions.

namespace {

using mpact::sim::generic::Instruction;

// Semantic functions.
using coralnpu::sim::CoralNPUVAbsd;
using coralnpu::sim::CoralNPUVAcc;
using coralnpu::sim::CoralNPUVAdd;
using coralnpu::sim::CoralNPUVAdd3;
using coralnpu::sim::CoralNPUVAdds;
using coralnpu::sim::CoralNPUVAddsu;
using coralnpu::sim::CoralNPUVAddw;
using coralnpu::sim::CoralNPUVAnd;
using coralnpu::sim::CoralNPUVClb;
using coralnpu::sim::CoralNPUVClz;
using coralnpu::sim::CoralNPUVCpop;
using coralnpu::sim::CoralNPUVDmulh;
using coralnpu::sim::CoralNPUVEq;
using coralnpu::sim::CoralNPUVEvn;
using coralnpu::sim::CoralNPUVEvnodd;
using coralnpu::sim::CoralNPUVGe;
using coralnpu::sim::CoralNPUVGt;
using coralnpu::sim::CoralNPUVHadd;
using coralnpu::sim::CoralNPUVHsub;
using coralnpu::sim::CoralNPUVLe;
using coralnpu::sim::CoralNPUVLt;
using coralnpu::sim::CoralNPUVMacc;
using coralnpu::sim::CoralNPUVMadd;
using coralnpu::sim::CoralNPUVMax;
using coralnpu::sim::CoralNPUVMin;
using coralnpu::sim::CoralNPUVMul;
using coralnpu::sim::CoralNPUVMulh;
using coralnpu::sim::CoralNPUVMuls;
using coralnpu::sim::CoralNPUVMulw;
using coralnpu::sim::CoralNPUVMv;
using coralnpu::sim::CoralNPUVMvp;
using coralnpu::sim::CoralNPUVNe;
using coralnpu::sim::CoralNPUVNot;
using coralnpu::sim::CoralNPUVOdd;
using coralnpu::sim::CoralNPUVOr;
using coralnpu::sim::CoralNPUVPadd;
using coralnpu::sim::CoralNPUVPsub;
using coralnpu::sim::CoralNPUVRev;
using coralnpu::sim::CoralNPUVRor;
using coralnpu::sim::CoralNPUVRSub;
using coralnpu::sim::CoralNPUVSel;
using coralnpu::sim::CoralNPUVShift;
using coralnpu::sim::CoralNPUVSlidehn;
using coralnpu::sim::CoralNPUVSlidehp;
using coralnpu::sim::CoralNPUVSlidevn;
using coralnpu::sim::CoralNPUVSlidevp;
using coralnpu::sim::CoralNPUVSll;
using coralnpu::sim::CoralNPUVSra;
using coralnpu::sim::CoralNPUVSrans;
using coralnpu::sim::CoralNPUVSrl;
using coralnpu::sim::CoralNPUVSub;
using coralnpu::sim::CoralNPUVSubs;
using coralnpu::sim::CoralNPUVSubsu;
using coralnpu::sim::CoralNPUVSubw;
using coralnpu::sim::CoralNPUVXor;
using coralnpu::sim::CoralNPUVZip;

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

class CoralNPUVectorInstructionsTest
    : public coralnpu::sim::test::CoralNPUVectorInstructionsTestBase {
 public:
  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2>
  void CoralNPUVectorBinaryOpHelper(absl::string_view name) {
    const auto name_with_type =
        absl::StrCat(name, CoralNPUTestTypeSuffix<TD>());

    // Test [VV, VX].{M} variants
    for (auto scalar : {kNonScalar, kIsScalar}) {
      for (auto stripmine : {kNonStripmine, kIsStripmine}) {
        auto op_name = absl::StrCat(name_with_type, "V", scalar ? "X" : "V",
                                    stripmine ? "M" : "");
        BinaryOpTestHelper<TD, TS1, TS2>(
            absl::bind_front(F<TD, TS1, TS2>::CoralNPUOp, scalar, stripmine),
            op_name, scalar, stripmine, F<TD, TS1, TS2>::Op);
      }
    }
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2, typename TNext1, typename... TNext>
  void CoralNPUVectorBinaryOpHelper(absl::string_view name) {
    CoralNPUVectorBinaryOpHelper<F, TD, TS1, TS2>(name);
    CoralNPUVectorBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename, typename> class F,
            bool is_signed = true>
  void CoralNPUVectorBinaryOpHelper(absl::string_view name) {
    if (is_signed) {
      CoralNPUVectorBinaryOpHelper<F, int8_t, int8_t, int8_t, int16_t, int16_t,
                                   int16_t, int32_t, int32_t, int32_t>(name);
    } else {
      CoralNPUVectorBinaryOpHelper<F, uint8_t, uint8_t, uint8_t, uint16_t,
                                   uint16_t, uint16_t, uint32_t, uint32_t,
                                   uint32_t>(name);
    }
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2>
  void CoralNPUHalftypeVectorBinaryOpHelper(absl::string_view name) {
    const auto name_with_type =
        absl::StrCat(name, CoralNPUTestTypeSuffix<TD>());

    // Vector OP single vector.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::CoralNPUOp, kNonStripmine),
        absl::StrCat(name_with_type, "V"), kNonScalar, kNonStripmine,
        F<TD, TS1, TS2>::Op, kHalftypeOp);

    // Vector OP single vector stripmined.
    BinaryOpTestHelper<TD, TS1, TS2>(
        absl::bind_front(F<TD, TS1, TS2>::CoralNPUOp, kIsStripmine),
        absl::StrCat(name_with_type, "VM"), kNonScalar, kIsStripmine,
        F<TD, TS1, TS2>::Op, kHalftypeOp);
  }

  template <template <typename, typename, typename> class F, typename TD,
            typename TS1, typename TS2, typename TNext1, typename... TNext>
  void CoralNPUHalftypeVectorBinaryOpHelper(absl::string_view name) {
    CoralNPUHalftypeVectorBinaryOpHelper<F, TD, TS1, TS2>(name);
    CoralNPUHalftypeVectorBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename, typename> class F, typename T>
  void CoralNPUVectorShiftBinaryOpHelper(absl::string_view name) {
    const auto name_with_type = absl::StrCat(name, CoralNPUTestTypeSuffix<T>());

    // Test {R}.[VV, VX].{M} variants.
    for (auto rounding : {kNonRounding, kIsRounding}) {
      for (auto scalar : {kNonScalar, kIsScalar}) {
        for (auto stripmine : {kNonStripmine, kIsStripmine}) {
          auto op_name = absl::StrCat(name_with_type, rounding ? "R" : "", "V",
                                      scalar ? "X" : "V", stripmine ? "M" : "");
          BinaryOpTestHelper<T, T, T>(
              absl::bind_front(F<T, T, T>::CoralNPUOp, rounding, scalar,
                               stripmine),
              op_name, scalar, stripmine,
              absl::bind_front(F<T, T, T>::Op, rounding));
        }
      }
    }
  }

  template <template <typename, typename, typename> class F, typename T,
            typename TNext1, typename... TNext>
  void CoralNPUVectorShiftBinaryOpHelper(absl::string_view name) {
    CoralNPUVectorShiftBinaryOpHelper<F, T>(name);
    CoralNPUVectorShiftBinaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename, typename> class F, typename TD, typename TS>
  void CoralNPUVectorUnaryOpHelper(absl::string_view name) {
    const auto name_with_type =
        absl::StrCat(name, CoralNPUTestTypeSuffix<TD>());

    // Vector OP single vector.
    UnaryOpTestHelper<TD, TS>(
        absl::bind_front(F<TD, TS>::CoralNPUOp, kNonStripmine),
        absl::StrCat(name_with_type, "V"), kNonStripmine, F<TD, TS>::Op);

    // Vector OP single vector stripmined.
    UnaryOpTestHelper<TD, TS>(
        absl::bind_front(F<TD, TS>::CoralNPUOp, kIsStripmine),
        absl::StrCat(name_with_type, "VM"), kIsStripmine, F<TD, TS>::Op);
  }

  template <template <typename, typename> class F, typename TD, typename TS,
            typename TNext1, typename... TNext>
  void CoralNPUVectorUnaryOpHelper(absl::string_view name) {
    CoralNPUVectorUnaryOpHelper<F, TD, TS>(name);
    CoralNPUVectorUnaryOpHelper<F, TNext1, TNext...>(name);
  }

  template <template <typename> class F, typename T>
  void CoralNPUSlideOpHelper(absl::string_view name, bool horizontal,
                             bool strip_mine) {
    const auto name_with_type = absl::StrCat(name, CoralNPUTestTypeSuffix<T>());

    for (int i = 1; i < 5; ++i) {
      BinaryOpTestHelper<T, T, T>(
          absl::bind_front(F<T>::CoralNPUOp, i, strip_mine),
          absl::StrCat(name_with_type, i, "V", strip_mine ? "M" : ""),
          kNonScalar, strip_mine, F<T>::Op,
          absl::bind_front(F<T>::kArgsGetter, horizontal, i), kNonHalftypeOp,
          kNonVmvpOp, kNonWidenDst);
    }
  }

  template <template <typename> class F, typename T, typename TNext1,
            typename... TNext>
  void CoralNPUSlideOpHelper(absl::string_view name, bool horizontal,
                             bool strip_mine) {
    CoralNPUSlideOpHelper<F, T>(name, horizontal, strip_mine);
    CoralNPUSlideOpHelper<F, TNext1, TNext...>(name, horizontal, strip_mine);
  }

  template <template <typename> class F, typename T>
  void CoralNPUShuffleOpHelper(absl::string_view name, bool widen_dst) {
    const auto name_with_type = absl::StrCat(name, CoralNPUTestTypeSuffix<T>());

    // Test [VV, VX].{M} variants.
    for (auto scalar : {kNonScalar, kIsScalar}) {
      for (auto stripmine : {kNonStripmine, kIsStripmine}) {
        auto op_name = absl::StrCat(name_with_type, "V", scalar ? "X" : "V",
                                    stripmine ? "M" : "");
        BinaryOpTestHelper<T, T, T>(
            absl::bind_front(F<T>::CoralNPUOp, scalar, stripmine), op_name,
            scalar, stripmine, F<T>::Op, F<T>::kArgsGetter, kNonHalftypeOp,
            kNonVmvpOp, widen_dst);
      }
    }
  }

  template <template <typename> class F, typename T, typename TNext1,
            typename... TNext>
  void CoralNPUShuffleOpHelper(absl::string_view name, bool widen_dst = false) {
    CoralNPUShuffleOpHelper<F, T>(name, widen_dst);
    CoralNPUShuffleOpHelper<F, TNext1, TNext...>(name, widen_dst);
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAdd<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VAdd) {
  CoralNPUVectorBinaryOpHelper<VAddOp>("VAdd");
}

// Vector subtract.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs1_ext - vs2_ext);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSub<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VSub) {
  CoralNPUVectorBinaryOpHelper<VSubOp>("VSub");
}

// Vector reverse subtract.
template <typename Vd, typename Vs1, typename Vs2>
struct VRSubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    int64_t vs1_ext = static_cast<int64_t>(vs1);
    int64_t vs2_ext = static_cast<int64_t>(vs2);
    return static_cast<Vd>(vs2_ext - vs1_ext);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVRSub<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VRsub) {
  CoralNPUVectorBinaryOpHelper<VRSubOp>("VRsub");
}

// Vector equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VEqOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 == vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVEq<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VEq) {
  CoralNPUVectorBinaryOpHelper<VEqOp>("VEq");
}

// Vector not equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VNeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 != vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVNe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VNe) {
  CoralNPUVectorBinaryOpHelper<VNeOp>("VNe");
}

// Vector less than.
template <typename Vd, typename Vs1, typename Vs2>
struct VLtOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 < vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVLt<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VLt) {
  CoralNPUVectorBinaryOpHelper<VLtOp>("VLt");
}

// Vector less than unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VLtu) {
  CoralNPUVectorBinaryOpHelper<VLtOp, kUnsigned>("VLtu");
}

// Vector less than or equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VLeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 <= vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVLe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VLe) {
  CoralNPUVectorBinaryOpHelper<VLeOp>("VLe");
}

// Vector less than or equal unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VLeu) {
  CoralNPUVectorBinaryOpHelper<VLeOp, kUnsigned>("VLeu");
}

// Vector greater than.
template <typename Vd, typename Vs1, typename Vs2>
struct VGtOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 > vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVGt<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VGt) {
  CoralNPUVectorBinaryOpHelper<VGtOp>("VGt");
}

// Vector greater than unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VGtu) {
  CoralNPUVectorBinaryOpHelper<VGtOp, kUnsigned>("VGtu");
}

// Vector greater than or equal.
template <typename Vd, typename Vs1, typename Vs2>
struct VGeOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >= vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVGe<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VGe) {
  CoralNPUVectorBinaryOpHelper<VGeOp>("VGe");
}

// Vector greater than or equal unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VGeu) {
  CoralNPUVectorBinaryOpHelper<VGeOp, kUnsigned>("VGeu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAbsd<Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAbsd) {
  CoralNPUVectorBinaryOpHelper<VAbsdOp, uint8_t, int8_t, int8_t, uint16_t,
                               int16_t, int16_t, uint32_t, int32_t, int32_t>(
      "VAbsd");
}

TEST_F(CoralNPUVectorInstructionsTest, VAbsdu) {
  CoralNPUVectorBinaryOpHelper<VAbsdOp, kUnsigned>("VAbsdu");
}

// Vector max.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaxOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return std::max(vs1, vs2); }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMax<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VMax) {
  CoralNPUVectorBinaryOpHelper<VMaxOp>("VMax");
}

// Vector max unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VMaxu) {
  CoralNPUVectorBinaryOpHelper<VMaxOp, kUnsigned>("VMaxu");
}

// Vector min.
template <typename Vd, typename Vs1, typename Vs2>
struct VMinOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return std::min(vs1, vs2); }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMin<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VMin) {
  CoralNPUVectorBinaryOpHelper<VMinOp>("VMin");
}

// Vector min unsigned.
TEST_F(CoralNPUVectorInstructionsTest, VMinu) {
  CoralNPUVectorBinaryOpHelper<VMinOp, kUnsigned>("VMinu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAdd3<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VAdd3) {
  CoralNPUVectorBinaryOpHelper<VAdd3Op>("VAdd3");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAdds<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAdds) {
  CoralNPUVectorBinaryOpHelper<VAddsOp>("VAdds");
}

// Vector saturated unsigned add.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddsuOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    uint64_t sum = static_cast<uint64_t>(vs1) + static_cast<uint64_t>(vs2);
    return std::min<uint64_t>(std::numeric_limits<Vd>::max(), sum);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAddsu<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAddsu) {
  CoralNPUVectorBinaryOpHelper<VAddsuOp, kUnsigned>("VAddsu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSubs<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSubs) {
  CoralNPUVectorBinaryOpHelper<VSubsOp>("VSubs");
}

// Vector saturated unsigned sub.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubsuOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 < vs2 ? 0 : vs1 - vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSubsu<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSubsu) {
  CoralNPUVectorBinaryOpHelper<VSubsuOp, kUnsigned>("VSubsu");
}

// Vector addition with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VAddwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) + static_cast<Vd>(vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAddw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAddw) {
  CoralNPUVectorBinaryOpHelper<VAddwOp, int16_t, int8_t, int8_t, int32_t,
                               int16_t, int16_t>("VAddwOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VAddwu) {
  CoralNPUVectorBinaryOpHelper<VAddwOp, uint16_t, uint8_t, uint8_t, uint32_t,
                               uint16_t, uint16_t>("VAddwuOp");
}

// Vector subtraction with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VSubwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) - static_cast<Vd>(vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSubw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSubw) {
  CoralNPUVectorBinaryOpHelper<VSubwOp, int16_t, int8_t, int8_t, int32_t,
                               int16_t, int16_t>("VSubwOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VSubwu) {
  CoralNPUVectorBinaryOpHelper<VSubwOp, uint16_t, uint8_t, uint8_t, uint32_t,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAcc<Vd, Vs2>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAcc) {
  CoralNPUVectorBinaryOpHelper<VAccOp, int16_t, int16_t, int8_t, int32_t,
                               int32_t, int16_t>("VAccOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VAccu) {
  CoralNPUVectorBinaryOpHelper<VAccOp, uint16_t, uint16_t, uint8_t, uint32_t,
                               uint32_t, uint16_t>("VAccuOp");
}

// Selects pairs from register
template <typename T>
static std::pair<T, T> PairwiseOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T>& vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T>& vs2_value, T rs2_value, bool halftype_op,
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
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVPadd<Vd, Vs2>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VPadd) {
  CoralNPUHalftypeVectorBinaryOpHelper<VPaddOp, int16_t, int8_t, int8_t,
                                       int32_t, int16_t, int16_t>("VPaddOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VPaddu) {
  CoralNPUHalftypeVectorBinaryOpHelper<VPaddOp, uint16_t, uint8_t, uint8_t,
                                       uint32_t, uint16_t, uint16_t>("VPaddOp");
}

// Vector packed sub
template <typename Vd, typename Vs1, typename Vs2>
struct VPsubOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) - static_cast<Vd>(vs2);
  }
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVPsub<Vd, Vs2>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VPsub) {
  CoralNPUHalftypeVectorBinaryOpHelper<VPsubOp, int16_t, int8_t, int8_t,
                                       int32_t, int16_t, int16_t>("VPsubOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VPsubu) {
  CoralNPUHalftypeVectorBinaryOpHelper<VPsubOp, uint16_t, uint8_t, uint8_t,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVHadd<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VHadd) {
  CoralNPUVectorBinaryOpHelper<VHaddOp>("VHadd");
}

TEST_F(CoralNPUVectorInstructionsTest, VHaddu) {
  CoralNPUVectorBinaryOpHelper<VHaddOp, kUnsigned>("VHaddu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVHadd<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VHaddr) {
  CoralNPUVectorBinaryOpHelper<VHaddrOp>("VHaddr");
}

TEST_F(CoralNPUVectorInstructionsTest, VHaddur) {
  CoralNPUVectorBinaryOpHelper<VHaddrOp, kUnsigned>("VHaddur");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVHsub<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VHsub) {
  CoralNPUVectorBinaryOpHelper<VHsubOp>("VHsub");
}

TEST_F(CoralNPUVectorInstructionsTest, VHsubu) {
  CoralNPUVectorBinaryOpHelper<VHsubOp, kUnsigned>("VHsubu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVHsub<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VHsubr) {
  CoralNPUVectorBinaryOpHelper<VHsubrOp>("VHsubr");
}

TEST_F(CoralNPUVectorInstructionsTest, VHsubur) {
  CoralNPUVectorBinaryOpHelper<VHsubrOp, kUnsigned>("VHsubur");
}

// Vector bitwise and.
template <typename Vd, typename Vs1, typename Vs2>
struct VAndOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 & vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVAnd<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VAnd) {
  CoralNPUVectorBinaryOpHelper<VAndOp, kUnsigned>("VAnd");
}

// Vector bitwise or.
template <typename Vd, typename Vs1, typename Vs2>
struct VOrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 | vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVOr<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VOr) {
  CoralNPUVectorBinaryOpHelper<VOrOp, kUnsigned>("VOr");
}

// Vector bitwise xor.
template <typename Vd, typename Vs1, typename Vs2>
struct VXorOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 ^ vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVXor<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VXor) {
  CoralNPUVectorBinaryOpHelper<VXorOp, kUnsigned>("VXor");
}

// Vector logical shift left.
template <typename Vd, typename Vs1, typename Vs2>
struct VSllOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 << (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSll<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSll) {
  CoralNPUVectorBinaryOpHelper<VSllOp, kUnsigned>("VSll");
}

// Vector logical shift right.
template <typename Vd, typename Vs1, typename Vs2>
struct VSrlOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >> (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSrl<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSrl) {
  CoralNPUVectorBinaryOpHelper<VSrlOp, kUnsigned>("VSrl");
}

// Vector arithmetic shift right.
template <typename Vd, typename Vs1, typename Vs2>
struct VSraOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) { return vs1 >> (vs2 & (sizeof(Vd) * 8 - 1)); }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSra<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSra) {
  CoralNPUVectorBinaryOpHelper<VSraOp>("VSra");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVRev<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VRev) {
  CoralNPUVectorBinaryOpHelper<VRevOp, uint8_t, uint8_t, uint8_t, uint16_t,
                               uint16_t, uint16_t, uint32_t, uint32_t,
                               uint32_t>("VRevOp");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVRor<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VRor) {
  CoralNPUVectorBinaryOpHelper<VRorOp, uint8_t, uint8_t, uint8_t, uint16_t,
                               uint16_t, uint16_t, uint32_t, uint32_t,
                               uint32_t>("VRorOp");
}

// Vector move pair.
template <typename T>
struct VMvpOp {
  static T Op(T vs1, T vs2) { return vs1; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMvp<T>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VMvp) {
  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::CoralNPUOp, kNonScalar, kNonStripmine),
      "VMvpVV", kNonScalar, kNonStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::CoralNPUOp, kNonScalar, kIsStripmine),
      "VMvpVVM", kNonScalar, kIsStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::CoralNPUOp, kIsScalar, kNonStripmine),
      "VMvpWVX", kIsScalar, kNonStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint32_t, uint32_t, uint32_t>(
      absl::bind_front(VMvpOp<uint32_t>::CoralNPUOp, kIsScalar, kIsStripmine),
      "VMvpWVXM", kIsScalar, kIsStripmine, VMvpOp<uint32_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint16_t, uint16_t, uint16_t>(
      absl::bind_front(VMvpOp<uint16_t>::CoralNPUOp, kIsScalar, kNonStripmine),
      "VMvpHVX", kIsScalar, kNonStripmine, VMvpOp<uint16_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint16_t, uint16_t, uint16_t>(
      absl::bind_front(VMvpOp<uint16_t>::CoralNPUOp, kIsScalar, kIsStripmine),
      "VMvpHVXM", kIsScalar, kIsStripmine, VMvpOp<uint16_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint8_t, uint8_t, uint8_t>(
      absl::bind_front(VMvpOp<uint8_t>::CoralNPUOp, kIsScalar, kNonStripmine),
      "VMvpBVX", kIsScalar, kNonStripmine, VMvpOp<uint8_t>::Op, kNonHalftypeOp,
      kVmvpOp);

  BinaryOpTestHelper<uint8_t, uint8_t, uint8_t>(
      absl::bind_front(VMvpOp<uint8_t>::CoralNPUOp, kIsScalar, kIsStripmine),
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

  static void CoralNPUOp(bool round, bool scalar, bool strip_mine,
                         Instruction* inst) {
    CoralNPUVShift<Vd>(round, scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VShift) {
  CoralNPUVectorShiftBinaryOpHelper<VShiftOp, int8_t, int16_t, int32_t, uint8_t,
                                    uint16_t, uint32_t>("VShift");
}

// Vector bitwise not.
template <typename Vd, typename Vs>
struct VNotOp {
  static Vd Op(Vs vs) { return ~vs; }
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVNot<Vs>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VNot) {
  CoralNPUVectorUnaryOpHelper<VNotOp, int32_t, int32_t>("VNot");
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
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVClb<Vs>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VClb) {
  CoralNPUVectorUnaryOpHelper<VClbOp, uint8_t, uint8_t, uint16_t, uint16_t,
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
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVClz<Vs>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VClz) {
  CoralNPUVectorUnaryOpHelper<VClzOp, uint8_t, uint8_t, uint16_t, uint16_t,
                              uint32_t, uint32_t>("VClz");
}

// Count the set bits.
template <typename Vd, typename Vs>
struct VCpopOp {
  static Vd Op(Vs vs) { return absl::popcount(vs); }
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVCpop<Vs>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VCpop) {
  CoralNPUVectorUnaryOpHelper<VCpopOp, uint8_t, uint8_t, uint16_t, uint16_t,
                              uint32_t, uint32_t>("VCpop");
}

// Count the set bits.
template <typename Vd, typename Vs>
struct VMvOp {
  static Vd Op(Vs vs) { return vs; }
  static void CoralNPUOp(bool strip_mine, Instruction* inst) {
    CoralNPUVMv<Vs>(strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMv) {
  CoralNPUVectorUnaryOpHelper<VMvOp, int32_t, int32_t>("VMv");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSrans<Vd, Vs1>(kNonRounding, scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSrans) {
  CoralNPUVectorBinaryOpHelper<VSransOp, int8_t, int16_t, int8_t, int16_t,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSrans<Vd, Vs1>(kIsRounding, scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSransr) {
  CoralNPUVectorBinaryOpHelper<VSransrOp, int8_t, int16_t, int8_t, int16_t,
                               int32_t, int16_t, uint8_t, uint16_t, uint8_t,
                               uint16_t, uint32_t, uint16_t>("VSransr");
}

TEST_F(CoralNPUVectorInstructionsTest, VSraqs) {
  CoralNPUVectorBinaryOpHelper<VSransOp, int8_t, int32_t, int8_t, uint8_t,
                               uint32_t, uint8_t>("VSraqs");
}

TEST_F(CoralNPUVectorInstructionsTest, VSraqsr) {
  CoralNPUVectorBinaryOpHelper<VSransrOp, int8_t, int32_t, int8_t, uint8_t,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMul<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VMul) {
  CoralNPUVectorBinaryOpHelper<VMulOp>("VMul");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMuls<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMuls) {
  CoralNPUVectorBinaryOpHelper<VMulsOp>("VMuls");
}

TEST_F(CoralNPUVectorInstructionsTest, VMulsu) {
  CoralNPUVectorBinaryOpHelper<VMulsOp, kUnsigned>("VMulsu");
}

// Vector elements multiplication with widening.
template <typename Vd, typename Vs1, typename Vs2>
struct VMulwOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return static_cast<Vd>(vs1) * static_cast<Vd>(vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMulw<Vd, Vs1>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMulw) {
  CoralNPUVectorBinaryOpHelper<VMulwOp, int16_t, int8_t, int8_t, int32_t,
                               int16_t, int16_t>("VMulwOp");
}

TEST_F(CoralNPUVectorInstructionsTest, VMulwu) {
  CoralNPUVectorBinaryOpHelper<VMulwOp, uint16_t, uint8_t, uint8_t, uint32_t,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMulh<Vd>(scalar, strip_mine, false /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMulh) {
  CoralNPUVectorBinaryOpHelper<VMulhOp>("VMulh");
}

TEST_F(CoralNPUVectorInstructionsTest, VMulhu) {
  CoralNPUVectorBinaryOpHelper<VMulhOp, kUnsigned>("VMulhu");
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMulh<Vd>(scalar, strip_mine, true /* round */, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMulhr) {
  CoralNPUVectorBinaryOpHelper<VMulhrOp>("VMulhr");
}

TEST_F(CoralNPUVectorInstructionsTest, VMulhur) {
  CoralNPUVectorBinaryOpHelper<VMulhrOp, kUnsigned>("VMulhur");
}

// Saturating signed doubling multiply returning high half with optional
// rounding.
template <typename T>
T CoralNPUVDmulhHelper(bool round, bool round_neg, T vs1, T vs2) {
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
    return CoralNPUVDmulhHelper<Vd>(kNonRounding, false /* round_neg*/, vs1,
                                    vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVDmulh<Vd>(scalar, strip_mine, kNonRounding, false /* round_neg*/,
                       inst);
  }
};

template <typename Vd, typename Vs1, typename Vs2>
struct VDmulhrOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return CoralNPUVDmulhHelper<Vd>(kIsRounding, false /* round_neg*/, vs1,
                                    vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVDmulh<Vd>(scalar, strip_mine, kIsRounding, false /* round_neg*/,
                       inst);
  }
};

template <typename Vd, typename Vs1, typename Vs2>
struct VDmulhrnOp {
  static Vd Op(Vs1 vs1, Vs2 vs2) {
    return CoralNPUVDmulhHelper<Vd>(kIsRounding, true /* round_neg*/, vs1, vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVDmulh<Vd>(scalar, strip_mine, kIsRounding, true /* round_neg*/,
                       inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VDmulh) {
  CoralNPUVectorBinaryOpHelper<VDmulhOp>("VDmulh");
}

TEST_F(CoralNPUVectorInstructionsTest, VDmulhr) {
  CoralNPUVectorBinaryOpHelper<VDmulhrOp>("VDmulhr");
}

TEST_F(CoralNPUVectorInstructionsTest, VDmulhrn) {
  CoralNPUVectorBinaryOpHelper<VDmulhrnOp>("VDmulhrn");
}

// Multiply accumulate.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaccOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) {
    return static_cast<int64_t>(vd) +
           static_cast<int64_t>(vs1) * static_cast<int64_t>(vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMacc<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMacc) {
  CoralNPUVectorBinaryOpHelper<VMaccOp>("VMacc");
}

// Multiply add.
template <typename Vd, typename Vs1, typename Vs2>
struct VMaddOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) {
    return static_cast<int64_t>(vs1) +
           static_cast<int64_t>(vd) * static_cast<int64_t>(vs2);
  }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVMadd<Vd>(scalar, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VMadd) {
  CoralNPUVectorBinaryOpHelper<VMaddOp>("VMadd");
}

// Slide next register by index.
template <typename T>
static std::pair<T, T> SlidenArgsGetter(
    bool horizontal, int index, int num_ops, int op_num, int dest_reg_sub_index,
    int element_index, int vd_size, bool widen_dst, int src1_widen_factor,
    int vs1_size, const std::vector<T>& vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T>& vs2_value, T rs2_value, bool halftype_op,
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
  static void CoralNPUOp(int index, bool strip_mine, Instruction* inst) {
    CoralNPUVSlidehn<T>(index, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSlidehn) {
  CoralNPUSlideOpHelper<VSlidehnOp, int8_t, int16_t, int32_t>(
      "VSlidehnOp", kHorizontal, true /* strip_mine */);
}

template <typename T>
struct VSlidevnOp {
  static constexpr auto kArgsGetter = SlidenArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void CoralNPUOp(int index, bool strip_mine, Instruction* inst) {
    CoralNPUVSlidevn<T>(index, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSliden) {
  CoralNPUSlideOpHelper<VSlidevnOp, int8_t, int16_t, int32_t>(
      "VSlidenOp", kVertical, false /* strip_mine */);
}

TEST_F(CoralNPUVectorInstructionsTest, VSlidevn) {
  CoralNPUSlideOpHelper<VSlidevnOp, int8_t, int16_t, int32_t>(
      "VSlidevnOp", kVertical, true /* strip_mine */);
}

// Slide previous register by index.
template <typename T>
static std::pair<T, T> SlidepArgsGetter(
    bool horizontal, int index, int num_ops, int op_num, int dest_reg_sub_index,
    int element_index, int vd_size, bool widen_dst, int src1_widen_factor,
    int vs1_size, absl::Span<const T> vs1_value, int vs2_size, bool s2_scalar,
    absl::Span<const T> vs2_value, T rs2_value, bool halftype_op,
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
  static void CoralNPUOp(int index, bool strip_mine, Instruction* inst) {
    CoralNPUVSlidehp<T>(index, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSlidehp) {
  CoralNPUSlideOpHelper<VSlidehpOp, int8_t, int16_t, int32_t>(
      "VSlidehpOp", kHorizontal, true /* strip_mine */);
}

template <typename T>
struct VSlidevpOp {
  static constexpr auto kArgsGetter = SlidepArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void CoralNPUOp(int index, bool strip_mine, Instruction* inst) {
    CoralNPUVSlidevp<T>(index, strip_mine, inst);
  }
};

TEST_F(CoralNPUVectorInstructionsTest, VSlidep) {
  CoralNPUSlideOpHelper<VSlidevpOp, int8_t, int16_t, int32_t>(
      "VSlidepOp", kVertical, false /* strip_mine */);
}

TEST_F(CoralNPUVectorInstructionsTest, VSlidevp) {
  CoralNPUSlideOpHelper<VSlidevpOp, int8_t, int16_t, int32_t>(
      "VSlidevpOp", kVertical, true /* strip_mine */);
}

// Select lanes from two operands with vector selection boolean.
template <typename Vd, typename Vs1, typename Vs2>
struct VSelOp {
  static Vd Op(Vd vd, Vs1 vs1, Vs2 vs2) { return vs1 & 1 ? vd : vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVSel<Vd>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VSel) {
  CoralNPUVectorBinaryOpHelper<VSelOp>("VSel");
}

// Select even/odd elements of concatenated registers.
template <typename T>
static std::pair<T, T> EvnOddOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T>& vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T>& vs2_value, T rs2_value, bool halftype_op,
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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVEvn<T>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VEvn) {
  CoralNPUShuffleOpHelper<VEvnOp, int8_t, int16_t, int32_t>("VEvn");
}

template <typename T>
struct VOddOp {
  static constexpr auto kArgsGetter = EvnOddOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs2; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVOdd<T>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VOdd) {
  CoralNPUShuffleOpHelper<VOddOp, int8_t, int16_t, int32_t>("VOdd");
}

template <typename T>
struct VEvnoddOp {
  static constexpr auto kArgsGetter = EvnOddOpArgsGetter<T>;
  static T Op(T vs1, T vs2) { return vs1; }
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVEvnodd<T>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VEvnodd) {
  CoralNPUShuffleOpHelper<VEvnoddOp, int8_t, int16_t, int32_t>("VEvnodd",
                                                               kWidenDst);
}

// Select even/odd elements of concatenated registers.
template <typename T>
static std::pair<T, T> ZipOpArgsGetter(
    int num_ops, int op_num, int dest_reg_sub_index, int element_index,
    int vd_size, bool widen_dst, int src1_widen_factor, int vs1_size,
    const std::vector<T>& vs1_value, int vs2_size, bool s2_scalar,
    const std::vector<T>& vs2_value, T rs2_value, bool halftype_op,
    bool vmvp_op) {
  auto src_index = (op_num * vs1_size + element_index +
                    dest_reg_sub_index * vs1_size * num_ops) /
                   2;

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
  static void CoralNPUOp(bool scalar, bool strip_mine, Instruction* inst) {
    CoralNPUVZip<T>(scalar, strip_mine, inst);
  }
};
TEST_F(CoralNPUVectorInstructionsTest, VZip) {
  CoralNPUShuffleOpHelper<VZipOp, int8_t, int16_t, int32_t>("VZip", kWidenDst);
}

}  // namespace
