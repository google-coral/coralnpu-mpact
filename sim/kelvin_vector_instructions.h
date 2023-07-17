#ifndef SIM_KELVIN_VECTOR_INSTRUCTIONS_H_
#define SIM_KELVIN_VECTOR_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

using mpact::sim::generic::Instruction;

// Vector 2-arg .vv, .vx arithmetic operations.
template <typename T>
void KelvinVAdd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSub(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVRSub(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVEq(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVNe(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVLt(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVLe(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVGt(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVGe(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVAbsd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMax(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMin(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVAdd3(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVAdds(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVAddsu(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSubs(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSubsu(bool scalar, bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVAddw(bool scalar, bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVSubw(bool scalar, bool strip_mine, Instruction *inst);

template <typename Td, typename Ts2>
void KelvinVAcc(bool scalar, bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVPadd(bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVPsub(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVHadd(bool scalar, bool strip_mine, bool round, Instruction *inst);

template <typename T>
void KelvinVHsub(bool scalar, bool strip_mine, bool round, Instruction *inst);

template <typename T>
void KelvinVAnd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVOr(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVXor(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVRev(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVRor(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMvp(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSll(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSra(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSrl(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVShift(bool round, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVNot(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVClb(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVClz(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVCpop(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMv(bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVSrans(bool round, bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMul(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMuls(bool scalar, bool strip_mine, Instruction *inst);

template <typename Td, typename Ts>
void KelvinVMulw(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMulh(bool scalar, bool strip_mine, bool round, Instruction *inst);

template <typename T>
void KelvinVDmulh(bool scalar, bool strip_mine, bool round, bool round_neg,
                  Instruction *inst);

template <typename T>
void KelvinVMacc(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVMadd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSlidevn(int index, Instruction *inst);

template <typename T>
void KelvinVSlidehn(int index, Instruction *inst);

template <typename T>
void KelvinVSlidevp(int index, Instruction *inst);

template <typename T>
void KelvinVSlidehp(int index, Instruction *inst);

template <typename T>
void KelvinVSel(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVEvn(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVOdd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVEvnodd(bool scalar, bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVZip(bool scalar, bool strip_mine, Instruction *inst);
}  // namespace kelvin::sim

#endif  // SIM_KELVIN_VECTOR_INSTRUCTIONS_H_
