#ifndef SIM_KELVIN_VECTOR_MEMORY_INSTRUCTIONS_H_
#define SIM_KELVIN_VECTOR_MEMORY_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

using mpact::sim::generic::Instruction;

template <typename T>
void KelvinVLd(bool has_length, bool has_stride, bool strip_mine,
               Instruction *inst);

template <typename T>
void KelvinVLdRegWrite(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinVSt(bool has_length, bool has_stride, bool strip_mine,
               Instruction *inst);

template <typename T>
void KelvinVStQ(bool strip_mine, Instruction *inst);

template <typename T>
void KelvinGetVl(bool strip_mine, bool is_rs1, bool is_rs2,
                 const mpact::sim::generic::Instruction *inst);

void KelvinVcGet(const mpact::sim::generic::Instruction *inst);

void KelvinAcSet(bool is_transpose,
                 const mpact::sim::generic::Instruction *inst);

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_VECTOR_MEMORY_INSTRUCTIONS_H_
