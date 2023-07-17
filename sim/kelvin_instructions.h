#ifndef SIM_KELVIN_INSTRUCTIONS_H_
#define SIM_KELVIN_INSTRUCTIONS_H_

#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

void KelvinIllegalInstruction(mpact::sim::generic::Instruction *inst);

void KelvinNopInstruction(mpact::sim::generic::Instruction *inst);

void KelvinIMpause(const mpact::sim::generic::Instruction *inst);

void KelvinLogInstruction(int log_mode, mpact::sim::generic::Instruction *inst);

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_INSTRUCTIONS_H_
