#ifndef SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_
#define SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_

#include <cstdint>

#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

using mpact::sim::generic::Instruction;

// Command structure for the convolution instruction.
typedef struct KelvinVConvCmd {
  uint32_t mode : 2;    // 31:30
  uint32_t start : 5;   // 29:25
  uint32_t stop : 5;    // 24:20
  uint32_t sbias1 : 9;  // 19:11
  uint32_t sdata1 : 1;  // 10
  uint32_t sbias2 : 9;  // 9:1
  uint32_t sdata2 : 1;  // 0
} vconv_cmd_t;

void KelvinVConv(Instruction *inst);

}  // namespace kelvin::sim

#endif  // SIM_KELVIN_VECTOR_CONVOLUTION_INSTRUCTIONS_H_
