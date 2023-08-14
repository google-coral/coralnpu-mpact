#include "sim/kelvin_vector_convolution_instructions.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "sim/kelvin_state.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"

namespace kelvin::sim {

using mpact::sim::generic::GetInstructionSource;
using mpact::sim::riscv::RV32VectorSourceOperand;

// Implement the 3-arg vector convolution (im2col + matmul)
// vs1 (narrow) represents the starting register of 8 vector registers
// vs3 (wide) is the starting register of group of up-to 8 vector
// registers. xs2 stores the convolution command.
// `vd` is not used in the op.
void KelvinVConv(Instruction *inst) {
  auto state = static_cast<KelvinState *>(inst->state());
  constexpr int kVectorLenInByte = kVectorLengthInBits / 8;
  constexpr int kVectorLenInWord = kVectorLenInByte / sizeof(uint32_t);

  vconv_cmd_t conv_cmd;
  auto reg_data = GetInstructionSource<uint32_t>(inst, 1, 0);
  memcpy(&conv_cmd, &reg_data, sizeof(conv_cmd));

  // Exam the content of the cmd.
  if (conv_cmd.mode != 0) {  // only supports 8-bit mode
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }
  if (conv_cmd.start > conv_cmd.stop) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }
  if (conv_cmd.start >= kVectorLenInWord || conv_cmd.stop >= kVectorLenInWord) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                /*epc=*/inst->address(), inst);
    return;
  }

  // Read the narrow source.
  auto vs1 = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto vs3 = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  AccArrayTemplate<std::array<uint8_t, kVectorLenInByte>> vec_narrow;
  for (int vec_idx = 0; vec_idx < vec_narrow.size(); ++vec_idx) {
    auto source_span = vs1->GetRegister(vec_idx)->data_buffer()->Get<uint8_t>();
    for (int j = 0; j < vec_narrow[vec_idx].size(); ++j) {
      vec_narrow[vec_idx][j] = source_span[j];
    }
  }

  // Prepare the accumulator.
  auto accumulator = state->acc_register();

  // Convert the biases to 9-bit signed values.
  int32_t sbias1 = (static_cast<int32_t>(conv_cmd.sbias1) << 23) >> 23;
  int32_t sbias2 = (static_cast<int32_t>(conv_cmd.sbias2) << 23) >> 23;

  // Multiply-Accumulate of conv(8x32xi8,  8x32xi8) -> 8x8xi32.
  // Internally they are broken into 4 groups to for accumulation to handle the
  // double-widening data without extra interleaving steps. Also, the operation
  // has both im2col and matmul in one shot (image data in `vs1`, filter/kernel
  // in `vs3`), so for the typical matmul, the input re-shuffling is required.
  //
  // Note the output of this op CANNOT be used directly, because it is still
  // in the double-widening format. It is expected to be followed by some
  // double-reduction instructions to read the 8-bit data back in order.
  constexpr int kInterleave[] = {0, 2, 1, 3};  // (ee, oe, eo, oo)
  constexpr int kQuadBase = 4;                 // For double-widening.
  constexpr int kQuadMask = kQuadBase - 1;
  for (int k = conv_cmd.start; k <= conv_cmd.stop; ++k) {
    auto wide_source_span =
        vs3->GetRegister(k - conv_cmd.start)->data_buffer()->Get<uint8_t>();
    for (int i = 0; i < vec_narrow.size(); ++i) {
      for (int j = 0; j < wide_source_span.size(); ++j) {
        // data1 (narrow) is transposed and broadcasted.
        uint8_t n = vec_narrow[i][kQuadBase * k + (j & kQuadMask)];
        int32_t sdata1 = conv_cmd.sdata1 ? static_cast<int8_t>(n) : n;
        uint8_t w = wide_source_span[j];
        int32_t sdata2 = conv_cmd.sdata2 ? static_cast<int8_t>(w) : w;
        const int rbase = i & ~kQuadMask;
        const int rquad = i & kQuadMask;
        const int word = j / kQuadBase;
        const int idx_i = rbase + kInterleave[word & kQuadMask];
        const int idx_j =
            rquad * (accumulator.size() / kQuadBase) + (word / kQuadBase);
        accumulator[idx_i][idx_j] += (sdata1 + sbias1) * (sdata2 + sbias2);
      }
    }
  }

  // Write the results back to the accumulation register
  for (int i = 0; i < state->acc_register().size(); ++i) {
    auto acc_array = state->acc_vec(i);
    *acc_array = accumulator[i];
  }
}

}  // namespace kelvin::sim
