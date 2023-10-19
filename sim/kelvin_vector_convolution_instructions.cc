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
#include "mpact/sim/generic/type_helpers.h"

namespace kelvin::sim {
namespace {
constexpr int kVectorLenInByte = kVectorLengthInBits / 8;
constexpr int kVectorLenInWord = kVectorLenInByte / sizeof(uint32_t);
constexpr int kDwRegisterProducts = 3;
}  // namespace

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).
using ::mpact::sim::generic::GetInstructionSource;
using ::mpact::sim::riscv::RV32VectorDestinationOperand;
using ::mpact::sim::riscv::RV32VectorSourceOperand;

// Implement the 3-arg vector convolution (im2col + matmul)
// vs1 (narrow) represents the starting register of 8 vector registers
// vs3 (wide) is the starting register of group of up-to 8 vector
// registers. xs2 stores the convolution command.
// `vd` is not used in the op.
void KelvinVConv(Instruction *inst) {
  auto state = static_cast<KelvinState *>(inst->state());

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

// Implements accumulation of 3 32-element 8bit*8bit Hadamard products.
// vs1 is the starting register of 9 vector activation registers, of which
//     three are selected.
// vs3 (wide) is the starting register of group of 3 vector registers.
// xs2 stores the convolution command.
// `vd` is used if |write_acc| is set to true.
void KelvinVDwconv(bool write_acc, Instruction *inst) {
  KelvinState *state = static_cast<KelvinState *>(inst->state());
  uint32_t reg_data = GetInstructionSource<uint32_t>(inst, 1, 0);
  vdwconv_u8_t dwconv_cmd;
  memcpy(&dwconv_cmd, &reg_data, sizeof(dwconv_cmd));

  int vs1_idx[3];
  switch (dwconv_cmd.regbase) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      vs1_idx[0] = dwconv_cmd.regbase;
      vs1_idx[1] = dwconv_cmd.regbase + 1;
      vs1_idx[2] = dwconv_cmd.regbase + 2;
      break;
    case 7:
      vs1_idx[0] = 1;
      vs1_idx[1] = 0;
      vs1_idx[2] = 2;
      break;
    case 8:
    case 9:
    case 10:
    case 11:
      vs1_idx[0] = (2 * dwconv_cmd.regbase) - 15;
      vs1_idx[1] = vs1_idx[0] + 1;
      vs1_idx[2] = 0;
      break;
    case 12:
    case 13:
    case 14:
    case 15:
      vs1_idx[0] = (2 * dwconv_cmd.regbase) - 22;
      vs1_idx[1] = 0;
      vs1_idx[2] = 1;
      break;
  }

  auto vs1 = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  absl::Span<uint32_t> vs10_span =
      vs1->GetRegister(vs1_idx[0])->data_buffer()->Get<uint32_t>();
  absl::Span<uint32_t> vs11_span =
      vs1->GetRegister(vs1_idx[1])->data_buffer()->Get<uint32_t>();
  absl::Span<uint32_t> vs12_span =
      vs1->GetRegister(vs1_idx[2])->data_buffer()->Get<uint32_t>();
  uint32_t a_data[kDwRegisterProducts * kVectorLenInWord];
  switch (dwconv_cmd.sparsity) {
    case 0:
      memcpy(a_data, vs10_span.data(), 8 * sizeof(uint32_t));
      memcpy(a_data + 8, vs11_span.data(), 8 * sizeof(uint32_t));
      memcpy(a_data + 16, vs12_span.data(), 8 * sizeof(uint32_t));
      break;
    case 1:
      a_data[0] = vs10_span[7];
      memcpy(a_data + 1, vs11_span.data(), 7 * sizeof(uint32_t));
      memcpy(a_data + 8, vs11_span.data(), 8 * sizeof(uint32_t));
      memcpy(a_data + 16, vs11_span.data() + 1, 7 * sizeof(uint32_t));
      a_data[23] = vs12_span[0];
      break;
    case 2:
      memcpy(a_data, vs10_span.data(), 8 * sizeof(uint32_t));
      memcpy(a_data + 8, vs10_span.data() + 1, 7 * sizeof(uint32_t));
      a_data[15] = vs11_span[0];
      memcpy(a_data + 16, vs10_span.data() + 2, 6 * sizeof(uint32_t));
      a_data[22] = vs11_span[0];
      a_data[23] = vs11_span[1];
      break;
    default:
      // Invalid state enum
      state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                  *mpact::sim::riscv::ExceptionCode::kIllegalInstruction,
                  /*epc=*/inst->address(), inst);
  }

  auto vs3 = static_cast<RV32VectorSourceOperand *>(inst->Source(2));
  int32_t *acc = reinterpret_cast<int32_t *>(state->dw_acc_vec(0));

  for (int r = 0; r < kDwRegisterProducts; r++) {
    absl::Span<uint8_t> a_span = absl::Span<uint8_t>(
        reinterpret_cast<uint8_t *>(a_data + (r * kVectorLenInWord)),
        kVectorLenInByte);
    absl::Span<uint8_t> b_span =
        vs3->GetRegister(r)->data_buffer()->Get<uint8_t>();

    for (int i = 0; i < kVectorLenInByte; i++) {
      int32_t a =
          dwconv_cmd.sdata1 ? static_cast<int8_t>(a_span[i]) : a_span[i];
      int32_t b =
          dwconv_cmd.sdata2 ? static_cast<int8_t>(b_span[i]) : b_span[i];
      a += dwconv_cmd.sbias1;
      b += dwconv_cmd.sbias2;

      constexpr static int interleave[4] = {0, 2, 1, 3};
      int acc_reg = interleave[(i & 0b11)];
      int reg_offset = i >> 2;
      acc[kVectorLenInWord * acc_reg + reg_offset] += a * b;
    }
  }

  if (!write_acc) {
    return;
  }

  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  for (int i = 0; i < 4; i++) {
    DataBuffer *dest_db = vd->AllocateDataBuffer(i);
    absl::Span<uint32_t> dest_span = dest_db->Get<uint32_t>();
    for (int j = 0; j < kVectorLenInWord; j++) {
      dest_span[j] = acc[i * kVectorLenInWord + j];
    }
    dest_db->Submit();
  }
}

}  // namespace kelvin::sim
