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

#include "sim/kelvin_vector_memory_instructions.h"

#include <algorithm>
#include <cstdint>

#include "sim/kelvin_state.h"
#include "absl/types/span.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/register.h"

namespace kelvin::sim {

using mpact::sim::generic::DataBuffer;
using mpact::sim::generic::GetInstructionSource;
using mpact::sim::riscv::LoadContext;
using mpact::sim::riscv::RV32VectorDestinationOperand;
using mpact::sim::riscv::RV32VectorSourceOperand;

// Vector load instruction with optional data length, stride and address
// register post-increment.
template <typename T>
void KelvinVLd(bool has_length, bool has_stride, bool strip_mine,
               Instruction *inst) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const uint32_t elts_per_register = vector_size_in_bytes / sizeof(T);

  const auto num_ops = strip_mine ? 4 : 1;
  auto addr = GetInstructionSource<uint32_t>(inst, 0, 0);
  // Check and exclude the cache invalidation bit. However, the semihost tests
  // use the memory space greater than the kelvin HW configuration and do not
  // comply to the magic bit setting. Exclude the check and mask for those
  // tests.
  if (state->max_physical_address() <=
      kKelvinMaxMemoryAddress) {  // exclude semihost tests
    addr &= kMemMask;
  }

  uint32_t elts_to_load = num_ops * elts_per_register;
  if (has_length) {
    auto length_arg = GetInstructionSource<uint32_t>(inst, 1, 0);
    elts_to_load = std::min(length_arg, elts_to_load);
  }

  uint32_t stride_elts = elts_per_register;
  if (has_stride) {
    auto stride_arg = GetInstructionSource<uint32_t>(inst, 1, 0);
    stride_elts = stride_arg;
  }

  auto *db_factory = inst->state()->db_factory();
  auto *address_db = db_factory->Allocate<uint64_t>(elts_to_load);
  auto *mask_db = db_factory->Allocate<bool>(elts_to_load);
  // Allocate the value data buffer that the loaded data is returned in.
  auto *value_db = db_factory->Allocate<T>(elts_to_load);

  auto addresses = address_db->Get<uint64_t>();
  auto masks = mask_db->Get<bool>();
  auto base = addr;
  auto elts_left = elts_to_load;
  for (int op_num = 0; op_num < num_ops; ++op_num) {
    uint32_t count = std::min(elts_left, elts_per_register);
    for (int i = 0; i < count; ++i) {
      addresses[op_num * elts_per_register + i] = base + i * sizeof(T);
      masks[op_num * elts_per_register + i] = true;
    }
    elts_left -= count;
    base += stride_elts * sizeof(T);
  }
  auto *context = new LoadContext(value_db);
  value_db->set_latency(0);
  state->LoadMemory(inst, address_db, mask_db, sizeof(T), value_db,
                    inst->child(), context);

  // Release the context and address_db. The others will be released elsewhere.
  context->DecRef();
  address_db->DecRef();
  mask_db->DecRef();

  const bool post_increment = inst->DestinationsSize() == 1;
  if (post_increment) {
    auto *reg =
        static_cast<
            mpact::sim::generic::RegisterDestinationOperand<uint32_t> *>(
            inst->Destination(0))
            ->GetRegister();

    if (elts_to_load > 0) {
      if (has_length && has_stride) {  // .tp
        addr += vector_size_in_bytes;
      } else if (!has_length && !has_stride &&
                 inst->SourcesSize() == 1) {  // .p.x
        addr += vector_size_in_bytes * num_ops;
      } else if (has_length) {  // .lp
        addr += elts_to_load * sizeof(T);
      } else if (has_stride) {  // .sp
        addr += stride_elts * sizeof(T) * num_ops;
      } else {  // .p.xx
        addr += GetInstructionSource<uint32_t>(inst, 1, 0) * sizeof(T);
      }
    }

    reg->data_buffer()->template Set<uint32_t>(0, addr);
  }
}
template void KelvinVLd<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVLd<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVLd<int32_t>(bool, bool, bool, Instruction *);

// VLd child instruction which writes data loaded to destination register(s).
template <typename T>
void KelvinVLdRegWrite(bool strip_mine, Instruction *inst) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const uint32_t elts_per_register = vector_size_in_bytes / sizeof(T);
  const auto num_ops = strip_mine ? 4 : 1;

  auto *context = static_cast<LoadContext *>(inst->context());
  auto values = context->value_db->template Get<T>();

  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer *dest_db = vd->AllocateDataBuffer(op_index);
    absl::Span<T> dest_span = dest_db->template Get<T>();

    for (int dst_element_index = 0; dst_element_index < elts_per_register;
         ++dst_element_index) {
      auto value_index = op_index * elts_per_register + dst_element_index;
      dest_span[dst_element_index] =
          value_index < context->value_db->template size<T>()
              ? values[value_index]
              : 0;
    }

    dest_db->Submit();
  }
}
template void KelvinVLdRegWrite<int8_t>(bool, Instruction *);
template void KelvinVLdRegWrite<int16_t>(bool, Instruction *);
template void KelvinVLdRegWrite<int32_t>(bool, Instruction *);

// Vector store instruction with the optional data length, stride and address
// register post-increment.
// Quad store stores either a quarter of the vector register content or the full
// register with xs2 stride.
template <typename T>
void VectorStoreHelper(bool has_length, bool has_stride, bool strip_mine,
                       bool is_quad, Instruction *inst) {
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const uint32_t elts_per_register = vector_size_in_bytes / sizeof(T);

  const auto num_ops = strip_mine ? 4 : 1;
  auto mem_addr = GetInstructionSource<uint32_t>(inst, 1, 0);
  if (state->max_physical_address() <=
      kKelvinMaxMemoryAddress) {  // exclude semihost tests
    mem_addr &= kMemMask;
  }
  auto vs = static_cast<RV32VectorSourceOperand *>(inst->Source(0));

  auto base_addr = mem_addr;

  uint32_t elts_to_store = num_ops * elts_per_register;
  if (has_length) {
    auto length_arg = GetInstructionSource<uint32_t>(inst, 2, 0);
    elts_to_store = std::min(length_arg, elts_to_store);
  }

  uint32_t stride_elts = elts_per_register;
  if (has_stride) {
    auto stride_arg = GetInstructionSource<uint32_t>(inst, 2, 0);
    stride_elts = stride_arg;
  }

  // Allocate the store memory
  auto *value_db = state->db_factory()->Allocate(elts_to_store * sizeof(T));
  auto *address_db = state->db_factory()->Allocate<uint64_t>(elts_to_store);
  auto *mask_db = state->db_factory()->Allocate<bool>(elts_to_store);
  auto addresses = address_db->Get<uint64_t>();
  auto value = value_db->Get<T>();
  auto mask = mask_db->Get<bool>();

  int address_index = 0;
  for (int op_num = 0; op_num < num_ops; op_num++) {
    auto source_span = vs->GetRegister(op_num)->data_buffer()->Get<T>();
    if (is_quad) {
      const uint32_t quad_size = elts_per_register / 4;
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < quad_size && address_index < elts_to_store; ++j) {
          addresses[address_index] = base_addr + j * sizeof(T);
          value[address_index] = source_span[i * quad_size + j];
          mask[address_index++] = true;
        }
        // Stride increase per quad_size.
        base_addr += stride_elts * sizeof(T);
      }
    } else {
      for (int i = 0; i < elts_per_register && address_index < elts_to_store;
           ++i) {
        addresses[address_index] = base_addr + i * sizeof(T);
        value[address_index] = source_span[i];
        mask[address_index++] = true;
      }
      base_addr += stride_elts * sizeof(T);
    }
  }
  state->StoreMemory(inst, address_db, mask_db, sizeof(T), value_db);
  value_db->DecRef();
  address_db->DecRef();
  mask_db->DecRef();

  const bool post_increment = inst->DestinationsSize() == 1;
  if (post_increment) {
    auto *reg =
        static_cast<
            mpact::sim::generic::RegisterDestinationOperand<uint32_t> *>(
            inst->Destination(0))
            ->GetRegister();
    if (elts_to_store > 0) {
      if (has_length && has_stride) {  // .tp
        mem_addr += vector_size_in_bytes;
      } else if (!has_length && !has_stride &&
                 inst->SourcesSize() == 2) {  // .p.x
        mem_addr += vector_size_in_bytes * num_ops;
      } else if (has_length) {  // .lp
        mem_addr += elts_to_store * sizeof(T);
      } else if (has_stride) {  // .sp
        const uint32_t quad_scale = is_quad ? 4 : 1;
        mem_addr += stride_elts * sizeof(T) * num_ops * quad_scale;
      } else {  // .p.xx
        mem_addr += GetInstructionSource<uint32_t>(inst, 2, 0) * sizeof(T);
      }
    }
    reg->data_buffer()->template Set<uint32_t>(0, mem_addr);
  }
}

template <typename T>
void KelvinVSt(bool has_length, bool has_stride, bool strip_mine,
               Instruction *inst) {
  VectorStoreHelper<T>(has_length, has_stride, strip_mine, /*is_quad=*/false,
                       inst);
}

template void KelvinVSt<int8_t>(bool, bool, bool, Instruction *);
template void KelvinVSt<int16_t>(bool, bool, bool, Instruction *);
template void KelvinVSt<int32_t>(bool, bool, bool, Instruction *);

// Duplicate a scalar value into a vector register.
template <typename T>
void KelvinVDup(bool strip_mine, Instruction *inst) {
  auto *state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  const uint32_t elts_per_register = vector_size_in_bytes / sizeof(T);
  const auto num_ops = strip_mine ? 4 : 1;

  // Gets destination register and scalar value.
  auto *vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto value = GetInstructionSource<T>(inst, 0);

  // Fill destination buffer and write to register.
  for (int op_index = 0; op_index < num_ops; ++op_index) {
    DataBuffer *dest_db = vd->AllocateDataBuffer(op_index);
    absl::Span<T> dest_span = dest_db->template Get<T>();
    for (int dst_element_index = 0; dst_element_index < elts_per_register;
         ++dst_element_index) {
      dest_span[dst_element_index] = value;
    }
    dest_db->Submit();
  }
}

template void KelvinVDup<int8_t>(bool, Instruction *);
template void KelvinVDup<int16_t>(bool, Instruction *);
template void KelvinVDup<int32_t>(bool, Instruction *);

template <typename T>
void KelvinVStQ(bool strip_mine, Instruction *inst) {
  VectorStoreHelper<T>(/*has_length=*/false, /*has_stride=*/true, strip_mine,
                       /*is_quad=*/true, inst);
}

template void KelvinVStQ<int8_t>(bool, Instruction *);
template void KelvinVStQ<int16_t>(bool, Instruction *);
template void KelvinVStQ<int32_t>(bool, Instruction *);

// Return the supported vl length. It starts with the maximum value based on
// vector_length and then is capped to the minimum by the additional inputs.
template <typename T>
void KelvinGetVl(bool strip_mine, bool is_rs1, bool is_rs2,
                 const mpact::sim::generic::Instruction *inst) {
  auto *dest_reg =
      static_cast<mpact::sim::generic::RegisterDestinationOperand<uint32_t> *>(
          inst->Destination(0))
          ->GetRegister();
  auto state = static_cast<KelvinState *>(inst->state());
  const int vector_size_in_bytes = state->vector_length() / 8;
  uint32_t vlen = vector_size_in_bytes / sizeof(T);
  if (strip_mine) {
    vlen *= 4;
  }

  if (is_rs1) {
    uint32_t rs1 = mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 0);
    vlen = std::min(vlen, rs1);
  }
  if (is_rs2) {
    uint32_t rs2 = mpact::sim::generic::GetInstructionSource<uint32_t>(inst, 1);
    vlen = std::min(vlen, rs2);
  }
  dest_reg->data_buffer()->Set<uint32_t>(0, vlen);
}
template void KelvinGetVl<int8_t>(bool, bool, bool, const Instruction *);
template void KelvinGetVl<int16_t>(bool, bool, bool, const Instruction *);
template void KelvinGetVl<int32_t>(bool, bool, bool, const Instruction *);

// Copy convolution accumulation registers into general vector register. In HW,
// it is set to be v48..55.
void KelvinVcGet(const mpact::sim::generic::Instruction *inst) {
  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  auto *state = static_cast<KelvinState *>(inst->state());
  const uint32_t kVecLenInWord = state->vector_length() / 32;
  for (int op_index = 0; op_index < kVecLenInWord; ++op_index) {
    DataBuffer *dest_db = vd->AllocateDataBuffer(op_index);
    absl::Span<uint32_t> dest_span = dest_db->Get<uint32_t>();
    auto *acc_vec = state->acc_vec(op_index);
    for (int i = 0; i < dest_span.size(); ++i) {
      dest_span[i] = (*acc_vec)[i];
    }
    acc_vec->fill(0);
    dest_db->Submit();
  }
}

// Copy the content from the general vector registers to convolution
// accumulation register. In HW, vs has to be 16-register aligned, and vd has
// to be set to v48.
void KelvinAcSet(bool is_transpose,
                 const mpact::sim::generic::Instruction *inst) {
  auto vs = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto *state = static_cast<KelvinState *>(inst->state());
  const uint32_t kVecLenInWord = state->vector_length() / 32;
  for (int op_index = 0; op_index < kVecLenInWord; ++op_index) {
    auto source_span =
        vs->GetRegister(op_index)->data_buffer()->Get<uint32_t>();
    for (int i = 0; i < source_span.size(); ++i) {
      if (is_transpose) {
        auto *acc_vec = state->acc_vec(i);
        (*acc_vec)[op_index] = source_span[i];
      } else {
        auto *acc_vec = state->acc_vec(op_index);
        (*acc_vec)[i] = source_span[i];
      }
    }
  }
}

// Copy the content from the source `vs1` banks to the `vd` banks to prepare the
// depthwise convolution. Due to compiler encoding, this op is typeless and only
// assumes `vs1` and `vd` content in 8-bit type.
void KelvinADwInit(const mpact::sim::generic::Instruction *inst) {
  auto *state = static_cast<KelvinState *>(inst->state());
  // Only set a quarter of the to prepare for double-widening in depth-wise
  // convolution.
  const uint32_t init_n = state->vector_length() / (8 * 4);
  constexpr int kInitSize = 4;
  auto vs = static_cast<RV32VectorSourceOperand *>(inst->Source(0));
  auto vd = static_cast<RV32VectorDestinationOperand *>(inst->Destination(0));
  for (int op_index = 0; op_index < kInitSize; ++op_index) {
    auto source_span = vs->GetRegister(op_index)->data_buffer()->Get<uint8_t>();
    uint8_t *dwacc_span =
        reinterpret_cast<uint8_t *>(state->dw_acc_vec(8 * op_index));
    for (int i = 0; i < 32; i++) {
      dwacc_span[i] = source_span[i];
    }

    DataBuffer *dest_db = vd->AllocateDataBuffer(op_index);
    absl::Span<uint8_t> dest_span = dest_db->Get<uint8_t>();
    for (int i = 0; i < init_n; ++i) {
      dest_span[i] = source_span[i];
    }
    dest_db->Submit();
  }
}

}  // namespace kelvin::sim
