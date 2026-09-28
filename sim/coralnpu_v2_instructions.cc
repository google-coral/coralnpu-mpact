// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sim/coralnpu_v2_instructions.h"

#include <cstdint>
#include <type_traits>

#include "sim/coralnpu_v2_state.h"
#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "riscv/riscv_f_instructions.h"
#include "riscv/riscv_i_instructions.h"
#include "riscv/riscv_instruction_helpers.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_vector_memory_instructions.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/operand_interface.h"
#include "mpact/sim/generic/type_helpers.h"

constexpr int kMaxEmul = 64;

namespace {

using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::MemoryPermission;
using ::mpact::sim::generic::GetInstructionSource;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::ExceptionCode;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.

// Checks if a scalar memory access is within a valid memory region with the
// required permissions. If not, it traps and returns false.
template <typename Register, typename ValueType>
bool AccessCheck(const Instruction* /*absl_nonnull*/ instruction,
                 ExceptionCode fault_exception_code) {
  using RegVal = typename Register::ValueType;
  using URegVal = std::make_unsigned_t<RegVal>;
  URegVal base = ::mpact::sim::generic::GetInstructionSource<URegVal>(
      instruction, /*index=*/0);
  RegVal offset = ::mpact::sim::generic::GetInstructionSource<RegVal>(
      instruction, /*index=*/1);
  URegVal address = base + offset;
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  MemoryPermission permission =
      (fault_exception_code == ExceptionCode::kLoadAccessFault)
          ? MemoryPermission::kRead
          : MemoryPermission::kWrite;
  if (!state->HasPermission(address, sizeof(ValueType), permission)) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/address,
                *fault_exception_code,
                /*epc=*/instruction->address(), instruction);
    return false;
  }
  return true;
}

// Verifies if a jump target address has execute permissions. If not, it traps
// and returns false.
bool IsJumpAllowed(uint64_t address,
                   const Instruction* /*absl_nonnull*/ instruction,
                   ExceptionCode fault_exception_code) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  if (!state->HasPermission(address,
                            ::coralnpu::sim::kCoralNPUV2InstructionSize,
                            MemoryPermission::kExecute)) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/address,
                *fault_exception_code,
                /*epc=*/address, instruction);
    return false;
  }
  return true;
}

// Checks the mask bit for a given vector element index.
bool CheckVectorMask(const uint8_t* src_masks, int i) {
  int index = i >> 3;
  int offset = i & 0b111;
  return ((src_masks[index] >> offset) & 0x1) != 0;
}

// Retrieves the raw mask data buffer from a vector source operand.
const uint8_t* GetMaskData(const Instruction* /*absl_nonnull*/ instruction,
                           int source_index) {
  if (source_index < 0) return nullptr;
  auto* src_mask_op =
      static_cast<RV32VectorSourceOperand*>(instruction->Source(source_index));
  return src_mask_op->GetRegister(0)->data_buffer()->Get<uint8_t>().data();
}

// Validates a single vector element's memory access against permissions and
// traps if it fails.
bool CheckVectorAccess(CoralNPUV2State* state,
                       const Instruction* /*absl_nonnull*/ instruction,
                       uint32_t address, int size,
                       MemoryPermission permission) {
  if (!state->HasPermission(address, size, permission)) {
    ExceptionCode fault_exception_code = (permission == MemoryPermission::kRead)
                                             ? ExceptionCode::kLoadAccessFault
                                             : ExceptionCode::kStoreAccessFault;
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/address,
                *fault_exception_code,
                /*epc=*/instruction->address(), instruction);
    return false;
  }
  return true;
}

// Performs a "fast path" check for contiguous or strided memory ranges to see
// if they fit within a single valid memory region. Returns true if the entire
// range has the required permissions.
bool HasRangePermission(CoralNPUV2State* state, uint32_t base, int64_t stride,
                        int element_count, int element_size,
                        MemoryPermission permission) {
  if (element_count <= 0) return true;

  int64_t span = static_cast<int64_t>(element_count - 1) * stride;
  // If the total span exceeds the 32-bit address space, fall back to slow path
  // to ensure element-by-element wrap-around checks trap correctly.
  if (span < -4294967295LL || span > 4294967295LL) return false;

  uint32_t start_addr = base;
  uint32_t end_addr = static_cast<uint32_t>(base + span);

  // Fall back to slow path if the 32-bit address wrapped around.
  if ((span > 0 && end_addr < start_addr) ||
      (span < 0 && start_addr < end_addr)) {
    return false;
  }

  uint32_t min_addr = span >= 0 ? start_addr : end_addr;
  uint32_t max_addr = span >= 0 ? end_addr : start_addr;

  // Use 64-bit arithmetic to safely detect if the end of the access overflows
  // the 32-bit address space. 0x100000000ULL is 0xFFFFFFFF + 1.
  uint64_t range_end = static_cast<uint64_t>(max_addr) + element_size;
  if (range_end > 0x100000000ULL) return false;

  return state->HasPermission(
      min_addr, static_cast<uint32_t>(range_end - min_addr), permission);
}

// Iterates over vector elements, applying masking and a provided validator
// function. Returns true if all elements pass validation.
template <typename AddressValidatorFunc>
bool ValidateVectorAccess(const Instruction* /*absl_nonnull*/ instruction,
                          int mask_source_index, int element_count,
                          bool ignore_vstart,
                          AddressValidatorFunc&& validator) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  int start = ignore_vstart ? 0 : rv_vector->vstart();

  const uint8_t* src_masks = GetMaskData(instruction, mask_source_index);

  for (int i = start; i < element_count; i++) {
    if (src_masks != nullptr && !CheckVectorMask(src_masks, i)) {
      continue;
    }
    if (!validator(state, i)) return false;
  }
  return true;
}

// Attempts a fast-path range check for strided vector accesses. If the fast
// path fails, it falls back to element-by-element validation.
template <typename AddressValidatorFunc>
bool ValidateStridedVectorAccess(const Instruction* /*absl_nonnull*/ instruction,
                                 int mask_source_index, int element_count,
                                 bool ignore_vstart, uint32_t base,
                                 int64_t stride, int element_size,
                                 MemoryPermission permission,
                                 AddressValidatorFunc&& validator) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  int start = ignore_vstart ? 0 : rv_vector->vstart();

  if (start < element_count &&
      HasRangePermission(state, base + start * stride, stride,
                         element_count - start, element_size, permission)) {
    return true;
  }

  return ValidateVectorAccess(instruction, mask_source_index, element_count,
                              ignore_vstart,
                              std::forward<AddressValidatorFunc>(validator));
}

// Extracts an index value from a vector of indices, potentially spanning
// multiple registers.
uint64_t GetIndexValue(
    const ::mpact::sim::riscv::RV32VectorSourceOperand* index_op, int i,
    int index_bytes, int indices_per_reg) {
  int reg_idx = i / indices_per_reg;
  int elem_idx = i % indices_per_reg;
  int byte_offset = elem_idx * index_bytes;

  absl::Span<const uint8_t> index_span =
      index_op->GetRegister(reg_idx)->data_buffer()->Get<uint8_t>();
  uint64_t index_val = 0;
  for (int b = 0; b < index_bytes; ++b) {
    index_val |=
        (static_cast<uint64_t>(index_span[byte_offset + b]) << (b * 8));
  }
  return index_val;
}

// Validates indexed vector memory accesses using a provided validator.
template <typename AddressValidatorFunc>
bool ValidateIndexedVectorAccess(
    const Instruction* /*absl_nonnull*/ instruction, int mask_source_index,
    int element_count, bool ignore_vstart,
    const ::mpact::sim::riscv::RV32VectorSourceOperand* index_op,
    int index_width, int indices_per_reg, AddressValidatorFunc&& validator) {
  return ValidateVectorAccess(
      instruction, mask_source_index, element_count, ignore_vstart,
      [&](CoralNPUV2State* state, int i) {
        uint32_t index_val = static_cast<uint32_t>(
            GetIndexValue(index_op, i, index_width, indices_per_reg));
        return validator(state, i, index_val);
      });
}

}  // namespace

namespace coralnpu::sim {

using ::coralnpu::sim::CoralNPUV2State;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RV32VectorSourceOperand;
using ::mpact::sim::riscv::RVFpRegister;

void CoralNPUV2Mpause(const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  state->MPause(instruction);
}

void CoralNPUV2Lw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed = AccessCheck<RV32Register, int32_t>(
      instruction, ExceptionCode::kLoadAccessFault);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int32_t>(instruction);
  }
}

void CoralNPUV2Lh(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed = AccessCheck<RV32Register, int16_t>(
      instruction, ExceptionCode::kLoadAccessFault);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int16_t>(instruction);
  }
}

void CoralNPUV2Lhu(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed = AccessCheck<RV32Register, uint16_t>(
      instruction, ExceptionCode::kLoadAccessFault);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, uint16_t>(instruction);
  }
}

void CoralNPUV2Lb(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed = AccessCheck<RV32Register, int8_t>(
      instruction, ExceptionCode::kLoadAccessFault);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, int8_t>(instruction);
  }
}

void CoralNPUV2Lbu(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_load_allowed = AccessCheck<RV32Register, uint8_t>(
      instruction, ExceptionCode::kLoadAccessFault);
  if (is_load_allowed) {
    ::mpact::sim::riscv::RVLoad<RV32Register, uint8_t>(instruction);
  }
}

void CoralNPUV2Sw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed = AccessCheck<RV32Register, uint32_t>(
      instruction, ExceptionCode::kStoreAccessFault);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint32_t>(instruction);
  }
}

void CoralNPUV2Sh(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed = AccessCheck<RV32Register, uint16_t>(
      instruction, ExceptionCode::kStoreAccessFault);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint16_t>(instruction);
  }
}

void CoralNPUV2Sb(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed = AccessCheck<RV32Register, uint8_t>(
      instruction, ExceptionCode::kStoreAccessFault);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RVStore<RV32Register, uint8_t>(instruction);
  }
}

void CoralNPUV2Fsw(const Instruction* /*absl_nonnull*/ instruction) {
  bool is_store_allowed = AccessCheck<RVFpRegister, uint64_t>(
      instruction, ExceptionCode::kStoreAccessFault);
  if (is_store_allowed) {
    ::mpact::sim::riscv::RV32::RiscVFSw(instruction);
  }
}

void CoralNPUV2Jal(const Instruction* /*absl_nonnull*/ instruction) {
  using RegVal = RV32Register::ValueType;
  using URegVal = std::make_unsigned_t<RegVal>;
  RegVal offset =
      ::mpact::sim::generic::GetInstructionSource<RegVal>(instruction, 0);
  URegVal address = instruction->address() + offset;
  if (IsJumpAllowed(address, instruction,
                    ExceptionCode::kInstructionAccessFault)) {
    ::mpact::sim::riscv::RV32::RiscVIJal(instruction);
  }
}

void CoralNPUV2Jalr(const Instruction* /*absl_nonnull*/ instruction) {
  using RegVal = RV32Register::ValueType;
  using URegVal = std::make_unsigned_t<RegVal>;
  URegVal base =
      ::mpact::sim::generic::GetInstructionSource<URegVal>(instruction, 0);
  RegVal offset =
      ::mpact::sim::generic::GetInstructionSource<RegVal>(instruction, 1);
  URegVal address = base + offset;
  if (IsJumpAllowed(address, instruction,
                    ExceptionCode::kInstructionAccessFault)) {
    ::mpact::sim::riscv::RV32::RiscVIJalr(instruction);
    return;
  }
  // If jump is not allowed, IsJumpAllowed has already called Trap.
  // We still need to update rd with pc+4 for Jalr.
  using RegType = RV32Register;
  using ValueType = typename RegType::ValueType;
  auto* dest = instruction->Destination(1);
  auto* db = dest->AllocateDataBuffer();
  if (db != nullptr) {
    db->Set<ValueType>(0, instruction->address() + instruction->size());
    db->Submit();
  }
}

void CoralNPUV2VlStrided(int element_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);
  int64_t stride = GetInstructionSource<int64_t>(instruction, 1);
  int emul = element_width * rv_vector->vector_length_multiplier() /
             rv_vector->selected_element_width();
  if ((emul > kMaxEmul) || (emul == 0)) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *ExceptionCode::kIllegalInstruction,
                /*epc=*/instruction->address(), instruction);
    return;
  }

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/2, vl,
      /*ignore_vstart=*/false, base, stride, element_width,
      MemoryPermission::kRead, [&](CoralNPUV2State* state, int i) {
        uint32_t address = base + i * stride;
        return CheckVectorAccess(state, instruction, address, element_width,
                                 MemoryPermission::kRead);
      });

  if (success) {
    ::mpact::sim::riscv::VlStrided(element_width, instruction);
  }
}

void CoralNPUV2Vlm(const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);
  int num_bytes = (rv_vector->vector_length() + 7) / 8;

  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/-1, num_bytes,
      /*ignore_vstart=*/true, base, /*stride=*/1, sizeof(uint8_t),
      MemoryPermission::kRead, [&](CoralNPUV2State* state, int i) {
        uint32_t address = base + i;
        return CheckVectorAccess(state, instruction, address, sizeof(uint8_t),
                                 MemoryPermission::kRead);
      });

  if (success) {
    ::mpact::sim::riscv::Vlm(instruction);
  }
}

void CoralNPUV2VlRegister(int num_regs, int element_width_bytes,
                          const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);
  int num_elements =
      rv_vector->vector_register_byte_length() * num_regs / element_width_bytes;

  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/-1, num_elements,
      /*ignore_vstart=*/true, base, element_width_bytes, element_width_bytes,
      MemoryPermission::kRead, [&](CoralNPUV2State* state, int i) {
        uint32_t address = base + i * element_width_bytes;
        return CheckVectorAccess(state, instruction, address,
                                 element_width_bytes, MemoryPermission::kRead);
      });

  if (success) {
    ::mpact::sim::riscv::VlRegister(num_regs, element_width_bytes, instruction);
  }
}

void CoralNPUV2VlIndexed(int index_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);
  auto* index_op =
      static_cast<RV32VectorSourceOperand*>(instruction->Source(1));

  int sew_bytes = rv_vector->selected_element_width();
  int vlen_bytes = rv_vector->vector_register_byte_length();
  int indices_per_reg = vlen_bytes / index_width;

  bool success = ValidateIndexedVectorAccess(
      instruction, /*mask_source_index=*/2, rv_vector->vector_length(),
      /*ignore_vstart=*/false, index_op, index_width, indices_per_reg,
      [&](CoralNPUV2State* state, int i, uint32_t index_val) {
        uint32_t address = base + index_val;
        return CheckVectorAccess(state, instruction, address, sew_bytes,
                                 MemoryPermission::kRead);
      });

  if (success) {
    ::mpact::sim::riscv::VlIndexed(index_width, instruction);
  }
}

void CoralNPUV2VlSegment(int element_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);

  // Source 2 is the number of fields - 1 (nf).
  int nf = GetInstructionSource<int32_t>(instruction, 2);
  int num_fields = nf + 1;
  int segment_byte_size = num_fields * element_width;

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/1, vl,
      /*ignore_vstart=*/false, base, segment_byte_size, segment_byte_size,
      MemoryPermission::kRead, [&](CoralNPUV2State* state, int i) {
        uint32_t address = base + i * segment_byte_size;
        return CheckVectorAccess(state, instruction, address, segment_byte_size,
                                 MemoryPermission::kRead);
      });

  if (success) {
    ::mpact::sim::riscv::VlSegment(element_width, instruction);
  }
}

void CoralNPUV2VlSegmentStrided(int element_width,
                                const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);
  int64_t segment_stride = GetInstructionSource<int64_t>(instruction, 1);

  // Source 3 is the number of fields - 1 (nf).
  int nf = GetInstructionSource<int32_t>(instruction, 3);
  int num_fields = nf + 1;

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/2, vl,
      /*ignore_vstart=*/false, base, segment_stride, num_fields * element_width,
      MemoryPermission::kRead, [&](CoralNPUV2State* state, int i) {
        uint32_t segment_address = base + i * segment_stride;
        for (int field = 0; field < num_fields; field++) {
          uint32_t address = segment_address + field * element_width;
          if (!CheckVectorAccess(state, instruction, address, element_width,
                                 MemoryPermission::kRead)) {
            return false;
          }
        }
        return true;
      });

  if (success) {
    ::mpact::sim::riscv::VlSegmentStrided(element_width, instruction);
  }
}

void CoralNPUV2VlSegmentIndexed(int index_width,
                                const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 0);

  // Get the index vector operand.
  auto* index_op =
      static_cast<RV32VectorSourceOperand*>(instruction->Source(1));

  // Source 3 is the number of fields - 1 (nf).
  int nf = GetInstructionSource<int32_t>(instruction, 3);
  int num_fields = nf + 1;

  int element_width = rv_vector->selected_element_width();
  int vlen_bytes = rv_vector->vector_register_byte_length();
  int indices_per_reg = vlen_bytes / index_width;

  bool success = ValidateIndexedVectorAccess(
      instruction, /*mask_source_index=*/2, rv_vector->vector_length(),
      /*ignore_vstart=*/false, index_op, index_width, indices_per_reg,
      [&](CoralNPUV2State* state, int i, uint32_t index_val) {
        uint32_t element_base_address = base + index_val;
        for (int field = 0; field < num_fields; field++) {
          uint32_t address = element_base_address + field * element_width;
          if (!CheckVectorAccess(state, instruction, address, element_width,
                                 MemoryPermission::kRead)) {
            return false;
          }
        }
        return true;
      });

  if (success) {
    ::mpact::sim::riscv::VlSegmentIndexed(index_width, instruction);
  }
}

void CoralNPUV2VsStrided(int element_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  int64_t stride = GetInstructionSource<int64_t>(instruction, 2);

  int emul = element_width * rv_vector->vector_length_multiplier() /
             rv_vector->selected_element_width();
  if ((emul > kMaxEmul) || (emul == 0)) {
    state->Trap(/*is_interrupt=*/false, /*trap_value=*/0,
                *ExceptionCode::kIllegalInstruction,
                /*epc=*/instruction->address(), instruction);
    return;
  }

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/3, vl,
      /*ignore_vstart=*/false, base, stride, element_width,
      MemoryPermission::kWrite, [&](CoralNPUV2State* s, int i) {
        uint32_t address = base + i * stride;
        return CheckVectorAccess(s, instruction, address, element_width,
                                 MemoryPermission::kWrite);
      });

  if (success) {
    ::mpact::sim::riscv::VsStrided(element_width, instruction);
  }
}

void CoralNPUV2Vsm(const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  int num_bytes = (rv_vector->vector_length() + 7) / 8;

  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/-1, num_bytes,
      /*ignore_vstart=*/true, base, /*stride=*/1, sizeof(uint8_t),
      MemoryPermission::kWrite, [&](CoralNPUV2State* s, int i) {
        uint32_t address = base + i;
        return CheckVectorAccess(s, instruction, address, sizeof(uint8_t),
                                 MemoryPermission::kWrite);
      });

  if (success) {
    ::mpact::sim::riscv::Vsm(instruction);
  }
}

void CoralNPUV2VsRegister(int num_regs,
                          const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  int num_elements =
      rv_vector->vector_register_byte_length() * num_regs / sizeof(uint64_t);

  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/-1, num_elements,
      /*ignore_vstart=*/true, base, sizeof(uint64_t), sizeof(uint64_t),
      MemoryPermission::kWrite, [&](CoralNPUV2State* s, int i) {
        uint32_t address = base + i * sizeof(uint64_t);
        return CheckVectorAccess(s, instruction, address, sizeof(uint64_t),
                                 MemoryPermission::kWrite);
      });

  if (success) {
    ::mpact::sim::riscv::VsRegister(num_regs, instruction);
  }
}

void CoralNPUV2VsIndexed(int index_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  auto* index_op =
      static_cast<RV32VectorSourceOperand*>(instruction->Source(2));

  int sew_bytes = rv_vector->selected_element_width();
  int vlen_bytes = rv_vector->vector_register_byte_length();
  int indices_per_reg = vlen_bytes / index_width;

  bool success = ValidateIndexedVectorAccess(
      instruction, /*mask_source_index=*/3, rv_vector->vector_length(),
      /*ignore_vstart=*/false, index_op, index_width, indices_per_reg,
      [&](CoralNPUV2State* s, int i, uint32_t index_val) {
        uint32_t address = base + index_val;
        return CheckVectorAccess(s, instruction, address, sew_bytes,
                                 MemoryPermission::kWrite);
      });

  if (success) {
    ::mpact::sim::riscv::VsIndexed(index_width, instruction);
  }
}

void CoralNPUV2VsSegment(int element_width,
                         const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  int nf = GetInstructionSource<int32_t>(instruction, 3);
  int num_fields = nf + 1;
  int segment_byte_size = num_fields * element_width;

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/2, vl,
      /*ignore_vstart=*/false, base, segment_byte_size, segment_byte_size,
      MemoryPermission::kWrite, [&](CoralNPUV2State* s, int i) {
        uint32_t address = base + i * segment_byte_size;
        return CheckVectorAccess(s, instruction, address, segment_byte_size,
                                 MemoryPermission::kWrite);
      });

  if (success) {
    ::mpact::sim::riscv::VsSegment(element_width, instruction);
  }
}

void CoralNPUV2VsSegmentStrided(int element_width,
                                const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  int64_t segment_stride = GetInstructionSource<int64_t>(instruction, 2);
  int num_fields = GetInstructionSource<int32_t>(instruction, 4) + 1;

  int vl = rv_vector->vector_length();
  bool success = ValidateStridedVectorAccess(
      instruction, /*mask_source_index=*/3, vl,
      /*ignore_vstart=*/false, base, segment_stride, num_fields * element_width,
      MemoryPermission::kWrite, [&](CoralNPUV2State* s, int i) {
        uint32_t segment_address = base + i * segment_stride;
        for (int field = 0; field < num_fields; field++) {
          uint32_t address = segment_address + field * element_width;
          if (!CheckVectorAccess(s, instruction, address, element_width,
                                 MemoryPermission::kWrite)) {
            return false;
          }
        }
        return true;
      });

  if (success) {
    ::mpact::sim::riscv::VsSegmentStrided(element_width, instruction);
  }
}

void CoralNPUV2VsSegmentIndexed(int index_width,
                                const Instruction* /*absl_nonnull*/ instruction) {
  CoralNPUV2State* state = static_cast<CoralNPUV2State*>(instruction->state());
  RiscVVectorState* rv_vector = state->rv_vector();
  uint64_t base = GetInstructionSource<uint64_t>(instruction, 1);
  auto* index_op =
      static_cast<RV32VectorSourceOperand*>(instruction->Source(2));
  int nf = GetInstructionSource<int32_t>(instruction, 4);
  int num_fields = nf + 1;
  int element_width = rv_vector->selected_element_width();
  int vlen_bytes = rv_vector->vector_register_byte_length();
  int indices_per_reg = vlen_bytes / index_width;

  bool success = ValidateVectorAccess(
      instruction, /*mask_source_index=*/3, rv_vector->vector_length(),
      /*ignore_vstart=*/false, [&](CoralNPUV2State* s, int i) {
        uint64_t index_val =
            GetIndexValue(index_op, i, index_width, indices_per_reg);
        uint32_t segment_address = base + index_val;
        for (int field = 0; field < num_fields; field++) {
          uint32_t address = segment_address + field * element_width;
          if (!CheckVectorAccess(s, instruction, address, element_width,
                                 MemoryPermission::kWrite)) {
            return false;
          }
        }
        return true;
      });

  if (success) {
    ::mpact::sim::riscv::VsSegmentIndexed(index_width, instruction);
  }
}

}  // namespace coralnpu::sim
