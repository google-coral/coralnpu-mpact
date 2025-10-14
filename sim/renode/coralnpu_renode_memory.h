/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_MEMORY_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_MEMORY_H_

#include <cstdint>
#include <vector>

#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/ref_count.h"
#include "mpact/sim/util/memory/memory_interface.h"

namespace coralnpu::sim::renode {

using ::mpact::sim::generic::DataBuffer;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::ReferenceCount;

// A memory interface class with memory blocks created, Initialize, and shared
// by Renode's MappedMemory module as an array of memory block pointers, the
// size of each block, and the total size of the memory (uint64_t
// memory_block_size_bytes, uint64_t memory_size_byte, uint8_t **
// memory_block_pointer_list). The class is tied to the CoralNPU configuration
// with memory size check.
class CoralNPURenodeMemory : public mpact::sim::util::MemoryInterface {
 public:
  CoralNPURenodeMemory(uint64_t block_size_bytes, uint64_t memory_size_bytes,
                       uint8_t** block_ptr_list, uint64_t base_address,
                       unsigned addressable_unit_size);
  CoralNPURenodeMemory(uint64_t block_size_bytes, uint64_t memory_size_bytes,
                       uint8_t** block_ptr_list)
      : CoralNPURenodeMemory(block_size_bytes, memory_size_bytes,
                             block_ptr_list, 0, 1) {}

  // The memory is not allocated by this class, so there is nothing to release
  // in the destructor.
  ~CoralNPURenodeMemory() override = default;

  // Implementation of the MemoryInterface methods.
  void Load(uint64_t address, DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override;

  void Load(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
            DataBuffer* db, Instruction* inst,
            ReferenceCount* context) override;

  // Convenience template function that calls the above function with the
  // element size as the sizeof() the template parameter type.
  template <typename T>
  void Load(DataBuffer* address_db, DataBuffer* mask_db, DataBuffer* db,
            Instruction* inst, ReferenceCount* context) {
    Load(address_db, mask_db, sizeof(T), db, inst, context);
  }

  void Store(uint64_t address, DataBuffer* db) override;
  void Store(DataBuffer* address_db, DataBuffer* mask_db, int el_size,
             DataBuffer* db) override;

  // Convenience template function that calls the above function with the
  // element size as the sizeof() the template parameter type.
  template <typename T>
  void Store(DataBuffer* address_db, DataBuffer* mask_db, DataBuffer* db) {
    Store(address_db, mask_db, sizeof(T), db);
  }

 private:
  void LoadStoreHelper(uint64_t address, uint8_t* byte_ptr, int size_in_units,
                       bool is_load);
  bool IsValidAddress(uint64_t address, uint64_t high_address);

  int addressable_unit_shift_;
  int addressable_unit_size_;
  int allocation_byte_size_;
  uint64_t memory_block_size_bytes_;
  uint64_t base_address_;
  uint64_t max_address_;

  std::vector<uint8_t*> block_map_;
};

}  // namespace coralnpu::sim::renode

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_CORALNPU_RENODE_MEMORY_H_
