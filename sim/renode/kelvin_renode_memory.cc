#include "sim/renode/kelvin_renode_memory.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "sim/kelvin_state.h"
#include "absl/base/macros.h"
#include "absl/numeric/bits.h"

namespace kelvin::sim::renode {

KelvinRenodeMemory::KelvinRenodeMemory(uint64_t block_size_bytes,
                                       uint64_t memory_size_bytes,
                                       uint8_t **block_ptr_list,
                                       uint64_t base_address,
                                       unsigned addressable_unit_size)
    : addressable_unit_size_(addressable_unit_size),
      allocation_byte_size_(block_size_bytes * addressable_unit_size),
      memory_block_size_bytes_(block_size_bytes),
      base_address_(base_address),
      max_address_(base_address + memory_size_bytes) {
  // Available memory should be greater than Kelvin's default 4MB address space.
  ABSL_HARDENING_ASSERT(max_address_ > kelvin::sim::kKelvinMaxMemoryAddress);
  uint64_t num_block = (max_address_ + block_size_bytes - 1) / block_size_bytes;
  // Build the block map.
  for (int i = 0; i < num_block; ++i) {
    block_map_.push_back(block_ptr_list[i]);
  }
  addressable_unit_shift_ = absl::bit_width(addressable_unit_size) - 1;
}

bool KelvinRenodeMemory::IsValidAddress(uint64_t address,
                                        uint64_t high_address) {
  return (address >= base_address_) && (high_address <= max_address_);
}

void KelvinRenodeMemory::Load(uint64_t address, DataBuffer *db,
                              Instruction *inst, ReferenceCount *context) {
  int size_in_units = db->size<uint8_t>() / addressable_unit_size_;
  uint64_t high = address + size_in_units;
  ABSL_HARDENING_ASSERT(IsValidAddress(address, high));
  ABSL_HARDENING_ASSERT(size_in_units > 0);
  uint8_t *byte_ptr = static_cast<uint8_t *>(db->raw_ptr());
  // Load the data into the data buffer.
  LoadStoreHelper(address, byte_ptr, size_in_units, true);
  // Execute the instruction to process and write back the load data.
  if (nullptr != inst) {
    if (db->latency() > 0) {
      inst->IncRef();
      if (context != nullptr) context->IncRef();
      inst->state()->function_delay_line()->Add(db->latency(),
                                                [inst, context]() {
                                                  inst->Execute(context);
                                                  if (context != nullptr)
                                                    context->DecRef();
                                                  inst->DecRef();
                                                });
    } else {
      inst->Execute(context);
    }
  }
}

void KelvinRenodeMemory::Load(DataBuffer *address_db, DataBuffer *mask_db,
                              int el_size, DataBuffer *db, Instruction *inst,
                              ReferenceCount *context) {
  auto mask_span = mask_db->Get<bool>();
  auto address_span = address_db->Get<uint64_t>();
  uint8_t *byte_ptr = static_cast<uint8_t *>(db->raw_ptr());
  int size_in_units = el_size / addressable_unit_size_;
  ABSL_HARDENING_ASSERT(size_in_units > 0);
  // This is either a gather load, or a unit stride load depending on size of
  // the address span.
  bool gather = address_span.size() > 1;
  for (unsigned i = 0; i < mask_span.size(); i++) {
    if (!mask_span[i]) continue;
    uint64_t address = gather ? address_span[i] : address_span[0] + i * el_size;
    uint64_t high = address + size_in_units;
    ABSL_HARDENING_ASSERT(IsValidAddress(address, high));
    LoadStoreHelper(address, &byte_ptr[el_size * i], size_in_units, true);
  }
  // Execute the instruction to process and write back the load data.
  if (nullptr != inst) {
    if (db->latency() > 0) {
      inst->IncRef();
      if (context != nullptr) context->IncRef();
      inst->state()->function_delay_line()->Add(db->latency(),
                                                [inst, context]() {
                                                  inst->Execute(context);
                                                  if (context != nullptr)
                                                    context->DecRef();
                                                  inst->DecRef();
                                                });
    } else {
      inst->Execute(context);
    }
  }
}

void KelvinRenodeMemory::Store(uint64_t address, DataBuffer *db) {
  int size_in_units = db->size<uint8_t>() / addressable_unit_size_;
  uint64_t high = address + size_in_units;
  ABSL_HARDENING_ASSERT(IsValidAddress(address, high));
  ABSL_HARDENING_ASSERT(size_in_units > 0);
  uint8_t *byte_ptr = static_cast<uint8_t *>(db->raw_ptr());
  LoadStoreHelper(address, byte_ptr, size_in_units, /*is_load*/ false);
}

void KelvinRenodeMemory::Store(DataBuffer *address_db, DataBuffer *mask_db,
                               int el_size, DataBuffer *db) {
  auto mask_span = mask_db->Get<bool>();
  auto address_span = address_db->Get<uint64_t>();
  uint8_t *byte_ptr = static_cast<uint8_t *>(db->raw_ptr());
  int size_in_units = el_size / addressable_unit_size_;
  ABSL_HARDENING_ASSERT(size_in_units > 0);
  // If the address_span.size() > 1, then this is a scatter store, otherwise
  // it's a unit stride store.
  bool scatter = address_span.size() > 1;
  for (unsigned i = 0; i < mask_span.size(); i++) {
    if (!mask_span[i]) continue;
    uint64_t address =
        scatter ? address_span[i] : address_span[0] + i * el_size;
    uint64_t high = address + size_in_units;
    ABSL_HARDENING_ASSERT(IsValidAddress(address, high));
    LoadStoreHelper(address, &byte_ptr[el_size * i], size_in_units,
                    /*is_load*/ false);
  }
}

void KelvinRenodeMemory::LoadStoreHelper(uint64_t address, uint8_t *byte_ptr,
                                         int size_in_units, bool is_load) {
  ABSL_HARDENING_ASSERT(address < max_address_);
  do {
    // Find the block in the map.
    uint64_t block_idx = address / memory_block_size_bytes_;

    uint8_t *block = block_map_[block_idx];

    int block_unit_offset = (address - block_idx * memory_block_size_bytes_);

    // Compute how many addressable units to load/store from/to the current
    // block.
    int store_size_in_units =
        std::min(size_in_units, allocation_byte_size_ - block_unit_offset);

    // Translate from unit size to byte size.
    int store_size_in_bytes = store_size_in_units << addressable_unit_shift_;
    int block_byte_offset = block_unit_offset << addressable_unit_shift_;

    if (is_load) {
      std::memcpy(byte_ptr, &block[block_byte_offset], store_size_in_bytes);
    } else {
      std::memcpy(&block[block_byte_offset], byte_ptr, store_size_in_bytes);
    }

    // Adjust address, data pointer and the remaining data left to be
    // loaded/stored.
    size_in_units -= store_size_in_units;
    byte_ptr += store_size_in_bytes;
    address += store_size_in_units;
  } while (size_in_units > 0);
}

}  // namespace kelvin::sim::renode
