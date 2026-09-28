#include "sim/coralnpu_v2_state.h"

#include <cstdint>
#include <memory>

#include "googletest/include/gtest/gtest.h"
#include "riscv/riscv_state.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace coralnpu::sim {
namespace {

using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;

TEST(CoralNPUV2StateTest, HasPermission) {
  auto memory = std::make_unique<FlatDemandMemory>();
  auto state = CreateCoralNPUV2State("test", RiscVXlen::RV32, memory.get());

  // Add a read-only region.
  state->AddMemoryRegion(0x1000, 0x1000, MemoryPermission::kRead);
  // Add a write-only region.
  state->AddMemoryRegion(0x2000, 0x1000, MemoryPermission::kWrite);
  // Add an execute-only region.
  state->AddMemoryRegion(0x3000, 0x1000, MemoryPermission::kExecute);
  // Add a read-write region.
  state->AddMemoryRegion(0x4000, 0x1000,
                         MemoryPermission::kRead | MemoryPermission::kWrite);

  // Check read-only region.
  EXPECT_TRUE(state->HasPermission(0x1000, 4, MemoryPermission::kRead));
  EXPECT_FALSE(state->HasPermission(0x1000, 4, MemoryPermission::kWrite));
  EXPECT_FALSE(state->HasPermission(0x1000, 4, MemoryPermission::kExecute));

  // Check write-only region.
  EXPECT_FALSE(state->HasPermission(0x2000, 4, MemoryPermission::kRead));
  EXPECT_TRUE(state->HasPermission(0x2000, 4, MemoryPermission::kWrite));
  EXPECT_FALSE(state->HasPermission(0x2000, 4, MemoryPermission::kExecute));

  // Check execute-only region.
  EXPECT_FALSE(state->HasPermission(0x3000, 4, MemoryPermission::kRead));
  EXPECT_FALSE(state->HasPermission(0x3000, 4, MemoryPermission::kWrite));
  EXPECT_TRUE(state->HasPermission(0x3000, 4, MemoryPermission::kExecute));

  // Check read-write region.
  EXPECT_TRUE(state->HasPermission(0x4000, 4, MemoryPermission::kRead));
  EXPECT_TRUE(state->HasPermission(0x4000, 4, MemoryPermission::kWrite));
  EXPECT_FALSE(state->HasPermission(0x4000, 4, MemoryPermission::kExecute));
  EXPECT_TRUE(state->HasPermission(
      0x4000, 4, MemoryPermission::kRead | MemoryPermission::kWrite));

  // Check out of bounds.
  EXPECT_FALSE(state->HasPermission(0x5000, 4, MemoryPermission::kRead));

  // Add two adjacent regions with the same permissions.
  state->AddMemoryRegion(0x6000, 0x1000,
                         MemoryPermission::kRead | MemoryPermission::kWrite);
  state->AddMemoryRegion(0x7000, 0x1000,
                         MemoryPermission::kRead | MemoryPermission::kWrite);

  // Check spanning multiple regions (should fail even if both have
  // permissions).
  EXPECT_FALSE(state->HasPermission(0x6FFC, 8, MemoryPermission::kRead));
}

TEST(CoralNPUV2StateTest, MepcLegalization) {
  auto memory = std::make_unique<FlatDemandMemory>();
  auto state = CreateCoralNPUV2State("test", RiscVXlen::RV32, memory.get());
  auto res = state->csr_set()->GetCsr("mepc");
  if (!res.ok()) {
    FAIL() << "Failed to get mepc CSR: " << res.status();
  }
  auto* mepc = res.value();

  // Bit 1 must be cleared when writing to mepc on CoralNPU (IALIGN=32).
  mepc->Write(static_cast<uint32_t>(0x2));
  EXPECT_EQ(mepc->AsUint32(), 0x0);
  EXPECT_EQ(mepc->GetUint32(), 0x0);

  // Bit 1 must remain 0 when writing 0xFFFFFFFF.
  mepc->Write(static_cast<uint32_t>(0xFFFFFFFF));
  EXPECT_EQ(mepc->AsUint32() & 0x2, 0x0);
  EXPECT_EQ(mepc->GetUint32() & 0x2, 0x0);
}

}  // namespace
}  // namespace coralnpu::sim
