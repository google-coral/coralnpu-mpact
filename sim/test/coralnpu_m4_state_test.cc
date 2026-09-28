// Copyright 2026 Google LLC
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

#include "sim/coralnpu_m4_state.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_zvt_state.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"

namespace {

using ::coralnpu::sim::CoralNPUM4State;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::util::FlatDemandMemory;

// The set of named tiles that exist for each TILE_ELEMENT_WIDTH, and the
// effective tile edge (EFFECTIVE_TILE_EDGE) over which row/col indices are
// valid. These come straight from the spec's tile-count table (15.1.1.1).
struct TileElementWidthConfig {
  int tile_element_width;
  std::vector<int> tiles;
  int effective_tile_edge;  // rows and cols range over [0,
                            // effective_tile_edge).
};

std::vector<TileElementWidthConfig> ConfigsForTe(int kTe) {
  return {
      {8, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, kTe},
      {16, {0, 2, 4, 6, 8, 10, 12, 14}, kTe},
      {32, {0, 4, 8, 12}, kTe},
      {64, {0, 2, 4, 6, 8, 10, 12, 14}, kTe / 2},
  };
}

class PunningTest : public ::testing::TestWithParam<int> {
 protected:
  PunningTest()
      : memory_(std::make_unique<FlatDemandMemory>()),
        state_(::coralnpu::sim::CreateCoralNPUM4State("test", RiscVXlen::RV32,
                                                      memory_.get())) {
    auto matrix_status = ::mpact::sim::riscv::RiscVZvtMatrixState::Create(
        state_.get(), /*kTe=*/GetParam());
    CHECK_OK(matrix_status);
    matrix_ = std::move(*matrix_status);
  }

  std::unique_ptr<FlatDemandMemory> memory_;
  std::unique_ptr<CoralNPUM4State> state_;
  std::unique_ptr<::mpact::sim::riscv::RiscVZvtMatrixState> matrix_;
};

// For every supported TILE_ELEMENT_WIDTH, the punning map from (tile, row, col)
// over the valid tiles/element-edge to byte offsets must be a *bijection* onto
// the whole tile buffer: each byte covered exactly once. This simultaneously
// verifies injectivity (no aliasing within a TILE_ELEMENT_WIDTH) and full
// coverage.
TEST_P(PunningTest, IsBijectionPerTew) {
  const int kTe = GetParam();
  const int kBufferBytes = 16 * kTe * kTe;
  ASSERT_EQ(matrix_->tile_buffer_bytes(), kBufferBytes);

  for (const auto& cfg : ConfigsForTe(kTe)) {
    std::vector<int> coverage(kBufferBytes, 0);
    const int kElemBytes = cfg.tile_element_width / 8;
    for (int tile : cfg.tiles) {
      for (int row = 0; row < cfg.effective_tile_edge; row++) {
        for (int col = 0; col < cfg.effective_tile_edge; col++) {
          int offset = matrix_->Punning(tile, row, col, cfg.tile_element_width);
          ASSERT_GE(offset, 0);
          ASSERT_LE(offset + kElemBytes, kBufferBytes)
              << "tile_element_width=" << cfg.tile_element_width
              << " tile=" << tile << " row=" << row << " col=" << col;
          for (int b = 0; b < kElemBytes; b++) coverage[offset + b]++;
        }
      }
    }
    for (int i = 0; i < kBufferBytes; i++) {
      EXPECT_EQ(coverage[i], 1)
          << "byte " << i << " covered " << coverage[i]
          << " times for tile_element_width=" << cfg.tile_element_width
          << " (kTe=" << kTe << ")";
    }
  }
}

// Round-trip element storage through GetElem/SetElem for a couple of TEWs.
TEST_P(PunningTest, GetSetElemRoundTrip) {
  const int kTe = GetParam();
  // TILE_ELEMENT_WIDTH=32 int32 tile mt0.
  for (int row = 0; row < kTe; row++) {
    for (int col = 0; col < kTe; col++) {
      matrix_->SetElem<int32_t>(0, row, col, 32, row * 100 + col);
    }
  }
  for (int row = 0; row < kTe; row++) {
    for (int col = 0; col < kTe; col++) {
      EXPECT_EQ((matrix_->GetElem<int32_t>(0, row, col, 32)), row * 100 + col);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(TileEdge, PunningTest,
                         ::testing::Values(4, 8, 16, 32));

}  // namespace
