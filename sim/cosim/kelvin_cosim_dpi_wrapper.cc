// Copyright 2025 Google LLC
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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "riscv/riscv32g_vec_decoder.h"
#include "riscv/riscv_csr.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "external/svdpi_h_file/file/svdpi.h"

// Include the DPI-C contract header.
#include "sim/cosim/kelvin_cosim_dpi.h"

constexpr int kKelvinVectorByteLength = 16;
constexpr uint32_t kKelvinStartAddress = 0;

namespace {
using ::mpact::sim::generic::DecoderInterface;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscV32GVecDecoder;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVTop;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

class MpactHandle {
 public:
  MpactHandle()
      : memory_(std::make_unique<FlatDemandMemory>()),
        rv_state_(CreateRVState(memory_.get())),
        rv_fp_state_(CreateFPState(rv_state_.get())),
        rvv_state_(CreateVectorState(rv_state_.get())),
        rv_decoder_(CreateDecoder(rv_state_.get(), memory_.get())),
        rv_top_(CreateRiscVTop(rv_state_.get(), rv_decoder_.get())) {
    absl::Status pc_write = rv_top_->WriteRegister("pc", kKelvinStartAddress);
    CHECK_OK(pc_write) << "Error writing to pc.";
  }

  uint32_t get_pc() {
    absl::StatusOr<uint64_t> read_reg_status = rv_top_->ReadRegister("pc");
    CHECK_OK(read_reg_status);
    if (!read_reg_status.ok()) {
      LOG(ERROR) << "[DPI] Failed to read pc.";
      return 0;
    }
    return static_cast<uint32_t>(read_reg_status.value());
  }

  RiscVTop* rv_top() { return rv_top_.get(); }

  RiscVState* rv_state() { return rv_state_.get(); }

 private:
  std::unique_ptr<RiscVState> CreateRVState(MemoryInterface* memory) {
    auto rv_state =
        std::make_unique<RiscVState>("RiscV32GV", RiscVXlen::RV32, memory);
    // Make sure the architectural and abi register aliases are added.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
      (void)rv_state->AddRegister<RV32Register>(reg_name);
      (void)rv_state->AddRegisterAlias<RV32Register>(reg_name,
                                                     kXRegisterAliases[i]);
    }
    return rv_state;
  }

  std::unique_ptr<RiscVFPState> CreateFPState(RiscVState* rv_state) {
    return std::make_unique<RiscVFPState>(rv_state->csr_set(), rv_state);
  }

  std::unique_ptr<RiscVVectorState> CreateVectorState(RiscVState* rv_state) {
    return std::make_unique<RiscVVectorState>(rv_state,
                                              kKelvinVectorByteLength);
  }

  std::unique_ptr<DecoderInterface> CreateDecoder(RiscVState* rv_state,
                                                  MemoryInterface* memory) {
    return std::make_unique<RiscV32GVecDecoder>(rv_state, memory);
  }

  std::unique_ptr<RiscVTop> CreateRiscVTop(RiscVState* rv_state,
                                           DecoderInterface* decoder) {
    return std::make_unique<RiscVTop>("KelvinPlaceholder", rv_state, decoder);
  }

  const std::unique_ptr<MemoryInterface> memory_;
  const std::unique_ptr<RiscVState> rv_state_;
  const std::unique_ptr<RiscVFPState> rv_fp_state_;
  const std::unique_ptr<RiscVVectorState> rvv_state_;
  const std::unique_ptr<DecoderInterface> rv_decoder_;
  const std::unique_ptr<RiscVTop> rv_top_;
};

MpactHandle* g_mpact_handle = nullptr;
}  // namespace

int mpact_init() {
  if (g_mpact_handle != nullptr) {
    LOG(ERROR) << "[DPI] mpact_init: g_mpact_handle is not null. "
               << "mpact_fini() must be called first.";
    return -1;
  }
  g_mpact_handle = new MpactHandle();
  return 0;
}

int mpact_reset() {
  if (g_mpact_handle != nullptr) {
    mpact_fini();
  }
  return mpact_init();
}

int mpact_step(const svLogicVecVal* instruction) {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_step: g_mpact_handle is null.";
    return -1;
  }
  uint32_t inst_word = instruction->aval;
  if (!g_mpact_handle->rv_top()
           ->WriteMemory(g_mpact_handle->get_pc(), &inst_word,
                         sizeof(inst_word))
           .ok()) {
    LOG(ERROR) << "[DPI] mpact_step: Failed to write instruction to memory.";
    return 1;
  }

  if (!g_mpact_handle->rv_top()->Step(1).ok()) {
    LOG(ERROR) << "[DPI] mpact_step: Failed to step the simulator.";
    return 2;
  }
  return 0;
}

bool mpact_is_halted() {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_is_halted: g_mpact_handle is null.";
    return false;
  }
  LOG(ERROR) << "[DPI] mpact_is_halted: Unimplemented.";
  return false;
}

uint32_t mpact_get_pc() {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_pc: g_mpact_handle is null.";
    return 0;
  }
  return g_mpact_handle->get_pc();
}

uint32_t mpact_get_gpr(uint32_t index) {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_gpr: g_mpact_handle is null.";
    return 0;
  }
  std::string reg_name =
      absl::StrCat(mpact::sim::riscv::RiscVState::kXregPrefix, index);
  mpact::sim::riscv::RiscVTop* rv_top = g_mpact_handle->rv_top();
  absl::StatusOr<uint64_t> read_reg_status = rv_top->ReadRegister(reg_name);
  if (!read_reg_status.ok()) {
    LOG(ERROR) << "[DPI] mpact_get_gpr: Failed to read register: " << reg_name;
    return 0;
  }
  return static_cast<uint32_t>(read_reg_status.value());
}

uint32_t mpact_get_csr(uint32_t address) {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_csr: g_mpact_handle is null.";
    return 0;
  }
  uint64_t csr_index = static_cast<uint64_t>(address);

  absl::StatusOr<mpact::sim::riscv::RiscVCsrInterface*> get_csr_status =
      g_mpact_handle->rv_state()->csr_set()->GetCsr(csr_index);

  if (!get_csr_status.ok()) {
    LOG(ERROR) << "[DPI] mpact_get_csr: Failed to get CSR: " << address;
    return 0;
  }
  mpact::sim::riscv::RiscVCsrInterface* csr = get_csr_status.value();
  return csr->AsUint32();
}

int mpact_fini() {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_fini: g_mpact_handle is null.";
    return -1;
  }
  delete g_mpact_handle;
  g_mpact_handle = nullptr;
  return 0;
}
