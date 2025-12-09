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

#include "sim/coralnpu_v2_state.h"
#include "sim/coralnpu_v2_user_decoder.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "riscv/riscv_fp_state.h"
#include "riscv/riscv_register.h"
#include "riscv/riscv_register_aliases.h"
#include "riscv/riscv_state.h"
#include "riscv/riscv_top.h"
#include "riscv/riscv_vector_state.h"
#include "mpact/sim/generic/core_debug_interface.h"
#include "mpact/sim/generic/data_buffer.h"
#include "mpact/sim/generic/decoder_interface.h"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/generic/type_helpers.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/memory/memory_interface.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"
#include "external/svdpi_h_file/file/svdpi.h"

// Include the DPI-C contract header.
#include "sim/cosim/coralnpu_cosim_dpi.h"

namespace {
using ::coralnpu::sim::CoralNPUV2State;
using ::coralnpu::sim::CoralNPUV2StateFactory;
using ::coralnpu::sim::CoralNPUV2UserDecoder;
using ::coralnpu::sim::kCoralnpuV2VectorByteLength;
using ::mpact::sim::generic::DecoderInterface;
using ::mpact::sim::generic::Instruction;
using ::mpact::sim::generic::operator*;  // NOLINT: clang-tidy false positive.
using ::mpact::sim::riscv::kFRegisterAliases;
using ::mpact::sim::riscv::kXRegisterAliases;
using ::mpact::sim::riscv::RiscVFPState;
using ::mpact::sim::riscv::RiscVState;
using ::mpact::sim::riscv::RiscVTop;
using ::mpact::sim::riscv::RiscVVectorState;
using ::mpact::sim::riscv::RiscVXlen;
using ::mpact::sim::riscv::RV32Register;
using ::mpact::sim::riscv::RVFpRegister;
using ::mpact::sim::riscv::RVVectorRegister;
using ::mpact::sim::util::ElfProgramLoader;
using ::mpact::sim::util::FlatDemandMemory;
using ::mpact::sim::util::MemoryInterface;

using HaltReason = ::mpact::sim::generic::CoreDebugInterface::HaltReason;

constexpr uint32_t kCoralNPUV2DefaultInitialMisaValue =
    (*::mpact::sim::riscv::RiscVXlen::RV32 << 30) |
    *::mpact::sim::riscv::IsaExtension::kIntegerMulDiv |
    *::mpact::sim::riscv::IsaExtension::kRVIBaseIsa |
    *::mpact::sim::riscv::IsaExtension::kSinglePrecisionFp |
    *::mpact::sim::riscv::IsaExtension::kVectorExtension;
constexpr uint32_t kCoralNPUV2DefaultItcmStartAddress = 0;
constexpr uint32_t kCoralNPUV2DefaultItcmLength = 0x2000;
constexpr uint32_t kCoralNPUV2DefaultDtcmStartAddress = 0x10000;
constexpr uint32_t kCoralNPUV2DefaultDtcmLength = 0x8000;

class MpactHandle {
 public:
  MpactHandle()
      : memory_(std::make_unique<FlatDemandMemory>()),
        elf_loader_(std::make_unique<ElfProgramLoader>(memory_.get())) {}

  void Init(sim_config_t* /*absl_nullable*/ config_data) {
    CHECK(!is_initialized_) << "[DPI] Init: is_initialized_ is already true.";
    if (config_data != nullptr) {
      state_ = CreateState(
          memory_.get(), config_data->itcm_start_address,
          config_data->itcm_length, config_data->initial_misa_value,
          config_data->dtcm_start_address, config_data->dtcm_length);
    } else {
      state_ = CreateState(
          memory_.get(), kCoralNPUV2DefaultItcmStartAddress,
          kCoralNPUV2DefaultItcmLength, kCoralNPUV2DefaultInitialMisaValue,
          kCoralNPUV2DefaultDtcmStartAddress, kCoralNPUV2DefaultDtcmLength);
    }
    rv_fp_state_ = CreateFPState(state_.get());
    state_->set_rv_fp(rv_fp_state_.get());
    rvv_state_ = CreateVectorState(state_.get());
    rv_decoder_ = CreateDecoder(state_.get(), memory_.get());
    rv_top_ = CreateRiscVTop(state_.get(), rv_decoder_.get());

    state_->AddMpauseHandler([this](const Instruction* inst) -> bool {
      LOG(INFO) << "Halting cosimulation due to mpause instruction.";
      cosimulation_halted_ = true;
      state_->Cease(inst);
      return true;
    });
    uint32_t pc_value = config_data == nullptr
                            ? kCoralNPUV2DefaultItcmStartAddress
                            : config_data->itcm_start_address;
    absl::Status pc_write = rv_top_->WriteRegister("pc", pc_value);
    CHECK_OK(pc_write) << "Error writing to pc.";
    is_initialized_ = true;
  }

  absl::Status load_program(const std::string& elf_file) {
    auto load_result = elf_loader_->LoadProgram(elf_file);
    if (!load_result.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to load program '", elf_file,
                       "': ", load_result.status().message()));
    }
    return absl::OkStatus();
  }

  uint32_t get_pc() {
    absl::StatusOr<uint64_t> read_reg_status = rv_top_->ReadRegister("pc");
    CHECK_OK(read_reg_status);
    return static_cast<uint32_t>(read_reg_status.value());
  }

  RiscVTop* rv_top() const { return rv_top_.get(); }

  CoralNPUV2State* rv_state() const { return state_.get(); }

  ElfProgramLoader* elf_loader() const { return elf_loader_.get(); }

  bool cosimulation_halted() const { return cosimulation_halted_; }

  bool is_initialized() const { return is_initialized_; }

 private:
  std::unique_ptr<CoralNPUV2State> CreateState(MemoryInterface* memory,
                                               uint32_t itcm_start_address,
                                               uint32_t itcm_length,
                                               uint32_t initial_misa_value,
                                               uint32_t dtcm_start_address,
                                               uint32_t dtcm_length) {
    auto state = std::make_unique<CoralNPUV2StateFactory>()
                     ->SetItcmRange(itcm_start_address, itcm_length)
                     ->SetInitialMisaValue(initial_misa_value)
                     ->AddLsuAccessRange(dtcm_start_address, dtcm_length)
                     ->Create("CoralNPUV2", RiscVXlen::RV32, memory, nullptr);

    // Make sure the architectural and abi register aliases are added.
    std::string reg_name;
    for (int i = 0; i < 32; i++) {
      reg_name = absl::StrCat(RiscVState::kXregPrefix, i);
      [[maybe_unused]] RV32Register* xreg =
          state->AddRegister<RV32Register>(reg_name);
      CHECK_OK(state->AddRegisterAlias<RV32Register>(reg_name,
                                                     kXRegisterAliases[i]));

      reg_name = absl::StrCat(RiscVState::kFregPrefix, i);
      [[maybe_unused]] RVFpRegister* freg =
          state->AddRegister<RVFpRegister>(reg_name);
      CHECK_OK(state->AddRegisterAlias<RVFpRegister>(reg_name,
                                                     kFRegisterAliases[i]));

      reg_name = absl::StrCat(RiscVState::kVregPrefix, i);
      [[maybe_unused]] RVVectorRegister* vreg =
          state->AddRegister<RVVectorRegister>(reg_name,
                                               kCoralnpuV2VectorByteLength);
    }
    return state;
  }

  std::unique_ptr<RiscVFPState> CreateFPState(CoralNPUV2State* state) {
    return std::make_unique<RiscVFPState>(state->csr_set(), state);
  }

  std::unique_ptr<RiscVVectorState> CreateVectorState(CoralNPUV2State* state) {
    return std::make_unique<RiscVVectorState>(state,
                                              kCoralnpuV2VectorByteLength);
  }

  std::unique_ptr<DecoderInterface> CreateDecoder(CoralNPUV2State* state,
                                                  MemoryInterface* memory) {
    return std::make_unique<CoralNPUV2UserDecoder>(state, memory);
  }

  std::unique_ptr<RiscVTop> CreateRiscVTop(CoralNPUV2State* state,
                                           DecoderInterface* decoder) {
    return std::make_unique<RiscVTop>("CoralNPUPlaceholder", state, decoder);
  }

  const std::unique_ptr<MemoryInterface> memory_;
  const std::unique_ptr<ElfProgramLoader> elf_loader_;
  std::unique_ptr<CoralNPUV2State> state_;
  std::unique_ptr<RiscVFPState> rv_fp_state_;
  std::unique_ptr<RiscVVectorState> rvv_state_;
  std::unique_ptr<DecoderInterface> rv_decoder_;
  std::unique_ptr<RiscVTop> rv_top_;

  // Flag to indicate if the simulation has halted due to an mpause
  // instruction. We add halting logic in the cosim library because single
  // stepping the rv_top object clears the halted flags.
  bool cosimulation_halted_ = false;

  // Flag to indicate if the simulation has been initialized. This is used to
  // support lazy initialization for the case that mpact_config is not called.
  bool is_initialized_ = false;
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

int mpact_config(sim_config_t* config_data) {
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_config: g_mpact_handle is null.";
    return -1;
  }
  if (config_data == nullptr) {
    LOG(ERROR) << "[DPI] mpact_config: config_data is null.";
    return -2;
  }
  g_mpact_handle->Init(config_data);
  return 0;
}

int mpact_load_program(const char* elf_file) {
  if (elf_file == nullptr) {
    LOG(ERROR) << "[DPI] mpact_init: received a null elf program.";
    return -1;
  }
  absl::Status status = g_mpact_handle->load_program(elf_file);
  if (!status.ok()) {
    LOG(ERROR) << "[DPI] Failed to load elf program '" << elf_file
               << "': " << status.message();
    return -1;
  }
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
  if (!g_mpact_handle->is_initialized()) {
    LOG(INFO) << "[DPI] mpact_step: Lazy initialization of g_mpact_handle.";
    g_mpact_handle->Init(nullptr);
  }
  if (g_mpact_handle->cosimulation_halted()) {
    LOG(ERROR) << "[DPI] mpact_step: Can not step when cosimulation is halted.";
    return 2;
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
  if (!g_mpact_handle->is_initialized()) {
    LOG(INFO) << "[DPI] mpact_is_halted: Lazy initialization of "
                 "g_mpact_handle.";
    g_mpact_handle->Init(nullptr);
  }
  LOG(ERROR) << "[DPI] mpact_is_halted: Unimplemented.";
  return false;
}

int mpact_get_vector_register(const char* name, svLogicVecVal* value) {
  if (value == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_vector_register: value is null.";
    return -1;
  }
  if (name == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_vector_register: name is null.";
    return -3;
  }
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_vector_register: g_mpact_handle is null.";
    return -2;
  }
  if (!g_mpact_handle->is_initialized()) {
    LOG(INFO) << "[DPI] mpact_get_vector_register: Lazy initialization of "
                 "g_mpact_handle.";
    g_mpact_handle->Init(nullptr);
  }
  std::string reg_name(name);
  RiscVTop* rv_top = g_mpact_handle->rv_top();
  absl::StatusOr<::mpact::sim::generic::DataBuffer*> vector_db =
      rv_top->GetRegisterDataBuffer(reg_name);
  if (!vector_db.ok()) {
    LOG(ERROR)
        << "[DPI] mpact_get_vector_register: Failed to get register data "
           "buffer: "
        << reg_name;
    return -4;
  }
  absl::Span<uint32_t> vector_data = (*vector_db)->Get<uint32_t>();
  for (int i = 0; i < vector_data.size(); ++i) {
    value[i].aval = vector_data[i];
    value[i].bval = 0;
  }
  return 0;
}

int mpact_get_register(const char* name, uint32_t* value) {
  if (value == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_register: value is null.";
    return -1;
  }
  if (name == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_register: name is null.";
    return -3;
  }
  *value = 0;
  if (g_mpact_handle == nullptr) {
    LOG(ERROR) << "[DPI] mpact_get_register: g_mpact_handle is null.";
    return -2;
  }
  if (!g_mpact_handle->is_initialized()) {
    LOG(INFO) << "[DPI] mpact_get_register: Lazy initialization of "
                 "g_mpact_handle.";
    g_mpact_handle->Init(nullptr);
  }
  std::string reg_name(name);
  RiscVTop* rv_top = g_mpact_handle->rv_top();
  absl::StatusOr<uint64_t> read_reg_status = rv_top->ReadRegister(reg_name);
  if (!read_reg_status.ok()) {
    LOG(ERROR) << "[DPI] mpact_get_register: Failed to read register: "
               << reg_name;
    return -4;
  }
  // CoralNPU V2 is a 32bit system. RiscVTop::ReadRegister outputs 64bit values
  // for both 32bit and 64bit systems. We can safely cast the value to uint32_t.
  *value = static_cast<uint32_t>(*read_reg_status);
  return 0;
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
