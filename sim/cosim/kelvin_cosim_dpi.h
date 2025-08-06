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

// Defines the DPI-C interface for cosimulation.
//
// These C-style functions allow a SystemVerilog testbench to control an
// MPACT-based golden reference model, running it in lock-step with a
// design under test (DUT).
//
// Note: This interface is designed for a single simulator instance and is not
// thread-safe.

#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_COSIM_KELVIN_COSIM_DPI_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_COSIM_KELVIN_COSIM_DPI_H_

#include <cstdint>

#include "external/svdpi_h_file/file/svdpi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the MPACT simulator. This function must be called before any
// other MPACT functions.
// Return 0 on success, non-zero on failure.
int mpact_init();

// Reset the MPACT simulator's architectural state.
// Return 0 on success, non-zero on failure.
int mpact_reset();

// Step the MPACT simulator by executing a single provided instruction.
// The instruction is provided as a SystemVerilog datatype - svLogicVecVal*.
// Return 0 on success, non-zero on failure.
int mpact_step(const svLogicVecVal* instruction);

// Check if the MPACT simulator has reached a halted state. Some tests may
// require the simulator to be halted before checking the results.
// Currently unimplemented and always returns false.
bool mpact_is_halted();

// Return the current value of the Program Counter (PC).
// On error, returns 0 and logs an error.
uint32_t mpact_get_pc();

// Return the value of the specified GPR. GPRs are selected by their index,
// where 0 is x0, 1 is x1, and so on.
// On error, returns 0 and logs an error.
uint32_t mpact_get_gpr(uint32_t index);

// Return the value of the specified CSR. CSRs are selected by their address.
// On error, returns 0 and logs an error.
uint32_t mpact_get_csr(uint32_t address);

// Finalize and clean up MPACT simulator resources.
// Return 0 on success, non-zero on failure.
int mpact_fini();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_COSIM_KELVIN_COSIM_DPI_H_
