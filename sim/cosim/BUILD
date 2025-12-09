# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This BUILD file defines the co-simulation library that bridges the
# UVM testbench with the MPACT-Sim model.



load("@cc_static_library_external//file:cc_static_libarary.bzl", "cc_static_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "coralnpu_cosim_lib",
    srcs = [
        "coralnpu_cosim_dpi_wrapper.cc",
    ],
    hdrs = [
        "coralnpu_cosim_dpi.h",
        "@svdpi_h_file//file",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sim:coralnpu_v2_state",
        "//sim:coralnpu_v2_user_decoder",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_fp_state",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_top",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
    alwayslink = True,
)

# This rule generates the shared library that is loaded and used by an external UVM testbench to
# instantiate and control coralnpu sim
cc_binary(
    name = "cosim_coralnpu",
    linkshared = True,
    deps = [
        ":coralnpu_cosim_lib",
    ],
)

cc_static_library(
    name = "coralnpu_cosim_lib_static",
    deps = [
        ":coralnpu_cosim_lib",
    ],
)
