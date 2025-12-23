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




cc_library(
    name = "coralnpu_simulator_headers",
    hdrs = [
        "coralnpu_simulator.h",
    ],
)

cc_library(
    name = "mpact_simulator",
    srcs = ["mpact_simulator.cc"],
    linkstatic = True,
    deps = [
        ":coralnpu_simulator_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-riscv//riscv:riscv32g_vec_decoder",
        "@com_google_mpact-riscv//riscv:riscv_fp_state",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_top",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
    alwayslink = True,
)

# generate libraries for external projects
cc_binary(
    name = "libcoralnpu_simulator_mpact.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [":mpact_simulator"],
)
