# Copyright 2023 Google LLC
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

# This project contains a small set of tests to test the functionality of
# the MPACT-Sim/Renode interface.

cc_test(
    name = "renode_mpact_test",
    size = "small",
    srcs = ["renode_mpact_test.cc"],
    data = [
        "//sim/test:testfiles/hello_world_mpause.bin",
        "//sim/test:testfiles/hello_world_mpause.elf",
        "//sim/test:testfiles/kelvin_ebreak.elf",
    ],
    deps = [
        "//sim/renode:kelvin_renode",
        "//sim/renode:renode_mpact",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_debug_info",
    ],
)

cc_test(
    name = "kelvin_renode_test",
    size = "small",
    srcs = ["kelvin_renode_test.cc"],
    data = [
        "//sim/test:testfiles/hello_world_mpause.bin",
        "//sim/test:testfiles/hello_world_mpause.elf",
        "//sim/test:testfiles/kelvin_ebreak.elf",
    ],
    deps = [
        "//sim:kelvin_top",
        "//sim/renode:kelvin_renode",
        "//sim/renode:renode_debug_interface",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_debug_info",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)
