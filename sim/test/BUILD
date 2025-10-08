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

# Unit tests for kelvin simulator.




exports_files(
    srcs = [
        "testfiles/hello_world_mpause.bin",
        "testfiles/hello_world_mpause.elf",
        "testfiles/hello_world_rv32imf.elf",
        "testfiles/kelvin_ebreak.elf",
        "testfiles/kelvin_perf_counters.elf",
        "testfiles/kelvin_vldvst.elf",
        "testfiles/rv32i.elf",
        "testfiles/rv32m.elf",
        "testfiles/rv32soft_fp.elf",
        "testfiles/rv32uf_fadd.elf",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "kelvin_encoding_test",
    size = "small",
    srcs = [
        "kelvin_encoding_test.cc",
    ],
    deps = [
        "//sim:kelvin_decoder",
        "//sim:kelvin_isa",
        "//sim:kelvin_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "kelvin_decoder_test",
    size = "small",
    srcs = [
        "kelvin_decoder_test.cc",
    ],
    data = [
        "testfiles/hello_world_rv32imf.elf",
    ],
    deps = [
        "//sim:kelvin_decoder",
        "//sim:kelvin_state",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_test(
    name = "kelvin_top_test",
    size = "small",
    srcs = [
        "kelvin_top_test.cc",
    ],
    data = [
        "testfiles/hello_world_mpause.bin",
        "testfiles/hello_world_mpause.elf",
        "testfiles/hello_world_rv32imf.elf",
        "testfiles/kelvin_ebreak.elf",
        "testfiles/kelvin_perf_counters.elf",
        "testfiles/kelvin_vldvst.elf",
        "testfiles/rv32i.elf",
        "testfiles/rv32m.elf",
        "testfiles/rv32soft_fp.elf",
        "testfiles/rv32uf_fadd.elf",
    ],
    deps = [
        "//sim:kelvin_state",
        "//sim:kelvin_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_library(
    name = "kelvin_vector_instructions_test_base",
    testonly = True,
    hdrs = ["kelvin_vector_instructions_test_base.h"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        "//sim:kelvin_state",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "kelvin_vector_instructions_test",
    srcs = ["kelvin_vector_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":kelvin_vector_instructions_test_base",
        "//sim:kelvin_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "kelvin_vector_convolution_instructions_test",
    srcs = [
        "kelvin_vector_convolution_instructions_test.cc",
        "testfiles/kelvin_vector_convolution_testdata.h",
    ],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":kelvin_vector_instructions_test_base",
        "//sim:kelvin_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "kelvin_vector_memory_instructions_test",
    srcs = ["kelvin_vector_memory_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":kelvin_vector_instructions_test_base",
        "//sim:kelvin_instructions",
        "//sim:kelvin_state",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "kelvin_log_instructions_test",
    srcs = ["kelvin_log_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":kelvin_vector_instructions_test_base",
        "//sim:kelvin_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "kelvin_v2_user_decoder_test",
    srcs = ["kelvin_v2_user_decoder_test.cc"],
    copts = ["-Werror"],
    deps = [
        "//sim:kelvin_v2_isa",
        "//sim:kelvin_v2_state",
        "//sim:kelvin_v2_user_decoder",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "kelvin_cosim_dpi_wrapper_test",
    srcs = ["kelvin_cosim_dpi_wrapper_test.cc"],
    copts = ["-Werror"],
    deps = [
        "//sim/cosim:kelvin_cosim_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
