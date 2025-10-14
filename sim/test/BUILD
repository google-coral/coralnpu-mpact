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

# Unit tests for CoralNPU simulator.




exports_files(
    srcs = [
        "testfiles/coralnpu_ebreak.elf",
        "testfiles/coralnpu_perf_counters.elf",
        "testfiles/coralnpu_vldvst.elf",
        "testfiles/hello_world_mpause.bin",
        "testfiles/hello_world_mpause.elf",
        "testfiles/hello_world_rv32imf.elf",
        "testfiles/rv32i.elf",
        "testfiles/rv32m.elf",
        "testfiles/rv32soft_fp.elf",
        "testfiles/rv32uf_fadd.elf",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "coralnpu_encoding_test",
    size = "small",
    srcs = ["coralnpu_encoding_test.cc"],
    deps = [
        "//sim:coralnpu_decoder",
        "//sim:coralnpu_isa",
        "//sim:coralnpu_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_decoder_test",
    size = "small",
    srcs = ["coralnpu_decoder_test.cc"],
    data = [
        "testfiles/hello_world_rv32imf.elf",
    ],
    deps = [
        "//sim:coralnpu_decoder",
        "//sim:coralnpu_state",
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
    name = "coralnpu_top_test",
    size = "small",
    srcs = ["coralnpu_top_test.cc"],
    data = [
        "testfiles/coralnpu_ebreak.elf",
        "testfiles/coralnpu_perf_counters.elf",
        "testfiles/coralnpu_vldvst.elf",
        "testfiles/hello_world_mpause.bin",
        "testfiles/hello_world_mpause.elf",
        "testfiles/hello_world_rv32imf.elf",
        "testfiles/rv32i.elf",
        "testfiles/rv32m.elf",
        "testfiles/rv32soft_fp.elf",
        "testfiles/rv32uf_fadd.elf",
    ],
    deps = [
        "//sim:coralnpu_state",
        "//sim:coralnpu_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_library(
    name = "coralnpu_vector_instructions_test_base",
    testonly = True,
    hdrs = ["coralnpu_vector_instructions_test_base.h"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        "//sim:coralnpu_state",
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
    name = "coralnpu_vector_instructions_test",
    srcs = ["coralnpu_vector_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":coralnpu_vector_instructions_test_base",
        "//sim:coralnpu_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "coralnpu_vector_convolution_instructions_test",
    srcs = [
        "coralnpu_vector_convolution_instructions_test.cc",
        "testfiles/coralnpu_vector_convolution_testdata.h",
    ],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":coralnpu_vector_instructions_test_base",
        "//sim:coralnpu_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "coralnpu_vector_memory_instructions_test",
    srcs = ["coralnpu_vector_memory_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":coralnpu_vector_instructions_test_base",
        "//sim:coralnpu_instructions",
        "//sim:coralnpu_state",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "coralnpu_log_instructions_test",
    srcs = ["coralnpu_log_instructions_test.cc"],
    copts = [
        "-Werror",
        "-Wvla-extension",
    ],
    deps = [
        ":coralnpu_vector_instructions_test_base",
        "//sim:coralnpu_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "coralnpu_v2_user_decoder_test",
    srcs = ["coralnpu_v2_user_decoder_test.cc"],
    copts = ["-Werror"],
    deps = [
        "//sim:coralnpu_v2_isa",
        "//sim:coralnpu_v2_state",
        "//sim:coralnpu_v2_user_decoder",
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
    name = "coralnpu_cosim_dpi_wrapper_test",
    srcs = ["coralnpu_cosim_dpi_wrapper_test.cc"],
    copts = ["-Werror"],
    deps = [
        "//sim/cosim:coralnpu_cosim_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
