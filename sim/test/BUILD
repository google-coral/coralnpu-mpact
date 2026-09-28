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

# load("@rules_cc//cc:defs.bzl", "cc_library")
# load("@rules_cc//cc:defs.bzl", "cc_test")

exports_files(
    srcs = [
        "testfiles/coralnpu_ebreak.elf",
        "testfiles/coralnpu_perf_counters.elf",
        "testfiles/coralnpu_v2_rvv_add_intrinsic.disassm",
        "testfiles/coralnpu_v2_rvv_add_intrinsic.elf",
        "testfiles/coralnpu_vldvst.elf",
        "testfiles/disassm_to_header.awk",
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
        "@mpact-sim//mpact/sim/generic:arch_state",
        "@mpact-sim//mpact/sim/util/memory",
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@com_github_serge1_elfio//:elfio",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/util/memory",
        "@mpact-sim//mpact/sim/util/program_loader:elf_loader",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@mpact-sim//mpact/sim/generic:core_debug_interface",
        "@mpact-sim//mpact/sim/util/memory",
        "@mpact-sim//mpact/sim/util/program_loader:elf_loader",
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
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:arch_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/generic:type_helpers",
        "@mpact-sim//mpact/sim/util/memory",
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
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@mpact-sim//mpact/sim/generic:instruction",
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
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:instruction",
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
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/random:distributions",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@mpact-sim//mpact/sim/generic:instruction",
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
        "@abseil-cpp//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_test(
    name = "coralnpu_v2_state_test",
    size = "small",
    srcs = ["coralnpu_v2_state_test.cc"],
    deps = [
        "//sim:coralnpu_v2_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_m4_state_test",
    size = "small",
    srcs = ["coralnpu_m4_state_test.cc"],
    deps = [
        "//sim:coralnpu_m4_state",
        "@abseil-cpp//absl/log:check",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_zvt_state",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_m4_encoding_test",
    size = "small",
    srcs = ["coralnpu_m4_encoding_test.cc"],
    deps = [
        "//sim:coralnpu_m4_isa",
        "//sim:coralnpu_m4_state",
        "//sim:coralnpu_m4_user_decoder",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:type_helpers",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_m4_config_test",
    size = "small",
    srcs = ["coralnpu_m4_config_test.cc"],
    deps = [
        "//sim:coralnpu_m4_state",
        "//sim:coralnpu_m4_user_decoder",
        "//sim:coralnpu_v2_state",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_zvt_state",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_m4_ops_test",
    size = "small",
    srcs = ["coralnpu_m4_ops_test.cc"],
    deps = [
        "//sim:coralnpu_m4_state",
        "//sim:coralnpu_m4_user_decoder",
        "//sim:coralnpu_v2_state",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_zvt_state",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/util/memory",
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
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/generic:type_helpers",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

genrule(
    name = "align_test_generated_h",
    srcs = ["testfiles/align_test.disassm"],
    outs = ["align_test_generated.h"],
    cmd = "awk -v header_guard=SIM_TEST_ALIGN_TEST_GENERATED_H_ -v nested_namespace=align_test -f $(location testfiles/disassm_to_header.awk) $< > $@",
    tools = ["testfiles/disassm_to_header.awk"],
)

cc_library(
    name = "align_test_generated",
    hdrs = ["align_test_generated.h"],
)

genrule(
    name = "frm_test_generated_h",
    srcs = ["testfiles/frm_test.disassm"],
    outs = ["frm_test_generated.h"],
    cmd = "awk -v header_guard=SIM_TEST_FRM_TEST_GENERATED_H_ -v nested_namespace=frm_test -f $(location testfiles/disassm_to_header.awk) $< > $@",
    tools = ["testfiles/disassm_to_header.awk"],
)

cc_library(
    name = "frm_test_generated",
    hdrs = ["frm_test_generated.h"],
)

cc_test(
    name = "coralnpu_cosim_dpi_wrapper_test",
    srcs = ["coralnpu_cosim_dpi_wrapper_test.cc"],
    copts = ["-Werror"],
    data = ["testfiles/coralnpu_v2_rvv_add_intrinsic.elf"],
    deps = [
        ":align_test_generated",
        ":frm_test_generated",
        "//sim:coralnpu_architecture",
        "//sim/cosim:coralnpu_cosim_lib",
        "@com_google_googletest//:gtest_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings:str_format",
        # INTERNAL_VERILATOR_DEPENDENCY,
    ],
)

cc_test(
    name = "coralnpu_v2_instructions_test",
    srcs = ["coralnpu_v2_instructions_test.cc"],
    copts = ["-Werror"],
    deps = [
        "//sim:coralnpu_v2_instructions",
        "//sim:coralnpu_v2_state",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_g",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_v",
        "@mpact-sim//mpact/sim/generic:arch_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/generic:type_helpers",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

genrule(
    name = "coralnpu_v2_rvv_add_intrinsic_generated_h",
    srcs = ["testfiles/coralnpu_v2_rvv_add_intrinsic.disassm"],
    outs = ["coralnpu_v2_rvv_add_intrinsic_generated.h"],
    cmd = "awk -v header_guard=SIM_TEST_CORALNPU_V2_RVV_ADD_INTRINSIC_GENERATED_H_ -v nested_namespace=coralnpu_v2_rvv_add_intrinsic -f $(location testfiles/disassm_to_header.awk) $< > $@",
    tools = ["testfiles/disassm_to_header.awk"],
)

cc_library(
    name = "coralnpu_v2_rvv_add_intrinsic_generated",
    hdrs = ["coralnpu_v2_rvv_add_intrinsic_generated.h"],
)

cc_test(
    name = "coralnpu_m3_encoding_test",
    size = "small",
    srcs = ["coralnpu_m3_encoding_test.cc"],
    deps = [
        "//sim:coralnpu_m3_isa",
        "//sim:coralnpu_m3_user_decoder",
        "//sim:coralnpu_v2_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_m3_getters_test",
    size = "small",
    srcs = ["coralnpu_m3_getters_test.cc"],
    deps = [
        "//sim:coralnpu_m3_isa",
        "//sim:coralnpu_m3_user_decoder",
        "//sim:coralnpu_v2_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

cc_test(
    name = "coralnpu_simulator_test",
    srcs = ["coralnpu_simulator_test.cc"],
    copts = ["-Werror"],
    data = [
        "testfiles/coralnpu_ebreak.elf",
        "testfiles/coralnpu_v2_rvv_add_intrinsic.elf",
    ],
    deps = [
        ":coralnpu_v2_rvv_add_intrinsic_generated",
        "//sim:coralnpu_architecture",
        "//sim:coralnpu_m3_user_decoder",
        "//sim:coralnpu_m4_state",
        "//sim:coralnpu_m4_user_decoder",
        "//sim:coralnpu_simulator",
        "//sim:coralnpu_v2_state",
        "//sim:coralnpu_v2_user_decoder",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/generic:core_debug_interface",
        "@mpact-sim//mpact/sim/generic:instruction",
        "@mpact-sim//mpact/sim/generic:type_helpers",
        "@mpact-sim//mpact/sim/util/memory",
        "@mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_test(
    name = "coralnpu_zfbfmin_test",
    size = "small",
    srcs = ["coralnpu_zfbfmin_test.cc"],
    deps = [
        ":coralnpu_zfbfmin_generated",
        "//sim:coralnpu_architecture",
        "//sim:coralnpu_simulator",
        "//sim:coralnpu_v2_state",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_fp_state",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@mpact-sim//mpact/sim/generic:core",
        "@mpact-sim//mpact/sim/util/memory",
    ],
)

genrule(
    name = "coralnpu_zfbfmin_generated_h",
    srcs = ["testfiles/coralnpu_zfbfmin.disassm"],
    outs = ["coralnpu_zfbfmin_generated.h"],
    cmd = "awk -v header_guard=SIM_TEST_CORALNPU_ZFBFMIN_GENERATED_H_ -v nested_namespace=coralnpu_zfbfmin -f $(location testfiles/disassm_to_header.awk) $< > $@",
    tools = ["testfiles/disassm_to_header.awk"],
)

cc_library(
    name = "coralnpu_zfbfmin_generated",
    hdrs = ["coralnpu_zfbfmin_generated.h"],
)

cc_test(
    name = "coralnpu_xstatus_test",
    size = "small",
    srcs = ["coralnpu_xstatus_test.cc"],
    deps = [
        "//sim:coralnpu_m4_state",
        "@com_google_googletest//:gtest_main",
        "@com_google_mpact-riscv//riscv:riscv_state",
    ],
)
