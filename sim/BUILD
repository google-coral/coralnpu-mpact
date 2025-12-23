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

# Build rules for CoralNPU simulator code.
load("@com_google_mpact-sim//mpact/sim/decoder:mpact_sim_isa.bzl", "mpact_bin_fmt_decoder", "mpact_isa_decoder")



package(default_visibility = ["//visibility:public"])

cc_library(
    name = "coralnpu_state",
    srcs = [
        "coralnpu_state.cc",
    ],
    hdrs = [
        "coralnpu_state.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
        "-fprotect-parens",
    ],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "coralnpu_instructions",
    srcs = [
        "coralnpu_instructions.cc",
        "coralnpu_vector_convolution_instructions.cc",
        "coralnpu_vector_instructions.cc",
        "coralnpu_vector_memory_instructions.cc",
    ],
    hdrs = [
        "coralnpu_instructions.h",
        "coralnpu_vector_convolution_instructions.h",
        "coralnpu_vector_instructions.h",
        "coralnpu_vector_memory_instructions.h",
    ],
    copts = ["-O3"],
    deps = [
        ":coralnpu_state",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

mpact_isa_decoder(
    name = "coralnpu_isa",
    src = "coralnpu.isa",
    includes = [
        "coralnpu_arith.isa",
        "coralnpu_base.isa",
        "coralnpu_memory.isa",
        "coralnpu_mul.isa",
        "coralnpu_shift.isa",
        "@com_google_mpact-riscv//riscv:riscv32zb.isa",
    ],
    isa_name = "CoralNPU",
    deps = [
        ":coralnpu_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_mpact-riscv//riscv:riscv_bitmanip_instructions",
        "@com_google_mpact-riscv//riscv:riscv_g",
    ],
)

mpact_bin_fmt_decoder(
    name = "coralnpu_bin_fmt",
    src = "coralnpu.bin_fmt",
    decoder_name = "CoralNPU",
    includes = [
        "coralnpu_arith.bin_fmt",
        "coralnpu_base.bin_fmt",
        "coralnpu_conv.bin_fmt",
        "coralnpu_format.bin_fmt",
        "coralnpu_memory.bin_fmt",
        "coralnpu_mul.bin_fmt",
        "coralnpu_shift.bin_fmt",
        "@com_google_mpact-riscv//riscv:riscv32zb.bin_fmt",
    ],
    deps = [
        ":coralnpu_isa",
    ],
)

cc_library(
    name = "coralnpu_decoder",
    srcs = [
        "coralnpu_encoding.cc",
        "decoder.cc",
    ],
    hdrs = [
        "coralnpu_encoding.h",
        "decoder.h",
    ],
    copts = ["-O3"],
    deps = [
        ":coralnpu_bin_fmt",
        ":coralnpu_isa",
        ":coralnpu_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_generic_decoder",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "coralnpu_top",
    srcs = [
        "coralnpu_action_point_memory_interface.cc",
        "coralnpu_top.cc",
    ],
    hdrs = [
        "coralnpu_action_point_memory_interface.h",
        "coralnpu_top.h",
    ],
    copts = ["-O3"],
    deps = [
        ":coralnpu_decoder",
        ":coralnpu_isa",
        ":coralnpu_state",
        "//sim/proto:trace_cc_proto",
        "//sim/renode:coralnpu_renode_memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_mpact-riscv//riscv:riscv_arm_semihost",
        "@com_google_mpact-riscv//riscv:riscv_fp_state",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:action_points",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:counters",
        "@com_google_mpact-sim//mpact/sim/generic:decode_cache",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_binary(
    name = "coralnpu_sim",
    srcs = ["coralnpu_sim.cc"],
    copts = ["-O3"],
    data = [
        "//sim/test:testfiles/hello_world_rv32imf.elf",
        "//sim/test:testfiles/rv32i.elf",
        "//sim/test:testfiles/rv32m.elf",
        "//sim/test:testfiles/rv32soft_fp.elf",
    ],
    deps = [
        ":coralnpu_state",
        ":coralnpu_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_mpact-riscv//riscv:debug_command_shell",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "coralnpu_v2_state",
    srcs = [
        "coralnpu_v2_state.cc",
    ],
    hdrs = [
        "coralnpu_v2_state.h",
    ],
    copts = [
        "-O3",
    ],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

mpact_isa_decoder(
    name = "coralnpu_v2_isa",
    src = "coralnpu_v2.isa",
    includes = [
        "@com_google_mpact-riscv//riscv:riscv32g.isa",
        "@com_google_mpact-riscv//riscv:riscv32zb.isa",
        "@com_google_mpact-riscv//riscv:riscv_vector.isa",
    ],
    isa_name = "CoralNPUV2",
    deps = [
        ":coralnpu_v2_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_mpact-riscv//riscv:riscv_bitmanip_instructions",
        "@com_google_mpact-riscv//riscv:riscv_g",
        "@com_google_mpact-riscv//riscv:riscv_v",
    ],
)

mpact_bin_fmt_decoder(
    name = "coralnpu_v2_bin_fmt",
    src = "coralnpu_v2.bin_fmt",
    decoder_name = "CoralNPUV2",
    includes = [
        "@com_google_mpact-riscv//riscv:riscv32g.bin_fmt",
        "@com_google_mpact-riscv//riscv:riscv_format16.bin_fmt",
        "@com_google_mpact-riscv//riscv:riscv_format32.bin_fmt",
        "@com_google_mpact-riscv//riscv:riscv_vector.bin_fmt",
    ],
    deps = [
        ":coralnpu_v2_isa",
    ],
)

cc_library(
    name = "coralnpu_v2_user_decoder",
    srcs = [
        "coralnpu_v2_encoding.cc",
        "coralnpu_v2_user_decoder.cc",
    ],
    hdrs = [
        "coralnpu_v2_encoding.h",
        "coralnpu_v2_user_decoder.h",
    ],
    copts = [
        "-O3",
    ],
    deps = [
        ":coralnpu_v2_bin_fmt",
        ":coralnpu_v2_getters",
        ":coralnpu_v2_isa",
        ":coralnpu_v2_state",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_mpact-riscv//riscv:riscv_encoding_common",
        "@com_google_mpact-riscv//riscv:riscv_generic_decoder",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:program_error",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "coralnpu_v2_getters",
    hdrs = ["coralnpu_v2_getters.h"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-riscv//riscv:riscv_encoding_common",
        "@com_google_mpact-riscv//riscv:riscv_getters",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
)

cc_library(
    name = "coralnpu_v2_instructions",
    srcs = [
        "coralnpu_v2_instructions.cc",
    ],
    hdrs = [
        "coralnpu_v2_instructions.h",
    ],
    copts = ["-O3"],
    deps = [
        ":coralnpu_v2_state",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_mpact-riscv//riscv:riscv_g",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

cc_library(
    name = "coralnpu_v2_simulator",
    srcs = ["coralnpu_v2_simulator.cc"],
    hdrs = ["coralnpu_v2_simulator.h"],
    copts = ["-O3"],
    deps = [
        ":coralnpu_v2_state",
        ":coralnpu_v2_user_decoder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_mpact-riscv//riscv:debug_command_shell",
        "@com_google_mpact-riscv//riscv:riscv_fp_state",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-riscv//riscv:riscv_top",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)

cc_binary(
    name = "coralnpu_v2_sim",
    srcs = ["coralnpu_v2_sim.cc"],
    copts = ["-O3"],
    deps = [
        ":coralnpu_v2_simulator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)
