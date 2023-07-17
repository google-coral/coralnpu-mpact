# Build rules for Kelvin simulator code.
load("@com_google_mpact-sim//mpact/sim/decoder:mpact_sim_isa.bzl", "mpact_bin_fmt_decoder", "mpact_isa_decoder")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kelvin_state",
    srcs = [
        "kelvin_state.cc",
    ],
    hdrs = [
        "kelvin_state.h",
    ],
    copts = [
        "-O3",
        "-ffp-model=strict",
        "-fprotect-parens",
    ],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-riscv//riscv:riscv_state",
    ],
)

cc_library(
    name = "kelvin_instructions",
    srcs = [
        "kelvin_instructions.cc",
        "kelvin_vector_instructions.cc",
        "kelvin_vector_memory_instructions.cc",
    ],
    hdrs = [
        "kelvin_instructions.h",
        "kelvin_vector_instructions.h",
        "kelvin_vector_memory_instructions.h",
    ],
    copts = ["-O3"],
    deps = [
        ":kelvin_state",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
    ],
)

mpact_isa_decoder(
    name = "kelvin_isa",
    src = "kelvin.isa",
    includes = [],
    isa_name = "Kelvin",
    deps = [
        ":kelvin_instructions",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_mpact-riscv//riscv:riscv_g",
    ],
)

mpact_bin_fmt_decoder(
    name = "kelvin_bin_fmt",
    src = "kelvin.bin_fmt",
    decoder_name = "Kelvin",
    includes = [],
    deps = [
        ":kelvin_isa",
    ],
)

cc_library(
    name = "kelvin_decoder",
    srcs = [
        "decoder.cc",
        "kelvin_encoding.cc",
    ],
    hdrs = [
        "decoder.h",
        "kelvin_encoding.h",
    ],
    copts = ["-O3"],
    deps = [
        ":kelvin_bin_fmt",
        ":kelvin_isa",
        ":kelvin_state",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "kelvin_top",
    srcs = [
        "kelvin_top.cc",
    ],
    hdrs = [
        "kelvin_top.h",
    ],
    copts = ["-O3"],
    deps = [
        ":kelvin_decoder",
        ":kelvin_isa",
        ":kelvin_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_mpact-riscv//riscv:riscv_arm_semihost",
        "@com_google_mpact-riscv//riscv:riscv_breakpoint",
        "@com_google_mpact-riscv//riscv:riscv_state",
        "@com_google_mpact-sim//mpact/sim/generic:arch_state",
        "@com_google_mpact-sim//mpact/sim/generic:component",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:decode_cache",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_binary(
    name = "kelvin_sim",
    srcs = ["kelvin_sim.cc"],
    copts = ["-O3"],
    data = [
        "//sim/test:testfiles/hello_world_rv32imf.elf",
        "//sim/test:testfiles/rv32i.elf",
        "//sim/test:testfiles/rv32m.elf",
        "//sim/test:testfiles/rv32soft_fp.elf",
    ],
    deps = [
        ":kelvin_top",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/log",
        "@com_google_mpact-riscv//riscv:debug_command_shell",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
)
