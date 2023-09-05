# Unit tests for kelvin simulator.

exports_files(
    srcs = [
        "testfiles/hello_world_mpause.bin",
        "testfiles/hello_world_mpause.elf",
        "testfiles/hello_world_rv32imf.elf",
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/functional:bind_front",
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
