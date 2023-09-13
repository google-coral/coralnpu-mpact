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
