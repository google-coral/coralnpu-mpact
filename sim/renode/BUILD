# Build Renode interface for kelvin

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "renode_mpact",
    srcs = [
        "renode_mpact.cc",
    ],
    hdrs = [
        "renode_mpact.h",
    ],
    deps = [
        ":renode_debug_interface",
        "//sim:kelvin_top",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
        "@com_google_mpact-sim//mpact/sim/util/program_loader:elf_loader",
    ],
    alwayslink = True,
)

cc_library(
    name = "renode_debug_interface",
    hdrs = ["renode_debug_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
    ],
)

cc_library(
    name = "kelvin_renode_memory",
    srcs = [
        "kelvin_renode_memory.cc",
    ],
    hdrs = [
        "kelvin_renode_memory.h",
    ],
    deps = [
        "//sim:kelvin_state",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/util/memory",
    ],
)

cc_library(
    name = "kelvin_renode",
    srcs = [
        "kelvin_renode.cc",
        "kelvin_renode_reigster_info.cc",
    ],
    hdrs = [
        "kelvin_renode.h",
        "kelvin_renode_register_info.h",
    ],
    deps = [
        ":renode_debug_interface",
        "//sim:kelvin_top",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_mpact-riscv//riscv:riscv_debug_info",
        "@com_google_mpact-sim//mpact/sim/generic:core",
        "@com_google_mpact-sim//mpact/sim/generic:core_debug_interface",
        "@com_google_mpact-sim//mpact/sim/generic:instruction",
        "@com_google_mpact-sim//mpact/sim/generic:type_helpers",
    ],
    alwayslink = True,
)

# This rule generates the shared library that is loaded and used by Renode to instantiate and
# control kelvin sim
cc_binary(
    name = "renode_kelvin",
    linkshared = True,
    deps = [
        ":kelvin_renode",
        ":renode_mpact",
    ],
)
