# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "2f51e0d90f6adf639fe6b23bee6dd86283326d86f8952e60cb26abeccc1e939b",
    strip_prefix = "mpact-riscv-641f0748c6b17aa8789d455e82fb8f1296f79d26",
    url = "https://github.com/google/mpact-riscv/archive/641f0748c6b17aa8789d455e82fb8f1296f79d26.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "f39cbbe26df267a6d0bd64756e46c8606d08b3f78cb407e3943862f79237bbf8",
    strip_prefix = "mpact-sim-0b5f0d18c69434e63f0ca20465067b5ef61f5046",
    url = "https://github.com/google/mpact-sim/archive/0b5f0d18c69434e63f0ca20465067b5ef61f5046.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
