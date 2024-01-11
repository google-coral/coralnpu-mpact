# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "68c65bae71654ce59f3bd047af9083840b4b463aee7a17ba7d75c84322534347",
    strip_prefix = "mpact-riscv-a4c534a97b107f3734e04c8e32d182e49d400ac3",
    url = "https://github.com/google/mpact-riscv/archive/a4c534a97b107f3734e04c8e32d182e49d400ac3.tar.gz",
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
