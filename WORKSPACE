# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "76e2b701cb93abaebbdd0fa43771cee20db84ca6eb5140f12f7c873d4743a9ac",
    strip_prefix = "mpact-riscv-e33928ebc238f4dd526719ee398869ef8c353b53",
    url = "https://github.com/google/mpact-riscv/archive/e33928ebc238f4dd526719ee398869ef8c353b53.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "f6e97fad35d9e218e4f9e0e8737ba0f5163e24393f502b47ca7fc5a9a7924d20",
    strip_prefix = "mpact-sim-7ea1334bedf7f6f4a58aa3b2636e9613d8704f39",
    url = "https://github.com/google/mpact-sim/archive/7ea1334bedf7f6f4a58aa3b2636e9613d8704f39.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
