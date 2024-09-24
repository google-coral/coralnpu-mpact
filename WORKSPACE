# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    integrity = "sha256-1UtiuOMLKJK5f1mXiWGfb4Lc1n1kWmSQNNZb80XiLY4=",
    strip_prefix = "mpact-riscv-3ed17ec6c5d9cf5fa35ea7100bfa9ae7799fa0d6",
    url = "https://github.com/google/mpact-riscv/archive/3ed17ec6c5d9cf5fa35ea7100bfa9ae7799fa0d6.tar.gz",
)

load("@com_google_mpact-riscv//:repos.bzl", "mpact_riscv_repos")
mpact_riscv_repos()

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")
mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")
mpact_sim_deps()

load("@com_google_mpact-sim//:protobuf_deps.bzl", "mpact_sim_protobuf_deps")
mpact_sim_protobuf_deps()
