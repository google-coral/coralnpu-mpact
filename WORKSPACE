# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "7c2ddf3c1980d138f4b9af539fba97e7aa8d5a812a9e15d62e40ccbbdae71a75",
    strip_prefix = "mpact-riscv-40bc408eda29dbb5ebf023d455bd841b2dae73c4",
    url = "https://github.com/google/mpact-riscv/archive/40bc408eda29dbb5ebf023d455bd841b2dae73c4.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "8bad24dffe9996762a4db6e074e38ba454ec2ae113c3cb849aa7d8250827d37b",
    strip_prefix = "mpact-sim-fc14a25478d2b8a15cc74451798471fe5d8ae5d2",
    url = "https://github.com/google/mpact-sim/archive/fc14a25478d2b8a15cc74451798471fe5d8ae5d2.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
