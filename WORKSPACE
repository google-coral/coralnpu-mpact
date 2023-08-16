# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "b293e55147a2ce2067a25032cef6b27f6450f3ef383e02d5607d53d910208e68",
    strip_prefix = "mpact-riscv-f66002025ec6c585839568112cabc1ce19d1b919",
    url = "https://github.com/google/mpact-riscv/archive/f66002025ec6c585839568112cabc1ce19d1b919.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "8b35b5f172f241fd207f5f54d81aacab1617dea16607b36e3417ef47f2366afc",
    strip_prefix = "mpact-sim-c4b68d7bdef36b9eb2e85f8dc22ccc4eb27ba4c5",
    url = "https://github.com/google/mpact-sim/archive/c4b68d7bdef36b9eb2e85f8dc22ccc4eb27ba4c5.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
