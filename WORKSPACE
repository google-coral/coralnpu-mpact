# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "244236ecf63f812eedf4e1c80e79276374d4c8a9222860220706522edf093fc8",
    strip_prefix = "mpact-riscv-8b0c6b7fa4f48d6dba99c1a4abba3bb548577cad",
    url = "https://github.com/google/mpact-riscv/archive/8b0c6b7fa4f48d6dba99c1a4abba3bb548577cad.tar.gz",
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
