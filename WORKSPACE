# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "b2b3ce1354d2da48ee195b72b682ecffeb0844b1fbd4f8e83a429d8b21194e61",
    strip_prefix = "mpact-riscv-a21d14041b9f034519e2f9fa359f2902b433cf23",
    url = "https://github.com/google/mpact-riscv/archive/a21d14041b9f034519e2f9fa359f2902b433cf23.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "240e6fa1cba9f26dd5e5343eeff6cc2f8a890cb1ead63c8f7a95323cb88b6593",
    strip_prefix = "mpact-sim-d3977cd11e560fe19c7ad5ee6b269d806ca6c768",
    url = "https://github.com/google/mpact-sim/archive/d3977cd11e560fe19c7ad5ee6b269d806ca6c768.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
