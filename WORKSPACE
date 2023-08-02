# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "de8672c59d454182a406fe859de4c87f86116dac320ae366aca60a3258370c1b",
    strip_prefix = "mpact-riscv-eadd26c36777070935ae85ff7f556185085dc188",
    url = "https://github.com/google/mpact-riscv/archive/eadd26c36777070935ae85ff7f556185085dc188.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "2405278024e66328217ab945e385fc4e3e028be19d71448ddeda2579679bd82d",
    strip_prefix = "mpact-sim-daacce22769d45ffb37207ad772751f6a7e03bd4",
    url = "https://github.com/google/mpact-sim/archive/daacce22769d45ffb37207ad772751f6a7e03bd4.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
