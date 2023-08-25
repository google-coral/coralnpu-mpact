# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "9b617e364fb64b49f2ddd83c9eb0c012d3afee1569ad85262e0ccaf8a29ae760",
    strip_prefix = "mpact-riscv-7b5ba3433b2c39752ca9a35d1b1ce48a7fec8722",
    url = "https://github.com/google/mpact-riscv/archive/7b5ba3433b2c39752ca9a35d1b1ce48a7fec8722.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "com_google_mpact-sim",
    sha256 = "1f0e6ea27b0487a5d997f85efaebdf60a1f1dbb478c30b25d3c6b41e9d4b4028",
    strip_prefix = "mpact-sim-e1b7b0adeb53875908995674a8555b68c4821903",
    url = "https://github.com/google/mpact-sim/archive/e1b7b0adeb53875908995674a8555b68c4821903.tar.gz",
)

load("@com_google_mpact-sim//:repos.bzl", "mpact_sim_repos")

mpact_sim_repos()

load("@com_google_mpact-sim//:deps.bzl", "mpact_sim_deps")

mpact_sim_deps()
