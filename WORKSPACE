# Setup bazel repository.
workspace(name = "kelvin_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "4e24df1e0b41f1ba04c8f72b1abd3d82b71e4517fa2fcd54c103134f535c0db6",
    strip_prefix = "mpact-riscv-92597b9bc9f07f7dedc0d380af70dbc3cf595339",
    url = "https://github.com/google/mpact-riscv/archive/92597b9bc9f07f7dedc0d380af70dbc3cf595339.tar.gz",
)

# Download only the single svdpi.h file.
http_file(
    name = "svdpi_h_file",
    downloaded_file_path = "svdpi.h",
    sha256 = "2528c8e529b66dd8e795c8a0fee326166cc51f7dee8fc6a0c6c930534fc780a6",
    urls = ["https://raw.githubusercontent.com/verilator/verilator/v5.028/include/vltstd/svdpi.h"],
)

load("@com_google_mpact-riscv//:repos.bzl", "mpact_riscv_repos")

mpact_riscv_repos()

load("@com_google_mpact-riscv//:dep_repos.bzl", "mpact_riscv_dep_repos")

mpact_riscv_dep_repos()

load("@com_google_mpact-riscv//:deps.bzl", "mpact_riscv_deps")

mpact_riscv_deps()

http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-0.26.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python3",
    python_version = "3.9",
)

http_file(
    name = "cc_static_library_external",
    downloaded_file_path = "cc_static_libarary.bzl",
    sha256 = "1287ce9f7e5fe31ad1b5937781531e4ab3f4656edabf650cca9ca720ceb31806",
    urls = ["https://raw.githubusercontent.com/project-oak/oak/fcceea755f0274d3a0eb7c0461b30af3dc28e40a/cc/build_defs.bzl"],
)
