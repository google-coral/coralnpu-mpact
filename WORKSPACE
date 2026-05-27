# Setup bazel repository.
workspace(name = "coralnpu_sim")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name = "rules_java",
    sha256 = "bbe7d94360cc9ed4607ec5fd94995fd1ec41e84257020b6f09e64055281ecb12",
    urls = ["https://github.com/bazelbuild/rules_java/releases/download/8.14.0/rules_java-8.14.0.tar.gz"],
)

http_archive(
    name = "com_google_protobuf",
    integrity = "sha256-EKDVjzmhqQnpXgDougtbHcZNApl/dBFRlTorNln254w=",
    strip_prefix = "protobuf-29.0",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v29.0/protobuf-29.0.tar.gz"],
)

http_archive(
    name = "rules_license",
    sha256 = "26d4021f6898e23b82ef953078389dd49ac2b5618ac564ade4ef87cced147b38",
    urls = ["https://github.com/bazelbuild/rules_license/releases/download/1.0.0/rules_license-1.0.0.tar.gz"],
)

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "38faef26745f34a82de0daf3b65a207c8d2ecf825f37484a4a27132512583574",
    strip_prefix = "mpact-riscv-cb68bd4a2cb80dea24d9760dc6397b5854ea41bd",
    url = "https://github.com/google/mpact-riscv/archive/cb68bd4a2cb80dea24d9760dc6397b5854ea41bd.tar.gz",
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
    sha256 = "690e0141724abb568267e003c7b6d9a54925df40c275a870a4d934161dc9dd53",
    strip_prefix = "rules_python-0.40.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python3",
    python_version = "3.11",
)

http_file(
    name = "cc_static_library_external",
    downloaded_file_path = "cc_static_libarary.bzl",
    sha256 = "1287ce9f7e5fe31ad1b5937781531e4ab3f4656edabf650cca9ca720ceb31806",
    urls = ["https://raw.githubusercontent.com/project-oak/oak/fcceea755f0274d3a0eb7c0461b30af3dc28e40a/cc/build_defs.bzl"],
)
