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
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    strip_prefix = "protobuf-29.0",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v29.0/protobuf-29.0.tar.gz"],
)

http_archive(
    name = "rules_license",
    sha256 = "26d4021f6898e23b82ef953078389dd49ac2b5618ac564ade4ef87cced147b38",
    urls = ["https://github.com/bazelbuild/rules_license/releases/download/1.0.0/rules_license-1.0.0.tar.gz"],
)

http_archive(
    name = "linenoise",
    build_file_content = """
cc_library(
    name = "linenoise",
    srcs = ["linenoise.c"],
    hdrs = ["linenoise.h"],
    visibility = ["//visibility:public"],
)
""",
    integrity = "sha256-l619QEHhHX+jlYGf13PBiS3qieUpI0I3ioNFaSzonCk=",
    strip_prefix = "linenoise-2.0",
    url = "https://github.com/antirez/linenoise/archive/refs/tags/2.0.tar.gz",
)

# MPACT-RiscV repo
http_archive(
    name = "com_google_mpact-riscv",
    sha256 = "06d89e9604ea7cc743e0c32d5ab3cf798e22905da19ebb819143b3b4d0df676a",
    strip_prefix = "mpact-riscv-e4f1e9c1b243954ff8388fd39019248b2ae7341a",
    url = "https://github.com/google/mpact-riscv/archive/e4f1e9c1b243954ff8388fd39019248b2ae7341a.tar.gz",
)

# MPACT-Sim repo
http_archive(
    name = "mpact-sim",
    sha256 = "2dc7e2463556f2e29bb6c2429833d9f672774dde79d4ced7f553703018c9e91c",
    strip_prefix = "mpact-sim-9c43949f80bef9978654473d9703ac29de30bc34",
    url = "https://github.com/google/mpact-sim/archive/9c43949f80bef9978654473d9703ac29de30bc34.tar.gz",
)

http_archive(
    name = "abseil-cpp",
    sha256 = "f50e5ac311a81382da7fa75b97310e4b9006474f9560ac46f54a9967f07d4ae3",
    strip_prefix = "abseil-cpp-20240722.0",
    url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.tar.gz",
)

http_archive(
    name = "com_google_googletest",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

http_archive(
    name = "com_googlesource_code_re2",
    repo_mapping = {
        "@com_google_absl": "@abseil-cpp",
    },
    sha256 = "4e6593ac3c71de1c0f322735bc8b0492a72f66ffccfad76e259fa21c41d27d8a",
    strip_prefix = "re2-2023-11-01",
    urls = ["https://github.com/google/re2/archive/refs/tags/2023-11-01/re2-2023-11-01.tar.gz"],
)

http_archive(
    name = "rules_cc",
    sha256 = "64cb81641305dcf7b3b3d5a73095ee8fe7444b26f7b72a12227d36e15cfbb6cb",
    strip_prefix = "rules_cc-0.1.3",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.3/rules_cc-0.1.3.tar.gz",
)

http_archive(
    name = "com_github_serge1_elfio",
    build_file = "@mpact-sim//:external/BUILD.elfio",
    sha256 = "caf49f3bf55a9c99c98ebea4b05c79281875783802e892729eea0415505f68c4",
    strip_prefix = "elfio-3.12",
    urls = ["https://github.com/serge1/ELFIO/releases/download/Release_3.12/elfio-3.12.tar.gz"],
)

http_file(
    name = "org_antlr_tool",
    sha256 = "bc13a9c57a8dd7d5196888211e5ede657cb64a3ce968608697e4f668251a8487",
    url = "https://www.antlr.org/download/antlr-4.13.1-complete.jar",
)

http_archive(
    name = "org_antlr4_cpp_runtime",
    add_prefix = "antlr4-runtime",
    build_file = "@mpact-sim//:external/BUILD.antlr4",
    sha256 = "d350e09917a633b738c68e1d6dc7d7710e91f4d6543e154a78bb964cfd8eb4de",
    strip_prefix = "runtime/src",
    urls = ["https://www.antlr.org/download/antlr4-cpp-runtime-4.13.1-source.zip"],
)

# Download only the single svdpi.h file.
http_file(
    name = "svdpi_h_file",
    downloaded_file_path = "svdpi.h",
    sha256 = "2528c8e529b66dd8e795c8a0fee326166cc51f7dee8fc6a0c6c930534fc780a6",
    urls = ["https://raw.githubusercontent.com/verilator/verilator/v5.028/include/vltstd/svdpi.h"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

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
