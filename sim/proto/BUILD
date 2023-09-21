# Protobuf for the kelvin sim trace file



package(default_visibility = ["//visibility:public"])

proto_library(
    name = "trace_proto",
    srcs = [
        "kelvin_trace.proto",
    ],
)

cc_proto_library(
    name = "trace_cc_proto",
    deps = [
        ":trace_proto",
    ],
)
