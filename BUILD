# Description:
# Implementation of the multimodal attribute extraction model.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"]) # Apache 2.0

exports_files(["LICENSE"])


# Utilities

py_binary(
    name= "json_preprocessing",
    srcs = ["utils/json_preprocessing.py"],
    srcs_version = "PY2AND3"
)

py_library(
    name = "utils",
    deps = [
        ":utils_",
        ":vgg_preprocessing",
    ]
)
    
py_library(
    name = "utils_",
    srcs = ["utils/utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name= "vgg_preprocessing",
    srcs = ["utils/vgg_preprocessing.py"],
    srcs_version = "PY2AND3"
)

# Network definitions.

py_library(
    name = "nets",
    deps = [
        ":deepsets",
        ":desc_encoder",
        ":image_encoder",
        ":mae",
        ":vgg",
    ],
)

py_library(
    name = "deepsets",
    srcs = ["nets/deepsets.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "deepsets_test",
    srcs = ["nets/deepsets_test.py"],
    deps = [
        ":deepsets",
    ],
)

py_library(
    name = "image_encoder",
    srcs = ["nets/image_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
    ]
)

py_test(
    name = "image_encoder_test",
    srcs = ["nets/image_encoder_test.py"],
    deps = [
        ":image_encoder",
    ],
)

py_library(
    name = "mae",
    srcs = ["nets/mae.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":deepsets",
        ":desc_encoder",
        ":image_encoder",
    ]
)

py_test(
    name = "mae_test",
    srcs = ["nets/mae_test.py"],
    deps = [
        ":mae",
    ],
)

py_library(
    name = "desc_encoder",
    srcs = ["nets/desc_encoder.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "desc_encoder_test",
    srcs = ["nets/desc_encoder_test.py"],
    deps = [
        ":desc_encoder",
    ],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "vgg_test",
    srcs = ["nets/vgg_test.py"],
    deps = [
        ":vgg",
    ],
)
