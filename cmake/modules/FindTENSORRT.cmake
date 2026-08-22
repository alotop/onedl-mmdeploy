# Copyright (c) OpenMMLab. All rights reserved.

if (NOT DEFINED TENSORRT_DIR)
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
endif ()
if (NOT TENSORRT_DIR)
    message(FATAL_ERROR "Please set TENSORRT_DIR with cmake -D option.")
endif()

find_path(
    TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

if (NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find TensorRT header NvInfer.h "
        "in TENSORRT_DIR: ${TENSORRT_DIR} or in CUDA_TOOLKIT_ROOT_DIR: "
        "${CUDA_TOOLKIT_ROOT_DIR}, please check if the path is correct.")
endif ()

set(_version_h "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")

if(EXISTS "${_version_h}")
    file(READ "${_version_h}" _contents)

    # 第一步：提取 NV_TENSORRT_MAJOR 的定义（可能是数字或宏名）
    string(REGEX MATCH "#define[ \t]+NV_TENSORRT_MAJOR[ \t]+([^ \t\r\n]+)" _dummy "${_contents}")
    if(CMAKE_MATCH_1)
        set(_token "${CMAKE_MATCH_1}")
        # 检查该 token 是否为纯数字
        if(_token MATCHES "^[0-9]+$")
            set(NV_TENSORRT_MAJOR "${_token}")
        else()
            # 否则视为宏名，继续查找该宏的定义
            string(REGEX MATCH "#define[ \t]+${_token}[ \t]+([0-9]+)" _dummy2 "${_contents}")
            if(CMAKE_MATCH_1)
                set(NV_TENSORRT_MAJOR "${CMAKE_MATCH_1}")
            else()
                message(WARNING "Could not parse macro ${_token} as number, defaulting to 8")
                set(NV_TENSORRT_MAJOR 8)
            endif()
        endif()
    else()
        message(WARNING "NV_TENSORRT_MAJOR not found in header, defaulting to 8")
        set(NV_TENSORRT_MAJOR 8)
    endif()
else()
    message(WARNING "NvInferVersion.h not found at ${_version_h}, defaulting to 8")
    set(NV_TENSORRT_MAJOR 8)
endif()

if(NV_TENSORRT_MAJOR GREATER 8)
    set(__TENSORRT_LIB_COMPONENTS nvinfer_10;nvinfer_plugin_10)
else()
    set(__TENSORRT_LIB_COMPONENTS nvinfer;nvinfer_plugin)
endif()
message(STATUS "TensorRT major version: ${NV_TENSORRT_MAJOR}")
message(STATUS "TensorRT library components: ${__TENSORRT_LIB_COMPONENTS}")

# set(__TENSORRT_LIB_COMPONENTS nvinfer;nvinfer_plugin)
foreach(__component ${__TENSORRT_LIB_COMPONENTS})
    find_library(
        __component_path ${__component}
        HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    if (NOT __component_path)
        message(FATAL_ERROR "Cannot find TensorRT lib ${__component} in "
            "TENSORRT_DIR: ${TENSORRT_DIR} or CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}, "
            "please check if the path is correct")
    endif()

    add_library(${__component} SHARED IMPORTED)
    set_property(TARGET ${__component} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    if (MSVC)
        set_target_properties(
            ${__component} PROPERTIES
            IMPORTED_IMPLIB_RELEASE ${__component_path}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    else()
        set_target_properties(
            ${__component} PROPERTIES
            IMPORTED_LOCATION_RELEASE ${__component_path}
            INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        )
    endif()
    unset(__component_path CACHE)
endforeach()

set(TENSORRT_LIBS ${__TENSORRT_LIB_COMPONENTS})
