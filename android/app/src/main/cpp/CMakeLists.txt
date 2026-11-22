# The analysis of fresniff as a shared library for the Android app: the sources of the command line program
# (FFmpeg decoding and encoding with the MediaCodec codecs of the device, analysis on the CPU) and a JNI entry.
#
#   FRESNIFF_OPENCV_ANDROID_SDK  the unpacked OpenCV Android SDK (the folder with sdk/)
#   FRESNIFF_FFMPEG_ANDROID      FFmpeg built by android/build-ffmpeg.sh (the folder with <abi>/include, <abi>/lib)
cmake_minimum_required(VERSION 3.22)
project(fresniff-android LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(FRESNIFF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

if(NOT EXISTS "${FRESNIFF_OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCVConfig.cmake")
    message(FATAL_ERROR "OpenCV Android SDK not found in '${FRESNIFF_OPENCV_ANDROID_SDK}' (see android/README.md)")
endif()
set(OpenCV_DIR "${FRESNIFF_OPENCV_ANDROID_SDK}/sdk/native/jni")
find_package(OpenCV REQUIRED)

set(_ffmpeg "${FRESNIFF_FFMPEG_ANDROID}/${ANDROID_ABI}")
if(NOT EXISTS "${_ffmpeg}/lib/libavcodec.so")
    message(FATAL_ERROR "FFmpeg for ${ANDROID_ABI} not found in '${_ffmpeg}', build it with android/build-ffmpeg.sh (see android/README.md)")
endif()
foreach(_lib avformat avcodec avutil)
    add_library(ffmpeg_${_lib} SHARED IMPORTED)
    set_target_properties(ffmpeg_${_lib} PROPERTIES
        IMPORTED_LOCATION "${_ffmpeg}/lib/lib${_lib}.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_ffmpeg}/include")
endforeach()

# the compute shaders of the GPU pipeline, embedded as SPIR-V (<name>.spv.inc), compiled with the NDK's glslc
set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
find_program(FRESNIFF_GLSLC glslc HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}" NO_DEFAULT_PATH)
if(NOT FRESNIFF_GLSLC)
    message(FATAL_ERROR "glslc not found in ${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}")
endif()
set(_shaders "")
foreach(_name yuv_diff motion_pyramid motion_lk motion_select motion_texture overlay_base overlay_blur overlay_draw ahb_import nv12_to_rgba)
    set(_src "${FRESNIFF_ROOT}/src/vulkan/${_name}.comp")
    set(_inc "${_shader_dir}/${_name}.spv.inc")
    add_custom_command(
        OUTPUT "${_inc}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_dir}"
        COMMAND ${FRESNIFF_GLSLC} -O --target-env=vulkan1.1 -mfmt=num -o "${_inc}" "${_src}"
        DEPENDS "${_src}"
        COMMENT "Compiling ${_name}.comp"
        VERBATIM)
    list(APPEND _shaders "${_inc}")
endforeach()

add_library(fresniff SHARED
    fresniff_jni.cpp
    ${FRESNIFF_ROOT}/src/android/vk_mediacodec_pipeline.cpp
    ${_shaders}
    ${FRESNIFF_ROOT}/src/main.cpp
    ${FRESNIFF_ROOT}/src/video_sources.cpp
    ${FRESNIFF_ROOT}/src/frame_analyzer.cpp
    ${FRESNIFF_ROOT}/src/diff_engine_cpu.cpp
    ${FRESNIFF_ROOT}/src/opencv_pipeline.cpp
    ${FRESNIFF_ROOT}/src/clock_skew.cpp
    ${FRESNIFF_ROOT}/src/motion.cpp
    ${FRESNIFF_ROOT}/src/motion_video.cpp
    ${FRESNIFF_ROOT}/src/osd_font.cpp
    ${FRESNIFF_ROOT}/src/osd_font_data.cpp
    ${FRESNIFF_ROOT}/src/resolution.cpp
    ${FRESNIFF_ROOT}/src/resolution_video.cpp
    ${FRESNIFF_ROOT}/src/video_writer.cpp
    ${FRESNIFF_ROOT}/src/ffmpeg/ffmpeg_decoder.cpp
    ${FRESNIFF_ROOT}/src/ffmpeg/ffmpeg_cpu_pipeline.cpp
    ${FRESNIFF_ROOT}/src/ffmpeg/ffmpeg_writer.cpp
    ${FRESNIFF_ROOT}/src/ffmpeg/yuv_picture.cpp
)
target_include_directories(fresniff PRIVATE ${FRESNIFF_ROOT}/src ${FRESNIFF_ROOT}/third_party ${OpenCV_INCLUDE_DIRS} "${_shader_dir}")
target_compile_definitions(fresniff PRIVATE FRESNIFF_LIBRARY=1 USE_FFMPEG=1 FRESNIFF_MEDIACODEC_VULKAN=1)
target_compile_options(fresniff PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# the OpenCV of the SDK as one shared library (libopencv_java4.so), FFmpeg as shared libraries
target_link_libraries(fresniff PRIVATE opencv_java ffmpeg_avformat ffmpeg_avcodec ffmpeg_avutil vulkan mediandk android log)
