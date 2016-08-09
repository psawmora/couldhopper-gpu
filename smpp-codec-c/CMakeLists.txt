if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "in-source build is not supported. Please build from another directory.")
endif ()

cmake_minimum_required(VERSION 2.8.7)
project(smpp_codec_c)


set(smpp_codec_c_VERSION_MAJOR 0)
set(smpp_codec_c_VERSION_MINOR 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")
if (DEFINED ENV{LIBRARY_OUTPUT_PATH})
    message(STATUS "LIBRARY_OUTPUT_PATH DEFINED=$ENV{LIBRARY_OUTPUT_PATH}")
else()
    set(ENV{LIBRARY_OUTPUT_PATH} ${CMAKE_SOURCE_DIR}/lib)
    message(STATUS "LIBRARY_OUTPUT_PATH=$ENV{LIBRARY_OUTPUT_PATH}")
endif ()
set(LIBRARAY_NAME "SMPPDecoder")
set(JAVA_INCLUDE_PATH $ENV{JAVA_HOME}/include)
set(JAVA_INCLUDE_PATH2 ${JAVA_INCLUDE_PATH}/linux)
message(STATUS "JAVA_INCLUDE_PATH=${JAVA_INCLUDE_PATH}")

include_directories(src/common)
include_directories(src/decoder)
include_directories(src/util)
include_directories(${JAVA_INCLUDE_PATH})
include_directories(${JAVA_INCLUDE_PATH2})

add_subdirectory(src/common)
add_subdirectory(src/decoder)
add_subdirectory(src/util)

link_libraries(${LIBRARAY_NAME} ${JNI_LIBRARIES})

add_library(${LIBRARAY_NAME} SHARED ${SOURCE_FILES})
install(TARGETS ${LIBRARAY_NAME} DESTINATION $ENV{LIBRARY_OUTPUT_PATH})