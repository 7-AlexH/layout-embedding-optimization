cmake_minimum_required(VERSION 3.16)
project(LayoutOpt)

set(OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/output" CACHE PATH "Output path for various files produced by the applications.")
set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Data path for various input files.")

set(CMAKE_CXX_STANDARD 17)

# Add -fPIC flag (required to build predicates.c)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ## Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using Clang (linux or apple) or GCC
    message("Using clang/gcc compiler flags")
    SET(BASE_CXX_FLAGS "-std=c++17 -Wall -Wextra -march=native")
    SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field")
    SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("Setting clang-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -fcolor-diagnostics")
        SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -fsanitize=address -fno-limit-debug-info")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message("Setting gcc-specific options")
        SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
        SET(CMAKE_CXX_FLAGS_DEBUG "-g3")
        SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
    endif()

    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    message("Using Visual Studio compiler flags")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"") # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"") # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"") # ignore truncation on initialization
    SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    # unrecognized
    message(FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]")
endif()

# Dependencies
# Warning: The order of these add_subdirectories matters, there are interdependencies.
set(GLOW_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}) # Viewer fonts will be placed here

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/extern/libtorch")

if(NOT TARGET glfw)
    add_subdirectory(extern/glfw)
endif()

if(NOT TARGET clean-core)
    add_subdirectory(extern/clean-core)
endif()

if(NOT TARGET typed-geometry)
    add_subdirectory(extern/typed-geometry)
endif()

if(NOT TARGET polymesh)
    add_subdirectory(extern/polymesh)
endif()

if(NOT TARGET glow)
    add_subdirectory(extern/glow)
endif()

if(NOT TARGET imgui)
    add_subdirectory(extern/imgui-lean)
endif()

if(NOT TARGET glow-extras)
    add_subdirectory(extern/glow-extras)
endif()

if(NOT TARGET geometry-central)
    add_subdirectory(extern/geometry-central)
endif()

find_package(OpenMP REQUIRED)

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/extern/libtorch") #make sure we use this libtorch and not system
find_package(Torch REQUIRED)
message(STATUS "Torch version: ${Torch_VERSION}")
set(CMAKE_CXX_FILES "${CMAKE_CXX_FILES} ${TORCH_CXX_FLAGS}")

# Find Eigen if not already present
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

# LayoutOpt Library (library directory)
file(GLOB_RECURSE LIBRARY_SOURCE_FILES
    "library/LayoutOpt/*.cc"
    "library/LayoutOpt/*.c")
 
add_library(LayoutOpt ${LIBRARY_SOURCE_FILES})
target_link_libraries(LayoutOpt PUBLIC imgui typed-geometry polymesh glow-extras OpenMP::OpenMP_CXX Eigen3::Eigen geometry-central ${TORCH_LIBRARIES})
target_include_directories(LayoutOpt PUBLIC library extern/libigl/include ${TORCH_INCLUDE_DIRS} )
target_compile_definitions(LayoutOpt PUBLIC OUTPUT_PATH="${OUTPUT_PATH}")
target_compile_definitions(LayoutOpt PUBLIC DATA_PATH="${DATA_PATH}")
target_link_libraries(LayoutOpt PRIVATE stdc++fs)

# Executable targets (apps directory)
file(GLOB_RECURSE APP_SOURCE_FILES "apps/*.cc")

foreach(APP_SOURCE_FILE ${APP_SOURCE_FILES})
    get_filename_component(APP_NAME ${APP_SOURCE_FILE} NAME_WE)
    message("Executable target: ${APP_NAME}")

    add_executable(${APP_NAME} ${APP_SOURCE_FILE})
    target_link_libraries(${APP_NAME} PRIVATE LayoutOpt)
endforeach()
