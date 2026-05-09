cmake_minimum_required(VERSION 3.20)

# =============================================================================
# プロジェクト設定
# =============================================================================
project(SuperTKMK
    VERSION 0.1.0
    DESCRIPTION "Accuracy-First Toroidal MHD Simulation and Visualization"
    LANGUAGES CXX CUDA
)

# C++ および CUDA 標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# CUDA / GPU 設定
# =============================================================================

# 対象 GPU アーキテクチャ（Ampere: sm_80/86, Hopper: sm_90 を想定）
# 実行環境に合わせて調整すること
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
    "Target CUDA GPU architectures (e.g. 80;86 for Ampere, 90 for Hopper)")

# CUDA バージョン確認（最低 12.0 を要求）
find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# 依存ライブラリ
# =============================================================================

# HDF5（シミュレーション状態の入出力）
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# =============================================================================
# コンパイルオプション
# =============================================================================

# Release ビルドのデフォルト最適化
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 警告設定
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
)

# CUDA 追加フラグ
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# =============================================================================
# ソースディレクトリ
# =============================================================================

# 各サブモジュールを追加
add_subdirectory(src/mhd_solver)
# add_subdirectory(src/advection_ct)
# add_subdirectory(src/visualization)

# =============================================================================
# 共通インクルードパス
# =============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${HDF5_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# =============================================================================
# インストール設定
# =============================================================================

install(DIRECTORY src/
    DESTINATION include/super-tkmk
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# =============================================================================
# テスト（CTest）
# =============================================================================

enable_testing()
# 各サブモジュールのテストはサブディレクトリの CMakeLists.txt で追加する

# =============================================================================
# ビルド情報サマリー
# =============================================================================

message(STATUS "")
message(STATUS "====== Super-TKMK Build Configuration ======")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA standard    : ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA version     : ${CUDAToolkit_VERSION}")
message(STATUS "  GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  HDF5 version     : ${HDF5_VERSION}")
message(STATUS "=============================================")
message(STATUS "")
