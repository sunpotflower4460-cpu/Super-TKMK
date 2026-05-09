# src/mhd_solver/CMakeLists.txt
#
# Godunov + Constrained Transport MHD ソルバライブラリ

cmake_minimum_required(VERSION 3.20)

# ─── ソースファイル ───────────────────────────────────────────────────────────

set(MHD_SOLVER_SOURCES
    cuda_mhd_ct.cu
    initial_conditions.cu
)

set(MHD_SOLVER_HEADERS
    cuda_mhd_ct.cuh
)

# ─── ライブラリターゲット ─────────────────────────────────────────────────────

add_library(mhd_solver STATIC ${MHD_SOLVER_SOURCES})

target_include_directories(mhd_solver
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(mhd_solver
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

# CUDA 分離コンパイル（デバイス間リンクが必要な場合に有効化）
set_target_properties(mhd_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ─── 単体テスト実行ファイル ───────────────────────────────────────────────────

add_executable(test_mhd_ct tests/test_mhd_ct.cu)

target_link_libraries(test_mhd_ct
    PRIVATE
        mhd_solver
        CUDA::cudart
)

set_target_properties(test_mhd_ct PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# CTest への登録
add_test(NAME mhd_ct_divb_check COMMAND test_mhd_ct)

# ─── インストール ─────────────────────────────────────────────────────────────

install(TARGETS mhd_solver
    ARCHIVE DESTINATION lib
)
install(FILES ${MHD_SOLVER_HEADERS}
    DESTINATION include/super-tkmk/mhd_solver
)
