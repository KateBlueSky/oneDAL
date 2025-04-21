package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**/*.hpp",
    ]),
    includes = [
        "include",
    ],
)

cc_library(
    name = "onemath",
    linkopts = [
        "-fsycl-max-parallel-link-jobs=16",
         "-lonemath_blas_cublas",
         "-lonemath_rng_curand",
         "-lonemath_blas_mklcpu",      
         "-lonemath_rng_mklcpu",
         "-lonemath_dft_cufft", 
         "-lonemath",
         "-lonemath_dft_mklcpu",
         "-lonemath_sparse_blas_cusparse",
         "-lonemath_lapack_cusolver",
         "-lonemath_sparse_blas_mklcpu",
         "-lonemath_lapack_mklcpu",
    ],
    srcs = [
         "lib/libonemath_blas_cublas.so",
         "lib/libonemath_rng_curand.so",
         "lib/libonemath_blas_mklcpu.so",      
         "lib/libonemath_rng_mklcpu.so",
         "lib/libonemath_dft_cufft.so", 
         "lib/libonemath.so",
         "lib/libonemath_dft_mklcpu.so",
         "lib/libonemath_sparse_blas_cusparse.so",
         "lib/libonemath_lapack_cusolver.so",
         "lib/libonemath_sparse_blas_mklcpu.so",
         "lib/libonemath_lapack_mklcpu.so",
    ],
    deps = [
        ":headers",
    ],
)
