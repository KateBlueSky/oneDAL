load("@onedal//dev/bazel:repos.bzl", "repos")

oneMath_repo = repos.prebuilt_libs_repo_rule(
    includes = [
        "include",
    ],
    libs = [
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
    build_template = "@onedal//dev/bazel/deps:oneMath.tpl.BUILD"
)

