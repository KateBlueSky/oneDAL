/* file: service_blas_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_BLAS_MKL_H__
#define __SERVICE_BLAS_MKL_H__

#include "services/daal_defines.h"
#include <mkl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <dnnl.hpp>
#include <mkl.h>
#include <immintrin.h>
#include <cstring>
#include <mutex>

#define __DAAL_MKLFN_CALL_BLAS(f_name, f_args) f_name f_args;

#define __DAAL_MKLFN_CALL_RETURN_BLAS(f_name, f_args, res) res = f_name f_args;

using namespace dnnl;

namespace daal
{
namespace internal
{
namespace mkl
{
template <typename fpType, CpuType cpu>
struct MklBlas
{};

/*
// Double precision functions definition
*/

template <CpuType cpu>
struct MklBlas<double, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                       const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                      const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                       const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                       const DAAL_INT * ldaty)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       double * beta, double * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                      const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                       const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, double * a, double * x, DAAL_INT * incx, double * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const double * a, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static double xxdot(const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        double res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(ddot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        mkl_set_num_threads_local(old_nthr);
        return res;
    }
};

/*
// Single precision functions definition
*/
static std::mutex xxgemm_mutex;

template <CpuType cpu>
struct MklBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                       const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                     const DAAL_INT * lda)
    {
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                      const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                      const DAAL_INT * ldaty)
    {
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
    }

    static void convert_bf16_to_f32(const uint16_t* src, float* dst, size_t size) {
        size_t i = 0;

        #ifdef __AVX512F__
    
        //std::cout << "using __AVX512F__\n";
        for (; i + 15 < size; i += 16) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)&src[i]);

            // Zero extend to 32 bits by shifting left 16 bits
            __m512i expanded = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_vals), 16);

            // Bitcast to float
            __m512 f32_vals = _mm512_castsi512_ps(expanded);

            _mm512_storeu_ps(&dst[i], f32_vals);
        }
        #endif

        // Scalar fallback
        for (; i < size; ++i) {
            uint32_t val = static_cast<uint32_t>(src[i]) << 16;
            std::memcpy(&dst[i], &val, sizeof(float));
        }
    }

    static void convert_f32_to_bf16(const float* src, uint16_t* dst, size_t size) {
        size_t i = 0;

        // Vectorized conversion using AVX512-BF16
        #ifdef __AVX512BF16__
            for (; i + 15 < size; i += 16) {
            __m512 f = _mm512_loadu_ps(&src[i]);
            __m256bh bf16 = _mm512_cvtneps_pbh(f);
        _   mm256_storeu_si256((__m256i*)(&dst[i]), (__m256i)bf16);
            }
        #endif

        // Scalar fallback for remaining values
        for (; i < size; ++i) {
            uint32_t val;
            memcpy(&val, &src[i], sizeof(float));
            dst[i] = static_cast<uint16_t>(val >> 16);
        }
    }


    static void xxgemm_oneDNN_(const char *transa, const char *transb, const int *p, const int *ny, const int *n,
                          const float *alpha, const float *a, const int *lda, const float *y, const int *ldy,
                          const float *beta, float *aty, const int *ldaty) {
    // Create an engine and stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Define memory dimensions
    memory::dims a_dims = { *p, *n };
    memory::dims y_dims = { *n, *ny };
    memory::dims aty_dims = { *p, *ny };

    // Create memory descriptors
    auto a_md = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
    auto y_md = memory::desc(y_dims, memory::data_type::f32, memory::format_tag::ab);
    auto aty_md = memory::desc(aty_dims, memory::data_type::f32, memory::format_tag::ab);

    // Create memory objects
    auto a_mem = memory(a_md, eng, const_cast<float*>(a));
    auto y_mem = memory(y_md, eng, const_cast<float*>(y));
    auto aty_mem = memory(aty_md, eng, aty);

    // Create matmul primitive descriptor
    auto matmul_pd = matmul::primitive_desc(eng, a_md, y_md, aty_md);

    // Create matmul primitive
    auto matmul_prim = matmul(matmul_pd);

    // Execute matmul
    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, y_mem},
        {DNNL_ARG_DST, aty_mem}
    });

    // Wait for the computation to finish
    s.wait();
}




    #define USE_REORDER
    static void xxgemm_oneDNN(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty) 
    {
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        auto start = std::chrono::high_resolution_clock::now();
        memory::dims a_dims = { *p, *n };
        memory::dims y_dims = { *n, *ny };
        memory::dims aty_dims = { *p, *ny };

        
        auto a_md_bf16 = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto b_md_bf16 = memory::desc(y_dims, memory::data_type::bf16, memory::format_tag::ab);
        auto c_md_bf16 = memory::desc(aty_dims, memory::data_type::bf16, memory::format_tag::ab);

        #ifdef USE_REORDER
        auto a_md_f32 = memory::desc(a_dims, memory::data_type::f32, memory::format_tag::ab);
        auto b_md_f32 = memory::desc(y_dims, memory::data_type::f32, memory::format_tag::ab);
        auto c_md_f32 = memory::desc(aty_dims, memory::data_type::f32, memory::format_tag::ab);
        auto a_mem_f32 = memory(a_md_f32, eng, const_cast<float*>(a));
        auto b_mem_f32 = memory(b_md_f32, eng, const_cast<float*>(y));
        auto c_mem_f32 = memory(c_md_f32, eng, aty);
        auto a_mem_bf16 = memory(a_md_bf16, eng);
        auto b_mem_bf16 = memory(b_md_bf16, eng);
        auto c_mem_bf16 = memory(c_md_bf16, eng); 
        reorder(a_mem_f32,a_mem_bf16).execute(s, a_mem_f32, a_mem_bf16);
        reorder(b_mem_f32,b_mem_bf16).execute(s, b_mem_f32, b_mem_bf16);
        #else
        std::vector<uint16_t> a_bf(*p * *n), y_bf(*n * *ny), aty_bf(*p * *ny);
        convert_f32_to_bf16(a, a_bf.data(), *p * *n);
        convert_f32_to_bf16(y, y_bf.data(), *n * *ny);
        auto a_mem_bf16 = memory(a_md_bf16, eng, a_bf.data());
        auto b_mem_bf16 = memory(b_md_bf16, eng, y_bf.data());
        auto c_mem_bf16 = memory(c_md_bf16, eng, aty_bf.data());
        #endif

        auto matmul_pd = matmul::primitive_desc(eng, a_md_bf16, b_md_bf16, c_md_bf16);
        auto matmul_prim = matmul(matmul_pd);

        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, a_mem_bf16},
            {DNNL_ARG_WEIGHTS, b_mem_bf16},
            {DNNL_ARG_DST, c_mem_bf16}
        });

        #ifdef USE_REORDER
        reorder(c_mem_bf16,c_mem_f32).execute(s, c_mem_bf16,c_mem_f32);
        s.wait();
        #else
        s.wait();
        convert_bf16_to_f32(aty_bf.data(), aty, *p * *ny);
        #endif
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "oneDNN bf16 GEMM time: " << elapsed << " sec\n";
    }


    #define USE_ONE_DNN

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty)
    {
        std::lock_guard<std::mutex> lock(xxgemm_mutex);
           
        #ifdef USE_ONE_DNN
         
        xxgemm_oneDNN_(transa, transb, p, ny, n, alpha, a, lda, y, ldy, beta, aty, ldaty);
        
        #else
        auto start = std::chrono::high_resolution_clock::now();
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,aty, (MKL_INT *)ldaty));
        mkl_set_num_threads_local(old_nthr);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "mkl f32 GEMM time: " << elapsed << " sec\n";
        #endif 
        
        
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       float * beta, float * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                      const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                       const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, float * a, float * x, DAAL_INT * incx, float * y, DAAL_INT * incy)
    {
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
    }

    static void xxaxpy(const DAAL_INT * n, const float * a, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        mkl_set_num_threads_local(old_nthr);
    }

    static float xxdot(const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        float res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(sdot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        mkl_set_num_threads_local(old_nthr);
        return res;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
