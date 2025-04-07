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
#include <chrono>
#include <iomanip>
#include <dnnl.hpp>

#define __DAAL_MKLFN_CALL_BLAS(f_name, f_args) f_name f_args;

#define __DAAL_MKLFN_CALL_RETURN_BLAS(f_name, f_args, res) res = f_name f_args;

#define USE_ONEDNN

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
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xsyrk execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: p = " << *p << ", n = " << *n << ", lda = " << *lda << ", ldata = " << *ldata << std::endl;
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const double * alpha, const double * a,
                       const DAAL_INT * lda, const double * beta, double * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xxsyrk execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: p = " << *p << ", n = " << *n << ", lda = " << *lda << ", ldata = " << *ldata << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                     const DAAL_INT * lda)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xsyr execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", lda = " << *lda << std::endl;
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const double * alpha, const double * x, const DAAL_INT * incx, double * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xxsyr execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", lda = " << *lda << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                      const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                      const DAAL_INT * ldaty)
    {
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "xgemm execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: p = " << *p << ", ny = " << *ny << ", n = " << *n << ", lda = " << *lda << ", ldy = " << *ldy << ", ldaty = " << *ldaty << std::endl;
    }

    
    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const double * alpha,
                       const double * a, const DAAL_INT * lda, const double * y, const DAAL_INT * ldy, const double * beta, double * aty,
                       const DAAL_INT * ldaty)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "xxgemm execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: p = " << *p << ", ny = " << *ny << ", n = " << *n << ", lda = " << *lda << ", ldy = " << *ldy << ", ldaty = " << *ldaty << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }


    

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a,
                      const DAAL_INT * lda, const double * b, const DAAL_INT * ldb, const double * beta, double * c, const DAAL_INT * ldc)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xsymm execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", ldb = " << *ldb << ", ldc = " << *ldc << std::endl;
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, double * alpha, double * a, DAAL_INT * lda, double * b, DAAL_INT * ldb,
                       double * beta, double * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dsymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xxsymm execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", ldb = " << *ldb << ", ldc = " << *ldc << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                      const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xgemv execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", incx = " << *incx << ", incy = " << *incy << std::endl;
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const double * alpha, const double * a, const DAAL_INT * lda,
                       const double * x, const DAAL_INT * incx, const double * beta, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(dgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xxgemv execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, double * a, double * x, DAAL_INT * incx, double * y, DAAL_INT * incy)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xaxpy execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
    }

    static void xxaxpy(const DAAL_INT * n, const double * a, const double * x, const DAAL_INT * incx, double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(daxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed = end - start;
        //std::cout << "xxaxpy execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static double xxdot(const DAAL_INT * n, const double * x, const DAAL_INT * incx, const double * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        auto start = std::chrono::high_resolution_clock::now();
        double res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(ddot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "xxdot execution time: " << std::fixed << std::setprecision(8) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
        return res;
    }
};

/*
// Single precision functions definition
*/

template <CpuType cpu>
struct MklBlas<float, cpu>
{
    typedef DAAL_INT SizeType;

    static void xsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xsyrk execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: p = " << *p << ", n = " << *n << ", lda = " << *lda << ", ldata = " << *ldata << std::endl;
    }

    static void xxsyrk(const char * uplo, const char * trans, const DAAL_INT * p, const DAAL_INT * n, const float * alpha, const float * a,
                       const DAAL_INT * lda, const float * beta, float * ata, const DAAL_INT * ldata)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssyrk, (uplo, trans, (MKL_INT *)p, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, beta, ata, (MKL_INT *)ldata));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xxsyrk execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: p = " << *p << ", n = " << *n << ", lda = " << *lda << ", ldata = " << *ldata << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                     const DAAL_INT * lda)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xsyr execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", lda = " << *lda << std::endl;
    }

    static void xxsyr(const char * uplo, const DAAL_INT * n, const float * alpha, const float * x, const DAAL_INT * incx, float * a,
                      const DAAL_INT * lda)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssyr, (uplo, (MKL_INT *)n, alpha, x, (MKL_INT *)incx, a, (MKL_INT *)lda));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xxsyr execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", lda = " << *lda << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }



    static void xxgemm_oneDNN(const char *transa, const char *transb, const int *p, const int *ny, const int *n,
                   const float *alpha, const float *a, const int *lda, const float *y, const int *ldy,
                   const float *beta, float *aty, const int *ldaty) {
    
        using namespace dnnl;            

        // Create a oneDNN engine and stream
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

        // Create a matrix multiplication primitive descriptor
        auto matmul_pd = matmul::primitive_desc(eng, a_md, y_md, aty_md);

        // Create a matrix multiplication primitive
        auto matmul_prim = matmul(matmul_pd);

        // Execute the matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        matmul_prim.execute(s, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, y_mem},
        {DNNL_ARG_DST, aty_mem}
        });
        s.wait();
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate and print the execution time
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "xxgemm (oneDNN) execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: p = " << *p << ", ny = " << *ny << ", n = " << *n << ", lda = " << *lda << ", ldy = " << *ldy << ", ldaty = " << *ldaty << std::endl;
    }

    static void xgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                      const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                      const DAAL_INT * ldaty)
    {

        #ifdef USE_ONEDNN
        int ip = static_cast<int>(*p);
        int iny = static_cast<int>(*ny);
        int in = static_cast<int>(*n);
        int ild = static_cast<int>(*lda);
        int ildy = static_cast<int>(*ldy);
        int ildaty = static_cast<int>(*ldaty);
            
        xxgemm_oneDNN(transa, transb, &ip, &iny, &in, alpha, a, &ild, y, &ildy, beta, aty, &ildaty);

        #else
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "xgemm execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: p = " << *p << ", ny = " << *ny << ", n = " << *n << ", lda = " << *lda << ", ldy = " << *ldy << ", ldaty = " << *ldaty << std::endl;
        #endif
    
    }

    static void xxgemm(const char * transa, const char * transb, const DAAL_INT * p, const DAAL_INT * ny, const DAAL_INT * n, const float * alpha,
                       const float * a, const DAAL_INT * lda, const float * y, const DAAL_INT * ldy, const float * beta, float * aty,
                       const DAAL_INT * ldaty)
    {

        #ifdef USE_ONEDNN
        int ip = static_cast<int>(*p);
        int iny = static_cast<int>(*ny);
        int in = static_cast<int>(*n);
        int ild = static_cast<int>(*lda);
        int ildy = static_cast<int>(*ldy);
        int ildaty = static_cast<int>(*ldaty);
        
        xxgemm_oneDNN(transa, transb, &ip, &iny, &in, alpha, a, &ild, y, &ildy, beta, aty, &ildaty);

        #else

        int old_nthr = mkl_set_num_threads_local(1);
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(sgemm, (transa, transb, (MKL_INT *)p, (MKL_INT *)ny, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, y, (MKL_INT *)ldy, beta,
                                       aty, (MKL_INT *)ldaty));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "xxgemm execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: p = " << *p << ", ny = " << *ny << ", n = " << *n << ", lda = " << *lda << ", ldy = " << *ldy << ", ldaty = " << *ldaty << std::endl;
        mkl_set_num_threads_local(old_nthr);
        #endif
    }

    static void xsymm(const char * side, const char * uplo, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a,
                      const DAAL_INT * lda, const float * b, const DAAL_INT * ldb, const float * beta, float * c, const DAAL_INT * ldc)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xsymm execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", ldb = " << *ldb << ", ldc = " << *ldc << std::endl;
    }

    static void xxsymm(char * side, char * uplo, DAAL_INT * m, DAAL_INT * n, float * alpha, float * a, DAAL_INT * lda, float * b, DAAL_INT * ldb,
                       float * beta, float * c, DAAL_INT * ldc)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(ssymm, (side, uplo, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, b, (MKL_INT *)ldb, beta, c, (MKL_INT *)ldc));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xxsymm execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", ldb = " << *ldb << ", ldc = " << *ldc << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                      const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "xgemv execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", incx = " << *incx << ", incy = " << *incy << std::endl;
    }

    static void xxgemv(const char * trans, const DAAL_INT * m, const DAAL_INT * n, const float * alpha, const float * a, const DAAL_INT * lda,
                       const float * x, const DAAL_INT * incx, const float * beta, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(sgemv, (trans, (MKL_INT *)m, (MKL_INT *)n, alpha, a, (MKL_INT *)lda, x, (MKL_INT *)incx, beta, y, (MKL_INT *)incy));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "xxgemv execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: m = " << *m << ", n = " << *n << ", lda = " << *lda << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static void xaxpy(DAAL_INT * n, float * a, float * x, DAAL_INT * incx, float * y, DAAL_INT * incy)
    {
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xaxpy execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
    }

    static void xxaxpy(const DAAL_INT * n, const float * a, const float * x, const DAAL_INT * incx, float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        //auto start = std::chrono::high_resolution_clock::now();
        __DAAL_MKLFN_CALL_BLAS(saxpy, ((MKL_INT *)n, a, x, (MKL_INT *)incx, y, (MKL_INT *)incy));
        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<float> elapsed = end - start;
        //std::cout << "xxaxpy execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        //std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
    }

    static float xxdot(const DAAL_INT * n, const float * x, const DAAL_INT * incx, const float * y, const DAAL_INT * incy)
    {
        int old_nthr = mkl_set_num_threads_local(1);
        auto start = std::chrono::high_resolution_clock::now();
        float res;
        __DAAL_MKLFN_CALL_RETURN_BLAS(sdot, ((MKL_INT *)n, x, (MKL_INT *)incx, y, (MKL_INT *)incy), res);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "xxdot execution time: " << std::fixed << std::setprecision(4) << elapsed.count() << " seconds" << std::endl;
        std::cout << "Parameters: n = " << *n << ", incx = " << *incx << ", incy = " << *incy << std::endl;
        mkl_set_num_threads_local(old_nthr);
        return res;
    }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif // __SERVICE_BLAS_MKL_H__
