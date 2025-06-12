/* file: kmeans_lloyd_impl.i */
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
//  Implementation of auxiliary functions used in Lloyd method
//  of K-means algorithm.
//--
*/

#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_defines.h"
#include "src/algorithms/service_error_handling.h"

#include "src/threading/threading.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_spblas.h"
#include "src/services/service_data_utils.h"

#include "src/algorithms/kmeans/kmeans_lloyd_helper.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <dnnl.hpp>

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;
using sycl::ext::oneapi::bfloat16;
using namespace dnnl; 

template <typename algorithmFPType, CpuType cpu>
struct TaskKMeansLloyd
{
    DAAL_NEW_DELETE();

    std::vector<uint16_t> get_chunk_bf16(const std::vector<uint16_t>& data, std::size_t start, std::size_t length) {
        if (start + length > data.size()) length = data.size() - start;
    return std::vector<uint16_t>(data.begin() + start, data.begin() + start + length);
    }

  

    // Specialization for float
    void convert_to_bf16(const float* src, uint16_t* dst, size_t size) {
        
        size_t i = 0;
        #ifdef __AVX512BF16__
        for (; i + 15 < size; i += 16) {
            __m512 f = _mm512_loadu_ps(&src[i]);
            __m256bh bf16 = _mm512_cvtneps_pbh(f);
            _mm256_storeu_si256((__m256i*)(&dst[i]), (__m256i)bf16);
        }
        #endif
        for (; i < size; ++i) {
            uint32_t val;
            memcpy(&val, &src[i], sizeof(float));
            dst[i] = static_cast<uint16_t>(val >> 16);
        }
    }

    // Specialization for double
    void convert_to_bf16(const double* src, uint16_t* dst, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            float f = static_cast<float>(src[i]);  // cast down
            uint32_t val;
            memcpy(&val, &f, sizeof(float));
            dst[i] = static_cast<uint16_t>(val >> 16);
        }
    }


    TaskKMeansLloyd(int _dim, int _clNum, algorithmFPType * _centroids, const size_t max_block_size)
    {
        dim      = _dim;
        clNum    = _clNum;
        cCenters = _centroids;
        cCenters_bf16.resize(clNum * dim); 
        convert_to_bf16(cCenters, cCenters_bf16.data(), clNum * dim);

        /* Allocate memory for all arrays inside TLS */
        tls_task = new daal::static_tls<TlsTask<algorithmFPType, cpu> *>([=]() -> TlsTask<algorithmFPType, cpu> * {
            return TlsTask<algorithmFPType, cpu>::create(dim, clNum, max_block_size);
        }); /* Allocate memory for all arrays inside TLS: end */

        //convert_f32_to_bf16(cCenters, cCenters_bf16.data(), clNum * dim);

        clSq = service_scalable_calloc<algorithmFPType, cpu>(clNum);
        if (clSq)
        {
            for (size_t k = 0; k < clNum; k++)
            {
                algorithmFPType sum = algorithmFPType(0);
                PRAGMA_FORCE_SIMD
                PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
                for (size_t j = 0; j < dim; j++)
                {
                    sum += (float)cCenters_bf16[k * dim + j] * (float)cCenters_bf16[k * dim + j] * 0.5;
                
                }
                clSq[k] = sum;
            }
        }
    }

    ~TaskKMeansLloyd()
    {
        if (tls_task)
        {
            tls_task->reduce([=](TlsTask<algorithmFPType, cpu> * tt) -> void { delete tt; });
            delete tls_task;
        }
        if (clSq)
        {
            service_scalable_free<algorithmFPType, cpu>(clSq);
        }
    }

    static SharedPtr<TaskKMeansLloyd<algorithmFPType, cpu> > create(int dim, int clNum, algorithmFPType * centroids, const size_t max_block_size)
    {
        SharedPtr<TaskKMeansLloyd<algorithmFPType, cpu> > result(new TaskKMeansLloyd<algorithmFPType, cpu>(dim, clNum, centroids, max_block_size));
        if (result.get() && (!result->tls_task || !result->clSq))
        {
            result.reset();
        }
    
        return result;
    }

    Status addNTToTaskThreadedDense(const NumericTable * const ntData, const algorithmFPType * const catCoef, const size_t blockSizeDefault,
                                    NumericTable * ntAssign = nullptr);

    Status addNTToTaskThreadedCSR(const NumericTable * const ntData, const algorithmFPType * const catCoef, const size_t blockSizeDefault,
                                  NumericTable * ntAssign = nullptr);

    template <Method method>
    Status addNTToTaskThreaded(const NumericTable * const ntData, const algorithmFPType * const catCoef, const size_t blockSizeDefault,
                               NumericTable * ntAssign = nullptr);

    template <typename centroidsFPType>
    int kmeansUpdateCluster(int jidx, centroidsFPType * s1);

    template <Method method>
    void kmeansComputeCentroids(int * clusterS0, algorithmFPType * clusterS1, double * auxData);

    void kmeansInsertCandidate(TlsTask<algorithmFPType, cpu> * tt, algorithmFPType value, size_t index);

    Status kmeansComputeCentroidsCandidates(algorithmFPType * cValues, size_t * cIndices, size_t & cNum);

    void kmeansClearClusters(algorithmFPType * goalFunc);

    daal::static_tls<TlsTask<algorithmFPType, cpu> *> * tls_task;
    algorithmFPType * clSq;
    algorithmFPType * cCenters;
    std::vector<uint16_t> cCenters_bf16;   

    int dim;
    int clNum;

    typedef typename Fp2IntSize<algorithmFPType>::IntT algIntType;
};






template <typename algorithmFPType, CpuType cpu>
Status TaskKMeansLloyd<algorithmFPType, cpu>::addNTToTaskThreadedDense(const NumericTable * const ntData, const algorithmFPType * const catCoef,
                                                                       const size_t blockSizeDefault, NumericTable * ntAssign)
{
    const size_t n = ntData->getNumberOfRows();
    const size_t n_columns = ntData->getNumberOfColumns();


    size_t nBlocks = n / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != n);


    engine eng(engine::kind::cpu, 0);

    //row_accessor<const algorithmFPType> accessor(const_cast<NumericTable *>(ntData));

    ReadRows<float, cpu> mtData(*const_cast<NumericTable *>(ntData), 0, n);
    const float * const data_input = mtData.get();
    std::vector<uint16_t> data_bf16(n * n_columns);
    convert_to_bf16(data_input, data_bf16.data(), n * n_columns);

    SafeStatus safeStat;
    daal::static_threader_for(nBlocks, [=, &safeStat](const int k, size_t tid) {
        stream s(eng);
        
        struct TlsTask<algorithmFPType, cpu> * tt = tls_task->local(tid);
        DAAL_CHECK_MALLOC_THR(tt);
        const size_t blockSize = (k == nBlocks - 1) ? n - k * blockSizeDefault : blockSizeDefault;

        //ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable *>(ntData), k * blockSizeDefault, blockSize);
        //DAAL_CHECK_BLOCK_STATUS_THR(mtData);
        //const uint16_t * const data = mtData.get();
        std::vector<uint16_t> data_bf16_chunk = get_chunk_bf16(data_bf16, k * blockSizeDefault * n_columns, blockSize * n_columns); 
        //const uint16_t * const data = data_bf16_chunk.data();

        const size_t p                           = dim;
        const size_t nClusters                   = clNum;
        //const uint16_t * const inClusters = cCenters_bf16.data();
        const algorithmFPType * const clustersSq = clSq;

        algorithmFPType * trg        = &(tt->goalFunc);
        algorithmFPType * x_clusters = tt->mklBuff;

        int * cS0             = tt->cS0;
        algorithmFPType * cS1 = tt->cS1;

        int * assignments = nullptr;
        WriteOnlyRows<int, cpu> assignBlock(ntAssign, k * blockSizeDefault, blockSize);
        if (ntAssign)
        {
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            assignments = assignBlock.get();
        }

        const char transa           = 't';
        const char transb           = 'n';
        const DAAL_INT _m           = blockSize;
        const DAAL_INT _n           = nClusters;
        const DAAL_INT _k           = p;
        const algorithmFPType alpha = -1.0;
        const DAAL_INT lda          = p;
        const DAAL_INT ldy          = p;
        const algorithmFPType beta  = 1.0;
        const DAAL_INT ldaty        = blockSize;
        memory::dims a_dims = { (int)_m, (int)_k };
        memory::dims b_dims = { (int)_k, (int)_n };
        memory::dims c_dims = { (int)_m, (int)_n };

        auto a_md_bf16 = memory::desc(a_dims, memory::data_type::bf16, memory::format_tag::nc);
        auto b_md_bf16 = memory::desc(b_dims, memory::data_type::bf16, memory::format_tag::nc);
        auto c_md_bf32 = memory::desc(c_dims, memory::data_type::f32, memory::format_tag::nc);


        auto a_mem_bf16 = memory(a_md_bf16, eng, data_bf16_chunk.data());
        auto b_mem_bf16 = memory(b_md_bf16, eng, cCenters_bf16.data());
        auto c_mem_bf32 = memory(c_md_bf32, eng, x_clusters);


        for (size_t j = 0; j < nClusters; j++)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < blockSize; i++)
            {
                x_clusters[i + j * blockSize] = clustersSq[j];
            }
        }



        auto matmul_pd = matmul::primitive_desc(eng, a_md_bf16, b_md_bf16, c_md_bf32);
        auto matmul_prim = matmul(matmul_pd);
        matmul_prim.execute(s, {
            { DNNL_ARG_SRC, a_mem_bf16 },
            { DNNL_ARG_WEIGHTS, b_mem_bf16 },
            { DNNL_ARG_DST, c_mem_bf32 }
        });


        s.wait();

        //BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, data, &lda, inClusters, &ldy, &beta, x_clusters, &ldaty);

        PRAGMA_ICC_OMP(simd simdlen(16))
        for (algIntType i = 0; i < (algIntType)blockSize; i++)
        {
            algorithmFPType minGoalVal = x_clusters[i];
            algIntType minIdx          = 0;

            for (algIntType j = 0; j < (algIntType)nClusters; j++)
            {
                algorithmFPType localGoalVal = x_clusters[i + j * blockSize];
                if (localGoalVal < minGoalVal)
                {
                    minGoalVal = localGoalVal;
                    minIdx     = j;
                }
            }

            minGoalVal *= 2.0;

            *((algIntType *)&(x_clusters[i])) = minIdx;
            x_clusters[i + blockSize]         = minGoalVal;
        }

        algorithmFPType goal = algorithmFPType(0);
        for (size_t i = 0; i < blockSize; i++)
        {
            const size_t minIdx        = *((algIntType *)&(x_clusters[i]));
            algorithmFPType minGoalVal = x_clusters[i + blockSize];

            PRAGMA_FORCE_SIMD
            for (size_t j = 0; j < p; j++)
            {
                cS1[minIdx * p + j] += (float)data_bf16_chunk[i * p + j];
                minGoalVal += (float)data_bf16_chunk[i * p + j] * (float)data_bf16_chunk[i * p + j];
            }

            kmeansInsertCandidate(tt, minGoalVal, k * blockSizeDefault + i);
            cS0[minIdx]++;

            goal += minGoalVal;

            if (ntAssign)
            {
                DAAL_ASSERT(minIdx <= services::internal::MaxVal<int>::get())
                assignments[i] = (int)minIdx;
            }
        } /* for (size_t i = 0; i < blockSize; i++) */

        *trg += goal;
    }); /* daal::threader_for( nBlocks, nBlocks, [=](int k) */
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status TaskKMeansLloyd<algorithmFPType, cpu>::addNTToTaskThreadedCSR(const NumericTable * const ntData, const algorithmFPType * const catCoef,
                                                                     const size_t blockSizeDefault, NumericTable * ntAssign)
{
    CSRNumericTableIface * ntDataCsr = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(ntData));

    const size_t n = ntData->getNumberOfRows();

    size_t nBlocks = n / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != n);

    SafeStatus safeStat;
    daal::static_threader_for(nBlocks, [=, &safeStat](const int k, size_t tid) {
        struct TlsTask<algorithmFPType, cpu> * tt = tls_task->local(tid);
        DAAL_CHECK_MALLOC_THR(tt);

        const size_t blockSize = (k == nBlocks - 1) ? n - k * blockSizeDefault : blockSizeDefault;

        ReadRowsCSR<algorithmFPType, cpu> dataBlock(ntDataCsr, k * blockSizeDefault, blockSize);
        DAAL_CHECK_BLOCK_STATUS_THR(dataBlock);

        const algorithmFPType * const data = dataBlock.values();
        const size_t * const colIdx        = dataBlock.cols();
        const size_t * const rowIdx        = dataBlock.rows();

        const size_t p                     = dim;
        const size_t nClusters             = clNum;
        const algorithmFPType * inClusters = cCenters;
        const algorithmFPType * clustersSq = clSq;

        algorithmFPType * trg        = &(tt->goalFunc);
        algorithmFPType * x_clusters = tt->mklBuff;

        int * cS0             = tt->cS0;
        algorithmFPType * cS1 = tt->cS1;

        int * assignments = nullptr;
        WriteOnlyRows<int, cpu> assignBlock(ntAssign, k * blockSizeDefault, blockSize);
        if (ntAssign)
        {
            DAAL_CHECK_BLOCK_STATUS_THR(assignBlock);
            assignments = assignBlock.get();
        }

        const char transa           = 'n';
        const DAAL_INT _n           = blockSize;
        const DAAL_INT _p           = p;
        const DAAL_INT _c           = nClusters;
        const algorithmFPType alpha = 1.0;
        const algorithmFPType beta  = 0.0;
        const char matdescra[6]     = { 'G', 0, 0, 'F', 0, 0 };

        SpBlasInst<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra, data, (DAAL_INT *)colIdx, (DAAL_INT *)rowIdx, inClusters,
                                                  &_p, &beta, x_clusters, &_n);

        size_t csrCursor = 0;
        for (size_t i = 0; i < blockSize; i++)
        {
            algorithmFPType minGoalVal = clustersSq[0] - x_clusters[i];
            size_t minIdx              = 0;

            for (size_t j = 0; j < nClusters; j++)
            {
                if (minGoalVal > clustersSq[j] - x_clusters[i + j * blockSize])
                {
                    minGoalVal = clustersSq[j] - x_clusters[i + j * blockSize];
                    minIdx     = j;
                }
            }

            minGoalVal *= 2.0;

            size_t valuesNum = rowIdx[i + 1] - rowIdx[i];
            for (size_t j = 0; j < valuesNum; j++)
            {
                cS1[minIdx * p + colIdx[csrCursor] - 1] += data[csrCursor];
                minGoalVal += data[csrCursor] * data[csrCursor];
                csrCursor++;
            }

            kmeansInsertCandidate(tt, minGoalVal, k * blockSizeDefault + i);

            *trg += minGoalVal;

            cS0[minIdx]++;

            if (ntAssign)
            {
                DAAL_ASSERT(minIdx <= services::internal::MaxVal<int>::get())
                assignments[i] = (int)minIdx;
            }
        }
    });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
template <Method method>
Status TaskKMeansLloyd<algorithmFPType, cpu>::addNTToTaskThreaded(const NumericTable * const ntData, const algorithmFPType * const catCoef,
                                                                  const size_t blockSizeDefault, NumericTable * ntAssign)
{
    if (method == lloydDense)
    {
        return addNTToTaskThreadedDense(ntData, catCoef, blockSizeDefault, ntAssign);
    }
    else if (method == lloydCSR)
    {
        return addNTToTaskThreadedCSR(ntData, catCoef, blockSizeDefault, ntAssign);
    }
    DAAL_ASSERT(false);
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
template <typename centroidsFPType>
int TaskKMeansLloyd<algorithmFPType, cpu>::kmeansUpdateCluster(int jidx, centroidsFPType * s1)
{
    int idx = (int)jidx;

    int s0 = 0;

    tls_task->reduce([&](TlsTask<algorithmFPType, cpu> * tt) -> void { s0 += tt->cS0[idx]; });

    tls_task->reduce([=](TlsTask<algorithmFPType, cpu> * tt) -> void {
        int j;
        PRAGMA_FORCE_SIMD
        for (j = 0; j < dim; j++)
        {
            s1[j] += tt->cS1[idx * dim + j];
        }
    });
    return s0;
}

template <typename algorithmFPType, CpuType cpu>
template <Method method>
void TaskKMeansLloyd<algorithmFPType, cpu>::kmeansComputeCentroids(int * clusterS0, algorithmFPType * clusterS1, double * auxData)
{
    if (method == defaultDense && auxData)
    {
        for (size_t i = 0; i < clNum; i++)
        {
            service_memset_seq<double, cpu>(auxData, 0.0, dim);
            clusterS0[i] = kmeansUpdateCluster<double>(i, auxData);

            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < dim; j++)
            {
                clusterS1[i * dim + j] = auxData[j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < clNum; i++)
        {
            service_memset_seq<algorithmFPType, cpu>(&clusterS1[i * dim], 0.0, dim);
            clusterS0[i] = kmeansUpdateCluster<algorithmFPType>(i, &clusterS1[i * dim]);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
void TaskKMeansLloyd<algorithmFPType, cpu>::kmeansInsertCandidate(TlsTask<algorithmFPType, cpu> * tt, algorithmFPType value, size_t index)
{
    size_t cPos = tt->cNum;
    while (cPos > 0 && tt->cValues[cPos - 1] < value)
    {
        if (cPos < clNum)
        {
            tt->cValues[cPos]  = tt->cValues[cPos - 1];
            tt->cIndices[cPos] = tt->cIndices[cPos - 1];
        }
        cPos--;
    }

    if (cPos < clNum)
    {
        tt->cValues[cPos]  = value;
        tt->cIndices[cPos] = index;
        if (tt->cNum < clNum)
        {
            tt->cNum++;
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
Status TaskKMeansLloyd<algorithmFPType, cpu>::kmeansComputeCentroidsCandidates(algorithmFPType * cValues, size_t * cIndices, size_t & cNum)
{
    cNum = 0;

    TArray<algorithmFPType, cpu> tmpValues(clNum);
    TArray<size_t, cpu> tmpIndices(clNum);
    DAAL_CHECK_MALLOC(tmpValues.get() && tmpIndices.get());

    algorithmFPType * tmpValuesPtr = tmpValues.get();
    size_t * tmpIndicesPtr         = tmpIndices.get();
    int result                     = 0;

    tls_task->reduce([&](TlsTask<algorithmFPType, cpu> * tt) -> void {
        size_t lcNum               = tt->cNum;
        algorithmFPType * lcValues = tt->cValues;
        size_t * lcIndices         = tt->cIndices;

        size_t cPos  = 0;
        size_t lcPos = 0;

        while (cPos + lcPos < clNum && (cPos < cNum || lcPos < lcNum))
        {
            if (cPos < cNum && (lcPos == lcNum || cValues[cPos] > lcValues[lcPos]))
            {
                tmpValuesPtr[cPos + lcPos]  = cValues[cPos];
                tmpIndicesPtr[cPos + lcPos] = cIndices[cPos];
                cPos++;
            }
            else
            {
                tmpValuesPtr[cPos + lcPos]  = lcValues[lcPos];
                tmpIndicesPtr[cPos + lcPos] = lcIndices[lcPos];
                lcPos++;
            }
        }
        cNum = cPos + lcPos;
        result |= daal::services::internal::daal_memcpy_s(cValues, cNum * sizeof(algorithmFPType), tmpValuesPtr, cNum * sizeof(algorithmFPType));
        result |= daal::services::internal::daal_memcpy_s(cIndices, cNum * sizeof(size_t), tmpIndicesPtr, cNum * sizeof(size_t));
    });

    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
void TaskKMeansLloyd<algorithmFPType, cpu>::kmeansClearClusters(algorithmFPType * goalFunc)
{
    if (clNum != 0)
    {
        clNum = 0;

        if (goalFunc != 0)
        {
            *goalFunc = (algorithmFPType)(0.0);

            tls_task->reduce([=](TlsTask<algorithmFPType, cpu> * tt) -> void { (*goalFunc) += tt->goalFunc; });
        }
    }
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
