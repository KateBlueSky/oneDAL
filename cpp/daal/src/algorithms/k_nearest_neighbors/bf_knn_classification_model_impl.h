/* file: bf_knn_classification_model_impl.h */
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

#ifndef __BF_KNN_CLASSIFICATION_MODEL_IMPL_
#define __BF_KNN_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include <iostream>

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace interface1
{
class Model::ModelImpl
{
public:
    ModelImpl(size_t nFeatures = 0) : _nFeatures(nFeatures) {}

    ~ModelImpl() {
        _data.reset();
        _labels.reset();

        std::cout <<"~~ModelImpl()" <<std::endl;  
        std::cout << "getLabelsConst() Use count: " << _labels.useCount() <<std::endl; 
        std::cout << "ptr1 use count: " << _data.useCount() << std::endl;
    }

    data_management::NumericTableConstPtr getData() const { 
        data_management::NumericTablePtr _data_temp = _data;    
        return _data_temp; 
        
    }

    data_management::NumericTablePtr getData() { 
        data_management::NumericTablePtr _data_temp = _data;
         std::cout << "ptr1 use count: " << _data_temp.useCount() << std::endl;
        return _data_temp; 
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);

        return services::Status();
    }

    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setData(const data_management::NumericTablePtr & value, bool copy)
    {
        std::cout <<"setData()" <<std::endl;
        services::Status s = setTable<algorithmFPType>(value, _data, copy);
        return s;
    }

    
    data_management::NumericTableConstPtr getLabels() const { 
        
        data_management::NumericTableConstPtr _return_labels = _labels;      
        std::cout << "getLabelsConst() Use count: " << _labels.useCount() <<std::endl; 
        std::cout <<"getLabelsConst()" <<std::endl;
        return _return_labels; 
    
    }

    data_management::NumericTablePtr getLabels() { 
        
        data_management::NumericTablePtr _return_labels = _labels;
        std::cout << "getLabels() Use count: " << _labels.useCount() <<std::endl; 
        std::cout <<"getLabels()" <<std::endl;
        return _return_labels; 
        
    }

    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setLabels(const data_management::NumericTablePtr & value, bool copy)
    {

        std::cout <<"setLabels()" <<std::endl;
        services::Status s = setTable<algorithmFPType>(value, _labels, copy);
        std::cout << "setLabels Use count: " << value.useCount() <<std::endl;
        return s;
    }


    size_t getNumberOfFeatures() const 
    { 
        return _nFeatures; 
        
    }

protected:
    template <typename algorithmFPType>
    DAAL_FORCEINLINE services::Status setTable(const data_management::NumericTablePtr & value, data_management::NumericTablePtr & dest, bool copy)
    {
        copy = true;
        if (!copy)
        {
            dest = value;
        }
        else
        {
            std::cout <<"setTable" <<std::endl;
            services::Status status;
            dest = data_management::HomogenNumericTable<algorithmFPType>::create(value->getNumberOfColumns(), value->getNumberOfRows(),
                                                                                 data_management::NumericTable::doAllocate, &status);
            DAAL_CHECK_STATUS_VAR(status);
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            DAAL_CHECK_STATUS_VAR(dest->getBlockOfRows(0, dest->getNumberOfRows(), data_management::writeOnly, destBD));
            DAAL_CHECK_STATUS_VAR(value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD));
            services::internal::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                              srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            DAAL_CHECK_STATUS_VAR(dest->releaseBlockOfRows(destBD));
            DAAL_CHECK_STATUS_VAR(value->releaseBlockOfRows(srcBD));
        }

        
        return services::Status();
    }

private:
    size_t _nFeatures;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
};

} // namespace interface1
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
