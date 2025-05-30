/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;
using model_t = model<task::regression>;
using input_t = train_input<task::regression>;
using result_t = train_result<task::regression>;
using descriptor_t = detail::descriptor_base<task::regression>;
using param_t = const detail::train_parameters<task::regression>;

template <typename Float>
static result_t call_train_kernel(const context_gpu& ctx,
                                  const descriptor_t& desc,
                                  const table& data,
                                  const table& responses,
                                  const table& weights) {
    train_kernel_hist_impl<Float, std::uint32_t, std::int32_t, task::regression> train_hist_impl(
        ctx);
    return train_hist_impl(desc, data, responses, weights);
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_train_kernel<Float>(ctx,
                                    desc,
                                    input.get_data(),
                                    input.get_responses(),
                                    input.get_weights());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::hist, Task> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const param_t& params,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::hist, task::regression>;
template struct train_kernel_gpu<double, method::hist, task::regression>;

} // namespace oneapi::dal::decision_forest::backend
