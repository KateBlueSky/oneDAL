/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/backend/primitives/rng/device_engine.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

// Swap oneMKL RNG with oneMATH RNG
#include <oneapi/math/rng.hpp>

namespace oneapi::dal::backend::primitives {

namespace bk = oneapi::dal::backend;

template <typename Distribution, typename Type>
sycl::event generate_rng(Distribution& distr,
                         device_engine& engine_,
                         std::int64_t count,
                         Type* dst,
                         const event_vector& deps) {
    auto engine_type = engine_.get_device_engine_base_ptr()->get_engine_type();

    if (engine_type == engine_type_internal::philox4x32x10) {
        auto& device_engine =
            *(static_cast<gen_philox*>(engine_.get_device_engine_base_ptr().get()))->get();
        return onemath::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mt19937) {
        auto& device_engine =
            *(static_cast<gen_mt19937*>(engine_.get_device_engine_base_ptr().get()))->get();
        return onemath::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mrg32k3a) {
        auto& device_engine =
            *(static_cast<gen_mrg32k*>(engine_.get_device_engine_base_ptr().get()))->get();
        return onemath::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mcg59) {
        auto& device_engine =
            *(static_cast<gen_mcg59*>(engine_.get_device_engine_base_ptr().get()))->get();
        return onemath::rng::generate(distr, device_engine, count, dst, deps);
    }
    else if (engine_type == engine_type_internal::mt2203) {
        auto& device_engine =
            *(static_cast<gen_mt2203*>(engine_.get_device_engine_base_ptr().get()))->get();
        return onemath::rng::generate(distr, device_engine, count, dst, deps);
    }
    else {
        throw std::runtime_error("Unsupported engine type in generate_rng");
    }
}

template <typename Type>
sycl::event uniform(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    Type a,
                    Type b,
                    const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) == sycl::usm::alloc::host) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    onemath::rng::uniform<Type> distr(a, b);
    engine_.skip_ahead_cpu(count);
    return generate_rng(distr, engine_, count, dst, deps);
}

template <typename Type>
sycl::event uniform_without_replacement(sycl::queue& queue,
                                        std::int64_t count,
                                        Type* dst,
                                        device_engine& engine_,
                                        Type a,
                                        Type b,
                                        const event_vector& deps) {
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);
    uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, state, a, b);
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
}

template <typename Type>
sycl::event shuffle(sycl::queue& queue,
                    std::int64_t count,
                    Type* dst,
                    device_engine& engine_,
                    const event_vector& deps) {
    Type idx[2];
    if (sycl::get_pointer_type(dst, engine_.get_queue().get_context()) ==
        sycl::usm::alloc::device) {
        throw domain_error(dal::detail::error_messages::unsupported_data_type());
    }
    void* state = engine_.get_host_engine_state();
    engine_.skip_ahead_gpu(count);

    for (std::int64_t i = 0; i < count; ++i) {
        uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
        std::swap(dst[idx[0]], dst[idx[1]]);
    }
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
}

template <typename Type>
sycl::event partial_fisher_yates_shuffle(sycl::queue& queue_,
                                         ndview<Type, 1>& result_array,
                                         std::int64_t top,
                                         std::int64_t seed,
                                         engine_type_internal method,
                                         const event_vector& deps) {
    device_engine eng_ = device_engine(queue_, seed, method);
    const auto casted_top = dal::detail::integral_cast<std::size_t>(top);
    const std::int64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT(casted_count < casted_top);
    auto indices_ptr = result_array.get_mutable_data();

    std::int64_t k = 0;
    std::size_t value = 0;
    auto state = eng_.get_host_engine_state();
    for (std::size_t i = 0; i < casted_count; i++) {
        uniform_dispatcher::uniform_by_cpu(1, &value, state, i, casted_top);
        for (std::size_t j = i; j > 0; j--) {
            if (value == dal::detail::integral_cast<std::size_t>(indices_ptr[j - 1])) {
                value = j - 1;
            }
        }
        if (value >= casted_top)
            continue;
        indices_ptr[i] = dal::detail::integral_cast<Type>(value);
        k++;
    }
    ONEDAL_ASSERT(k == count);
    return queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
}

#define INSTANTIATE_UNIFORM(F)                                         \
    template ONEDAL_EXPORT sycl::event uniform(sycl::queue& queue,     \
                                               std::int64_t count_,    \
                                               F* dst,                 \
                                               device_engine& engine_, \
                                               F a,                    \
                                               F b,                    \
                                               const event_vector& deps);

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(std::int32_t)

#define INSTANTIATE_UWR(F)                                                                 \
    template ONEDAL_EXPORT sycl::event uniform_without_replacement(sycl::queue& queue,     \
                                                                   std::int64_t count_,    \
                                                                   F* dst,                 \
                                                                   device_engine& engine_, \
                                                                   F a,                    \
                                                                   F b,                    \
                                                                   const event_vector& deps);

INSTANTIATE_UWR(float)
INSTANTIATE_UWR(double)
INSTANTIATE_UWR(std::int32_t)

#define INSTANTIATE_SHUFFLE(F)                                         \
    template ONEDAL_EXPORT sycl::event shuffle(sycl::queue& queue,     \
                                               std::int64_t count_,    \
                                               F* dst,                 \
                                               device_engine& engine_, \
                                               const event_vector& deps);

INSTANTIATE_SHUFFLE(std::int32_t)

#define INSTANTIATE_PARTIAL_SHUFFLE(F)                                                           \
    template ONEDAL_EXPORT sycl::event partial_fisher_yates_shuffle(sycl::queue& queue,          \
                                                                    ndview<F, 1>& a,             \
                                                                    std::int64_t top,            \
                                                                    std::int64_t seed,           \
                                                                    engine_type_internal method, \
                                                                    const event_vector& deps);

INSTANTIATE_PARTIAL_SHUFFLE(std::int32_t)
INSTANTIATE_PARTIAL_SHUFFLE(std::int64_t)

} // namespace oneapi::dal::backend::primitives
// namespace oneapi::dal::backend::primitives
