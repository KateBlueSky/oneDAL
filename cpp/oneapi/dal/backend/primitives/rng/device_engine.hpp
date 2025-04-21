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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"

// Replaced MKL with oneMATH
#include <oneapi/math/rng.hpp>

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

class gen_base {
public:
    virtual ~gen_base() = default;
    virtual engine_type_internal get_engine_type() const = 0;
    virtual void skip_ahead_gpu(std::int64_t nSkip) = 0;
};

class gen_mt2203 : public gen_base {
public:
    explicit gen_mt2203() = delete;
    gen_mt2203(sycl::queue queue, std::int64_t seed, std::int64_t engine_idx = 0)
            : _gen(queue, seed, engine_idx) {}

    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mt2203;
    }

    void skip_ahead_gpu(std::int64_t) override {}

    onemath::rng::mt2203* get() {
        return &_gen;
    }

protected:
    onemath::rng::mt2203 _gen;
};

class gen_philox : public gen_base {
public:
    explicit gen_philox() = delete;
    gen_philox(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    engine_type_internal get_engine_type() const override {
        return engine_type_internal::philox4x32x10;
    }

    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    onemath::rng::philox4x32x10* get() {
        return &_gen;
    }

protected:
    onemath::rng::philox4x32x10 _gen;
};

class gen_mrg32k : public gen_base {
public:
    explicit gen_mrg32k() = delete;
    gen_mrg32k(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mrg32k3a;
    }

    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    onemath::rng::mrg32k3a* get() {
        return &_gen;
    }

protected:
    onemath::rng::mrg32k3a _gen;
};

class gen_mt19937 : public gen_base {
public:
    explicit gen_mt19937() = delete;
    gen_mt19937(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mt19937;
    }

    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    onemath::rng::mt19937* get() {
        return &_gen;
    }

protected:
    onemath::rng::mt19937 _gen;
};

class gen_mcg59 : public gen_base {
public:
    explicit gen_mcg59() = delete;
    gen_mcg59(sycl::queue queue, std::int64_t seed) : _gen(queue, seed) {}

    engine_type_internal get_engine_type() const override {
        return engine_type_internal::mcg59;
    }

    void skip_ahead_gpu(std::int64_t nSkip) override {
        skip_ahead(_gen, nSkip);
    }

    onemath::rng::mcg59* get() {
        return &_gen;
    }

protected:
    onemath::rng::mcg59 _gen;
};

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
// namespace oneapi::dal::backend::primitives
