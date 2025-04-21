/*******************************************************************************
* Copyright 2022 Intel Corporation
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

// Replacing MKL with oneMATH
//#include <oneapi/math/blas.hpp>
#include "oneapi/math.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Convert oneDAL `ndorder` to oneMATH `layout`
inline constexpr oneapi::math::layout order_as_layout(ndorder order) {
    return (order == ndorder::c) ? oneapi::math::layout::R /* row-major */
                                 : oneapi::math::layout::C /* column-major */;
}

/// Convert oneDAL `transpose` to oneMATH `transpose`
inline constexpr oneapi::math::transpose transpose_to_onemath(primitives::transpose trans) {
    return (trans == primitives::transpose::trans) ? oneapi::math::transpose::trans
                                                   : oneapi::math::transpose::nontrans;
}

/// Mapping based on Fortran-style order
inline constexpr oneapi::math::transpose f_order_as_transposed(ndorder order) {
    return (order == ndorder::f) ? oneapi::math::transpose::trans : oneapi::math::transpose::nontrans;
}

/// Mapping based on C-style order
inline constexpr oneapi::math::transpose c_order_as_transposed(ndorder order) {
    return (order == ndorder::c) ? oneapi::math::transpose::trans : oneapi::math::transpose::nontrans;
}

/// Flip between upper/lower triangle
inline constexpr oneapi::math::uplo flip_uplo(oneapi::math::uplo order) {
    constexpr auto upper = oneapi::math::uplo::upper;
    constexpr auto lower = oneapi::math::uplo::lower;
    return (order == upper) ? lower : upper;
}

/// Identity mapping (just for semantic clarity)
inline constexpr oneapi::math::uplo ident_uplo(oneapi::math::uplo order) {
    constexpr auto upper = oneapi::math::uplo::upper;
    constexpr auto lower = oneapi::math::uplo::lower;
    return (order == upper) ? upper : lower;
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
 // namespace oneapi::dal::backend::primitives
