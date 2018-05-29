// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_RANDOMUTILS_HPP
#define NETKET_RANDOMUTILS_HPP

#include <Eigen/Dense>
#include <complex>
#include <random>

namespace netket {
using default_random_engine = std::mt19937;

template <typename T>
using enable_if_real_type = typename std::enable_if<std::is_floating_point<T>::value>::type;

template<typename T>
struct is_complex{
    static constexpr bool value{false};
};

template<typename T>
struct is_complex<std::complex<T >>{
    static constexpr bool value{true};
};


template<typename T>
using enable_if_complex_type = typename std::enable_if<is_complex<T>::value>::type;


template <typename Derive, typename Enable=void>
void RandomGaussian(Eigen::MatrixBase<Derive> &, std::default_random_engine &, double) {
}

template <typename Derive, typename Enable=void>
void RandomGaussian(Eigen::MatrixBase<Derive> &par,
                    int seed, double sigma) {
  std::default_random_engine generator(seed);
  RandomGaussian<Derive, Enable>(par, generator, sigma);
}

template<typename Derive, enable_if_real_type<typename Derive::Scalar>>
void RandomGaussian(Eigen::MatrixBase<Derive> &par,
                    std::default_random_engine &generator, double sigma) {
  std::normal_distribution<double> distribution(0, sigma);
  for (int i = 0; i < par.size(); i++) {
    par(i) = distribution(generator);
  }
}

template<typename Derive, enable_if_complex_type<typename Derive::Scalar>>
void RandomGaussian(Eigen::MatrixBase<Derive> &par,
                    std::default_random_engine &generator, double sigma) {
  std::normal_distribution<double> distribution(0, sigma);
  for (int i = 0; i < par.size(); i++) {
    par(i) = std::complex<double>(distribution(generator),
                                  distribution(generator));
  }
}

}  // namespace netket

#endif
