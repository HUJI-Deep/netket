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

#ifndef NETKET_ABSTRACT_LAYER_HH
#define NETKET_ABSTRACT_LAYER_HH

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <fstream>
#include <Lookup/lookup.hpp>
#include <netket.hpp>


namespace netket{
/**
  Abstract class for Neural Network layer.
  This class prototypes the methods needed
  by a class satisfying the Layer concept.
*/
template<typename T> class AbstractLayer {

    public:

        using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using TensorType = Eigen::Tensor<T, 3>;
        using StateType = T;
        using LookupType = Lookup<T>;

        virtual void to_json(json &j) const = 0;

        /**
        Member function returning the number of variational parameters.
        @return Number of variational parameters in the Machine.
        */
        virtual int Npar() const =0;

        /**
        Member function returning the current set of parameters in the machine.
        */
        virtual void GetParameters(VectorType &out_params, int start_idx) const =0;

        /**
        Member function setting the current set of parameters in the machine.
        */
        virtual void SetParameters(const VectorType &pars, int start_idx)=0;


        /**
        Member function providing a random initialization of the parameters.
        @param seed is the see of the random number generator.
        @param sigma is the variance of the gaussian.
        */
        virtual void InitRandomPars(std::default_random_engine &generator, double sigma)=0;


        /**
        Member function returning the number of output units.
        @return Number of output units of the Layer.
        */
        virtual Eigen::array<int , 3> Noutput() const =0;


        /**
        Member function computing the logarithm of the wave function for a given visible vector.
        Given the current set of parameters, this function should compute the value
        of the logarithm of the wave function from scratch.
        @param t a constant reference to previous layer output.
        @return Logarithm of the layer output.
        */
        virtual void LogVal(const TensorType &input_tensor, TensorType &output_tensor)=0;

        /**
        Member function computing the logarithm of the wave function for a given visible vector.
        Given the current set of parameters, this function should comput the value
        of the logarithm of the wave function using the information provided in the look-up table,
        to speed up the computation.
        @param t a constant reference to previous layer output.
        @param lt a constant eference to the look-up table.
        @return Logarithm of the layer output.
        */
        virtual void LogVal(const TensorType &layer_input, TensorType &output_tensor, LookupType &lt)=0;

        /**
        Member function initializing the look-up tables.
        If needed, a Machine can make use of look-up tables
        to speed up some critical functions. For example,
        to speed up the calculation of wave-function ratios.
        The state of a look-up table depends on the visible units.
        This function should initialize the look-up tables
        making sure that memory in the table is also being allocated.
        @param v a constant reference to the visible configuration.
        @param lt a reference to the look-up table to be initialized.
        */
        virtual void InitLookup(const Eigen::VectorXd &v, LookupType &lt)=0;

        /**
        Member function updating the look-up tables.
        If needed, a Machine can make use of look-up tables
        to speed up some critical functions. For example,
        to speed up the calculation of wave-function ratios.
        The state of a look-up table depends on the visible units.
        This function should update the look-up tables
        when the state of visible units is changed according
        to the information stored in toflip and newconf
        @param v a constant reference to the current visible configuration.
        @param tochange a constant reference to a vector containing the indeces of the units to be modified.
        @param newconf a constant reference to a vector containing the new values of the visible units:
        here newconf(i)=v'(tochange(i)), where v' is the new visible state.
        @param lt a reference to the look-up table to be updated.
        */
        virtual void UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                                  const std::vector<double> &newconf, LookupType &lt)=0;


        /**
        Member function computing the derivative of the logarithm of the wave function for a given visible vector.
        @param input_tensor a constant reference to a visible configuration.
        @param next_layer_gradient a constant reference to a visible configuration.
        @return Derivatives of the logarithm of the wave function with respect to the set of parameters.
        */
        virtual void
        DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient, Eigen::Map<VectorType> &plat_offsets_grad,
               TensorType &layer_gradient)=0;


        /**
        Member function computing the derivative of the logarithm of the wave function for a given visible vector.
        @param input_tensor a constant reference to a visible configuration.
        @param next_layer_gradient a constant reference to a visible configuration.
        @return Derivatives of the logarithm of the wave function with respect to the set of parameters.
        */
        virtual void
        DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient, Eigen::Map<VectorType> &plat_offsets_grad)=0;
    };
}

#endif
