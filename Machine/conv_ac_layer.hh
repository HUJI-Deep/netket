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

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <Json/json.hh>

#include "abstract_layer.hh"

#ifndef NETKET_CONV_AC_LAYER_HH
#define NETKET_CONV_AC_LAYER_HH

namespace netket {

using namespace std;
using namespace Eigen;


template<typename T>
class ConvACLayer : public AbstractLayer<T> {

    using VectorType=typename AbstractLayer<T>::VectorType;
    using MatrixType=typename AbstractLayer<T>::MatrixType;

    int number_of_input_channels_;
    int number_of_output_channels_{};
    int kernel_width_{};
    int kernel_height_{};
    int strides_width_{};
    int strides_height_{};

    int my_mpi_node_{};

    MatrixType offsets_weights_{};


public:

    using StateType=typename ConvACLayer<T>::StateType;
    using LookupType=typename ConvACLayer<T>::LookupType;

    ConvACLayer(const json &pars, int number_of_input_channels) :
            number_of_input_channels_(number_of_input_channels) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_node_);
        from_json(pars);
    }

    void GetParameters(VectorType &out_params, int start_idx) override {

    }

    void SetParameters(const VectorType &pars, int start_idx) override {

    }

    void InitLookup(const Eigen::VectorXd &v, LookupType &lt) override {

    }

    void
    UpdateLookup(const Eigen::VectorXd &v, const std::vector<int> &tochange,
                 const std::vector<double> &newconf, LookupType &lt) override {

    }


    int Noutput() const override {
        return number_of_output_channels_;
    }


    int Nvisible() const {
        return number_of_input_channels_;
    }

    int Npar() const {
        return number_of_input_channels_ * number_of_output_channels_ *
               kernel_height_ * kernel_width_;
    }

    void InitRandomPars(std::default_random_engine& generator,double sigma) {

    }


    VectorType DerLog(const VectorXd &v) {
        return VectorType{};
    }

    //Value of the logarithm of the wave-function
    T LogVal(const VectorXd &v) {
        return T{};
    }

    //Value of the logarithm of the wave-function
    //using pre-computed look-up tables for efficiency
    T LogVal(const VectorXd &v, LookupType &lt) {
        return T{};
    }

    //Difference between logarithms of values, when one or more visible variables are being flipped
    VectorType LogValDiff(const VectorXd &v,
                          const vector<vector<int> > &tochange,
                          const vector<vector<double>> &newconf) {

        return VectorType{};
    }

    //Difference between logarithms of values, when one or more visible variables are being flipped
    //Version using pre-computed look-up tables for efficiency on a small number of spin flips
    T LogValDiff(const VectorXd &v, const vector<int> &tochange,
                 const vector<double> &newconf, const LookupType &lt) {
        return T{};
    }

    void to_json(json &j) const {
        j["Name"] = "RbmSpin";
        j["number_of_output_channels"] = number_of_output_channels_;
        j["kernel_width"] = kernel_width_;
        j["kernel_height"] = kernel_height_;
        j["strides_width"] = strides_width_;
        j["strides_height"] = strides_height_;
        j["offsets_weights"] = offsets_weights_;
    }

    int read_layer_param_from_json(const json &pars, const string &param_name) {
        if (FieldExists(pars, param_name)) {
            return FieldVal(pars, param_name);
        } else {
            if (my_mpi_node_ == 0) {
                cerr
                        << "# Error while constructing ConvACLayer from Json input: missing attribute \""
                        << param_name << "\"" << endl;
            }
            std::abort();
        }
    }

    void assert_json_layer_name(const json &pars) {
        if (pars.at("Name") != "ConvACLayer") {
            if (my_mpi_node_ == 0) {
                cerr << "# Error while constructing ConvACLayer from Json input"
                     << endl;
            }
            std::abort();
        }
    }

    void from_json(const json &pars) {
        assert_json_layer_name(pars);
        number_of_output_channels_ = read_layer_param_from_json(pars,
                                                                "number_of_output_channels");
        kernel_width_ = read_layer_param_from_json(pars, "kernel_width");
        kernel_height_ = read_layer_param_from_json(pars, "kernel_height");
        strides_width_ = read_layer_param_from_json(pars, "strides_width");
        strides_height_ = read_layer_param_from_json(pars, "strides_height");
        if (FieldExists(pars, "offsets_weights_")) {
            offsets_weights_ = pars["offsets_weights_"];
        }
    }
};


}

#endif
