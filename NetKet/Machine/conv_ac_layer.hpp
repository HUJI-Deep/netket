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

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"
#include "abstract_layer.hpp"
#include "ggemm_cpu.hpp"

#ifndef NETKET_CONV_AC_LAYER_HH
#define NETKET_CONV_AC_LAYER_HH

namespace netket {

using namespace std;
using namespace Eigen;


template<typename T>
class ConvACLayer : public AbstractLayer<T> {

    using VectorType=typename AbstractLayer<T>::VectorType;
    using MatrixType=typename AbstractLayer<T>::MatrixType;
    using TensorType=typename AbstractLayer<T>::TensorType;

    int number_of_input_channels_;
    int input_width_;
    int input_height_;
    int interlaced_top_buff_len_{};
    int output_height_{};
    int output_width_{};
    int number_of_output_channels_{};
    int kernel_width_{};
    int kernel_height_{};
    int strides_width_{};
    int strides_height_{};
    int padding_width_{};
    int padding_height_{};
    int num_regions_{};

    int my_mpi_node_{};

    TensorMap <Eigen::Tensor<T, 4>> offsets_weights_{NULL, 0, 0, 0, 0};
    TensorMap <Eigen::Tensor<T, 4>> transposed_offsets_weights_{NULL, 0, 0, 0,
                                                                0};
    Tensor<T, 1> padded_offsets_weights_{};
    Tensor<T, 1> padded_transposed_offsets_weights_{};
    Tensor<T, 1> top_expanded_data_{};
    Tensor<T, 1> top_expanded_diff_{};
    Tensor<T, 1> col_buffer_{};
    Tensor<T, 1> col_grad_buffer_{};

    std::unique_ptr<Matrix<T, 2, 1>[]> interlaced_top_buff_{};

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

    void create_tensors() {
        const int offsets_padding_size = ggemm_padded_output_size(
                number_of_output_channels_, number_of_input_channels_);
        const int total_offsets_size = Npar() + offsets_padding_size;
        padded_offsets_weights_ = Tensor<T, 1>(total_offsets_size);
        padded_transposed_offsets_weights_ = Tensor<T, 1>(total_offsets_size);
        new(&offsets_weights_)  Eigen::TensorMap<Eigen::Tensor<T, 4>>(
                padded_offsets_weights_.data(), kernel_height_, kernel_width_,
                number_of_output_channels_, number_of_input_channels_);
        new(&transposed_offsets_weights_) Eigen::TensorMap<Eigen::Tensor<T, 4>>(
                padded_transposed_offsets_weights_.data(), kernel_height_,
                kernel_width_, number_of_output_channels_,
                number_of_input_channels_);
        const int col_buffer_padding = ggemm_padded_output_size(
                number_of_input_channels_, output_height_ * output_width_);
        int top_expanded_size =
                num_regions_ * number_of_output_channels_ * output_height_ *
                output_width_ + col_buffer_padding;
        top_expanded_data_ = Tensor<T, 1>(top_expanded_size);
        top_expanded_diff_ = Tensor<T, 1>(top_expanded_size);
        int buffer_size =
                num_regions_ * number_of_input_channels_ * output_height_ *
                output_width_ + col_buffer_padding;
        col_buffer_ = Tensor<T, 1>(buffer_size);
        col_grad_buffer_ = Tensor<T, 1>(buffer_size);
        interlaced_top_buff_len_ =
                number_of_output_channels_ * output_height_ * output_width_ *
                num_regions_;
        interlaced_top_buff_.reset(
                new Matrix<T, 2, 1>[interlaced_top_buff_len_]);
    }

    void update_layer_properties(const json &pars) {
        number_of_output_channels_ = read_layer_param_from_json(pars,
                                                                "number_of_output_channels");
        kernel_width_ = read_layer_param_from_json(pars, "kernel_width");
        kernel_height_ = read_layer_param_from_json(pars, "kernel_height");
        strides_width_ = read_layer_param_from_json(pars, "strides_width");
        strides_height_ = read_layer_param_from_json(pars, "strides_height");
        padding_width_ = read_layer_param_from_json(pars, "padding_width");
        padding_height_ = read_layer_param_from_json(pars, "padding_height");
        num_regions_ = kernel_height_ * kernel_width_;
        output_height_ = dimension_out_size(input_height_,
                                            padding_height_,
                                            kernel_height_,
                                            strides_height_, true);
        output_width_ = dimension_out_size(input_width_, padding_width_,
                                           kernel_width_,
                                           strides_width_, true);
    }


public:

    using StateType=typename ConvACLayer<T>::StateType;
    using LookupType=typename ConvACLayer<T>::LookupType;

    ConvACLayer(const json &pars, int number_of_input_channels,
                int input_height, int input_width) :
            number_of_input_channels_(number_of_input_channels),
            input_height_(input_height),
            input_width_(input_width) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_node_);
        from_json(pars);
    }

    void GetParameters(VectorType &out_params, int start_idx) const override {
        int k = start_idx;
        for (int j = 0; j < kernel_height_; j++) {
            for (int i = 0; i < kernel_width_; i++) {
                for (int p = 0; p < number_of_output_channels_; p++) {
                    for (int c = 0; c < number_of_input_channels_; c++) {
                        out_params(k) = offsets_weights_(j, i, p, c);
                        k++;
                    }
                }
            }
        }
    }

    void SetParameters(const VectorType &pars, int start_idx) override {
        int k = start_idx;
        for (int j = 0; j < kernel_height_; j++) {
            for (int i = 0; i < kernel_width_; i++) {
                for (int p = 0; p < number_of_output_channels_; p++) {
                    for (int c = 0; c < number_of_input_channels_; c++) {
                        offsets_weights_(j, i, p, c) = pars[k];
                        transposed_offsets_weights_(j, i, c, p) = pars[k];
                        k++;
                    }
                }
            }
        }
    }

    void InitLookup(const VectorXd &v, LookupType &lt) override {

    }

    void
    UpdateLookup(const VectorXd &v, const std::vector<int> &tochange,
                 const std::vector<double> &newconf, LookupType &lt) override {

    }


    Eigen::array<int, 3> Noutput() const override {
        return Eigen::array<int, 3>{number_of_output_channels_, output_height_,
                                    output_width_};
    }


    int Nvisible() const {
        return number_of_input_channels_;
    }

    int Npar() const {
        return number_of_input_channels_ * number_of_output_channels_ *
               kernel_height_ * kernel_width_;
    }

    void InitRandomPars(std::default_random_engine &generator, double sigma) {
        Map<VectorType> par(offsets_weights_.data(), Npar());
        netket::RandomGaussian(par, generator, sigma);
        SetParameters(par, 0);
    }

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &plat_offsets_grad,
                TensorType &layer_gradient) {
        DerLog(input_tensor, next_layer_gradient, plat_offsets_grad);
        col_grad_buffer_.setZero();
        Matrix<T, 2, 1> vec;
        vec << 1, 0;
        ggemm_readc_cpu
                <false, false, T, Matrix<T, 2, 1>, T, Matrix<T, 2, 1>,
                        mex_backward_bottom_finite<T>, ggemm_add<T>, false,
                        no_op<T, Matrix<T, 2, 1>>, false,
                        true, true, true>
                (number_of_input_channels_, output_height_ * output_width_,
                 number_of_output_channels_,
                 padded_transposed_offsets_weights_.data(),
                 interlaced_top_buff_.get(), col_buffer_.data(),
                 col_grad_buffer_.data(), 0, vec, num_regions_);
        channels_first_col2im_cpu(col_grad_buffer_.data(),
                                  number_of_input_channels_, input_height_,
                                  input_width_,
                                  kernel_height_, kernel_width_,
                                  padding_height_, padding_height_,
                                  strides_height_,
                                  strides_width_,
                                  layer_gradient.data(), true);
    }

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &plat_offsets_grad) {
        Eigen::TensorMap<Eigen::Tensor<T, 2>> src(next_layer_gradient.data(), 1,
                                                  number_of_output_channels_ *
                                                  output_height_ *
                                                  output_width_);
        Eigen::TensorMap<Eigen::Tensor<T, 2>> dst(top_expanded_diff_.data(),
                                                  num_regions_,
                                                  number_of_output_channels_ *
                                                  output_height_ *
                                                  output_width_);
        Eigen::array<int, 2> bcast_gates2({num_regions_, 1});
        dst = src.broadcast(bcast_gates2);
        Eigen::TensorMap<Eigen::Tensor<T, 4>> offsets_grad(
                plat_offsets_grad.data(), Npar());
        offsets_grad.setZero();
        channels_first_im2col_cpu<T>(
                input_tensor.data(),
                number_of_input_channels_, input_height_, input_width_,
                kernel_height_, kernel_width_,
                padding_height_, padding_width_,
                strides_height_, strides_width_,
                col_buffer_.data(), true, T(0));
        ggemm_2ops_cpu
                <T, T, T, uint8_t,
                        ggemm_add<T, uint8_t>, ggemm_max<T>,
                        softmax<T>, ggemm_add<T>, false,
                        softmax_activation<T>, true,
                        true, true, true>
                (number_of_output_channels_, output_height_ * output_width_,
                 number_of_input_channels_, padded_offsets_weights_.data(),
                 col_buffer_.data(),
                 top_expanded_data_.data(),
                 -INFINITY, 0, 0, num_regions_);
        interlace_cpu<T>(interlaced_top_buff_len_, top_expanded_data_.data(),
                         top_expanded_diff_.data(),
                         interlaced_top_buff_.get());
        Matrix<T, 2, 1> vec;
        vec << 1, 0;
        ggemm_readc_cpu
                <false, true, Matrix<T, 2, 1>, T, T, Matrix<T, 2, 1>,
                        mex_backward_offsets_finite<T>, ggemm_add<T>, true,
                        no_op<T, Matrix<T, 2, 1>>, false,
                        true, true, true>
                (number_of_output_channels_, number_of_input_channels_,
                 output_height_ * output_width_, interlaced_top_buff_.get(),
                 col_buffer_.data(), padded_offsets_weights_.data(),
                 offsets_grad.data(), 0,
                 vec, num_regions_);
    }

    void LogVal(const TensorType &input_tensor, TensorType &output_tensor) {
        output_tensor.setZero();
        col_buffer_.setZero();
        channels_first_im2col_cpu<T>(
                input_tensor.data(),
                number_of_input_channels_, input_height_, input_width_,
                kernel_height_, kernel_width_,
                padding_height_, padding_width_,
                strides_height_, strides_width_,
                col_buffer_.data(), true, T(0));
        ggemm_2ops_cpu<T, T, T, uint8_t,
                ggemm_add<T, uint8_t>, ggemm_max<T>,
                softmax<T>, ggemm_add<T>, true,
                softmax_activation<T>, true,
                true, true, false>
                (number_of_output_channels_, output_height_ * output_width_,
                 number_of_input_channels_, padded_offsets_weights_.data(),
                 col_buffer_.data(),
                 output_tensor.data(), -INFINITY, 0, 0, num_regions_);
    }

    void LogVal(const TensorType &layer_input, TensorType &output_tensor,
                LookupType &lt) {
        LogVal(layer_input, output_tensor);
    }


    void to_json(json &j) const {
        j["Name"] = "RbmSpin";
        j["number_of_output_channels"] = number_of_output_channels_;
        j["kernel_width"] = kernel_width_;
        j["kernel_height"] = kernel_height_;
        j["strides_width"] = strides_width_;
        j["strides_height"] = strides_height_;
        VectorType params_vector(Npar());
        GetParameters(params_vector, 0);
        j["offsets_weights"] = params_vector;
    }


    void from_json(const json &pars) {
        assert_json_layer_name(pars);
        update_layer_properties(pars);
        create_tensors();
        if (FieldExists(pars, "offsets_weights_")) {
            SetParameters(pars["offsets_weights_"], 0);
        }
    }
};
}

#endif
