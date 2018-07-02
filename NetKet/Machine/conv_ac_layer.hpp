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

#ifndef NETKET_CONV_AC_LAYER_HH
#define NETKET_CONV_AC_LAYER_HH

#define MINUS_INFINITY (-1e38)

namespace netket {

using namespace std;
using namespace Eigen;

template<typename T>
struct MyMinusOp {
    EIGEN_EMPTY_STRUCT_CTOR(MyMinusOp)

    typedef T result_type;

    T operator()(const T &a, const double &b) const { return a - T{b}; }
};

template<typename T>
class ConvACLayer : public AbstractLayer<T> {

    using VectorType=typename AbstractLayer<T>::VectorType;
    using MatrixType=typename AbstractLayer<T>::MatrixType;
    using TensorType=typename AbstractLayer<T>::TensorType;

    bool init_in_log_space_;
    bool normalize_input_channels_;
    int number_of_input_channels_;
    int input_height_;
    int input_width_;
    int output_height_{};
    int output_width_{};
    int number_of_output_channels_{};
    int kernel_width_{};
    int kernel_height_{};
    int strides_width_{};
    int strides_height_{};
    int padding_width_{};
    int padding_height_{};
    int tensor_lookup_index_{-1};
    int large_tensor_lookup_index_{-1};
    int my_mpi_node_{};

    Tensor<T, 4> offsets_weights_{};
    Tensor<T, 4> input_patches_{};
    Tensor<T, 5> chain_rule_before_sum_{};
    Tensor<T, 6> input_chain_rule_before_sum_{};
    Tensor<T, 6> input_element_wise_sum_{};
    Tensor<T, 5> element_wise_sum_{};
    Tensor<T, 2> single_element_wise_sum_{};
    Tensor<T, 1> single_logsumexp_input_channel_{};
    Tensor<double , 1> single_max_per_input_channel_{};
    Tensor<T, 4> logsumexp_input_channel_{};
    Tensor<double , 4> max_per_input_channel_{};
    Tensor<double , 4> max_per_spatial_location_{};
    Tensor<T, 5> input_logsumexp_input_channel_{};
    Tensor<double , 5> input_max_per_input_channel_{};
    Tensor<double , 3> input_max_per_spatial_location_{};
    VectorXd input_vector_;

    template<typename ParamType>
    void read_layer_param_from_json(const json &pars, const string &param_name,
                                    ParamType &param_value) {
        if (FieldExists(pars, param_name)) {
            param_value = FieldVal(pars, param_name);
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
        input_patches_ = Tensor<T, 4>(number_of_input_channels_, kernel_height_,
                                      kernel_width_,
                                      output_height_ * output_width_);
        offsets_weights_ = Tensor<T, 4>(number_of_output_channels_,
                                        number_of_input_channels_,
                                        kernel_height_, kernel_width_);
        chain_rule_before_sum_ = Tensor<T, 5>(number_of_output_channels_, number_of_input_channels_, kernel_height_, kernel_width_, output_height_ * output_width_);
        element_wise_sum_ = Tensor<T, 5>(number_of_output_channels_, number_of_input_channels_, kernel_height_, kernel_width_, output_height_ * output_width_);
        single_element_wise_sum_ = Tensor<T, 2>(number_of_output_channels_, number_of_input_channels_);
        input_element_wise_sum_  = Tensor<T, 6>(number_of_output_channels_, number_of_input_channels_, kernel_height_, kernel_width_, input_height_ , input_width_);
        input_chain_rule_before_sum_ = Tensor<T, 6>(number_of_output_channels_, number_of_input_channels_, kernel_height_, kernel_width_, input_height_ , input_width_);
        logsumexp_input_channel_ = Tensor<T, 4>(number_of_output_channels_, kernel_height_, kernel_width_, output_height_ * output_width_);
        single_logsumexp_input_channel_ = Tensor<T, 1>(number_of_output_channels_);
        input_logsumexp_input_channel_ = Tensor<T, 5>(number_of_output_channels_, kernel_height_, kernel_width_, input_height_, input_width_);
        max_per_input_channel_ = Tensor<double, 4>(number_of_output_channels_, kernel_height_, kernel_width_, output_height_ * output_width_);
        input_max_per_input_channel_ = Tensor<double, 5>(number_of_output_channels_, kernel_height_, kernel_width_, input_height_ , input_width_);
        max_per_spatial_location_ = Tensor<double, 4>(number_of_output_channels_, number_of_input_channels_, kernel_height_, kernel_width_);
        input_max_per_spatial_location_ = Tensor<double, 3>(number_of_input_channels_, input_height_, input_width_);
        single_max_per_input_channel_ = Tensor<double, 1>(number_of_output_channels_);
    }

    /**
     * A helper function to calculate the output dimension's size give the original
     * size of the image, padding, patch size and stride.
     * @param  image_size The size of the dimension in the original image
     * @param  pad_size_before   The amount of padding to apply to the original image
     * @param  pad_size_after   The amount of padding to apply to the original image
     * @param  patch_size The size of the dimension in the patch taken from the image
     * @param  stride     The patch's stride over the original image
     * @param  round_down Whether to round down or up when calculating the size
     * @return            The output size of the patch image
     * @remarks round_down can be used to control pooling/conv style im2col method.
     */
    inline int
    dimension_out_size(const int image_size, const int pad_size_before,
                       const int pad_size_after, const int patch_size,
                       const int stride, const bool round_down) {
        if (round_down) {
            return (image_size + pad_size_before + pad_size_after -
                    patch_size) / stride + 1;
        } else {
            return static_cast<int>(std::ceil(
                    static_cast<float>(image_size + pad_size_before +
                                       pad_size_after - patch_size) /
                    stride)) + 1;
        }
    }

    void update_layer_properties(const json &pars) {
        read_layer_param_from_json(pars, "number_of_output_channels",
                                   number_of_output_channels_);
        read_layer_param_from_json(pars, "kernel_width", kernel_width_);
        read_layer_param_from_json(pars, "kernel_height", kernel_height_);
        read_layer_param_from_json(pars, "strides_width", strides_width_);
        read_layer_param_from_json(pars, "strides_height", strides_height_);
        read_layer_param_from_json(pars, "padding_width", padding_width_);
        read_layer_param_from_json(pars, "padding_height", padding_height_);
        read_layer_param_from_json(pars, "init_in_log_space",
                                   init_in_log_space_);
        read_layer_param_from_json(pars, "normalize_input_channels",
                                   normalize_input_channels_);
        output_height_ = dimension_out_size(input_height_,
                                            padding_height_, 0,
                                            kernel_height_,
                                            strides_height_, true);
        output_width_ = dimension_out_size(input_width_, padding_width_, 0,
                                           kernel_width_,
                                           strides_width_, true);
    }

    void
    sum_convolution(const TensorType &input_tensor) {
        if (kernel_width_ == strides_width_ && kernel_height_ == strides_height_
            && padding_width_ == 0 && padding_height_ == 0 &&
            number_of_output_channels_ == 1) {
            Eigen::array<long, 5> shape{1, number_of_input_channels_, kernel_height_, kernel_width_, 1};
            auto reshaped_offsets = offsets_weights_.reshape(shape);
            auto reshaped_input = input_tensor.reshape(shape);
            element_wise_sum_ = reshaped_input + reshaped_offsets;
            return;
        }
        Eigen::array<pair<int, int>, 3> paddings{make_pair(0, 0),
                                                 make_pair(padding_height_, 0),
                                                 make_pair(padding_width_, 0)};
        auto padded_input = input_tensor.pad(paddings, 0);
        input_patches_ = padded_input.extract_image_patches(kernel_height_,
                                                            kernel_width_,
                                                            strides_height_,
                                                            strides_width_, 1,
                                                            1,
                                                            Eigen::PADDING_VALID);
        Eigen::array<long, 5> kernel_shape{number_of_output_channels_,
                                           number_of_input_channels_,
                                           kernel_height_,kernel_width_, 1};
        Eigen::array<long, 5> kernel_bcast_shape(
                {1, 1, 1, 1, input_patches_.dimension(3)});
        assert(input_patches_.dimension(0) == offsets_weights_.dimension(1));
        assert(input_patches_.dimension(1) == offsets_weights_.dimension(2));
        assert(input_patches_.dimension(2) == offsets_weights_.dimension(3));
        Eigen::array<long, 5> patches_shape{1, input_patches_.dimension(0),
                                            input_patches_.dimension(1),
                                            input_patches_.dimension(2),
                                            input_patches_.dimension(3)};
        Eigen::array<long, 5> patches_bcast_shape(
                {number_of_output_channels_, 1, 1, 1, 1});
        auto broadcasted_kernel = offsets_weights_.reshape(
                kernel_shape).broadcast(kernel_bcast_shape);
        auto broadcasted_patches = input_patches_.reshape(
                patches_shape).broadcast(patches_bcast_shape);
        element_wise_sum_ =  broadcasted_patches + broadcasted_kernel;
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
        Eigen::TensorMap<Eigen::Tensor<T, 4>> out_params_mapping(
                out_params.data() + start_idx, number_of_output_channels_, number_of_input_channels_,
                kernel_height_, kernel_width_);
        out_params_mapping = offsets_weights_;
    }

    void SetParameters(const VectorType &pars, int start_idx) override {
        VectorType &non_const_pars = const_cast<VectorType &>(pars);
        Eigen::TensorMap<const Eigen::Tensor<T, 4>> pars_mapping(
                non_const_pars.data() + start_idx, number_of_output_channels_, number_of_input_channels_, kernel_height_,
                kernel_width_);
        offsets_weights_ = pars_mapping;
        if (normalize_input_channels_) {
            Eigen::array<long, 1> input_channel_axis({1});
            Eigen::array<long, 4> logsumexp_shape{number_of_output_channels_, 1,
                                                  kernel_height_,
                                                  kernel_width_};
            Eigen::array<long, 4> logsumexp_bcast_shape{1,
                                                        number_of_input_channels_,
                                                        1, 1};
            auto max_per_input_channel = offsets_weights_.real().maximum(
                    input_channel_axis).cwiseMax(MINUS_INFINITY).eval();
            auto max_per_input_channel_broadcasted = max_per_input_channel.reshape(
                    logsumexp_shape).broadcast(logsumexp_bcast_shape);
            auto logsumexp_input_channel = ((offsets_weights_.real() -
                                             max_per_input_channel_broadcasted).exp().sum(
                    input_channel_axis).log() + max_per_input_channel).eval();
            auto logsumexp_input_channel_broadcasted = logsumexp_input_channel.reshape(
                    logsumexp_shape).broadcast(logsumexp_bcast_shape);
            offsets_weights_ = offsets_weights_.binaryExpr(
                    logsumexp_input_channel_broadcasted, MyMinusOp<T>());
        }
    }

    void InitLookup(const TensorType &input_tensor, TensorType &output_tensor,
                    LookupType &lt) override {
        if (tensor_lookup_index_ < 0 ||
            lt.TensorSize() <= tensor_lookup_index_) {
            tensor_lookup_index_ = lt.AddTensor(number_of_output_channels_,
                                                output_height_, output_width_);
        }
        if (large_tensor_lookup_index_ < 0 ||
            lt.LargeTensorSize() <= large_tensor_lookup_index_) {
            large_tensor_lookup_index_ = lt.AddLargeTensor(
                    number_of_output_channels_,
                    kernel_height_, kernel_width_, output_height_,
                    output_width_);
        }
        Eigen::array<long, 5> lookup_shape(
                {number_of_output_channels_, kernel_height_, kernel_width_,
                 output_height_, output_width_});
        Forward(input_tensor, output_tensor);
        lt.T(tensor_lookup_index_) = output_tensor;
        lt.L_T(large_tensor_lookup_index_) = logsumexp_input_channel_.reshape(lookup_shape);
    }

    void
    UpdateLookup(const TensorType &input_tensor,
                 const Matrix<bool, Dynamic, Dynamic> &input_changed,
                 TensorType &output_tensor,
                 Matrix<bool, Dynamic, Dynamic> &out_to_change,
                 LookupType &lt) override {
        LogValFromDiff(input_tensor, input_changed, output_tensor, out_to_change, lt, true);
//        InitLookup(input_tensor, output_tensor, lt);
    }

    void
    UpdateLookupFromOneHotDiff(const VectorXd &input_tensor,
                 const vector<int> &tochange,
                 const vector<double> &newconf,
                 Matrix<bool, Dynamic, Dynamic> &out_to_change,
                 TensorType &output_tensor,
                 LookupType &lt) override {
        LogValFromOneHotDiff(input_tensor, tochange, newconf, out_to_change, output_tensor,
                             lt, true);
    }


    Eigen::array<int, 3> Noutput() const override {
        return Eigen::array<int, 3>{number_of_output_channels_, output_height_,
                                    output_width_};
    }

    int Npar() const {
        return number_of_input_channels_ * number_of_output_channels_ *
               kernel_height_ * kernel_width_;
    }

    void InitRandomPars(std::default_random_engine &generator, double sigma) {
        Map<VectorType> par(offsets_weights_.data(), Npar());
        netket::RandomGaussian<Map<VectorType>, true>(par, generator, sigma);
        if (init_in_log_space_) {
            par = par.unaryExpr(
                    Eigen::internal::scalar_log_op<std::complex<double>>());
        }
        SetParameters(par, 0);
    }

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &flat_offsets_grad,
                TensorType &input_gradient) {
        DerLog(input_tensor, next_layer_gradient, flat_offsets_grad);
        if (kernel_width_ == strides_width_ && kernel_height_ == strides_height_
            && padding_width_ == 0 && padding_height_ == 0 &&
            number_of_output_channels_ == 1) {
            Eigen::TensorMap<Eigen::Tensor<T, 3>> offsets_grad(
                    flat_offsets_grad.data(), number_of_input_channels_,
                    kernel_height_, kernel_width_);
            input_gradient = offsets_grad;
//            todo logsumexp over number_of_output_channels_ will cause this to work when number_of_output_channels_ != 1
            return;
        }
        T log_zero = -std::numeric_limits<double>::infinity();
        Eigen::array<long, 6> input_shape{1, number_of_input_channels_, 1, 1,
                                          input_height_, input_width_};
        Eigen::array<long, 6> input_bcast_shape{number_of_output_channels_, 1,
                                                kernel_height_, kernel_width_,
                                                1, 1};
        Eigen::array<long, 6> offsets_shape{number_of_output_channels_,
                                            number_of_input_channels_,
                                            kernel_height_, kernel_width_, 1,
                                            1};
        Eigen::array<long, 6> offsets_bcast_shape{1, 1, 1, 1, input_height_,
                                                  input_width_};
        auto input_broadcasted = input_tensor.reshape(input_shape).broadcast(
                input_bcast_shape);
        auto offsets_broadcasted = offsets_weights_.reshape(
                offsets_shape).broadcast(offsets_bcast_shape);
        input_element_wise_sum_ = input_broadcasted + offsets_broadcasted;
        Eigen::array<long, 1> input_channel_axis{1};
        input_max_per_input_channel_ = input_element_wise_sum_.real().maximum(
                input_channel_axis).cwiseMax(MINUS_INFINITY);
        Eigen::array<long, 6> logsumexp_shape{number_of_output_channels_, 1,
                                              kernel_height_, kernel_width_,
                                              input_height_, input_width_};
        Eigen::array<long, 6> logsumexp_bcast_shape(
                {1, number_of_input_channels_, 1, 1, 1, 1});
        auto max_per_input_channel_broadcasted = input_max_per_input_channel_.reshape(
                logsumexp_shape).broadcast(logsumexp_bcast_shape);
        input_logsumexp_input_channel_ = (input_element_wise_sum_ -
                                        max_per_input_channel_broadcasted).exp().sum(
                input_channel_axis).log() + input_max_per_input_channel_;
        auto logsumexp_input_channel_broadcasted = input_logsumexp_input_channel_.reshape(
                logsumexp_shape).broadcast(logsumexp_bcast_shape);
//          todo fill spatial_gradients with log_zero according to strides and padding
        auto spatial_gradients =
                input_element_wise_sum_ - logsumexp_input_channel_broadcasted;
        Eigen::array<pair<int, int>, 3> gradient_paddings{make_pair(0, 0),
                                                          make_pair(0,
                                                                    kernel_height_ -
                                                                    strides_height_),
                                                          make_pair(0,
                                                                    kernel_width_ -
                                                                    strides_width_)};
        auto padded_next_layer_gradient = next_layer_gradient.pad(
                gradient_paddings, log_zero);
        auto next_layer_gradient_patches = padded_next_layer_gradient.extract_image_patches(
                kernel_height_ / strides_height_,
                kernel_width_ / strides_width_, 1,
                1, 1, 1,
                Eigen::PADDING_VALID).reverse(Eigen::array<long, 4>{1, 2});
        Eigen::array<long, 6> next_layer_gradient_shape{
                number_of_output_channels_,
                1,
                kernel_height_ / strides_height_,
                kernel_width_ / strides_width_,
                output_height_,
                output_width_};
        Eigen::array<long, 6> next_layer_gradient_bcast_shape{
                1, number_of_input_channels_, 1, 1, 1, 1};
        auto next_layer_gradient_patches_broadcasted = next_layer_gradient_patches.reshape(
                next_layer_gradient_shape).broadcast(
                next_layer_gradient_bcast_shape);
        input_chain_rule_before_sum_ =
                spatial_gradients + next_layer_gradient_patches_broadcasted;
        Eigen::array<long, 3> spatial_location_axis{0, 2, 3};
        input_max_per_spatial_location_ = input_chain_rule_before_sum_.real().maximum(
                spatial_location_axis).cwiseMax(MINUS_INFINITY);
        Eigen::array<long, 6> chain_rule_logsumexp_shape{
                1, number_of_input_channels_, 1, 1, input_height_,
                input_width_};
        Eigen::array<long, 6> chain_rule_logsumexp_bcast_shape(
                {number_of_output_channels_, 1, kernel_height_, kernel_width_,
                 1, 1});
        auto max_per_spatial_location_broadcasted = input_max_per_spatial_location_.reshape(
                chain_rule_logsumexp_shape).broadcast(
                chain_rule_logsumexp_bcast_shape);
        input_gradient = (input_chain_rule_before_sum_ -
                          max_per_spatial_location_broadcasted).exp().sum(
                spatial_location_axis).log() + input_max_per_spatial_location_;
    }

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &flat_offsets_grad) {
        Eigen::TensorMap<Eigen::Tensor<T, 4>> offsets_grad(
                flat_offsets_grad.data(), number_of_output_channels_,
                number_of_input_channels_, kernel_height_,
                kernel_width_);
        Eigen::array<long, 5> logsumexp_shape{number_of_output_channels_, 1,
                                              input_patches_.dimension(1),
                                              input_patches_.dimension(2),
                                              input_patches_.dimension(3)};
        Eigen::array<long, 5> logsumexp_bcast_shape(
                {1, number_of_input_channels_, 1, 1, 1});
        auto logsumexp_input_channel_broadcasted = logsumexp_input_channel_.reshape(
                logsumexp_shape).broadcast(logsumexp_bcast_shape);
        auto spatial_gradients =
                element_wise_sum_ - logsumexp_input_channel_broadcasted;
        auto next_layer_gradient_broadcasted = next_layer_gradient.reshape(
                Eigen::array<long, 5>{number_of_output_channels_, 1, 1, 1,
                                      output_height_ *
                                      output_width_}).broadcast(
                Eigen::array<long, 5>{1, number_of_input_channels_,
                                      kernel_height_, kernel_width_, 1});
        chain_rule_before_sum_ = (spatial_gradients + next_layer_gradient_broadcasted);
        Eigen::array<long, 1> spatial_location_axis{4};
        max_per_spatial_location_ = chain_rule_before_sum_.real().maximum(
                spatial_location_axis).cwiseMax(MINUS_INFINITY);
        Eigen::array<long, 5> chain_rule_logsumexp_shape{
                number_of_output_channels_, number_of_input_channels_,
                kernel_height_, kernel_width_, 1};
        Eigen::array<long, 5> chain_rule_logsumexp_bcast_shape(
                {1, 1, 1, 1, output_height_ * output_width_});
        auto max_per_spatial_location_broadcasted = max_per_spatial_location_.reshape(
                chain_rule_logsumexp_shape).broadcast(
                chain_rule_logsumexp_bcast_shape);
        offsets_grad = (chain_rule_before_sum_ -
                        max_per_spatial_location_broadcasted).exp().sum(
                spatial_location_axis).log() + max_per_spatial_location_;
    }

    void
    LogVal(const TensorType &input_tensor, TensorType &output_tensor) override {
//        todo check if calling to Forward is slower and if no replace this with Forward
        sum_convolution(input_tensor);
        Eigen::array<long, 1> input_channel_axis{1};
        max_per_input_channel_ = element_wise_sum_.real().maximum(
                input_channel_axis).cwiseMax(MINUS_INFINITY);
        Eigen::array<long, 5> logsumexp_shape{number_of_output_channels_, 1,
                                              input_patches_.dimension(1),
                                              input_patches_.dimension(2),
                                              input_patches_.dimension(3)};
        Eigen::array<long, 5> logsumexp_bcast_shape(
                {1, number_of_input_channels_, 1, 1, 1});
        auto max_per_input_channel_broadcasted = max_per_input_channel_.reshape(
                logsumexp_shape).broadcast(logsumexp_bcast_shape);
        auto logsumexp_input_channel = ((element_wise_sum_ -
                                         max_per_input_channel_broadcasted).exp().sum(
                input_channel_axis).log() + max_per_input_channel_);
        auto sum_over_spatial_kernel = logsumexp_input_channel.sum(
                Eigen::array<long, 2>{1, 2});
        output_tensor = sum_over_spatial_kernel.reshape(
                Eigen::array<long, 3>{number_of_output_channels_,
                                      output_height_, output_width_});
    }

    void
    Forward(const TensorType &input_tensor, TensorType &output_tensor) override {
        sum_convolution(input_tensor);
        Eigen::array<long, 1> input_channel_axis{1};
        max_per_input_channel_ = element_wise_sum_.real().maximum(
                input_channel_axis).cwiseMax(MINUS_INFINITY);
        Eigen::array<long, 5> logsumexp_shape{number_of_output_channels_, 1,
                                              input_patches_.dimension(1),
                                              input_patches_.dimension(2),
                                              input_patches_.dimension(3)};
        Eigen::array<long, 5> logsumexp_bcast_shape(
                {1, number_of_input_channels_, 1, 1, 1});
        auto max_per_input_channel_broadcasted = max_per_input_channel_.reshape(
                logsumexp_shape).broadcast(logsumexp_bcast_shape);
        logsumexp_input_channel_ = ((element_wise_sum_ -
                                     max_per_input_channel_broadcasted).exp().sum(
                input_channel_axis).log() + max_per_input_channel_);
        auto sum_over_spatial_kernel = logsumexp_input_channel_.sum(
                Eigen::array<long, 2>{1, 2});
        output_tensor = sum_over_spatial_kernel.reshape(
                Eigen::array<long, 3>{number_of_output_channels_,
                                      output_height_, output_width_});
    }

    void LogValFromOneHotDiff(const VectorXd &orig_input_vector,
                              const vector<int> &tochange,
                              const vector<double> &newconf,
                              Matrix<bool, Dynamic, Dynamic> &out_to_change,
                              TensorType &output_tensor,
                              const LookupType &lt) override {
        LogValFromOneHotDiff(orig_input_vector, tochange, newconf, out_to_change, output_tensor,
                             const_cast<LookupType &>(lt), false);
    }

    void LogValFromOneHotDiff(const VectorXd &orig_input_vector,
                              const vector<int> &tochange,
                              const vector<double> &newconf,
                              Matrix<bool, Dynamic, Dynamic> &out_to_change,
                              TensorType &output_tensor,
                              LookupType &lt, bool update_lookup) {
        input_vector_ = orig_input_vector;
        output_tensor = lt.T(tensor_lookup_index_);
        for (int i = 0; i < tochange.size(); ++i) {
            if (input_vector_(tochange[i]) == newconf[i]) {
                continue;
            }
            input_vector_(tochange[i]) = newconf[i];
            int w = tochange[i] % input_width_;
            int h = tochange[i] / input_width_;

            auto diff = (offsets_weights_.chip(0, 1) -
                         offsets_weights_.chip(1, 1)).reverse(
                    Eigen::array<bool, 3>{false, true, true});
            for (int j = 0;
                 (j < kernel_height_) && (h + j < output_height_); ++j) {
                for (int k = 0;
                     (k < kernel_width_) && (w + k < output_width_); ++k) {
                    out_to_change(h + j, w + k) = true;
                    auto old_val = output_tensor.chip(h + j, 1).chip(w + k, 1);
                    if (newconf[i] == 1) {
                        output_tensor.chip(h + j, 1).chip(w + k, 1) =
                                old_val + diff.chip(j, 1).chip(k, 1);
                    } else {
                        output_tensor.chip(h + j, 1).chip(w + k, 1) =
                                old_val - diff.chip(j, 1).chip(k, 1);
                    }
                }
            }
        }
        if (update_lookup){
            lt.T(tensor_lookup_index_) = output_tensor;
        }
    }

    void LogValFromDiff(const TensorType &input_tensor,
                        const Matrix<bool, Dynamic, Dynamic> &input_changed,
                        TensorType &output_tensor,
                        Matrix<bool, Dynamic, Dynamic> &out_to_change,
                        const LookupType &lt) override {
        LogValFromDiff(input_tensor, input_changed, output_tensor, out_to_change, const_cast<LookupType &>(lt), false);
    }

    void LogValFromDiff(const TensorType &input_tensor,
                        const Matrix<bool, Dynamic, Dynamic> &input_changed,
                        TensorType &output_tensor,
                        Matrix<bool, Dynamic, Dynamic> &out_to_change,
                        LookupType &lt, bool update_lookup) {
        if (strides_height_ != 1 || strides_width_ != 1) {
            out_to_change.setOnes();
            if (update_lookup){
                InitLookup(input_tensor, output_tensor, lt);
            }
            else{
                LogVal(input_tensor, output_tensor);
            }
            return;
        }
        Eigen::array<long, 1> input_channel_axis{1};
        Eigen::array<long, 2> logsumexp_shape{number_of_output_channels_, 1};
        Eigen::array<long, 2> logsumexp_bcast_shape(
                {1, number_of_input_channels_});
        output_tensor = lt.T(tensor_lookup_index_);
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                if (!input_changed(h, w)) {
                    continue;
                }
                for (int j = 0;
                     (j < kernel_height_) && (h + j < output_height_); ++j) {
                    for (int k = 0;
                         (k < kernel_width_) && (w + k < output_width_); ++k) {
                        single_element_wise_sum_ =
                                input_tensor.chip(h, 1).chip(w, 1)
                                        .reshape(
                                                logsumexp_bcast_shape).broadcast(
                                                logsumexp_shape)
                                + offsets_weights_.chip(kernel_height_ - j - 1,
                                                        2).chip(
                                        kernel_width_ - k - 1, 2);
                        single_max_per_input_channel_ = single_element_wise_sum_.real().maximum(
                                input_channel_axis).cwiseMax(MINUS_INFINITY);
                        auto max_per_input_channel_broadcasted = single_max_per_input_channel_.reshape(
                                logsumexp_shape).broadcast(
                                logsumexp_bcast_shape);
                        single_logsumexp_input_channel_ = (single_element_wise_sum_ -
                                                           max_per_input_channel_broadcasted).exp().sum(
                                input_channel_axis).log() + single_max_per_input_channel_;
                        out_to_change(h + j, w + k) = true;
                        for (int c = 0; c < number_of_output_channels_; c++) {
                            output_tensor(c, h + j, w + w) +=
                                    single_logsumexp_input_channel_(c) -
                                    lt.L_T(large_tensor_lookup_index_)(c, kernel_height_ - j - 1, kernel_width_ - w - 1, h + j,w + k);
                            if (update_lookup){
                                lt.L_T(large_tensor_lookup_index_)(c, kernel_height_ - j - 1, kernel_width_ - w - 1, h + j,w + k) = single_logsumexp_input_channel_(c);
                            }
                        }
                    }
                }
            }
        }
        if (update_lookup){
            lt.T(tensor_lookup_index_) = output_tensor;
        }
    }

    void LogVal(const TensorType &layer_input, TensorType &output_tensor,
                const LookupType &lt) override {
        output_tensor = lt.T(tensor_lookup_index_);
    }


    void to_json(json &j) const override {
        j["Name"] = "RbmSpin";
        j["number_of_output_channels"] = number_of_output_channels_;
        j["kernel_width"] = kernel_width_;
        j["kernel_height"] = kernel_height_;
        j["strides_width"] = strides_width_;
        j["strides_height"] = strides_height_;
        j["init_in_log_space"] = init_in_log_space_;
        j["normalize_input_channels"] = normalize_input_channels_;
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
