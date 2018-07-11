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

#include "abstract_layer.hpp"

namespace netket {

using namespace Eigen;


template<typename T>
class ToOneHotLayer : public AbstractLayer<T> {

    int input_height_;
    int input_width_;

    Tensor<T, 2> log_zero_{};
    Tensor<T, 2> log_one_{};

public:

    using VectorType=typename AbstractLayer<T>::VectorType;
    using MatrixType=typename AbstractLayer<T>::MatrixType;
    using TensorType=typename AbstractLayer<T>::TensorType;
    using StateType=typename ConvACLayer<T>::StateType;
    using LookupType=typename ConvACLayer<T>::LookupType;


    ToOneHotLayer(const json &pars, int input_height, int input_width) :
            input_height_(input_height),
            input_width_(input_width) {
        log_zero_  = Tensor<T, 2>(input_height, input_width);
        log_one_  = Tensor<T, 2>(input_height, input_width);
        log_zero_.setConstant(-std::numeric_limits<double>::infinity());
        log_one_.setZero();
    }

    void to_json(json &j) const override {
        j["Name"] = "ToOneHotLayer";
    }

    int Npar() const override {
        return 0;
    }

    void GetParameters(VectorType &out_params, int start_idx) const override {

    }

    void SetParameters(const VectorType &pars, int start_idx) override {

    }

    void InitRandomPars(std::default_random_engine &generator,
                        double sigma) override {

    }

    Eigen::array<int, 3> Noutput() const override {
        return Eigen::array<int, 3>{2, input_height_,
                                    input_width_};
    }

    void
    LogVal(const TensorType &input_tensor, TensorType &output_tensor) override {
        assert(input_tensor.dimension(0) == 1);
        auto two_d_input = input_tensor.chip(0,0);
        auto is_one = two_d_input == T{1};
        output_tensor.chip(0, 0) = is_one.select(log_one_, log_zero_);
        output_tensor.chip(1, 0) = is_one.select(log_zero_, log_one_);
    }

    void
    Forward(const TensorType &input_tensor, TensorType &output_tensor) override {
        LogVal(input_tensor, output_tensor);
    }

    void LogVal(const TensorType &layer_input, TensorType &output_tensor,
                const LookupType &lt) override {
        LogVal(layer_input, output_tensor);
    }

    void InitLookup(const TensorType &input_tensor, TensorType &output_tensor, LookupType &lt) override {
        LogVal(input_tensor, output_tensor);
    }

    void
    UpdateLookup(const TensorType &input_tensor, const Matrix<bool, Dynamic, Dynamic> &input_changed,
                 TensorType &output_tensor, Matrix<bool, Dynamic, Dynamic> &out_to_change, LookupType &lt) override {
        LogVal(input_tensor, output_tensor);
    }

    void
    UpdateLookupFromOneHotDiff(const Eigen::VectorXd &input_tensor,
                               const std::vector<int> &tochange,
                               const std::vector<double> &newconf,
                               Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &out_to_change,
                               TensorType &output_tensor,
                               LookupType &lt) override{
        assert(false);
    };

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &plat_offsets_grad,
                TensorType &layer_gradient) override {
        assert(false);
    }

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &plat_offsets_grad) override {

    }

    void LogValFromOneHotDiff(const VectorXd &orig_input_vector,
                              const vector<int> &tochange,
                              const vector<double> &newconf,
                              Matrix<bool, Dynamic, Dynamic> &out_to_change,
                              TensorType &output_tensor, const LookupType &lt) override{
        assert(false);
    }

    void LogValFromDiff(const TensorType &input_tensor, const Matrix<bool, Dynamic, Dynamic> &input_changed,
                        TensorType &output_tensor, Matrix<bool, Dynamic, Dynamic> &out_to_change,
                        const LookupType &lt) override{
        assert(false);
    }

};

}