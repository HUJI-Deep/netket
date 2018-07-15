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

#ifndef NETKET_CONV_AC_HH
#define NETKET_CONV_AC_HH

#include "abstract_machine.hpp"
#include "conv_ac_layer.hpp"
#include "to_one_hot_layer.hpp"
#include "add_bias_layer.hpp"
#include "sum_pooling_layer.hpp"
#include <cmath>

namespace netket {

using namespace std;
using namespace Eigen;

template<typename T>
T normalized(T number);

template<>
std::complex<double> normalized(std::complex<double> number){
    double imag = std::fmod(number.imag(), 2 * M_PI);
    if (imag < 0){
        imag = 2 * M_PI + imag;
    }
    return std::complex<double>{number.real(), imag};
}

template< >
double normalized(double number){
    return number;
}

template<typename T>
class ConvAC : public AbstractMachine<T> {
public:


    using VectorType=typename AbstractMachine<T>::VectorType;
    using MatrixType=typename AbstractMachine<T>::MatrixType;
    using StateType = typename AbstractMachine<T>::StateType;
    using LookupType = typename AbstractMachine<T>::LookupType;
    using TensorType=typename AbstractLayer<T>::TensorType;

    int number_of_visible_units_{};
    int visible_width_{};
    int visible_height_{};
    int my_mpi_node_{};
    bool is_first_layer_one_hot_{};
    bool is_second_layer_add_bias_{};
    bool fast_lookup_{};
    vector<unique_ptr<AbstractLayer<T>>> layers_;
    vector<TensorType> values_tensors_;
    vector<TensorType> input_gradient_tensors_;
    Eigen::Matrix<T, Dynamic, 1> input_buffer_;
    vector<Matrix<bool, Dynamic, Dynamic>> input_changed_;
    LookupType lt_;

    const Hilbert &hilbert_;


    ConvAC(const Hilbert &hilbert, const json &pars) :
            number_of_visible_units_(hilbert.Size()),
            hilbert_(hilbert) {
        InirMPI();
        from_json(pars);
    }

    void InirMPI() {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_node_);
        if (my_mpi_node_ == 0) {
            cout << "# ConvAC Initizialized with nvisible = "
                 << number_of_visible_units_ << endl;
        }
    }

    int Npar() const override {
        int num_of_params = 0;
        for (auto const &layer: layers_) {
            num_of_params += layer->Npar();
        }
        return num_of_params;
    }

    VectorType GetParameters() override {
        VectorType parameters(Npar());
        int start_idx = 0;
        for (auto const &layer: layers_) {
            layer->GetParameters(parameters, start_idx);
            start_idx += layer->Npar();
        }
        return parameters;
    }

    void SetParameters(const VectorType &parameters) override {
        int start_idx = 0;
        for (auto const &layer: layers_) {
            layer->SetParameters(parameters, start_idx);
            start_idx += layer->Npar();
        }
    }

    void InitRandomPars(int seed, double sigma) override {
        std::default_random_engine generator(seed);
        for (auto const &layer: layers_) {
            layer->InitRandomPars(generator, sigma);
        }
    }

    int Nvisible() const override {
        return number_of_visible_units_;
    }

    T LogVal(const VectorXd &v) override {
        input_buffer_ = v;
        TensorMap<Tensor<T, 2, RowMajor>> input_tensor_swaped(input_buffer_.data(), visible_height_,
                                                              visible_width_);
        values_tensors_[0] = input_tensor_swaped.swap_layout().reshape(Eigen::array<long, 3>{1, visible_height_, visible_width_});
        for (int i = 0; i < layers_.size(); ++i) {
            layers_[i]->LogVal(values_tensors_[i], values_tensors_[i+1]);
//            InfoMessage() << "Layer " << i << " stats: max_abs = " << values_tensors_[i+1].abs().maximum() << ", min_abs = " << values_tensors_[i+1].abs().minimum()
//                          << ", max_img = " << values_tensors_[i+1].imag().maximum()<< endl;
        }
        Eigen::Tensor<T, 0> sum_result(
                values_tensors_[values_tensors_.size() - 1].sum());
        return normalized(sum_result(0));
    }

    T LogVal(const VectorXd &v, const LookupType &lt) override {
        return lt.V(0)(0);
    }

    void InitLookup(const VectorXd &v, LookupType &lt) override {
        if (lt.VectorSize() == 0) {
            lt.AddVector(1);
        }
        input_buffer_ = v;
        TensorMap<Tensor<T, 2, RowMajor>> input_tensor_swaped(input_buffer_.data(), visible_height_,
                                                              visible_width_);
        values_tensors_[0] = input_tensor_swaped.swap_layout().reshape(Eigen::array<long, 3>{1, visible_height_, visible_width_});
        for (int i = 0; i < layers_.size(); ++i) {
            layers_[i]->InitLookup(values_tensors_[i], values_tensors_[i+1], lt);
        }
        Eigen::Tensor<T, 0> sum_result(
                values_tensors_[values_tensors_.size() - 1].sum());
        lt.V(0)(0) = normalized(sum_result(0));
    }

    void UpdateLookup(const VectorXd &orig_vector, const vector<int> &tochange,
                      const vector<double> &newconf, LookupType &lt) override {
        T orig_log_value = lt.V(0)(0);
        if (is_first_layer_one_hot_ && fast_lookup_ ) {
            int first_layer_to_calc_diff;
            if (is_second_layer_add_bias_){
                VectorXd new_vector(orig_vector);
                for (std::size_t s = 0; s < tochange.size(); s++) {
                    new_vector[tochange[s]] = newconf[s];
                }
                input_buffer_ = new_vector;
                TensorMap<Tensor<T, 2, RowMajor>> input_tensor_swaped(input_buffer_.data(), visible_height_,
                                                                      visible_width_);
                values_tensors_[0] = input_tensor_swaped.swap_layout().reshape(Eigen::array<long, 3>{1, visible_height_, visible_width_});
                input_changed_[0].setZero();
                input_changed_[1].setZero();
                for (int i = 0; i < tochange.size(); ++i) {
                    int h = tochange[i] % visible_height_;
                    int w = tochange[i] / visible_height_;
                    input_changed_[0](h, w) = true;
                }
                layers_[0]->UpdateLookup(values_tensors_[0], input_changed_[0],
                                           values_tensors_[1], input_changed_[1], lt);
                first_layer_to_calc_diff = 1;
            }
            else{
                first_layer_to_calc_diff = 2;
                input_changed_[2].setZero();
                layers_[1]->UpdateLookupFromOneHotDiff(orig_vector, tochange, newconf,
                                                 input_changed_[2], values_tensors_[2],
                                                 lt);
            }
            for (int i = first_layer_to_calc_diff; i < layers_.size(); ++i) {
                input_changed_[i+1].setZero();
                layers_[i]->UpdateLookup(values_tensors_[i], input_changed_[i],
                                           values_tensors_[i+1], input_changed_[i+1], lt);
            }
        }

        VectorXd new_vector(orig_vector);
        for (std::size_t s = 0; s < tochange.size(); s++) {
            new_vector[tochange[s]] = newconf[s];
        }
        InitLookup(new_vector, lt);
    }

    VectorType
    LogValDiff(const VectorXd &orig_vector,
               const vector<vector<int> > &tochange,
               const vector<vector<double>> &newconf) override {
        InitLookup(orig_vector, lt_);
        const std::size_t nconn = tochange.size();
        VectorType logvaldiffs = VectorType::Zero(nconn);
        for (std::size_t k = 0; k < nconn; k++) {
            logvaldiffs(k) = LogValDiff(orig_vector, tochange[k], newconf[k], lt_);
        }
        return logvaldiffs;
    }

    T LogValDiff(const VectorXd &orig_vector, const vector<int> &tochange,
                 const vector<double> &newconf, const LookupType &lt) override {
        T orig_log_value = lt.V(0)(0);
        if (is_first_layer_one_hot_ && fast_lookup_ ) {
            int first_layer_to_calc_diff;
            if (is_second_layer_add_bias_){
                VectorXd new_vector(orig_vector);
                for (std::size_t s = 0; s < tochange.size(); s++) {
                    new_vector[tochange[s]] = newconf[s];
                }
                input_buffer_ = new_vector;
                for (std::size_t s = 0; s < tochange.size(); s++) {
                    input_buffer_[tochange[s]] = newconf[s];
                }
                TensorMap<Tensor<T, 2, RowMajor>> input_tensor_swaped(input_buffer_.data(), visible_height_,
                                                                      visible_width_);
                values_tensors_[0] = input_tensor_swaped.swap_layout().reshape(Eigen::array<long, 3>{1, visible_height_, visible_width_});
                input_changed_[0].setZero();
                input_changed_[1].setZero();
                for (int i = 0; i < tochange.size(); ++i) {
                    int h = tochange[i] % visible_height_;
                    int w = tochange[i] / visible_height_;
                    input_changed_[0](h, w) = true;
                }
                layers_[0]->LogValFromDiff(values_tensors_[0], input_changed_[0],
                                           values_tensors_[1], input_changed_[1], lt);
                first_layer_to_calc_diff = 1;
            }
            else{
                first_layer_to_calc_diff = 2;
                input_changed_[2].setZero();
                layers_[1]->LogValFromOneHotDiff(orig_vector, tochange, newconf,
                                                 input_changed_[2], values_tensors_[2],
                                                 lt);
            }
            for (int i = first_layer_to_calc_diff; i < layers_.size(); ++i) {
                input_changed_[i+1].setZero();
                layers_[i]->LogValFromDiff(values_tensors_[i], input_changed_[i],
                                           values_tensors_[i+1], input_changed_[i+1], lt);
            }
            Eigen::Tensor<T, 0> sum_result(
                    values_tensors_[values_tensors_.size() - 1].sum());
            return normalized(sum_result(0) - orig_log_value);
        }


        VectorXd new_vector(orig_vector);
        for (std::size_t s = 0; s < tochange.size(); s++) {
            new_vector[tochange[s]] = newconf[s];
        }
        return LogVal(new_vector) - orig_log_value;
    }

    VectorType DerLog(const VectorXd &v) override {
        input_buffer_ = v;
        TensorMap<Tensor<T, 2, RowMajor>> input_tensor_swaped(input_buffer_.data(), visible_height_,
                                                             visible_width_);
        values_tensors_[0] = input_tensor_swaped.swap_layout().reshape(Eigen::array<long, 3>{1, visible_height_, visible_width_});
        for (int i = 0; i < layers_.size(); ++i) {
            layers_[i]->Forward(values_tensors_[i], values_tensors_[i+1]);
        }
        VectorType all_layers__gradient(Npar());
        int params_id = Npar();
        Map<VectorType> params_gradient{NULL, 0};
        for (int i = layers_.size() - 1; i > 0; --i) {
            int layer_num_of_params = layers_[i]->Npar();
            params_id -= layer_num_of_params;
            new(&params_gradient) Map<VectorType>(
                    all_layers__gradient.data() + params_id,
                    layer_num_of_params);
            if (i > 1 || !is_first_layer_one_hot_ ){
                layers_[i]->DerLog(values_tensors_[i],
                                   input_gradient_tensors_[i], params_gradient,
                                   input_gradient_tensors_[i - 1]);
            }
            else{
                layers_[i]->DerLog(values_tensors_[i],
                                   input_gradient_tensors_[i], params_gradient);
            }
        }
        int layer_num_of_params = layers_[0]->Npar();
        new(&params_gradient) Map<VectorType>(
                all_layers__gradient.data(), layer_num_of_params);
        layers_[0]->DerLog(values_tensors_[0], input_gradient_tensors_[0],
                           params_gradient);
        return all_layers__gradient.unaryExpr(
                Eigen::internal::scalar_exp_op<std::complex<double>>());
    }

    void to_json(json &j) const override {
        j["Machine"]["Name"] = "ConvAC";
        j["Machine"]["Nvisible"] = number_of_visible_units_;
        j["Machine"]["visible_width"] = visible_width_;
        j["Machine"]["visible_height"] = visible_height_;
        j["Machine"]["fast_lookup"] = fast_lookup_;
        j["Machine"]["Layers"] = json::array();
        for (auto const &layer: layers_) {
            json layer_node;
            layer->to_json(layer_node);
            j["Machine"]["Layers"].push_back(layer_node);
        }
    }

    void from_json(const json &pars) override {
        if (pars.at("Machine").at("Name") != "ConvAC") {
            if (my_mpi_node_ == 0) {
                std::cerr
                        << "# Error while constructing RbmSpin from Json input"
                        << endl;
            }
            std::abort();
        }
        if (FieldExists(pars["Machine"], "visible_width")) {
            visible_width_ = pars["Machine"]["visible_width"];
        } else {
            visible_width_ = 1;
        }
        if (FieldExists(pars["Machine"], "visible_height")) {
            visible_height_ = pars["Machine"]["visible_height"];
        } else {
            visible_height_ = hilbert_.Size();
        }
        if (FieldExists(pars["Machine"], "fast_lookup")) {
            fast_lookup_ = pars["Machine"]["fast_lookup"];
        } else {
            fast_lookup_ = true;
        }

        number_of_visible_units_ = visible_width_ * visible_height_;
        if (number_of_visible_units_ != hilbert_.Size()) {
            if (my_mpi_node_ == 0) {
                cerr
                        << "# Number of visible units is incompatible with given Hilbert space"
                        << endl;
            }
            std::abort();
        }
        if (!FieldExists(pars["Machine"], "Layers")) {
            if (my_mpi_node_ == 0) {
                cerr << "# ConvAC Machines must have layers attribute" << endl;
            }
            std::abort();
        }
        input_buffer_ = Eigen::Matrix<T, Dynamic, 1>(number_of_visible_units_);
        int input_dimension = 1;
        int input_height = visible_height_;
        int input_width = visible_width_;
        values_tensors_.clear();
        values_tensors_.push_back(
                TensorType(input_dimension, input_height, input_width));
        is_first_layer_one_hot_ = false;
        is_second_layer_add_bias_ = false;
        int i = 0;
        layers_.clear();
        input_gradient_tensors_.clear();
        input_changed_.clear();
        input_changed_.push_back(Matrix<bool, Dynamic, Dynamic>(input_height, input_width));
        for (auto const &layer: pars["Machine"]["Layers"]) {
            if (FieldVal(layer, "Name") == "ConvACLayer") {
                layers_.push_back(std::unique_ptr<ConvACLayer<T>>(
                        new ConvACLayer<T>(layer, input_dimension, input_height,
                                           input_width)));
            } else if (FieldVal(layer, "Name") == "ToOneHotLayer") {
                if (i == 0) {
                    is_first_layer_one_hot_ = true;
                }
                layers_.push_back(std::unique_ptr<ToOneHotLayer<T>>(
                        new ToOneHotLayer<T>(layer, input_height,
                                             input_width)));
            }
            else if (FieldVal(layer, "Name") == "AddBiasLayer") {
                if (i == 1) {
                    is_second_layer_add_bias_ = true;
                }
                layers_.push_back(std::unique_ptr<AddBiasLayer<T>>(
                        new AddBiasLayer<T>(layer, input_dimension, input_height,
                                             input_width)));
            }
            else if (FieldVal(layer, "Name") == "SumPoolingLayer") {
                layers_.push_back(std::unique_ptr<SumPoolingLayer<T>>(
                        new SumPoolingLayer<T>(layer, input_dimension, input_height,
                                            input_width)));
            }
            else {
                if (my_mpi_node_ == 0) {
                    cerr << "Unknown layers type : " << FieldVal(layer, "Name")
                         << endl;
                }
                std::abort();
            }
            if (my_mpi_node_ == 0) {
                std::cout << "Adding Layer with " << layers_.back()->Npar()
                          << " params" << std::endl;
            }
            auto output_dims = layers_.back()->Noutput();
            input_dimension = output_dims[0];
            input_height = output_dims[1];
            input_width = output_dims[2];
            values_tensors_.push_back(
                    TensorType(input_dimension, input_height, input_width));
            input_gradient_tensors_.push_back(
                    TensorType(input_dimension, input_height, input_width));
            input_changed_.push_back(Matrix<bool, Dynamic, Dynamic>(input_height, input_width));
            ++i;
        }
        input_gradient_tensors_.back().setZero();
    }
};
}

#endif //NETKET_CONV_AC_HH
