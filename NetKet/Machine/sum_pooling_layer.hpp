#ifndef NETKET_SUM_POOLING_LAYER_HPP
#define NETKET_SUM_POOLING_LAYER_HPP

#include "abstract_layer.hpp"

namespace netket {

using namespace Eigen;


template<typename T>
class SumPoolingLayer : public AbstractLayer<T> {

    int input_height_;
    int input_width_;
    int number_of_input_channels_;

public:

    using VectorType=typename AbstractLayer<T>::VectorType;
    using MatrixType=typename AbstractLayer<T>::MatrixType;
    using TensorType=typename AbstractLayer<T>::TensorType;
    using StateType=typename ConvACLayer<T>::StateType;
    using LookupType=typename ConvACLayer<T>::LookupType;


    SumPoolingLayer(const json &pars, int number_of_input_channels, int input_height, int input_width) :
            input_height_(input_height),
            input_width_(input_width), number_of_input_channels_(number_of_input_channels) {
    }

    void to_json(json &j) const override {
        j["Name"] = "SumPoolingLayer";
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
        return Eigen::array<int, 3>{number_of_input_channels_, 1, 1};
    }

    void
    LogVal(const TensorType &input_tensor, TensorType &output_tensor) override {
        output_tensor = input_tensor.sum(Eigen::array<int, 2>{1, 2}).reshape(Noutput());
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
        LogValFromDiff(input_tensor, input_changed, output_tensor, out_to_change, lt);
    }

    void
    UpdateLookupFromOneHotDiff(const Eigen::VectorXd &input_vector,
                               const std::vector<int> &tochange,
                               const std::vector<double> &newconf,
                               Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &out_to_change,
                               TensorType &output_tensor,
                               LookupType &lt) override{
        LogValFromOneHotDiff(input_vector, tochange, newconf, out_to_change, output_tensor, lt);
    };

    void DerLog(const TensorType &input_tensor, TensorType &next_layer_gradient,
                Eigen::Map<VectorType> &plat_offsets_grad,
                TensorType &layer_gradient) override {
        layer_gradient = next_layer_gradient.broadcast(Eigen::array<int, 3>{number_of_input_channels_, input_height_, input_width_});
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
        LogVal(input_tensor, output_tensor);
        out_to_change(0,0) = true;
    }

};

}

#endif //NETKET_SUM_POOLING_LAYER_HPP
