#ifndef NETKET_CONV_AC_HH
#define NETKET_CONV_AC_HH

#include "abstract_machine.hpp"
#include "conv_ac_layer.hpp"

namespace netket {

using namespace std;
using namespace Eigen;

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
    vector<unique_ptr<AbstractLayer<T>>> layers_;

    const Hilbert &hilbert_;


    ConvAC(const Hilbert &hilbert, const json &pars) :
            number_of_visible_units_(hilbert.Size()),
            hilbert_(hilbert) {
        from_json(pars);
        InirMPI();
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
        for(auto const& layer: layers_) {
            num_of_params += layer->Npar();
        }
        return num_of_params;
    }

    VectorType GetParameters() override {
        VectorType parameters(Npar());
        int start_idx = 0;
        for(auto const& layer: layers_) {
            layer->GetParameters(parameters, start_idx);
            start_idx += layer->Npar();
        }
        return parameters;
    }

    void SetParameters(const VectorType &parameters) override {
        int start_idx = 0;
        for(auto const& layer: layers_) {
            layer->SetParameters(parameters, start_idx);
            start_idx += layer->Npar();
        }
    }

    void InitRandomPars(int seed, double sigma) override {
        std::default_random_engine generator(seed);
        for(auto const& layer: layers_) {
            layer->InitRandomPars(generator, sigma);
        }
    }

    int Nvisible() const override {
        return number_of_visible_units_;
    }

    T LogVal(const VectorXd &v) override {
        return T{};
    }

    T LogVal(const VectorXd &v, LookupType &lt) override {
        return T{};
    }

    void InitLookup(const VectorXd &v, LookupType &lt) override {
        for(auto const& layer: layers_) {
            layer->InitLookup(v, lt);
        }
    }

    void UpdateLookup(const VectorXd &v, const vector<int> &tochange,
                      const vector<double> &newconf, LookupType &lt) override {
        for(auto const& layer: layers_) {
            layer->UpdateLookup(v, tochange, newconf, lt);
        }
    }

    VectorType
    LogValDiff(const VectorXd &orig_vector, const vector<vector<int> > &tochange,
               const vector<vector<double>> &newconf) override {
        T orig_log_value = LogVal(orig_vector);
        const std::size_t nconn = tochange.size();
        VectorType logvaldiffs = VectorType::Zero(nconn);
        for (std::size_t k = 0; k < nconn; k++) {
            VectorXd new_vector(orig_vector);
            for (std::size_t s = 0; s < tochange[k].size(); s++) {
                new_vector[tochange[k][s]] = newconf[k][s];
            }
            logvaldiffs(k) = LogVal(new_vector) - orig_log_value;
        }
        return logvaldiffs;
    }

    T LogValDiff(const VectorXd &v, const vector<int> &tochange,
                 const vector<double> &newconf, const LookupType &lt) override {
        T orig_log_value = LogVal(v);
        VectorXd new_vector(v);
        for (std::size_t s = 0; s < tochange.size(); s++) {
            new_vector[tochange[s]] = newconf[s];
        }
        return LogVal(new_vector) - orig_log_value;
    }

    VectorType DerLog(const VectorXd &v) override {
        return VectorType{};
    }

    void to_json(json &j) const override {
        j["Machine"]["Name"] = "ConvAC";
        j["Machine"]["Nvisible"] = number_of_visible_units_;
        j["layers"] = json::array();
        for(auto const& layer: layers_) {
            json layer_node;
            layer->to_json(layer_node);
            j["layers"].push_back(layer_node);
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
        }
        else{
            visible_width_ = 1;
        }
        if (FieldExists(pars["Machine"], "visible_height")) {
            visible_height_ = pars["Machine"]["visible_height"];
        }
        else{
            visible_height_ = hilbert_.Size();
        }
        number_of_visible_units_ = visible_width_ * visible_height_;
        if (number_of_visible_units_ != hilbert_.Size()) {
            if (my_mpi_node_ == 0) {
                cerr << "# Number of visible units is incompatible with given Hilbert space" << endl;
            }
            std::abort();
        }
        if (FieldExists(pars["Machine"], "Layers")) {
            if (my_mpi_node_ == 0) {
                cerr << "# ConvAC Machines must have layers attribute" << endl;
            }
            std::abort();
        }
        int input_dimension = number_of_visible_units_;
        for(auto const& layer: pars["Machine"]["Layers"]) {
            layers_.push_back(std::unique_ptr<ConvACLayer<T>>(new ConvACLayer<T>(layer, input_dimension)));
            input_dimension = layers_.back()->Noutput();
        }
    }
};
}

#endif //NETKET_CONV_AC_HH
