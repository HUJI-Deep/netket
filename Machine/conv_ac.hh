#ifndef NETKET_CONV_AC_HH
#define NETKET_CONV_AC_HH

#include "abstract_machine.hh"

namespace netket{

using namespace std;
using namespace Eigen;

template<typename T> class ConvAC : public AbstractMachine<T> {
public:


    using VectorType=typename AbstractMachine<T>::VectorType;
    using MatrixType=typename AbstractMachine<T>::MatrixType;

    int number_of_visible_units_;
    int my_mpi_node_;
    unsigned int num_of_layers;

    const Hilbert & hilbert_;


    ConvAC(const Hilbert & hilbert,const json & pars):
        number_of_visible_units_(hilbert.Size()),
        hilbert_(hilbert){

        from_json(pars);

        MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_node_);

        if(my_mpi_node_==0){
            cout<<"# ConvAC Initizialized with nvisible = "<<number_of_cisible_units_<<endl;
        }
    }

    int Npar() const override {
        return 0;
    }

    VectorType GetParameters() override {
        return nullptr;
    }

    void SetParameters(const VectorType &pars) override {

    }

    void InitRandomPars(int seed, double sigma) override {

    }

    int Nvisible() const override {
        return 0;
    }

    T LogVal(const VectorXd &v) override {
        return nullptr;
    }

    T LogVal(const VectorXd &v, LookupType &lt) override {
        return nullptr;
    }

    void InitLookup(const VectorXd &v, LookupType &lt) override {

    }

    void UpdateLookup(const VectorXd &v, const vector<int> &tochange,
                      const vector<double> &newconf, LookupType &lt) override {

    }

    VectorType
    LogValDiff(const VectorXd &v, const vector<vector<int> > &tochange,
               const vector<vector<double>> &newconf) override {
        return nullptr;
    }

    T LogValDiff(const VectorXd &v, const vector<int> &toflip,
                 const vector<double> &newconf, const LookupType &lt) override {
        return nullptr;
    }

    VectorType DerLog(const VectorXd &v) override {
        return nullptr;
    }

    void to_json(json &j) const override {

    }

    void from_json(const json & pars) override {
        if(pars.at("Machine").at("Name")!="RbmSpin"){
            if(my_mpi_node_==0){
                std::cerr<<"# Error while constructing RbmSpin from Json input"<<endl;
            }
            std::abort();
        }
        if(FieldExists(pars["Machine"],"Nvisible")){
          number_of_visible_units_=pars["Machine"]["Nvisible"];
        }
        if(number_of_visible_units_!=hilbert_.Size()){
          if(mynode_==0){
            cerr<<"# Number of visible units is incompatible with given Hilbert space"<<endl;
          }
          std::abort();
        }
        if(FieldExists(pars["Machine"],"Layers")){
          if(mynode_==0){
            cerr<<"# ConvAC Machines must have layers attribute"<<endl;
          }
          std::abort();
        }
    }
}
}

#endif //NETKET_CONV_AC_HH
