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

#ifndef NETKET_MACHINE_HH
#define NETKET_MACHINE_HH


#include <fstream>
#include <memory>

#include "abstract_machine.hh"
#include "rbm_spin.hh"
#include "rbm_spin_symm.hh"
#include "rbm_multival.hh"
#include "conv_ac.hh"

namespace netket{

template<class T> class Machine:public AbstractMachine<T>{
  using Ptype=std::unique_ptr<AbstractMachine<T>>;

  Ptype m_;

  const Hilbert & hilbert_;

  int mynode_;

public:

  using VectorType=typename AbstractMachine<T>::VectorType;
  using MatrixType=typename AbstractMachine<T>::MatrixType;
  using StateType=typename AbstractMachine<T>::StateType;
  using LookupType=typename AbstractMachine<T>::LookupType;


  Machine(const Hilbert & hilbert,const json & pars):
    hilbert_(hilbert){
    CheckInput(pars);
    Init(hilbert_,pars);
    InitParameters(pars);
  }

  Machine(const Hamiltonian & hamiltonian,const json & pars):
    hilbert_(hamiltonian.GetHilbert()){
    CheckInput(pars);
    Init(hilbert_,pars);
    InitParameters(pars);
  }

  Machine(const Graph & graph,const Hilbert & hilbert,const json & pars):
    hilbert_(hilbert){
    CheckInput(pars);
    Init(hilbert_,pars);
    Init(graph,hilbert,pars);
    InitParameters(pars);
  }

  Machine(const Graph & graph,const Hamiltonian & hamiltonian,const json & pars):
    hilbert_(hamiltonian.GetHilbert()){
    CheckInput(pars);
    Init(hilbert_,pars);
    Init(graph,hilbert_,pars);
    InitParameters(pars);
  }

  void Init(const Hilbert & hilbert,const json & pars){
    if(pars["Machine"]["Name"]=="RbmSpin"){
      m_=Ptype(new RbmSpin<T>(hilbert,pars));
    }
    else if(pars["Machine"]["Name"]=="RbmMultival"){
      m_=Ptype(new RbmMultival<T>(hilbert,pars));
    }
    else if(pars["Machine"]["Name"]=="ConvAC"){
      m_=Ptype(new ConvAC<T>(hilbert,pars));
    }
  }

  void Init(const Graph & graph,const Hilbert & hilbert,const json & pars){
    if(pars["Machine"]["Name"]=="RbmSpinSymm"){
      m_=Ptype(new RbmSpinSymm<T>(graph,hilbert,pars));
    }
  }

  void InitParameters(const json & pars){
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    if(FieldOrDefaultVal(pars["Machine"],"InitRandom",true)){
      double sigma_rand=FieldOrDefaultVal(pars["Machine"],"SigmaRand",0.1);
      m_->InitRandomPars(1232,sigma_rand);
      if(mynode==0)
      cout<<"# Machine initialized with random parameters"<<endl;
    }

    if(FieldExists(pars["Machine"],"InitFile")){
      std::string filename=pars["Machine"]["InitFile"];

      std::ifstream ifs (filename);

      if (ifs.is_open()) {
        json jmachine;
        ifs >> jmachine;
        m_->from_json(jmachine);
      }
      else{
        if(mynode==0)
        std::cerr<< "Error opening file : "<<filename<<endl;
        std::abort();
      }

      if(mynode==0)
      cout<<"# Machine initialized from file: "<<filename<<endl;
    }
  }

  void CheckInput(const json & pars){
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    if(!FieldExists(pars,"Machine")){
      if(mynode==0)
      std::cerr<<"Machine is not defined in the input"<<std::endl;
      std::abort();
    }

    if(!FieldExists(pars["Machine"],"Name")){
      if(mynode==0)
      std::cerr<<"Machine Name is not defined in the input"<<std::endl;
      std::abort();
    }

    std::set<std::string> machines=
    {
      "RbmSpin",
      "RbmSpinSymm",
      "RbmMultival",
      "ConvAC"
    };

    const auto name=pars["Machine"]["Name"];

    if(machines.count(name)==0){
      std::cerr<<"Machine "<<name<<" not found."<<std::endl;
      std::abort();
    }
  }

  //returns the number of variational parameters
  int Npar()const{
    return m_->Npar();
  }

  int Nvisible()const{
    return m_->Nvisible();
  }

  //Initializes Lookup tables
  void InitLookup(const VectorXd & v,LookupType & lt){
    return m_->InitLookup(v,lt);
  }

  //Updates Lookup tables
  void UpdateLookup(const VectorXd & v,const vector<int>  & tochange,
    const vector<double> & newconf,LookupType & lt){

    return m_->UpdateLookup(v,tochange,newconf,lt);
  }

  VectorType DerLog(const VectorXd & v){
    return m_->DerLog(v);
  }

  MatrixType DerLogDiff(const VectorXd & v,
    const vector<vector<int> >  & toflip,
    const vector<vector<double>> & newconf){

    return m_->DerLogDiff(v,toflip,newconf);
  }

  VectorType GetParameters(){
    return m_->GetParameters();
  }

  void SetParameters(const VectorType & pars){
    return m_->SetParameters(pars);
  }

  //Value of the logarithm of the wave-function
  T LogVal(const VectorXd & v){
    return m_->LogVal(v);
  }

  //Value of the logarithm of the wave-function
  //using pre-computed look-up tables for efficiency
  T LogVal(const VectorXd & v,LookupType & lt){
    return m_->LogVal(v,lt);
  }

  //Difference between logarithms of values, when one or more visible variables are being flipped
  VectorType LogValDiff(const VectorXd & v,
    const vector<vector<int> >  & toflip,
    const vector<vector<double>> & newconf){

    return m_->LogValDiff(v,toflip,newconf);
  }

  //Difference between logarithms of values, when one or more visible variables are being flipped
  //Version using pre-computed look-up tables for efficiency on a small number of spin flips
  T LogValDiff(const VectorXd & v,const vector<int>  & toflip,
      const vector<double> & newconf,const LookupType & lt){

    return m_->LogValDiff(v,toflip,newconf,lt);
  }

  void InitRandomPars(int seed,double sigma){
    return m_->InitRandomPars(seed,sigma);
  }

  const Hilbert& GetHilbert()const{
    return hilbert_;
  }

  void to_json(json &j)const{
    m_->to_json(j);
  }

  void from_json(const json&j){
    m_->from_json(j);
  }
};
}
#endif
