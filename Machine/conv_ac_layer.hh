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

#ifndef NETKET_CONV_AC_LAYER_HH
#define NETKET_CONV_AC_LAYER_HH

namespace netket{

using namespace std;
using namespace Eigen;


template<typename T> class ConvACLayer : public AbstractLayer<T>{

  using VectorType=typename AbstractLayer<T>::VectorType;
  using MatrixType=typename AbstractLayer<T>::MatrixType;

  int number_of_input_channels_;
  int number_of_output_channels_;
  int kernel_width_;
  int kernel_height_;
  int strides_width_;
  int strides_height_;

  MatrixType offsets_weights_;


public:

  using StateType=typename AbstractMachine<T>::StateType;
  using LookupType=typename AbstractMachine<T>::LookupType;

  ConvACLayer(const json & pars, int number_of_input_channels):
    number_of_input_channels_(number_of_input_channels){
    from_json(pars);
  }

  void Init(){
    W_.resize(nv_,nh_);
    a_.resize(nv_);
    b_.resize(nh_);

    thetas_.resize(nh_);
    lnthetas_.resize(nh_);
    thetasnew_.resize(nh_);
    lnthetasnew_.resize(nh_);

    npar_=nv_*nh_;

    if(usea_){
      npar_+=nv_;
    }
    else{
      a_.setZero();
    }

    if(useb_){
      npar_+=nh_;
    }
    else{
      b_.setZero();
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if(mynode_==0){
      cout<<"# RBM Initizialized with nvisible = "<<nv_<<" and nhidden = "<<nh_<<endl;
      cout<<"# Using visible bias = "<<usea_<<endl;
      cout<<"# Using hidden bias  = "<<useb_<<endl;
    }
  }

  int Nvisible()const{
    return number_of_input_channels_;
  }

  int Npar()const{
    return number_of_input_channels_ * number_of_output_channels_ * kernel_height_ * kernel_width_;
  }

  void InitRandomPars(int seed,double sigma){

    VectorType par(npar_);

    RandomGaussian(par,seed,sigma);

    SetParameters(par);
  }


  void InitLookup(const VectorXd & v,LookupType & lt){
    if(lt.VectorSize()==0){
      lt.AddVector(b_.size());
    }
    if(lt.V(0).size()!=b_.size()){
      lt.V(0).resize(b_.size());
    }

    lt.V(0)=(W_.transpose()*v+b_);
  }

  void UpdateLookup(const VectorXd & v,const vector<int>  & tochange,
    const vector<double> & newconf,LookupType & lt){

    if(tochange.size()!=0){

      for(std::size_t s=0;s<tochange.size();s++){
        const int sf=tochange[s];
        lt.V(0)+=W_.row(sf)*(newconf[s]-v(sf));
      }

    }
  }

  VectorType DerLog(const VectorXd & v){
    VectorType der(npar_);

    int k=0;

    if(usea_){
      for(;k<nv_;k++){
        der(k)=v(k);
      }
    }

    RbmSpin::tanh(W_.transpose()*v+b_,lnthetas_);

    if(useb_){
      for(int p=0;p<nh_;p++){
        der(k)=lnthetas_(p);
        k++;
      }
    }

    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        der(k)=lnthetas_(j)*v(i);
        k++;
      }
    }
    return der;
  }


  VectorType GetParameters(){

    VectorType pars(npar_);

    int k=0;

    if(usea_){
      for(;k<nv_;k++){
        pars(k)=a_(k);
      }
    }

    if(useb_){
      for(int p=0;p<nh_;p++){
        pars(k)=b_(p);
        k++;
      }
    }

    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        pars(k)=W_(i,j);
        k++;
      }
    }

    return pars;
  }


  void SetParameters(const VectorType & pars){
    int k=0;

    if(usea_){
      for(;k<nv_;k++){
        a_(k)=pars(k);
      }
    }

    if(useb_){
      for(int p=0;p<nh_;p++){
        b_(p)=pars(k);
        k++;
      }
    }

    for(int i=0;i<nv_;i++){
      for(int j=0;j<nh_;j++){
        W_(i,j)=pars(k);
        k++;
      }
    }
  }

  //Value of the logarithm of the wave-function
  T LogVal(const VectorXd & v){
    RbmSpin::lncosh(W_.transpose()*v+b_,lnthetas_);

    return (v.dot(a_)+lnthetas_.sum());
  }

  //Value of the logarithm of the wave-function
  //using pre-computed look-up tables for efficiency
  T LogVal(const VectorXd & v,LookupType & lt){
    RbmSpin::lncosh(lt.V(0),lnthetas_);

    return (v.dot(a_)+lnthetas_.sum());
  }

  //Difference between logarithms of values, when one or more visible variables are being flipped
  VectorType LogValDiff(const VectorXd & v,
    const vector<vector<int> >  & tochange,
    const vector<vector<double>> & newconf){


    const std::size_t nconn=tochange.size();
    VectorType logvaldiffs=VectorType::Zero(nconn);

    thetas_=(W_.transpose()*v+b_);
    RbmSpin::lncosh(thetas_,lnthetas_);

    T logtsum=lnthetas_.sum();

    for(std::size_t k=0;k<nconn;k++){

      if(tochange[k].size()!=0){

        thetasnew_=thetas_;

        for(std::size_t s=0;s<tochange[k].size();s++){
          const int sf=tochange[k][s];

          logvaldiffs(k)+=a_(sf)*(newconf[k][s]-v(sf));

          thetasnew_+=W_.row(sf)*(newconf[k][s]-v(sf));
        }

        RbmSpin::lncosh(thetasnew_,lnthetasnew_);
        logvaldiffs(k)+=lnthetasnew_.sum() - logtsum;

      }
    }
    return logvaldiffs;
  }

  //Difference between logarithms of values, when one or more visible variables are being flipped
  //Version using pre-computed look-up tables for efficiency on a small number of spin flips
  T LogValDiff(const VectorXd & v,const vector<int>  & tochange,
    const vector<double> & newconf,const LookupType & lt){

    T logvaldiff=0.;

    if(tochange.size()!=0){

      RbmSpin::lncosh(lt.V(0),lnthetas_);

      thetasnew_=lt.V(0);

      for(std::size_t s=0;s<tochange.size();s++){
        const int sf=tochange[s];

        logvaldiff+=a_(sf)*(newconf[s]-v(sf));

        thetasnew_+=W_.row(sf)*(newconf[s]-v(sf));
      }

      RbmSpin::lncosh(thetasnew_,lnthetasnew_);
      logvaldiff+=(lnthetasnew_.sum()-lnthetas_.sum());
    }
    return logvaldiff;
  }

  void to_json(json &j)const{
    j["Machine"]["Name"]="RbmSpin";
    j["Machine"]["Nvisible"]=nv_;
    j["Machine"]["Nhidden"]=nh_;
    j["Machine"]["UseVisibleBias"]=usea_;
    j["Machine"]["UseHiddenBias"]=useb_;
    j["Machine"]["a"]=a_;
    j["Machine"]["b"]=b_;
    j["Machine"]["W"]=W_;
  }

  int read_layer_param_from_json(const json & pars, const string & param_name){
    if(FieldExists(pars["Layer"],param_name)){
      return FieldVal(pars["Layer"],param_name);
    }
    else{
      if(mynode_==0){
        cerr<<"# Error while constructing ConvACLayer from Json input: missing attribute \"" << param_name << "\""<<endl;
      }
      std::abort();
    }    
  }

  void assert_json_layer_name(const json & pars){
    if(pars.at("Layer").at("Name")!="ConvACLayer"){
      if(mynode_==0){
        cerr<<"# Error while constructing ConvACLayer from Json input"<<endl;
      }
      std::abort();
    }
  }

  void from_json(const json & pars){
    assert_json_layer_name(pars);   
    number_of_output_channels_ = read_layer_param_from_json(pars, "number_of_output_channels");
    kernel_width_ = read_layer_param_from_json(pars, "kernel_width");
    kernel_height_ = read_layer_param_from_json(pars, "kernel_height");
    strides_width_ = read_layer_param_from_json(pars, "strides_width");
    strides_height_ = read_layer_param_from_json(pars, "strides_height");
    Init();
    if (FieldExists(pars["Layer"],"offsets_weights_")){
      offsets_weights_ = pars["Layer"]["offsets_weights_"]
    }
  }
};


}

#endif
