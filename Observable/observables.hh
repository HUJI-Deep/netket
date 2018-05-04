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

#ifndef NETKET_OBSERVABLES_HH
#define NETKET_OBSERVABLES_HH

#include <vector>
#include <string>
#include "netket.hh"

namespace netket{

class Observables{

  std::vector<Observable> observables_;

public:

  using MatType=LocalOperator::MatType;

  Observables(const Hilbert & hilbert,const json & pars){

    if(FieldExists(pars,"Observables")){
      auto obspar=pars["Observables"];

      if(obspar.is_array()){
        //multiple observables case
        for(int i=0;i<obspar.size();i++){
          observables_.push_back(Observable(hilbert,obspar[i]));
        }
      }
      else{
        //single observable case
        observables_.push_back(Observable(hilbert,obspar));
      }
    }

  }

  Observable & operator()(int i){
    assert(i<observables_.size());
    return observables_[i];
  }

  const int Size()const{
    return observables_.size();
  }

};
}
#endif
