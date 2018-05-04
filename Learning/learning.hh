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

#ifndef NETKET_LEARNING_HH
#define NETKET_LEARNING_HH

namespace netket{
  class AbstractStepper;
  class Sgd;
  class AdaDelta;
  class AdaMax;
  class Rprop;

  class GroundState;
  class MatrixReplacement;
}

#include "netket.hh"
#include "abstract_stepper.hh"

#include "sgd.hh"
#include "ada_delta.hh"
#include "ada_max.hh"
#include "rprop.hh"



namespace netket{
  class Stepper:public AbstractStepper{
    using Ptype=std::unique_ptr<AbstractStepper>;
    Ptype s_;
  public:
    Stepper(const json & pars);
    void Init(const VectorXd & pars);
    void Init(const VectorXcd & pars);
    void Update(const VectorXd & grad,VectorXd & pars);
    void Update(const VectorXcd & grad,VectorXd & pars);
    void Update(const VectorXcd & grad,VectorXcd & pars);
    void Reset();
  };

  class Learning {
  public:
    Learning(const json & pars);
  };

}

#include "matrix_replacement.hh"
#include "ground_state.hh"


#endif
