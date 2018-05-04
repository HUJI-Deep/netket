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

#ifndef NETKET_GRAPH_HH
#define NETKET_GRAPH_HH


namespace netket{

  class AbstractGraph;
  class Hypercube;
  class CustomGraph;
}

#include "abstract_graph.hh"
#include "distance.hh"
#include "hypercube.hh"
#include "custom_graph.hh"

namespace netket{
  class Graph: public AbstractGraph{
    using Ptype=std::unique_ptr<AbstractGraph>;
    Ptype g_;
  public:
    Graph(const json & pars);
    int Nsites()const;
    std::vector<std::vector<int>> AdjacencyList()const;
    std::vector<std::vector<int>> SymmetryTable()const;
    std::vector<std::vector<int>> Distances()const;
    bool IsBipartite()const;
  };
}


#endif
