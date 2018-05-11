
#include "Json/json.hh"
#include <vector>
#include <string>
#include <fstream>

std::vector<json> GetMachineInputs(){

  std::vector<json> input_tests;
  json pars;

  //Ising 1d
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 1},
      {"Pbc", true}
    }},
    {"Machine",{
      {"Name", "RbmSpin"},
      {"Alpha", 1.0}
    }},
    {"Hamiltonian",{
      {"Name", "Ising"},
      {"h", 1.0}
    }}
  };
  input_tests.push_back(pars);

  //Heisenberg 1d
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 1},
      {"Pbc", true}
    }},
    {"Machine",{
      {"Name", "RbmSpinSymm"},
      {"Alpha", 1.0}
    }},
    {"Hamiltonian",{
      {"Name", "Heisenberg"},
      {"TotalSz", 0}
    }}
  };
  input_tests.push_back(pars);

  //Bose-Hubbard 1d with symmetric machine
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 1},
      {"Pbc", true}
    }},
    {"Machine",{
      {"Name", "RbmSpinSymm"},
      {"Alpha", 1.0}
    }},
    {"Hamiltonian",{
      {"Name", "BoseHubbard"},
      {"U", 4.0},
      {"Nmax", 4},
      {"Nbosons", 20}
    }}
  };
  input_tests.push_back(pars);


  //Bose-Hubbard 1d with non-symmetric rbm machine
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 1},
      {"Pbc", true}
    }},
    {"Machine",{
      {"Name", "RbmSpin"},
      {"Alpha", 2.0}
    }},
    {"Hamiltonian",{
      {"Name", "BoseHubbard"},
      {"U", 4.0},
      {"Nmax", 4},
      {"Nbosons", 20}
    }}
  };
  input_tests.push_back(pars);

  //Bose-Hubbard 1d with multi-val rbm
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 10},
      {"Dimension", 1},
      {"Pbc", true}
    }},
    {"Machine",{
      {"Name", "RbmMultival"},
      {"Alpha", 2.0}
    }},
    {"Hamiltonian",{
      {"Name", "BoseHubbard"},
      {"U", 4.0},
      {"Nmax", 3},
      {"Nbosons", 20}
    }}
  };
  input_tests.push_back(pars);


  return input_tests;
}