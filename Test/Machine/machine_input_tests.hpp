
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetMachineInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

//   Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  input_tests.push_back(pars);

  pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "ConvAC"}, {"Alpha", 1.0}, {"visible_height", 10}, {"Layers", {{
                  {"Name", "ConvACLayer"},
                  {"kernel_width", 1},
                  {"kernel_height", 3},
                  {"padding_width", 0},
                  {"padding_height", 1},
                  {"strides_width", 1},
                  {"strides_height", 1},{"init_in_log_space", true}, {"normalize_input_channels", false},
                  {"number_of_output_channels", 2}
          }}}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  input_tests.push_back(pars);

  pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 3}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "ConvAC"}, {"Alpha", 1.0},{"Layers", {{{"Name", "ToOneHotLayer"}},
                 {{"Name", "ConvACLayer"},
                 {"kernel_width", 1},
                 {"kernel_height", 3},
                 {"padding_width", 0},
                 {"padding_height", 1},
                 {"strides_width", 1},
                 {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                 {"number_of_output_channels", 1}},
                 {{"Name", "ConvACLayer"},
                   {"kernel_width", 1},
                   {"kernel_height", 3},
                   {"padding_width", 0},
                   {"padding_height", 1},
                   {"strides_width", 1},
                   {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                   {"number_of_output_channels", 1}}
         }}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  input_tests.push_back(pars);

    pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 3}, {"Dimension", 1}, {"Pbc", true}}},
            {"Machine", {{"Name", "ConvAC"}, {"visible_height", 1}, {"visible_width", 3}, {"Alpha", 1.0},{"Layers", {{{"Name", "ToOneHotLayer"}},
                   {{"Name", "ConvACLayer"},
                            {"kernel_width", 3},
                            {"kernel_height", 1},
                             {"padding_width", 1},
                            {"padding_height", 0},
                            {"strides_width", 1},
                            {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                            {"number_of_output_channels", 1}},
                    {{"Name", "ConvACLayer"},
                            {"kernel_width", 3},
                            {"kernel_height", 1},
                            {"padding_width", 1},
                            {"padding_height", 0},
                            {"strides_width", 1},
                            {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                            {"number_of_output_channels", 1}}, {{"Name", "SumPoolingLayer"}}
            }}}},
            {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
    input_tests.push_back(pars);
//
  pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 3}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "ConvAC"}, {"Alpha", 1.0}, {"fast_lookup", true},{"Layers", {{{"Name", "ToOneHotLayer"}},  {{"Name", "AddBiasLayer"}},{
                   {"Name", "ConvACLayer"},
                   {"kernel_width", 1},
                   {"kernel_height", 3},
                   {"padding_width", 0},
                   {"padding_height", 0},
                   {"strides_width", 1},
                   {"strides_height", 3},{"init_in_log_space", true}, {"normalize_input_channels", false},
                   {"number_of_output_channels", 2}},
//                                                                                                    {{"Name", "AddBiasLayer"}},
                   {{"Name", "ConvACLayer"},
                    {"kernel_width", 1},
                    {"kernel_height", 1},
                    {"padding_width", 0},
                    {"padding_height", 0},
                    {"strides_width", 1},
                    {"strides_height", 1},{"init_in_log_space", true}, {"normalize_input_channels", false},
                    {"number_of_output_channels", 1}},
                                                                                            }}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  input_tests.push_back(pars);
    pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 2}, {"Pbc", true}}},
            {"Machine", {{"Name", "ConvAC"}, {"Alpha", 1.0}, {"visible_width", 10}, {"visible_height", 10}, {"fast_lookup", true},
                                {"Layers", {{{"Name", "ToOneHotLayer"}},
               {{"Name", "ConvACLayer"},
                       {"kernel_width", 3},
                       {"kernel_height", 3},
                       {"padding_width", 1},
                       {"padding_height", 1},
                       {"strides_width", 1},
                       {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                       {"number_of_output_channels", 1}},
                {{"Name", "ConvACLayer"},
                 {"kernel_width", 10},
                 {"kernel_height", 10},
                 {"padding_width", 0},
                 {"padding_height", 0},
                 {"strides_width", 10},
                 {"strides_height", 10},{"init_in_log_space", false}, {"normalize_input_channels", false},
                 {"number_of_output_channels", 1}},
               {{"Name", "ConvACLayer"},
                       {"kernel_width", 1},
                       {"kernel_height", 1},
                       {"padding_width", 0},
                       {"padding_height", 0},
                       {"strides_width", 1},
                       {"strides_height", 1},{"init_in_log_space", false}, {"normalize_input_channels", false},
                       {"number_of_output_channels", 1}},
//            {{"Name", "ConvACLayer"},
//              {"kernel_width", 3},
//              {"kernel_height", 3},
//              {"padding_width", 1},
//              {"padding_height", 1},
//              {"strides_width", 1},
//              {"strides_height", 1},{"init_in_log_space", true}, {"normalize_input_channels", false},
//              {"number_of_output_channels", 1}},
                                                                                                                                           }}}},
            {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
    input_tests.push_back(pars);
  pars = {{"Graph", {{"Name", "Hypercube"}, {"L", 3}, {"Dimension", 2}, {"Pbc", true}}},
          {"Machine", {{"Name", "ConvAC"}, {"Alpha", 1.0}, {"SigmaRand", 1.0}, {"visible_width", 3}, {"visible_height", 3}, {"fast_lookup", true},{"Layers", {{{"Name", "ToOneHotLayer"}},
                                                                                                    {{"Name", "ConvACLayer"},
                                                {"kernel_width", 3},
                                                {"kernel_height", 3},
                                                {"padding_width", 1},
                                                {"padding_height", 1},
                                                {"strides_width", 1},
                                                {"strides_height", 1},{"dirichlet_init", true}, {"normalize_input_channels", false},
                                                {"number_of_output_channels", 1}},

            {{"Name", "ConvACLayer"},
              {"kernel_width", 3},
              {"kernel_height", 3},
              {"padding_width", 1},
              {"padding_height", 1},
              {"strides_width", 1},
              {"strides_height", 1},{"dirichlet_init", true}, {"normalize_input_channels", false},
              {"number_of_output_channels", 1}},
                                                                                            }}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with symmetric machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 4}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with non-symmetric rbm machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 4}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with multi-val rbm
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmMultival"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 3}}}};
  input_tests.push_back(pars);

  return input_tests;
}
