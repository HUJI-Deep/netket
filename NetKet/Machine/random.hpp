#ifndef NETKET_RANDOM_HH
#define NETKET_RANDOM_HH

#include <complex>
#include <random>


template <typename T> class Random{
public:
    static void RandomGaussian(Eigen::Matrix<double, Eigen::Dynamic, 1> &par,
                               std::default_random_engine &generator, double sigma) {

        std::normal_distribution<double> distribution(0, sigma);
        for (int i = 0; i < par.size(); i++) {
            par(i) = distribution(generator);
        }
    }

    static void
    RandomGaussian(Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> &par,
                   std::default_random_engine &generator, double sigma) {
        std::normal_distribution<double> distribution(0, sigma);
        for (int i = 0; i < par.size(); i++) {
            par(i) = std::complex<double>(distribution(generator),
                                          distribution(generator));
        }
    }

    inline static void RandomGaussian(Eigen::Matrix<T, Eigen::Dynamic, 1> &par,
                               int seed, double sigma) {
        std::default_random_engine generator(seed);
        Random<T>::RandomGaussian(par, generator, sigma);
    }

};

#endif //NETKET_RANDOM_HH
