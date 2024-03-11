#pragma once

#include <memory>
#include "WaveFunctions.h"
#include "../Particle.h"

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(
        const double alpha,
        double beta
        );
    double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles);
    double LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles);
    arma::vec QuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, const int index);
    double w(std::vector<std::unique_ptr<class Particle>> &particles, const int index, const arma::vec step);
    arma::vec getParameters() {return m_parameters; };
    arma::vec dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles);

private:
    double m_alpha;
    double m_beta = 1.0;
    arma::vec m_beta_z;
    arma::vec m_parameters;
};