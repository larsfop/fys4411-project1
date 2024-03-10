#pragma once

#include <memory>
#include "WaveFunctions.h"
#include "../Particle.h"

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(
        std::vector<std::unique_ptr<class Particle>> &particles,
        const double alpha,
        double beta
        );
    double Wavefunction();
    double LocalEnergy();
    arma::vec QuantumForce(const int index);
    double w(const int index, const arma::vec step);

private:
    double m_alpha;
    double m_beta = 1.0;
    int m_numberofparticles;
    int m_numberofdimensions;
    std::vector<std::unique_ptr<class Particle>> m_particles;
    arma::vec m_beta_z;
};