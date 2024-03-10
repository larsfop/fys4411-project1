#pragma once

#include <vector>
#include <armadillo>

class WaveFunction
{
private:
    
public:
    virtual ~WaveFunction() = default;

    virtual double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual double LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles) = 0;
    virtual arma::vec QuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, const int index) = 0;
    virtual double w(std::vector<std::unique_ptr<class Particle>> &particles, const int index, const arma::vec step) = 0;
    virtual arma::vec getParameters() = 0;
};

