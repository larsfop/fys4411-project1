#pragma once

#include <vector>
#include <armadillo>

class WaveFunction
{
private:
    
public:
    virtual ~WaveFunction() = default;

    virtual double Wavefunction() = 0;
    virtual double LocalEnergy() = 0;
    virtual arma::vec QuantumForce(const int index) = 0;
    virtual double w(const int index, const arma::vec step) = 0;
};

