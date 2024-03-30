#pragma once

#include <memory>
#include "WaveFunctions.h"
#include "../Particle.h"

class SimpleGaussian : public WaveFunction
{
public:
    SimpleGaussian(const double alpha, double beta);
    double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    double LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles);
    arma::vec QuantumForce(std::vector<std::unique_ptr<class Particle>> &particles, const int index);
    double w(std::vector<std::unique_ptr<class Particle>> &particles, const int index, const arma::vec step);
    arma::vec getParameters() {return m_parameters; };
    arma::vec dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles);
    void ChangeParameters(const double alpha, const double beta);
    arma::vec QuantumForce(
        std::vector<std::unique_ptr<class Particle>> &particles,
        const int index,
        const arma::vec Step
    );
    double geta() {return 0; };

private:
    double m_alpha;
    double m_beta;
    arma::vec m_beta_z;
    arma::vec m_parameters;
};


class SimpleGaussianNumerical : public SimpleGaussian
{
public:
    SimpleGaussianNumerical(double alpha, double beta, double dx);
    double DoubleDerivative(std::vector<std::unique_ptr<class Particle>> &particles);
    double EvaluateSingleParticle(class Particle particle);
    double EvaluateSingleParticle(class Particle particle, double step, double step_index);

private:
    double m_dx;
    double m_alpha;
    arma::vec m_beta_z;
};