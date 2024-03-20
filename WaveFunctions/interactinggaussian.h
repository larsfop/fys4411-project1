
#include <memory>
#include "WaveFunctions.h"
#include "../Particle.h"

class InteractingGaussian : public WaveFunction
{
public:
    InteractingGaussian(
        const double alpha, 
        const double beta
    );
    double Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles);
    double DoubleDerivative(
        std::vector<std::unique_ptr<class Particle>> &particles
    );
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
    double geta() {return m_a; };

private:
    double m_alpha;
    double m_beta;
    double m_a;
    double m_gamma;
    arma::vec m_beta_z;
    arma::vec m_gamma_z;
    arma::vec m_parameters;
};