
#include <memory>
#include "simplegaussian.h"

#include <iostream>
#include <iomanip>
using namespace std;

SimpleGaussian::SimpleGaussian(
    const double alpha,
    double beta
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_beta_z = {1.0, 1.0, beta};
    m_parameters = {alpha, beta};
}

double SimpleGaussian::Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double r2 = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j)*m_beta_z(j);
        }
    }
    return std::exp(-m_alpha*r2);
}

double SimpleGaussian::DoubleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double d2psi = 0;

    // Calculate the constant term, where the z dimension is multiplied by 
    // an additional beta   
    double term = 0;
    for (int j = 0; j < numberofdimensions; j++)
    {
        term += m_beta_z(j);
    }
    term *= 2*m_alpha;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();

        // sum and square the position term i.e. (x^2 + y^2 + beta*z^2)
        double r2 = arma::sum(arma::square(pos%m_beta_z));
        d2psi += 4*m_alpha*m_alpha*r2;
    }
    return d2psi - term;
}

double SimpleGaussian::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double kinetic = DoubleDerivative(particles);

    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        potential += arma::sum(arma::square(pos));
        // for (int j = 0; j < numberofdimensions; j++)
        // {
        //     potential += pos(j)*pos(j);
        // }
    }

    // E_L = alpha(2 + beta) + r^2(1 - 4alpha)
    return 0.5*(-kinetic + potential);
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce = pos%m_beta_z;
    return -4*m_alpha*qforce;
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce = pos%m_beta_z + Step;
    return -4*m_alpha*qforce; 
}

double SimpleGaussian::w(std::vector<std::unique_ptr<class Particle>> &particles,
    const int index, 
    const arma::vec step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    double dr2 = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        //dr2 += (pos(i) + step(i))*(pos(i) + step(i)) - pos(i)*pos(i)*m_beta_z(i);
        dr2 += (2*pos(i) + step(i))*step(i)*m_beta_z(i);
    }
    return std::exp(-2*m_alpha*dr2);
}

// Take the derivative of the the wavefunction as a function of the parameters alpha, beta
arma::vec SimpleGaussian::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();

    arma::vec derivative(2);
    arma::vec r2(3);
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2(j) += pos(j)*pos(j);
        }
    }
    derivative(0) = -(r2(0) + r2(1) + m_beta*r2(2));
    derivative(1) = -m_alpha*r2(2);
    return derivative; // ex. Psi[alpha]/Psi -> Psi[alpha] = dPsi/dalpha
}

void SimpleGaussian::ChangeParameters(const double alpha, const double beta)
{
    m_alpha = alpha,
    m_beta = beta;
    m_parameters = {alpha, beta};
}