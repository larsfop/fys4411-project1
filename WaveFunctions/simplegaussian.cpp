
#include <memory>
#include "simplegaussian.h"

#include <iostream>
using namespace std;

SimpleGaussian::SimpleGaussian(
    const double alpha,
    double beta
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_beta_z = {1, 1, beta};
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

double SimpleGaussian::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double E = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            E += m_alpha*m_beta_z(j) + 0.5*pos(j)*pos(j)*(1 - 4.0*m_alpha*m_alpha*m_beta_z(j)*m_beta_z(j));
        }
    }
    return E/numberofdimensions;
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce = pos;
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) *= -4*m_alpha*m_beta_z(i);
    }
    return qforce;
}

arma::vec SimpleGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce = pos + Step;
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) *= -4*m_alpha*m_beta_z(i);
    }
    return qforce; 
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
    return derivative; // ex. Psi[alpha]/Psi
}

void SimpleGaussian::ChangeParameters(const double alpha, const double beta)
{
    m_alpha = alpha,
    m_beta = beta;
    m_parameters = {alpha, beta};
}