
#include <memory>
#include "simplegaussian.h"

SimpleGaussian::SimpleGaussian(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const double alpha,
    double beta
)
{
    m_particles = particles;
    m_numberofdimensions = particles[0]->getNumberofDimensions();
    m_numberofparticles = particles.size();
    m_alpha = alpha;
    m_beta = beta;
    m_beta_z = {1, 1, beta};
}

double SimpleGaussian::Wavefunction()
{
    double r2 = 0;
    for (int i = 0; i < m_numberofparticles; i++)
    {
        arma::vec pos = m_particles[i]->getPosition();
        for (int j = 0; j < m_numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j)*m_beta_z(j)*m_beta_z(j);
        }
    }
    return std::exp(-m_alpha*r2);
}

double SimpleGaussian::LocalEnergy()
{
    double E = 0;
    for (int i = 0; i < m_numberofparticles; i++)
    {
        arma::vec pos = m_particles[i]->getPosition();
        for (int j = 0; j < m_numberofdimensions; j++)
        {
            E += m_alpha*m_beta_z(j) + 0.5*pos(j)*pos(j)*(1 - 4*m_alpha*m_alpha*m_beta_z(j)*m_beta_z(j));
        }
    }
    return E/m_numberofdimensions;
}

arma::vec SimpleGaussian::QuantumForce(const int index)
{
    arma::vec pos = m_particles[index]->getPosition();
    arma::vec qforce = pos;
    for (int i = 0; i < m_numberofdimensions; i++)
    {
        qforce(i) *= -4*m_alpha*m_beta_z(i);
    }
    return qforce;
}

double SimpleGaussian::w(const int index, const arma::vec step)
{
    arma::vec pos = m_particles[index]->getPosition();
    double dr2 = 0;
    for (int i = 0; i < m_numberofdimensions; i++)
    {
        dr2 += (pos(i) + step(i))*(pos(i) + step(i)) - pos(i)*pos(i);
    }
    return exp(-2*m_alpha*dr2);
}