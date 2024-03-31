
#include "interactinggaussian.h"

#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;

InteractingGaussian::InteractingGaussian(
    const double alpha, 
    const double beta
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_a = 0.0043;
    m_parameters = {alpha, beta};
    m_beta_z = {1.0, 1.0, beta};
    m_gamma_z = {1, 1, beta};
}


double InteractingGaussian::Wavefunction(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    double r2 = 0;
    double f = 1;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = i+1; j < numberofparticles; j++)
        {
            arma::vec posj = particles[j]->getPosition();
            double rij = arma::norm(pos - posj);
            f *= std::max(1 - m_a/rij, 0.0);
        }
        for (int j = 0; j < numberofdimensions; j++)
        {
            r2 += pos(j)*pos(j)*m_beta_z(j);
        }
    }
    return std::exp(-m_alpha*r2)*f;
}

double InteractingGaussian::DoubleDerivative(
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    arma::vec beta_z2 = arma::square(m_beta_z);
    double d2phi = 0;
    double alpha2 = m_alpha*m_alpha;
    double term1 = 0;
    double term2 = 0;
    double term3 = 0;

    double constant = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        constant += m_beta_z(i);
    }
    constant *= 2*m_alpha*numberofparticles;

    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();

        // Calculate the double and single derivative of phi, divided by phi
        arma::vec posj(numberofdimensions);
        arma::vec dphi(numberofdimensions);
        arma::vec v(numberofdimensions);
        for (int j = 0; j < numberofdimensions; j++)
        {
            d2phi += 4*alpha2*pos(j)*pos(j)*beta_z2(j);
            dphi(j) = -2*m_alpha*pos(j);
        }

        // Calculate the three interaction terms
        for (int j = 0; j < i; j++)
        {
            posj = particles[j]->getPosition();
            double rkj = arma::norm(pos - posj);

            double up = m_a/(rkj*(rkj - m_a));
            double upp = m_a*(m_a - 2*rkj)/(rkj*rkj*(rkj - m_a)*(rkj - m_a));

            v += (pos - posj)/rkj * up;

            term3 += upp + 2/rkj*up;
        }
        for (int j = i+1; j < numberofparticles; j++)
        {
            posj = particles[j]->getPosition();
            double rkj = arma::norm(pos - posj);

            double up = m_a/(rkj*(rkj - m_a));
            double upp = m_a*(m_a - 2*rkj)/(rkj*rkj*(rkj - m_a)*(rkj - m_a));

            v += (pos - posj)/rkj * up;

            term3 += upp + 2/rkj*up;
        }
        for (int j = 0; j < numberofdimensions; j++)
        {
            term1 += dphi(j)*v(j);
            term2 += v(j)*v(j);
        }
    }
    return d2phi - constant + 2*term1 + term2 + term3;
}

double InteractingGaussian::LocalEnergy(std::vector<std::unique_ptr<class Particle>> &particles)
{
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    arma::vec gamma_z2 = arma::square(m_gamma_z);
    double kinetic = DoubleDerivative(particles);
    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        for (int j = 0; j < numberofdimensions; j++)
        {
            potential += pos(j)*pos(j)*gamma_z2(j);
        }
    }
    return 0.5*(-kinetic + potential);
}

arma::vec InteractingGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) = pos(i)*m_beta_z(i);
    }
    arma::vec up(numberofdimensions);
    for (int j = 0; j < index; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos - posj);

        //up -= (pos - posj)/(rkj*(rkj - m_a)*(rkj - m_a));
        up -= (pos - posj)/(rkj*rkj*(rkj - m_a));
    }
    for (int j = index+1; j < numberofparticles; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos - posj);

        //up -= (pos - posj)/(rkj*(rkj - m_a)*(rkj - m_a));
        up -= (pos - posj)/(rkj*rkj*(rkj - m_a));
    }
    return -4*m_alpha*qforce + 2*m_a*up;
}

arma::vec InteractingGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index,
    const arma::vec Step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        qforce(i) = pos(i)*m_beta_z(i) + Step(i);
    }
    arma::vec up(numberofdimensions);
    for (int j = 0; j < index; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos + Step - posj);

        up -= (pos - posj)/(rkj*(rkj - m_a)*(rkj - m_a));
    }
    for (int j = index+1; j < numberofparticles; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos + Step - posj);

        up -= (pos - posj)/(rkj*(rkj - m_a)*(rkj - m_a));
    }
    return -4*m_alpha*qforce + 2*m_a*up;
}

double InteractingGaussian::w(std::vector<std::unique_ptr<class Particle>> &particles,
    const int index, 
    const arma::vec step
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    arma::vec pos = particles[index]->getPosition();
    double dr2 = 0;
    double interaction = 1;
    for (int i = 0; i < numberofdimensions; i++)
    {
        dr2 += (2*pos(i) + step(i))*step(i)*m_beta_z(i);
    }
    for (int i = 0; i < index; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double rki_n = arma::norm(pos + step - posi);
        double rki_o = arma::norm(pos - posi);

        interaction *= std::max(1 - m_a/rki_n, 0.0)/std::max(1 - m_a/rki_o, 0.0);
    }
    for (int i = index+1; i < numberofparticles; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double rki_n = arma::norm(pos + step - posi);
        double rki_o = arma::norm(pos - posi);

        //if (rki_o<0)
        //{
        //    interaction *= 1e6;
        //}
        //else
        //{
            double numerator = std::max(1 - m_a/rki_n, 0.0);
            double denominator = std::max(1 - m_a/rki_o, 0.0);
            interaction = numerator/denominator;
            //interaction *= std::max(1 - m_a/rki_n, 0.0)/std::max(1 - m_a/rki_o, 0.0);
        //}
    }
    return std::exp(-2*m_alpha*dr2)*interaction*interaction;
}

// Take the derivative of the the wavefunction as a function of the parameters alpha, beta
arma::vec InteractingGaussian::dPsidParam(std::vector<std::unique_ptr<class Particle>> &particles)
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

void InteractingGaussian::ChangeParameters(const double alpha, const double beta)
{
    m_alpha = alpha,
    m_beta = beta;
    m_parameters = {alpha, beta};
}