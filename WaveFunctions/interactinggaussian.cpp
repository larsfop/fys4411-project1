
#include "interactinggaussian.h"

#include <iostream>
using namespace std;

InteractingGaussian::InteractingGaussian(
    const double alpha, 
    const double beta,
    const double a,
    const double gamma
)
{
    m_alpha = alpha;
    m_beta = beta;
    m_a = a;
    m_parameters = {alpha, beta};
    m_beta_z = {1.0, 1.0, beta};
    m_gamma = gamma;
    m_gamma_z = {1, 1, gamma};
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
            f *= (1 - m_a/rij) * (rij > m_a);
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
    double d2phi = 0;
    double alpha2 = m_alpha*m_alpha;
    arma::vec v(numberofdimensions);
    double term1 = 0;
    double term2 = 0;
    double term3 = 0;

    double constant = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        constant += m_gamma_z(i);
    }
    constant *= 2*m_alpha*numberofparticles;

    for (int i  = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        arma::vec r2 = arma::square(pos%m_beta_z);

        // Calculate the double and single derivative of phi, divided by phi
        arma::vec posj(numberofdimensions);
        arma::vec dphi(numberofdimensions);
        for (int j = 0; j < numberofdimensions; j++)
        {
            d2phi += 4*alpha2*r2(j);
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
        // term1 += 2*arma::dot(dphi,v);
        // term2 += arma::dot(v,v);
        for (int j = 0; j < 3; j++)
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
    double kinetic = DoubleDerivative(particles);
    arma::vec gamma_z = {1, 1, m_gamma};
    //cout << m_gamma << endl;
    //gamma_z.print();
    double potential = 0;
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos = particles[i]->getPosition();
        potential += arma::sum(arma::square(pos%gamma_z));
    }
    return 0.5*(-kinetic + potential)/numberofdimensions;
}

arma::vec InteractingGaussian::QuantumForce(
    std::vector<std::unique_ptr<class Particle>> &particles,
    const int index
)
{
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    arma::vec pos = particles[index]->getPosition();
    arma::vec qforce = pos % m_beta_z;
    double up = 0;
    for (int j = 0; j < index; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos - posj);

        up += m_a/(rkj*(rkj - m_a));
    }
    for (int j = index+1; j < numberofparticles; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos - posj);

        up += m_a/(rkj*(rkj - m_a));
    }
    return -4*m_alpha*qforce + 2*up;
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
    arma::vec qforce = pos%m_beta_z + Step;
    double up = 0;
    for (int j = 0; j < index; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos + Step - posj);

        up += m_a/(rkj*(rkj - m_a));
    }
    for (int j = index+1; j < numberofparticles; j++)
    {
        arma::vec posj = particles[j]->getPosition();
        double rkj = arma::norm(pos + Step - posj);

        up += m_a/(rkj*(rkj - m_a));
    }
    return -4*m_alpha*qforce + 2*up;
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
        //dr2 += (pos(i) + step(i))*(pos(i) + step(i)) - pos(i)*pos(i)*m_beta_z(i);
        dr2 += (2*pos(i) + step(i))*step(i)*m_beta_z(i);
    }
    for (int i = 0; i < index; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double rki_n = arma::norm(pos + step - posi);
        double rki_o = arma::norm(pos - posi);

        interaction *= (1 - m_a/rki_n)/(1 - m_a/rki_o);
    }
    for (int i = index+1; i < numberofparticles; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double rki_n = arma::norm(pos + step - posi);
        double rki_o = arma::norm(pos - posi);

        interaction *= (1 - m_a/rki_n)/(1 - m_a/rki_o);
    }
    return std::exp(-2*m_alpha*dr2)*interaction;
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