
#include "metropolishastings.h"
#include "../Particle.h"

#include <iostream>
using namespace std;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng) : MonteCarlo(std::move(rng)) {}

bool MetropolisHastings::Step(
    double stepsize,
    class WaveFunction &wavefunction,
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    double sqrt_dt = sqrt(stepsize);
    int numberofparticles = particles.size();
    int numberofdimension = particles[0]->getNumberofDimensions();
    int index = m_rng->NextInt(numberofparticles-1);
    int D = 0.5;

    arma::vec qforce = wavefunction.QuantumForce(particles, index);
    arma::vec params = wavefunction.getParameters(); // params = {alpha, beta}
    arma::vec beta_z = {1, 1, params(1)};

    arma::vec pos = particles[index]->getPosition();
    arma::vec step(numberofdimension);
    //std::vector<double> step = std::vector<double>();
    //arma::vec qforcenew = qforce;
    for (int i = 0; i < numberofdimension; i++)
    {
        step(i) = sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D;
        //step.push_back(sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D);
        // qforcenew(i) += step(i);
    }
    arma::vec qforcenew = wavefunction.QuantumForce(particles, index, step);
    double greens = 0;
    for (int i = 0; i < numberofdimension; i++)
    {
        greens += 0.5*(qforce(i) + qforcenew(i))*(D*stepsize*0.5*\
        (qforce(i) - qforcenew(i)) - (pos(i) + step(i)) + pos(i));
        // greens += 0.5*qforce(i)*(1 + step[i])*D*stepsize*0.5*\
        // (qforce(i)*(1 - step[i]) - (1 + step(i))*pos(i)*beta_z(i));
    }
    double G = std::exp(greens);
    double w_Psi = wavefunction.w(particles, index, step);
    cout << typeid(G).name() << "\t\t" << typeid(w_Psi).name() << endl;
    cout << w_Psi*G << endl;
    double w = wavefunction.w(particles, index, step) * exp(greens);
    cout << "w : " << w << endl;
    cout << wavefunction.w(particles, index, step) << endl;
    cout << exp(greens) << endl;
    cout << wavefunction.w(particles, index, step)*exp(greens) << endl;

    if(m_rng->NextDouble() <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}