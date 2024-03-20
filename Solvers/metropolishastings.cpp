
#include "metropolishastings.h"
#include "../Particle.h"

#include <iostream>
#include <iomanip>
using namespace std;

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng) : MonteCarlo(std::move(rng)) {}

bool MetropolisHastings::Step(
    double stepsize,
    class WaveFunction &wavefunction,
    std::vector<std::unique_ptr<class Particle>> &particles,
    int index
)
{
    double sqrt_dt = sqrt(stepsize);
    int numberofparticles = particles.size();
    int numberofdimensions = particles[0]->getNumberofDimensions();
    index = m_rng->NextInt(numberofparticles-1);
    //int index = 0;
    double D = 0.5;

    arma::vec qforce = wavefunction.QuantumForce(particles, index);
    arma::vec params = wavefunction.getParameters(); // params = {alpha, beta}
    arma::vec beta_z = {1, 1, params(1)};

    arma::vec pos = particles[index]->getPosition();
    arma::vec step(numberofdimensions);
    //std::vector<double> step = std::vector<double>();
    //arma::vec qforcenew = qforce;
    for (int i = 0; i < numberofdimensions; i++)
    {
        step(i) = sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D;
        //step.push_back(sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D);
        // qforcenew(i) += step(i);
    }

    double a = wavefunction.geta();
    for (int i = 0; i < index; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double dr = arma::norm((pos+step) - posi);
        
        if (dr <= a)
        {
            return false;
        }
    }
    for (int i = index+1; i < numberofparticles; i++)
    {
        arma::vec posi = particles[i]->getPosition();
        double dr = arma::norm((pos+step) - posi);
        
        if (dr <= a)
        {
            return false;
        }
    }

    arma::vec qforcenew = wavefunction.QuantumForce(particles, index, step);
    double greens = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        greens += 0.5*(qforce(i) + qforcenew(i))*(D*stepsize*0.5*\
        (qforce(i) - qforcenew(i)) - (pos(i) + step(i)) + pos(i));
    }

    double w = wavefunction.w(particles, index, step) * exp(greens);
    double random = m_rng->NextDouble();
    // cout << "new MC cycle" << endl;
    // cout << wavefunction.w(particles, index, step) << "  " << exp(greens) << endl;
    // cout << pos(0) << " " << pos(1) << " " << pos(2) << endl;
    // cout << qforce(0) << " " << qforce(1) << " " << qforce(2) << endl;
    // cout << std::fixed << setprecision(10);
    // cout << random << "  " << w << endl;
    if(random <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}