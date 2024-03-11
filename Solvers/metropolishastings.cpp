
#include "metropolishastings.h"
#include "../Particle.h"

MetropolisHastings::MetropolisHastings(std::unique_ptr<class Random> rng) : MonteCarlo(std::move(rng)) {}

bool MetropolisHastings::Step(
    double stepsize,
    class WaveFunction &wavefunction,
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    double sqrt_dt = sqrt(stepsize);
    int numberofdimension = particles.size();
    int index = m_rng->NextInt(numberofdimension-1);
    int D = 0.5;

    arma::vec qforce = wavefunction.QuantumForce(particles, index);
    arma::vec params = wavefunction.getParameters();
    arma::vec beta_z = {1, 1, params(1)};

    arma::vec pos = particles[index]->getPosition();
    //arma::vec step(numberofdimension, arma::fill::zeros);
    std::vector<double> step = std::vector<double>();
    arma::vec qforcenew = qforce;
    double greens = 0;
    for (int i = 0; i < numberofdimension; i++)
    {
        //step(i) = stepsize * m_rng->NextGaussian();
        step.push_back(sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D);
        qforcenew(i) += step[i];
        greens += 0.5*(qforce(i) + qforcenew(i))*(D*stepsize*0.5*\
        (qforce(i) - qforcenew(i)) - pos(i)*pos(i)*beta_z(i));
        // greens += 0.5*qforce(i)*(1 + step[i])*D*stepsize*0.5*\
        // (qforce(i)*(1 - step[i]) - pos(i)*pos(i)*beta_z(i));
    }

    double w = wavefunction.w(particles, index, step) * exp(greens);

    if(m_rng->NextDouble() <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}