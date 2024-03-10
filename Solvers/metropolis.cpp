#include <memory>
#include "metropolis.h"
#include "../Particle.h"

Metropolis::Metropolis(std::unique_ptr<class Random> rng) : MonteCarlo(std::move(rng)) {}

bool Metropolis::Step(
    double stepsize,
    class WaveFunction &wavefunction,
    std::vector<std::unique_ptr<class Particle>> &particles
)
{
    int numberofdimension = particles.size();
    int index = m_rng->NextInt(numberofdimension-1);

    arma::vec step(numberofdimension, arma::fill::zeros);
    for (int i = 0; i < numberofdimension; i++)
    {
        step(i) = stepsize * m_rng->NextGaussian();
    }
    double w = wavefunction.w(index, step);

    if(m_rng->NextDouble() <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}