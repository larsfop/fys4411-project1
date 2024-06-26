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
    int numberofdimension = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int index = m_rng->NextInt(numberofparticles-1);

    // calculate next step
    arma::vec step(numberofdimension, arma::fill::zeros);
    for (int i = 0; i < numberofdimension; i++)
    {
        step(i) = stepsize * (m_rng->NextDouble() - 0.5);
    }
    double w = wavefunction.w(particles, index, step);

    // check if the new position is accepted
    if(m_rng->NextDouble() <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}