
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
    int numberofdimensions = particles[0]->getNumberofDimensions();
    int numberofparticles = particles.size();
    int index = m_rng->NextInt(numberofparticles-1);
    double D = 0.5; // diffusion coefficient

    arma::vec qforce = wavefunction.QuantumForce(particles, index);
    arma::vec params = wavefunction.getParameters(); // params = {alpha, beta}
    arma::vec beta_z = {1, 1, params(1)}; // there to make the program more general for different betas

    // calculate the new step for the particle
    arma::vec pos = particles[index]->getPosition();
    arma::vec step(numberofdimensions);
    for (int i = 0; i < numberofdimensions; i++)
    {
        step(i) = sqrt_dt*m_rng->NextGaussian() + qforce(i)*stepsize*D;
    }

    // then compute the quantum force with the new step
    arma::vec qforcenew = wavefunction.QuantumForce(particles, index, step);
    double greens = 0;
    for (int i = 0; i < numberofdimensions; i++)
    {
        greens += 0.5*(qforce(i) + qforcenew(i))*(D*stepsize*0.5*\
        (qforce(i) - qforcenew(i)) - (pos(i) + step(i)) + pos(i));
    }

    // check whether the new positions is accepted or not
    // the energy will be calculated anyway
    double w = wavefunction.w(particles, index, step) * exp(greens);
    double random = m_rng->NextDouble();
    if(random <= w)
    {
        particles.at(index)->ChangePosition(step);

        return true;
    }

    return false;
}