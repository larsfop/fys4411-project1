
#include "system.h"
#include "sampler.h"
#include "Particle.h"
#include "WaveFunctions/WaveFunctions.h"
#include "InitialState/initialstate.h"
#include "Solvers/montecarlo.h"

System::System(
    std::unique_ptr<class WaveFunction> wavefunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles
)
{
    m_numberofparticles = particles.size();
    m_numberofdimensions = particles[0]->getNumberofDimensions();
    m_wavefunction = std::move(wavefunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
}

std::unique_ptr<class Sampler> System::RunMetropolisSteps(
    double steplength,
    int numberofMetropolisSteps
)
{
    auto sampler = std::make_unique<Sampler>(
        m_numberofparticles,
        m_numberofdimensions,
        steplength,
        numberofMetropolisSteps
    );
    for (int i = 0; i < numberofMetropolisSteps; i++)
    {
        bool acceptedStep = m_solver->Step(steplength, *m_wavefunction, m_particles);
        sampler->Sample(acceptedStep, this);
    }
    sampler->ComputeAverages();
    return sampler;
}

double System::ComputeLocalEnergy()
{
    return m_wavefunction->LocalEnergy(m_particles);
}

arma::vec System::getParameters()
{
    return m_wavefunction->getParameters();
}