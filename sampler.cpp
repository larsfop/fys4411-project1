
#include <memory>
#include "sampler.h"
#include "system.h"

Sampler::Sampler(
    int numberofparticles,
    int numberofdimensions,
    double steplength,
    int numberofMetropolisSteps
)
{
    m_stepnumber = 0;
    m_numberofMetropolisSteps = numberofMetropolisSteps;
    m_numberofparticles = numberofparticles;
    m_numberofdimensions = numberofdimensions;
    m_energy = 0;
    m_DeltaEnergy = 0;
    m_steplength = steplength;
    m_numberofacceptedsteps = 0;
}

void Sampler::Sample(bool acceptedstep, class System *system)
{
    auto localenergy = system->ComputeLocalEnergy();
    m_DeltaEnergy += localenergy;
    m_stepnumber++;
    m_numberofacceptedsteps += acceptedstep;
}

void Sampler::ComputeAverages()
{
    m_energy = m_DeltaEnergy/m_numberofMetropolisSteps;
}