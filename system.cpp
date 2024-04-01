
#include "system.h"
#include "sampler.h"
#include "Particle.h"
#include "WaveFunctions/WaveFunctions.h"
#include "InitialState/initialstate.h"
#include "Solvers/montecarlo.h"

// The main bus that combines all of the systems that makes the program run
System::System(
    std::unique_ptr<class WaveFunction> wavefunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles,
    std::string Filename,
    bool Printout
)
{
    m_numberofparticles = particles.size();
    m_numberofdimensions = particles[0]->getNumberofDimensions();
    m_wavefunction = std::move(wavefunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);

    m_Filename = Filename;
    m_Printout = Printout;
}

// runs all of the MC cycles
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
        bool acceptedStep;
        acceptedStep = m_solver->Step(steplength, *m_wavefunction, m_particles);
        sampler->Sample(acceptedStep, this);

        // if you want to do some extra printing for blocking and one-body
        if (m_Printout)
        {
            sampler->SampleEnergies(i);
            sampler->SamplePositions(this);
            sampler->SampleHist(this);
        }
    }
        
    sampler->ComputeDerivatives();
    sampler->ComputeAverages();

    return sampler;
}

int System::RunEquilibrationSteps(
        double stepLength,
        int numberOfEquilibrationSteps)
{
    int acceptedSteps = 0;

    for (int i = 0; i < numberOfEquilibrationSteps; i++) {
        acceptedSteps += m_solver->Step(stepLength, *m_wavefunction, m_particles);
    }

    for (int i = 0; i < m_numberofparticles; i++)
    {
        m_particles[i]->SetEquilibrationPositions();
    }

    return acceptedSteps;
}

// helper functions for other classes that can't access them directly
double System::ComputeLocalEnergy()
{
    return m_wavefunction->LocalEnergy(m_particles);
}

arma::vec System::ComputeDerivatives()
{
    return m_wavefunction->dPsidParam(m_particles);
}

arma::vec System::getParameters()
{
    return m_wavefunction->getParameters();
}

arma::vec System::getPosition(int index)
{
    return m_particles[index]->getPosition();
}

// optimization algorithm
std::unique_ptr<class Sampler> System::FindOptimalParameters(
    double steplength,
    int numberofMetropolisSteps,
    double learningrate,
    double tolerance,
    int maxiterations
)
{
    int nparams = m_wavefunction->getParameters().n_elem;

    arma::vec params = m_wavefunction->getParameters();
    double gradient = 1;

    auto sampler = std::make_unique<Sampler>(
        m_numberofparticles,
        m_numberofdimensions,
        steplength,
        numberofMetropolisSteps
    );

    int iterations = 0;
    // tolerance and iteration given in main file
    while (gradient > tolerance && iterations < maxiterations)
    {
        sampler = this->RunMetropolisSteps(steplength, numberofMetropolisSteps);
        arma::vec energyderivatives = sampler->getEnergyDerivatives();

        gradient = 0;
        for (int i = 0; i < nparams-1; i++)
        {
            params(i) -= learningrate*energyderivatives(i)/m_numberofparticles;
            // gradient only computed for tolerance
            gradient += std::abs(energyderivatives(i));
        }

        sampler->setFilename(m_Filename);
        sampler->WritetoFile();
        
        m_wavefunction->ChangeParameters(params(0), params(1));
        sampler->setParameters(params(0), params(1));
        // after each optimization reset the position to before the start
        for (int i = 0; i < m_numberofparticles; i++)
        {
            m_particles[i]->SetPositionsToEquilibration();
        }

        iterations++;
    }
    return sampler;
}

// The optimizations algorithms clumsier little brother
// does the same just worse
std::unique_ptr<class Sampler> System::VaryParameters(
    double steplength,
    int numberofMetropolisSteps,
    int maxvariations
)
{
    int nparams = m_wavefunction->getParameters().n_elem;

    arma::vec params = m_wavefunction->getParameters();
    double alpha = params(0);
    double beta = params(1);

    auto sampler = std::make_unique<Sampler>(
        m_numberofparticles,
        m_numberofdimensions,
        steplength,
        numberofMetropolisSteps
    );

    double DeltaParams = 0.02;
    for (int i = 0; i < maxvariations; i++)
    {
        sampler = this->RunMetropolisSteps(steplength, numberofMetropolisSteps);
        sampler->setFilename(m_Filename);
        sampler->WritetoFile();

        alpha += DeltaParams;
        m_wavefunction->ChangeParameters(alpha, beta);
        sampler->setParameters(alpha, beta);

        for (int j = 0; j < m_numberofparticles; j++)
        {
            m_particles[j]->SetPositionsToEquilibration();
        }
    }

    return sampler;
}