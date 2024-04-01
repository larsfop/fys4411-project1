
#include "system.h"
#include "sampler.h"
#include "Particle.h"
#include "WaveFunctions/WaveFunctions.h"
#include "InitialState/initialstate.h"
#include "Solvers/montecarlo.h"

#include <iostream>
#include <stdio.h>
using std::cout;
using std::endl;

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
    //int j = 0;
    auto t1 = std::chrono::system_clock::now();
    for (int i = 0; i < numberofMetropolisSteps; i++)
    {
        bool acceptedStep;
        for (int j = 0; j < m_numberofparticles; j++)
        {
            acceptedStep = m_solver->Step(steplength, *m_wavefunction, m_particles, j);
        }

        sampler->Sample(acceptedStep, this);
        if (m_Printout)
        {
            sampler->WriteEnergiestoFile(*this, i+1);
        }
    }
        
    sampler->ComputeDerivatives();
    sampler->ComputeAverages();
    //sampler->WritetoFile(*this);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> time = t2 - t1;
    sampler->SetTime(time);

    return sampler;
}

int System::RunEquilibrationSteps(
        double stepLength,
        int numberOfEquilibrationSteps)
{
    int acceptedSteps = 0;

    for (int i = 0; i < numberOfEquilibrationSteps; i++) {
        for (int j = 0; j < m_numberofparticles; j++)
        {
            acceptedSteps += m_solver->Step(stepLength, *m_wavefunction, m_particles, j);
        }
    }

    for (int i = 0; i < m_numberofparticles; i++)
    {
        m_particles[i]->SetEquilibrationPositions();
    }

    return acceptedSteps;
}

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
    while (gradient > tolerance && iterations < maxiterations)
    {
        //cout << "Iteration : " << iterations+1 << endl;
        sampler = this->RunMetropolisSteps(steplength, numberofMetropolisSteps);
        arma::vec energyderivatives = sampler->getEnergyDerivatives();

        gradient = 0;
        for (int i = 0; i < nparams-1; i++)
        {
            params(i) -= learningrate*energyderivatives(i)/m_numberofparticles;
            //cout << "Parameter " << i+1 << " : " << params(i) << endl;
            //cout << "EnergyDerivative " << i+1 << " : " << energyderivatives(i) << endl;
            gradient += std::abs(energyderivatives(i));
        }

        sampler->setFilename(m_Filename);
        //sampler->ComputeAverages();
        sampler->WritetoFile();
        
        m_wavefunction->ChangeParameters(params(0), params(1));
        sampler->setParameters(params(0), params(1));
        for (int i = 0; i < m_numberofparticles; i++)
        {
            m_particles[i]->SetPositionsToEquilibration();
        }

        iterations++;
    }
    //cout << "Max iterations : " << iterations << endl;
    return sampler;
}

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