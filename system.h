#pragma once

#include <memory>
#include <vector>
#include <armadillo>
#include <omp.h>

class System
{
public:
    System(
        std::unique_ptr<class WaveFunction> wavefunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles
    );
    std::unique_ptr<class Sampler> RunMetropolisSteps(
        double stepLength,
        int numberofMetropolisSteps
    );
    int RunEquilibrationSteps(
        double steplength,
        int numberofMetropolisSteps
    );
    std::unique_ptr<class Sampler> FindOptimalParameters(
        double steplength,
        int numberofMetropolisSteps,
        double learningrate,
        double tolerance,
        int maxiterations
    );
    double ComputeLocalEnergy();
    arma::vec ComputeDerivatives();
    arma::vec getParameters();

private:
    int m_numberofparticles;
    int m_numberofdimensions;

    std::unique_ptr<class WaveFunction> m_wavefunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};