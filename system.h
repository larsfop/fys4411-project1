#pragma once

#include <memory>
#include <vector>

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
    double ComputeLocalEnergy();


private:
    int m_numberofparticles;
    int m_numberofdimensions;

    std::unique_ptr<class WaveFunction> m_wavefunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};