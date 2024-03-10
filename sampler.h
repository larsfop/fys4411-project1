#pragma once

#include <memory>

class Sampler
{
public:
    Sampler(
        int numberofparticles,
        int numberofdimensions,
        double steplength,
        int numberofMetropolisSteps
    );
    void Sample(bool acceptedstep, class System *system);
    void printOutput(class System &system);
    void ComputeAverages();
    double getEnergy() {return m_energy; };

private:
    int m_stepnumber;
    int m_numberofMetropolisSteps;
    int m_numberofparticles;
    int m_numberofdimensions;
    int m_numberofacceptedsteps;
    double m_energy;
    double m_DeltaEnergy;
    double m_steplength;
};