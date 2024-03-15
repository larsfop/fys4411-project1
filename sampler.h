#pragma once

#include <memory>
#include <armadillo>

class Sampler
{
public:
    Sampler(
        int numberofparticles,
        int numberofdimensions,
        double steplength,
        int numberofMetropolisSteps,
        int numberofthreads
    );
    void Sample(bool acceptedstep, class System *system);
    void printOutput(class System &system);
    void ComputeAverages();
    double getEnergy() {return m_Energy; };
    arma::vec getEnergyDerivatives() {return m_EnergyDerivative; };
    void CreateFile();
    void WritetoFile(System &system);
    void WriteEnergiestoFile(System &system, int iteration);

private:
    int m_stepnumber;
    int m_numberofMetropolisSteps;
    int m_numberofparticles;
    int m_numberofdimensions;
    int m_numberofacceptedsteps;
    int m_numberofthreads;
    double m_Energy;
    double m_Energy2;
    double m_variance;
    double m_DeltaEnergy;
    double m_steplength;
    arma::vec m_DeltaPsi;
    arma::vec m_PsiEnergyDerivative;
    arma::vec m_EnergyDerivative;
    std::string m_Filename;
};