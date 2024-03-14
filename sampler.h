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
        int numberofMetropolisSteps
    );
    void Sample(bool acceptedstep, class System *system);
    void printOutput(class System &system);
    void ComputeAverages();
    double getEnergy() {return m_energy; };
    arma::vec getEnergyDerivatives() {return m_EnergyDerivative; };
    void CreateFile();
    void WritetoFile(System &system);

private:
    int m_stepnumber;
    int m_numberofMetropolisSteps;
    int m_numberofparticles;
    int m_numberofdimensions;
    int m_numberofacceptedsteps;
    double m_energy;
    double m_DeltaEnergy;
    double m_steplength;
    arma::vec m_DeltaPsi;
    arma::vec m_PsiEnergyDerivative;
    arma::vec m_EnergyDerivative;
    std::string m_Filename;
};