#pragma once

#include <memory>
#include <armadillo>
#include <vector>

class Sampler
{
public:
    Sampler(
        int numberofparticles,
        int numberofdimensions,
        double steplength,
        int numberofMetropolisSteps
    );
    Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers);
    void Sample(bool acceptedstep, class System *system);
    void printOutput(class System &system);
    void printOutput();
    void ComputeAverages();
    double getEnergy() {return m_Energy; };
    arma::vec getEnergyDerivatives() {return m_EnergyDerivative; };
    void CreateFile();
    void WritetoFile(System &system);
    void WriteEnergiestoFile(System &system, int iteration);
    void setParameters(double alpha, double beta);

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
    arma::vec m_params;
    std::string m_Filename;
};