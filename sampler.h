#pragma once

#include <memory>
#include <armadillo>
#include <vector>

#include "Particle.h"
class Sampler
{
public:
    Sampler(
        int numberofparticles,
        int numberofdimensions,
        double steplength,
        int numberofMetropolisSteps
    );
    Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers, std::string Filename, bool Printout);
    void Sample(bool acceptedstep, class System *system);
    void printOutput(class System &system);
    void printOutput();
    void SampleEnergies(int iteration);
    void SamplePositions(class System *system);
    void SampleHist(class System *system);
    void ComputeDerivatives();
    void ComputeAverages();
    double getEnergy() {return m_Energy; };
    arma::vec getEnergyDerivatives() {return m_EnergyDerivative; };
    void WritetoFile();
    void WriteEnergiestoFile();
    void setParameters(double alpha, double beta);
    void setFilename(std::string Filename) {m_Filename = Filename; };
    void SetTime(std::chrono::duration<double> time) {m_time = time; };

private:
    int m_stepnumber;
    int m_numberofMetropolisSteps;
    int m_numberofparticles;
    int m_numberofdimensions;
    int m_numberofacceptedsteps;
    int m_numberofthreads;
    int m_Nx = 100;
    int m_Ny = 100;
    int m_Nz = 100;
    double m_Energy;
    double m_Energy2;
    double m_variance;
    double m_DeltaEnergy;
    double m_steplength;
    arma::vec m_DeltaPsi;
    arma::vec m_PsiEnergyDerivative;
    arma::vec m_EnergyDerivative;
    arma::vec m_params;
    arma::vec m_energies;
    arma::mat m_positions;
    arma::Col<int> m_hist;
    std::string m_Filename;
    std::chrono::duration<double> m_time;
};