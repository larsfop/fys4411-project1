
#include <memory>
#include "sampler.h"
#include "system.h"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;

using std::setw;
using std::setprecision;
using std::fixed;
using std::scientific;

Sampler::Sampler(
    int numberofparticles,
    int numberofdimensions,
    double steplength,
    int numberofMetropolisSteps,
    int numberofthreads
)
{
    m_stepnumber = 0;
    m_numberofMetropolisSteps = numberofMetropolisSteps;
    m_numberofparticles = numberofparticles;
    m_numberofdimensions = numberofdimensions;
    m_Energy = 0;
    m_DeltaEnergy = 0;
    m_steplength = steplength;
    m_numberofacceptedsteps = 0;
    m_numberofthreads = numberofthreads;

    m_DeltaPsi = arma::vec(2);
    m_PsiEnergyDerivative = arma::vec(2);
    m_EnergyDerivative = arma::vec(2);

    m_Filename = "Results.dat";
}

void Sampler::Sample(bool acceptedstep, class System *system)
{
    auto localenergy = system->ComputeLocalEnergy();
    // cout << localenergy << endl;
    m_Energy += localenergy;
    m_Energy2 += localenergy*localenergy;
    m_stepnumber++;
    m_numberofacceptedsteps += acceptedstep;

    arma::vec dparams = system->ComputeDerivatives();
    m_DeltaPsi += dparams;
    m_PsiEnergyDerivative += dparams*localenergy;
}

void Sampler::ComputeAverages()
{
    m_Energy /= m_numberofMetropolisSteps;
    m_Energy2 /= m_numberofMetropolisSteps;
    m_DeltaPsi /= m_numberofMetropolisSteps;
    m_PsiEnergyDerivative /= m_numberofMetropolisSteps;
    m_variance = m_Energy2 - m_Energy*m_Energy;
    // m_PsiEnergyDerivative.print();
    // m_DeltaPsi.print();
    // cout << m_Energy << endl;
    m_EnergyDerivative = 2*(m_PsiEnergyDerivative - m_DeltaPsi*m_Energy);

    //m_stepnumber /= m_numberofthreads;
    //m_numberofacceptedsteps /= m_numberofthreads;
}

void Sampler::printOutput(System &system)
{
    auto params = system.getParameters();


    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberofparticles << endl;
    cout << " Number of dimensions : " << m_numberofdimensions << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberofMetropolisSteps) << endl;
    cout << " Step length used : " << m_steplength << endl;
    cout << " Ratio of accepted steps: " << ((double) m_numberofacceptedsteps) / ((double) m_numberofMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << params.n_elem << endl;
    for (unsigned int i=0; i < params.n_elem; i++) {
        cout << " Parameter " << i+1 << " : " << params(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_Energy << endl;
    cout << " Variance : " << m_variance << endl;
    cout << endl;
}

void Sampler::CreateFile()
{
    int width = 20;
    std::ofstream ofile(m_Filename, std::ofstream::trunc);
    ofile << setw(width-8) << "alpha"
            << setw(width) << "EnergyDerivative"
            << setw(width) << "Energy"
            << endl;
    ofile.close();
}

void Sampler::WritetoFile(System &system)
{
    auto params = system.getParameters();
    int width = 20;

    std::ofstream ofile(m_Filename, std::ofstream::app);

    // ofile << setprecision(6);
    ofile << setw(width-8) << params(0)
            << setw(width) << m_EnergyDerivative(0)
            << setw(width) << params(1)
            << setw(width) << m_EnergyDerivative(1)
            << setw(width) << m_Energy
            << endl;
    ofile.close();
}

void Sampler::WriteEnergiestoFile(System &system, int iteration)
{
    std::ofstream ofile("Energies.dat", std::ofstream::app);
    ofile << m_Energy/iteration << endl;
    ofile.close(); 
}
