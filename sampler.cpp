
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

// setup basic sampler that does the work during VMC
Sampler::Sampler(
    int numberofparticles,
    int numberofdimensions,
    double steplength,
    int numberofMetropolisSteps
)
{
    m_stepnumber = 0;
    m_numberofMetropolisSteps = numberofMetropolisSteps;
    m_numberofparticles = numberofparticles;
    m_numberofdimensions = numberofdimensions;
    m_Energy = 0;
    m_Energy2 = 0;
    m_DeltaEnergy = 0;
    m_steplength = steplength;
    m_numberofacceptedsteps = 0;

    m_DeltaPsi = arma::vec(2);
    m_PsiEnergyDerivative = arma::vec(2);
    m_params = arma::vec(2);
    m_energies = arma::vec(numberofMetropolisSteps);
    m_positions = arma::mat(numberofdimensions,numberofparticles);
    m_hist = arma::Col<int>(numberofparticles+1);
}

// combines the earlier mentioned samplers
Sampler::Sampler(std::vector<std::unique_ptr<class Sampler>> &samplers, std::string Filename, bool Printout)
{
    m_numberofthreads = samplers.size();

    m_numberofMetropolisSteps = samplers[0]->m_numberofMetropolisSteps;
    m_numberofparticles = samplers[0]->m_numberofparticles;
    m_numberofdimensions = samplers[0]->m_numberofdimensions;
    m_steplength = samplers[0]->m_steplength;

    m_Energy = 0;
    m_Energy2 = 0;
    m_numberofacceptedsteps = 0;
    m_stepnumber = 0;

    m_Filename = samplers[0]->m_Filename;
    int N_params = samplers[0]->m_params.n_elem;
    m_DeltaPsi = arma::vec(N_params);
    m_PsiEnergyDerivative = arma::vec(N_params);
    m_params = arma::vec(N_params);
    m_energies = arma::vec(m_numberofMetropolisSteps);
    m_positions = arma::mat(m_numberofdimensions,m_numberofparticles);
    m_hist = arma::Col<int>(m_numberofparticles+1);
    for (auto &sampler : samplers)
    {
        m_Energy += sampler->m_Energy;
        m_Energy2 += sampler->m_Energy2;
        //m_variance += sampler->m_variance;
        
        m_DeltaPsi += sampler->m_DeltaPsi;
        m_PsiEnergyDerivative += sampler->m_PsiEnergyDerivative;
        //m_EnergyDerivative += sampler->m_EnergyDerivative;

        m_stepnumber += sampler->m_stepnumber;
        m_numberofacceptedsteps += sampler->m_numberofacceptedsteps; 

        for (int i = 0; i < N_params; i++)
        {
            m_params(i) += sampler->m_params(i);
        }

        if (Printout)
        {
            m_energies += sampler->m_energies;
        }

        m_positions += sampler->m_positions;
        m_hist += sampler->m_hist;
    }

    m_Energy /= m_numberofthreads*m_numberofparticles;
    m_Energy2 /= m_numberofthreads*m_numberofparticles*m_numberofparticles;
    m_DeltaPsi /= m_numberofthreads;
    m_PsiEnergyDerivative /= m_numberofthreads;

    m_variance = m_Energy2 - m_Energy*m_Energy;
    m_EnergyDerivative = 2*(m_PsiEnergyDerivative - m_DeltaPsi*m_Energy);

    m_stepnumber /= m_numberofthreads;
    m_numberofacceptedsteps /= m_numberofthreads;

    m_params /= m_numberofthreads;

    m_energies /= m_numberofthreads;
    m_positions /= m_numberofthreads;

    m_Filename = Filename;
}

// basic sampler during VMC
void Sampler::Sample(bool acceptedstep, class System *system)
{
    auto localenergy = system->ComputeLocalEnergy();
    m_Energy += localenergy;
    m_Energy2 += localenergy*localenergy;
    m_stepnumber++;
    m_numberofacceptedsteps += acceptedstep;

    arma::vec dparams = system->ComputeDerivatives();
    m_DeltaPsi += dparams;
    m_PsiEnergyDerivative += dparams*localenergy;

    m_params = system->getParameters();
}

// here comes specific sampler for specific reasons
void Sampler::ComputeDerivatives()
{
    double Energy = m_Energy/m_numberofMetropolisSteps;
    arma::vec DeltaPsi = m_DeltaPsi/m_numberofMetropolisSteps;
    arma::vec PsiEnergyDerivative = m_PsiEnergyDerivative/m_numberofMetropolisSteps;

    m_EnergyDerivative = 2*(PsiEnergyDerivative - DeltaPsi*Energy);
}

void Sampler::SampleEnergies(int iteration)
{
    m_energies(iteration) = m_Energy/(iteration+1);
}

void Sampler::SamplePositions(class System *system)
{
    for (int i = 0; i < m_numberofparticles; i++)
    {
        arma::vec pos = system->getPosition(i);
        for (int j = 0; j < m_numberofdimensions; j++)
        {
            m_positions(j,i) = pos(j);
        }
    }
}

void Sampler::SampleHist(class System *system)
{
    arma::vec pos = system->getPosition(0);
    arma::vec dr(m_numberofparticles-1);
    for (int i = 1; i < m_numberofparticles; i++)
    {
        arma::vec posi = system->getPosition(i);
        dr(i-1) = arma::norm(pos-posi);
    }
    arma::Col<int> tmp(dr.n_elem);
    double dr_max = arma::max(dr);
    for (int i = 0; i < dr.n_elem; i++)
    {
        tmp(i) = (int) std::round(dr(i)/dr_max*m_numberofparticles);
    }
    for (int i : tmp)
    {
        m_hist(i) += 1;
    }
}

void Sampler::ComputeAverages()
{
    m_Energy /= m_numberofMetropolisSteps;
    m_Energy2 /= m_numberofMetropolisSteps;
    m_variance = m_Energy2 - m_Energy*m_Energy;
}

void Sampler::printOutput(System &system)
{
    auto params = system.getParameters();


    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberofparticles << endl;
    cout << " Number of dimensions : " << m_numberofdimensions << endl;
    cout << " Number of Metropolis steps run : 2^" << std::log2(m_numberofMetropolisSteps) << endl;
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

void Sampler::printOutput()
{
    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of threads  : " << m_numberofthreads << endl;
    cout << " Number of particles  : " << m_numberofparticles << endl;
    cout << " Number of dimensions : " << m_numberofdimensions << endl;
    cout << " Number of Metropolis steps run :" << m_numberofMetropolisSteps << " (2^" << std::log2(m_numberofMetropolisSteps) << ")" << endl;
    cout << " Step length used : " << m_steplength << endl;
    cout << " Ratio of accepted steps: " << ((double) m_numberofacceptedsteps) / ((double) m_numberofMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << m_params.n_elem << endl;
    for (unsigned int i=0; i < m_params.n_elem; i++) {
        cout << " Parameter " << i+1 << " : " << m_params(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_Energy << endl;
    cout << " Variance : " << m_variance << endl;
    cout << endl;
}

void Sampler::WritetoFile()
{
    int width = 16;

    std::ofstream ofile(m_Filename, std::ofstream::app);

    ofile << setw(width-6) << m_numberofMetropolisSteps
            << setw(width) << m_numberofacceptedsteps
            << setw(width) << m_numberofdimensions
            << setw(width) << m_numberofparticles
            << setw(width) << m_steplength
            << setw(width) << m_params(0)
            << setw(width) << m_EnergyDerivative(0)
            << setw(width) << m_params(1)
            << setw(width) << m_EnergyDerivative(1)
            << setw(width) << m_Energy
            << setw(width) << m_variance
            << setw(width) << m_time.count()
            << setw(width) << omp_get_thread_num()
            << endl;
    ofile.close();
}

void Sampler::WriteEnergiestoFile()
{
    int width = 16;
    std::string Filename = "Outputs/Energies2.bin";
    m_energies.save(Filename, arma::raw_binary);
    m_positions.save("Outputs/Positions.dat", arma::raw_ascii);
    m_hist.save("Outputs/hist_IW.dat", arma::raw_ascii);
}

void Sampler::setParameters(double alpha, double beta)
{
    m_params = {alpha, beta};
}
