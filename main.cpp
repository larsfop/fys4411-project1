
#include <memory>
#include <iostream>
#include <vector>
#include <time.h>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#include <map>

#include "Particle.h"
#include "Math/random.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/interactinggaussian.h"
#include "InitialState/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "sampler.h"
#include "system.h"

using namespace std;

int main(int argc, const char *argv[])
{
    int seed, numberofMetropolisSteps, numberofparticles, numberofdimensions, maxvariations;
    double alpha, beta, steplength;
    double dx = 1e-1;
    string Filename;
    bool OptimizeForParameters, Interacting, Hastings, NumericalDer, Printout;

    string input;
    ifstream ifile("config");
    while (getline(ifile, input))
    {
        string name = input.substr(0, input.find("="));
        string value = input.substr(input.find("=")+1);
        if (name == "seed")
        {seed = stoi(value); }
        else if (name == "MetropolisSteps")
        {numberofMetropolisSteps = 1 << stoi(value); }
        else if (name == "MaxVariations")
        {maxvariations = stoi(value); }
        else if (name == "Particles")
        {numberofparticles = stoi(value); }
        else if (name == "Dimensions")
        {numberofdimensions = stoi(value); }
        else if (name == "alpha")
        {alpha = stod(value); }
        else if (name == "beta")
        {beta = stod(value); }
        else if (name == "Steplength")
        {steplength = stod(value); }
        else if (name == "threadsused")
        {omp_set_num_threads(stoi(value)); }
        else if (name == "Filename")
        {Filename = value; }
        else if (name == "OptimizeForParameters")
        {OptimizeForParameters = (bool) stoi(value); }
        else if (name == "Interacting")
        {Interacting = (bool) stoi(value); }
        else if (name == "Hastings")
        {Hastings = (bool) stoi(value); }
        else if (name == "NumericalDer")
        {NumericalDer = (bool) stoi(value); }
        else if (name=="Printout")
        {Printout = (bool) stoi(value); }
    }

    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            string input = argv[i];
            string name = input.substr(0, input.find("="));
            string value = input.substr(input.find("=")+1);
            if (name == "seed")
            {seed = stoi(value); }
            else if (name == "MetropolisSteps")
            {numberofMetropolisSteps = 1 << stoi(value); }
            else if (name == "MaxVariations")
            {maxvariations = stoi(value); }
            else if (name == "Particles")
            {numberofparticles = stoi(value); }
            else if (name == "Dimensions")
            {numberofdimensions = stoi(value); }
            else if (name == "alpha")
            {alpha = stod(value); }
            else if (name == "beta")
            {beta = stod(value); }
            else if (name == "Steplength")
            {steplength = stod(value); }
            else if (name == "threadsused")
            {omp_set_num_threads(stoi(value)); }
            else if (name == "Filename")
            {Filename = value; }
            else if (name == "OptimizeForParameters")
            {OptimizeForParameters = (bool) stoi(value); }   
            else if (name == "Interacting")
            {Interacting = (bool) stoi(value); }
            else if (name == "Hastings")
            {Hastings = (bool) stoi(value); }
            else if (name == "NumericalDer")
            {NumericalDer = (bool) stoi(value); }
            else if (name=="Printout")
            {Printout = (bool) stoi(value); }
        }
    }

    int numberofEquilibrationSteps = 1e5;

    double eta = 0.1;
    double tol = 1e-5;
    int maxiter = 1e2;

    string Path = "Outputs/";
    int width = 16;
    Filename = Path + Filename + ".dat";
    ofstream outfile(Filename);
    outfile << setw(width-6) << "MC-cycles"
            << setw(width) << "Accepted_Steps"
            << setw(width) << "Dimensions"
            << setw(width) << "Particles"
            << setw(width) << "Steplength"
            << setw(width) << "alpha"
            << setw(width) << "dalpha"
            << setw(width) << "beta"
            << setw(width) << "dbeta"
            << setw(width) << "Energy"
            << setw(width) << "Variance"
            << setw(width) << "Time"
            << setw(width) << "Thread"
            << endl;
    outfile.close();

    std::ofstream ofile("Energies.dat");
    ofile.close();

    auto t1 = std::chrono::system_clock::now();

    std::vector<std::unique_ptr<class Sampler>> samplers;
    #pragma omp parallel
    {
        int threadnumber = omp_get_thread_num();
        auto rng = std::make_unique<Random>(seed+threadnumber);

        std::vector<std::unique_ptr<class Particle>> particles;
        if (Hastings)
        {
            particles = SetupRandomNormalInitialStates(
                numberofdimensions,
                numberofparticles,
                *rng,
                std::sqrt(steplength)
            );
        }
        else
        {
            particles = SetupRandomUniformInitialState(
                numberofdimensions,
                numberofparticles,
                *rng,
                steplength
            );
        }

        std::unique_ptr<class WaveFunction> wavefunction;
        std::unique_ptr<class MonteCarlo> solver;

        if (Interacting)
        {
            wavefunction = std::make_unique<class InteractingGaussian>(alpha, beta);
        }
        else
        {
            wavefunction = std::make_unique<class SimpleGaussian>(alpha, beta);
        }

        if (Hastings)
        {
            solver = std::make_unique<class MetropolisHastings>(std::move(rng));
        }
        else
        {
            solver = std::make_unique<class Metropolis>(std::move(rng));
        }

        auto system = std::make_unique<System>(
            std::move(wavefunction),
            std::move(solver),
            std::move(particles),
            Filename,
            Printout
        );

        auto acceptedEquilibrationSteps = system->RunEquilibrationSteps(
            steplength,
            numberofEquilibrationSteps
        );

        std::unique_ptr<Sampler> sampler;
        if (OptimizeForParameters)
        {
            sampler = system->FindOptimalParameters(
                steplength,
                numberofMetropolisSteps,
                eta,
                tol,
                maxiter
            );
        }
        else
        {
            sampler = system->VaryParameters(
                steplength,
                numberofMetropolisSteps,
                maxvariations
            ); 
        }

        samplers.push_back(std::move(sampler));
    }
    std::unique_ptr<Sampler> sampler = std::make_unique<class Sampler>(samplers, Filename);

    //sampler->WritetoFile();

    sampler->printOutput();
    
    auto t2 = std::chrono::system_clock::now();

    std::chrono::duration<double> time = t2 - t1;
    cout << fixed << setprecision(3) << endl;
    cout << "Time : " << time.count() << " seconds" << endl;

    if (NumericalDer)
    {
        auto t1 = std::chrono::system_clock::now();

        auto rng = std::make_unique<Random>(seed);

        auto particles = SetupRandomUniformInitialState(
            numberofdimensions,
            numberofparticles,
            *rng,
            steplength
        );

        auto system = std::make_unique<System>(
            std::make_unique<class SimpleGaussianNumerical>(alpha, beta, dx),
            std::make_unique<class Metropolis>(std::move(rng)),
            std::move(particles),
            Filename,
            Printout
        );

        auto acceptedEquilibrationSteps = system->RunEquilibrationSteps(
            steplength,
            numberofEquilibrationSteps
        );

        auto sampler = system->VaryParameters(
            steplength,
            numberofMetropolisSteps,
            maxvariations
        );

        sampler->printOutput();

        auto t2 = std::chrono::system_clock::now();
        std::chrono::duration<double> time = t2 - t1;
        cout << "Numerical time : " << time.count() << " seconds" << endl;
    }

    return 0;
}