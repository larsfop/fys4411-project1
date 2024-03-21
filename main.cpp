
#include <memory>
#include <iostream>
#include <vector>
#include <time.h>
#include <iomanip>
#include <stdio.h>
#include <chrono>

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

// Profiling shows more than half the time running is spent computing 
// the double derivative for the interacting case, especially for many particles

int main(int argc, const char *argv[])
{
    int seed = 1234;

    // Number of MonteCarlo steps given in binary powers
    // i.e. = 2^{terminal input}
    int numberofMetropolisSteps = 1 << atoi(argv[1]);
    // int numberofMetropolisSteps = stod(argv[1]);
    int numberofparticles = atoi(argv[2]);
    int numberofdimensions = atoi(argv[3]);
    int numberofEquilibrationSteps = 1e5;

    double alpha = atof(argv[4]);
    double beta = atof(argv[5]);
    double steplength = 0.01;

    double eta = 0.1;
    double tol = 1e-7;
    int maxiter = 1e3;

    bool OptimizeForParameters = false;

    omp_set_num_threads(atoi(argv[6]));

    string Path = "Outputs/";
    int width = 20;
    string Filename = Path + argv[7] + ".dat";
    ofstream outfile(Filename);
    outfile << setw(width-8) << "alpha"
            << setw(width) << "dalpha"
            << setw(width) << "beta"
            << setw(width) << "dbeta"
            << setw(width) << "Energy"
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

        auto particles = SetupRandomNormalInitialStates(
            numberofdimensions,
            numberofparticles,
            *rng,
            std::sqrt(steplength)
        );

        auto system = std::make_unique<System>(
            std::make_unique<InteractingGaussian>(alpha, beta),
            std::make_unique<MetropolisHastings>(std::move(rng)),
            std::move(particles)
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
            sampler = system->RunMetropolisSteps(
                steplength,
                numberofMetropolisSteps
            ); 
        }

        samplers.push_back(std::move(sampler));
    }
    std::unique_ptr<Sampler> sampler = std::make_unique<class Sampler>(samplers);
    sampler->printOutput();

    /*auto rng = std::make_unique<Random>(seed);
    // auto particles = SetupRandomUniformInitialState(
    //     numberofdimensions,
    //     numberofparticles,
    //     *rng,
    //     steplength
    // );

    auto particles = SetupRandomNormalInitialStates(
        numberofdimensions,
        numberofparticles,
        *rng,
        sqrt(steplength)
    );

    auto system = std::make_unique<System>(
        std::make_unique<SimpleGaussian>(alpha, beta),
        std::make_unique<MetropolisHastings>(std::move(rng)),
        std::move(particles)
    );
    // auto acceptedEquilibrationSteps = system->RunEquilibrationSteps(
    //     steplength,
    //     numberofMetropolisSteps
    // );
    auto sampler = system->FindOptimalParameters(
        steplength,
        numberofMetropolisSteps,
        eta,
        tol,
        maxiter
    );

    // auto sampler = system->RunMetropolisSteps(
    //     steplength,
    //     numberofMetropolisSteps
    // );*/

    auto t2 = std::chrono::system_clock::now();

    std::chrono::duration<double> time = t2 - t1;
    cout << fixed << setprecision(3) << endl;
    cout << "Time : " << time.count() << " seconds" << endl;

    // sampler->printOutput(*system);

    return 0;
}