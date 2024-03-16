
#include <memory>
#include <iostream>
#include <vector>
#include <time.h>
#include <iomanip>
#include <stdio.h>

#include "Particle.h"
#include "Math/random.h"
#include "WaveFunctions/simplegaussian.h"
#include "InitialState/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "sampler.h"
#include "system.h"

using namespace std;

int main(int argc, const char *argv[])
{
    int seed = 1234;

    int numberofdimensions = atoi(argv[4]);
    int numberofparticles = atoi(argv[3]);
    int numberofMetropolisSteps = stod(argv[1]);
    int numberofEquilibrationSteps = 1e2;

    double alpha = atof(argv[5]);
    double beta = atof(argv[6]);
    double steplength = 0.01;

    double eta = 0.1;
    double tol = 1e-7;
    int maxiter = stod(argv[2]); //1e3;

    omp_set_num_threads(atoi(argv[7]));

    int width = 20;
    string Filename = "Results.dat";
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

    double t_total;
    clock_t t1,t2;

    t1 = clock();

    int numberofthreads = omp_get_num_threads();
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
            std::make_unique<SimpleGaussian>(alpha, beta),
            std::make_unique<MetropolisHastings>(std::move(rng)),
            std::move(particles)
        );

        auto sampler = system->FindOptimalParameters(
            steplength,
            numberofMetropolisSteps,
            eta,
            tol,
            maxiter
        );

        samplers.push_back(std::move(sampler));
        //samplers[threadnumber] = std::move(sampler);
        // gives bus error
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

    t2 = clock();

    // sampler->printOutput(*system);

    t_total = t2 - t1;
    double time = ((double) (t_total))/CLOCKS_PER_SEC;
    cout << "Time : " << time << endl;

    return 0;
}