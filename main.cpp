
#include <memory>
#include <iostream>
#include <vector>

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

    int numberofdimensions = atoi(argv[3]);
    int numberofparticles = atoi(argv[2]);
    int numberofMetropolisSteps = stod(argv[1]);
    int numberofEquilibrationSteps = 1e2;

    double alpha = 0.3;
    double beta = 0.7;
    double steplength = 0.01;

    double eta = 0.1/numberofparticles;
    double tol = 1e-7;
    int maxiter = 1e3;

    auto rng = std::make_unique<Random>(seed);
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
    // );

    sampler->printOutput(*system);

    return 0;
}