
#include <memory>
#include <iostream>
#include <vector>

#include "Particle.h"
#include "Math/random.h"
#include "WaveFunctions/simplegaussian.h"
#include "InitialState/initialstate.h"
#include "Solvers/metropolis.h"
#include "sampler.h"
#include "system.h"

using namespace std;

int main()
{
    int seed = 1234;

    int numberofdimensions = 3;
    int numberofparticles = 1;
    int numberofMetropolisSteps = 1e6;
    int numberofEquilibrationSteps = 1e2;

    double alpha = 0.5;
    double beta = 1.0;
    double steplength = 0.1;

    auto rng = std::make_unique<Random>(seed);
    auto particles = SetupRandomUniformInitialState(
        numberofdimensions,
        numberofparticles,
        *rng,
        steplength
    );
    auto system = std::make_unique<System>(
        std::make_unique<SimpleGaussian>(alpha, beta),
        std::make_unique<Metropolis>(std::move(rng)),
        std::move(particles)
    );

    auto sampler = system->RunMetropolisSteps(
        steplength,
        numberofMetropolisSteps
    );

    sampler->printOutput(*system);

    return 0;
}