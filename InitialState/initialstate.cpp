#include <memory>
#include <initialstate.h>

#include <iostream>
using namespace std;
std::vector<std::unique_ptr<Particle>> SetupRandomUniformInitialState(
    const int numberofdimension,
    const int numberofparticles,
    Random &rng,
    const double stepsize
)
{
    auto particles = std::vector<std::unique_ptr<Particle>>();
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos(numberofdimension);
        for (int j = 0; j < numberofdimension; j++)
        {
            pos(j) = stepsize * (rng.NextDouble() - 0.5);
        }
        particles.push_back(std::make_unique<Particle>(pos));
    }
    return particles;
}

std::vector<std::unique_ptr<Particle>> SetupRandomNormalInitialStates(
    const int numberofdimensions,
    const int numberofparticles,
    Random &rng,
    const double sqrt_dt
)
{
    auto particles = std::vector<std::unique_ptr<Particle>>();
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos(numberofdimensions);
        for (int j = 0; j < numberofdimensions; j++)
        {
            pos(j) = sqrt_dt * rng.NextGaussian();
        }
        particles.push_back(std::make_unique<Particle>(pos));
    }
    return particles;
}