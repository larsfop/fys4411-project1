#include <memory>
#include <initialstate.h>

std::vector<std::unique_ptr<Particle>> SetupRandomUniformInitialState(
    const int numberofdimension,
    const int numberofparticles,
    std::unique_ptr<class Random> &rng,
    const double stepsize
)
{
    auto particles = std::vector<std::unique_ptr<Particle>>();
    for (int i = 0; i < numberofparticles; i++)
    {
        arma::vec pos(numberofdimension, arma::fill::zeros);
        for (int j = 0; j < numberofdimension; j++)
        {
            pos(j) = stepsize * (rng->NextDouble() - 0.5);
        }
        particles.push_back(std::make_unique<Particle>(pos));
    }
    return particles;
}