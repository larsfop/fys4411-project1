#pragma once

#include <memory>
#include <vector>
#include "../Particle.h"
#include "Math/random.h"

std::vector<std::unique_ptr<Particle>> SetupRandomUniformInitialState(
    const int numberofdimension,
    const int numberofparticles,
    Random &rng,
    const double stepsize
);

std::vector<std::unique_ptr<Particle>> SetupRandomNormalInitialStates(
    const int numberofdimensions,
    const int numberofparticles,
    Random &rng,
    const double sqrt_dt
);