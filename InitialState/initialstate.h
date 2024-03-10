#pragma once

#include <memory>
#include <vector>
#include "../Particle.h"
#include "Math/random.h"

std::vector<std::unique_ptr<Particle>> SetupRandomUniformInitialState(
    const int numberofdimension,
    const int numberofparticles,
    std::unique_ptr<class Random> &rng,
    const double stepsize
);