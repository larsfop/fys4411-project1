#pragma once

#include <memory>
#include <vector>
#include "Math/random.h"

class MonteCarlo
{
public:
    MonteCarlo(std::unique_ptr<class Random> rng);
    virtual ~MonteCarlo() = default;

    virtual bool Step(
        double stepsize,
        class WaveFunction &wavefunction,
        std::vector<std::unique_ptr<class Particle>> &particles,
        int index
    ) = 0;

protected:
    std::unique_ptr<class Random> m_rng;
};