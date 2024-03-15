#pragma once

#include <random>

class Random
{
private:
    std::mt19937_64 m_generator;
    int m_seed;

public:
    Random(int seed)
    {
        m_generator = std::mt19937_64(seed);
        m_seed = seed;
    }

    int NextInt(int upper)
    {
        std::uniform_int_distribution<int> rng(0.0, upper);
        return rng(m_generator);
    }

    double NextDouble()
    {
        std::uniform_real_distribution<double> rng(0.0, 1.0);
        return rng(m_generator);
    }

    double NextGaussian()
    {
        std::normal_distribution<double> rng(0.0, 1.0);
        return rng(m_generator);
    }

    int getSeed()
    {
        return m_seed;
    }

    void set_seed(int seed)
    {
        m_generator.seed(seed);
        m_seed = seed;
    }
};