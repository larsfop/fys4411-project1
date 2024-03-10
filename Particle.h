#pragma once

#include <armadillo>
#include <vector>

class Particle
{
    public:
        Particle(const std::vector<double> &position);
        void ChangePosition(const std::vector<double> step);
        int getNumberofDimensions() {return m_NumberofDimensions; };
        std::vector<double> getPosition() {return m_Position; };

    private:
        int m_NumberofDimensions;
        std::vector<double> m_Position = std::vector<double>();
};