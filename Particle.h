#pragma once

#include <armadillo>

class Particle
{
    public:
        Particle(arma::vec &position);
        void ChangePosition(const arma::vec step);
        int getNumberofDimensions() {return m_NumberofDimensions; };
        arma::vec getPosition() {return m_Position; };

    private:
        int m_NumberofDimensions;
        arma::vec m_Position;
};