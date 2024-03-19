#pragma once

#include <armadillo>
#include <vector>

class Particle
{
    public:
        Particle(const arma::vec &position);
        void ChangePosition(const arma::vec step);
        int getNumberofDimensions() {return m_NumberofDimensions; };
        //std::vector<double> getPosition() {return m_Position; };
        arma::vec getPosition() {return m_Position; };
        void ResetPosition();
        void SetEquilibrationPositions();
        void SetPositionsToEquilibration();

    private:
        int m_NumberofDimensions;
        //std::vector<double> m_Position = std::vector<double>();
        arma::vec m_Position;
        arma::vec m_InitialPositions;
        arma::vec m_EquilibrationPositions;
};