
#include "Particle.h"
#include "Math/random.h"

#include <iostream>
using namespace std;

Particle::Particle(const arma::vec &position)
{
    m_NumberofDimensions = position.size();
    m_InitialPositions = position;
    m_Position = position;
    m_EquilibrationPositions = arma::vec(m_NumberofDimensions);
}

void Particle::ChangePosition(const arma::vec step)
{
    m_Position += step;
}

void Particle::ResetPosition()
{
    m_Position = m_InitialPositions;
}

void Particle::SetEquilibrationPositions()
{
    m_EquilibrationPositions = m_Position;
}

void Particle::SetPositionsToEquilibration()
{
    m_Position = m_EquilibrationPositions;
}