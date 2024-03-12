
#include "Particle.h"
#include "Math/random.h"

#include <iostream>
using namespace std;

Particle::Particle(const arma::vec &position)
{
    m_NumberofDimensions = position.size();
    //m_Position(m_NumberofDimensions);
    m_InitialPositions = position;
    m_Position = position;
}

void Particle::ChangePosition(const arma::vec step)
{
    m_Position += step;
}

void Particle::ResetPosition()
{
    m_Position = m_InitialPositions;
}