
#include "Particle.h"
#include "Math/random.h"

#include <iostream>
using namespace std;

Particle::Particle(const std::vector<double> &position)
{
    m_NumberofDimensions = position.size();
    //m_Position(m_NumberofDimensions);
    m_Position = position;
}

void Particle::ChangePosition(const std::vector<double> step)
{
    for (int i = 0; i < step.size(); i++)
    {
        m_Position[i] += step[i];
    }
}