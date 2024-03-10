
#include "Particle.h"
#include "Math/random.h"

Particle::Particle(arma::vec &position)
{
    m_NumberofDimensions = position.n_elem;
    m_Position(m_NumberofDimensions);
}

void Particle::ChangePosition(const arma::vec step)
{
    m_Position += step;
}