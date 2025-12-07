#pragma once

class LearningRateScheduler
{
public:
    LearningRateScheduler();
    LearningRateScheduler(float baseRate, float minRate, int warmupSteps, int flatSteps, int decaySteps);
    float GetLearningRate(int step) const;

private:
    float m_baseRate = 1e-3f; // Initial peak learning rate
    float m_minRate = 1e-4f; // Floor learning rate at end of schedule
    int m_warmupSteps = 10000;
    int m_flatSteps = 100000;
    int m_decaySteps = 100000;
};
