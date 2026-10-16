#pragma once

#include <vector>
#include <functional>

// Define types if not already in a shared header
using State = std::vector<double>;

// Forward declaration of the ODE function
State dSdt(const double tau, const State& Y, const double up[3], const double pp[12], const double sp[4], const SimParamStruct params, float* pSnap, int segment);
