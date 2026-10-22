#pragma once

#include <vector>
#include <functional>

// Define types if not already in a shared header
using State = std::vector<double>;
// #include "OdeFunc.h"
#include <cmath>

// Forward declarations for helper functions used inside dSdt
void loadp(double* pArr, float* pSnap, const State& st, double tau0, SimParamStruct params, int segment, int activeSteps);
void interp(double* pArr, const State& st, double t, double* p, double* dpdx, double* dpdy, double* dpdt, double* dpdtdx, SimParamStruct params, int segment, int activeSteps);

State dSdt(const double tau, const State& Y, const double up[3], const double pp[12], const double sp[4], const SimParamStruct params, float* pSnap, int segment, int activeSteps) {
    // Radial Oscillation
    double R = Y[0];        // Dimensionless Radius
    double U = Y[3];        // Dimensionless Wall velocity
    double px = Y[1];       // Dimensionless position
    double vx = Y[4];       // Dimensionless translation velocity

    State F(Y.size(), 0.0);

    if (R <= 1e-9) {
        return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; 
    }

    double pb, dpdx, dpdy, dpdt, dpdtdx;
    double pArr[8];

    loadp(pArr, pSnap, Y, tau, params, segment, activeSteps);
    interp(pArr, Y, tau, &pb, &dpdx, &dpdy, &dpdt, &dpdtdx, params, segment, activeSteps);

    double inv_R = 1.0 / R;
    double base = up[2] * inv_R;
    if (base <= 0.0) base = 1e-9; 
    double p = (double)std::pow(base, sp[0]);

    // -- Radial Oscillation --
    double N = (up[0] + up[1] * U) * p; 
    N -= pp[0] * (1 + pp[5] * U) + pp[1] * inv_R + pp[2] * U * inv_R;
    N -= 1.5 * (1.0 - pp[5] * U * (1.0 / 3.0)) * U * U;
    N -= (1 + pp[5] * U) * pp[3] * pb + pp[4] * dpdt * R;
    N += 0.25 * vx * vx * params.c * params.c / params.f / params.f / pp[11] / pp[11];
    
    double D = R - pp[5] * R * U + pp[2] * pp[5];
    double rD = 1.0 / D;

    // -- Translational motion --
    double Fb = -pp[7] * R * R * R * dpdx;
    const double pi = 3.14159265359;
    // double u = -params.Pa / (params.rho * params.c) * std::sin(2.0 * pi * px) * std::cos(2.0 * pi * tau);
    double u = dpdtdx / params.rho / params.c / 4 / pi / pi / params.f * params.c / params.f;
    double Fd = -pp[8] * R * (vx * sp[3] - u);
    
    double dv = 3 * (Fb + Fd) * pp[6] * inv_R * inv_R * inv_R - 3.0 * U * inv_R * vx;

    F[0] = U;
    F[1] = vx;
    F[2] = 0.0;
    F[3] = rD * N;
    F[4] = dv;
    F[5] = 0.0;
    // F[6] = 0.0;

    return { F[0], F[1], F[2], F[3], F[4], F[5] };
    // return { F[0], F[1], F[2], F[3], F[4], F[5], F[6] };
}
