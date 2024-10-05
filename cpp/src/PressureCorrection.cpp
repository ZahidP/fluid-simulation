#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

#include "PressureCorrection.hpp"

/**
 * @brief Variant of the SIMPLE algorithm for pressure correction in CFD simulations.
 *
 * This implementation includes methods for calculating intermediate velocities,
 * applying gravity, performing advection, solving the pressure equation using
 * the Jacobi method (both serial and parallel), and enforcing incompressibility.
 *
 * Additional considerations:
 * - Central difference schemes can introduce numerical diffusion.
 * - Semi-Lagrangian advection and smaller time steps can mitigate diffusion effects.
 */
PressureCorrection::PressureCorrection(int numRows, int numCols,
                                       double timeStep, double cellSize)
    : GridSolver(numRows, numCols, timeStep, cellSize) {
    if (numRows <= 0 || numCols <= 0) {
        throw std::invalid_argument("Number of rows and columns must be positive.");
    }
    if (timeStep <= 0.0 || cellSize <= 0.0) {
        throw std::invalid_argument("Time step and cell size must be positive.");
    }
    // Initialize grid or other members as needed
}

void PressureCorrection::calculateIntermediateVelocity(float gravity) {
    size_t rows = grid.size();
    size_t cols = grid[0].size();

    double nu = 0.01; // Example viscosity value; adjust as needed
    double dy = dx;

    gravity = gravity * dt;
    float densityThreshold = 0.1f; // Adjust this value based on your simulation

    for (int i = 1; i < static_cast<int>(rows) - 1; ++i) {
        for (int j = 1; j < static_cast<int>(cols) - 1; ++j) {
            grid[i][j].u_star =
                grid[i][j].U +
                dt * nu *
                    ((grid[i + 1][j].U - 2.0f * grid[i][j].U + grid[i - 1][j].U) / (dx * dx) +
                     (grid[i][j + 1].U - 2.0f * grid[i][j].U + grid[i][j - 1].U) / (dy * dy));

            grid[i][j].v_star =
                grid[i][j].V +
                dt * nu *
                    ((grid[i + 1][j].V - 2.0f * grid[i][j].V + grid[i - 1][j].V) / (dx * dx) +
                     (grid[i][j + 1].V - 2.0f * grid[i][j].V + grid[i][j - 1].V) / (dy * dy));
        }
    }
}

void PressureCorrection::setCell(size_t row, size_t col, float U, float V,
                                 float density) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    grid[row][col] = {U, V, density, 0.0f, 1}; // Initialize pressure to 0 and fluid flag to 1
}

PressureCorrection::Cell PressureCorrection::getCell(size_t row,
                                                     size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    return grid[row][col];
}

void PressureCorrection::updateVelocityFromGravity(float gravity) {
    gravity = gravity * dt;
    float densityThreshold = 0.1f; // Adjust this value based on your simulation
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            if (grid[i][j].density > densityThreshold) {
                grid[i][j].V += gravity * grid[i][j].density; // Applied density
            } else {
                grid[i][j].V = 0.0f; // Reset velocity in near-empty cells
            }
        }
    }
}

void PressureCorrection::updateVelocitiesFromPressure() {
    /**
     * Updates the velocity fields based on the pressure gradients.
     * This is part of the explicit pressure correction step in the SIMPLE algorithm.
     */
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // Update U (horizontal velocity)
            if (j < cols - 2) {
                float pressureGradient =
                    (grid[i][j].pressure - grid[i][j - 1].pressure) / dx;
                grid[i][j].U = grid[i][j].u_star - pressureGradient * dt *
                                                   grid[i][j].fluid /
                                                   grid[i][j].density;
            }

            // Update V (vertical velocity)
            if (i < rows - 2) {
                float pressureGradient =
                    (grid[i][j].pressure - grid[i - 1][j].pressure) / dx; // Changed from dx to dy
                grid[i][j].V = grid[i][j].v_star - pressureGradient * dt *
                                                   grid[i][j].fluid /
                                                   grid[i][j].density;
            }
        }
    }
}

float PressureCorrection::interpolate(float x, float y, int field) {
    /* Assumes x and y are already in grid coordinates (i.e., integers represent
     * grid points) */

    // Clamp x and y to [0, cols - 2] and [0, rows - 2] to avoid out-of-bounds
    x = std::max(0.0f, std::min(x, static_cast<float>(cols - 2)));
    y = std::max(0.0f, std::min(y, static_cast<float>(rows - 2)));

    int j = static_cast<int>(std::floor(x)); // column index
    int i = static_cast<int>(std::floor(y)); // row index
    float fx = x - j;
    float fy = y - i;

    float v00 = (field == 0)   ? grid[i][j].U
               : (field == 1) ? grid[i][j].V
                              : grid[i][j].density;
    float v10 = (field == 0)   ? grid[i][j + 1].U
               : (field == 1) ? grid[i][j + 1].V
                              : grid[i][j + 1].density;
    float v01 = (field == 0)   ? grid[i + 1][j].U
               : (field == 1) ? grid[i + 1][j].V
                              : grid[i + 1][j].density;
    float v11 = (field == 0)   ? grid[i + 1][j + 1].U
               : (field == 1) ? grid[i + 1][j + 1].V
                              : grid[i + 1][j + 1].density;

    float result = (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 +
                   (1.0f - fx) * fy * v01 + fx * fy * v11;

    return result;
}

void PressureCorrection::semiLagrangianAdvection(bool advectDensity, bool advectU, bool advectV) {
    std::vector<std::vector<Cell>> newGrid = grid;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // Trace particle back in time
            float x = j - grid[i][j].U * dt / dx;
            float y = i - grid[i][j].V * dt / dx; // Corrected to dy

            // Clamp coordinates to grid boundaries
            x = std::max(0.5f, std::min(static_cast<float>(cols) - 1.5f, x));
            y = std::max(0.5f, std::min(static_cast<float>(rows) - 1.5f, y));

            if (advectDensity) {
                newGrid[i][j].density = interpolate(x, y, 2);
            }
            if (advectU) {
                newGrid[i][j].U = interpolate(x, y, 0);
            }
            if (advectV) {
                newGrid[i][j].V = interpolate(x, y, 1);
            }
        }
    }
    grid = newGrid;
}


void PressureCorrection::solvePressureJacobi(int iterations) {
    /**
     * Solves the pressure Poisson equation using the Jacobi iterative method.
     * This method is serial and may be slower for large grids.
     *
     * @param iterations Number of Jacobi iterations to perform.
     */
    std::vector<std::vector<float>> newPressure(rows,
                                                std::vector<float>(cols, 0.0f));

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                float divergence = (grid[i][j + 1].u_star - grid[i][j - 1].u_star +
                                    grid[i + 1][j].v_star - grid[i - 1][j].v_star) /
                                   (2.0f * dx);

                float sumPressure = grid[i - 1][j].pressure * grid[i - 1][j].fluid +
                                    grid[i + 1][j].pressure * grid[i + 1][j].fluid +
                                    grid[i][j - 1].pressure * grid[i][j - 1].fluid +
                                    grid[i][j + 1].pressure * grid[i][j + 1].fluid;

                float sumFluid = grid[i - 1][j].fluid + grid[i + 1][j].fluid +
                                 grid[i][j - 1].fluid + grid[i][j + 1].fluid;

                newPressure[i][j] =
                    grid[i][j].fluid *
                    (sumPressure - divergence * dx * dx * grid[i][j].density) /
                    (sumFluid + 1e-8f);
            }
        }
        // Update pressures
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                grid[i][j].pressure = newPressure[i][j];
            }
        }
    }
}

void PressureCorrection::solvePressureJacobiParallel(int iterations) {
    /**
     * Solves the pressure Poisson equation using the Jacobi iterative method.
     * This method is parallelized using OpenMP for better performance on large grids.
     *
     * @param iterations Number of Jacobi iterations to perform.
     */
    std::vector<std::vector<float>> newPressure(rows,
                                                std::vector<float>(cols, 0.0f));

    for (int iter = 0; iter < iterations; ++iter) {
        // Parallelize the pressure computation loop
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                float divergence = (grid[i][j + 1].u_star - grid[i][j - 1].u_star +
                                    grid[i + 1][j].v_star - grid[i - 1][j].v_star) /
                                   (2.0f * dx);

                float sumPressure = grid[i - 1][j].pressure * grid[i - 1][j].fluid +
                                    grid[i + 1][j].pressure * grid[i + 1][j].fluid +
                                    grid[i][j - 1].pressure * grid[i][j - 1].fluid +
                                    grid[i][j + 1].pressure * grid[i][j + 1].fluid;

                float sumFluid = grid[i - 1][j].fluid + grid[i + 1][j].fluid +
                                 grid[i][j - 1].fluid + grid[i][j + 1].fluid;

                newPressure[i][j] =
                    grid[i][j].fluid *
                    (sumPressure - divergence * dx * dx * grid[i][j].density) /
                    (sumFluid + 1e-6f);
            }
        }

        // Parallelize the pressure update loop
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                grid[i][j].pressure = newPressure[i][j];
            }
        }
    }
}

void PressureCorrection::forceIncompressibility(int iterations) {
    /**
     * Adjusts the velocity fields to enforce incompressibility by correcting
     * the divergence using central differences.
     *
     * @param iterations Number of correction iterations to perform.
     */
    float h = dx;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                // Calculate the local divergence using central differences of the velocity components
                float divergence = (grid[i][j + 1].U - grid[i][j - 1].U + grid[i + 1][j].V -
                                    grid[i - 1][j].V) /
                                   (2.0f * h);

                // Adjust velocities to correct divergence
                grid[i][j - 1].U -= divergence * h / 2.0f;
                grid[i][j + 1].U += divergence * h / 2.0f;
                grid[i - 1][j].V -= divergence * h / 2.0f;
                grid[i + 1][j].V += divergence * h / 2.0f;
            }
        }
    }
}

bool PressureCorrection::isObstacle(int i, int j) {
    // Implement this based on how you're representing obstacles
    // For example:
    return !grid[i][j].fluid; // Assuming you have a 'fluid' flag in your Cell struct
}
