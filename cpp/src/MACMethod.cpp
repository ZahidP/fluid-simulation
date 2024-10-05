#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <omp.h>


#include "MACMethod.hpp"
#include <algorithm>

/**
  * ok i think i am actually doing pressure correction in a different order, and the pressure correction code is doing the advect step twice
  *
 */

MACMethod::MACMethod(int numRows, int numCols, double timeStep, double cellSize)
    : GridSolver(numRows, numCols, timeStep, cellSize) {
}

double MACMethod::interpolate(float x, float y, int field) {
    /* Assumes x and y are already in grid coordinates (i.e., integers represent grid points) */
    
    // Check bounds
    if (y < 0 || y >= grid.size()-1 || x < 0 || x >= grid[0].size()-1) {
        throw std::out_of_range("Coordinates out of grid bounds");
    }

    int j = std::floor(x);  // column index
    int i = std::floor(y);  // row index
    float fx = x - j;
    float fy = y - i;

    double v00 = (field == 0) ? grid[i][j].U : (field == 1) ? grid[i][j].V : grid[i][j].density;
    double v10 = (field == 0) ? grid[i][j+1].U : (field == 1) ? grid[i][j+1].V : grid[i][j+1].density;
    double v01 = (field == 0) ? grid[i+1][j].U : (field == 1) ? grid[i+1][j].V : grid[i+1][j].density;
    double v11 = (field == 0) ? grid[i+1][j+1].U : (field == 1) ? grid[i+1][j+1].V : grid[i+1][j+1].density;

    double result = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11;

    return result;
}

void MACMethod::setCell(size_t row, size_t col, float U, float V, float density) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    grid[row][col] = {U, V, density, 0, 1};
}

MACMethod::Cell MACMethod::getCell(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    return grid[row][col];
}

void MACMethod::updateVelocityFromGravity(float gravity) {
    gravity = gravity * dt;
    float densityThreshold = 0.02;  // Adjust this value based on your simulation
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            if (grid[i][j].density > densityThreshold) {
                grid[i][j].V += gravity * grid[i][j].density;  // Make gravity force proportional to density
            } else {
                grid[i][j].V = 0.0f;  // Reset velocity in near-empty cells
            }
        }
    }
}    

void MACMethod::updateVelocitiesFromPressure() {
    /**
     * So there are two solver schemes. This is meant to be used for the explicit pressure correction.
     */
    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // Update U (horizontal velocity)
            if (j < cols - 2) {
                float nonFluidAdjustment = grid[i][j].fluid;// * grid[i][j+1].fluid;
                float pressureGradient = (grid[i][j+1].pressure - grid[i][j].pressure) / dx;
                grid[i][j].U -= pressureGradient * dt * nonFluidAdjustment / grid[i][j].density;
            }

            // Update V (vertical velocity)
            if (i < rows - 2) {
                float nonFluidAdjustment = grid[i][j].fluid;// * grid[i+1][j].fluid;
                float pressureGradient = (grid[i+1][j].pressure - grid[i][j].pressure) / dx;
                grid[i][j].V -= pressureGradient * dt * nonFluidAdjustment / grid[i][j].density;
            }
        }
    }
}
    
void MACMethod::semiLagrangianAdvection() {
    std::vector<std::vector<Cell>> newGrid = grid;

    for (size_t i = 1; i < rows - 1; ++i) {
        for (size_t j = 1; j < cols - 1; ++j) {
            // Trace particle back in time
            float x = j - grid[i][j].U * dt / dx;
            float y = i - grid[i][j].V * dt / dx;

            // Clamp coordinates to grid boundaries
            x = std::max(0.5, std::min<double>(static_cast<double>(cols) - 1.5, x));
            y = std::max(0.5, std::min<double>(static_cast<double>(rows) - 1.5, y));

            // Interpolate and update values
            newGrid[i][j].U = interpolate(x, y, 0);
            newGrid[i][j].V = interpolate(x, y, 1);
            newGrid[i][j].density = interpolate(x, y, 2);
        }
    }

    grid = newGrid;
}


void MACMethod::solvePressureJacobi(int iterations) {
    /**
     * Solves the pressure Poisson equation using the Jacobi iterative method.
     * Updates the pressure field to enforce incompressibility.
     */
    std::vector<std::vector<float>> newPressure(rows, std::vector<float>(cols, 0.0f));

    for (int iter = 0; iter < iterations; ++iter) {
        // Iterate over inner grid cells
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                // Compute divergence of velocity field
                float divergence = (grid[i][j + 1].U - grid[i][j - 1].U +
                                    grid[i + 1][j].V - grid[i - 1][j].V) / (2.0f * dx);
                
                // Coefficients for the Poisson equation
                float a = grid[i][j - 1].fluid;
                float b = grid[i][j + 1].fluid;
                float c = grid[i - 1][j].fluid;
                float d = grid[i + 1][j].fluid;
                
                float sumPressure = grid[i][j - 1].pressure * a +
                                     grid[i][j + 1].pressure * b +
                                     grid[i - 1][j].pressure * c +
                                     grid[i + 1][j].pressure * d;
                
                float sumFluid = a + b + c + d + 1e-6f; // Avoid division by zero
                
                // Jacobi update formula
                newPressure[i][j] = (sumPressure - divergence * dx * dx * grid[i][j].density) / sumFluid;
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


void MACMethod::solvePressureJacobiParallel(int iterations) {
    /**
     * Solves the pressure Poisson equation using the Jacobi iterative method with OpenMP parallelization.
     */
    std::vector<std::vector<float>> newPressure(rows, std::vector<float>(cols, 0.0f));
    
    // Optionally set the number of threads or allow OpenMP to decide
    // omp_set_num_threads(omp_get_max_threads());

    for (int iter = 0; iter < iterations; ++iter) {
        // Parallelize the pressure computation loop
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                // Compute divergence of velocity field
                float divergence = (grid[i][j + 1].U - grid[i][j - 1].U +
                                    grid[i + 1][j].V - grid[i - 1][j].V) / (2.0f * dx);
                
                // Coefficients for the Poisson equation
                float a = grid[i][j - 1].fluid;
                float b = grid[i][j + 1].fluid;
                float c = grid[i - 1][j].fluid;
                float d = grid[i + 1][j].fluid;
                
                float sumPressure = grid[i][j - 1].pressure * a +
                                     grid[i][j + 1].pressure * b +
                                     grid[i - 1][j].pressure * c +
                                     grid[i + 1][j].pressure * d;
                
                float sumFluid = a + b + c + d + 1e-6f; // Avoid division by zero
                
                // Jacobi update formula
                if (grid[i][j].fluid == false) {
                    // what should the pressure be for non-fluid cells?
                    // it would affect the pressure gradient.
                    newPressure[i][j] = 2;
                } else {
                    newPressure[i][j] = (sumPressure - divergence * dx * dx * grid[i][j].density) / sumFluid;
                }

            }
        }

        // Update pressures in parallel
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                grid[i][j].pressure = newPressure[i][j];
            }
        }
    }
}




void MACMethod::forceIncompressibility(int iterations) {
    float h = dx;
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                // calculate the local divergence using
                // central differences of the velocity components
                float div = (grid[i][j+1].U - grid[i][j-1].U +
                             grid[i+1][j].V - grid[i-1][j].V) / (2.0 * h);
                
                grid[i][j-1].U -= div * h / 2.0;
                grid[i][j+1].U += div * h / 2.0;
                grid[i-1][j].V -= div * h / 2.0;
                grid[i+1][j].V += div * h / 2.0;
            }
        }
    }
}
