//
//  SmokeGrid.cpp
//  fluid-simulation
//
//

#include "PressureFree.hpp"

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <algorithm>

PressureFree::PressureFree(int numRows, int numCols, double timeStep, double cellSize)
    : GridSolver(numRows, numCols, timeStep, cellSize) {
    printf("PressureFree: grid size is %zu x %zu\n", grid.size(), grid[0].size());
}


double PressureFree::interpolate(float x, float y, int field) {
    float h = cellSize;
//    int i = static_cast<int>(y / h);
//    int j = static_cast<int>(x / h);
//
    int i = static_cast<int>(x);
    int j = static_cast<int>(y);
//    float fx = (x - j * h) / h;
//    float fy = (y - i * h) / h;
    float fx = x - i;
    float fy = y - j;

    double v00 = (field == 0) ? grid[i][j].U : (field == 1) ? grid[i][j].V : grid[i][j].density;
    double v10 = (field == 0) ? grid[i+1][j].U : (field == 1) ? grid[i+1][j].V : grid[i+1][j].density;
    double v01 = (field == 0) ? grid[i][j+1].U : (field == 1) ? grid[i][j+1].V : grid[i][j+1].density;
    double v11 = (field == 0) ? grid[i+1][j+1].U : (field == 1) ? grid[i+1][j+1].V : grid[i+1][j+1].density;

    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11;
}

void PressureFree::setObstacle(int cx, int cy, int R) {
    int rows = static_cast<int>(grid.size());
    int cols = static_cast<int>(grid[0].size());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int dx = i - cx;
            int dy = j - cy;
            // Calculate distance from the center
            if (std::sqrt(dx * dx + dy * dy) <= R) {
                grid[i][j].fluid = 0;
            }
        }
    }
}

void PressureFree::setCell(size_t row, size_t col, float U, float V, float density) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    grid[row][col] = {U, V, density, 0, 1};
}

PressureFree::Cell PressureFree::getCell(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Cell coordinates out of range");
    }
    return grid[row][col];
}

void PressureFree::updateVelocityFromGravity(float gravity) {
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

void PressureFree::semiLagrangianAdvection() {
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
            newGrid[i][j].U = interpolate(y, x, 0);
            newGrid[i][j].V = interpolate(y, x, 1);
            newGrid[i][j].density = interpolate(y, x, 2);
            // std::cout << "Updated densities: " << newGrid[i][j].density << std::endl;
        }
    }

    grid = newGrid;
}

void PressureFree::solvePressureProjectionGaussSeidel(int iterations) {
    /**
     * This equation ensures that the velocity field remains divergence-free (incompressible) by solving for a pressure field that, when applied, will correct any divergence in the velocity field.
     * It's a second-order central difference approximation of the Laplacian in 2D.
     * In a standard discretization, you would see something like:
     * (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - 4p[i,j]) / (dx²)
     * This code effectively reorganizes this equation to solve for p[i,j]
     * Gauss-seidel is an iterative method. That can converge faster than the Jacobi method.
     *
     * NOTE: A non-zero divergence violates the incompressibility condition
     * So we keep repeatedly solving for the updated velocities and pressures until they converge to a steady state
     * where divergence = 0.
     */
    float scaling_factor = dx / dt;
        float two_dx = 2.0f * dx;
    for (int iter = 0; iter < iterations; ++iter) {
        //std::cout << "GS iter: " << iter << std::endl;
        for (size_t i = 1; i < rows - 1; ++i) {
            for (size_t j = 1; j < cols - 1; ++j) {
                float is_fluid = grid[i][j].fluid;
                float overrelax = 1.8;
                
                // this is a central difference approximation of the divergence
                // TODO: should this be / (2 * dx)
                float divergence = overrelax * is_fluid * (
                    (grid[i][j+1].U - grid[i][j-1].U) / two_dx +
                    (grid[i+1][j].V - grid[i-1][j].V) / two_dx
                );

                
                float fluid_up = grid[i-1][j].fluid;
                float fluid_down = grid[i+1][j].fluid;
                float fluid_left = grid[i][j-1].fluid;
                float fluid_right = grid[i][j+1].fluid;
                
                float n_fluid_neighbors = fluid_up + fluid_down + fluid_left + fluid_right;
                
                // this becomes pressure when multiplied by the scaling factor
                float pressure_correction = -divergence / (n_fluid_neighbors + 1e-6);
                
                float scaling_factor = dx / dt;
                
                grid[i][j].pressure = pressure_correction * scaling_factor;
                
                // basically we want the divergence to "flow out" into the neighboring cells velocities
                // but only if that cell is a fluid
                grid[i][j].U += fluid_right * pressure_correction;
                grid[i][j-1].U -= fluid_left * pressure_correction;
                grid[i][j].V += fluid_down * pressure_correction;
                grid[i-1][j].V -= fluid_up * pressure_correction;
            }
            //std::cout << "Updated velocites: " << grid[i][1].U << std::endl;
        }
    }
}
