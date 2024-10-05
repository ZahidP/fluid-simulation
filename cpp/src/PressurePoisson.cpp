//#include <vector>
//#include <stdexcept>
//#include <cmath>
//#include <iostream>
//#include <algorithm>
//#include <cstddef>
//#include <omp.h>
//
//
//#include "SIMPLE.hpp"
//#include <algorithm>
//
///**
//  * SIMPLE algorithm
//  *
// */
//
//SIMPLE::SIMPLE(int numRows, int numCols, double timeStep, double cellSize)
//    : rows(numRows), cols(numCols), dt(timeStep), dx(cellSize) {
//    grid.resize(rows, std::vector<Cell>(cols, {0.0, 0.0, 0.1, 0, 1}));
//}
//
//double SIMPLE::interpolate(float x, float y, int field) {
//    /**
//     * Assumes x and y are already in grid coordinates (i.e., integers represent grid points)
//     */
//    int i = static_cast<int>(x);
//    int j = static_cast<int>(y);
//    float fx = x - i;
//    float fy = y - j;
//
//    double v00 = (field == 0) ? grid[i][j].U : (field == 1) ? grid[i][j].V : grid[i][j].density;
//    double v10 = (field == 0) ? grid[i+1][j].U : (field == 1) ? grid[i+1][j].V : grid[i+1][j].density;
//    double v01 = (field == 0) ? grid[i][j+1].U : (field == 1) ? grid[i][j+1].V : grid[i][j+1].density;
//    double v11 = (field == 0) ? grid[i+1][j+1].U : (field == 1) ? grid[i+1][j+1].V : grid[i+1][j+1].density;
//
//    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11;
//}
//
//void SIMPLE::setObstacle(int row, int col, bool isObstacle) {
//    if (row >= rows || col >= cols) {
//        throw std::out_of_range("Cell coordinates out of range");
//    }
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            grid[row + i][col + j].fluid = isObstacle ? 0 : 1;
//        }
//    }
//}
//
//void SIMPLE::setCell(size_t row, size_t col, float U, float V, float density) {
//    if (row >= rows || col >= cols) {
//        throw std::out_of_range("Cell coordinates out of range");
//    }
//    grid[row][col] = {U, V, density, 0, 1};
//}
//
//SIMPLE::Cell SIMPLE::getCell(size_t row, size_t col) const {
//    if (row >= rows || col >= cols) {
//        throw std::out_of_range("Cell coordinates out of range");
//    }
//    return grid[row][col];
//}
//
//size_t SIMPLE::getRows() const {
//    return rows;
//}
//
//size_t SIMPLE::getCols() const {
//    return cols;
//}
//
//void SIMPLE::updateVelocityFromGravity(float gravity) {
//    gravity = gravity * dt;
//    float densityThreshold = 0.02;  // Adjust this value based on your simulation
//    for (size_t i = 1; i < rows - 1; ++i) {
//        for (size_t j = 1; j < cols - 1; ++j) {
//            if (grid[i][j].density > densityThreshold) {
//                grid[i][j].V += gravity * grid[i][j].density;  // Make gravity force proportional to density
//            } else {
//                grid[i][j].V = 0.0f;  // Reset velocity in near-empty cells
//            }
//        }
//    }
//}
//
//void SIMPLE::pumpWater(double velocity) {
//    grid[static_cast<int>(rows / 2)][0].U = velocity;
//    grid[static_cast<int>(rows / 2)][0].V = 0;
//    grid[static_cast<int>(rows / 2)][0].density = 1;
//    grid[static_cast<int>(rows / 2) + 1][0].U = velocity;
//    grid[static_cast<int>(rows / 2) + 1][0].V = 0;
//    grid[static_cast<int>(rows / 2) + 1][0].density = 1;
//    grid[static_cast<int>(rows / 2) + 2][0].U = velocity;
//    grid[static_cast<int>(rows / 2) + 2][0].V = 0;
//    grid[static_cast<int>(rows / 2) + 2][0].density = 1;
//}
//
//void SIMPLE::updateVelocitiesFromPressure() {
//    /**
//     * So there are two solver schemes. This is meant to be used for the explicit pressure correction.
//     */
//    for (size_t i = 1; i < rows - 1; ++i) {
//        for (size_t j = 1; j < cols - 1; ++j) {
//            // Update U (horizontal velocity)
//            if (j < cols - 2) {
//                float pressureGradient = (grid[i][j+1].pressure - grid[i][j].pressure) / dx;
//                grid[i][j].U -= pressureGradient * dt * grid[i][j].fluid;
//            }
//
//            // Update V (vertical velocity)
//            if (i < rows - 2) {
//                float pressureGradient = (grid[i+1][j].pressure - grid[i][j].pressure) / dx;
//                grid[i][j].V -= pressureGradient * dt * grid[i][j].fluid;
//            }
//        }
//    }
//}
//    
//void SIMPLE::semiLagrangianAdvection() {
//    std::vector<std::vector<Cell>> newGrid = grid;
//
//    for (size_t i = 1; i < rows - 1; ++i) {
//        for (size_t j = 1; j < cols - 1; ++j) {
//            // Trace particle back in time
//            float x = j - grid[i][j].U * dt / dx;
//            float y = i - grid[i][j].V * dt / dx;
//
//            // Clamp coordinates to grid boundaries
//            x = std::max(0.5, std::min<double>(static_cast<double>(cols) - 1.5, x));
//            y = std::max(0.5, std::min<double>(static_cast<double>(rows) - 1.5, y));
//
//            // Interpolate and update values
//            newGrid[i][j].U = interpolate(y, x, 0);
//            newGrid[i][j].V = interpolate(y, x, 1);
//            newGrid[i][j].density = interpolate(y, x, 2);
//        }
//    }
//
//    grid = newGrid;
//}
//
//
//void SIMPLE::advectSmoke() {
//    std::vector<std::vector<Cell>> newGrid = grid;
//    for (int i = 1; i < grid.size() - 1; ++i) {
//        for (int j = 1; j < grid[i].size() - 1; ++j) {
//            if (grid[i][j].fluid) {
//                // Calculate average velocities
//                float u = (grid[i][j].U + grid[i+1][j].U) * 0.5;
//                float v = (grid[i][j].V + grid[i][j+1].V) * 0.5;
//                
//                // Trace particle back in time
//                float x = j * dx + 0.5f * dx - dt * u;
//                float y = i * dx + 0.5f * dx - dt * v;
//                
//                // Clamp coordinates to grid boundaries
//                x = std::max(0.5, std::min<double>(static_cast<double>(grid[i].size()) - 1.5, x));
//                y = std::max(0.5, std::min<double>(static_cast<double>(grid.size()) - 1.5, y));
//                
//                // Update density using interpolation
//                newGrid[i][j].density = interpolate(y, x, 2);
//            }
//            std::cout << "Updated densities: " << newGrid[i][j].density << std::endl;
//        }
//    }
//    
//    // Update the grid with new values
//    for (int i = 1; i < grid.size() - 1; ++i) {
//        for (int j = 1; j < grid[i].size() - 1; ++j) {
//            grid[i][j].density = newGrid[i][j].density;
//        }
//    }
//}
//    
//void SIMPLE::solvePressureJacobi(int iterations) {
//    /**
//     The Jacobi solver updates a new array first, and then replaces the original array with the updated array. Unlike the Gauss-Seidel method
//      it does not update the array in place.
//     */
//    std::vector<std::vector<float>> newPressure(rows, std::vector<float>(cols, 0.0));
//
//    for (int iter = 0; iter < iterations; ++iter) {
//        // std::cout << "Jacobi Iter: " << iter << std::endl;
//        for (size_t i = 1; i < rows - 1; ++i) {
//            for (size_t j = 1; j < cols - 1; ++j) {
//                float divergence = (grid[i][j+1].U - grid[i][j-1].U +
//                                             grid[i+1][j].V - grid[i-1][j].V) / (2 * dx);
//                
//                float sumPressure = grid[i-1][j].pressure * grid[i-1][j].fluid +
//                grid[i+1][j].pressure * grid[i+1][j].fluid +
//                grid[i][j-1].pressure * grid[i][j-1].fluid +
//                grid[i][j+1].pressure * grid[i][j+1].fluid;
//                
//                float sumFluid =
//                grid[i-1][j].fluid +
//                grid[i+1][j].fluid +
//                grid[i][j-1].fluid +
//                grid[i][j+1].fluid;
//                
//                newPressure[i][j] = grid[i][j].fluid * (sumPressure - divergence * dx * dx * grid[i][j].density) /
//                                    (sumFluid + 1e-6f);  //
////                std::cout << "New pressure: " << newPressure[i][j]  << std::endl;
//            }
//        }
//        // Update pressures
//        // std::cout << "Update pressures: " << iter << std::endl;
//        for (size_t i = 1; i < rows - 1; ++i) {
//            for (size_t j = 1; j < cols - 1; ++j) {
//                grid[i][j].pressure = newPressure[i][j];
//            }
//        }
//    }
//};
//
//void SIMPLE::solvePressureJacobiParallel(int iterations) {
//        std::vector<std::vector<float>> newPressure(rows, std::vector<float>(cols, 0.0));
//
////            for (int iter = 0; iter < iterations; ++iter) {
////                #pragma omp parallel for collapse(2) schedule(dynamic)
////                for (size_t i = 1; i < rows - 1; ++i) {
////                    for (size_t j = 1; j < cols - 1; ++j) {
////                        float center = 1.0 - grid[i][j].fluid;
////                        float divergence = center * (grid[i][j+1].U - grid[i][j-1].U +
////                                                     grid[i+1][j].V - grid[i-1][j].V) / (2 * dx);
////
////                        float sumPressure =
////                            grid[i-1][j].pressure * (1.0 - grid[i-1][j].solid) +
////                            grid[i+1][j].pressure * (1.0 - grid[i+1][j].solid) +
////                            grid[i][j-1].pressure * (1.0 - grid[i][j-1].solid) +
////                            grid[i][j+1].pressure * (1.0 - grid[i][j+1].solid);
////
////                        float sumSolid =
////                            (1.0 - grid[i-1][j].solid) +
////                            (1.0 - grid[i+1][j].solid) +
////                            (1.0 - grid[i][j-1].solid) +
////                            (1.0 - grid[i][j+1].solid);
////
////                        newPressure[i][j] = center * (sumPressure - divergence * dx * dx) /
////                            (sumSolid + 1e-6);  // Add small value to avoid division by zero
////                    }
////                }
////
////                // Update pressures
////                #pragma omp parallel for collapse(2) schedule(dynamic)
////                for (size_t i = 1; i < rows - 1; ++i) {
////                    for (size_t j = 1; j < cols - 1; ++j) {
////                        grid[i][j].pressure = newPressure[i][j];
////                    }
////                }
////            }
//    }
//
//
//
//void SIMPLE::forceIncompressibility(int iterations) {
//    float h = dx;
//    for (int iter = 0; iter < iterations; ++iter) {
//        for (size_t i = 1; i < rows - 1; ++i) {
//            for (size_t j = 1; j < cols - 1; ++j) {
//                // calculate the local divergence using
//                // central differences of the velocity components
//                float div = (grid[i][j+1].U - grid[i][j-1].U +
//                             grid[i+1][j].V - grid[i-1][j].V) / (2.0 * h);
//                
//                grid[i][j-1].U -= div * h / 2.0;
//                grid[i][j+1].U += div * h / 2.0;
//                grid[i-1][j].V -= div * h / 2.0;
//                grid[i+1][j].V += div * h / 2.0;
//            }
//        }
//    }
//}
//
//
//void SIMPLE::applyPressure() {
//    for (size_t i = 1; i < rows - 1; ++i) {
//        for (size_t j = 1; j < cols - 1; ++j) {
//            float centerFluid = grid[i][j].fluid;
//            float leftFluid = grid[i][j-1].fluid;
//            float rightFluid = grid[i][j+1].fluid;
//            float topFluid = grid[i-1][j].fluid;
//            float bottomFluid = grid[i+1][j].fluid;
//
//            float densityFactor = 2.0f / (grid[i][j].density + grid[i][j-1].density);
//            grid[i][j].U -= centerFluid * leftFluid * (grid[i][j].pressure - grid[i][j-1].pressure) / dx * densityFactor;
//            
//            densityFactor = 2.0f / (grid[i][j+1].density + grid[i][j].density);
//            grid[i][j+1].U -= rightFluid * centerFluid * (grid[i][j+1].pressure - grid[i][j].pressure) / dx * densityFactor;
//            
//            densityFactor = 2.0f / (grid[i][j].density + grid[i-1][j].density);
//            grid[i][j].V -= centerFluid * topFluid * (grid[i][j].pressure - grid[i-1][j].pressure) / dx * densityFactor;
//            
//            densityFactor = 2.0f / (grid[i+1][j].density + grid[i][j].density);
//            grid[i+1][j].V -= bottomFluid * centerFluid * (grid[i+1][j].pressure - grid[i][j].pressure) / dx * densityFactor;
//        }
//    }
//}
