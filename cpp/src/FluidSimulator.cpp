#include "FluidSimulator.hpp"
#include "GridSolver.hpp"
#include <SFML/Graphics.hpp>
#include <stdexcept>
#include <iostream>
#include <chrono>

FluidSimulator::FluidSimulator(
        int r, 
        int c,
        std::vector<int> obstacleLocations,
        std::vector<PumpData> pumpData,
        double timeStep, int cellSize, sf::RenderWindow* win, 
        sf::RenderTexture* texture,
        std::string method
    )
    : grid(r, c, timeStep, cellSize), rows(r), cols(c),
        obstacleLocations(obstacleLocations),
        dx(cellSize), window(win), texture(texture), pressureFreeGrid(r, c, timeStep, cellSize),
        mac_grid(r, c, timeStep, cellSize), method(method),
        pumpData(pumpData)
{
    pumpVelocity = 150;
    pumpHeight = 20;
    cellShape.setSize(sf::Vector2f(cellSize, cellSize));
    grid.pumpWater(pumpData);
    if (rows > 30) {
        if (obstacleLocations.size() > 0) {
            grid.setObstacle(obstacleLocations[0], 30, static_cast<int>(r / 25));
            grid.setObstacle(obstacleLocations[1], 70, static_cast<int>(r / 28));
            pressureFreeGrid.setObstacle(obstacleLocations[0], 25, static_cast<int>(r / 22));
            pressureFreeGrid.setObstacle(obstacleLocations[1], 50, static_cast<int>(r / 20));
            mac_grid.setObstacle(obstacleLocations[0], 30, static_cast<int>(r / 25));
            mac_grid.setObstacle(obstacleLocations[1], 70, static_cast<int>(r / 20));
        }
    }
    pressureFreeGrid.pumpWater(pumpData);
    mac_grid.pumpWater(pumpData);
}


GridSolver FluidSimulator::solveNextStep(int iters, bool showTimings, float timestamp, int timestep) {
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration;
    
    float gravity = 0;

    if (method == "pressure_correction") {
        start = std::chrono::high_resolution_clock::now();
        grid.pumpWater(pumpData);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        grid.calculateIntermediateVelocity(gravity);
        start = std::chrono::high_resolution_clock::now();
        grid.solvePressureJacobiParallel(iters);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "solvePressureJacobi: " << duration.count() << " ms\n";
        }

        start = std::chrono::high_resolution_clock::now();
        grid.updateVelocitiesFromPressure();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "updateVelocitiesFromPressure: " << duration.count() << " ms\n";
        }
        // advect the density of particles only
        start = std::chrono::high_resolution_clock::now();
        grid.semiLagrangianAdvection(true, true, true);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "semiLagrangianAdvection: " << duration.count() << " ms\n";
        }
        //grid.captureTimeStep(iters, timestamp);
        return grid;
    }
    else if (method == "pressure_free") {
        start = std::chrono::high_resolution_clock::now();
        pressureFreeGrid.pumpWater(pumpData);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        start = std::chrono::high_resolution_clock::now();
        pressureFreeGrid.solvePressureProjectionGaussSeidel(iters);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "solvePressureProjectionGaussSeidel: " << duration.count() << " ms\n";
        }

        start = std::chrono::high_resolution_clock::now();
        pressureFreeGrid.updateVelocityFromGravity(gravity * 0.1);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "updateVelocityFromGravity: " << duration.count() << " ms\n";
        }

        start = std::chrono::high_resolution_clock::now();
        pressureFreeGrid.semiLagrangianAdvection();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "semiLagrangianAdvection: " << duration.count() << " ms\n";
        }
        pressureFreeGrid.captureTimeStep(iters, timestamp, timestep);
        return pressureFreeGrid;
    } else if (method == "mac") {
        start = std::chrono::high_resolution_clock::now();
        mac_grid.pumpWater(pumpData);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "pumpWater: " << duration.count() << " ms\n";
        }
        start = std::chrono::high_resolution_clock::now();
        mac_grid.solvePressureJacobiParallel(iters);
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "solvePressureJacobi: " << duration.count() << " ms\n";
        }
        start = std::chrono::high_resolution_clock::now();
        mac_grid.updateVelocitiesFromPressure();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "updateVelocitiesFromPressure: " << duration.count() << " ms\n";
        }
        start = std::chrono::high_resolution_clock::now();
        mac_grid.semiLagrangianAdvection();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if (showTimings) {
            std::cout << "semiLagrangianAdvection: " << duration.count() << " ms\n";
        }
        mac_grid.captureTimeStep(iters, timestamp, timestep);
        return mac_grid;
    }
    else {
        throw std::out_of_range("Bad argument");
    }
}

void FluidSimulator::render(bool renderTexture) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            
            double normalizedPressure;
            
            // Color based on pressure
            if (method == "pressure_free") {
                normalizedPressure = pressureFreeGrid.grid[i][j].density / 1.0;
            } else if (method == "pressure_correction") {
                normalizedPressure = (grid.grid[i][j].density) / 1.0;
            } else if (method == "mac") {
                normalizedPressure = (mac_grid.grid[i][j].density) / 1.0;
            } else {
                throw;
            }
            sf::Color fillColor;
                        
            fillColor = sf::Color(
                static_cast<sf::Uint8>(normalizedPressure * 255),
                static_cast<sf::Uint8>(normalizedPressure * 255),
                static_cast<sf::Uint8>(normalizedPressure * 255)
            );
            if (method == "pressure_free") {
                if (pressureFreeGrid.grid[i][j].fluid == 0) {
                    fillColor = sf::Color(
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(200)
                    );
                }
            } else if (method == "pressure_correction") {
                if (grid.grid[i][j].fluid == 0) {
                    fillColor = sf::Color(
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(200)
                    );
                }
            } else if (method == "mac") {
                if (mac_grid.grid[i][j].fluid == 0) {
                    fillColor = sf::Color(
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(100),
                            static_cast<sf::Uint8>(200)
                    );
                }
            }

            sf::Color outlineColor(128, 128, 128); // Gray outline

            cellShape.setPosition(j * dx, i * dx);
            cellShape.setFillColor(fillColor);
            cellShape.setOutlineColor(outlineColor);
            cellShape.setOutlineThickness(0); // Set the thickness of the outline
            if (renderTexture) {
                texture->draw(cellShape);
            }
            window->draw(cellShape);
        }
    }
}
