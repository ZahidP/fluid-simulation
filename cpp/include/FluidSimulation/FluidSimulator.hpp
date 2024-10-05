#ifndef FLUIDSIMULATOR_HPP
#define FLUIDSIMULATOR_HPP

#include <vector>
#include <cmath>
#include <SFML/Graphics.hpp>
#include "PressureCorrection.hpp"
#include "PressureFree.hpp"
#include "MACMethod.hpp"

class FluidSimulator {
private:
    PressureCorrection grid;
    PressureFree pressureFreeGrid;
    MACMethod mac_grid;
    int rows, cols;
    double dx;
    sf::RenderWindow* window;
    sf::RenderTexture* texture;
    sf::RectangleShape cellShape;
    int cellSize;
    float pumpVelocity;
    float pumpHeight;
    std::string method;
    std::vector<int> obstacleLocations;
    std::vector<float> pumpUs;
    std::vector<float> pumpVs;
    std::vector<PumpData> pumpData;
    

public:
    FluidSimulator(int r, int c,
                   std::vector<int> obstacleLocations, 
                   std::vector<PumpData> pumpData,
                   double timeStep, int cellSize,
                   sf::RenderWindow* win, sf::RenderTexture* texture, std::string method);
    
    GridSolver solveNextStep(int iters = 10, bool showTimings = false, float timestamp = 0.0, int timestep = 0);
    void render(bool renderTexture);

    // Other methods...
};

#endif // FLUIDSIMULATOR_HPP
