//
//  SmokeGrid.hpp
//  fluid-simulation
//
//

#ifndef PressureFree_hpp
#define PressureFree_hpp

#include <stdio.h>

#include <vector>
#include <cstddef>
#include "GridSolver.hpp"

class PressureFree: public GridSolver {
private:
    double interpolate(float x, float y, int field);

public:
    PressureFree(int numRows, int numCols, double timeStep, double cellSize);
    void setCell(size_t row, size_t col, float U, float V, float density);
    Cell getCell(size_t row, size_t col) const;
    void updateVelocityFromGravity(float gravity);
    double computeDivergence();
    void forceIncompressibility(int iterations);
    void semiLagrangianAdvection();
    void solvePressureProjectionGaussSeidel(int iterations);
    void setObstacle(int cx, int cy, int R);
    float sampleField(float x, float y, int field);
};

#endif /* PressureFree_hpp */
