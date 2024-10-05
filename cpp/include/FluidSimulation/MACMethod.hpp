#ifndef Grid_hpp
#define Grid_hpp

#include <vector>
#include <cstddef>
#include "GridSolver.hpp"

class MACMethod: public GridSolver {
private:
    double interpolate(float x, float y, int field);

public:
    MACMethod(int numRows, int numCols, double timeStep, double cellSize);
    void setCell(size_t row, size_t col, float U, float V, float density);
    Cell getCell(size_t row, size_t col) const;
    void updateVelocityFromGravity(float gravity);
    double computeDivergence();
    void forceIncompressibility(int iterations);
    void semiLagrangianAdvection();
    void solvePressureJacobi(int iterations);
    void applyPressure();
    void solvePressureJacobiParallel(int iterations);
    void updateVelocitiesFromPressure();
    void advectSmoke();
    float sampleField(float x, float y, int field);
};

#endif // GRID_H
