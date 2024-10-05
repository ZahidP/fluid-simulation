//
//  SmokeGrid.hpp
//  fluid-simulation
//
//

#ifndef PressureCorrection_hpp
#define PressureCorrection_hpp

#include <stdio.h>

#include <cstddef>
#include <vector>
#include "GridSolver.hpp"

class PressureCorrection: public GridSolver {
private:
    float interpolate(float x, float y, int field);

public:
  PressureCorrection(int numRows, int numCols, double timeStep,
                     double cellSize);
  void setCell(size_t row, size_t col, float U, float V, float density);
  Cell getCell(size_t row, size_t col) const;
  size_t getRows() const;
  size_t getCols() const;
  void updateVelocityFromGravity(float gravity);
  double computeDivergence();
  void forceIncompressibility(int iterations);
  void semiLagrangianAdvection(bool advectDensity, bool advectU, bool advectV);
  void calculateIntermediateVelocity(float gravity);
  void updateVelocitiesFromPressure();
  void applyPressure();
  void solvePressureJacobi(int iterations);
  void solvePressureJacobiParallel(int iterations);
  bool isObstacle(int i, int j);
  float sampleField(float x, float y, int field);
};

#endif /* PressureCorrection_hpp */
