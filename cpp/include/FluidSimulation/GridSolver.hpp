#ifndef GRIDSOLVER_H
#define GRIDSOLVER_H

#include <vector>
#include <cmath>  // For std::sqrt

struct Cell {
    double U;
    double V;
    double density;
    double pressure;
    int fluid = 1;
};

struct PumpData {
    double U;
    double V;
    double density;
    double startLocation;
    double pumpHeight = 10;
};

class GridSolver {
protected:
    struct Cell {
      double U;
      double V;
      double density;
      double pressure;
      int fluid = 1;
      double u_star = 0;
      double v_star = 0;
    };
    struct SimulationTimestep {
        int iter;
        float time;
        std::vector<std::vector<Cell>> grid;
        int timestep;
    };
    size_t rows;
    size_t cols;
    double dt; // Time step
    int cellSize;
    double dx; // Grid cell size
    double interpolate(float x, float y, int field);

public:
    void setObstacle(int cx, int cy, int R);
    void pumpWater(std::vector<PumpData> pumpData);
    std::vector<std::vector<Cell>> grid;
    GridSolver(int numRows, int numCols, double timeStep, double cellSize);
    void captureTimeStep(int sim_index, float time, float timestep);
    std::vector<SimulationTimestep> gridRecords;
    void exportToCSV(const std::string& filename);
};

#endif // GRIDSOLVER_H
