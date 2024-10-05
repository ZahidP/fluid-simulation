//
//  GridSolver.cpp
//  fluid-simulation
//
//

#include <fstream>
#include "GridSolver.hpp"
#include <string>
#include <sstream>
#include <sys/stat.h>  // For checking file existence
#include <iostream>
#include <iostream>
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()

float randomNumber() {
    // Seed the random number generator
    srand(time(0));
    
    // Generate a random number between -5 and 5
    int random_number = rand() % 21 - 10;
    
    // Print the random number
    return random_number;
}

// Function to check if a file exists
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Function to generate a unique filename
std::string generateUniqueFilename(const std::string& baseFilename) {
    std::string filename = baseFilename;
    std::string extension = ".csv";
    int fileIndex = 1;

    // Check if the file exists and keep incrementing the number suffix
    while (fileExists(filename + extension)) {
        std::ostringstream newFilename;
        newFilename << baseFilename << "_" << fileIndex;
        filename = newFilename.str();
        fileIndex++;
    }

    return filename + extension;
}

GridSolver::GridSolver(int numRows, int numCols, double timeStep, double cellSize)
    : rows(numRows), cols(numCols), dt(timeStep), dx(cellSize) {
    printf("initializing to %d, %d \n", static_cast<int>(rows), static_cast<int>(cols));
    grid.resize(rows, std::vector<Cell>(cols, {0.0, 0.0, 0.05, 0, 1}));
}


void GridSolver::setObstacle(int cx, int cy, int R) {
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

void GridSolver::pumpWater(
    std::vector<PumpData> pumpData
) {
    //printf("pumpWater: %d\n", pumpStartRows[0]);
    for (auto& pump : pumpData) {
        for (int i = 0; i < pump.pumpHeight; ++i) {
            int startRow = pump.startLocation;
            grid[startRow + i][1].U = pump.U;
            grid[startRow + i][1].V = pump.V + randomNumber();
            grid[startRow + i][1].density = pump.density;
            // this is due to boundary conditions in pressure_correction (fix)
            grid[startRow + i][2].U = pump.U;
            grid[startRow + i][2].V = pump.V + randomNumber();
            grid[startRow + i][2].density = pump.density;
            
            grid[startRow + i][3].U = pump.U;
            grid[startRow + i][3].V = pump.V + randomNumber();
            grid[startRow + i][3].density = pump.density;
        }
    }
}

void GridSolver::exportToCSV(const std::string& filename) {
    
    printf("Exporting to CSV");
    
    std::string updatedFileName = generateUniqueFilename(filename);
    
    std::ofstream file(updatedFileName);
    
    if (!file.is_open()) {
        printf("Failed to open file: %s\n", filename.c_str());
        return;
    }
    
    // Write header
    file << "iter,time,timestep,row,col,u,v,density,pressure,is_fluid\n";
    
    for (const auto& record : gridRecords) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                Cell cell = record.grid[row][col];
                file << record.iter << ","
                     << record.time << ","
                     << record.timestep << ","
                     << row << ","
                     << col << ","
                     << cell.U << ","
                     << cell.V << ","
                     << cell.density << ","
                     << cell.pressure << ","
                     << cell.fluid << "\n";
            }
        }
    }
    file.close();
    std::cout << "File saved as " << updatedFileName << std::endl;
    printf("Export complete.\n");
}

void GridSolver::captureTimeStep(int sim_index, float time, float timestep) {
    std::vector<std::vector<Cell>> gridCopy;
    for (const auto& row : grid) {
        std::vector<Cell> newRow;
        for (const auto& cell : row) {
            newRow.push_back(cell);
        }
        gridCopy.push_back(newRow);
    }
    
    SimulationTimestep simulationStep;
    simulationStep.iter = sim_index;
    simulationStep.time = time;
    simulationStep.grid = grid;
    simulationStep.timestep = timestep;
    gridRecords.push_back(simulationStep);
}
