#include <vector>
#include <cmath>
#include <SFML/Graphics.hpp>
#include "FluidSimulator.hpp"
#include "GridSolver.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>
#include <unistd.h>
#include <string>
#include <filesystem>

std::string generateOutputPath(const std::string& filename) {
    const char* outputPath = std::getenv("OUTPUT_PATH");
    std::string basePath;

    if (outputPath != NULL) {
        basePath = std::string(outputPath);
    } else {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            basePath = std::string(cwd);
        } else {
            // Handle error
            std::cerr << "Error getting current working directory" << std::endl;
            return filename;  // Fallback to just the filename
        }
    }

    return basePath + "/" + filename;
}


// Main function
int main() {
    int rows = 120;
    int cols = 120;
    int cellSize = 3; // Increased for better visibility
    // int iterations = 100;
    double timestep = 0.02;
    std::vector<int> obstacleLocations = {63, 70};//{50, 60};//{60, 107};
    
    std::vector<PumpData> pumpData = {
        {140, 0, 0.8, 60, 25},
//        {90, -25, 0.8, 85, 8}
    };

    int max_threads = omp_get_max_threads();
    printf("Maximum number of threads available: %d\n", max_threads);
    
    #ifdef _OPENMP
        printf("OpenMP is enabled. Number of threads: %d\n", omp_get_max_threads());
    #else
        printf("OpenMP is not enabled.\n");
    #endif

    sf::RenderWindow window(sf::VideoMode(cols * cellSize, rows * cellSize), "Fluid Simulator");
    sf::RenderTexture renderTexture;
    
    if (!renderTexture.create(cols * cellSize, rows * cellSize)) {
        std::cerr << "Failed to create render texture" << std::endl;
        return -1;
    }
    std::string method = "mac";
    FluidSimulator simulator(
        rows, cols,
        obstacleLocations,
        pumpData,
        timestep, cellSize,
        &window, &renderTexture, method
    );
    
    int i = 0;
    
    GridSolver solver = GridSolver(rows, cols, timestep, cellSize);
    // Create a RenderTexture to render off-screen
    
    // Get the current working directory (C++17 std::filesystem)
    std::filesystem::path currentPath = std::filesystem::current_path();

    // Get the parent directory of the current directory
    std::filesystem::path parentDirectory = currentPath.parent_path();
    
    std::string baseFileName = "data/" + method;
    solver.exportToCSV(generateOutputPath(baseFileName));

    // Main loop
    while (window.isOpen() & (i < 500)) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        sf::RectangleShape cellShape;
        
        window.clear(sf::Color::White);
        renderTexture.clear();
        
        cellShape.setSize(sf::Vector2f(cellSize, cellSize));
        
        int steps = 20;
        if (method == "pressure_correction" || method == "mac") {
            steps = steps * 3;
        }
        solver = simulator.solveNextStep(steps, i < 5, timestep * i, i);
        i = i + 1;
        bool shouldRenderTexture;
        
        bool save_screenshots = false;
        
        if (i < 120 && i > 40 && save_screenshots) {
            shouldRenderTexture = true;
        } else {
            shouldRenderTexture = false;
        }
        
        simulator.render(shouldRenderTexture);
        window.display();
        // Draw on the render texture (off-screen)
        renderTexture.display(); 
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // Save a frame to PNG
        
        if (shouldRenderTexture && save_screenshots) {
            sf::Image screenshot = renderTexture.getTexture().copyToImage();

            // Create a filename with 'i' as a suffix
            std::string filename = "assets/screenshot_" + std::to_string(i) + ".png";

            std::string screenshot_path = generateOutputPath(filename);

            // Save the image in the parent directory
            if (screenshot.saveToFile(screenshot_path)) {
                std::cout << "Screenshot saved as " << screenshot_path << std::endl;
            } else {
                std::cerr << "Failed to save screenshot!" << std::endl;
            }
        }
    }
    // extension is added in exportToCSV
    baseFileName = "data/" + method;
    //solver.exportToCSV(generateOutputPath(baseFileName));

    return 0;
}
