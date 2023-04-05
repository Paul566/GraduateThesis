#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include "TestReader.h"
#include "SimplexInBallTestReader.h"
#include "GeneralSolver.h"

namespace fs = std::filesystem;


int main () {
    std::ofstream test_results;
    test_results.open("/home/paul/Documents/GraduateThesis/GeneralSolver/test-results/simplex-in-ball-2d");

    std::string path = "/home/paul/Documents/GraduateThesis/tests/2d/simplex-in-ball/";
    for (const auto & test_file: fs::directory_iterator(path)) {
        SimplexInBallTestReader test_reader = SimplexInBallTestReader(test_file.path(), 2);
        GeneralSolver solver_instance(test_reader, 2);

        std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now();
        solver_instance.Solve();
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        double time_in_seconds = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count()) / 1000000.;

        test_results << std::setprecision(12) << time_in_seconds << "," << std::abs(solver_instance.t - 1.) << "\n";
        std::cout << std::setprecision(12) << test_file << "\t" << time_in_seconds << "\t" << std::abs(solver_instance.t - 1.) << "\n";
    }

    test_results.close();
    return 0;
}
