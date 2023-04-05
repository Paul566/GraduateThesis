#ifndef GENERALSOLVER_GENERALSOLVER_H
#define GENERALSOLVER_GENERALSOLVER_H

#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <memory>
#include "TestReader.h"
#include "SimplexInBallTestReader.h"

class GeneralSolver {
    struct gridpoint_hash
    {
        size_t operator()(const std::tuple<std::vector<double>, double, double>& gridpoint) const {
            std::hash<double> double_hasher;
            size_t answer = 0;
            for (double coordinate : std::get<0>(gridpoint)) {
                answer ^= double_hasher(coordinate) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
            }
            return answer;
        }
    };

    struct Face {
        Face();
        ~Face();

        std::vector<std::tuple<std::vector<double>, double, double>> gridpoints;
        std::vector<std::shared_ptr<Face>> children;
        bool is_suspicious;
        bool is_root;
    };

private:
    TestReader& test_reader_;
    double dimension;
    Face root;

    void UpdateTAndX();
    void get_grid(const Face& vertex, std::unordered_set<std::tuple<std::vector<double>, double, double>, gridpoint_hash>& grid_data);

public:
    double t;
    std::vector<double> x;
    GeneralSolver(const TestReader &test_reader, int dimension);
    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
