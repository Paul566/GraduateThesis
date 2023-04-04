#ifndef GENERALSOLVER_GENERALSOLVER_H
#define GENERALSOLVER_GENERALSOLVER_H

#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include "TestReader.h"
#include "SimplexInBallTestReader.h"

class GeneralSolver {
    struct gridpoint_hash
    {
        size_t operator()(const std::tuple<std::vector<double>, double, double> &gridpoint) const {
            std::hash<double> double_hasher;
            size_t answer = 0;
            for (double coordinate : std::get<0>(gridpoint)) {
                answer ^= double_hasher(coordinate) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
            }
            return answer;
        }
    };

private:
    TestReader& test_reader_;
    double dimension;

    void UpdateTAndX();

public:
    double t;
    std::vector<double> x;
    std::unordered_set<std::tuple<std::vector<double>, double, double>, gridpoint_hash> grid_data; // vector of (p, s(p, A), s(p, B))
    GeneralSolver(const TestReader &test_reader, int dimension);
    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
