#ifndef GENERALSOLVER_GENERALSOLVER_H
#define GENERALSOLVER_GENERALSOLVER_H

#include <vector>
#include <tuple>
#include <unordered_set>
#include "TestReader.h"
#include "SimplexInBallTestReader.h"

class GeneralSolver {

private:
    TestReader& test_reader_;
    double dimension;

    void UpdateTAndX();

public:
    double t;
    std::vector<double> x;
    std::vector<std::tuple<std::vector<double>, double, double>> grid_data; // vector of (p, s(p, A), s(p, B))
    GeneralSolver(const TestReader &test_reader, int dimension);
    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
