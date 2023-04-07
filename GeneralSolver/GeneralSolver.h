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
    std::shared_ptr<Face> root;
    int max_iterations;
    // delta is the length of the longest edge of the leaf simplex from the triangulation,
    // it bounds the fineness of the grid from above
    double delta;

    void UpdateTAndX();

    void GetGrid(const std::shared_ptr<Face>& face, std::unordered_set<std::tuple<std::vector<double>, double, double>, gridpoint_hash>& grid_data);

    double SubdivideFace(const std::shared_ptr<Face>& face);
    [[nodiscard]] std::vector<double> SphericalBarycenter(const std::vector<std::vector<double>>& vertices) const;
    std::pair<double, std::vector<std::vector<std::vector<double>>>> SubdivideSphericalSimplex(std::vector<std::vector<double>> simplex);
    void SubdivideSuspiciousFaces(const std::shared_ptr<Face>& face);

    static std::vector<double> Normalize(std::vector<double> vec);
    static double dist(std::vector<double> vec1, std::vector<double> vec2);
    std::vector<std::vector<double>> ExtractBasedVectors();

public:
    double t;
    std::vector<double> x;
    GeneralSolver(const TestReader &test_reader, int dimension_, int max_iterations_=5);
    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
