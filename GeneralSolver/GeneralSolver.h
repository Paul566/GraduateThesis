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
    struct Gridpoint {
        struct Hash {
            size_t operator()(const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &gridpoint) const;
        };
        struct Compare {
            size_t operator() (const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &a,
                               const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &b) const;
        };
    };

    struct Face {
        Face();

        ~Face();

        std::vector<std::shared_ptr<std::tuple<std::vector<double>, double, double>>> gridpoints;
        std::vector<std::shared_ptr<Face>> children;
        bool is_suspicious;
        bool is_root;
    };

private:
    TestReader &test_reader_;
    int dimension;
    std::shared_ptr<Face> root;
    int max_iterations;
    double tolerance = 1e-12;
    // delta is the length of the longest edge of the leaf simplex from the triangulation,
    // it bounds the fineness of the grid from above
    bool b_is_unit_ball;
    bool b_is_smooth;
    double delta;
    double h_b_b_hat{}; // Hausdorff distance between B and its approximation
    double x_error;
    double diameter_a; // an upper bound on the diameter of A
    double diameter_b; // an upper bound on the diameter of B

    void UpdateTAndX();

    double SubdivideFace(const std::shared_ptr<Face> &face);

    [[nodiscard]] std::vector<double> SphericalBarycenter(const std::vector<std::vector<double>> &vertices) const;

    // SubdivideSphericalSimplex also adds the new vertices to grid_data
    std::pair<double, std::vector<std::vector<std::vector<double>>>>
    SubdivideSphericalSimplex(std::vector<std::vector<double>> simplex);

    void SubdivideSuspiciousFaces(const std::shared_ptr<Face> &face);

    static std::vector<double> Normalize(std::vector<double> vec);

    static double DotProduct(std::vector<double> a, std::vector<double> b);

    static double Norm(const std::vector<double>& vec);

    static double Dist(std::vector<double> vec1, std::vector<double> vec2);

    void UpdateHBBHat();

    // DirectedErrorBound gets an upper bound for (x_current - x_precise, p)
    double DirectedErrorBound(const std::vector<double>& p) const;

    void UpdateBasedPoints();

    void MarkSuspiciousFaces(std::shared_ptr<Face> &face);

    // CheckIfFaceSuspicious also updates x_error if encounters possible errors greater than current x_error;
    bool CheckIfFaceSuspicious(std::shared_ptr<Face>& face);

    // BasedSimplexContainingFace returns dim based points such that they contain the face in their conic hull
    std::optional<std::vector<std::tuple<std::vector<double>, double>>> BasedSimplexContainingFace(const std::shared_ptr<Face> &face);

    // SimplexContainsPoint returns true if point is in the conic hull of the vectors from simplex
    bool SimplexContainsPoint(const std::vector<std::vector<double>> &simplex, const std::vector<double> &point) const;

    void RemoveSuspiciousFaces(const std::shared_ptr<Face> &face);

public:
    double t;
    std::vector<double> x;
    // grid_data consists of tuples (point p, s(p, A), s(p, B))
    std::unordered_set<std::shared_ptr<std::tuple<std::vector<double>, double, double>>, Gridpoint::Hash, Gridpoint::Compare> grid_data;
    // based_gridpoints_data consists of tuples (based point p, upper bound for (x_current - x_precise, p))
    std::vector<std::tuple<std::vector<double>, double>> based_points_data;

    GeneralSolver(const TestReader &test_reader, int dimension_, int max_iterations_ = 5, bool b_is_unit_ball_ = false, bool b_is_smooth_ = false);

    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
