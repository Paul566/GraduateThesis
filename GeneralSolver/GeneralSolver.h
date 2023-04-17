#ifndef GENERALSOLVER_GENERALSOLVER_H
#define GENERALSOLVER_GENERALSOLVER_H

#include <vector>
#include <tuple>
#include <utility>
#include <unordered_set>
#include <memory>
#include "or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534/include/eigen3/Eigen/Dense"
#include "TestReader.h"
#include "SimplexInBallTestReader.h"

class GeneralSolver {
    struct Gridpoint {
        struct Hash {
            size_t operator()(const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &gridpoint) const;
        };

        struct Compare {
            size_t operator()(const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &a,
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
    double norm_a; // an upper bound on ||A||
    double norm_b; // an upper bound on ||B||
    // index i of based_face_decompositions is a decomposition for a matrix with columns that are equal
    // p_1, .. \widehat{p_i}, .. p_{dim + 1}
    std::vector<std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXf>>> based_face_decompositions;
    // index i of x_error_vertices is P^{-1} @ d, where P is matrix with rows that are equal
    // p_1, .. \widehat{p_i}, .. p_{dim + 1}, d is a vector of bounds (x_error, p_k) for the corresponding rows of P
    std::vector<std::shared_ptr<std::vector<double>>> x_error_vertices;
    // keys are gridpoints, values are pairs of (i, l, r), where
    // i is the index of a based point such that the rest based points contain the gridpoint in their conic hull
    // l = (x - P^{-1} @ d, q) + t * s(q, B) - s(q, A),
    // r = t * ||B|| + ||A|| + ||P^{-1} @ d|| if not b_is_unit_ball, else ||A|| + ||P^{-1} @ d||
    // P^{-1} @ d is the corresponding element of x_error_vertices
    std::unordered_map<std::shared_ptr<std::tuple<std::vector<double>, double, double>>, std::tuple<int, double, double>, Gridpoint::Hash, Gridpoint::Compare> grid_gaps_data;

    int faces;
    int non_sus_faces;

    void UpdateTAndX();

    double SubdivideFace(const std::shared_ptr<Face> &face);

    [[nodiscard]] std::vector<double> SphericalBarycenter(const std::vector<std::vector<double>> &vertices) const;

    std::pair<double, std::vector<std::vector<std::vector<double>>>>
    SubdivideSphericalSimplex(std::vector<std::vector<double>> simplex);

    void SubdivideSuspiciousFaces(const std::shared_ptr<Face> &face);

    static std::vector<double> Normalize(std::vector<double> vec);

    static double DotProduct(std::vector<double> a, std::vector<double> b);

    static double Norm(const std::vector<double> &vec);

    static double Dist(std::vector<double> vec1, std::vector<double> vec2);

    void UpdateHBBHat();

    // DirectedErrorBound gets an upper bound for (x_current - x_precise, p)
    double DirectedErrorBound(const std::vector<double> &p) const;

    void UpdateBasedPoints();

    void UpdateBasedFaceDecompositions();

    void UpdateXErrorVertices();

    std::tuple<int, double, double> GetGridGapData(const std::tuple<std::vector<double>, double, double>& q);

    void UpdateGridGapData(const std::shared_ptr<Face>& face);

    void UpdateGridData(const std::shared_ptr<Face>& face);

    void MarkSuspiciousFaces(std::shared_ptr<Face> &face);

    bool CheckIfFaceSuspicious(std::shared_ptr<Face> &face);

    void RemoveSuspiciousFaces(std::shared_ptr<Face> &face);

public:
    double t;
    std::vector<double> x;
    // grid_data consists of tuples (point p, s(p, A), s(p, B))
    std::unordered_set<std::shared_ptr<std::tuple<std::vector<double>, double, double>>, Gridpoint::Hash, Gridpoint::Compare> grid_data;
    // based_gridpoints_data consists of tuples (based point p, upper bound for (x_current - x_precise, p))
    std::vector<std::tuple<std::vector<double>, double>> based_points_data;

    GeneralSolver(const TestReader &test_reader, int dimension_, int max_iterations_ = 5, bool b_is_unit_ball_ = false,
                  bool b_is_smooth_ = false);

    void Solve();
};


#endif //GENERALSOLVER_GENERALSOLVER_H
