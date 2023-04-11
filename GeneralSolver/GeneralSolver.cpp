#include <memory>
#include <tuple>
#include <vector>
#include <stdexcept>
#include "GeneralSolver.h"
#include "or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534/include/ortools/linear_solver/linear_solver.h"
#include "or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534/include/eigen3/Eigen/Dense"

GeneralSolver::GeneralSolver(const TestReader &test_reader, int dimension_, int max_iterations_, bool b_is_unit_ball_,
                             bool b_is_smooth_)
        : test_reader_(const_cast<TestReader &>(test_reader)) {
    dimension = dimension_;
    max_iterations = max_iterations_;
    t = 0.;
    x = std::vector<double>(static_cast<size_t>(dimension), 0.);
    delta = sqrt(2.);
    b_is_unit_ball = b_is_unit_ball_;
    b_is_smooth = b_is_smooth_;
    h_b_b_hat = INFINITY;
    x_error = INFINITY;

    std::vector<std::shared_ptr<std::tuple<std::vector<double>, double, double>>> plus_identity;
    std::vector<std::shared_ptr<std::tuple<std::vector<double>, double, double>>> minus_identity;
    for (int i = 0; i < dimension; ++i) {
        std::vector<double> p1(static_cast<size_t>(dimension), 0.);
        std::vector<double> p2(static_cast<size_t>(dimension), 0.);

        p1[i] = 1.;
        p2[i] = -1.;

        auto gridpoint1_tuple = std::tuple<std::vector<double>, double, double>(
                {p1, test_reader_.SupportA(p1), test_reader_.SupportB(p1)});
        auto gridpoint2_tuple = std::tuple<std::vector<double>, double, double>(
                {p2, test_reader_.SupportA(p2), test_reader_.SupportB(p2)});

        auto gridpoint1_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(gridpoint1_tuple);
        auto gridpoint2_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(gridpoint2_tuple);

        plus_identity.push_back(gridpoint1_ptr);
        minus_identity.push_back(gridpoint2_ptr);

        diameter_a += (test_reader_.SupportA(p1) - test_reader_.SupportA(p2)) * (test_reader_.SupportA(p1) - test_reader_.SupportA(p2));
        diameter_b += (test_reader_.SupportB(p1) - test_reader_.SupportB(p2)) * (test_reader_.SupportB(p1) - test_reader_.SupportB(p2));
    }
    diameter_a = sqrt(diameter_a);
    diameter_b = sqrt(diameter_b);

    root = std::make_shared<Face>();
    root->is_root = true;
    for (int i = 0; i < pow(2, dimension); ++i) {
        std::shared_ptr<Face> face(new Face());
        for (int j = 0; j < dimension; ++j) {
            if (i & (1 << j)) {
                face->gridpoints.push_back(plus_identity[j]);
                grid_data.insert(plus_identity[j]);
            } else {
                face->gridpoints.push_back(minus_identity[j]);
                grid_data.insert(minus_identity[j]);
            }
        }
        root->children.push_back(face);
    }
}

void GeneralSolver::Solve() {
    for (int i = 0; i < max_iterations; ++i) {
        SubdivideSuspiciousFaces(root);
        UpdateTAndX();
        UpdateHBBHat();
        UpdateBasedPoints();
        x_error = 0.;
        MarkSuspiciousFaces(root); // x_error gets updated here
    }
}

void GeneralSolver::UpdateTAndX() {
    std::unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("GLOP"));

    double inf = operations_research::MPSolver::infinity();

    operations_research::MPVariable *const t_to_optimize = solver->MakeNumVar(-inf, inf, "t");
    std::vector<operations_research::MPVariable *> x_to_optimize;
    for (int i = 0; i < this->dimension; ++i) {
        operations_research::MPVariable *x_i = solver->MakeNumVar(-inf, inf, "x");
        x_to_optimize.push_back(x_i);
    }

    for (const auto &gridpoint: grid_data) {
        operations_research::MPConstraint *const constraint = solver->MakeRowConstraint(std::get<1>(*gridpoint), inf);
        constraint->SetCoefficient(t_to_optimize, std::get<2>(*gridpoint));
        for (int i = 0; i < this->dimension; ++i)
            constraint->SetCoefficient(x_to_optimize[i], std::get<0>(*gridpoint)[i]);
    }

    operations_research::MPObjective *const objective = solver->MutableObjective();
    objective->SetCoefficient(t_to_optimize, 1);
    objective->SetMinimization();

    solver->Solve();

    this->t = objective->Value();
    for (int i = 0; i < this->dimension; ++i)
        this->x[i] = x_to_optimize[i]->solution_value();
}

double GeneralSolver::SubdivideFace(const std::shared_ptr<Face> &face) {
    // returns max length of the edges int the subdivision
    // also updates grid_data

    std::vector<std::vector<double>> current_simplex;

    for (const auto &gridpoint: face->gridpoints) {
        current_simplex.push_back(std::get<0>(*gridpoint));
    }

    auto subdivision_result = SubdivideSphericalSimplex(current_simplex);
    auto new_simplices = subdivision_result.second;

    for (const auto &simplex: new_simplices) {
        std::shared_ptr<Face> child_face(new Face());
        for (const std::vector<double> &vertex: simplex) {
            std::tuple<std::vector<double>, double, double> new_gridpoint(
                    {vertex, test_reader_.SupportA(vertex), test_reader_.SupportB(vertex)});
            auto new_gridpoint_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(new_gridpoint);
            child_face->gridpoints.push_back(new_gridpoint_ptr);
        }
        face->children.push_back(child_face);
    }

    return subdivision_result.first;
}

std::vector<double> GeneralSolver::SphericalBarycenter(const std::vector<std::vector<double>> &vertices) const {
    std::vector<double> ans(static_cast<size_t>(dimension), 0.);
    for (auto vertex: vertices) {
        for (int i = 0; i < dimension; ++i) {
            ans[i] += vertex[i];
        }
    }
    return Normalize(ans);
}

std::vector<double> GeneralSolver::Normalize(std::vector<double> vec) {
    double norm = 0;
    for (double coordinate: vec)
        norm += coordinate * coordinate;
    norm = sqrt(norm);

    std::vector<double> ans(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        ans[i] = vec[i] / norm;
    }

    return ans;
}

std::pair<double, std::vector<std::vector<std::vector<double>>>>
GeneralSolver::SubdivideSphericalSimplex(std::vector<std::vector<double>> simplex) {
    // returns a pair (max diameter of the simplex in the subdivision, simplices of subdivision)

    std::vector<double> barycenter = SphericalBarycenter(simplex);
    std::vector<std::vector<std::vector<double>>> ans;
    if (simplex.size() == 2) {
        std::vector<std::vector<double>> subsimplex1;
        std::vector<std::vector<double>> subsimplex2;
        subsimplex1.push_back(simplex[0]);
        subsimplex1.push_back(barycenter);
        subsimplex2.push_back(simplex[1]);
        subsimplex2.push_back(barycenter);
        ans.push_back(subsimplex1);
        ans.push_back(subsimplex2);

        std::tuple<std::vector<double>, double, double> barycenter_tuple(barycenter, test_reader_.SupportA(barycenter),
                                                                         test_reader_.SupportB(barycenter));
        grid_data.insert(std::make_shared<std::tuple<std::vector<double>, double, double>>(barycenter_tuple));

        return {Dist(barycenter, simplex[0]), ans};
    }

    double max_diameter = 0.;
    for (int i = 0; i < simplex.size(); ++i) {
        std::vector<std::vector<double>> subsimplex;
        for (int j = 0; j < simplex.size(); ++j) {
            if (j != i) {
                subsimplex.push_back(simplex[j]);
            }
        }

        auto subresult = SubdivideSphericalSimplex(subsimplex);
        auto subsimplex_subdivision = subresult.second;
        if (subresult.first > max_diameter)
            max_diameter = subresult.first;

        for (auto subface: subsimplex_subdivision) {
            subface.push_back(barycenter);
            ans.push_back(subface);
            for (const auto &vertex: subface) {
                double diameter_candidate = Dist(vertex, barycenter);
                if (diameter_candidate > max_diameter)
                    max_diameter = diameter_candidate;
            }
        }
    }

    return {max_diameter, ans};
}

void GeneralSolver::SubdivideSuspiciousFaces(const std::shared_ptr<Face> &face) {
    // also updates delta

    if (face->is_root) // happens one time at the beginning of dfs
        delta = 0;

    if ((face->is_root) || (!face->children.empty())) {
        for (const auto &child: face->children) {
            SubdivideSuspiciousFaces(child);
        }
    } else {
        double candidate_delta = SubdivideFace(face);
        if (candidate_delta > delta)
            delta = candidate_delta;
    }
}

double GeneralSolver::Dist(std::vector<double> vec1, std::vector<double> vec2) {
    double ans = 0.;
    for (int i = 0; i < vec1.size(); ++i)
        ans += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    return sqrt(ans);
}

void GeneralSolver::UpdateHBBHat() {
    if (b_is_unit_ball) {
        h_b_b_hat = 4 * delta * delta;
    } else {
        // TODO
        throw std::runtime_error("B is not a unit ball, this is not yet supported");
    }
}

void GeneralSolver::UpdateBasedPoints() {
    based_points_data.clear();

    for (const auto &gridpoint: grid_data) {
        if (DotProduct(std::get<0>(*gridpoint), x) + t * std::get<2>(*gridpoint) - std::get<1>(*gridpoint) <
            tolerance) {
            based_points_data.emplace_back(std::get<0>(*gridpoint), DirectedErrorBound(std::get<0>(*gridpoint)));
        }
    }

    if (based_points_data.size() != dimension + 1) {
        // TODO
        throw std::runtime_error("greater than dim+1 based vectors, implement this");
    }
}

double GeneralSolver::DotProduct(std::vector<double> a, std::vector<double> b) {
    double ans = 0.;
    for (int i = 0; i < a.size(); ++i) {
        ans += a[i] * b[i];
    }
    return ans;
}

double GeneralSolver::DirectedErrorBound(const std::vector<double> &p) const {
    if (b_is_smooth) {
        return t * h_b_b_hat;
    } else {
        // TODO
        throw std::runtime_error("the boundary of B is not smooth, this is not yet supported");
    }
}

std::optional<std::vector<std::tuple<std::vector<double>, double>>>
GeneralSolver::BasedSimplexContainingFace(const std::shared_ptr<GeneralSolver::Face> &face) {
    for (int i = 0; i < dimension + 1; ++i) {
        std::vector<std::vector<double>> current_based_simplex;
        for (int j = 0; j < dimension + 1; ++j) {
            if (i != j)
                current_based_simplex.push_back(std::get<0>(based_points_data[j]));
        }

        bool found_simplex = true;
        for (const auto& gridpoint : face->gridpoints) {
            if (!SimplexContainsPoint(current_based_simplex, std::get<0>(*gridpoint))) {
                found_simplex = false;
                break;
            }
        }

        if (found_simplex) {
            std::vector<std::tuple<std::vector<double>, double>> ans;
            for (int j = 0; j < dimension + 1; ++j) {
                if (i != j)
                    ans.push_back(based_points_data[j]);
            }
            return ans;
        }
    }
    return std::nullopt;
}

bool GeneralSolver::SimplexContainsPoint(const std::vector<std::vector<double>> &simplex, const std::vector<double> &point) const {
    Eigen::MatrixXf A(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            A(i, j) = static_cast<float>(simplex[j][i]);
        }
    }

    Eigen::VectorXf b(dimension);
    for (int i = 0; i < dimension; ++i) {
        b(i) = static_cast<float>(point[i]);
    }

    Eigen::VectorXf coords = A.colPivHouseholderQr().solve(b);
    return std::all_of(coords.begin(), coords.end(), [](float c) { return c >= 0.; });
}

bool GeneralSolver::CheckIfFaceSuspicious(std::shared_ptr<Face>& face) {
    auto based_simplex = BasedSimplexContainingFace(face);
    if (based_simplex == std::nullopt)
        return true;

    Eigen::MatrixXf A(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            A(i, j) = static_cast<float>(std::get<0>(based_simplex.value()[j])[i]);
        }
    }

    Eigen::VectorXf b(dimension);
    for (int i = 0; i < dimension; ++i) {
        b(i) = static_cast<float>(std::get<1>(based_simplex.value()[i]));
    }

    Eigen::VectorXf error_bound = A.colPivHouseholderQr().solve(b);
    if ((A * error_bound - b).norm() > 1e-6) { // the matrix is singular
        return true;
    }

    // update x_error:
    float error_bound_norm = error_bound.norm();
    if (error_bound_norm > x_error)
        x_error = error_bound_norm;

    for (const auto& vertex : face->gridpoints) {
        auto p = std::get<0>(*vertex);
        double supp_a = std::get<1>(*vertex);
        double supp_b = std::get<2>(*vertex);
        if (DotProduct(x, p) + t * supp_b - supp_a > delta * (diameter_a + t * diameter_b + Norm(x)) + error_bound_norm) {
            return false;
        }
        if (b_is_unit_ball) {
            if (DotProduct(x, p) + t * supp_b - supp_a > delta * diameter_a + error_bound_norm) {
                return false;
            }
        }
    }
    return true;
}

double GeneralSolver::Norm(const std::vector<double>& vec) {
    double norm = 0.;
    for (double coord : vec) {
        norm += coord * coord;
    }
    return sqrt(norm);
}

void GeneralSolver::MarkSuspiciousFaces(std::shared_ptr<Face> &face) {
    if ((face->is_root) || (!face->children.empty())) {
        for (auto &child: face->children) {
            MarkSuspiciousFaces(child);
        }
    } else {
        face->is_suspicious = CheckIfFaceSuspicious(face);
    }
}

GeneralSolver::Face::Face() {
    is_root = false;
    is_suspicious = true;
}

GeneralSolver::Face::~Face() = default;

size_t GeneralSolver::Gridpoint::Hash::operator()(
        const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &gridpoint) const {
    std::hash<double> double_hasher;
    size_t answer = 0;
    for (double coordinate: std::get<0>(*gridpoint)) {
        answer ^= double_hasher(coordinate) + 0x9e3779b9 + (answer << 6) + (answer >> 2);
    }
    return answer;
}

size_t
GeneralSolver::Gridpoint::Compare::operator()(const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &a,
                                              const std::shared_ptr<std::tuple<std::vector<double>, double, double>> &b) const {
    return std::get<0>(*a) == std::get<0>(*b);
}
