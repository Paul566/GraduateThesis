#include <memory>
#include <tuple>
#include <vector>
#include <stdexcept>
#include "GeneralSolver.h"
#include "or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534/include/ortools/linear_solver/linear_solver.h"

GeneralSolver::GeneralSolver(const TestReader &test_reader, int dimension_, int max_iterations_, bool b_is_unit_ball_)
        : test_reader_(const_cast<TestReader &>(test_reader)) {
    dimension = dimension_;
    max_iterations = max_iterations_;
    t = 0.;
    x = std::vector<double> (static_cast<size_t>(dimension), 0.);
    delta = sqrt(2.);
    b_is_unit_ball = b_is_unit_ball_;

    std::vector<std::shared_ptr<std::tuple<std::vector<double>, double, double>>> plus_identity;
    std::vector<std::shared_ptr<std::tuple<std::vector<double>, double, double>>> minus_identity;
    for (int i = 0; i < dimension; ++i) {
        std::vector<double> p1(static_cast<size_t>(dimension), 0.);
        std::vector<double> p2(static_cast<size_t>(dimension), 0.);

        p1[i] = 1.;
        p2[i] = -1.;

        auto gridpoint1_tuple = std::tuple<std::vector<double>, double, double>({p1, test_reader_.SupportA(p1), test_reader_.SupportB(p1)});
        auto gridpoint2_tuple = std::tuple<std::vector<double>, double, double>({p2, test_reader_.SupportA(p2), test_reader_.SupportB(p2)});

        auto gridpoint1_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(gridpoint1_tuple);
        auto gridpoint2_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(gridpoint2_tuple);

        plus_identity.push_back(gridpoint1_ptr);
        minus_identity.push_back(gridpoint2_ptr);
    }

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
    }
}

void GeneralSolver::UpdateTAndX() {
    std::unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("GLOP"));

    double inf = operations_research::MPSolver::infinity();

    operations_research::MPVariable *const t_to_optimize = solver->MakeNumVar(- inf, inf, "t");
    std::vector<operations_research::MPVariable *> x_to_optimize;
    for (int i = 0; i < this->dimension; ++i) {
        operations_research::MPVariable * x_i = solver->MakeNumVar(- inf, inf, "x");
        x_to_optimize.push_back(x_i);
    }

    for (const auto& gridpoint : grid_data) {
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

double GeneralSolver::SubdivideFace(const std::shared_ptr<Face>& face) {
    // returns max length of the edges int the subdivision
    // also updates grid_data

    std::vector<std::vector<double>> current_simplex;
    double ans = 0.;

    for (const auto& gridpoint : face->gridpoints) {
        current_simplex.push_back(std::get<0>(*gridpoint));
    }

    auto subdivision_result = SubdivideSphericalSimplex(current_simplex);
    auto new_simplices = subdivision_result.second;

    for (const auto& simplex : new_simplices) {
        std::shared_ptr<Face> child_face(new Face());
        for (const std::vector<double>& vertex : simplex) {
            std::tuple<std::vector<double>, double, double> new_gridpoint({vertex, test_reader_.SupportA(vertex), test_reader_.SupportB(vertex)});
            auto new_gridpoint_ptr = std::make_shared<std::tuple<std::vector<double>, double, double>>(new_gridpoint);
            child_face->gridpoints.push_back(new_gridpoint_ptr);
        }
        face->children.push_back(child_face);
    }

    return subdivision_result.first;
}

std::vector<double> GeneralSolver::SphericalBarycenter(const std::vector<std::vector<double>>& vertices) const {
    std::vector<double> ans(static_cast<size_t>(dimension), 0.);
    for (auto vertex : vertices) {
        for (int i = 0; i < dimension; ++i) {
            ans[i] += vertex[i];
        }
    }
    return Normalize(ans);
}

std::vector<double> GeneralSolver::Normalize(std::vector<double> vec) {
    double norm = 0;
    for (double coordinate : vec)
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

        std::tuple<std::vector<double>, double, double> barycenter_tuple(barycenter, test_reader_.SupportA(barycenter), test_reader_.SupportB(barycenter));
        grid_data.insert(std::make_shared<std::tuple<std::vector<double>, double, double>>(barycenter_tuple));

        return {dist(barycenter, simplex[0]), ans};
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

        for (auto subface : subsimplex_subdivision) {
            subface.push_back(barycenter);
            ans.push_back(subface);
            for (const auto& vertex : subface) {
                double diameter_candidate = dist(vertex, barycenter);
                if (diameter_candidate > max_diameter)
                    max_diameter = diameter_candidate;
            }
        }
    }

    return {max_diameter, ans};
}

void GeneralSolver::SubdivideSuspiciousFaces(const std::shared_ptr<Face>& face) {
    // also updates delta

    if (face->is_root) // happens one time at the beginning of dfs
        delta = 0;

    if ((face->is_root) || (!face->children.empty())) {
        for (const auto& child : face->children) {
            SubdivideSuspiciousFaces(child);
        }
    } else {
        double candidate_delta = SubdivideFace(face);
        if (candidate_delta > delta)
            delta = candidate_delta;
    }
}

double GeneralSolver::dist(std::vector<double> vec1, std::vector<double> vec2) {
    double ans = 0.;
    for (int i = 0; i < vec1.size(); ++i)
        ans += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    return sqrt(ans);
}

void GeneralSolver::UpdateHBBHat() {
    if (b_is_unit_ball) {
        h_b_b_hat = 4 * delta * delta;
    } else {
        //TODO
        throw std::runtime_error("B is not a unit ball, this is not yet supported");
    }
}

std::vector<std::vector<double>> GeneralSolver::ExtractBasedVectors() {
    //TODO
    return {};
}

GeneralSolver::Face::Face() {
    is_root = false;
    is_suspicious = true;
}

GeneralSolver::Face::~Face() = default;
