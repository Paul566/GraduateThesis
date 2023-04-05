#include <tuple>
#include <utility>
#include <vector>
#include "GeneralSolver.h"
#include "or-tools_x86_64_Ubuntu-20.04_cpp_v9.6.2534/include/ortools/linear_solver/linear_solver.h"

GeneralSolver::GeneralSolver(const TestReader &test_reader, int dimension_)
        : test_reader_(const_cast<TestReader &>(test_reader)) {
    dimension = dimension_;
    t = 0.;
    x = std::vector<double> (static_cast<size_t>(dimension), 0.);

    std::vector<std::tuple<std::vector<double>, double, double>> plus_identity;
    std::vector<std::tuple<std::vector<double>, double, double>> minus_identity;
    for (int i = 0; i < dimension; ++i) {
        std::vector<double> p1(static_cast<size_t>(dimension), 0.);
        std::vector<double> p2(static_cast<size_t>(dimension), 0.);

        p1[i] = 1.;
        p2[i] = -1.;

        plus_identity.emplace_back(p1, test_reader_.SupportA(p1), test_reader_.SupportB(p1));
        minus_identity.emplace_back(p2, test_reader_.SupportA(p2), test_reader_.SupportB(p2));
    }

    root = Face();
    root.is_root = true;
    for (int i = 0; i < pow(2, dimension); ++i) {
        std::shared_ptr<Face> face(new Face());
        for (int j = 0; j < dimension; ++j) {
            if (i & (1 << j)) {
                face->gridpoints.push_back(plus_identity[j]);
            } else {
                face->gridpoints.push_back(minus_identity[j]);
            }
        }
        root.children.push_back(face);
    }

}

void GeneralSolver::Solve() {

    UpdateTAndX();
}

void GeneralSolver::UpdateTAndX() {
    std::unordered_set<std::tuple<std::vector<double>, double, double>, gridpoint_hash> grid_data;
    get_grid(root, grid_data); // unordered_set of tuples (p, suppA, suppB)

    std::unique_ptr<operations_research::MPSolver> solver(operations_research::MPSolver::CreateSolver("GLOP"));

    double inf = operations_research::MPSolver::infinity();

    operations_research::MPVariable *const t_to_optimize = solver->MakeNumVar(- inf, inf, "t");
    std::vector<operations_research::MPVariable *> x_to_optimize;
    for (int i = 0; i < this->dimension; ++i) {
        operations_research::MPVariable * x_i = solver->MakeNumVar(- inf, inf, "x");
        x_to_optimize.push_back(x_i);
    }

    for (const auto& gridpoint : grid_data) {
        operations_research::MPConstraint *const constraint = solver->MakeRowConstraint(std::get<1>(gridpoint), inf);
        constraint->SetCoefficient(t_to_optimize, std::get<2>(gridpoint));
        for (int i = 0; i < this->dimension; ++i)
            constraint->SetCoefficient(x_to_optimize[i], std::get<0>(gridpoint)[i]);
    }

    operations_research::MPObjective *const objective = solver->MutableObjective();
    objective->SetCoefficient(t_to_optimize, 1);
    objective->SetMinimization();

    solver->Solve();

    this->t = objective->Value();
    for (int i = 0; i < this->dimension; ++i)
        this->x[i] = x_to_optimize[i]->solution_value();
}

void GeneralSolver::get_grid(const GeneralSolver::Face& vertex,
                             std::unordered_set<std::tuple<std::vector<double>, double, double>, gridpoint_hash>& grid_data) {
    for (const auto& gridpoint : vertex.gridpoints) {
        grid_data.insert(gridpoint);
    }

    for (const auto child : vertex.children) {
        get_grid(*child, grid_data);
    }
}

GeneralSolver::Face::Face() {
    is_root = false;
    is_suspicious = true;
}

GeneralSolver::Face::~Face() = default;
