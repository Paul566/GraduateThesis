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
}

void GeneralSolver::Solve() {
    for (int i = 0; i < dimension; ++i) {
        std::vector<double> gridpoint1(static_cast<size_t>(dimension), 0.);
        gridpoint1[i] = 1.;
        std::vector<double> gridpoint2(static_cast<size_t>(dimension), 0.);
        gridpoint2[i] = -1.;
        grid_data.emplace_back(gridpoint1, test_reader_.SupportA(gridpoint1), test_reader_.SupportB(gridpoint1));
        grid_data.emplace_back(gridpoint2, test_reader_.SupportA(gridpoint1), test_reader_.SupportB(gridpoint1));
    }

    UpdateTAndX();
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

    for (auto const& [p, suppA, suppB] : grid_data) {
        operations_research::MPConstraint *const constraint = solver->MakeRowConstraint(suppA, inf);

        constraint->SetCoefficient(t_to_optimize, suppB);
        for (int i = 0; i < this->dimension; ++i)
            constraint->SetCoefficient(x_to_optimize[i], p[i]);
    }

    operations_research::MPObjective *const objective = solver->MutableObjective();
    objective->SetCoefficient(t_to_optimize, 1);
    objective->SetMinimization();

    solver->Solve();

    this->t = objective->Value();
    for (int i = 0; i < this->dimension; ++i)
        this->x[i] = x_to_optimize[i]->solution_value();
}
