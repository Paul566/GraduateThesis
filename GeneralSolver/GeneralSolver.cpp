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

        norm_a += std::max(test_reader_.SupportA(p1), test_reader_.SupportA(p2)) *
                  std::max(test_reader_.SupportA(p1), test_reader_.SupportA(p2));
        norm_b += std::max(test_reader_.SupportB(p1), test_reader_.SupportB(p2)) *
                  std::max(test_reader_.SupportB(p1), test_reader_.SupportB(p2));
    }
    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);

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
        UpdateGridData(root);
        UpdateTAndX();
        UpdateHBBHat();
        UpdateBasedPoints();
        UpdateBasedFaceDecompositions();
        UpdateXErrorVertices();
        UpdateGridGapData(root);

        faces = 0;
        non_sus_faces = 0;

        MarkSuspiciousFaces(root);
        RemoveSuspiciousFaces(root);
    }
    std::cout << "REPORT\n";

    std::cout << "non sus faces " << non_sus_faces << "\n";
    std::cout << "faces " << faces << "\n";
    std::cout << "delta " << delta << "\n";
    std::cout << "h_b_b_hat " << h_b_b_hat << "\n";

    /*std::cout << "some grid gap data: \n";
    int i = 0;
    for (auto [key, val] : grid_gaps_data) {
        i++;
        if (i > 10000)
            break;
        if (i % 1000 == 0)
            std::cout << std::get<1>(val) << " " << std::get<2>(val) << "\n";
    }*/

    std::cout << "END REPORT\n";
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

    if (dimension == 3) { // in case dim=3 use midpoint subdivision
        return Subdivide2DSimplex(simplex[0], simplex[1], simplex[2]);
    }

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
        h_b_b_hat = 1. / (1. - delta * delta / 2.) - 1.;
    } else {
        // TODO
        throw std::runtime_error("B is not a unit ball, this is not yet supported");
    }
}

void GeneralSolver::UpdateBasedPoints() {
    based_points_data.clear();

    std::vector<std::pair<double, std::vector<double>>> gaps_and_points;
    for (const auto &gridpoint: grid_data) {
        gaps_and_points.emplace_back(DotProduct(std::get<0>(*gridpoint), x) + t * std::get<2>(*gridpoint) - std::get<1>(*gridpoint), std::get<0>(*gridpoint));
    }

    std::sort(gaps_and_points.begin(), gaps_and_points.end());

    int index = 0;

    while (based_points_data.size() < dimension + 1) {
        bool new_point = true;
        for (auto known_based_point : based_points_data) { // check if this is really a new based point
            if (Dist(std::get<0>(known_based_point), std::get<1>(gaps_and_points[index])) < delta) {
                new_point = false;
                break;
            }
        }

        if (new_point) {
            based_points_data.emplace_back(std::get<1>(gaps_and_points[index]), DirectedErrorBound(std::get<1>(gaps_and_points[index])));
        }

        index++;

        if (index >= gaps_and_points.size()) {
            throw std::runtime_error("could not find dim+1 distinct based points");
        }
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

double GeneralSolver::Norm(const std::vector<double> &vec) {
    double norm = 0.;
    for (double coord: vec) {
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

void GeneralSolver::UpdateBasedFaceDecompositions() {
    based_face_decompositions.clear();

    for (int absent_i = 0; absent_i < dimension + 1; ++absent_i) {

        Eigen::MatrixXf P_transposed(dimension, dimension);
        for (int i = 0; i < dimension + 1; ++i) {
            for (int j = 0; j < dimension; ++j) {
                if (i > absent_i)
                    P_transposed(j, i - 1) = static_cast<float>(std::get<0>(based_points_data[i])[j]);
                if (i < absent_i)
                    P_transposed(j, i) = static_cast<float>(std::get<0>(based_points_data[i])[j]);
            }
        }

        based_face_decompositions.push_back(
                std::make_shared<Eigen::ColPivHouseholderQR<Eigen::MatrixXf>>(P_transposed.colPivHouseholderQr()));
    }
}

std::tuple<int, double, double>
GeneralSolver::GetGridGapData(const std::tuple<std::vector<double>, double, double> &q) {
    Eigen::VectorXf q_eigen(dimension);
    for (int i = 0; i < dimension; ++i)
        q_eigen[i] = static_cast<float>(std::get<0>(q)[i]);

    for (int absent_index = 0; absent_index < dimension + 1; ++absent_index) {
        Eigen::VectorXf coords = based_face_decompositions[absent_index]->solve(q_eigen);
        if (!std::all_of(coords.begin(), coords.end(), [](float c) { return c >= -1e-6; }))
            continue;

        // at this point we know that this absent_index is the right one

        Eigen::VectorXf x_eigen(dimension);
        Eigen::VectorXf x_error_vertex(dimension);
        for (int i = 0; i < dimension; ++i) {
            x_eigen[i] = static_cast<float>(x[i]);
            x_error_vertex[i] = static_cast<float>((*x_error_vertices[absent_index])[i]);
        }

        double l = (x_eigen - x_error_vertex).dot(q_eigen) + t * std::get<2>(q) - std::get<1>(q);
        double r = 0.;
        if (b_is_unit_ball)
            r = norm_a + x_error_vertex.norm();
        else
            r = t * norm_b + norm_a + x_error_vertex.norm();

        return {absent_index, l, r};
    }

    // if we are here, then the based simplex containing the gridpoint is degenerate

    return {0, -INFINITY, INFINITY};
}

void GeneralSolver::UpdateGridData(const std::shared_ptr<Face> &face) {
    if (face->is_root) {
        grid_data.clear();
    }

    if (face->children.empty()) {
        for (const auto &gridpoint: face->gridpoints) {
            grid_data.insert(gridpoint);
        }
    } else {
        for (const auto &child: face->children) {
            UpdateGridData(child);
        }
    }
}

void GeneralSolver::UpdateGridGapData(const std::shared_ptr<Face> &face) {
    if (face->is_root) {
        grid_gaps_data.clear();
    }

    if (face->children.empty()) {
        for (const auto &gridpoint: face->gridpoints) {
            if (grid_gaps_data.find(gridpoint) == grid_gaps_data.end()) {
                grid_gaps_data[gridpoint] = GetGridGapData(*gridpoint);
            }
        }
    } else {
        for (const auto &child: face->children) {
            UpdateGridGapData(child);
        }
    }
}

bool GeneralSolver::CheckIfFaceSuspicious(std::shared_ptr<Face> &face) {
    faces++;

    bool face_inside_one_based_face = true;
    for (int i = 1; i < dimension; ++i) {
        if (std::get<0>(grid_gaps_data[face->gridpoints[i]]) != std::get<0>(grid_gaps_data[face->gridpoints[0]])) {
            face_inside_one_based_face = false;
            break;
        }
    }

    for (int i = 0; i < dimension; ++i) {
        double max_edge = 0.; // max length of an edge in face incident to the i-th vertex
        for (int j = 0; j < dimension; ++j) {
            double edge_length = Dist(std::get<0>(*face->gridpoints[i]), std::get<0>(*face->gridpoints[j]));
            if (edge_length > max_edge) {
                max_edge = edge_length;
            }
        }

        double estimation = std::get<1>(grid_gaps_data[face->gridpoints[i]]) - max_edge * std::get<2>(grid_gaps_data[face->gridpoints[i]]);
        if ((face_inside_one_based_face) && (estimation > 0)) {
            non_sus_faces++;
            return false;
        }
        if ((! face_inside_one_based_face) && (estimation <= 0)) {
            return true;
        }
    }

    if (! face_inside_one_based_face) {
        non_sus_faces++;
        return false;
    }

    return true;
}

void GeneralSolver::UpdateXErrorVertices() {
    x_error_vertices.clear();
    for (int absent_i = 0; absent_i < dimension + 1; ++absent_i) {

        Eigen::MatrixXf P(dimension, dimension);
        for (int i = 0; i < dimension + 1; ++i) {
            for (int j = 0; j < dimension; ++j) {
                if (i > absent_i)
                    P(i - 1, j) = static_cast<float>(std::get<0>(based_points_data[i])[j]);
                if (i < absent_i)
                    P(i, j) = static_cast<float>(std::get<0>(based_points_data[i])[j]);
            }
        }

        Eigen::VectorXf d(dimension); // vector of bounds on (x - \widehat{x}, p)
        for (int i = 0; i < dimension + 1; ++i) {
            if (i > absent_i)
                d(i - 1) = static_cast<float>(DirectedErrorBound(std::get<0>(based_points_data[i])));
            if (i < absent_i)
                d(i) = static_cast<float>(DirectedErrorBound(std::get<0>(based_points_data[i])));
        }

        Eigen::VectorXf x_error_eigen = P.colPivHouseholderQr().solve(d);
        std::vector<double> x_error_vector;
        for (float c: x_error_eigen) {
            x_error_vector.push_back(c);
        }
        x_error_vertices.push_back(std::make_shared<std::vector<double>>(x_error_vector));
    }
}

void GeneralSolver::RemoveSuspiciousFaces(std::shared_ptr<Face> &face) {
    bool face_was_leaf = face->children.empty();
    std::vector<int> indices_to_delete;

    for (int i = static_cast<int>(face->children.size()) - 1; i >= 0; i--) {
        if (! face->children[i]->children.empty()) { // child was not a leaf
            RemoveSuspiciousFaces(face->children[i]);
        }

        if (! face->children[i]->is_suspicious) { // if child is a leaf and is suspicious, or it became a suspicious leaf
            indices_to_delete.push_back(i);
        }
    }

    for (int index : indices_to_delete) {
        face->children[index] = face->children.back();
        face->children.pop_back();
    }

    bool face_became_leaf = face->children.empty();

    if ((face_became_leaf) && (! face_was_leaf)) {
        face->is_suspicious = false;
    }
}

std::pair<double, std::vector<std::vector<std::vector<double>>>>
GeneralSolver::Subdivide2DSimplex(const std::vector<double>& v1, const std::vector<double>& v2,
                                  const std::vector<double>& v3) {
    // returns a pair (max diameter of the simplex in the subdivision, simplices of subdivision)
    // subdivides a triangle in the following way:
    //
    //       /\          /\
    //      /__\    ->  /\/\
    //

    if (dimension != 3) {
        throw std::runtime_error("this is not a three-dimensional problem, but trying to use Subdivide2DSimplex");
    }

    std::vector<double> c1({v2[0] + v3[0], v2[1] + v3[1], v2[2] + v3[2]});
    std::vector<double> c2({v1[0] + v3[0], v1[1] + v3[1], v1[2] + v3[2]});
    std::vector<double> c3({v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]});

    c1 = Normalize(c1);
    c2 = Normalize(c2);
    c3 = Normalize(c3);

    double max_edge = std::max(Dist(c1, c2), std::max(Dist(c1, c3), Dist(c2, c3)));

    std::vector<std::vector<double>> simplex1;
    std::vector<std::vector<double>> simplex2;
    std::vector<std::vector<double>> simplex3;
    std::vector<std::vector<double>> simplex4;

    simplex1.push_back(v1);
    simplex1.push_back(c2);
    simplex1.push_back(c3);

    simplex2.push_back(v2);
    simplex2.push_back(c1);
    simplex2.push_back(c3);

    simplex3.push_back(v3);
    simplex3.push_back(c1);
    simplex3.push_back(c2);

    simplex4.push_back(c1);
    simplex4.push_back(c2);
    simplex4.push_back(c3);

    std::vector<std::vector<std::vector<double>>> subdivision;

    subdivision.push_back(simplex1);
    subdivision.push_back(simplex2);
    subdivision.push_back(simplex3);
    subdivision.push_back(simplex4);

    return {max_edge, subdivision};
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
