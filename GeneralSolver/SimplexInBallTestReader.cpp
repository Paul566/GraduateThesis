#include <fstream>
#include "SimplexInBallTestReader.h"


SimplexInBallTestReader::SimplexInBallTestReader(const std::string& path, int dimension_) {
    dimension = dimension_;

    std::fstream test_file(path, std::ios_base::in);

    std::string line;
    getline(test_file, line);
    getline(test_file, line);

    vertices_.reserve(dimension + 1);
    for (int i = 0; i <= dimension; ++i) {
        std::vector<double> vertex(dimension);
        for (int j = 0; j < dimension; ++j) {
            test_file >> vertex[j];
            if (test_file.peek() == ',')
                test_file.ignore();
        }
        vertices_.push_back(vertex);
    }
}

double SimplexInBallTestReader::SupportA(const std::vector<double>& p) {
    return support_polyhedron(p, vertices_);
}

double SimplexInBallTestReader::SupportB(const std::vector<double>& p) {
    return 1.;
}
