#include <cmath>
#include "TestReader.h"


double TestReader::support_polyhedron(const std::vector<double>& p, const std::vector<std::vector<double>>& vertices) const {
    double ans = - std::numeric_limits<double>::infinity();

    for (auto vertex : vertices) {
        double dot_product = 0.;
        for (int i = 0; i < this->dimension; ++i)
            dot_product += vertex[i] * p[i];
        if (dot_product > ans)
            ans = dot_product;
    }
    return ans;
}

TestReader::~TestReader() = default;

