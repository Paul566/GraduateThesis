#ifndef GENERALSOLVER_TESTREADER_H
#define GENERALSOLVER_TESTREADER_H

#include <string>
#include <vector>


class TestReader {

protected:
    double support_polyhedron(const std::vector<double>& p, const std::vector<std::vector<double>>& vertices) const;

public:
    int dimension{};

    ~TestReader();
    virtual double SupportA(const std::vector<double>& p) = 0;
    virtual double SupportB(const std::vector<double>& p) = 0;
};


#endif //GENERALSOLVER_TESTREADER_H
