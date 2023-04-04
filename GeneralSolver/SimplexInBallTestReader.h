#ifndef GENERALSOLVER_SIMPLEXINBALLTESTREADER_H
#define GENERALSOLVER_SIMPLEXINBALLTESTREADER_H

#include "TestReader.h"


class SimplexInBallTestReader : public TestReader {
private:
    std::vector<std::vector<double>> vertices_;

public:
    SimplexInBallTestReader(const std::string& path, int dimension);
    double SupportA(const std::vector<double>& p) override;
    double SupportB(const std::vector<double>& p) override;
};


#endif //GENERALSOLVER_SIMPLEXINBALLTESTREADER_H
