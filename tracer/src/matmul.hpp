#pragma once

#include "Halide.h"
#include <stdio.h>

using namespace Halide;

namespace matmul {
    // based on MESA 4x4 inverse
    Func inverse(Func mat) {
        Func inv("inv");
        Var mi, mj;

        inv(mi, mj) = 0.0f;

        inv(0, 0) = mat(1, 1)  * mat(2, 2) * mat(3, 3) -
                 mat(1, 1)  * mat(3, 2) * mat(2, 3) -
                 mat(1, 2)  * mat(2, 1)  * mat(3, 3) +
                 mat(1, 2)  * mat(3, 1)  * mat(2, 3) +
                 mat(1, 3) * mat(2, 1)  * mat(3, 2) -
                 mat(1, 3) * mat(3, 1)  * mat(2, 2);

        inv(0, 1) = -mat(0, 1)  * mat(2, 2) * mat(3, 3) +
                    mat(0, 1)  * mat(3, 2) * mat(2, 3) +
                    mat(0, 2)  * mat(2, 1)  * mat(3, 3) -
                    mat(0, 2)  * mat(3, 1)  * mat(2, 3) -
                    mat(0, 3) * mat(2, 1)  * mat(3, 2) +
                    mat(0, 3) * mat(3, 1)  * mat(2, 2);

        inv(0, 2) = mat(0, 1)  * mat(1, 2) * mat(3, 3) -
                    mat(0, 1)  * mat(3, 2) * mat(1, 3) -
                    mat(0, 2)  * mat(1, 1) * mat(3, 3) +
                    mat(0, 2)  * mat(3, 1) * mat(1, 3) +
                    mat(0, 3) * mat(1, 1) * mat(3, 2) -
                    mat(0, 3) * mat(3, 1) * mat(1, 2);

        inv(0, 3) = -mat(0, 1)  * mat(1, 2) * mat(2, 3) +
                    mat(0, 1)  * mat(2, 2) * mat(1, 3) +
                    mat(0, 2)  * mat(1, 1) * mat(2, 3) -
                    mat(0, 2)  * mat(2, 1) * mat(1, 3) -
                    mat(0, 3) * mat(1, 1) * mat(2, 2) +
                    mat(0, 3) * mat(2, 1) * mat(1, 2);

        inv(1, 0) = -mat(1, 0)  * mat(2, 2) * mat(3, 3) +
                    mat(1, 0)  * mat(3, 2) * mat(2, 3) +
                    mat(1, 2)  * mat(2, 0) * mat(3, 3) -
                    mat(1, 2)  * mat(3, 0) * mat(2, 3) -
                    mat(1, 3) * mat(2, 0) * mat(3, 2) +
                    mat(1, 3) * mat(3, 0) * mat(2, 2);

        inv(1, 1) = mat(0, 0)  * mat(2, 2) * mat(3, 3) -
                    mat(0, 0)  * mat(3, 2) * mat(2, 3) -
                    mat(0, 2)  * mat(2, 0) * mat(3, 3) +
                    mat(0, 2)  * mat(3, 0) * mat(2, 3) +
                    mat(0, 3) * mat(2, 0) * mat(3, 2) -
                    mat(0, 3) * mat(3, 0) * mat(2, 2);

        inv(1, 2) = -mat(0, 0)  * mat(1, 2) * mat(3, 3) +
                    mat(0, 0)  * mat(3, 2) * mat(1, 3) +
                    mat(0, 2)  * mat(1, 0) * mat(3, 3) -
                    mat(0, 2)  * mat(3, 0) * mat(1, 3) -
                    mat(0, 3) * mat(1, 0) * mat(3, 2) +
                    mat(0, 3) * mat(3, 0) * mat(1, 2);

        inv(1, 3) = mat(0, 0)  * mat(1, 2) * mat(2, 3) -
                    mat(0, 0)  * mat(2, 2) * mat(1, 3) -
                    mat(0, 2)  * mat(1, 0) * mat(2, 3) +
                    mat(0, 2)  * mat(2, 0) * mat(1, 3) +
                    mat(0, 3) * mat(1, 0) * mat(2, 2) -
                    mat(0, 3) * mat(2, 0) * mat(1, 2);

        inv(2, 0) = mat(1, 0)  * mat(2, 1) * mat(3, 3) -
                    mat(1, 0)  * mat(3, 1) * mat(2, 3) -
                    mat(1, 1)  * mat(2, 0) * mat(3, 3) +
                    mat(1, 1)  * mat(3, 0) * mat(2, 3) +
                    mat(1, 3) * mat(2, 0) * mat(3, 1) -
                    mat(1, 3) * mat(3, 0) * mat(2, 1);

        inv(2, 1) = -mat(0, 0)  * mat(2, 1) * mat(3, 3) +
                    mat(0, 0)  * mat(3, 1) * mat(2, 3) +
                    mat(0, 1)  * mat(2, 0) * mat(3, 3) -
                    mat(0, 1)  * mat(3, 0) * mat(2, 3) -
                    mat(0, 3) * mat(2, 0) * mat(3, 1) +
                    mat(0, 3) * mat(3, 0) * mat(2, 1);

        inv(2, 2) = mat(0, 0)  * mat(1, 1) * mat(3, 3) -
                    mat(0, 0)  * mat(3, 1) * mat(1, 3) -
                    mat(0, 1)  * mat(1, 0) * mat(3, 3) +
                    mat(0, 1)  * mat(3, 0) * mat(1, 3) +
                    mat(0, 3) * mat(1, 0) * mat(3, 1) -
                    mat(0, 3) * mat(3, 0) * mat(1, 1);

        inv(2, 3) = -mat(0, 0)  * mat(1, 1) * mat(2, 3) +
                    mat(0, 0)  * mat(2, 1) * mat(1, 3) +
                    mat(0, 1)  * mat(1, 0) * mat(2, 3) -
                    mat(0, 1)  * mat(2, 0) * mat(1, 3) -
                    mat(0, 3) * mat(1, 0) * mat(2, 1) +
                    mat(0, 3) * mat(2, 0) * mat(1, 1);

        inv(3, 0) = -mat(1, 0) * mat(2, 1) * mat(3, 2) +
                    mat(1, 0) * mat(3, 1) * mat(2, 2) +
                    mat(1, 1) * mat(2, 0) * mat(3, 2) -
                    mat(1, 1) * mat(3, 0) * mat(2, 2) -
                    mat(1, 2) * mat(2, 0) * mat(3, 1) +
                    mat(1, 2) * mat(3, 0) * mat(2, 1);

        inv(3, 1) = mat(0, 0) * mat(2, 1) * mat(3, 2) -
                    mat(0, 0) * mat(3, 1) * mat(2, 2) -
                    mat(0, 1) * mat(2, 0) * mat(3, 2) +
                    mat(0, 1) * mat(3, 0) * mat(2, 2) +
                    mat(0, 2) * mat(2, 0) * mat(3, 1) -
                    mat(0, 2) * mat(3, 0) * mat(2, 1);

        inv(3, 2) = -mat(0, 0) * mat(1, 1) * mat(3, 2) +
                    mat(0, 0) * mat(3, 1) * mat(1, 2) +
                    mat(0, 1) * mat(1, 0) * mat(3, 2) -
                    mat(0, 1) * mat(3, 0) * mat(1, 2) -
                    mat(0, 2) * mat(1, 0) * mat(3, 1) +
                    mat(0, 2) * mat(3, 0) * mat(1, 1);

        inv(3, 3) = mat(0, 0) * mat(1, 1) * mat(2, 2) -
                    mat(0, 0) * mat(2, 1) * mat(1, 2) -
                    mat(0, 1) * mat(1, 0) * mat(2, 2) +
                    mat(0, 1) * mat(2, 0) * mat(1, 2) +
                    mat(0, 2) * mat(1, 0) * mat(2, 1) -
                    mat(0, 2) * mat(2, 0) * mat(1, 1);

        Expr det;
        det = 1.0f / (mat(0, 0) * inv(0, 0) +
                      mat(1, 0) * inv(0, 1) +
                      mat(2, 0) * inv(0, 2) +
                      mat(3, 0) * inv(0, 3));

        Func invOut("invOut");

        invOut(mi, mj) = inv(mi, mj) * det;

        //if (det == 0) return false;

        return invOut;
    }

    Func product(Func a, Func b) {
        Var mi, mj;

        Func prod;
        RDom k(0, 4);
        prod(mi, mj) = sum(a(mi, k) * b(k, mj));

        return prod;
    }
}
