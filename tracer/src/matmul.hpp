#pragma once

#include "Halide.h"
#include <stdio.h>
#include <exception>

using namespace Halide;

namespace matmul {
    // based on MESA 4x4 inverse
    Func inverse(Func mat) {
        Func inv("inv");
        Var mi, mj;

        inv(mi, mj) = 0.0f;

        inv.bound(mi, 0, 4).unroll(mi);
        inv.bound(mj, 0, 4).unroll(mj);

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

        inv.compute_root();

        Expr det;
        det = 1.0f / (mat(0, 0) * inv(0, 0) +
                      mat(1, 0) * inv(0, 1) +
                      mat(2, 0) * inv(0, 2) +
                      mat(3, 0) * inv(0, 3));

        Func invOut("invOut");

        invOut(mi, mj) = inv(mi, mj) * det;
        invOut.bound(mi, 0, 4).unroll(mi);
        invOut.bound(mj, 0, 4).unroll(mj);

        //if (det == 0) return false;

        return invOut;
    }

    Func product(Func a, Func b, unsigned int t = 4) {
        Var mi, mj, mk;

        Func prod;
        RDom k(0, Expr(t));
        prod(mi, mj) = sum(a(mi, k) * b(k, mj));

        prod.compute_root();

        return prod;
    }

    template <typename T>
    Func product(Func a, T b) {
        Var mi, mj;

        Func prod;
        prod(mi, mj) = a(mi, mj) * b;

        prod.compute_inline();

        return prod;
    }
}

template <unsigned int N>
class TupleVec final {
    Tuple _data;

  public:

    TupleVec() {}
    TupleVec(Tuple const& value) : _data(value) {
        if (value.size() != N) {
            throw std::out_of_range("template size does not match runtime tuple size!");
        }
    }
    explicit TupleVec(Expr const& single_dim) {
        std::vector<Expr> data(N);
        for (unsigned int i = 0; i < N; i++) {
            data[i] = single_dim;
        }
        _data = Tuple(data);
    }

    TupleVec(TupleVec const&) = default;
    TupleVec(TupleVec&&) = default;

    TupleVec& operator=(TupleVec const&) = default;
    TupleVec& operator=(TupleVec&&) = default;

    Tuple const& get() const {
        return _data;
    };

    Expr operator[](unsigned int i) const {
        if (i < N) {
            return _data[i];
        } else {
            throw std::out_of_range("out of bounds on TupleVec[]");
        }
    }

    Expr& operator[](unsigned int i) {
        if (i < N) {
            return _data[i];
        } else {
            throw std::out_of_range("out of bounds on TupleVec[]");
        }
    }
};

template <unsigned int N>
TupleVec<N> operator+(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] + rhs[i];
        }

        return Tuple(output);
    } else {
        throw std::out_of_range("out of bounds on TupleVec+");
    }
}

template <unsigned int N>
TupleVec<N> operator+(Tuple const& lhs, TupleVec<N> const& rhs) {
    return rhs + lhs;
}

template <unsigned int N>
TupleVec<N> operator+(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs + rhs;
}

template <unsigned int N>
TupleVec<N> operator+(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] + rhs;
    }

    return Tuple(output);
}

template <unsigned int N>
TupleVec<N> operator+(Expr const& lhs, TupleVec<N> const& rhs) {
    return rhs + lhs;
}

template <unsigned int N>
TupleVec<N> operator-(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] - rhs[i];
        }

        return Tuple(output);
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
TupleVec<N> operator-(Tuple const& lhs, TupleVec<N> const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs[i] - rhs.get()[i];
        }

        return Tuple(output);
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
TupleVec<N> operator-(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs - rhs.get();
}

template <unsigned int N>
TupleVec<N> operator-(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] - rhs;
    }

    return Tuple(output);
}

template <unsigned int N>
TupleVec<N> operator-(Expr const& lhs, TupleVec<N> const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = rhs.get()[i] - lhs;
    }

    return Tuple(output);;
}

template <unsigned int N>
TupleVec<N> operator*(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] * rhs;
    }

    return Tuple(output);
}

template <unsigned int N>
TupleVec<N> operator*(Expr const& lhs, TupleVec<N> const& rhs) {
    return rhs * lhs;
}

template <unsigned int N>
TupleVec<N> operator/(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] / rhs;
    }

    return Tuple(output);
}

template <unsigned int N>
TupleVec<N> operator/(Expr const& lhs, TupleVec<N> const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = rhs / lhs.get()[i];
    }

    return Tuple(output);
}

Expr dot(Tuple const& lhs, Tuple const& rhs) {
    if (lhs.size() == rhs.size()) {
        Expr result = lhs[0] * rhs[0];
        for (unsigned int i = 1; i < lhs.size(); i++) {
            result += lhs[i] * rhs[i];
        }

        return result;
    } else {
        throw std::out_of_range("out of bounds on TupleVec dot");
    }
}

template <unsigned int N>
Expr dot(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return dot(lhs.get(), rhs.get());
}

template <unsigned int N>
Expr dot(TupleVec<N> const& lhs, Tuple const& rhs) {
    return dot(lhs.get(), rhs);
}

template <unsigned int N>
Expr dot(Tuple const& lhs, TupleVec<N> const& rhs) {
    return dot(lhs, rhs.get());
}

Expr norm(Tuple const& vec) {
    Expr result = vec[0] * vec[0];
    for (unsigned int i = 1; i < vec.size(); i++) {
        result += vec[i] * vec[i];
    }

    return Halide::sqrt(result);
}

template <unsigned int N>
Expr norm(TupleVec<N> const& vec) {
    return norm(vec.get());
}
