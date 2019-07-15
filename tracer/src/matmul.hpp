#pragma once

#include "Halide.h"
#include <stdio.h>
#include <exception>

using namespace Halide;

namespace matmul {
    // based on MESA 4x4 inverse
    inline Func inverse(Func mat) {
        Func inv("inv");
        Var mi, mj;

        inv(mi, mj) = 0.0f;

        //inv.bound(mi, 0, 4).unroll(mi);
        //inv.bound(mj, 0, 4).unroll(mj);

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

        //inv.compute_root();

        Expr det;
        det = 1.0f / (mat(0, 0) * inv(0, 0) +
                      mat(1, 0) * inv(0, 1) +
                      mat(2, 0) * inv(0, 2) +
                      mat(3, 0) * inv(0, 3));

        Func invOut("invOut");

        invOut(mi, mj) = inv(mi, mj) * det;
        //invOut.bound(mi, 0, 4).unroll(mi);
        //invOut.bound(mj, 0, 4).unroll(mj);

        //if (det == 0) return false;

        return invOut;
    }

    inline Func product(Func a, Func b, unsigned int t = 4) {
        Var mi, mj, mk;

        Func prod;
        RDom k(0, Expr(t));
        //prod(mi, mj) = sum(a(mi, k) * b(k, mj));
        prod(mi, mj) = 0.0f;
        prod(mi, mj) += a(mi, k) * b(k, mj);

        //prod.compute_root();

        return prod;
    }

    template <typename T>
    inline Func product(Func a, T b) {
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
    TupleVec(std::initializer_list<Expr> list) : _data(list) {
        if (list.size() != N) {
            throw std::out_of_range("template size does not match runtime tuple size!");
        }
    }
    template <class T, typename std::enable_if< std::is_convertible<T, Expr>::value, T >::type>
    TupleVec(std::initializer_list<T> list) : _data(list) {
        if (list.size() != N) {
            throw std::out_of_range("template size does not match runtime tuple size!");
        }
    }

    // Replace with default?
    TupleVec(TupleVec const& other) : _data(other._data) {}
    TupleVec(TupleVec&& other) : _data(other._data)  {}

    // Replace with default?
    TupleVec& operator=(TupleVec const& other) {
        _data = other._data;
    }
    TupleVec& operator=(TupleVec&& other) {
        _data = other._data;
    };

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

template <typename T, unsigned int N>
inline TupleVec<N> cast(TupleVec<N> const& v) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = cast<T>(v[i]);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator+(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] + rhs[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec+");
    }
}

template <unsigned int N>
inline TupleVec<N> operator+(Tuple const& lhs, TupleVec<N> const& rhs) {
    return rhs + lhs;
}

template <unsigned int N>
inline TupleVec<N> operator+(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs + rhs.get();
}

template <unsigned int N>
inline TupleVec<N> operator+(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] + rhs;
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator+(Expr const& lhs, TupleVec<N> const& rhs) {
    return rhs + lhs;
}

template <unsigned int N>
inline TupleVec<N> operator-(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] - rhs[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
inline TupleVec<N> operator-(Tuple const& lhs, TupleVec<N> const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs[i] - rhs.get()[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
inline TupleVec<N> operator-(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs - rhs.get();
}

template <unsigned int N>
inline TupleVec<N> operator-(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] - rhs;
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator-(Expr const& lhs, TupleVec<N> const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs - rhs.get()[i];
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator*(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] * rhs[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec+");
    }
}

template <unsigned int N>
inline TupleVec<N> operator*(Tuple const& lhs, TupleVec<N> const& rhs) {
    return rhs * lhs;
}

template <unsigned int N>
inline TupleVec<N> operator*(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs * rhs.get();
}

template <unsigned int N>
inline TupleVec<N> operator*(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] * rhs;
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator*(Expr const& lhs, TupleVec<N> const& rhs) {
    return rhs * lhs;
}

template <unsigned int N>
inline TupleVec<N> operator/(TupleVec<N> const& lhs, Tuple const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs.get()[i] / rhs[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
inline TupleVec<N> operator/(Tuple const& lhs, TupleVec<N> const& rhs) {
    if (rhs.size() == N) {
        std::vector<Expr> output(N);

        for (unsigned int i = 0; i < rhs.size(); i++) {
            output[i] = lhs[i] / rhs.get()[i];
        }

        return TupleVec<N>(Tuple(output));
    } else {
        throw std::out_of_range("out of bounds on TupleVec-");
    }
}

template <unsigned int N>
inline TupleVec<N> operator/(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return lhs / rhs.get();
}

template <unsigned int N>
inline TupleVec<N> operator/(TupleVec<N> const& lhs, Expr const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = lhs.get()[i] / rhs;
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> operator/(Expr const& lhs, TupleVec<N> const& rhs) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = rhs / lhs.get()[i];
    }

    return TupleVec<N>(Tuple(output));
}

inline Expr dot(Tuple const& lhs, Tuple const& rhs) {
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
inline Expr dot(TupleVec<N> const& lhs, TupleVec<N> const& rhs) {
    return dot(lhs.get(), rhs.get());
}

template <unsigned int N>
inline Expr dot(TupleVec<N> const& lhs, Tuple const& rhs) {
    return dot(lhs.get(), rhs);
}

template <unsigned int N>
inline Expr dot(Tuple const& lhs, TupleVec<N> const& rhs) {
    return dot(lhs, rhs.get());
}

inline Expr norm(Tuple const& vec) {
    Expr result = vec[0] * vec[0];
    for (unsigned int i = 1; i < vec.size(); i++) {
        result += vec[i] * vec[i];
    }

    return Halide::sqrt(result);
}

template <unsigned int N>
inline Expr norm(TupleVec<N> const& vec) {
    return norm(vec.get());
}

template <unsigned int N>
inline TupleVec<N> abs(TupleVec<N> const& vec) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = abs(vec[i]);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> min(TupleVec<N> const& vec, Tuple const& other) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = min(vec[i], other[i]);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> min(Tuple const& vec, TupleVec<N> const& other) {
    return min(TupleVec<N>(vec), other);
}

template <unsigned int N>
inline TupleVec<N> min(TupleVec<N> const& vec, TupleVec<N> const& other) {
    return min(vec, TupleVec<N>(other));
}

template <unsigned int N>
inline TupleVec<N> min(TupleVec<N> const& vec, Expr const& other) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = min(vec[i], other);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> min(Expr const& other, TupleVec<N> const& vec) {
    return min(vec, other);
}

template <unsigned int N>
inline TupleVec<N> max(TupleVec<N> const& vec, Tuple const& other) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = max(vec[i], other[i]);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> max(Tuple const& vec, TupleVec<N> const& other) {
    return max(TupleVec<N>(vec), other);
}

template <unsigned int N>
inline TupleVec<N> max(TupleVec<N> const& vec, TupleVec<N> const& other) {
    return max(vec, TupleVec<N>(other));
}

template <unsigned int N>
inline TupleVec<N> max(TupleVec<N> const& vec, Expr const& other) {
    std::vector<Expr> output(N);

    for (unsigned int i = 0; i < N; i++) {
        output[i] = max(vec[i], other);
    }

    return TupleVec<N>(Tuple(output));
}

template <unsigned int N>
inline TupleVec<N> max(Expr const& other, TupleVec<N> const& vec) {
    return max(vec, other);
}

template <unsigned int N>
inline TupleVec<N> build(std::function<Expr(unsigned int)> f) {
    std::vector<Expr> result(N);

    for (unsigned int i = 0; i < N; i++) {
        result[i] = f(i);
    }

    return TupleVec<N>(Tuple(result));
}

inline Tuple apply(Tuple const& vec, std::function<Expr(const Expr&)> f) {
    std::vector<Expr> result(vec.size());

    for (unsigned int i = 0; i < vec.size(); i++) {
        result[i] = f(vec[i]);
    }

    return Tuple(result);
}

template <unsigned int N>
inline TupleVec<N> apply(TupleVec<N> const& vec, std::function<Expr(const Expr&)> f) {
    return TupleVec<N>(apply(vec.get(), f));
}

inline Tuple apply(Tuple const& vec, std::function<Expr(const Expr&, unsigned int)> f) {
    std::vector<Expr> result(vec.size());

    for (unsigned int i = 0; i < vec.size(); i++) {
        result[i] = f(vec[i], i);
    }

    return Tuple(result);
}

template <unsigned int N>
inline TupleVec<N> apply(TupleVec<N> const& vec, std::function<Expr(const Expr&, unsigned int)> f) {
    return TupleVec<N>(apply(vec.get(), f));
}
