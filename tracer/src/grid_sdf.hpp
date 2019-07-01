#pragma once

#include "Halide.h"

#include "matmul.hpp"

using namespace Halide;

inline void apply_auto_schedule(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(
            F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string, Internal::Function>::iterator fit;
    for (fit = flist.begin(); fit != flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        std::cout << "Warning: applying default schedule to " << f.name() << std::endl;
    }
    std::cout << std::endl;
}

class GridSDF {
  public:

    GridSDF(Halide::Func _buffer, TupleVec<3> _p0, TupleVec<3> _p1, int n0, int n1,
            int n2) :
        buffer(_buffer),
        p0(_p0),
        p1(_p1),
        n({n0, n1, n2}), nx(n0), ny(n1), nz(n2) { }

    Halide::Func buffer;
    TupleVec<3> p0;
    TupleVec<3> p1;
    TupleVec<3> n;

    int nx, ny, nz;
};

// For debugging analytical functions
inline GridSDF to_grid_sdf(std::function<Expr(TupleVec<3>)> sdf,
                    TupleVec<3> p0,
                    TupleVec<3> p1,
                    int nx, int ny, int nz) {
    Var dx("dx"), dy("dy"), dz("dz");
    Func field_func("field_func");
    field_func(dx, dy, dz) = sdf(TupleVec<3>({
        (dx / cast<float>(nx)) * (p1[0] - p0[0]) + p0[0],
        (dy / cast<float>(ny)) * (p1[1] - p0[1]) + p0[1],
        (dz / cast<float>(nz)) * (p1[2] - p0[2]) + p0[2]
    }));
    Halide::Buffer<float> buffer = field_func.realize(nx, ny, nz);

    return GridSDF(Func(buffer), p0, p1, nx, ny, nz);
}
