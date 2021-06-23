# 1 "/home/shirsch/bricklib/stencils/3axis.hip.cpp"
//
// Created by Samantha Hirsch on 6/22/21.
//

#include "stencils_hip.hip.h"
#include <iostream>
#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include "gpuvfold.h"

#define COEFF_SIZE 129

void d3pt7hip() {
    unsigned *grid_ptr;

    BrickInfo<3> bInfo = init_grid<3>(grid_ptr, {STRIDEB, STRIDEB, STRIDEB});
    unsigned *grid_dev;
    {
        unsigned size = (STRIDEB * STRIDEB * STRIDEB) * sizeof(unsigned);
        hipMalloc(&grid_dev, size);
        hipMemcpy(grid_dev, grid_ptr, size, hipMemcpyHostToDevice);
    }
    auto grid = (unsigned (*)[STRIDEB][STRIDEB]) grid_dev;

    BrickInfo<3> *binfo_dev = movBrickInfoDeep(bInfo, hipMemcpyHostToDevice);

    // Create out data
    unsigned data_size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
    bElem *in_ptr = randomArray({STRIDE, STRIDE, STRIDE});
    bElem *out_ptr = zeroArray({STRIDE, STRIDE, STRIDE});
    bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_ptr;
    bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_ptr;

    // Copy over the coefficient array
    bElem *coeff_dev;
    {
        unsigned size = COEFF_SIZE * sizeof(bElem);
        hipMalloc(&coeff_dev, size);
        hipMemcpy(coeff_dev, coeff, size, hipMemcpyHostToDevice);
    }

    // Copy over the data array
    bElem *in_dev, *out_dev;
    {
        hipMalloc(&in_dev, data_size);
        hipMalloc(&out_dev, data_size);
        hipMemcpy(in_dev, in_ptr, data_size, hipMemcpyHostToDevice);
        hipMemcpy(out_dev, out_ptr, data_size, hipMemcpyHostToDevice);
    }

    // Create the bricks
    auto bsize = cal_size<BDIM>::value;
    auto bstorage = BrickStorage::allocate(bInfo.nbricks, bsize * 2);
    Brick<Dim<BDIM>, Dim<VFOLD>> bIn(&bInfo, bstorage, 0);
    Brick<Dim<BDIM>, Dim<VFOLD>> bOut(&bInfo, bstorage, bsize);

    // Copy data to the bricks
    copyToBrick<3>({STRIDEG, STRIDEG, STRIDEG}, {PADDING, PADDING, PADDING}, {0, 0, 0}, in_ptr, grid_ptr, bIn);
    BrickStorage bstorage_dev = movBrickStorage(bstorage, hipMemcpyHostToDevice);

    auto arr_func = [&arr_in, &arr_out]() -> void {
        _TILEFOR arr_out[k][j][i] = coeff[5] * arr_in[k + 1][j][i] + coeff[6] * arr_in[k - 1][j][i] +
                                    coeff[3] * arr_in[k][j + 1][i] + coeff[4] * arr_in[k][j - 1][i] +
                                    coeff[1] * arr_in[k][j][i + 1] + coeff[2] * arr_in[k][j][i - 1] +
                                    coeff[0] * arr_in[k][j][i];
    };

    auto brick_func = [&grid, &binfo_dev, &bstorage_dev, &coeff_dev]() -> void {
        auto bSize = cal_size<BDIM>::value;
        Brick <Dim<BDIM>, Dim<VFOLD>> bIn(binfo_dev, bstorage_dev, 0);
        Brick <Dim<BDIM>, Dim<VFOLD>> bOut(binfo_dev, bstorage_dev, bSize);
        dim3 block(NB, NB, NB), thread(BDIM);
        // d3pt7_brick << < block, thread >> > (grid, bIn, bOut, coeff_dev);
    };

    auto cuarr_func = [&in_dev, &out_dev, &coeff_dev]() -> void {
        bElem(*arr_in)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in_dev;
        bElem(*arr_out)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out_dev;
        dim3 block(NB, NB, NB), thread(BDIM);
        // d3pt7_arr << < block, thread >> > (arr_in, arr_out, coeff_dev);
    };

    std::cout << "d3pt7" << std::endl;
    arr_func();
    std::cout << "Arr: " << hiptime_func(cuarr_func) << std::endl;
    std::cout << "Bricks: " << hiptime_func(brick_func) << std::endl;

    free(in_ptr);
    free(out_ptr);
    free(grid_ptr);
    free(bInfo.adj);
    hipFree(in_dev);
    hipFree(out_dev);
}
