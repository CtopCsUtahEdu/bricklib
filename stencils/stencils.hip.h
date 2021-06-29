//
// Created by Samantha Hirsch on 6/21/21
//

#ifndef BRICK_STENCILS_HIP_H
#define BRICK_STENCILS_HIP_H

#include <brick-hip.h>
#include "stencils.h"

#define HIP_ITER 100

template<typename T>
double hiptime_func(T func) {
    func();
    hipEvent_t start_event, stop_event;
    hipDeviceSynchronize();
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    hipEventRecord(start_event);
    for (int i = 0; i < HIP_ITER; i++) {
        func();
    }
    hipEventRecord(stop_event);
    hipEventSynchronize(stop_event);
    hipEventSynchronize(start_event);
    float elapsed;
    hipEventElapsedTime(&elapsed, start_event, stop_event);
    const unsigned ms_per_s = 1000;
    return elapsed / HIP_ITER / ms_per_s;
}

void d3pt7hip();
void d3condhip();

#endif // BRICK_STENCILS_HIP_H
