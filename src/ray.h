#pragma once
#include "vec3.h"

struct Ray {
    vec3 origin;
    vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const vec3& origin, const vec3& direction)
        : origin(origin), direction(direction.normalize()) {}

    __host__ __device__ vec3 at(float t) const {
        return origin + direction * t;
    }
};