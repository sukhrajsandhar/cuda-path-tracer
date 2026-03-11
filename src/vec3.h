#pragma once
#include <math.h>

struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ vec3 operator+(const vec3& v) const { return vec3(x+v.x, y+v.y, z+v.z); }
    __host__ __device__ vec3 operator-(const vec3& v) const { return vec3(x-v.x, y-v.y, z-v.z); }
    __host__ __device__ vec3 operator*(float t)        const { return vec3(x*t, y*t, z*t); }
    __host__ __device__ vec3 operator*(const vec3& v)  const { return vec3(x*v.x, y*v.y, z*v.z); }
    __host__ __device__ vec3 operator/(float t)        const { return vec3(x/t, y/t, z/t); }
    __host__ __device__ vec3& operator+=(const vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }

    __host__ __device__ float dot(const vec3& v)  const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ float length()            const { return sqrtf(x*x + y*y + z*z); }
    __host__ __device__ vec3  normalize()         const { float l = length(); return vec3(x/l, y/l, z/l); }
    __host__ __device__ vec3  cross(const vec3& v) const {
        return vec3(y*v.z - z*v.y,
                    z*v.x - x*v.z,
                    x*v.y - y*v.x);
    }
};

inline __host__ __device__ vec3 operator*(float t, const vec3& v) { return v * t; }