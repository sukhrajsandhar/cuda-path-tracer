#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "stb_image_write.h"


struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    int material; // 0=diffuse, 1=metal, 2=glass
    float fuzz;   // metal fuzziness
};


__device__ bool hitSphere(const Sphere& s, const Ray& r, float tMin, float tMax, float& t) {
    vec3 oc = r.origin - s.center;
    float a = r.direction.dot(r.direction);
    float b = 2.0f * oc.dot(r.direction);
    float c = oc.dot(oc) - s.radius * s.radius;
    float disc = b*b - 4*a*c;
    if (disc < 0) return false;
    float sqrtDisc = sqrtf(disc);
    float root = (-b - sqrtDisc) / (2*a);
    if (root < tMin || root > tMax) {
        root = (-b + sqrtDisc) / (2*a);
        if (root < tMin || root > tMax) return false;
    }
    t = root;
    return true;
}


__device__ vec3 randomInUnitSphere(curandState* state) {
    vec3 p;
    do {
        p = vec3(curand_uniform(state),
                 curand_uniform(state),
                 curand_uniform(state)) * 2.0f - vec3(1,1,1);
    } while (p.dot(p) >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - n * 2.0f * v.dot(n);
}


__device__ vec3 getColor(Ray r, Sphere* spheres, int numSpheres, curandState* state) {
    vec3 attenuation(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < 8; bounce++) {
        float closest = 1e20f;
        int hitIdx = -1;

        for (int i = 0; i < numSpheres; i++) {
            float t;
            if (hitSphere(spheres[i], r, 0.001f, closest, t)) {
                closest = t;
                hitIdx = i;
            }
        }

        if (hitIdx < 0) {
            float t = 0.5f * (r.direction.normalize().y + 1.0f);
            vec3 sky = vec3(1,1,1) * (1-t) + vec3(0.5f, 0.7f, 1.0f) * t;
            return attenuation * sky;
        }

        Sphere& s = spheres[hitIdx];
        vec3 hitPoint = r.at(closest);
        vec3 normal = (hitPoint - s.center).normalize();

        if (s.material == 0) {
            vec3 target = hitPoint + normal + randomInUnitSphere(state);
            attenuation = attenuation * s.color;
            r = Ray(hitPoint, target - hitPoint);
        } else if (s.material == 1) {
            vec3 reflected = reflect(r.direction.normalize(), normal);
            reflected = reflected + randomInUnitSphere(state) * s.fuzz;
            attenuation = attenuation * s.color;
            r = Ray(hitPoint, reflected);
        }
    }
    return vec3(0,0,0);
}


__global__ void initRand(curandState* states, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    curand_init(1984 + x + y * width, 0, 0, &states[x + y * width]);
}


__global__ void render(unsigned char* output, int width, int height,
                       int samples, Sphere* spheres, int numSpheres,
                       curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    curandState* state = &states[x + y * width];


    vec3 camPos(0, 1, 4);
    vec3 camTarget(0, 0, -1);
    vec3 camUp(0, 1, 0);
    float fov = 60.0f * 3.14159f / 180.0f;
    float halfW = tanf(fov / 2);
    float halfH = halfW * height / width;

    vec3 w = (camPos - camTarget).normalize();
    vec3 u = camUp.cross(w).normalize();
    vec3 v = w.cross(u);

    vec3 color(0, 0, 0);
    for (int s = 0; s < samples; s++) {
        float pu = (x + curand_uniform(state)) / width;
        float pv = (y + curand_uniform(state)) / height;
        vec3 dir = (u * (2*pu - 1) * halfW) +
                   (v * (1 - 2*pv) * halfH) - w;
        Ray ray(camPos, dir);
        color += getColor(ray, spheres, numSpheres, state);
    }
    color = color / (float)samples;

    int idx = (y * width + x) * 3;
    output[idx]   = (unsigned char)(sqrtf(color.x) * 255);
    output[idx+1] = (unsigned char)(sqrtf(color.y) * 255);
    output[idx+2] = (unsigned char)(sqrtf(color.z) * 255);
}


int main() {
    const int WIDTH   = 1200;
    const int HEIGHT  = 675;
    const int SAMPLES = 64;

    Sphere scene[] = {
        {vec3(0, -100.5f, -1), 100.0f, vec3(0.5f, 0.7f, 0.3f), 0, 0},
        {vec3(0, 0, -1.5f),   0.5f,   vec3(0.8f, 0.3f, 0.3f), 0, 0},
        {vec3(-1.1f, 0, -1.5f), 0.5f, vec3(0.8f, 0.8f, 0.8f), 1, 0.1f},
        {vec3(1.1f, 0, -1.5f),  0.5f, vec3(0.8f, 0.6f, 0.2f), 1, 0.3f},
        {vec3(0, 1.2f, -1.5f),  0.3f, vec3(0.3f, 0.5f, 0.9f), 0, 0},
    };

    int numSpheres = 5;
    int imageSize  = WIDTH * HEIGHT * 3;

    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(scene));
    cudaMemcpy(d_spheres, scene, sizeof(scene), cudaMemcpyHostToDevice);

    unsigned char* d_output;
    curandState*   d_states;
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_states, WIDTH * HEIGHT * sizeof(curandState));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    initRand<<<gridSize, blockSize>>>(d_states, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    printf("Rendering %dx%d with %d samples per pixel...\n", WIDTH, HEIGHT, SAMPLES);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    render<<<gridSize, blockSize>>>(d_output, WIDTH, HEIGHT,
                                    SAMPLES, d_spheres, numSpheres, d_states);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Render time: %.2f ms\n", ms);

    unsigned char* output = new unsigned char[imageSize];
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);
    stbi_write_png("../output/render.png", WIDTH, HEIGHT, 3, output, WIDTH * 3);
    printf("Saved output/render.png\n");

    cudaFree(d_output);
    cudaFree(d_states);
    cudaFree(d_spheres);
    delete[] output;

    return 0;
}

