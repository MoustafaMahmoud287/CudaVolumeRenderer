#ifndef VOLUME_RENDERER_H
#define VOLUME_RENDERER_H


#include <cuda_runtime.h>
#include <vector>
#include "VolumeTexture.cuh"
#include "CudaUtils.cuh"

class VolumeRenderer {
private:
    // the output image (R,G,B,A for every pixel)
    uchar4* d_output = nullptr;

    // screen Dimensions
    int width = 0;
    int height = 0;

    // CUDA Launch Parameters
    dim3 blockSize{16, 16, 1}; // for now 16x16
    dim3 gridSize;  // calculated based on width/height

public:
    VolumeRenderer() = default;
    ~VolumeRenderer() { cleanup(); }

    // if window resize or at the begining
    void resize(int w, int h);

    // 2. the main command
    void render(const VolumeTexture& volume);

    // get data back to CPU if needed
    std::vector<uchar4> getFrameBuffer();

    void cleanup();

    //static function so if we use it without creating object
   static void launchKernel(uchar4* d_output, cudaTextureObject_t texObj, int width, int height, float angleX, float angleY, float opc, float3 boxSize); // d_output is pointer on gpu memory
};

#endif // !VOLUME_RENDERER_H
