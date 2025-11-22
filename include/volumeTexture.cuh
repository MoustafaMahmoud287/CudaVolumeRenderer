#ifndef VOLUME_TEXTURE_CUH
#define VOLUME_TEXTURE_CUH

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>

class VolumeTexture {
private:
    // the opaque CUDA storage container
    cudaArray_t contentArray = nullptr;

    // the handle the kernel uses to read the data
    cudaTextureObject_t textureObject = 0;

public:
    VolumeTexture() = default;
    ~VolumeTexture() { cleanup(); }

    // function to load Image data to the aligned texture
    void loadVolume(const std::vector<uint8_t>& hostData, int width, int height, int depth);

    // helper to clean up memory
    void cleanup();

    // Getter
    cudaTextureObject_t getTexture() const { return textureObject; }
};

#endif // !VOLUME_TEXTURE_CUH