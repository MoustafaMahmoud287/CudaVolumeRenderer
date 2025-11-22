#include "volumeRenderer.cuh"
#include "ray.cuh"
#include "CudaUtils.cuh"

namespace {
__global__ void renderKernel(cudaTextureObject_t volumeTex, uchar4* d_output, int width, int height, float angleX, float angleY, float opc, float3 boxSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x >= width || y >= height) return;

    // 1. Normalize Screen Coordinates (-1.0 to +1.0)
    // We adjust 'u' by the window aspect ratio so the volume doesn't stretch when window resizes
    float u = ((x / (float)width) - 0.5f) * 2.0f;
    float v = ((y / (float)height) - 0.5f) * 2.0f;
    u *= (float)width / (float)height;

    // 2. Ray Setup (Camera at Z = -2.0 to see full rotation)
    Ray pixelRay(make_float3(u, v, -2.0f), make_float3(0.0f, 0.0f, 1.0f));

    float t = 0.0f;
    float dt = 1.0f / 256.0f;
    float3 accColor = make_float3(0.0f, 0.0f, 0.0f);
    float accAlpha = 0.0f;

    // Calculate center based on the actual box size
    float3 center = boxSize * 0.5f;

    // pre-calculate sines and cosines to save performance
    float cY = cosf(angleY); // (left/right)
    float sY = sinf(angleY);
    float cX = cosf(angleX); // (up/down)
    float sX = sinf(angleX);

    while (t < 4.0f && accAlpha < 0.99f) { // Increased max t to 4.0 for rotation space
        float3 rawPos = pixelRay.point_at(t);

        // --- 3D ROTATION MATH ---

        // 1. Use rawPos (centered at 0,0,0 relative to camera ray)
        float3 p = rawPos;
        float3 temp = p;

        // 2. Rotate around Y-Axis (Left/Right)
        p.x = temp.x * cY - temp.z * sY;
        p.z = temp.x * sY + temp.z * cY;

        // update temp for next rotation
        temp = p;

        // 3. Rotate around X-Axis (Up/Down)
        p.y = temp.y * cX - temp.z * sX;
        p.z = temp.y * sX + temp.z * cX;

        // 4. Shift to Volume Coordinates (0 to boxSize)
        float3 pos = p + center;
        // ------------------------

        // check bounds using boxSize
        float density = 0.0f;
        if (pos.x >= 0.0f && pos.x <= boxSize.x &&
            pos.y >= 0.0f && pos.y <= boxSize.y &&
            pos.z >= 0.0f && pos.z <= boxSize.z) {

            // normalize back to 0..1 for texture lookup
            density = tex3D<float>(volumeTex, pos.x / boxSize.x, pos.y / boxSize.y, pos.z / boxSize.z);
        }

        float4 pixColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        //  for human head

            if (density > 0.6f) { // bones
                pixColor = make_float4(1.0f, 1.0f, 1.0f, density * 0.9f);
            }
            else if (density > 0.3f) { // muscle
                pixColor = make_float4(0.8f, 0.2f, 0.2f, density * 0.2f);
            }
            else if (density > 0.15f) { // skin
                pixColor = make_float4(1.0f, 0.8f, 0.6f, density * 0.05f);
            }


        // for lobseter
        /*if (density > 0.5f) {
            pixColor = make_float4(1.0f, 0.2f, 0.1f, density * 0.9f);
        }
        else if (density > 0.25f) {
            pixColor = make_float4(0.9f, 0.8f, 0.8f, density * 0.2f);
        }
        else if (density > 0.15f) {
            pixColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }*/

        if (pixColor.w > 0.0 && density > opc) {
            float alpha = pixColor.w * 0.5f;
            float visibility = 1.0f - accAlpha;
            accColor.x += pixColor.x * alpha * visibility;
            accColor.y += pixColor.y * alpha * visibility;
            accColor.z += pixColor.z * alpha * visibility;
            accAlpha += alpha * visibility;
        }
        t += dt;
    }

    d_output[index] = make_uchar4(
        (unsigned char)(fminf(accColor.x, 1.0f) * 255.0f),
        (unsigned char)(fminf(accColor.y, 1.0f) * 255.0f),
        (unsigned char)(fminf(accColor.z, 1.0f) * 255.0f),
        255
        );
}
}

void VolumeRenderer::resize(int w, int h) {
    if (w == width && h == height) return; // if there is no change

    width = w; // update the width
    height = h; // update the height

    if (d_output != nullptr) { // delete the old Image
        cudaFree(d_output);
        d_output = nullptr;
    }

    CHECK_CUDA(cudaMalloc((void**)&d_output, sizeof(uchar4) * width * height)); // malloc for the new size

    gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // the new gridsize
}

void VolumeRenderer::render(const VolumeTexture& volume) {
    // Legacy render function (assumes 1x1x1 box)
    float3 defaultBox = make_float3(1.0f, 1.0f, 1.0f);
    renderKernel<<<gridSize, blockSize>>> ( // call the gpu rendering function
        volume.getTexture(), // the volume texture
        d_output, // the out put buffer
        width, // the screen width
        height, // the screen height
        0.0f, 0.0f, 0.0f, defaultBox
        );

    CHECK_CUDA(cudaGetLastError()); // check for errors
    CHECK_CUDA(cudaDeviceSynchronize()); // synchronize with the cpu
}

std::vector<uchar4> VolumeRenderer::getFrameBuffer() {
    std::vector<uchar4> frameBuffer(width * height);
    CHECK_CUDA(cudaMemcpy(frameBuffer.data(), d_output, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
    return frameBuffer;
}

void VolumeRenderer::cleanup() {
    if (d_output != nullptr) { // delete the Image
        cudaFree(d_output);
        d_output = nullptr;
    }
}

void VolumeRenderer::launchKernel(uchar4* d_output, cudaTextureObject_t texObj, int width, int height, float angleX, float angleY, float opc, float3 boxSize) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    renderKernel<<<gridSize, blockSize>>>(texObj, d_output, width, height, angleX, angleY, opc, boxSize);
    CHECK_CUDA(cudaGetLastError());
}
