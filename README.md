# CUDA Volume Ray-Caster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![Tech](https://img.shields.io/badge/C%2B%2B-17-green)
![Tech](https://img.shields.io/badge/Qt-6.8-green)
![Tech](https://img.shields.io/badge/CUDA-12-76B900)

> *Real-time volumetric rendering of 8-bit raw medical datasets using a custom CUDA ray-marching kernel.*

## 📖 Overview

This project is a high-performance volumetric visualization engine designed to bridge the gap between **compute (CUDA)** and **interface (Qt 6)**. 

Unlike standard visualization tools that rely on CPU-based rendering or heavy libraries like VTK, this project implements a **raw C++ Ray-Marching kernel from scratch**. It achieves **Zero-Copy latency** by mapping OpenGL Pixel Buffer Objects (PBOs) directly into the CUDA memory space, allowing for real-time interaction (135+ FPS) on consumer-grade hardware (GTX 1650).

## ✨ Key Features

* **Custom CUDA Kernel:** Implemented stochastic ray-marching with front-to-back alpha accumulation and trilinear texture interpolation.
* **Zero-Copy Architecture:** Utilized `cuda_gl_interop` to write kernel outputs directly to the OpenGL framebuffer, bypassing the CPU and PCIe bus completely.
* **Qt Quick (QML) Integration:** Decoupled the Rendering Thread from the Main GUI Thread using `QQuickFramebufferObject` for a responsive, lock-free UI.
* **Interactive 3D Rotation:** Implemented Euler angle rotation matrices directly within the kernel to support arbitrary Pitch/Yaw manipulation via mouse drag.
* **Aspect Ratio Correction:** Logic to handle non-cubic datasets (e.g., flattened 256x256x56 scans) without visual distortion.
* **Dynamic Transfer Functions:** Hardcoded classification logic for distinguishing Tissue vs. Bone (Human Head) or Shell vs. Meat (Lobster).

## 🛠️ Technical Stack

* **Core Language:** C++17
* **Compute:** NVIDIA CUDA 12.x
* **Graphics:** OpenGL 4.5
* **UI Framework:** Qt 6.8 (QML & Qt Quick)
* **Build System:** CMake 3.18+
* **Compiler:** MSVC 2022 (Windows)

## 🏗️ System Architecture

The application uses a Producer-Consumer model between the CPU and GPU:

1.  **Initialization:**
    * Raw binary data (`.raw`) is loaded into Host RAM.
    * Data is uploaded to a **CUDA 3D Texture Object** (VRAM) for hardware-accelerated sampling.
2.  **Render Loop:**
    * **Qt (Main Thread):** Captures mouse events and updates rotation properties.
    * **Sync:** Data is safely copied to the Render Thread.
    * **CUDA (Compute Stream):** Maps the OpenGL PBO, ray-casts the volume, writes pixels, and unmaps.
    * **OpenGL (Render Stream):** Blits the PBO to a texture and draws a fullscreen quad.

## 📦 Build Instructions

### Prerequisites
* Visual Studio 2022 (C++ Desktop Dev workload).
* CUDA Toolkit v11.0+.
* Qt 6 (MSVC 64-bit).
* CMake.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MoustafaMahmoud287/CudaVolumeRenderer.git](https://github.com/MoustafaMahmoud287/CudaVolumeRenderer.git)
    cd CudaVolumeRenderer
    ```

2.  **Configure with CMake:**
    ```bash
    mkdir build && cd build
    cmake .. -G "Visual Studio 17 2022" -A x64
    ```

3.  **Build:**
    ```bash
    cmake --build . --config Release
    ```
    *Note: The `synthetic_skull.raw` or configured dataset will be automatically copied to the build directory via CMake rules.*

4.  **Run:**
    ```bash
    ./Release/QtCudaVis.exe
    ```

## 🎮 Controls

* **Left Click & Drag:** Rotate the volume (Yaw/Pitch).
* **Resize Window:** The renderer automatically resizes the grid and PBO to match the new resolution.

## 🔮 Future Roadmap

* [ ] Implement 1D Transfer Function Texture lookup (Gradient Editor UI).
* [ ] Add Phong Shading by calculating gradients (normals) on the fly.
* [ ] Implement Empty Space Skipping (Octrees) for large datasets (>512^3).

## 📄 License

This project is licensed under the MIT License.

---

**Built by [Moustafa Mahmoud](https://github.com/MoustafaMahmoud287)**
