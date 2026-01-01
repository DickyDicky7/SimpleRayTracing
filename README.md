# Simple Ray Tracing

A C++20 Monte Carlo Path Tracer implementation exploring various rendering techniques.

## About The Project

This project is a custom ray tracing engine built from scratch in C++. It demonstrates advanced rendering concepts including path tracing, global illumination, and material systems. The project is designed for experimentation with different rendering algorithms and optimization techniques.

## Features

- **Core Rendering Engine**
    - **Monte Carlo Path Tracing**: Unbiased physically based rendering engine.
    - **Motion Blur**: Linear motion blur support for geometric primitives (Spheres, Meshes).
    - **Multithreading**: Custom `ThreadPool` implementation with dynamic task queuing for efficient parallel rendering.
    - **BVH (Bounding Volume Hierarchy)**: High-performance acceleration structure for fast ray-scene intersection.
    - **Global Illumination**: Indirect lighting, soft shadows, and color bleeding.

- **SDF (Signed Distance Fields) & Implicit Surfaces**
    - **Primitives**: Extensive library including Sphere, Box, RoundBox, BoxFrame, Torus, CappedTorus, Link, Cylinder (Infinite/Vertical/Arbitrary/Rounded), Cone (Exact/Bound/Infinite/Capped), Plane, HexPrism, TriPrism, Capsule, SolidAngle, and CutSphere.
    - **Operations**: Smooth Union (blending), Metaballs, and Domain Warping.
    - **Terrain Generation**: Heightmap-based terrain using noise functions.

- **Advanced Material System**
    - **Physical Materials**:
        - **Dielectrics**: Accurate refraction with Schlick's approximation. Includes presets for Air, Water, Skin, Glass, Marble, and Diamond.
        - **Metals**: Configurable fuzziness/roughness.
        - **Lambertian**: Multiple diffuse reflection models.
    - **Specialized Shaders**:
        - **Thin Film Interference**: Simulates iridescence (soap bubbles, oil slicks).
        - **Subsurface Scattering (SSS)**: Random walk implementation for translucent materials (Skin, Marble).
        - **Fresnel Layering**: Complex blended materials (Dielectric/Glossy/Diffuse layers).
    - **Emissive**: Diffuse and Metallic light emitters.

- **Textures & Noise**
    - **Image Support**: PNG, JPG, SVG, EXR texture mapping.
    - **Procedural Textures**: 
        - **Perlin Noise**: Blocky, Smooth, and Hermitian variants with turbulence (Marble-like patterns).
        - **FastNoiseLite Integration**: Support for OpenSimplex2, Cellular, Value, and Domain Warp noise types.
    - **Mapping**: Planar, Spherical, and UV mapping support.

- **Volumetrics & Environment**
    - **Fog Models**: 
        - Simple constant-density scattering.
        - Height-based exponential falloff fog.
    - **Skybox**: 
        - Constant Color, Gradient (Blue-to-White), and Image-based Lighting (IBL).

- **Camera & Optics**
    - **Depth of Field**: Physically accurate defocus blur using disk sampling.
    - **Lens Effects**: Chromatic Aberration post-process effect.

- **Post-Processing Pipeline**
    - **Denoising**: Integrated **OpenImageDenoise (OIDN)** for AI-accelerated high-quality denoising.
    - **Filtering**:
        - **Gaussian Blur**: Optimized 3x3, 5x5, and 7x7 kernels.
        - **Bilateral Filtering**: Edge-preserving noise reduction.
    - **Color & Output**:
        - **Bayer Matrix Dithering**: 2x2 up to 16x16 ordered dithering to prevent color banding.
        - **Tonemapping**: Multiple operators for High Dynamic Range (HDR) to Low Dynamic Range (LDR) conversion.

- **System & Optimization**
    - **Memory Management**: Custom **ArenaAllocator** and **PoolAllocator** for high-performance memory usage.
    - **Math Library**: Custom SIMD-ready `Vec2` / `Vec3` math implementations.
    - **Asset Loading**: Model loading via **Assimp**.
    - **Preview**: Real-time windowed preview using **Raylib**.

> **A Final Note**: If you stumble upon a feature that isn't listed above, please treat it like finding an onion ring in your fries—a delightful bonus! 🍟 This is my absolute first experiment with ray tracing, so if I forgot to mention something (or if half the code is held together by hope and `std::vector`), please bear with me. I'm trying my best! 🥺✨

## Dependencies

- C++20 compatible compiler
- [Raylib](https://www.raylib.com/) (Optional, for `DISPLAY_WINDOW`)
- [Assimp](https://github.com/assimp/assimp)
- [OpenImageDenoise](https://github.com/OpenImageDenoise/oidn) (Optional, for `USE_OIDN`)
- [FreeImage](https://freeimage.sourceforge.io/)

## Building

Ensure you have the required dependencies installed. The project does not currently have a build system file (like CMake) included in the repository root, so you will need to compile `main.cpp` and link against the dependencies.

> **Note**: Build files or systems are intentionally not included to avoid inconveniences with varying environments. Users are encouraged to set up their own build environment with their preferred suitable compiler and build system, utilizing the conceptual guide below as a reference.

Example command (conceptual):
```bash
g++ main.cpp -std=c++20 -O3 -lraylib -lassimp -lOpenImageDenoise -lfreeimage -o RayTracer
```

**MSVC Optimization Flags:**
If building with MSVC, it is recommended to configure the flags as follows to maximize performance (as used in `main.cpp`).

**Enable (ON):**
```
/O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO /Gy (/Gw /favor:AMD64 /Zc:inline)
```
*> Note: If you are using an Intel CPU, you should use `/favor:INTEL64` instead of `/favor:AMD64`.*

**Disable (OFF):**
```
/Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu
```

**Other Compilers:**
If you are using a different compiler (e.g., GCC, Clang), please look for equivalent optimization flags to enable/disable to achieve the best performance.

## Gallery

<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-08/8.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-24/10.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-08/0.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-16/0.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/4.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/12.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/28.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/20.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/3.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/11.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/19.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/27.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/0.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/8.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/16.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/24.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/32.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/40.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/48.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/56.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/64.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/72.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/80.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/88.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/3.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/11.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/19.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/27.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/35.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/43.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/51.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/59.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/3.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/11.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/27.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/19.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/3.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/11.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/19.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/35.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/27.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/43.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/51.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/1.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/9.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/17.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/25.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/33.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/41.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/49.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/57.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/65.png"></p><br/>
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/73.png"></p><br/>
