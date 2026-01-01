<div align="center"><h1>Simple Ray Tracing</h1></div>
<!--
<div align="center"><h1>Simple Ray Tracing</h1></div>
-->



<div align="center"><p>A C++20 Monte Carlo Path Tracer implementation exploring various rendering techniques.</p></div>
<!--
<div align="center"><p>A C++20 Monte Carlo Path Tracer implementation exploring various rendering techniques.</p></div>
-->



<div align="center"><h2>About The Project</h2></div>
<!--
<div align="center"><h2>About The Project</h2></div>
-->



<div align="center"><p>This project is a custom ray tracing engine built from scratch in C++. It demonstrates advanced rendering concepts including path tracing, global illumination, and material systems. The project is designed for experimentation with different rendering algorithms and optimization techniques.</p></div>
<!--
<div align="center"><p>This project is a custom ray tracing engine built from scratch in C++. It demonstrates advanced rendering concepts including path tracing, global illumination, and material systems. The project is designed for experimentation with different rendering algorithms and optimization techniques.</p></div>
-->



<div align="center"><h2>Features</h2></div>
<!--
<div align="center"><h2>Features</h2></div>
-->



<div align="center"><p><strong>Core Rendering Engine</strong></p></div>
<!--
<div align="center"><p><strong>Core Rendering Engine</strong></p></div>
-->
<div align="center"><p><strong>Monte Carlo Path Tracing</strong>: Unbiased physically based rendering engine.</p></div>
<!--
<div align="center"><p><strong>Monte Carlo Path Tracing</strong>: Unbiased physically based rendering engine.</p></div>
-->
<div align="center"><p><strong>Motion Blur</strong>: Linear motion blur support for geometric primitives (Spheres, Meshes).</p></div>
<!--
<div align="center"><p><strong>Motion Blur</strong>: Linear motion blur support for geometric primitives (Spheres, Meshes).</p></div>
-->
<div align="center"><p><strong>Multithreading</strong>: Custom <code>ThreadPool</code> implementation with dynamic task queuing for efficient parallel rendering.</p></div>
<!--
<div align="center"><p><strong>Multithreading</strong>: Custom <code>ThreadPool</code> implementation with dynamic task queuing for efficient parallel rendering.</p></div>
-->
<div align="center"><p><strong>BVH (Bounding Volume Hierarchy)</strong>: High-performance acceleration structure for fast ray-scene intersection.</p></div>
<!--
<div align="center"><p><strong>BVH (Bounding Volume Hierarchy)</strong>: High-performance acceleration structure for fast ray-scene intersection.</p></div>
-->
<div align="center"><p><strong>Global Illumination</strong>: Indirect lighting, soft shadows, and color bleeding.</p></div>
<!--
<div align="center"><p><strong>Global Illumination</strong>: Indirect lighting, soft shadows, and color bleeding.</p></div>
-->



<div align="center"><p><strong>SDF (Signed Distance Fields) & Implicit Surfaces</strong></p></div>
<!--
<div align="center"><p><strong>SDF (Signed Distance Fields) & Implicit Surfaces</strong></p></div>
-->
<div align="center"><p><strong>Primitives</strong>: Extensive library including Sphere, Box, RoundBox, BoxFrame, Torus, CappedTorus, Link, Cylinder (Infinite/Vertical/Arbitrary/Rounded), Cone (Exact/Bound/Infinite/Capped), Plane, HexPrism, TriPrism, Capsule, SolidAngle, and CutSphere.</p></div>
<!--
<div align="center"><p><strong>Primitives</strong>: Extensive library including Sphere, Box, RoundBox, BoxFrame, Torus, CappedTorus, Link, Cylinder (Infinite/Vertical/Arbitrary/Rounded), Cone (Exact/Bound/Infinite/Capped), Plane, HexPrism, TriPrism, Capsule, SolidAngle, and CutSphere.</p></div>
-->
<div align="center"><p><strong>Operations</strong>: Smooth Union (blending), Metaballs, and Domain Warping.</p></div>
<!--
<div align="center"><p><strong>Operations</strong>: Smooth Union (blending), Metaballs, and Domain Warping.</p></div>
-->
<div align="center"><p><strong>Terrain Generation</strong>: Heightmap-based terrain using noise functions.</p></div>
<!--
<div align="center"><p><strong>Terrain Generation</strong>: Heightmap-based terrain using noise functions.</p></div>
-->



<div align="center"><p><strong>Advanced Material System</strong></p></div>
<!--
<div align="center"><p><strong>Advanced Material System</strong></p></div>
-->
<div align="center"><p><strong>Physical Materials</strong>:</p></div>
<!--
<div align="center"><p><strong>Physical Materials</strong>:</p></div>
-->
<div align="center"><p><strong>Dielectrics</strong>: Accurate refraction with Schlick's approximation. Includes presets for Air, Water, Skin, Glass, Marble, and Diamond.</p></div>
<!--
<div align="center"><p><strong>Dielectrics</strong>: Accurate refraction with Schlick's approximation. Includes presets for Air, Water, Skin, Glass, Marble, and Diamond.</p></div>
-->
<div align="center"><p><strong>Metals</strong>: Configurable fuzziness/roughness.</p></div>
<!--
<div align="center"><p><strong>Metals</strong>: Configurable fuzziness/roughness.</p></div>
-->
<div align="center"><p><strong>Lambertian</strong>: Multiple diffuse reflection models.</p></div>
<!--
<div align="center"><p><strong>Lambertian</strong>: Multiple diffuse reflection models.</p></div>
-->
<div align="center"><p><strong>Specialized Shaders</strong>:</p></div>
<!--
<div align="center"><p><strong>Specialized Shaders</strong>:</p></div>
-->
<div align="center"><p><strong>Thin Film Interference</strong>: Simulates iridescence (soap bubbles, oil slicks).</p></div>
<!--
<div align="center"><p><strong>Thin Film Interference</strong>: Simulates iridescence (soap bubbles, oil slicks).</p></div>
-->
<div align="center"><p><strong>Subsurface Scattering (SSS)</strong>: Random walk implementation for translucent materials (Skin, Marble).</p></div>
<!--
<div align="center"><p><strong>Subsurface Scattering (SSS)</strong>: Random walk implementation for translucent materials (Skin, Marble).</p></div>
-->
<div align="center"><p><strong>Fresnel Layering</strong>: Complex blended materials (Dielectric/Glossy/Diffuse layers).</p></div>
<!--
<div align="center"><p><strong>Fresnel Layering</strong>: Complex blended materials (Dielectric/Glossy/Diffuse layers).</p></div>
-->
<div align="center"><p><strong>Emissive</strong>: Diffuse and Metallic light emitters.</p></div>
<!--
<div align="center"><p><strong>Emissive</strong>: Diffuse and Metallic light emitters.</p></div>
-->



<div align="center"><p><strong>Textures & Noise</strong></p></div>
<!--
<div align="center"><p><strong>Textures & Noise</strong></p></div>
-->
<div align="center"><p><strong>Image Support</strong>: PNG, JPG, SVG, EXR texture mapping.</p></div>
<!--
<div align="center"><p><strong>Image Support</strong>: PNG, JPG, SVG, EXR texture mapping.</p></div>
-->
<div align="center"><p><strong>Procedural Textures</strong>:</p></div>
<!--
<div align="center"><p><strong>Procedural Textures</strong>:</p></div>
-->
<div align="center"><p><strong>Perlin Noise</strong>: Blocky, Smooth, and Hermitian variants with turbulence (Marble-like patterns).</p></div>
<!--
<div align="center"><p><strong>Perlin Noise</strong>: Blocky, Smooth, and Hermitian variants with turbulence (Marble-like patterns).</p></div>
-->
<div align="center"><p><strong>FastNoiseLite Integration</strong>: Support for OpenSimplex2, Cellular, Value, and Domain Warp noise types.</p></div>
<!--
<div align="center"><p><strong>FastNoiseLite Integration</strong>: Support for OpenSimplex2, Cellular, Value, and Domain Warp noise types.</p></div>
-->
<div align="center"><p><strong>Mapping</strong>: Planar, Spherical, and UV mapping support.</p></div>
<!--
<div align="center"><p><strong>Mapping</strong>: Planar, Spherical, and UV mapping support.</p></div>
-->



<div align="center"><p><strong>Volumetrics & Environment</strong></p></div>
<!--
<div align="center"><p><strong>Volumetrics & Environment</strong></p></div>
-->
<div align="center"><p><strong>Fog Models</strong>:</p></div>
<!--
<div align="center"><p><strong>Fog Models</strong>:</p></div>
-->
<div align="center"><p>Simple constant-density scattering.</p></div>
<!--
<div align="center"><p>Simple constant-density scattering.</p></div>
-->
<div align="center"><p>Height-based exponential falloff fog.</p></div>
<!--
<div align="center"><p>Height-based exponential falloff fog.</p></div>
-->
<div align="center"><p><strong>Skybox</strong>:</p></div>
<!--
<div align="center"><p><strong>Skybox</strong>:</p></div>
-->
<div align="center"><p>Constant Color, Gradient (Blue-to-White), and Image-based Lighting (IBL).</p></div>
<!--
<div align="center"><p>Constant Color, Gradient (Blue-to-White), and Image-based Lighting (IBL).</p></div>
-->



<div align="center"><p><strong>Camera & Optics</strong></p></div>
<!--
<div align="center"><p><strong>Camera & Optics</strong></p></div>
-->
<div align="center"><p><strong>Depth of Field</strong>: Physically accurate defocus blur using disk sampling.</p></div>
<!--
<div align="center"><p><strong>Depth of Field</strong>: Physically accurate defocus blur using disk sampling.</p></div>
-->
<div align="center"><p><strong>Lens Effects</strong>: Chromatic Aberration post-process effect.</p></div>
<!--
<div align="center"><p><strong>Lens Effects</strong>: Chromatic Aberration post-process effect.</p></div>
-->



<div align="center"><p><strong>Post-Processing Pipeline</strong></p></div>
<!--
<div align="center"><p><strong>Post-Processing Pipeline</strong></p></div>
-->
<div align="center"><p><strong>Denoising</strong>: Integrated <strong>OpenImageDenoise (OIDN)</strong> for AI-accelerated high-quality denoising.</p></div>
<!--
<div align="center"><p><strong>Denoising</strong>: Integrated <strong>OpenImageDenoise (OIDN)</strong> for AI-accelerated high-quality denoising.</p></div>
-->
<div align="center"><p><strong>Filtering</strong>:</p></div>
<!--
<div align="center"><p><strong>Filtering</strong>:</p></div>
-->
<div align="center"><p><strong>Gaussian Blur</strong>: Optimized 3x3, 5x5, and 7x7 kernels.</p></div>
<!--
<div align="center"><p><strong>Gaussian Blur</strong>: Optimized 3x3, 5x5, and 7x7 kernels.</p></div>
-->
<div align="center"><p><strong>Bilateral Filtering</strong>: Edge-preserving noise reduction.</p></div>
<!--
<div align="center"><p><strong>Bilateral Filtering</strong>: Edge-preserving noise reduction.</p></div>
-->
<div align="center"><p><strong>Color & Output</strong>:</p></div>
<!--
<div align="center"><p><strong>Color & Output</strong>:</p></div>
-->
<div align="center"><p><strong>Bayer Matrix Dithering</strong>: 2x2 up to 16x16 ordered dithering to prevent color banding.</p></div>
<!--
<div align="center"><p><strong>Bayer Matrix Dithering</strong>: 2x2 up to 16x16 ordered dithering to prevent color banding.</p></div>
-->
<div align="center"><p><strong>Tonemapping</strong>: Multiple operators for High Dynamic Range (HDR) to Low Dynamic Range (LDR) conversion.</p></div>
<!--
<div align="center"><p><strong>Tonemapping</strong>: Multiple operators for High Dynamic Range (HDR) to Low Dynamic Range (LDR) conversion.</p></div>
-->



<div align="center"><p><strong>System & Optimization</strong></p></div>
<!--
<div align="center"><p><strong>System & Optimization</strong></p></div>
-->
<div align="center"><p><strong>Memory Management</strong>: Custom <strong>ArenaAllocator</strong> and <strong>PoolAllocator</strong> for high-performance memory usage.</p></div>
<!--
<div align="center"><p><strong>Memory Management</strong>: Custom <strong>ArenaAllocator</strong> and <strong>PoolAllocator</strong> for high-performance memory usage.</p></div>
-->
<div align="center"><p><strong>Math Library</strong>: Custom SIMD-ready <code>Vec2</code> / <code>Vec3</code> math implementations.</p></div>
<!--
<div align="center"><p><strong>Math Library</strong>: Custom SIMD-ready <code>Vec2</code> / <code>Vec3</code> math implementations.</p></div>
-->
<div align="center"><p><strong>Asset Loading</strong>: Model loading via <strong>Assimp</strong>.</p></div>
<!--
<div align="center"><p><strong>Asset Loading</strong>: Model loading via <strong>Assimp</strong>.</p></div>
-->
<div align="center"><p><strong>Preview</strong>: Real-time windowed preview using <strong>Raylib</strong>.</p></div>
<!--
<div align="center"><p><strong>Preview</strong>: Real-time windowed preview using <strong>Raylib</strong>.</p></div>
-->



<div align="center"><blockquote><p><strong>A Final Note</strong>: If you stumble upon a feature that isn't listed above, please treat it like finding an onion ring in your fries—a delightful bonus! 🍟 This is my absolute first experiment with ray tracing, so if I forgot to mention something (or if half the code is held together by hope and <code>std::vector</code>), please bear with me. I'm trying my best! 🥺✨</p></blockquote></div>
<!--
<div align="center"><blockquote><p><strong>A Final Note</strong>: If you stumble upon a feature that isn't listed above, please treat it like finding an onion ring in your fries—a delightful bonus! 🍟 This is my absolute first experiment with ray tracing, so if I forgot to mention something (or if half the code is held together by hope and <code>std::vector</code>), please bear with me. I'm trying my best! 🥺✨</p></blockquote></div>
-->



<div align="center"><h2>Dependencies</h2></div>
<!--
<div align="center"><h2>Dependencies</h2></div>
-->



<div align="center"><p>- C++20 compatible compiler</p></div>
<!--
<div align="center"><p>- C++20 compatible compiler</p></div>
-->
<div align="center"><p>- <a href="https://www.raylib.com/">Raylib</a> (Optional, for <code>DISPLAY_WINDOW</code>)</p></div>
<!--
<div align="center"><p>- <a href="https://www.raylib.com/">Raylib</a> (Optional, for <code>DISPLAY_WINDOW</code>)</p></div>
-->
<div align="center"><p>- <a href="https://github.com/assimp/assimp">Assimp</a></p></div>
<!--
<div align="center"><p>- <a href="https://github.com/assimp/assimp">Assimp</a></p></div>
-->
<div align="center"><p>- <a href="https://github.com/OpenImageDenoise/oidn">OpenImageDenoise</a> (Optional, for <code>USE_OIDN</code>)</p></div>
<!--
<div align="center"><p>- <a href="https://github.com/OpenImageDenoise/oidn">OpenImageDenoise</a> (Optional, for <code>USE_OIDN</code>)</p></div>
-->
<div align="center"><p>- <a href="https://freeimage.sourceforge.io/">FreeImage</a></p></div>
<!--
<div align="center"><p>- <a href="https://freeimage.sourceforge.io/">FreeImage</a></p></div>
-->



<div align="center"><h2>Building</h2></div>
<!--
<div align="center"><h2>Building</h2></div>
-->



<div align="center"><p>Ensure you have the required dependencies installed. The project does not currently have a build system file (like CMake) included in the repository root, so you will need to compile <code>main.cpp</code> and link against the dependencies.</p></div>
<!--
<div align="center"><p>Ensure you have the required dependencies installed. The project does not currently have a build system file (like CMake) included in the repository root, so you will need to compile <code>main.cpp</code> and link against the dependencies.</p></div>
-->



<div align="center"><blockquote><p><strong>Note</strong>: Build files or systems are intentionally not included to avoid inconveniences with varying environments. Users are encouraged to set up their own build environment with their preferred suitable compiler and build system, utilizing the conceptual guide below as a reference.</p></blockquote></div>
<!--
<div align="center"><blockquote><p><strong>Note</strong>: Build files or systems are intentionally not included to avoid inconveniences with varying environments. Users are encouraged to set up their own build environment with their preferred suitable compiler and build system, utilizing the conceptual guide below as a reference.</p></blockquote></div>
-->



<div align="center"><p>Example command (conceptual):</p></div>
<!--
<div align="center"><p>Example command (conceptual):</p></div>
-->
<div align="center"><pre><code>g++ main.cpp -std=c++20 -O3 -lraylib -lassimp -lOpenImageDenoise -lfreeimage -o RayTracer</code></pre></div>
<!--
<div align="center"><pre><code>g++ main.cpp -std=c++20 -O3 -lraylib -lassimp -lOpenImageDenoise -lfreeimage -o RayTracer</code></pre></div>
-->



<div align="center"><p><strong>MSVC Optimization Flags:</strong></p></div>
<!--
<div align="center"><p><strong>MSVC Optimization Flags:</strong></p></div>
-->
<div align="center"><p>If building with MSVC, it is recommended to configure the flags as follows to maximize performance (as used in <code>main.cpp</code>).</p></div>
<!--
<div align="center"><p>If building with MSVC, it is recommended to configure the flags as follows to maximize performance (as used in <code>main.cpp</code>).</p></div>
-->



<div align="center"><p><strong>Enable (ON):</strong></p></div>
<!--
<div align="center"><p><strong>Enable (ON):</strong></p></div>
-->
<div align="center"><pre><code>/O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO /Gy (/Gw /favor:AMD64 /Zc:inline)</code></pre></div>
<!--
<div align="center"><pre><code>/O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO /Gy (/Gw /favor:AMD64 /Zc:inline)</code></pre></div>
-->
<div align="center"><p><em>&gt; Note: If you are using an Intel CPU, you should use <code>/favor:INTEL64</code> instead of <code>/favor:AMD64</code>.</em></p></div>
<!--
<div align="center"><p><em>&gt; Note: If you are using an Intel CPU, you should use <code>/favor:INTEL64</code> instead of <code>/favor:AMD64</code>.</em></p></div>
-->



<div align="center"><p><strong>Disable (OFF):</strong></p></div>
<!--
<div align="center"><p><strong>Disable (OFF):</strong></p></div>
-->
<div align="center"><pre><code>/Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu</code></pre></div>
<!--
<div align="center"><pre><code>/Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu</code></pre></div>
-->



<div align="center"><p><strong>Other Compilers:</strong></p></div>
<!--
<div align="center"><p><strong>Other Compilers:</strong></p></div>
-->
<div align="center"><p>If you are using a different compiler (e.g., GCC, Clang), please look for equivalent optimization flags to enable/disable to achieve the best performance.</p></div>
<!--
<div align="center"><p>If you are using a different compiler (e.g., GCC, Clang), please look for equivalent optimization flags to enable/disable to achieve the best performance.</p></div>
-->
<div align="center"><h2>Gallery</h2></div>
<!--
<div align="center"><h2>Gallery</h2></div>
-->



<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-08/8.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-08/8.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-24/10.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-04-24/10.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-08/0.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-08/0.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-16/0.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-16/0.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/4.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/4.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/12.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/12.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/28.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/28.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/20.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-05-24/20.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/3.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/3.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/11.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/11.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/19.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/19.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/27.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-06-16/27.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/0.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/0.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/8.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/8.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/16.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/16.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/24.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/24.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/32.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/32.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/40.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/40.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/48.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/48.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/56.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/56.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/64.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/64.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/72.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/72.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/80.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/80.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/88.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-08/88.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/3.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/3.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/11.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/11.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/19.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/19.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/27.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/27.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/35.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/35.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/43.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/43.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/51.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/51.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/59.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-07-24/59.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/3.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/3.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/11.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/11.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/27.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/27.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/19.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-08/19.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/3.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/3.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/11.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/11.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/19.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/19.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/35.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/35.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/27.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/27.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/43.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/43.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/51.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-08-16/51.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/1.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/1.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/9.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/9.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/17.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/17.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/25.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/25.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/33.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/33.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/41.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/41.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/49.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/49.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/57.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/57.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/65.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/65.png">######-->
<p align="center"><img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/73.png"></p><br/>
<!--##############<img src="https://raw.githubusercontent.com/DickyDicky7/SimpleRayTracingRenderResults/refs/heads/main/2025-09-08/73.png">######-->


