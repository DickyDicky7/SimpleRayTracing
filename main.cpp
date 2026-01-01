#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

        #define SCENE_001
//      #define SCENE_001
//      #define SCENE_002
//      #define SCENE_002
//      #define SCENE_003
//      #define SCENE_003

//      #define OLD_IMAGE_SAMPLING
        #define NEW_IMAGE_SAMPLING

        #define USE_OIDN
//      #define USE_OIDN

//      #define FOG_SIMPLE
//      #define FOG_SIMPLE
//      #define FOG_HEIGHT
//      #define FOG_HEIGHT

//      #define DISPLAY_WINDOW
//      #define DISPLAY_WINDOW

#include <string_view>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <array>
#include "SDF.h"
#include <iomanip>
#include <cstdint>
#include <numbers>
#include "ThreadPool.h"
#include <chrono>
#include <random>
#include <vector>
#include "Lazy.h"
#include "Vec2.h"
#include "Vec3.h"
#include <span>
#include "Image.h"
#include "ImagePNG.h"
#include "ImageJPG.h"
#include "ImageSVG.h"
#include "ImageEXR.h"
  #include "PostProcessing.h"
//#include "PostProcessing.h"
  #include <assimp/Importer.hpp>
//#include <assimp/Importer.hpp>
  #include <assimp/scene.h>
//#include <assimp/scene.h>
  #include <assimp/postprocess.h>
//#include <assimp/postprocess.h>
  #include "FastNoiseLite.h"
//#include "FastNoiseLite.h"
#ifdef USE_OIDN
  #include <OpenImageDenoise/oidn.hpp>
//#include <OpenImageDenoise/oidn.hpp>
#endif
#ifdef FOG_SIMPLE
    constexpr Color3 _EX_SCATTERING_FOG_SIMPLE_DENSITY_{ 0.09f, 0.09f, 0.09f };
//  constexpr Color3 _EX_SCATTERING_FOG_SIMPLE_DENSITY_{ 0.09f, 0.09f, 0.09f };
    constexpr Color3 _IN_SCATTERING_FOG_SIMPLE_DENSITY_{ 0.01f, 0.01f, 0.01f };
//  constexpr Color3 _IN_SCATTERING_FOG_SIMPLE_DENSITY_{ 0.01f, 0.01f, 0.01f };
    constexpr Color3 _FOG_SIMPLE_COLOR_{ 0.5f, 0.6f, 0.7f };
//  constexpr Color3 _FOG_SIMPLE_COLOR_{ 0.5f, 0.6f, 0.7f };
#endif
#ifdef FOG_HEIGHT
    constexpr float  _FOG_HEIGHT_DENSITY_ = 0.01f;
//  constexpr float  _FOG_HEIGHT_DENSITY_ = 0.01f;
    constexpr float  _FOG_HEIGHT_FALLOFF_ = 0.08f;
//  constexpr float  _FOG_HEIGHT_FALLOFF_ = 0.08f;
    constexpr Color3 _FOG_HEIGHT_COLOR_{ 0.5f, 0.6f, 0.7f };
//  constexpr Color3 _FOG_HEIGHT_COLOR_{ 0.5f, 0.6f, 0.7f };
#endif

    #ifdef DISPLAY_WINDOW
//  #ifdef DISPLAY_WINDOW
    namespace rl
//  namespace rl
    {
        extern "C"
//      extern "C"
        {
            #include <raylib.h>
//          #include <raylib.h>
        }
    }
    #endif
//  #endif

    enum class BackgroundType : std::uint8_t
//  enum class BackgroundType : std::uint8_t
{
    BLUE_LERP_WHITE = 0,
//  BLUE_LERP_WHITE = 0,
    DARK_ROOM_SPACE = 1,
//  DARK_ROOM_SPACE = 1,
    SKY_BOX         = 2,
//  SKY_BOX         = 2,
};

    static inline const constexpr float Safe(bool expression, float valueT, float valueF) { if (expression) { return valueT; } else { return valueF; } }
//  static inline const constexpr float Safe(bool expression, float valueT, float valueF) { if (expression) { return valueT; } else { return valueF; } }

static inline float Sample1LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    return 0.0f;
}
static inline float Sample2LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    int pixelX = static_cast<int>(std::floor(x));
    int pixelY = static_cast<int>(std::floor(y));

    float deltaX = x - pixelX;
    float deltaY = y - pixelY;

    int currPixelX = std::clamp(pixelX + 0, 0, imgW - 1);
    int nextPixelX = std::clamp(pixelX + 1, 0, imgW - 1);
    int currPixelY = std::clamp(pixelY + 0, 0, imgH - 1);
    int nextPixelY = std::clamp(pixelY + 1, 0, imgH - 1);

    size_t indexOfTLPixelWithValueAtColorChannel = (static_cast<size_t>(currPixelY) * imgW + currPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfTRPixelWithValueAtColorChannel = (static_cast<size_t>(currPixelY) * imgW + nextPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfBLPixelWithValueAtColorChannel = (static_cast<size_t>(nextPixelY) * imgW + currPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfBRPixelWithValueAtColorChannel = (static_cast<size_t>(nextPixelY) * imgW + nextPixelX) * numberOfColorChannels + colorChannel;

    float valueLerpTop = (1.0f - deltaX) * rgbs[indexOfTLPixelWithValueAtColorChannel] + deltaX * rgbs[indexOfTRPixelWithValueAtColorChannel];
    float valueLerpBot = (1.0f - deltaX) * rgbs[indexOfBLPixelWithValueAtColorChannel] + deltaX * rgbs[indexOfBRPixelWithValueAtColorChannel];
    float valueLerpVer = (1.0f - deltaY) * valueLerpTop
                       +         deltaY  * valueLerpBot
                       ;

    return valueLerpVer;
//  return valueLerpVer;

    return 0.0f;
}
static inline float Sample3LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    return 0.0f;
}

  static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { return std::powf(linearSpaceComponent, 1.0f / 2.0f); }
//static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { return std::powf(linearSpaceComponent, 1.0f / 2.0f); }
  static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return std::powf(gammasSpaceComponent,        2.0f); }
//static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return std::powf(gammasSpaceComponent,        2.0f); }

    constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
//  constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
    constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();
//  constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();

static
inline float Random()
{
    thread_local static std::random_device rd;// Non-deterministic seed source
//  thread_local static std::random_device rd;// Non-deterministic seed source
    thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//  thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    thread_local static std::mt19937 generator  (rd())   ;
//  thread_local static std::mt19937 generator/*(rd())*/ ;
    return distribution(generator);
//  return distribution(generator);

//    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
////  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//    static std::mt19937 generator ;
////  static std::mt19937 generator ;
//    return distribution(generator);
////  return distribution(generator);
}

//       thread_local static uint32_t seed = 123456789;
////     thread_local static uint32_t seed = 123456789;
//static inline float Random()
//{
//    seed ^= seed << 13;
////  seed ^= seed << 13;
//    seed ^= seed >> 17;
////  seed ^= seed >> 17;
//    seed ^= seed << 5 ;
////  seed ^= seed << 5 ;
//    return (seed & 0xFFFFFF) / 16777216.0f; // Normalize to [ 0.0f , 1.0f ]
////  return (seed & 0xFFFFFF) / 16777216.0f; // Normalize to [ 0.0f , 1.0f ]
//}


static
inline float Random(float min, float max)
{
    return min + (max - min) * Random();
//  return min + (max - min) * Random();
}

    static inline int RandomInt(int min, int max) { return int(Random((float)min, float(max + 1))); }
//  static inline int RandomInt(int min, int max) { return int(Random((float)min, float(max + 1))); }

struct Interval
{
    float min = positiveInfinity;
    float max = negativeInfinity;

    static const Interval empty;
//  static const Interval empty;
    static const Interval universe;
//  static const Interval universe;

    bool Contains (float x) const { return min <= x && x <= max; }
//  bool Contains (float x) const { return min <= x && x <= max; }
    bool Surrounds(float x) const { return min <  x && x <  max; }
//  bool Surrounds(float x) const { return min <  x && x <  max; }

    float Clamp(float x) const
    {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    Interval Expand(float delta) const { float padding = delta / 2.0f; return Interval{ .min = min - padding, .max = max + padding }; }
//  Interval Expand(float delta) const { float padding = delta / 2.0f; return Interval{ .min = min - padding, .max = max + padding }; }
};

    const Interval Interval::empty    { positiveInfinity, negativeInfinity };
//  const Interval Interval::empty    { positiveInfinity, negativeInfinity };
    const Interval Interval::universe { negativeInfinity, positiveInfinity };
//  const Interval Interval::universe { negativeInfinity, positiveInfinity };


struct AABB2D
{
    Interval intervalAxisX;
    Interval intervalAxisY;
//  Interval intervalAxisZ;
};
struct AABB3D
{
    Interval intervalAxisX;
    Interval intervalAxisY;
    Interval intervalAxisZ;
};


    inline static std::string GetCurrentDateTime()
//  inline static std::string GetCurrentDateTime()
{
        std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm); return std::string(buffer);
//      std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm); return std::string(buffer);
}














static inline Vec3 SampleRGB1LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y)
{
    return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
}
static inline Vec3 SampleRGB2LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y)
{
    int pixelX = static_cast<int>(std::floor(x));
    int pixelY = static_cast<int>(std::floor(y));

    float deltaX = x - pixelX;
    float deltaY = y - pixelY;

    int currPixelX = std::clamp(pixelX + 0, 0, imgW - 1);
    int nextPixelX = std::clamp(pixelX + 1, 0, imgW - 1);
    int currPixelY = std::clamp(pixelY + 0, 0, imgH - 1);
    int nextPixelY = std::clamp(pixelY + 1, 0, imgH - 1);

    size_t indexOfTLPixel = (static_cast<size_t>(currPixelY) * imgW + currPixelX) * 3;
    size_t indexOfTRPixel = (static_cast<size_t>(currPixelY) * imgW + nextPixelX) * 3;
    size_t indexOfBLPixel = (static_cast<size_t>(nextPixelY) * imgW + currPixelX) * 3;
    size_t indexOfBRPixel = (static_cast<size_t>(nextPixelY) * imgW + nextPixelX) * 3;

    float valueLerpTopR = (1.0f - deltaX) * rgbs[indexOfTLPixel + 0] + deltaX * rgbs[indexOfTRPixel + 0];
    float valueLerpBotR = (1.0f - deltaX) * rgbs[indexOfBLPixel + 0] + deltaX * rgbs[indexOfBRPixel + 0];
    float valueLerpVerR = (1.0f - deltaY) * valueLerpTopR
                        +         deltaY  * valueLerpBotR
                        ;

    float valueLerpTopG = (1.0f - deltaX) * rgbs[indexOfTLPixel + 1] + deltaX * rgbs[indexOfTRPixel + 1];
    float valueLerpBotG = (1.0f - deltaX) * rgbs[indexOfBLPixel + 1] + deltaX * rgbs[indexOfBRPixel + 1];
    float valueLerpVerG = (1.0f - deltaY) * valueLerpTopG
                        +         deltaY  * valueLerpBotG
                        ;

    float valueLerpTopB = (1.0f - deltaX) * rgbs[indexOfTLPixel + 2] + deltaX * rgbs[indexOfTRPixel + 2];
    float valueLerpBotB = (1.0f - deltaX) * rgbs[indexOfBLPixel + 2] + deltaX * rgbs[indexOfBRPixel + 2];
    float valueLerpVerB = (1.0f - deltaY) * valueLerpTopB
                        +         deltaY  * valueLerpBotB
                        ;

    return Vec3{ .x = valueLerpVerR, .y = valueLerpVerG, .z = valueLerpVerB };
//  return Vec3{ .x = valueLerpVerR, .y = valueLerpVerG, .z = valueLerpVerB };

    return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
}
static inline Vec3 SampleRGB3LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y)
{
    return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  return Vec3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
}





  enum class NoisePerlinType : std::uint8_t
//enum class NoisePerlinType : std::uint8_t
  {
      BLOCKY = 0,
//    BLOCKY = 0,
      SMOOTH = 1,
//    SMOOTH = 1,
      SMOOTH_HERMITIAN = 2,
//    SMOOTH_HERMITIAN = 2,
      SMOOTH_SHIFT_OFF = 3,
//    SMOOTH_SHIFT_OFF = 3,
  };


  enum class NoisePerlinProcedureType : std::uint8_t
//enum class NoisePerlinProcedureType : std::uint8_t
  {
      NOISE_BASIC = 0,
//    NOISE_BASIC = 0,
      NOISE_NORMALIZED = 1,
//    NOISE_NORMALIZED = 1,
      TURBULENCE_1 = 2,
//    TURBULENCE_1 = 2,
      TURBULENCE_2 = 3,
//    TURBULENCE_2 = 3,
  };


  struct NoisePerlin
//struct NoisePerlin
  {
      std::array<Vec3, 256> randomFloat3s;
//    std::array<Vec3, 256> randomFloat3s;
      std::array<int , 256> permutationsX;
//    std::array<int , 256> permutationsX;
      std::array<int , 256> permutationsY;
//    std::array<int , 256> permutationsY;
      std::array<int , 256> permutationsZ;
//    std::array<int , 256> permutationsZ;
      NoisePerlinType noisePerlinType;
//    NoisePerlinType noisePerlinType;
      NoisePerlinProcedureType noisePerlinProcedureType;
//    NoisePerlinProcedureType noisePerlinProcedureType;
  };


  inline static void Generate(NoisePerlin& np)
//inline static void Generate(NoisePerlin& np)
  {
      switch (np.noisePerlinType)
//    switch (np.noisePerlinType)
      {
          case NoisePerlinType::BLOCKY:
//        case NoisePerlinType::BLOCKY:
          case NoisePerlinType::SMOOTH:
//        case NoisePerlinType::SMOOTH:
          case NoisePerlinType::SMOOTH_HERMITIAN:
//        case NoisePerlinType::SMOOTH_HERMITIAN:
          {
              for (Vec3& randomFloat3 : np.randomFloat3s)
//            for (Vec3& randomFloat3 : np.randomFloat3s)
              {
                  randomFloat3.x = Random();
//                randomFloat3.x = Random();
              }
              for (int i = 000; i < 256; ++i)
              {
                  np.permutationsX[i] = i;
                  np.permutationsY[i] = i;
                  np.permutationsZ[i] = i;
              }
              for (int i = 255; i > 000; --i)
              {
                  int targetX = RandomInt(0, i);
                  int targetY = RandomInt(0, i);
                  int targetZ = RandomInt(0, i);
                  std::swap(np.permutationsX[i], np.permutationsX[targetX]);
                  std::swap(np.permutationsY[i], np.permutationsY[targetY]);
                  std::swap(np.permutationsZ[i], np.permutationsZ[targetZ]);
              }
          }
          break;
//        break;


          case NoisePerlinType::SMOOTH_SHIFT_OFF:
//        case NoisePerlinType::SMOOTH_SHIFT_OFF:
          {
              for (Vec3& randomFloat3 : np.randomFloat3s)
//            for (Vec3& randomFloat3 : np.randomFloat3s)
              {
                  randomFloat3 = Normalize(Vec3{ .x = Random(-1.0f, +1.0f), .y = Random(-1.0f, +1.0f), .z = Random(-1.0f, +1.0f) });
//                randomFloat3 = Normalize(Vec3{ .x = Random(-1.0f, +1.0f), .y = Random(-1.0f, +1.0f), .z = Random(-1.0f, +1.0f) });
              }
              for (int i = 000; i < 256; ++i)
              {
                  np.permutationsX[i] = i;
                  np.permutationsY[i] = i;
                  np.permutationsZ[i] = i;
              }
              for (int i = 255; i > 000; --i)
              {
                  int targetX = RandomInt(0, i);
                  int targetY = RandomInt(0, i);
                  int targetZ = RandomInt(0, i);
                  std::swap(np.permutationsX[i], np.permutationsX[targetX]);
                  std::swap(np.permutationsY[i], np.permutationsY[targetY]);
                  std::swap(np.permutationsZ[i], np.permutationsZ[targetZ]);
              }
          }
          break;
//        break;


          default:
//        default:
          {

          }
          break;
//        break;
      }
  }


  inline static float GetNoiseValue(const NoisePerlin& np, const Point3& p)
//inline static float GetNoiseValue(const NoisePerlin& np, const Point3& p)
  {
      float noisePerlinResult = 0.0f;
//    float noisePerlinResult = 0.0f;

      switch (np.noisePerlinType)
//    switch (np.noisePerlinType)
      {
          case NoisePerlinType::BLOCKY:
//        case NoisePerlinType::BLOCKY:
          {
              int i = static_cast<int>(4 * p.x) & 255;
              int j = static_cast<int>(4 * p.y) & 255;
              int k = static_cast<int>(4 * p.z) & 255;
              noisePerlinResult = np.randomFloat3s[np.permutationsX[i] ^
                                                   np.permutationsY[j] ^
                                                   np.permutationsZ[k]].x;
          }
          break;
//        break;


          case NoisePerlinType::SMOOTH:
//        case NoisePerlinType::SMOOTH:
          {
              float u = p.x - std::floor(p.x);
              float v = p.y - std::floor(p.y);
              float w = p.z - std::floor(p.z);

              int i = static_cast<int>(std::floor(p.x));
              int j = static_cast<int>(std::floor(p.y));
              int k = static_cast<int>(std::floor(p.z));

              float c[2][2][2]{};
//            float c[2][2][2]{};

              for (int di = 0; di < 2; ++di)
              for (int dj = 0; dj < 2; ++dj)
              for (int dk = 0; dk < 2; ++dk)
                  c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
                                                   np.permutationsY[(j + dj) & 255] ^
                                                   np.permutationsZ[(k + dk) & 255]].x;
//                c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
//                                                 np.permutationsY[(j + dj) & 255] ^
//                                                 np.permutationsZ[(k + dk) & 255]].x;

//            trilinear interpolation
//            trilinear interpolation
              float accum = 0.0f;
//            float accum = 0.0f;
              for (int ii = 0; ii < 2; ++ii)
              for (int jj = 0; jj < 2; ++jj)
              for (int kk = 0; kk < 2; ++kk)
                    accum+= ((ii * u) + (1 - ii) * (1 - u))
                          * ((jj * v) + (1 - jj) * (1 - v))
                          * ((kk * w) + (1 - kk) * (1 - w))
                          * c[ii][jj][kk];
//                  accum+= ((ii * u) + (1 - ii) * (1 - u))
//                        * ((jj * v) + (1 - jj) * (1 - v))
//                        * ((kk * w) + (1 - kk) * (1 - w))
//                        * c[ii][jj][kk];
              noisePerlinResult = accum;
//            noisePerlinResult = accum;
//            trilinear interpolation
//            trilinear interpolation
          }
          break;
//        break;


          case NoisePerlinType::SMOOTH_HERMITIAN:
//        case NoisePerlinType::SMOOTH_HERMITIAN:
          {
              float u = p.x - std::floor(p.x);
              float v = p.y - std::floor(p.y);
              float w = p.z - std::floor(p.z);
//            A standard trick is to use a Hermite cubic to round off the interpolation (Improvement with Hermitian Smoothing)
//            A standard trick is to use a Hermite cubic to round off the interpolation (Improvement with Hermitian Smoothing)
              u = u * u * (3.0f - 2.0f * u);
              v = v * v * (3.0f - 2.0f * v);
              w = w * w * (3.0f - 2.0f * w);
//            A standard trick is to use a Hermite cubic to round off the interpolation (Improvement with Hermitian Smoothing)
//            A standard trick is to use a Hermite cubic to round off the interpolation (Improvement with Hermitian Smoothing)

              int i = static_cast<int>(std::floor(p.x));
              int j = static_cast<int>(std::floor(p.y));
              int k = static_cast<int>(std::floor(p.z));

              float c[2][2][2]{};
//            float c[2][2][2]{};

              for (int di = 0; di < 2; ++di)
              for (int dj = 0; dj < 2; ++dj)
              for (int dk = 0; dk < 2; ++dk)
                  c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
                                                   np.permutationsY[(j + dj) & 255] ^
                                                   np.permutationsZ[(k + dk) & 255]].x;
//                c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
//                                                 np.permutationsY[(j + dj) & 255] ^
//                                                 np.permutationsZ[(k + dk) & 255]].x;

//            trilinear interpolation
//            trilinear interpolation
              float accum = 0.0f;
//            float accum = 0.0f;
              for (int ii = 0; ii < 2; ++ii)
              for (int jj = 0; jj < 2; ++jj)
              for (int kk = 0; kk < 2; ++kk)
                    accum+= ((ii * u) + (1 - ii) * (1 - u))
                          * ((jj * v) + (1 - jj) * (1 - v))
                          * ((kk * w) + (1 - kk) * (1 - w))
                          * c[ii][jj][kk];
//                  accum+= ((ii * u) + (1 - ii) * (1 - u))
//                        * ((jj * v) + (1 - jj) * (1 - v))
//                        * ((kk * w) + (1 - kk) * (1 - w))
//                        * c[ii][jj][kk];
              noisePerlinResult = accum;
//            noisePerlinResult = accum;
//            trilinear interpolation
//            trilinear interpolation
          }
          break;
//        break;


          case NoisePerlinType::SMOOTH_SHIFT_OFF:
//        case NoisePerlinType::SMOOTH_SHIFT_OFF:
          {
              float u = p.x - std::floor(p.x);
              float v = p.y - std::floor(p.y);
              float w = p.z - std::floor(p.z);

              int i = static_cast<int>(std::floor(p.x));
              int j = static_cast<int>(std::floor(p.y));
              int k = static_cast<int>(std::floor(p.z));

              Vec3 c[2][2][2]{};
//            Vec3 c[2][2][2]{};

              for (int di = 0; di < 2; ++di)
              for (int dj = 0; dj < 2; ++dj)
              for (int dk = 0; dk < 2; ++dk)
                  c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
                                                   np.permutationsY[(j + dj) & 255] ^
                                                   np.permutationsZ[(k + dk) & 255]];
//                c[di][dj][dk] = np.randomFloat3s[np.permutationsX[(i + di) & 255] ^
//                                                 np.permutationsY[(j + dj) & 255] ^
//                                                 np.permutationsZ[(k + dk) & 255]];

//            perlin interpolation
//            perlin interpolation
              float uu = u * u * (3.0f - 2.0f * u);
              float vv = v * v * (3.0f - 2.0f * v);
              float ww = w * w * (3.0f - 2.0f * w);
              float accum = 0.0f;
//            float accum = 0.0f;
              for (int ii = 0; ii < 2; ++ii)
              for (int jj = 0; jj < 2; ++jj)
              for (int kk = 0; kk < 2; ++kk)
              {
                  Vec3 weightV{ .x = u - ii, .y = v - jj, .z = w - kk };
//                Vec3 weightV{ .x = u - ii, .y = v - jj, .z = w - kk };
                    accum+= ((ii * uu) + (1 - ii) * (1 - uu))
                          * ((jj * vv) + (1 - jj) * (1 - vv))
                          * ((kk * ww) + (1 - kk) * (1 - ww))
                          * Dot(c[ii][jj][kk], weightV);
//                  accum+= ((ii * uu) + (1 - ii) * (1 - uu))
//                        * ((jj * vv) + (1 - jj) * (1 - vv))
//                        * ((kk * ww) + (1 - kk) * (1 - ww))
//                        * Dot(c[ii][jj][kk], weightV);
              }
              noisePerlinResult = accum;
//            noisePerlinResult = accum;
//            perlin interpolation
//            perlin interpolation
          }
          break;
//        break;


          default:
//        default:
          {

          }
          break;
//        break;
      }

      return noisePerlinResult;
//    return noisePerlinResult;
  }


  inline static float GetTurbulenceValue(const NoisePerlin& np, const Point3& p, std::uint8_t depth)
//inline static float GetTurbulenceValue(const NoisePerlin& np, const Point3& p, std::uint8_t depth)
  {
      float  accum = 0.0f;
//    float  accum = 0.0f;
      Point3 tempP = p;
//    Point3 tempP = p;
      float weight = 1.0f;
//    float weight = 1.0f;
      for (std::uint8_t i = 0; i < depth; ++i)
//    for (std::uint8_t i = 0; i < depth; ++i)
      {
          accum  += weight * GetNoiseValue(np, tempP);
//        accum  += weight * GetNoiseValue(np, tempP);
          weight *= 0.5f;
//        weight *= 0.5f;
          tempP  *= 2.0f;
//        tempP  *= 2.0f;
      }
      return std::fabs(accum);
//    return std::fabs(accum);
  }


    enum class TextureType : std::int8_t
//  enum class TextureType : std::int8_t
{
    SOLID_COLOR = 0,
//  SOLID_COLOR = 0,
    CHECKER_TEXTURE_1 = 1,
//  CHECKER_TEXTURE_1 = 1,
    CHECKER_TEXTURE_2 = 2,
//  CHECKER_TEXTURE_2 = 2,
    IMAGE_TEXTURE = 3,
//  IMAGE_TEXTURE = 3,
    NOISE_PERLIN = 4,
//  NOISE_PERLIN = 4,
};


    struct Texture
//  struct Texture
{
    Color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  Color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
    float scale = 1.0;
//  float scale = 1.0;
    int imageIndex = -1; // png(s) jpg(s) svg(s) exr(s) etc.
//  int imageIndex = -1; // png(s) jpg(s) svg(s) exr(s) etc.
    int noiseIndex = -1; // perlin origin(s) perlin smooth(s)
//  int noiseIndex = -1; // perlin origin(s) perlin smooth(s)
    int oTileTextureIndex = -1;
//  int oTileTextureIndex = -1;
    int eTileTextureIndex = -1;
//  int eTileTextureIndex = -1;
    TextureType type = TextureType::SOLID_COLOR;
//  TextureType type = TextureType::SOLID_COLOR;
};



    static inline struct TexturesDatabase { std::vector<Texture> textures; } texturesDatabase;
//  static inline struct TexturesDatabase { std::vector<Texture> textures; } texturesDatabase;

    static inline struct ImagesDatabase { std::vector<Image> images; Image imageBasedLighting; } imagesDatabase;
//  static inline struct ImagesDatabase { std::vector<Image> images; Image imageBasedLighting; } imagesDatabase;

    static inline struct NoisesDatabase { std::vector<NoisePerlin> noisePerlins; } noisesDatabase;
//  static inline struct NoisesDatabase { std::vector<NoisePerlin> noisePerlins; } noisesDatabase;



    static inline Color3 ExecuteNoisePerlinProcedure(const Texture& texture, float uTextureCoordinate, float vTextureCoordinate, const Point3& point)
//  static inline Color3 ExecuteNoisePerlinProcedure(const Texture& texture, float uTextureCoordinate, float vTextureCoordinate, const Point3& point)
    {
        const NoisePerlin& noisePerlin = noisesDatabase.noisePerlins[texture.noiseIndex];
//      const NoisePerlin& noisePerlin = noisesDatabase.noisePerlins[texture.noiseIndex];
        Color3 noisePerlinProcedureResult;
//      Color3 noisePerlinProcedureResult;
        switch (noisePerlin.noisePerlinProcedureType)
//      switch (noisePerlin.noisePerlinProcedureType)
        {
            case NoisePerlinProcedureType::NOISE_BASIC:
//          case NoisePerlinProcedureType::NOISE_BASIC:
            {
                noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(noisePerlin, point * texture.scale);
//              noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(noisePerlin, point * texture.scale);
            }
            break;
//          break;


            case NoisePerlinProcedureType::NOISE_NORMALIZED:
//          case NoisePerlinProcedureType::NOISE_NORMALIZED:
            {
                noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * (1.0f + GetNoiseValue(noisePerlin, point * texture.scale)) * 0.5f;
//              noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * (1.0f + GetNoiseValue(noisePerlin, point * texture.scale)) * 0.5f;
            }
            break;
//          break;


            case NoisePerlinProcedureType::TURBULENCE_1:
//          case NoisePerlinProcedureType::TURBULENCE_1:
            {
                noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetTurbulenceValue(noisePerlin, point * texture.scale, 7);
//              noisePerlinProcedureResult = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetTurbulenceValue(noisePerlin, point * texture.scale, 7);
            }
            break;
//          break;


            case NoisePerlinProcedureType::TURBULENCE_2:
//          case NoisePerlinProcedureType::TURBULENCE_2:
            {
                noisePerlinProcedureResult = Color3{ .x = 0.5f, .y = 0.5f, .z = 0.5f } * (1.0f + std::sin(texture.scale * point.z + 10.0f * GetTurbulenceValue(noisePerlin, point * texture.scale, 7)));
//              noisePerlinProcedureResult = Color3{ .x = 0.5f, .y = 0.5f, .z = 0.5f } * (1.0f + std::sin(texture.scale * point.z + 10.0f * GetTurbulenceValue(noisePerlin, point * texture.scale, 7)));
            }
            break;
//          break;


            default:
//          default:
            {
                noisePerlinProcedureResult = {};
//              noisePerlinProcedureResult = {};
            }
            break;
//          break;
        }
        return noisePerlinProcedureResult;
//      return noisePerlinProcedureResult;
    }



    static inline Color3 Value(int textureIndex, float uTextureCoordinate, float vTextureCoordinate, const Point3& point)
//  static inline Color3 Value(int textureIndex, float uTextureCoordinate, float vTextureCoordinate, const Point3& point)
{
    const Texture& texture = texturesDatabase.textures[textureIndex];
//  const Texture& texture = texturesDatabase.textures[textureIndex];
    switch (texture.type)
//  switch (texture.type)
    {
        case TextureType::SOLID_COLOR:
//      case TextureType::SOLID_COLOR:
        {
            return texture.albedo;
//          return texture.albedo;
        }
        break;
//      break;


        case TextureType::CHECKER_TEXTURE_1:
//      case TextureType::CHECKER_TEXTURE_1:
        {
            float textureInverseScale = 1.0f / texture.scale;
//          float textureInverseScale = 1.0f / texture.scale;
            int pointX = static_cast<int>(std::floor(textureInverseScale * point.x));
            int pointY = static_cast<int>(std::floor(textureInverseScale * point.y));
            int pointZ = static_cast<int>(std::floor(textureInverseScale * point.z));
            if ((pointX + pointY + pointZ) % 2 == 0)
//          if ((pointX + pointY + pointZ) % 2 == 0)
            {
                return Value(texture.eTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
//              return Value(texture.eTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
            }
            else
            {
                return Value(texture.oTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
//              return Value(texture.oTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
            }
        }
        break;
//      break;


        case TextureType::CHECKER_TEXTURE_2:
//      case TextureType::CHECKER_TEXTURE_2:
        {
            float textureInverseScale = 1.0f / texture.scale;
//          float textureInverseScale = 1.0f / texture.scale;
            int remappedUTextureCoordinate = static_cast<int>(std::floor(textureInverseScale * uTextureCoordinate));
            int remappedVTextureCoordinate = static_cast<int>(std::floor(textureInverseScale * vTextureCoordinate));
            if ((remappedUTextureCoordinate + remappedVTextureCoordinate) % 2 == 0)
//          if ((remappedUTextureCoordinate + remappedVTextureCoordinate) % 2 == 0)
            {
                return Value(texture.eTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
//              return Value(texture.eTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
            }
            else
            {
                return Value(texture.oTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
//              return Value(texture.oTileTextureIndex, uTextureCoordinate, vTextureCoordinate, point);
            }
        }
        break;
//      break;


        case TextureType::IMAGE_TEXTURE:
//      case TextureType::IMAGE_TEXTURE:
        {
#ifdef OLD_IMAGE_SAMPLING
            const Image& image = imagesDatabase.images[texture.imageIndex];
//          const Image& image = imagesDatabase.images[texture.imageIndex];

            Interval rgbRange{ 0.0f, 1.0f };
//          Interval rgbRange{ 0.0f, 1.0f };
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
            uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
//          uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
            vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);
//          vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);

            std::uint16_t imagePixelX = static_cast<std::uint16_t>(uTextureCoordinate * (image.w - 1));
            std::uint16_t imagePixelY = static_cast<std::uint16_t>(vTextureCoordinate * (image.h - 1));

            size_t imagePixelIndex = (static_cast<size_t>(imagePixelY) * image.w + imagePixelX) * 3 /* number of color channels */;
//          size_t imagePixelIndex = (static_cast<size_t>(imagePixelY) * image.w + imagePixelX) * 3 /* number of color channels */;

            return Color3{ .x = image.rgbs[imagePixelIndex + 0],
                           .y = image.rgbs[imagePixelIndex + 1],
                           .z = image.rgbs[imagePixelIndex + 2],
                         };
#endif
#ifdef NEW_IMAGE_SAMPLING
            const Image& image = imagesDatabase.images[texture.imageIndex];
//          const Image& image = imagesDatabase.images[texture.imageIndex];

            Interval rgbRange{ 0.0f, 1.0f };
//          Interval rgbRange{ 0.0f, 1.0f };
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
            uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
//          uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
            vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);
//          vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);

            float imagePixelX = uTextureCoordinate * (image.w - 1);
            float imagePixelY = vTextureCoordinate * (image.h - 1);

            return SampleRGB2LinearInterpolation(image.rgbs, image.w, image.h, imagePixelX, imagePixelY);
//          return SampleRGB2LinearInterpolation(image.rgbs, image.w, image.h, imagePixelX, imagePixelY);
#endif
        }
        break;
//      break;


        case TextureType::NOISE_PERLIN:
//      case TextureType::NOISE_PERLIN:
        {
            return ExecuteNoisePerlinProcedure(texture, uTextureCoordinate, vTextureCoordinate, point);
//          return ExecuteNoisePerlinProcedure(texture, uTextureCoordinate, vTextureCoordinate, point);
        }
        break;
//      break;


        default:
//      default:
        {
            return {};
//          return {};
        }
        break;
//      break;
    }
    return {};
//  return {};
}



struct Ray
{
    Vec3 ori;
    Vec3 dir;
    float time;
//  float time;

    Point3 Marching(float t) const { return ori + dir * t; }
//  point3 Marching(float t) const { return ori + dir * t; }
};



    static inline Point3 Marching(Point3 ori, Vec3 dir, float t) { return ori + dir * t; }
//  static inline Point3 Marching(Point3 ori, Vec3 dir, float t) { return ori + dir * t; }



static bool HitAABB(const Ray& ray, Interval rayT, const AABB2D& aabb2d)
{
    return true;
}
static bool HitAABB(const Ray& ray, Interval rayT, const AABB3D& aabb3d)
{
    const float& rayDirAxisXInverse = 1.0f / ray.dir.x;
    const float& tX0 = (aabb3d.intervalAxisX.min - ray.ori.x) * rayDirAxisXInverse;
    const float& tX1 = (aabb3d.intervalAxisX.max - ray.ori.x) * rayDirAxisXInverse;
    if (tX0 < tX1)
    {
        if (tX0 > rayT.min) { rayT.min = tX0; }
        if (tX1 < rayT.max) { rayT.max = tX1; }
    }
    else
    {
        if (tX1 > rayT.min) { rayT.min = tX1; }
        if (tX0 < rayT.max) { rayT.max = tX0; }
    }
    if (rayT.max <= rayT.min)
    {
        return false;
    }

    const float& rayDirAxisYInverse = 1.0f / ray.dir.y;
    const float& tY0 = (aabb3d.intervalAxisY.min - ray.ori.y) * rayDirAxisYInverse;
    const float& tY1 = (aabb3d.intervalAxisY.max - ray.ori.y) * rayDirAxisYInverse;
    if (tY0 < tY1)
    {
        if (tY0 > rayT.min) { rayT.min = tY0; }
        if (tY1 < rayT.max) { rayT.max = tY1; }
    }
    else
    {
        if (tY1 > rayT.min) { rayT.min = tY1; }
        if (tY0 < rayT.max) { rayT.max = tY0; }
    }
    if (rayT.max <= rayT.min)
    {
        return false;
    }

    const float& rayDirAxisZInverse = 1.0f / ray.dir.z;
    const float& tZ0 = (aabb3d.intervalAxisZ.min - ray.ori.z) * rayDirAxisZInverse;
    const float& tZ1 = (aabb3d.intervalAxisZ.max - ray.ori.z) * rayDirAxisZInverse;
    if (tZ0 < tZ1)
    {
        if (tZ0 > rayT.min) { rayT.min = tZ0; }
        if (tZ1 < rayT.max) { rayT.max = tZ1; }
    }
    else
    {
        if (tZ1 > rayT.min) { rayT.min = tZ1; }
        if (tZ0 < rayT.max) { rayT.max = tZ0; }
    }
    if (rayT.max <= rayT.min)
    {
        return false;
    }


    return true;
}



    inline static Vec3 GenRandom(                    ) { return Vec3 { Random(        ), Random(        ), Random(        ) }; }
    inline static Vec3 GenRandom(float min, float max) { return Vec3 { Random(min, max), Random(min, max), Random(min, max) }; }
    inline static Vec3 GenRandomUnitVector()
//  inline static Vec3 GenRandomUnitVector()
    {
//  Rejection Sampling Method (References: Las-Vegas algorithm vs Monte-Carlo algorithm)
//  Rejection Sampling Method (References: Las-Vegas algorithm vs Monte-Carlo algorithm)
/*
        while (true)
        {
            const Vec3& p = GenRandom(-1.0f, +1.0f);
//          const Vec3& p = GenRandom(-1.0f, +1.0f);
            const float& pLengthSquared = p.LengthSquared();
//          const float& pLengthSquared = p.LengthSquared();
            if (pLengthSquared <= 1.0000f && pLengthSquared > 1e-4f)
//          if (pLengthSquared <= 1.0000f && pLengthSquared > 1e-4f)
            {
                return p / std::sqrt(pLengthSquared);
//              return p / std::sqrt(pLengthSquared);
            }
        }
*/
//  Archimedes' Method
//  Archimedes' Method

        float theta = Random(+0.0f, +2.0f * std::numbers::pi_v<float>); // Longitude          (uniform)
//      float theta = Random(+0.0f, +2.0f * std::numbers::pi_v<float>); // Longitude          (uniform)
        float z     = Random(-1.0f, +1.0f                            ); // Height on cylinder (uniform)
//      float z     = Random(-1.0f, +1.0f                            ); // Height on cylinder (uniform)

        float r = std::sqrt(1.0f - z * z); // Radius of the circle at this height
//      float r = std::sqrt(1.0f - z * z); // Radius of the circle at this height

        float x = r * std::cos(theta);
//      float x = r * std::cos(theta);
        float y = r * std::sin(theta);
//      float y = r * std::sin(theta);

        return { x, y, z };
//      return { x, y, z };

    }
    inline static Vec3 GenRandomUnitVectorOnHemisphere(const Vec3& normal)
//  inline static Vec3 GenRandomUnitVectorOnHemisphere(const Vec3& normal)
    {
        const Vec3& randomUnitVector = GenRandomUnitVector();
//      const Vec3& randomUnitVector = GenRandomUnitVector();
        if (Dot(randomUnitVector, normal) > 0.0f)
//      if (Dot(randomUnitVector, normal) > 0.0f)
        {
            return  randomUnitVector;
        }
        else
        {
            return -randomUnitVector;
        }
    }
    inline static Vec3 GenRandomPointInsideNormalizedDisk()
    {
//  rejection sampling
//  rejection sampling
/*
        while (true)
//      while (true)
        {
            Point3 point { .x = Random(-1.0f , +1.0f), .y = Random(-1.0f , +1.0f), .z = 0.0f };
//          Point3 point { .x = Random(-1.0f , +1.0f), .y = Random(-1.0f , +1.0f), .z = 0.0f };
            if (point.LengthSquared() < 1.0f)
//          if (point.LengthSquared() < 1.0f)
            {
                return point;
//              return point;
            }
        }
*/
//  polar sampling
//  polar sampling
        float r = std::sqrt(Random()); // radius (sqrt ensures uniform distribution)
//      float r = std::sqrt(Random()); // radius (sqrt ensures uniform distribution)
        float theta = Random(0.0f, 2.0f * std::numbers::pi_v<float>); // angle in radians
//      float theta = Random(0.0f, 2.0f * std::numbers::pi_v<float>); // angle in radians
        return { r * std::cos(theta), r * std::sin(theta), 0.0f };
//      return { r * std::cos(theta), r * std::sin(theta), 0.0f };
    }
    inline static Vec3 DefocusDiskSample(const Point3& diskCenter, const Vec3& defocusDiskRadiusU, const Vec3& defocusDiskRadiusV)
//  inline static Vec3 DefocusDiskSample(const Point3& diskCenter, const Vec3& defocusDiskRadiusU, const Vec3& defocusDiskRadiusV)
    {
        Point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
//      Point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
        return diskCenter + randomPointInsideNormalizedDisk.x * defocusDiskRadiusU
                          + randomPointInsideNormalizedDisk.y * defocusDiskRadiusV;
    }

    inline static Vec3 Reflect(const Vec3& incomingVector, const Vec3& normal) { return incomingVector - 2.0f * Dot(incomingVector, normal) * normal; }
//  inline static Vec3 Reflect(const Vec3& incomingVector, const Vec3& normal) { return incomingVector - 2.0f * Dot(incomingVector, normal) * normal; }

    inline static Vec3 Refract(const Vec3& incomingVector, const Vec3& normal, float ratioOfEtaiOverEtat)
    {
        const float& cosTheta = std::fminf(Dot(-incomingVector, normal), 1.0f);
//      const float& cosTheta = std::fminf(Dot(-incomingVector, normal), 1.0f);
        const Vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
//      const Vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
        const Vec3& refractedRayParallel = -std::sqrtf(std::fabsf(1.0f - refractedRayPerpendicular.LengthSquared())) * normal;
//      const Vec3& refractedRayParallel = -std::sqrtf(std::fabsf(1.0f - refractedRayPerpendicular.LengthSquared())) * normal;
        return refractedRayPerpendicular + refractedRayParallel;
//      return refractedRayPerpendicular + refractedRayParallel;
    }

    inline static float Reflectance(float cosine, float ratioOfEtaiOverEtat)
    {
        // Use Schlick's approximation for reflectance.
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - ratioOfEtaiOverEtat) / (1.0f + ratioOfEtaiOverEtat);
//      float r0 = (1.0f - ratioOfEtaiOverEtat) / (1.0f + ratioOfEtaiOverEtat);
        r0 = r0 * r0;
//      r0 = r0 * r0;
        float temp = 1.0f - cosine;
//      float temp = 1.0f - cosine;
        return r0 + (1.0f - r0) * temp * temp * temp * temp * temp;
//      return r0 + (1.0f - r0) * temp * temp * temp * temp * temp;
//      return r0 + (1.0f - r0) * std::powf((1.0f - cosine), 5.0f);
//      return r0 + (1.0f - r0) * std::powf((1.0f - cosine), 5.0f);
    }



    enum class MaterialType       : std::uint8_t
//  enum class MaterialType       : std::uint8_t
{
    LambertianDiffuseReflectance1 = 0,
//  LambertianDiffuseReflectance1 = 0,
    LambertianDiffuseReflectance2 = 1,
//  LambertianDiffuseReflectance2 = 1,
    Metal                         = 2,
//  Metal                         = 2,
    MetalFuzzy1                   = 3,
//  MetalFuzzy1                   = 3,
    MetalFuzzy2                   = 4,
//  MetalFuzzy2                   = 4,
    Dielectric                    = 5,
//  Dielectric                    = 5,
    LightDiffuse                  = 6,
//  LightDiffuse                  = 6,
    LightMetalic                  = 7,
//  LightMetalic                  = 7,
    Isotropic1                    = 8,
//  Isotropic1                    = 8,
    Isotropic2                    = 9,
//  Isotropic2                    = 9,
    FresnelBlendedDielectricGlossyDiffuse1 = 10, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
//  FresnelBlendedDielectricGlossyDiffuse1 = 10, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
    FresnelBlendedDielectricGlossyDiffuse2 = 11, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
//  FresnelBlendedDielectricGlossyDiffuse2 = 11, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
    FresnelBlendedDielectricGlossyDiffuse3 = 12, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
//  FresnelBlendedDielectricGlossyDiffuse3 = 12, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
    FresnelBlendedDielectricGlossyDiffuse4 = 13, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
//  FresnelBlendedDielectricGlossyDiffuse4 = 13, // render opaque dielectrics like polished marbles, polished stones, or dense plastics, etc.
    ThinFilmDielectric1                    = 14,
//  ThinFilmDielectric1                    = 14,
    ThinFilmDielectric2                    = 15,
//  ThinFilmDielectric2                    = 15,
    ThinFilmDielectric3                    = 16,
//  ThinFilmDielectric3                    = 16,
    ThinFilmDielectric4                    = 17,
//  ThinFilmDielectric4                    = 17,
    ThinFilmMetal                          = 18,
//  ThinFilmMetal                          = 18,
    SubsurfaceRandomWalk1                  = 19, // homogeneous SSS via one-bounce random walk
//  SubsurfaceRandomWalk1                  = 19, // homogeneous SSS via one-bounce random walk
    SubsurfaceRandomWalk2                  = 20, // homogeneous SSS via one-bounce random walk
//  SubsurfaceRandomWalk2                  = 20, // homogeneous SSS via one-bounce random walk
};


    enum class MaterialDielectric : std::uint8_t
//  enum class MaterialDielectric : std::uint8_t
{
    NOTHING = 0,
//  NOTHING = 0,
    AIR     = 1,
//  AIR     = 1,
    WATER   = 2,
//  WATER   = 2,
    SKIN    = 3,
//  SKIN    = 3,
    GLASS   = 4,
//  GLASS   = 4,
    MARBLE  = 5,
//  MARBLE  = 5,
    DIAMOND = 6,
//  DIAMOND = 6,
};


constexpr inline static float GetRefractionIndex(MaterialDielectric materialDielectric)
{
    switch ( materialDielectric )
    {
        case MaterialDielectric::NOTHING: return 1.000000f; break;
//      case MaterialDielectric::NOTHING: return 1.000000f; break;
        case MaterialDielectric::AIR    : return 1.000293f; break;
//      case MaterialDielectric::AIR    : return 1.000293f; break;
        case MaterialDielectric::WATER  : return 1.333000f; break;
//      case MaterialDielectric::WATER  : return 1.333000f; break;
        case MaterialDielectric::SKIN   : return 1.400000f; break;
//      case MaterialDielectric::SKIN   : return 1.400000f; break;
        case MaterialDielectric::GLASS  : return 1.500000f; break;
//      case MaterialDielectric::GLASS  : return 1.500000f; break;
        case MaterialDielectric::MARBLE : return 1.550000f; break;
//      case MaterialDielectric::MARBLE : return 1.550000f; break;
        case MaterialDielectric::DIAMOND: return 2.400000f; break;
//      case MaterialDielectric::DIAMOND: return 2.400000f; break;
                                 default: return 0.000000f; break;
//                               default: return 0.000000f; break;
    }
}


    struct Material
//  struct Material
{
    Color3 absorption; Color3 scattering; float asymmetryParameter /* range: [-1 ; +1] | skin ~ 0.8-0.95 */; float layer1Roughness; float layer1Thickness; float layer0IOR; float layer1IOR; float layer2IOR; std::uint8_t textureIndex; MaterialType materialType;
//  Color3 absorption; Color3 scattering; float asymmetryParameter /* range: [-1 ; +1] | skin ~ 0.8-0.95 */; float layer1Roughness; float layer1Thickness; float layer0IOR; float layer1IOR; float layer2IOR; std::uint8_t textureIndex; MaterialType materialType;
};

    struct MaterialScatteredResult
//  struct MaterialScatteredResult
{
    Ray scatteredRay; Color3 attenuation; Color3 emission; bool isScattered;
//  Ray scatteredRay; Color3 attenuation; Color3 emission; bool isScattered;
};

    enum class GeometryType : std::uint8_t
//  enum class GeometryType : std::uint8_t
{
//  SPHERE = 0,
    SPHERE = 0,
//  SPHERE = 0,
//  PRIMITIVE = 1,
    PRIMITIVE = 1,
//  PRIMITIVE = 1,
//  IMPLICIT_SURFACE = 2,
    IMPLICIT_SURFACE = 2,
//  IMPLICIT_SURFACE = 2,
};

    struct SDFHitResult
//  struct SDFHitResult
{
    float distance; std::uint8_t materialIndex;
//  float distance; std::uint8_t materialIndex;
};

    struct Geometry
//  struct Geometry
{
//  with triangle: vertex 0 normal == vertex 1 normal == vertex 2 normal => @flat@ shading
//  with triangle: vertex 0 normal == vertex 1 normal == vertex 2 normal => @flat@ shading
//  with triangle: vertex 0 normal != vertex 1 normal != vertex 2 normal => smooth shading
//  with triangle: vertex 0 normal != vertex 1 normal != vertex 2 normal => smooth shading
    union { struct Sphere { Point3 center; float radius; } sphere; struct Primitive { Point3 vertex0; Point3 vertex1; Point3 vertex2; Vec3 vertex0FrontFaceNormal; Vec3 vertex1FrontFaceNormal; Vec3 vertex2FrontFaceNormal; float frontFaceVertex0U; float frontFaceVertex0V; float frontFaceVertex1U; float frontFaceVertex1V; float frontFaceVertex2U; float frontFaceVertex2V; float backFaceVertex0U; float backFaceVertex0V; float backFaceVertex1U; float backFaceVertex1V; float backFaceVertex2U; float backFaceVertex2V; bool perVertexFrontFaceNormalAvailable; } primitive; struct ImplicitSurface { int maxSteps; float maxMarchingDistance; float hitEpsilon; std::uint8_t sdfIndex; } implicitSurface; }; AABB3D aabb3d; Vec3 movingDirection; std::uint8_t materialIndex; GeometryType geometryType;
//  union { struct Sphere { Point3 center; float radius; } sphere; struct Primitive { Point3 vertex0; Point3 vertex1; Point3 vertex2; Vec3 vertex0FrontFaceNormal; Vec3 vertex1FrontFaceNormal; Vec3 vertex2FrontFaceNormal; float frontFaceVertex0U; float frontFaceVertex0V; float frontFaceVertex1U; float frontFaceVertex1V; float frontFaceVertex2U; float frontFaceVertex2V; float backFaceVertex0U; float backFaceVertex0V; float backFaceVertex1U; float backFaceVertex1V; float backFaceVertex2U; float backFaceVertex2V; bool perVertexFrontFaceNormalAvailable; } primitive; struct ImplicitSurface { int maxSteps; float maxMarchingDistance; float hitEpsilon; std::uint8_t sdfIndex; } implicitSurface; }; AABB3D aabb3d; Vec3 movingDirection; std::uint8_t materialIndex; GeometryType geometryType;

};






    static inline struct MaterialsDatabase { std::vector<Material> materials; } materialsDatabase;
//  static inline struct MaterialsDatabase { std::vector<Material> materials; } materialsDatabase;






    inline static float SdfSphere(const Point3& p, float radius)
//  inline static float SdfSphere(const Point3& p, float radius)
    {
        return p.Length() - radius;
//      return p.Length() - radius;
    }
    inline static float SdfTorus(const Point3& p, const Vec2& t)
//  inline static float SdfTorus(const Point3& p, const Vec2& t)
    {
        Vec2 q = { .x = Vec2{ p.x, p.z }.Length() - t.x, .y = p.y };
//      Vec2 q = { .x = Vec2{ p.x, p.z }.Length() - t.x, .y = p.y };
        return q.Length() - t.y;
//      return q.Length() - t.y;
    }
    inline static float SdfSmoothUnion(float d1, float d2, float k)
//  inline static float SdfSmoothUnion(float d1, float d2, float k)
    {
        float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
//      float h = std::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
        return BlendLinear(d2, d1, h) - k * h * (1.0f - h);
//      return BlendLinear(d2, d1, h) - k * h * (1.0f - h);
    }
    FastNoiseLite* noise;
//  FastNoiseLite* noise;

    // p: current point in space (vec3)
    // p: current point in space (vec3)
    // frequency: how "zoomed in" the noise is. Higher frequency = smaller, more frequent features.
    // frequency: how "zoomed in" the noise is. Higher frequency = smaller, more frequent features.
    // amplitude: how high the mountains are. The maximum displacement height.
    // amplitude: how high the mountains are. The maximum displacement height.
    inline static float TerrainSDF(const Vec3& p, float frequency = 1.0f, float amplitude = 15.0f)
    {
        // 1. Get the 2D noise value for the current XZ position.
        // 1. Get the 2D noise value for the current XZ position.
        // We multiply the position by frequency to scale the noise.
        // We multiply the position by frequency to scale the noise.
        float noiseValue = noise->GetNoise(p.x * frequency, p.z * frequency);
//      float noiseValue = noise->GetNoise(p.x * frequency, p.z * frequency);

        // 2. The height of the terrain at this XZ position.
        // 2. The height of the terrain at this XZ position.
        float terrainHeight = noiseValue * amplitude;
//      float terrainHeight = noiseValue * amplitude;

        // 3. The SDF is the vertical distance to that height.
        // 3. The SDF is the vertical distance to that height.
        return p.y - terrainHeight;
//      return p.y - terrainHeight;
    }
    inline static float SdfMetaballs(const Point3& p)
//  inline static float SdfMetaballs(const Point3& p)
    {
//        Point3 p1 = p - Point3{ -8.0f, 0.0f, 0.0f };
////      Point3 p1 = p - Point3{ -8.0f, 0.0f, 0.0f };
//        Point3 p2 = p - Point3{ +8.0f, 0.0f, 0.0f };
////      Point3 p2 = p - Point3{ +8.0f, 0.0f, 0.0f };
//        float sphere1Dist = SdfSphere(p1, 10.00f);
////      float sphere1Dist = SdfSphere(p1, 10.00f);
//        float sphere2Dist = SdfSphere(p2, 10.00f);
////      float sphere2Dist = SdfSphere(p2, 10.00f);
//        return SdfSmoothUnion(sphere1Dist, sphere2Dist, 0.8f);
////      return SdfSmoothUnion(sphere1Dist, sphere2Dist, 0.8f);
        return TerrainSDF(sdf3d::OpTranslate(p, { 0, -15, 0 }), 1.0f, 15.0f);
//      return TerrainSDF(sdf3d::OpTranslate(p, { 0, -15, 0 }), 1.0f, 15.0f);
/*
        return sdf3d::SDFTorus(sdf3d::OpTwist(p), { 9,5 });
//      return sdf3d::SDFTorus(sdf3d::OpTwist(p), { 9,5 });
*/
    }
    inline static SDFHitResult SDFIndex0(const Point3& p)
//  inline static SDFHitResult SDFIndex0(const Point3& p)
    {
        SDFHitResult sdfHitResult{};
//      SDFHitResult sdfHitResult{};
        Point3 transformedP = sdf3d::OpTranslate(p, { 0, -15, 0 });
//      Point3 transformedP = sdf3d::OpTranslate(p, { 0, -15, 0 });
        sdfHitResult.distance = TerrainSDF(transformedP, 1.0f, 15.0f);
//      sdfHitResult.distance = TerrainSDF(transformedP, 1.0f, 15.0f);
        if (transformedP.y <= -2.0f)
//      if (transformedP.y <= -2.0f)
        {
            sdfHitResult.materialIndex = 3;
//          sdfHitResult.materialIndex = 3;
        }
        else
        {
            sdfHitResult.materialIndex = 4;
//          sdfHitResult.materialIndex = 4;
        }
        return sdfHitResult;
//      return sdfHitResult;
    }
    inline static Vec3 CalculateImplicitNormal(float (*sdf)(const Point3& p), const Point3& p)
//  inline static Vec3 CalculateImplicitNormal(float (*sdf)(const Point3& p), const Point3& p)
    {
        constexpr float epsilon = 1e-4f;
//      constexpr float epsilon = 1e-4f;
        Vec3 normal =
//      Vec3 normal =
        {
            .x = sdf(Point3{p.x           + epsilon, p.y, p.z}) - sdf(Point3{p.x           - epsilon, p.y, p.z}),
//          .x = sdf(Point3{p.x           + epsilon, p.y, p.z}) - sdf(Point3{p.x           - epsilon, p.y, p.z}),
            .y = sdf(Point3{p.x, p.y      + epsilon     , p.z}) - sdf(Point3{p.x, p.y      - epsilon     , p.z}),
//          .y = sdf(Point3{p.x, p.y      + epsilon     , p.z}) - sdf(Point3{p.x, p.y      - epsilon     , p.z}),
            .z = sdf(Point3{p.x, p.y, p.z + epsilon          }) - sdf(Point3{p.x, p.y, p.z - epsilon          }),
//          .z = sdf(Point3{p.x, p.y, p.z + epsilon          }) - sdf(Point3{p.x, p.y, p.z - epsilon          }),
        };
        return Normalize(normal);
//      return Normalize(normal);
    }
    using SDFFunctionPointer = SDFHitResult(*)(const Point3& p);
//  using SDFFunctionPointer = SDFHitResult(*)(const Point3& p);
    inline static Vec3 CalculateImplicitNormal(const SDFFunctionPointer& sdf, const Point3& p)
//  inline static Vec3 CalculateImplicitNormal(const SDFFunctionPointer& sdf, const Point3& p)
    {
        constexpr float epsilon = 1e-4f;
//      constexpr float epsilon = 1e-4f;
        Vec3 normal =
//      Vec3 normal =
        {
            .x = sdf(Point3{p.x           + epsilon, p.y, p.z}).distance - sdf(Point3{p.x           - epsilon, p.y, p.z}).distance,
//          .x = sdf(Point3{p.x           + epsilon, p.y, p.z}).distance - sdf(Point3{p.x           - epsilon, p.y, p.z}).distance,
            .y = sdf(Point3{p.x, p.y      + epsilon     , p.z}).distance - sdf(Point3{p.x, p.y      - epsilon     , p.z}).distance,
//          .y = sdf(Point3{p.x, p.y      + epsilon     , p.z}).distance - sdf(Point3{p.x, p.y      - epsilon     , p.z}).distance,
            .z = sdf(Point3{p.x, p.y, p.z + epsilon          }).distance - sdf(Point3{p.x, p.y, p.z - epsilon          }).distance,
//          .z = sdf(Point3{p.x, p.y, p.z + epsilon          }).distance - sdf(Point3{p.x, p.y, p.z - epsilon          }).distance,
        };
        return Normalize(normal);
//      return Normalize(normal);
    }
    static inline struct SDFsDatabase { SDFFunctionPointer sdfs[1] = { SDFIndex0, }; } sdfsDatabase;
//  static inline struct SDFsDatabase { SDFFunctionPointer sdfs[1] = { SDFIndex0, }; } sdfsDatabase;






    inline static bool IsStationary(Geometry& g)
//  inline static bool IsStationary(Geometry& g)
{
    return g.movingDirection.x == 0.0f
        && g.movingDirection.y == 0.0f
        && g.movingDirection.z == 0.0f;
}

    inline static void CalculateAABB3D(Geometry& g)
//  inline static void CalculateAABB3D(Geometry& g)
{
    if (IsStationary(g))
//  if (IsStationary(g))
    {
        switch (g.geometryType)
//      switch (g.geometryType)
        {
        case GeometryType::SPHERE:
//      case GeometryType::SPHERE:
            {
                g.aabb3d.intervalAxisX.min = g.sphere.center.x - g.sphere.radius;
                g.aabb3d.intervalAxisX.max = g.sphere.center.x + g.sphere.radius;
                g.aabb3d.intervalAxisY.min = g.sphere.center.y - g.sphere.radius;
                g.aabb3d.intervalAxisY.max = g.sphere.center.y + g.sphere.radius;
                g.aabb3d.intervalAxisZ.min = g.sphere.center.z - g.sphere.radius;
                g.aabb3d.intervalAxisZ.max = g.sphere.center.z + g.sphere.radius;
            }
            break;
//          break;
        case GeometryType::PRIMITIVE:
//      case GeometryType::PRIMITIVE:
            {
                constexpr float padding = 1e-4f; // COULD BE LOWER
//              constexpr float padding = 1e-4f; // COULD BE LOWER
                g.aabb3d.intervalAxisX.min = std::fminf(g.primitive.vertex0.x, std::fminf(g.primitive.vertex1.x, g.primitive.vertex2.x)) - padding;
                g.aabb3d.intervalAxisY.min = std::fminf(g.primitive.vertex0.y, std::fminf(g.primitive.vertex1.y, g.primitive.vertex2.y)) - padding;
                g.aabb3d.intervalAxisZ.min = std::fminf(g.primitive.vertex0.z, std::fminf(g.primitive.vertex1.z, g.primitive.vertex2.z)) - padding;
                g.aabb3d.intervalAxisX.max = std::fmaxf(g.primitive.vertex0.x, std::fmaxf(g.primitive.vertex1.x, g.primitive.vertex2.x)) + padding;
                g.aabb3d.intervalAxisY.max = std::fmaxf(g.primitive.vertex0.y, std::fmaxf(g.primitive.vertex1.y, g.primitive.vertex2.y)) + padding;
                g.aabb3d.intervalAxisZ.max = std::fmaxf(g.primitive.vertex0.z, std::fmaxf(g.primitive.vertex1.z, g.primitive.vertex2.z)) + padding;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }
    else
    {
        switch (g.geometryType)
//      switch (g.geometryType)
        {
        case GeometryType::SPHERE:
//      case GeometryType::SPHERE:
            {
                const Point3& destinationCenter = Marching(g.sphere.center, g.movingDirection, 1.0f);
//              const Point3& destinationCenter = Marching(g.sphere.center, g.movingDirection, 1.0f);
                g.aabb3d.intervalAxisX.min = std::fminf(g.sphere.center.x, destinationCenter.x) - g.sphere.radius;
                g.aabb3d.intervalAxisX.max = std::fmaxf(g.sphere.center.x, destinationCenter.x) + g.sphere.radius;
                g.aabb3d.intervalAxisY.min = std::fminf(g.sphere.center.y, destinationCenter.y) - g.sphere.radius;
                g.aabb3d.intervalAxisY.max = std::fmaxf(g.sphere.center.y, destinationCenter.y) + g.sphere.radius;
                g.aabb3d.intervalAxisZ.min = std::fminf(g.sphere.center.z, destinationCenter.z) - g.sphere.radius;
                g.aabb3d.intervalAxisZ.max = std::fmaxf(g.sphere.center.z, destinationCenter.z) + g.sphere.radius;
            }
            break;
//          break;
        case GeometryType::PRIMITIVE:
//      case GeometryType::PRIMITIVE:
            {
                constexpr float padding = 1e-4f; // COULD BE LOWER
//              constexpr float padding = 1e-4f; // COULD BE LOWER
                const Point3& destinationVertex0 = Marching(g.primitive.vertex0, g.movingDirection, 1.0f);
                const Point3& destinationVertex1 = Marching(g.primitive.vertex1, g.movingDirection, 1.0f);
                const Point3& destinationVertex2 = Marching(g.primitive.vertex2, g.movingDirection, 1.0f);
                float minXOri = std::fminf(g.primitive.vertex0.x, std::fminf(g.primitive.vertex1.x, g.primitive.vertex2.x));
                float minYOri = std::fminf(g.primitive.vertex0.y, std::fminf(g.primitive.vertex1.y, g.primitive.vertex2.y));
                float minZOri = std::fminf(g.primitive.vertex0.z, std::fminf(g.primitive.vertex1.z, g.primitive.vertex2.z));
                float maxXOri = std::fmaxf(g.primitive.vertex0.x, std::fmaxf(g.primitive.vertex1.x, g.primitive.vertex2.x));
                float maxYOri = std::fmaxf(g.primitive.vertex0.y, std::fmaxf(g.primitive.vertex1.y, g.primitive.vertex2.y));
                float maxZOri = std::fmaxf(g.primitive.vertex0.z, std::fmaxf(g.primitive.vertex1.z, g.primitive.vertex2.z));
                float minXDes = std::fminf(destinationVertex0.x, std::fminf(destinationVertex1.x, destinationVertex2.x));
                float minYDes = std::fminf(destinationVertex0.y, std::fminf(destinationVertex1.y, destinationVertex2.y));
                float minZDes = std::fminf(destinationVertex0.z, std::fminf(destinationVertex1.z, destinationVertex2.z));
                float maxXDes = std::fmaxf(destinationVertex0.x, std::fmaxf(destinationVertex1.x, destinationVertex2.x));
                float maxYDes = std::fmaxf(destinationVertex0.y, std::fmaxf(destinationVertex1.y, destinationVertex2.y));
                float maxZDes = std::fmaxf(destinationVertex0.z, std::fmaxf(destinationVertex1.z, destinationVertex2.z));
                g.aabb3d.intervalAxisX.min = std::fminf(minXOri, minXDes) - padding;
                g.aabb3d.intervalAxisX.max = std::fmaxf(maxXOri, maxXDes) + padding;
                g.aabb3d.intervalAxisY.min = std::fminf(minYOri, minYDes) - padding;
                g.aabb3d.intervalAxisY.max = std::fmaxf(maxYOri, maxYDes) + padding;
                g.aabb3d.intervalAxisZ.min = std::fminf(minZOri, minZDes) - padding;
                g.aabb3d.intervalAxisZ.max = std::fmaxf(maxZOri, maxZDes) + padding;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }
}

    inline static void CalculateAABB3D(std::vector<Geometry>& geometries, AABB3D& aabb3d)
//  inline static void CalculateAABB3D(std::vector<Geometry>& geometries, AABB3D& aabb3d)
{
    for (Geometry& g : geometries)
//  for (Geometry& g : geometries)
    {
        CalculateAABB3D(g);
//      CalculateAABB3D(g);
        aabb3d.intervalAxisX.min = std::fminf(g.aabb3d.intervalAxisX.min, aabb3d.intervalAxisX.min);
        aabb3d.intervalAxisX.max = std::fmaxf(g.aabb3d.intervalAxisX.max, aabb3d.intervalAxisX.max);
        aabb3d.intervalAxisY.min = std::fminf(g.aabb3d.intervalAxisY.min, aabb3d.intervalAxisY.min);
        aabb3d.intervalAxisY.max = std::fmaxf(g.aabb3d.intervalAxisY.max, aabb3d.intervalAxisY.max);
        aabb3d.intervalAxisZ.min = std::fminf(g.aabb3d.intervalAxisZ.min, aabb3d.intervalAxisZ.min);
        aabb3d.intervalAxisZ.max = std::fmaxf(g.aabb3d.intervalAxisZ.max, aabb3d.intervalAxisZ.max);
    }
}

    struct RayHitResult
//  struct RayHitResult
{
    Point3 at; Vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace; std::uint8_t materialIndex;
//  Point3 at; Vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace; std::uint8_t materialIndex;
};










    static inline float FresnelAmplitudeS(float iIOR, float tIOR, float iCosTheta, float tCosTheta)
//  static inline float FresnelAmplitudeS(float iIOR, float tIOR, float iCosTheta, float tCosTheta)
    {
        if (iCosTheta < 0.0f) return 0.0f;
    //  if (iCosTheta < 0.0f) return 0.0f;
        float iIORMultiplyICosTheta = iIOR * iCosTheta;
    //  float iIORMultiplyICosTheta = iIOR * iCosTheta;
        float tIORMultiplyTCosTheta = tIOR * tCosTheta;
    //  float tIORMultiplyTCosTheta = tIOR * tCosTheta;
        return (iIORMultiplyICosTheta - tIORMultiplyTCosTheta) / (iIORMultiplyICosTheta + tIORMultiplyTCosTheta);
    //  return (iIORMultiplyICosTheta - tIORMultiplyTCosTheta) / (iIORMultiplyICosTheta + tIORMultiplyTCosTheta);
    }
    static inline float FresnelAmplitudeP(float iIOR, float tIOR, float iCosTheta, float tCosTheta)
//  static inline float FresnelAmplitudeP(float iIOR, float tIOR, float iCosTheta, float tCosTheta)
    {
        if (iCosTheta < 0.0f) return 0.0f;
    //  if (iCosTheta < 0.0f) return 0.0f;
        float tIORMultiplyICosTheta = tIOR * iCosTheta;
    //  float tIORMultiplyICosTheta = tIOR * iCosTheta;
        float iIORMultiplyTCosTheta = iIOR * tCosTheta;
    //  float iIORMultiplyTCosTheta = iIOR * tCosTheta;
        return (tIORMultiplyICosTheta - iIORMultiplyTCosTheta) / (tIORMultiplyICosTheta + iIORMultiplyTCosTheta);
    //  return (tIORMultiplyICosTheta - iIORMultiplyTCosTheta) / (tIORMultiplyICosTheta + iIORMultiplyTCosTheta);
    }
    static inline float ThinFilmReflectanceForWavelength(float iCosThetaFromExternalLayerToFilmLayer /* cosine of the angle of incidence in the external medium */, float filmThickness /* film thickness in nanometers */, float lambda /* wavelength of light in nanometers */, float externalLayerIOR /* IOR of the medium outside the film */, float filmLayerIOR /* IOR of the film */, float substrateLayerIOR /* IOR of the material beneath the film */)
//  static inline float ThinFilmReflectanceForWavelength(float iCosThetaFromExternalLayerToFilmLayer /* cosine of the angle of incidence in the external medium */, float filmThickness /* film thickness in nanometers */, float lambda /* wavelength of light in nanometers */, float externalLayerIOR /* IOR of the medium outside the film */, float filmLayerIOR /* IOR of the film */, float substrateLayerIOR /* IOR of the material beneath the film */)
    {
        // Avoid division by zero at grazing
        // Avoid division by zero at grazing
        if (iCosThetaFromExternalLayerToFilmLayer < 1e-4f)
            iCosThetaFromExternalLayerToFilmLayer = 1e-4f;

        // Angle in film (Snell's Law: externalLayerIOR * iSinThetaFromExternalLayerToFilmLayer = filmLayerIOR * tSinThetaFromExternalLayerToFilmLayer)
        // Angle in film (Snell's Law: externalLayerIOR * iSinThetaFromExternalLayerToFilmLayer = filmLayerIOR * tSinThetaFromExternalLayerToFilmLayer)
        float sqISinThetaFromExternalLayerToFilmLayer = 1.0f - iCosThetaFromExternalLayerToFilmLayer
                                                             * iCosThetaFromExternalLayerToFilmLayer;
        float sqTSinThetaFromExternalLayerToFilmLayer = (externalLayerIOR / filmLayerIOR) *
                                                        (externalLayerIOR / filmLayerIOR) *
              sqISinThetaFromExternalLayerToFilmLayer ;

        // Total Internal Reflection at external-film interface (unlikely if filmLayerIOR > externalLayerIOR)
        // Total Internal Reflection at external-film interface (unlikely if filmLayerIOR > externalLayerIOR)
        if (sqTSinThetaFromExternalLayerToFilmLayer >= 1.0f)
//      if (sqTSinThetaFromExternalLayerToFilmLayer >= 1.0f)
        {
            return 1.0f;
//          return 1.0f;
        }
        float tCosThetaFromExternalLayerToFilmLayer = std::sqrtf(1.0f - sqTSinThetaFromExternalLayerToFilmLayer);
//      float tCosThetaFromExternalLayerToFilmLayer = std::sqrtf(1.0f - sqTSinThetaFromExternalLayerToFilmLayer);

        // Angle in substrate (Snell's Law: filmLayerIOR * iSinThetaFromFilmLayerToSubstrateLayer = substrateLayerIOR * tSinThetaFromFilmLayerToSubstrateLayer)
        // Angle in substrate (Snell's Law: filmLayerIOR * iSinThetaFromFilmLayerToSubstrateLayer = substrateLayerIOR * tSinThetaFromFilmLayerToSubstrateLayer)
        // Notes: iSinThetaFromFilmLayerToSubstrateLayer = tSinThetaFromExternalLayerToFilmLayer
        // Notes: iSinThetaFromFilmLayerToSubstrateLayer = tSinThetaFromExternalLayerToFilmLayer
        // Notes: iCosThetaFromFilmLayerToSubstrateLayer = tCosThetaFromExternalLayerToFilmLayer
        // Notes: iCosThetaFromFilmLayerToSubstrateLayer = tCosThetaFromExternalLayerToFilmLayer
        float sqTSinThetaFromFilmLayerToSubstrateLayer = (filmLayerIOR / substrateLayerIOR) * (filmLayerIOR / substrateLayerIOR) * sqTSinThetaFromExternalLayerToFilmLayer;
//      float sqTSinThetaFromFilmLayerToSubstrateLayer = (filmLayerIOR / substrateLayerIOR) * (filmLayerIOR / substrateLayerIOR) * sqTSinThetaFromExternalLayerToFilmLayer;

        // If sqTSinThetaFromFilmLayerToSubstrateLayer >= 1.0f then TIR occurs at film-substrate interface. This affects phase shift and amplitude at r12. For simplicity here -> we'll assume no TIR for substrate or that it's handled by complex IORs for metals.
        // If sqTSinThetaFromFilmLayerToSubstrateLayer >= 1.0f then TIR occurs at film-substrate interface. This affects phase shift and amplitude at r12. For simplicity here -> we'll assume no TIR for substrate or that it's handled by complex IORs for metals.

        float tCosThetaFromFilmLayerToSubstrateLayer = 1.0f; // Placeholder! Actual calculation more complex with complex IORs
//      float tCosThetaFromFilmLayerToSubstrateLayer = 1.0f; // Placeholder! Actual calculation more complex with complex IORs
        if (sqTSinThetaFromFilmLayerToSubstrateLayer < 1.0f && substrateLayerIOR > 0.0f)
//      if (sqTSinThetaFromFilmLayerToSubstrateLayer < 1.0f && substrateLayerIOR > 0.0f)
        {
            // only if not TIR and substrate is not vacuum/perfect reflector placeholder
            // only if not TIR and substrate is not vacuum/perfect reflector placeholder
            tCosThetaFromFilmLayerToSubstrateLayer = std::sqrtf(1.0f - sqTSinThetaFromFilmLayerToSubstrateLayer);
//          tCosThetaFromFilmLayerToSubstrateLayer = std::sqrtf(1.0f - sqTSinThetaFromFilmLayerToSubstrateLayer);
        }


        // Fresnel amplitude reflection coefficients (for electric field E)
        // Fresnel amplitude reflection coefficients (for electric field E)
        // r01: external -> film      interface
        // r01: external -> film      interface
        // r12: film     -> substrate interface
        // r12: film     -> substrate interface
        // For unpolarized light -> we average s- and p-polarization reflectance
        // For unpolarized light -> we average s- and p-polarization reflectance
        float r01S = FresnelAmplitudeS(externalLayerIOR, filmLayerIOR, iCosThetaFromExternalLayerToFilmLayer, tCosThetaFromExternalLayerToFilmLayer);
        float r01P = FresnelAmplitudeP(externalLayerIOR, filmLayerIOR, iCosThetaFromExternalLayerToFilmLayer, tCosThetaFromExternalLayerToFilmLayer);

        float r12S = FresnelAmplitudeS(filmLayerIOR, substrateLayerIOR, tCosThetaFromExternalLayerToFilmLayer, tCosThetaFromFilmLayerToSubstrateLayer);
        float r12P = FresnelAmplitudeP(filmLayerIOR, substrateLayerIOR, tCosThetaFromExternalLayerToFilmLayer, tCosThetaFromFilmLayerToSubstrateLayer);



        //              @Phase shift (delta) due to path length difference in the film
        //              @Phase shift (delta) due to path length difference in the film
        // Optical path length difference =                      2 * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer (or iCosThetaFromFilmLayerToSubstrateLayer)
        // Optical path length difference =                      2 * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer (or iCosThetaFromFilmLayerToSubstrateLayer)
        //              @Phase difference = (2 * PI / lambda) * (2 * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer (or iCosThetaFromFilmLayerToSubstrateLayer))
        //              @Phase difference = (2 * PI / lambda) * (2 * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer (or iCosThetaFromFilmLayerToSubstrateLayer))
        float delta = (4.0f * std::numbers::pi_v<float> * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer) / lambda;
//      float delta = (4.0f * std::numbers::pi_v<float> * filmLayerIOR * filmThickness * tCosThetaFromExternalLayerToFilmLayer) / lambda;


        // Interference calculation for amplitudes
        // Interference calculation for amplitudes
        // R_s = |r01_s + r12_s * exp(i*delta)|^2 / |1 + r01_s * r12_s * exp(i*delta)|^2
        // R_s = |r01_s + r12_s * exp(i*delta)|^2 / |1 + r01_s * r12_s * exp(i*delta)|^2
        // but simplified using intensities (approximating phase shifts for dielectrics)
        // but simplified using intensities (approximating phase shifts for dielectrics)
        // A more general form (from https://www.gamedev.net/tutorials/programming/graphics/thin-film-interference-for-computer-graphics-r2962/)
        // A more general form (from https://www.gamedev.net/tutorials/programming/graphics/thin-film-interference-for-computer-graphics-r2962/)
        // r_s = (r01_s + r12_s * std::cos(delta) - I * r12_s * std::sin(delta)) / (1 + r01_s*r12_s*std::cos(delta) - I * r01_s*r12_s*std::sin(delta))
        // r_s = (r01_s + r12_s * std::cos(delta) - I * r12_s * std::sin(delta)) / (1 + r01_s*r12_s*std::cos(delta) - I * r01_s*r12_s*std::sin(delta))
        // Reflectance_s = |r_s|^2
        // Reflectance_s = |r_s|^2

        // Simplified approach using intensities (less accurate for phase shifts but common)
        // Simplified approach using intensities (less accurate for phase shifts but common)
        // R = (r01^2 + r12^2 + 2*r01*r12*cos(delta)) / (1 + r01^2*r12^2 + 2*r01*r12*cos(delta))
        // R = (r01^2 + r12^2 + 2*r01*r12*cos(delta)) / (1 + r01^2*r12^2 + 2*r01*r12*cos(delta))
        // This is for intensity. We should use amplitudes.
        // This is for intensity. We should use amplitudes.


        // Using amplitudes (complex numbers simplified)
        // Using amplitudes (complex numbers simplified)
        // For s-polarization:
        // For s-polarization:
        float numSReal =       + r01S + r12S * std::cos(delta);
        float numSImag =              - r12S * std::sin(delta); // exp(-i * delta) or exp(+i * delta) convention matters
        float denSReal =  1.0f + r01S * r12S * std::cos(delta);
        float denSImag =       - r01S * r12S * std::sin(delta);
        float RSNum = numSReal * numSReal + numSImag * numSImag;
        float RSDen = denSReal * denSReal + denSImag * denSImag;
        float RS = RSNum / RSDen;
//      float RS = RSNum / RSDen;
        if (RSDen == 0.0f) RS = 1.0f;
//      if (RSDen == 0.0f) RS = 1.0f;


        // For p-polarization:
        // For p-polarization:
        float numPReal =         r01P + r12P * std::cos(delta);
        float numPImag =              - r12P * std::sin(delta);
        float denPReal =  1.0f + r01P * r12P * std::cos(delta);
        float denPImag =       - r01P * r12P * std::sin(delta);
        float RPNum = numPReal * numPReal + numPImag * numPImag;
        float RPDen = denPReal * denPReal + denPImag * denPImag;
        float RP = RPNum / RPDen;
//      float RP = RPNum / RPDen;
        if (RPDen == 0.0f) RP = 1.0f;
//      if (RPDen == 0.0f) RP = 1.0f;


        return 0.5f * (RS + RP); // Average for unpolarized light
//      return 0.5f * (RS + RP); // Average for unpolarized light
    }
    // Calculate Fresnel term with thin-film interference for RGB
    // Calculate Fresnel term with thin-film interference for RGB
    static inline Color3 FresnelThinFilm(const Vec3& V /* view vector (outgoing) */, const Vec3& N /* surface normal */, const Material& material)
//  static inline Color3 FresnelThinFilm(const Vec3& V /* view vector (outgoing) */, const Vec3& N /* surface normal */, const Material& material)
    {
        float iCosThetaFromExternalLayerToFilmLayer = Dot(V, N); // Assuming V is already normalized and points away from surface
//      float iCosThetaFromExternalLayerToFilmLayer = Dot(V, N); // Assuming V is already normalized and points away from surface
        if (iCosThetaFromExternalLayerToFilmLayer <= 0.0f) return { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // No reflection if looking from behind or grazing edge on
//      if (iCosThetaFromExternalLayerToFilmLayer <= 0.0f) return { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // No reflection if looking from behind or grazing edge on


        // Representative wavelengths for R G B (in nanometers)
        // Representative wavelengths for R G B (in nanometers)
        constexpr float lambdaR = 630.0f;
//      constexpr float lambdaR = 630.0f;
        constexpr float lambdaG = 532.0f;
//      constexpr float lambdaG = 532.0f;
        constexpr float lambdaB = 465.0f;
//      constexpr float lambdaB = 465.0f;


        float  externalLayerIOR = material.layer0IOR;
//      float  externalLayerIOR = material.layer0IOR;
        float      filmLayerIOR = material.layer1IOR;
//      float      filmLayerIOR = material.layer1IOR;
        float substrateLayerIOR = material.layer2IOR;
//      float substrateLayerIOR = material.layer2IOR;
        float filmThickness = material.layer1Thickness; // assumed to be in nanometers
//      float filmThickness = material.layer1Thickness; // assumed to be in nanometers


        float R = ThinFilmReflectanceForWavelength(iCosThetaFromExternalLayerToFilmLayer, filmThickness, lambdaR, externalLayerIOR, filmLayerIOR, substrateLayerIOR);
        float G = ThinFilmReflectanceForWavelength(iCosThetaFromExternalLayerToFilmLayer, filmThickness, lambdaG, externalLayerIOR, filmLayerIOR, substrateLayerIOR);
        float B = ThinFilmReflectanceForWavelength(iCosThetaFromExternalLayerToFilmLayer, filmThickness, lambdaB, externalLayerIOR, filmLayerIOR, substrateLayerIOR);


        return { .x = R, .y = G, .z = B };
//      return { .x = R, .y = G, .z = B };
    }










    // Higher totalExtinction => Shorter free-flight distances => Light scatters/absorbs more frequently
    // Higher totalExtinction => Shorter free-flight distances => Light scatters/absorbs more frequently
    inline static float SampleFreeFlightDistance(float totalExtinction)
//  inline static float SampleFreeFlightDistance(float totalExtinction)
    {
        // guard epsilon = 1e-6f
        // guard epsilon = 1e-6f
        // Exponential free-flight distance
        // Exponential free-flight distance
        float sample = std::fmaxf(1e-6f, Random());
//      float sample = std::fmaxf(1e-6f, Random());
        return -std::logf(sample) / std::fmaxf(1e-6f, totalExtinction);
//      return -std::logf(sample) / std::fmaxf(1e-6f, totalExtinction);
    }
    // Henyey–Greenstein phase sampling (sample scattered ray direction based on transmit direction)
    // Henyey–Greenstein phase sampling (sample scattered ray direction based on transmit direction)
    inline static Vec3 SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(const Vec3& transmitDirection, float asymmetryParameter)
//  inline static Vec3 SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(const Vec3& transmitDirection, float asymmetryParameter)
    {
        float     polarAngleSample = Random();
//      float     polarAngleSample = Random();
        float azimuthalAngleSample = Random();
//      float azimuthalAngleSample = Random();

        float cosTheta;
//      float cosTheta;
        if (std::fabsf(asymmetryParameter) < 1e-4f)
//      if (std::fabsf(asymmetryParameter) < 1e-4f)
        {
            cosTheta = 1.0f - 2.0f * polarAngleSample;
//          cosTheta = 1.0f - 2.0f * polarAngleSample;
        }
        else
//      else
        {
            float temp = (1.0f - asymmetryParameter * asymmetryParameter              ) / (1.0f - asymmetryParameter + 2.0f * asymmetryParameter * polarAngleSample);
//          float temp = (1.0f - asymmetryParameter * asymmetryParameter              ) / (1.0f - asymmetryParameter + 2.0f * asymmetryParameter * polarAngleSample);
            cosTheta   = (1.0f + asymmetryParameter * asymmetryParameter - temp * temp) / (                            2.0f * asymmetryParameter                   );
//          cosTheta   = (1.0f + asymmetryParameter * asymmetryParameter - temp * temp) / (                            2.0f * asymmetryParameter                   );
            cosTheta   = std::clamp(cosTheta, -1.0f, 1.0f);
//          cosTheta   = std::clamp(cosTheta, -1.0f, 1.0f);
        }
        float sinTheta = std::sqrtf(std::fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
//      float sinTheta = std::sqrtf(std::fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * std::numbers::pi_v<float> * azimuthalAngleSample;
//      float phi = 2.0f * std::numbers::pi_v<float> * azimuthalAngleSample;

        // build an ONB (Orthonormal Basis) around forward-axis-w = transmitDirection
        // build an ONB (Orthonormal Basis) around forward-axis-w = transmitDirection
        // build an orthonormal basis (u,v,w) so you can rotate the sampled local direction (in spherical coords) into world space
        // build an orthonormal basis (u,v,w) so you can rotate the sampled local direction (in spherical coords) into world space
        // Frisvad’s fast ONB (branchless & stable)
        // Frisvad’s fast ONB (branchless & stable)
        Vec3 w = Normalize(transmitDirection);
//      Vec3 w = Normalize(transmitDirection);
        float sign = std::copysign(1.0f, w.z);
//      float sign = std::copysign(1.0f, w.z);
        float stabilityFactor = -1.0f / (sign + w.z);
//      float stabilityFactor = -1.0f / (sign + w.z);
        float shorthand = w.x * w.y * stabilityFactor;
//      float shorthand = w.x * w.y * stabilityFactor;
        Vec3 u = Normalize(Vec3{ 1.0f + sign * w.x * w.x * stabilityFactor, sign * shorthand, -sign * w.x });
//      Vec3 u = Normalize(Vec3{ 1.0f + sign * w.x * w.x * stabilityFactor, sign * shorthand, -sign * w.x });
        Vec3 v = Cross(w, u);
//      Vec3 v = Cross(w, u);

        // Local Spherical -> World
        // Local Spherical -> World
        // Vec3 local = Vec3{ sinTheta * std::cosf(phi), sinTheta * std::sinf(phi), cosTheta };
        // Vec3 local = Vec3{ sinTheta * std::cosf(phi), sinTheta * std::sinf(phi), cosTheta };
        return Normalize(sinTheta * std::cosf(phi) * u + sinTheta * std::sinf(phi) * v + cosTheta * w);
//      return Normalize(sinTheta * std::cosf(phi) * u + sinTheta * std::sinf(phi) * v + cosTheta * w);
    }










    inline static MaterialScatteredResult Scatter(const Ray& rayIn, const RayHitResult& rayHitResult)
//  inline static MaterialScatteredResult Scatter(const Ray& rayIn, const RayHitResult& rayHitResult)
{
    MaterialScatteredResult materialScatteredResult {};
//  MaterialScatteredResult materialScatteredResult {};
    const Material& material = materialsDatabase.materials[rayHitResult.materialIndex];
//  const Material& material = materialsDatabase.materials[rayHitResult.materialIndex];
    switch (material.materialType)
//  switch (material.materialType)
    {

    case MaterialType::LambertianDiffuseReflectance1:
//  case MaterialType::LambertianDiffuseReflectance1:
        {
            Vec3 scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          Vec3 scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//              materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
            }
            else
            {
                materialScatteredResult.scatteredRay.dir = scatteredDirection;
//              materialScatteredResult.scatteredRay.dir = scatteredDirection;
            }
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case MaterialType::LambertianDiffuseReflectance2:
//  case MaterialType::LambertianDiffuseReflectance2:
        {
            Vec3 scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
//          Vec3 scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//              materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
            }
            else
            {
                materialScatteredResult.scatteredRay.dir = scatteredDirection;
//              materialScatteredResult.scatteredRay.dir = scatteredDirection;
            }
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case MaterialType::Metal:
//  case MaterialType::Metal:
        {
            Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case MaterialType::MetalFuzzy1:
//  case MaterialType::MetalFuzzy1:
        {
            Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            reflectionScatteredDirection = Normalize(reflectionScatteredDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          reflectionScatteredDirection = Normalize(reflectionScatteredDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = Dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//          materialScatteredResult.isScattered = Dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case MaterialType::MetalFuzzy2:
//  case MaterialType::MetalFuzzy2:
        {
            Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          Vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            reflectionScatteredDirection = Normalize(reflectionScatteredDirection + material.layer1Roughness * GenRandomUnitVector());
//          reflectionScatteredDirection = Normalize(reflectionScatteredDirection + material.layer1Roughness * GenRandomUnitVector());
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = Dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//          materialScatteredResult.isScattered = Dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case MaterialType::Dielectric:
//  case MaterialType::Dielectric:
        {
            Vec3 diffuseRayDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
//          Vec3 diffuseRayDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());

            float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
            if (rayHitResult.isFrontFace) [[likely]] { ratioOfEtaiOverEtat = material.layer0IOR / material.layer1IOR; }
//          if (rayHitResult.isFrontFace) [[likely]] { ratioOfEtaiOverEtat = material.layer0IOR / material.layer1IOR; }
            Vec3 normalizedIncomingRayDirection = Normalize(rayIn.dir);
//          Vec3 normalizedIncomingRayDirection = Normalize(rayIn.dir);

            float cosTheta = std::fminf(Dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
//          float cosTheta = std::fminf(Dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
            float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
//          float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
            bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
//          bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
            Vec3 scatteredRayDirection;
//          Vec3 scatteredRayDirection;

            // chia material.layer1Roughness ra làm 2 property, 1 cho reflect, 1 cho refract
            // chia material.layer1Roughness ra làm 2 property, 1 cho reflect, 1 cho refract
            if ( notAbleToRefract )
            {
                 scatteredRayDirection = BlendLinear(Reflect(normalizedIncomingRayDirection, rayHitResult.normal), diffuseRayDirection, material.layer1Roughness);
//               scatteredRayDirection = BlendLinear(Reflect(normalizedIncomingRayDirection, rayHitResult.normal), diffuseRayDirection, material.layer1Roughness);
            }
            else
            {
                 scatteredRayDirection = BlendLinear(Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat), -diffuseRayDirection, material.layer1Roughness);
//               scatteredRayDirection = BlendLinear(Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat), -diffuseRayDirection, material.layer1Roughness);
            }

            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
            materialScatteredResult.scatteredRay.dir = Normalize(materialScatteredResult.scatteredRay.dir);
//          materialScatteredResult.scatteredRay.dir = Normalize(materialScatteredResult.scatteredRay.dir);

/*
            // Absorption
            // Absorption
            if (!rayHitResult.isFrontFace)
//          if (!rayHitResult.isFrontFace)
            {
                materialScatteredResult.attenuation *= Color3
//              materialScatteredResult.attenuation *= Color3
                {
                    std::expf(-rayHitResult.minT * 1.0f),
                    std::expf(-rayHitResult.minT * 1.0f),
                    std::expf(-rayHitResult.minT * 1.0f),
                };
            }
*/
        }
        break;
//      break;



    case MaterialType::LightDiffuse:
//  case MaterialType::LightDiffuse:
        {
            materialScatteredResult.scatteredRay.ori = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.scatteredRay.ori = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.scatteredRay.dir = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.scatteredRay.dir = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.scatteredRay.time  = 0.0f;
//          materialScatteredResult.scatteredRay.time  = 0.0f;
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.emission = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.emission = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.isScattered = false;
//          materialScatteredResult.isScattered = false;
        }
        break;
//      break;



    case MaterialType::LightMetalic:
//  case MaterialType::LightMetalic:
        {
            materialScatteredResult.scatteredRay.ori = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.scatteredRay.ori = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.scatteredRay.dir = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.scatteredRay.dir = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.scatteredRay.time  = 0.0f;
//          materialScatteredResult.scatteredRay.time  = 0.0f;
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.emission = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.emission = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.isScattered = false;
//          materialScatteredResult.isScattered = false;
        }
        break;
//      break;



    case MaterialType::Isotropic1:
//  case MaterialType::Isotropic1:
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
//          materialScatteredResult.scatteredRay.dir = GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case MaterialType::Isotropic2:
//  case MaterialType::Isotropic2:
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = GenRandomUnitVector();
//          materialScatteredResult.scatteredRay.dir = GenRandomUnitVector();
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case MaterialType::FresnelBlendedDielectricGlossyDiffuse1:
//  case MaterialType::FresnelBlendedDielectricGlossyDiffuse1:
    {
        float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment
//      float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment

        Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
//      Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
        float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal
//      float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal

//      incident = outside environment / transmitted = inside environment
//      incident = outside environment / transmitted = inside environment
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
            ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//          ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
        }
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment

//      Calculate Fresnel reflectance probability using Schlick's approximation
//      Calculate Fresnel reflectance probability using Schlick's approximation
        float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);
//      float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);

        Vec3 scatteredDirection; Color3 attenuationColor;
//      Vec3 scatteredDirection; Color3 attenuationColor;

        materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails
//      materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails

//      Specular reflection branch
//      Specular reflection branch
        if (Random() < reflectanceProbability)
//      if (Random() < reflectanceProbability)
        {
//          Add fuzz/gloss/roughness for polished materials
//          Add fuzz/gloss/roughness for polished materials
            scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          Ensure scattered ray is above the surface
//          Ensure scattered ray is above the surface
            if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
//          if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
            {
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
                materialScatteredResult.isScattered = false;
//              materialScatteredResult.isScattered = false;
            }
            attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
//          attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
        }
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
        else
        {
//          Lambertian-like scattering
//          Lambertian-like scattering
            scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                scatteredDirection = rayHitResult.normal;
//              scatteredDirection = rayHitResult.normal;
            }
            attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        }

        if (materialScatteredResult.isScattered)
//      if (materialScatteredResult.isScattered)
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = attenuationColor;
//          materialScatteredResult.attenuation = attenuationColor;
        }
        else
        {
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
        }
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
    }
    break;
//  break;



    case MaterialType::FresnelBlendedDielectricGlossyDiffuse2:
//  case MaterialType::FresnelBlendedDielectricGlossyDiffuse2:
    {
        float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment
//      float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment

        Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
//      Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
        float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal
//      float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal

//      incident = outside environment / transmitted = inside environment
//      incident = outside environment / transmitted = inside environment
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
            ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//          ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
        }
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment

//      Calculate Fresnel reflectance probability using Schlick's approximation
//      Calculate Fresnel reflectance probability using Schlick's approximation
        float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);
//      float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);

        Vec3 scatteredDirection; Color3 attenuationColor;
//      Vec3 scatteredDirection; Color3 attenuationColor;

        materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails
//      materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails

//      Specular reflection branch
//      Specular reflection branch
        if (Random() < reflectanceProbability)
//      if (Random() < reflectanceProbability)
        {
//          Add fuzz/gloss/roughness for polished materials
//          Add fuzz/gloss/roughness for polished materials
            scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
//          scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
//          Ensure scattered ray is above the surface
//          Ensure scattered ray is above the surface
            if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
//          if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
            {
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
                materialScatteredResult.isScattered = false;
//              materialScatteredResult.isScattered = false;
            }
            attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
//          attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
        }
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
        else
        {
//          Lambertian-like scattering
//          Lambertian-like scattering
            scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
//          scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                scatteredDirection = rayHitResult.normal;
//              scatteredDirection = rayHitResult.normal;
            }
            attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        }

        if (materialScatteredResult.isScattered)
//      if (materialScatteredResult.isScattered)
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = attenuationColor;
//          materialScatteredResult.attenuation = attenuationColor;
        }
        else
        {
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
        }
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
    }
    break;
//  break;



    case MaterialType::FresnelBlendedDielectricGlossyDiffuse3:
//  case MaterialType::FresnelBlendedDielectricGlossyDiffuse3:
    {
        float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment
//      float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment

        Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
//      Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
        float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal
//      float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal

//      incident = outside environment / transmitted = inside environment
//      incident = outside environment / transmitted = inside environment
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
            ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//          ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
        }
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment

//      Calculate Fresnel reflectance probability using Schlick's approximation
//      Calculate Fresnel reflectance probability using Schlick's approximation
        float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);
//      float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);

        Vec3 scatteredDirection; Color3 attenuationColor;
//      Vec3 scatteredDirection; Color3 attenuationColor;

        materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails
//      materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails

//      Specular reflection branch
//      Specular reflection branch
        if (Random() < reflectanceProbability)
//      if (Random() < reflectanceProbability)
        {
//          Add fuzz/gloss/roughness for polished materials
//          Add fuzz/gloss/roughness for polished materials
            scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          Ensure scattered ray is above the surface
//          Ensure scattered ray is above the surface
            if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
//          if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
            {
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
                materialScatteredResult.isScattered = false;
//              materialScatteredResult.isScattered = false;
            }
            attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
//          attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
        }
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
        else
        {
//          Lambertian-like scattering
//          Lambertian-like scattering
            scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
//          scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVector());
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                scatteredDirection = rayHitResult.normal;
//              scatteredDirection = rayHitResult.normal;
            }
            attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        }

        if (materialScatteredResult.isScattered)
//      if (materialScatteredResult.isScattered)
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = attenuationColor;
//          materialScatteredResult.attenuation = attenuationColor;
        }
        else
        {
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
        }
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
    }
    break;
//  break;



    case MaterialType::FresnelBlendedDielectricGlossyDiffuse4:
//  case MaterialType::FresnelBlendedDielectricGlossyDiffuse4:
    {
        float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment
//      float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR; // incident = inside environment / transmitted = outside environment

        Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
//      Vec3 normalizedRayInDirection = Normalize(rayIn.dir);
        float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal
//      float cosThetaIncident = std::fminf(Dot(-normalizedRayInDirection, rayHitResult.normal), 1.0f); // Cosine of angle between view and normal

//      incident = outside environment / transmitted = inside environment
//      incident = outside environment / transmitted = inside environment
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
            ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//          ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
        }
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment
//      else case (isFrontFace is false) -> incident = inside environment / transmitted = outside environment

//      Calculate Fresnel reflectance probability using Schlick's approximation
//      Calculate Fresnel reflectance probability using Schlick's approximation
        float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);
//      float reflectanceProbability = Reflectance(cosThetaIncident, ratioOfEtaiOverEtat);

        Vec3 scatteredDirection; Color3 attenuationColor;
//      Vec3 scatteredDirection; Color3 attenuationColor;

        materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails
//      materialScatteredResult.isScattered = true; // Assume scattered unless specular reflection fails

//      Specular reflection branch
//      Specular reflection branch
        if (Random() < reflectanceProbability)
//      if (Random() < reflectanceProbability)
        {
//          Add fuzz/gloss/roughness for polished materials
//          Add fuzz/gloss/roughness for polished materials
            scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
//          scatteredDirection = Normalize(Reflect(normalizedRayInDirection, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
//          Ensure scattered ray is above the surface
//          Ensure scattered ray is above the surface
            if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
//          if (Dot(scatteredDirection, rayHitResult.normal) <= 0.0f)
            {
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
//              Fallback or absorb: For simplicity, make it a perfect reflection if fuzz pushes it below horizon or just let it be absorbed (isScattered = false). Given MetalFuzzy behavior, let's mark as not scattered if it goes wrong.
                materialScatteredResult.isScattered = false;
//              materialScatteredResult.isScattered = false;
            }
            attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
//          attenuationColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular highlights on dielectrics are typically white
        }
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
//      Diffuse reflection branch (light "entered" coating layer, scattered, and exited)
        else
        {
//          Lambertian-like scattering
//          Lambertian-like scattering
            scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          scatteredDirection = Normalize(rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            if (scatteredDirection.NearZero()) [[unlikely]]
//          if (scatteredDirection.NearZero()) [[unlikely]]
            {
                scatteredDirection = rayHitResult.normal;
//              scatteredDirection = rayHitResult.normal;
            }
            attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          attenuationColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        }

        if (materialScatteredResult.isScattered)
//      if (materialScatteredResult.isScattered)
        {
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = attenuationColor;
//          materialScatteredResult.attenuation = attenuationColor;
        }
        else
        {
            materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
//          materialScatteredResult.attenuation = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Absorbed
        }
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f }; // Polished materials are not emissive
    }
    break;
//  break;



    case MaterialType::ThinFilmDielectric1: // e.g. soap bubble
//  case MaterialType::ThinFilmDielectric1: // e.g. soap bubble
    {
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);
//      Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);

        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
//      float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
        averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability
//      averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability

        Vec3 scatteredRayDirection; bool reflects = false;
//      Vec3 scatteredRayDirection; bool reflects = false;

        if (Random() < averageReflectance)
//      if (Random() < averageReflectance)
        {
            reflects = true;
//          reflects = true;
            scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
            if (material.layer1Roughness > 0.0f)
//          if (material.layer1Roughness > 0.0f)
            {
                scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//              scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            }
        }
        else
        {
            float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
            if (rayHitResult.isFrontFace)
//          if (rayHitResult.isFrontFace)
            {
                ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//              ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
            }
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
//          scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
            // If TIR for refraction, it must reflect
            // If TIR for refraction, it must reflect
            // Refract might return zero vector on TIR
            // Refract might return zero vector on TIR
            if (scatteredRayDirection.LengthSquared() < 1e-4f)
//          if (scatteredRayDirection.LengthSquared() < 1e-4f)
            {
                reflects = true;
//              reflects = true;
                scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//              scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
                if (material.layer1Roughness > 0.0f)
//              if (material.layer1Roughness > 0.0f)
                {
                    scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//                  scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
                }
            }
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        if (reflects)
//      if (reflects)
        {
            materialScatteredResult.attenuation = fresnelColor; // The interference is the color
//          materialScatteredResult.attenuation = fresnelColor; // The interference is the color
        }
        else
        {
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
//          materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
            materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
//          materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
            materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
//          materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
            materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
//          materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
        }
        materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::ThinFilmDielectric2: // e.g. soap bubble
//  case MaterialType::ThinFilmDielectric2: // e.g. soap bubble
    {
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);
//      Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);

        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
//      float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
        averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability
//      averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability

        Vec3 scatteredRayDirection; bool reflects = false;
//      Vec3 scatteredRayDirection; bool reflects = false;

        if (Random() < averageReflectance)
//      if (Random() < averageReflectance)
        {
            reflects = true;
//          reflects = true;
            scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
            if (material.layer1Roughness > 0.0f)
//          if (material.layer1Roughness > 0.0f)
            {
                scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
//              scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
            }
        }
        else
        {
            float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
            if (rayHitResult.isFrontFace)
//          if (rayHitResult.isFrontFace)
            {
                ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//              ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
            }
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
//          scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
            // If TIR for refraction, it must reflect
            // If TIR for refraction, it must reflect
            // Refract might return zero vector on TIR
            // Refract might return zero vector on TIR
            if (scatteredRayDirection.LengthSquared() < 1e-4f)
//          if (scatteredRayDirection.LengthSquared() < 1e-4f)
            {
                reflects = true;
//              reflects = true;
                scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//              scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
                if (material.layer1Roughness > 0.0f)
//              if (material.layer1Roughness > 0.0f)
                {
                    scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
//                  scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
                }
            }
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        if (reflects)
//      if (reflects)
        {
            materialScatteredResult.attenuation = fresnelColor; // The interference is the color
//          materialScatteredResult.attenuation = fresnelColor; // The interference is the color
        }
        else
        {
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
//          materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
            materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
//          materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
            materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
//          materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
            materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
//          materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
        }
        materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::ThinFilmDielectric3: // e.g. soap bubble
//  case MaterialType::ThinFilmDielectric3: // e.g. soap bubble
    {
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);
//      Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);

        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
//      float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
        averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability
//      averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability

        Vec3 scatteredRayDirection; bool reflects = false;
//      Vec3 scatteredRayDirection; bool reflects = false;

        if (Random() < averageReflectance)
//      if (Random() < averageReflectance)
        {
            reflects = true;
//          reflects = true;
            scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
            if (material.layer1Roughness > 0.0f)
//          if (material.layer1Roughness > 0.0f)
            {
                scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//              scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            }
        }
        else
        {
            float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
            if (rayHitResult.isFrontFace)
//          if (rayHitResult.isFrontFace)
            {
                ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//              ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
            }
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
//          scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
            // If TIR for refraction, it must reflect
            // If TIR for refraction, it must reflect
            // Refract might return zero vector on TIR
            // Refract might return zero vector on TIR
            if (scatteredRayDirection.LengthSquared() < 1e-4f)
//          if (scatteredRayDirection.LengthSquared() < 1e-4f)
            {
                reflects = true;
//              reflects = true;
                scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//              scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
                if (material.layer1Roughness > 0.0f)
//              if (material.layer1Roughness > 0.0f)
                {
                    scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
//                  scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
                }
            }
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        if (reflects)
//      if (reflects)
        {
            materialScatteredResult.attenuation = fresnelColor; // The interference is the color
//          materialScatteredResult.attenuation = fresnelColor; // The interference is the color
        }
        else
        {
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
//          materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
            materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
//          materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
            materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
//          materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
            materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
//          materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
        }
        materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::ThinFilmDielectric4: // e.g. soap bubble
//  case MaterialType::ThinFilmDielectric4: // e.g. soap bubble
    {
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        // This material primarily reflects and refracts. The main difference from a standard dielectric is the colored Fresnel reflection.
        Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);
//      Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);

        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        // Decide to reflect or refract based on an *average* or luminance of fresnelColor. This is an approximation. A more advanced way would be spectral sampling.
        float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
//      float averageReflectance = (fresnelColor.x + fresnelColor.y + fresnelColor.z) / 3.0f;
        averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability
//      averageReflectance = std::clamp(averageReflectance, 0.0f, 1.0f); // Ensure valid probability

        Vec3 scatteredRayDirection; bool reflects = false;
//      Vec3 scatteredRayDirection; bool reflects = false;

        if (Random() < averageReflectance)
//      if (Random() < averageReflectance)
        {
            reflects = true;
//          reflects = true;
            scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
            if (material.layer1Roughness > 0.0f)
//          if (material.layer1Roughness > 0.0f)
            {
                scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
//              scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVector());
            }
        }
        else
        {
            float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          float ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
            if (rayHitResult.isFrontFace)
//          if (rayHitResult.isFrontFace)
            {
                ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
//              ratioOfEtaiOverEtat = 1.0f / ratioOfEtaiOverEtat;
            }
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            // Note: This simplified refraction doesn't consider the substrate yet. A full model would trace rays from external into film, then from film to substrate or film to external.
            scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
//          scatteredRayDirection = Normalize(Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat));
            // If TIR for refraction, it must reflect
            // If TIR for refraction, it must reflect
            // Refract might return zero vector on TIR
            // Refract might return zero vector on TIR
            if (scatteredRayDirection.LengthSquared() < 1e-4f)
//          if (scatteredRayDirection.LengthSquared() < 1e-4f)
            {
                reflects = true;
//              reflects = true;
                scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
//              scatteredRayDirection = Reflect(rayIn.dir, rayHitResult.normal);
                if (material.layer1Roughness > 0.0f)
//              if (material.layer1Roughness > 0.0f)
                {
                    scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//                  scatteredRayDirection = Normalize(scatteredRayDirection + material.layer1Roughness * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
                }
            }
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        if (reflects)
//      if (reflects)
        {
            materialScatteredResult.attenuation = fresnelColor; // The interference is the color
//          materialScatteredResult.attenuation = fresnelColor; // The interference is the color
        }
        else
        {
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            // Transmitted light for simple thin film usually retains base color or is white-ish. A proper model would also apply Fresnel at the second interface (film:substrate).
            materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
//          materialScatteredResult.attenuation = Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } - fresnelColor; // Basic energy conservation for transmission
            materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
//          materialScatteredResult.attenuation.x = std::fmaxf(0.0f, materialScatteredResult.attenuation.x);
            materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
//          materialScatteredResult.attenuation.y = std::fmaxf(0.0f, materialScatteredResult.attenuation.y);
            materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
//          materialScatteredResult.attenuation.z = std::fmaxf(0.0f, materialScatteredResult.attenuation.z);
        }
        materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      materialScatteredResult.attenuation *= Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::ThinFilmMetal:
//  case MaterialType::ThinFilmMetal:
    {
        // For a metal substrate, there's usually no transmission into the metal. The reflection is modulated by the thin film interference.
        // For a metal substrate, there's usually no transmission into the metal. The reflection is modulated by the thin film interference.
        Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);
//      Color3 fresnelColor = FresnelThinFilm(-rayIn.dir, rayHitResult.normal, material);

        Vec3 scatteredRayDirection;
//      Vec3 scatteredRayDirection;

        // @Rough surface with film
        // @Rough surface with film
        if (material.layer1Roughness > 0.0f)
//      if (material.layer1Roughness > 0.0f)
        {
            // Importance sample a microfacet normal H. For simplicity, using isotropic GGX sampling with average roughness. A true anisotropic film on rough surface is even more complex.
            // Importance sample a microfacet normal H. For simplicity, using isotropic GGX sampling with average roughness. A true anisotropic film on rough surface is even more complex.
            float averageRoughness = material.layer1Roughness;
//          float averageRoughness = material.layer1Roughness;
            Vec2 xi = { .x = Random(), .y = Random() };
//          Vec2 xi = { .x = Random(), .y = Random() };
            // Simplified GGX H sampling (replace with a proper GGX/Beckmann sampler if available)
            // Simplified GGX H sampling (replace with a proper GGX/Beckmann sampler if available)
            float a = averageRoughness * averageRoughness;
//          float a = averageRoughness * averageRoughness;
            float phi = 2.0f * std::numbers::pi_v<float> * xi.x;
//          float phi = 2.0f * std::numbers::pi_v<float> * xi.x;
            float cosThetaHSq = (1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y);
//          float cosThetaHSq = (1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y);
            float cosThetaH = std::sqrtf(       cosThetaHSq);
//          float cosThetaH = std::sqrtf(       cosThetaHSq);
            float sinThetaH = std::sqrtf(1.0f - cosThetaHSq);
//          float sinThetaH = std::sqrtf(1.0f - cosThetaHSq);

            Vec3 HLocal{ .x = sinThetaH * std::cosf(phi), .y = sinThetaH * std::sinf(phi), .z = cosThetaH };
//          Vec3 HLocal{ .x = sinThetaH * std::cosf(phi), .y = sinThetaH * std::sinf(phi), .z = cosThetaH };

            // Transform HLocal from shading space (normal is { 0 , 0 , 1 }) to world space
            // Transform HLocal from shading space (normal is { 0 , 0 , 1 }) to world space
            Vec3                                          tempT   { .x = 1.0f, .y = 0.0f, .z = 0.0f };
//          Vec3                                          tempT   { .x = 1.0f, .y = 0.0f, .z = 0.0f };
            if (std::fabsf(rayHitResult.normal.x) > 0.9f) tempT = { .x = 0.0f, .y = 1.0f, .z = 0.0f };
//          if (std::fabsf(rayHitResult.normal.x) > 0.9f) tempT = { .x = 0.0f, .y = 1.0f, .z = 0.0f };
            Vec3 TShade = Normalize(Cross(tempT, rayHitResult.normal        ));
//          Vec3 TShade = Normalize(Cross(tempT, rayHitResult.normal        ));
            Vec3 BShade =           Cross(       rayHitResult.normal, TShade) ;
//          Vec3 BShade =           Cross(       rayHitResult.normal, TShade) ;
            Vec3 H = HLocal.x * TShade + HLocal.y * BShade + HLocal.z * rayHitResult.normal;
//          Vec3 H = HLocal.x * TShade + HLocal.y * BShade + HLocal.z * rayHitResult.normal;

            scatteredRayDirection = Normalize(Reflect(rayIn.dir, H));
//          scatteredRayDirection = Normalize(Reflect(rayIn.dir, H));
        }
        // Smooth surface with film
        // Smooth surface with film
        else
        {
            scatteredRayDirection = Normalize(Reflect(rayIn.dir, rayHitResult.normal));
//          scatteredRayDirection = Normalize(Reflect(rayIn.dir, rayHitResult.normal));
        }

        if (Dot(scatteredRayDirection, rayHitResult.normal) <= 0.0f)
//      if (Dot(scatteredRayDirection, rayHitResult.normal) <= 0.0f)
        {
            materialScatteredResult.isScattered = false;
//          materialScatteredResult.isScattered = false;
            return materialScatteredResult;
//          return materialScatteredResult;
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        // Attenuation is the film's colored reflectance + base metal's color (F0 of metal). This is a common approximation: Modulate base metal F0 with interference color. A more physically correct way involves complex IORs for metal in ThinFilmReflectanceForWavelength.
        // Attenuation is the film's colored reflectance + base metal's color (F0 of metal). This is a common approximation: Modulate base metal F0 with interference color. A more physically correct way involves complex IORs for metal in ThinFilmReflectanceForWavelength.
        Color3 baseMetalF0 = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      Color3 baseMetalF0 = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
        materialScatteredResult.attenuation = fresnelColor * baseMetalF0; // Modulate
//      materialScatteredResult.attenuation = fresnelColor * baseMetalF0; // Modulate
        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::SubsurfaceRandomWalk1:
//  case MaterialType::SubsurfaceRandomWalk1:
    {
        // Homogeneous medium just beneath the surface
        // Homogeneous medium just beneath the surface

        // Choose which side we’re going into:
        // Choose which side we’re going into:
        // If we’re outside (front face), enter; if already inside (back face), just continue.
        // If we’re outside (front face), enter; if already inside (back face), just continue.
        float eta = material.layer1IOR / material.layer0IOR;
//      float eta = material.layer1IOR / material.layer0IOR;
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
              eta = material.layer0IOR / material.layer1IOR;
//            eta = material.layer0IOR / material.layer1IOR;
        }

        // Try to refract into the medium; if TIR, reflect (acts like a glossy dielectric at grazing angles)
        // Try to refract into the medium; if TIR, reflect (acts like a glossy dielectric at grazing angles)
        Vec3 transmitDirection;
//      Vec3 transmitDirection;
        bool totalInternalReflection = false;
//      bool totalInternalReflection = false;

        float cosTheta = std::fminf(Dot(-rayIn.dir, rayHitResult.normal), 1.0f);
//      float cosTheta = std::fminf(Dot(-rayIn.dir, rayHitResult.normal), 1.0f);
        Vec3  tangentialPartOfTheRefractedDirection = eta * (rayIn.dir + cosTheta * rayHitResult.normal);
//      Vec3  tangentialPartOfTheRefractedDirection = eta * (rayIn.dir + cosTheta * rayHitResult.normal);
        float leftoverEnergy = 1.0f - tangentialPartOfTheRefractedDirection.LengthSquared();
//      float leftoverEnergy = 1.0f - tangentialPartOfTheRefractedDirection.LengthSquared();
        if (leftoverEnergy < 0.0f)
//      if (leftoverEnergy < 0.0f)
        {
            totalInternalReflection = true;
//          totalInternalReflection = true;
        }
        if (!totalInternalReflection)
//      if (!totalInternalReflection)
        {
            Vec3 parallel = -std::sqrtf(std::fmaxf(0.0f, leftoverEnergy)) * rayHitResult.normal;
//          Vec3 parallel = -std::sqrtf(std::fmaxf(0.0f, leftoverEnergy)) * rayHitResult.normal;
            transmitDirection = Normalize(tangentialPartOfTheRefractedDirection + parallel);
//          transmitDirection = Normalize(tangentialPartOfTheRefractedDirection + parallel);
        }

        if (totalInternalReflection)
//      if (totalInternalReflection)
        {
            // Specular reflection fallback
            // Specular reflection fallback
            Vec3 reflect = Normalize(Reflect(rayIn.dir, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
//          Vec3 reflect = Normalize(Reflect(rayIn.dir, rayHitResult.normal) + material.layer1Roughness * GenRandomUnitVector());
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflect;
//          materialScatteredResult.scatteredRay.dir = reflect;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//          materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = Dot(reflect, rayHitResult.normal) > 0.0f;
//          materialScatteredResult.isScattered = Dot(reflect, rayHitResult.normal) > 0.0f;
            break;
//          break;
        }

        // We are in the medium now: do one interior free-flight and a phase-function scatter
        // We are in the medium now: do one interior free-flight and a phase-function scatter
        Color3 totalExtinction
//      Color3 totalExtinction
        {
            .x = material.absorption.x + material.scattering.x,
            .y = material.absorption.y + material.scattering.y,
            .z = material.absorption.z + material.scattering.z,
        };

        // Sample distance with a single luminance-based total extinction then do per-channel Beer–Lambert attenuation
        // Sample distance with a single luminance-based total extinction then do per-channel Beer–Lambert attenuation
        float luminanceBasedTotalExtinction = 0.3333f * (totalExtinction.x + totalExtinction.y + totalExtinction.z);
//      float luminanceBasedTotalExtinction = 0.3333f * (totalExtinction.x + totalExtinction.y + totalExtinction.z);
        float freeFlightDistance = SampleFreeFlightDistance(luminanceBasedTotalExtinction);
//      float freeFlightDistance = SampleFreeFlightDistance(luminanceBasedTotalExtinction);

        // Interior scatter point (push a bit inside to avoid self-hit)
        // Interior scatter point (push a bit inside to avoid self-hit)
        Vec3 normal = rayHitResult.normal;
//      Vec3 normal = rayHitResult.normal;
        if (rayHitResult.isFrontFace) normal = -rayHitResult.normal;
//      if (rayHitResult.isFrontFace) normal = -rayHitResult.normal;
        Vec3 interiorScatterPoint = rayHitResult.at + normal * 1e-4f + freeFlightDistance * transmitDirection;
//      Vec3 interiorScatterPoint = rayHitResult.at + normal * 1e-4f + freeFlightDistance * transmitDirection;

/*
        // Beer–Lambert attenuation from entry to scatter
        // Beer–Lambert attenuation from entry to scatter
        Color3 beerLambertAttenuation
//      Color3 beerLambertAttenuation
        {
            .x = std::expf(-material.absorption.x * freeFlightDistance),
            .y = std::expf(-material.absorption.y * freeFlightDistance),
            .z = std::expf(-material.absorption.z * freeFlightDistance),
//          .x = std::expf(-material.scattering.x * freeFlightDistance),
//          .y = std::expf(-material.scattering.y * freeFlightDistance),
//          .z = std::expf(-material.scattering.z * freeFlightDistance),
//          .x = std::expf(-totalExtinction.x * freeFlightDistance),
//          .y = std::expf(-totalExtinction.y * freeFlightDistance),
//          .z = std::expf(-totalExtinction.z * freeFlightDistance),
        };
*/

        // Single-scatter albedo factor
        // Single-scatter albedo factor
        Color3 singleScatterWeight
//      Color3 singleScatterWeight
        {
        };
        if (totalExtinction.x > 0.0f) singleScatterWeight.x = material.scattering.x / totalExtinction.x;
        if (totalExtinction.y > 0.0f) singleScatterWeight.y = material.scattering.y / totalExtinction.y;
        if (totalExtinction.z > 0.0f) singleScatterWeight.z = material.scattering.z / totalExtinction.z;


        // Sample new interior direction from phase function
        // Sample new interior direction from phase function
        Vec3 interiorDirection = SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(transmitDirection, material.asymmetryParameter);
//      Vec3 interiorDirection = SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(transmitDirection, material.asymmetryParameter);

        // Return a ray starting at the interior scatter point heading somewhere inside.
        // Return a ray starting at the interior scatter point heading somewhere inside.
        // On the next intersection it will hit the inner surface and exit naturally.
        // On the next intersection it will hit the inner surface and exit naturally.
        materialScatteredResult.scatteredRay.ori = interiorScatterPoint;
//      materialScatteredResult.scatteredRay.ori = interiorScatterPoint;
        materialScatteredResult.scatteredRay.dir = interiorDirection;
//      materialScatteredResult.scatteredRay.dir = interiorDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;

        Color3 textureColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//      Color3 textureColor = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);

        materialScatteredResult.attenuation = Color3
//      materialScatteredResult.attenuation = Color3
        {
//          .x = textureColor.x * beerLambertAttenuation.x * singleScatterWeight.x,
//          .y = textureColor.y * beerLambertAttenuation.y * singleScatterWeight.y,
//          .z = textureColor.z * beerLambertAttenuation.z * singleScatterWeight.z,
            .x = textureColor.x                            * singleScatterWeight.x,
            .y = textureColor.y                            * singleScatterWeight.y,
            .z = textureColor.z                            * singleScatterWeight.z,
        };

        materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//      materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
    }
    break;
//  break;



    case MaterialType::SubsurfaceRandomWalk2:
//  case MaterialType::SubsurfaceRandomWalk2:
    {
        // 1. Fresnel Reflection vs. Refraction
        // 1. Fresnel Reflection vs. Refraction
        // Decide if the ray reflects off the surface or enters the medium based on the Fresnel equations.
        // Decide if the ray reflects off the surface or enters the medium based on the Fresnel equations.
        float ratioOfEtaiOverEtat;
//      float ratioOfEtaiOverEtat;
        if (rayHitResult.isFrontFace)
//      if (rayHitResult.isFrontFace)
        {
            ratioOfEtaiOverEtat = material.layer0IOR / material.layer1IOR;
//          ratioOfEtaiOverEtat = material.layer0IOR / material.layer1IOR;
        }
        else
        {
            // Ray is inside the medium and trying to exit.
            // Ray is inside the medium and trying to exit.
            ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
//          ratioOfEtaiOverEtat = material.layer1IOR / material.layer0IOR;
        }
        float cosTheta = std::fminf(Dot(-rayIn.dir, rayHitResult.normal), 1.0f);
//      float cosTheta = std::fminf(Dot(-rayIn.dir, rayHitResult.normal), 1.0f);
        float reflectanceProbability = Reflectance(cosTheta, ratioOfEtaiOverEtat);
//      float reflectanceProbability = Reflectance(cosTheta, ratioOfEtaiOverEtat);


        // 2. Probabilistically choose between reflection and subsurface scattering
        // 2. Probabilistically choose between reflection and subsurface scattering
        if (Random() < reflectanceProbability)
//      if (Random() < reflectanceProbability)
        {
            // --- Surface Reflection (like a glossy dielectric) ---
            // --- Surface Reflection (like a glossy dielectric) ---
            Vec3 reflectedDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          Vec3 reflectedDirection = Reflect(rayIn.dir, rayHitResult.normal);
            if (material.layer1Roughness > 0.0f)
//          if (material.layer1Roughness > 0.0f)
            {
                 reflectedDirection = Normalize(reflectedDirection + material.layer1Roughness * GenRandomUnitVector());
//               reflectedDirection = Normalize(reflectedDirection + material.layer1Roughness * GenRandomUnitVector());
            }

            materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = reflectedDirection, .time = rayIn.time, };
//          materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = reflectedDirection, .time = rayIn.time, };
            materialScatteredResult.attenuation = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular reflection is white
//          materialScatteredResult.attenuation = { .x = 1.0f, .y = 1.0f, .z = 1.0f }; // Specular reflection is white
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = (Dot(reflectedDirection, rayHitResult.normal) > 0.0f); // Ensure reflection is valid
//          materialScatteredResult.isScattered = (Dot(reflectedDirection, rayHitResult.normal) > 0.0f); // Ensure reflection is valid
        }
        else
        {
            // --- Subsurface Scattering (Random Walk) ---
            // --- Subsurface Scattering (Random Walk) ---
            Vec3 refractedDirection = Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat);
//          Vec3 refractedDirection = Refract(rayIn.dir, rayHitResult.normal, ratioOfEtaiOverEtat);
            if ( refractedDirection.LengthSquared() < 1e-6 )
//          if ( refractedDirection.LengthSquared() < 1e-6 )
            {
                // Handle Total Internal Reflection
                // Handle Total Internal Reflection
                Vec3 reflectedDirection = Reflect(rayIn.dir, rayHitResult.normal);
//              Vec3 reflectedDirection = Reflect(rayIn.dir, rayHitResult.normal);
                materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = reflectedDirection, .time = rayIn.time, };
//              materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = reflectedDirection, .time = rayIn.time, };
                materialScatteredResult.attenuation = { .x = 1.0f, .y = 1.0f, .z = 1.0f };
//              materialScatteredResult.attenuation = { .x = 1.0f, .y = 1.0f, .z = 1.0f };
                materialScatteredResult.isScattered = true;
//              materialScatteredResult.isScattered = true;
            }
            else
            {
                // Ray has entered the medium, perform the one-bounce random walk
                // Ray has entered the medium, perform the one-bounce random walk
                Color3 totalExtinction = material.absorption + material.scattering;
//              Color3 totalExtinction = material.absorption + material.scattering;
                float luminanceBasedTotalExtinction = (totalExtinction.x + totalExtinction.y + totalExtinction.z) / 3.0f;
//              float luminanceBasedTotalExtinction = (totalExtinction.x + totalExtinction.y + totalExtinction.z) / 3.0f;

                // If the medium doesn't absorb or scatter, just treat it as a simple transparent dielectric
                // If the medium doesn't absorb or scatter, just treat it as a simple transparent dielectric
                if (luminanceBasedTotalExtinction < 1e-6f)
//              if (luminanceBasedTotalExtinction < 1e-6f)
                {
                    materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = refractedDirection, .time = rayIn.time, };
//                  materialScatteredResult.scatteredRay = { .ori = rayHitResult.at, .dir = refractedDirection, .time = rayIn.time, };
                    materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
//                  materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at);
                    materialScatteredResult.isScattered = true;
//                  materialScatteredResult.isScattered = true;
                }
                else
                {
                    // Sample the distance the light travels inside the medium
                    // Sample the distance the light travels inside the medium
                    float freeFlightDistance = SampleFreeFlightDistance(luminanceBasedTotalExtinction);
//                  float freeFlightDistance = SampleFreeFlightDistance(luminanceBasedTotalExtinction);

                    // Calculate the point where the light scatters inside the medium
                    // Calculate the point where the light scatters inside the medium
                    Point3 interiorPoint = rayHitResult.at + refractedDirection * (freeFlightDistance + 1e-4f);
//                  Point3 interiorPoint = rayHitResult.at + refractedDirection * (freeFlightDistance + 1e-4f);

                    // Sample a new direction after scattering using the Henyey-Greenstein phase function
                    // Sample a new direction after scattering using the Henyey-Greenstein phase function
                    Vec3 newDirection = SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(refractedDirection, material.asymmetryParameter);
//                  Vec3 newDirection = SampleScatteredDirectionUsingHenyeyGreensteinPhaseFunction(refractedDirection, material.asymmetryParameter);

                    materialScatteredResult.scatteredRay = { .ori = interiorPoint, .dir = newDirection, .time = rayIn.time, };
//                  materialScatteredResult.scatteredRay = { .ori = interiorPoint, .dir = newDirection, .time = rayIn.time, };

                    // Attenuation from Beer-Lambert Law (absorption and out-scattering)
                    // Attenuation from Beer-Lambert Law (absorption and out-scattering)
                    Color3 beerLambertAttenuation
//                  Color3 beerLambertAttenuation
                    {
                        .x = expf(-totalExtinction.x * freeFlightDistance),
                        .y = expf(-totalExtinction.y * freeFlightDistance),
                        .z = expf(-totalExtinction.z * freeFlightDistance),
                    };

                    // Albedo: The probability of scattering vs. absorbing
                    // Albedo: The probability of scattering vs. absorbing
                    Color3 singleScatterAlbedo
//                  Color3 singleScatterAlbedo
                    {
                        .x = material.scattering.x / Safe(totalExtinction.x > 1e-6f, totalExtinction.x, 1.0f),
                        .y = material.scattering.y / Safe(totalExtinction.y > 1e-6f, totalExtinction.y, 1.0f),
                        .z = material.scattering.z / Safe(totalExtinction.z > 1e-6f, totalExtinction.z, 1.0f),
                    };

                    // Final attenuation combines surface color, albedo, and Beer-Lambert law
                    // Final attenuation combines surface color, albedo, and Beer-Lambert law
                    materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) * singleScatterAlbedo * beerLambertAttenuation;
//                  materialScatteredResult.attenuation = Value(material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) * singleScatterAlbedo * beerLambertAttenuation;
                    materialScatteredResult.isScattered = true;
//                  materialScatteredResult.isScattered = true;
                }
            }
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
        }
    }
    break;
//  break;



    default:
//  default:
        break;
//      break;
    }
    return materialScatteredResult;
//  return materialScatteredResult;
}




inline
static RayHitResult RayHit(const Geometry& geo
//                        (const Geometry& geo
                          ,const Ray     & ray
//                        ,const Ray     & ray
                          ,const Interval& rayT
//                        ,const Interval& rayT
                          )
//                        )
{
    switch (geo.geometryType)
//  switch (geo.geometryType)
    {
        case GeometryType::SPHERE:
//      case GeometryType::SPHERE:
        {
            const Point3& currentSphereCenterByIncomingRayTime = Marching(geo.sphere.center, geo.movingDirection, ray.time);
//          const Point3& currentSphereCenterByIncomingRayTime = Marching(geo.sphere.center, geo.movingDirection, ray.time);
            const Vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
//          const Vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
            const float& a = ray.dir.LengthSquared();
//          const float& a = ray.dir.LengthSquared();
            const float& h = Dot(ray.dir, fromSphereCenterToRayOrigin);
//          const float& h = Dot(ray.dir, fromSphereCenterToRayOrigin);
            const float& c = fromSphereCenterToRayOrigin.LengthSquared() - geo.sphere.radius * geo.sphere.radius;
//          const float& c = fromSphereCenterToRayOrigin.LengthSquared() - geo.sphere.radius * geo.sphere.radius;
            const float& discriminant = h * h - a * c;
//          const float& discriminant = h * h - a * c;
            RayHitResult rayHitResult { .materialIndex = geo.materialIndex };
//          RayHitResult rayHitResult { .materialIndex = geo.materialIndex };
//          rayHitResult.hitted = discriminant >= 0.0f;
//          rayHitResult.hitted = discriminant >= 0.0f;
            if (discriminant < 0.0f)
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
            }
            else
            {
                float sqrtDiscriminant = std::sqrt(discriminant);
//              float sqrtDiscriminant = std::sqrt(discriminant);

                float t = (h - sqrtDiscriminant) / a;
//              float t = (h - sqrtDiscriminant) / a;

                if (!rayT.Surrounds(t))
//              if (!rayT.Surrounds(t))
                {
                    t = (h + sqrtDiscriminant) / a;
//                  t = (h + sqrtDiscriminant) / a;

                    if (!rayT.Surrounds(t))
//                  if (!rayT.Surrounds(t))
                    {
                        rayHitResult.hitted = false;
//                      rayHitResult.hitted = false;
                        return rayHitResult;
//                      return rayHitResult;
                    }
                }

                rayHitResult.hitted = true;
//              rayHitResult.hitted = true;

                rayHitResult.minT = t;
//              rayHitResult.minT = t;

                rayHitResult.at = ray.Marching(rayHitResult.minT);
//              rayHitResult.at = ray.Marching(rayHitResult.minT);

                const Vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / geo.sphere.radius;
//              const Vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / geo.sphere.radius;

                rayHitResult.isFrontFace = Dot(ray.dir, outwardNormal) < 0.0f;
//              rayHitResult.isFrontFace = Dot(ray.dir, outwardNormal) < 0.0f;
                if (rayHitResult.isFrontFace)
//              if (rayHitResult.isFrontFace)
                {
                    rayHitResult.normal =  outwardNormal;
//                  rayHitResult.normal =  outwardNormal;
                }
                else
//              else
                {
                    rayHitResult.normal = -outwardNormal;
//                  rayHitResult.normal = -outwardNormal;
                }

                // GEOGRAPHIC COORDINATE
                // GEOGRAPHIC COORDINATE
                // normalizedSurfacePoint (also known as outwardNormal): a given point on the sphere of radius one and centered at the origin <0 0 0>
                // normalizedSurfacePoint (also known as outwardNormal): a given point on the sphere of radius one and centered at the origin <0 0 0>
                // uSurfaceCoordinate: returned value [0,1] of angle around the Y axis from X=-1
                // uSurfaceCoordinate: returned value [0,1] of angle around the Y axis from X=-1
                // vSurfaceCoordinate: returned value [0,1] of angle                   from Y=-1 to Y=+1
                // vSurfaceCoordinate: returned value [0,1] of angle                   from Y=-1 to Y=+1
                // <+1 +0 +0> yields <+0.50 +0.50> | <-1 +0 +0> yields <0.00 0.50>
                // <+0 +1 +0> yields <+0.50 +1.00> | <+0 -1 +0> yields <0.50 0.00>
                // <+0 +0 +1> yields <+0.25 +0.50> | <+0 +0 -1> yields <0.75 0.50>

                float theta = std::acos (-outwardNormal.y); // latitude
//              float theta = std::acos (-outwardNormal.y); // latitude
                float phi   = std::atan2(-outwardNormal.z, outwardNormal.x) + std::numbers::pi_v<float>; // longitude
//              float phi   = std::atan2(-outwardNormal.z, outwardNormal.x) + std::numbers::pi_v<float>; // longitude

                rayHitResult.uSurfaceCoordinate = phi   / (2.0f * std::numbers::pi_v<float>);
//              rayHitResult.uSurfaceCoordinate = phi   / (2.0f * std::numbers::pi_v<float>);
                rayHitResult.vSurfaceCoordinate = theta /         std::numbers::pi_v<float> ;
//              rayHitResult.vSurfaceCoordinate = theta /         std::numbers::pi_v<float> ;
            }

            return rayHitResult;
//          return rayHitResult;
        }
        break;
//      break;


        case GeometryType::PRIMITIVE:
//      case GeometryType::PRIMITIVE:
        {
            const Point3& currentPrimitiveVertex0ByIncomingRayTime = Marching(geo.primitive.vertex0, geo.movingDirection, ray.time);
            const Point3& currentPrimitiveVertex1ByIncomingRayTime = Marching(geo.primitive.vertex1, geo.movingDirection, ray.time);
            const Point3& currentPrimitiveVertex2ByIncomingRayTime = Marching(geo.primitive.vertex2, geo.movingDirection, ray.time);

            // Moller Trumbore Algorithm
//          // Moller Trumbore Algorithm

            RayHitResult rayHitResult{ .materialIndex = geo.materialIndex };
//          RayHitResult rayHitResult{ .materialIndex = geo.materialIndex };

            // Small epsilon to handle floating-point inaccuracies
//          // Small epsilon to handle floating-point inaccuracies
            constexpr float EPSILON = 1e-4f;
//          constexpr float EPSILON = 1e-4f;

            // Calculate two edges of the primitive from its vertices
//          // Calculate two edges of the primitive from its vertices
            Vec3 primitiveEdge1 = currentPrimitiveVertex1ByIncomingRayTime - currentPrimitiveVertex0ByIncomingRayTime;
//          Vec3 primitiveEdge1 = currentPrimitiveVertex1ByIncomingRayTime - currentPrimitiveVertex0ByIncomingRayTime;
            Vec3 primitiveEdge2 = currentPrimitiveVertex2ByIncomingRayTime - currentPrimitiveVertex0ByIncomingRayTime;
//          Vec3 primitiveEdge2 = currentPrimitiveVertex2ByIncomingRayTime - currentPrimitiveVertex0ByIncomingRayTime;

            // Compute the determinant (used for intersection and barycentric coordinates)
//          // Compute the determinant (used for intersection and barycentric coordinates)
            Vec3 rayDirectionCrossPrimitiveEdge2 = Cross(ray.dir, primitiveEdge2);
//          Vec3 rayDirectionCrossPrimitiveEdge2 = Cross(ray.dir, primitiveEdge2);
            float determinant = Dot(primitiveEdge1, rayDirectionCrossPrimitiveEdge2);
//          float determinant = Dot(primitiveEdge1, rayDirectionCrossPrimitiveEdge2);

            // If determinant is near zero, the ray is parallel to the primitive plane (no intersection)
//          // If determinant is near zero, the ray is parallel to the primitive plane (no intersection)
            if (abs(determinant) < EPSILON)
//          if (abs(determinant) < EPSILON)
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult;
//              return rayHitResult;
            }

            float inverseDeterminant = 1.0f / determinant;
//          float inverseDeterminant = 1.0f / determinant;
            Vec3 vectorFromPrimitiveVertex0ToRayOrigin = ray.ori - currentPrimitiveVertex0ByIncomingRayTime;
//          Vec3 vectorFromPrimitiveVertex0ToRayOrigin = ray.ori - currentPrimitiveVertex0ByIncomingRayTime;

            // Compute the w1 barycentric coordinate
//          // Compute the w1 barycentric coordinate
            float w1Barycentric = inverseDeterminant * Dot(vectorFromPrimitiveVertex0ToRayOrigin, rayDirectionCrossPrimitiveEdge2);
//          float w1Barycentric = inverseDeterminant * Dot(vectorFromPrimitiveVertex0ToRayOrigin, rayDirectionCrossPrimitiveEdge2);
            if  ( w1Barycentric < 0.0f
            ||    w1Barycentric > 1.0f )
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult; // Intersection lies outside the primitive
//              return rayHitResult; // Intersection lies outside the primitive
            }

            // Compute the w2 barycentric coordinate
//          // Compute the w2 barycentric coordinate
            Vec3 rayOriginCrossPrimitiveEdge1 = Cross(vectorFromPrimitiveVertex0ToRayOrigin, primitiveEdge1);
//          Vec3 rayOriginCrossPrimitiveEdge1 = Cross(vectorFromPrimitiveVertex0ToRayOrigin, primitiveEdge1);
            float w2Barycentric = inverseDeterminant * Dot(ray.dir, rayOriginCrossPrimitiveEdge1);
//          float w2Barycentric = inverseDeterminant * Dot(ray.dir, rayOriginCrossPrimitiveEdge1);
            if  ( w2Barycentric < 0.0f
            ||    w1Barycentric +
                  w2Barycentric > 1.0f )
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult; // Intersection lies outside the primitive
//              return rayHitResult; // Intersection lies outside the primitive
            }

            // Compute the distance from ray origin to intersection point (t)
//          // Compute the distance from ray origin to intersection point (t)
            float t = inverseDeterminant * Dot(primitiveEdge2, rayOriginCrossPrimitiveEdge1);
//          float t = inverseDeterminant * Dot(primitiveEdge2, rayOriginCrossPrimitiveEdge1);

            if (!rayT.Surrounds(t))
//          if (!rayT.Surrounds(t))
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult;
//              return rayHitResult;
            }

            // Valid intersection if t > EPSILON (ignore hits behind the ray origin)
//          // Valid intersection if t > EPSILON (ignore hits behind the ray origin)
            if (t > EPSILON)
//          if (t > EPSILON)
            {
                rayHitResult.hitted = true;
//              rayHitResult.hitted = true;

                rayHitResult.minT = t;
//              rayHitResult.minT = t;

                rayHitResult.at = ray.Marching(rayHitResult.minT);
//              rayHitResult.at = ray.Marching(rayHitResult.minT);

                float w0Barycentric = 1.0f - w1Barycentric - w2Barycentric;
//              float w0Barycentric = 1.0f - w1Barycentric - w2Barycentric;
                Vec3 interpolatedFrontFaceNormal = Normalize(w0Barycentric * geo.primitive.vertex0FrontFaceNormal + w1Barycentric * geo.primitive.vertex1FrontFaceNormal + w2Barycentric * geo.primitive.vertex2FrontFaceNormal);
//              Vec3 interpolatedFrontFaceNormal = Normalize(w0Barycentric * geo.primitive.vertex0FrontFaceNormal + w1Barycentric * geo.primitive.vertex1FrontFaceNormal + w2Barycentric * geo.primitive.vertex2FrontFaceNormal);
                rayHitResult.isFrontFace = Dot(ray.dir, interpolatedFrontFaceNormal) < 0.0f;
//              rayHitResult.isFrontFace = Dot(ray.dir, interpolatedFrontFaceNormal) < 0.0f;
                if (rayHitResult.isFrontFace)
//              if (rayHitResult.isFrontFace)
                {
                    rayHitResult.normal =  interpolatedFrontFaceNormal;
//                  rayHitResult.normal =  interpolatedFrontFaceNormal;

                    rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0U + w1Barycentric * geo.primitive.frontFaceVertex1U + w2Barycentric * geo.primitive.frontFaceVertex2U;
//                  rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0U + w1Barycentric * geo.primitive.frontFaceVertex1U + w2Barycentric * geo.primitive.frontFaceVertex2U;
                    rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0V + w1Barycentric * geo.primitive.frontFaceVertex1V + w2Barycentric * geo.primitive.frontFaceVertex2V;
//                  rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0V + w1Barycentric * geo.primitive.frontFaceVertex1V + w2Barycentric * geo.primitive.frontFaceVertex2V;
                }
                else
//              else
                {
                    rayHitResult.normal = -interpolatedFrontFaceNormal;
//                  rayHitResult.normal = -interpolatedFrontFaceNormal;

                    rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive. backFaceVertex0U + w1Barycentric * geo.primitive. backFaceVertex1U + w2Barycentric * geo.primitive. backFaceVertex2U;
//                  rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive. backFaceVertex0U + w1Barycentric * geo.primitive. backFaceVertex1U + w2Barycentric * geo.primitive. backFaceVertex2U;
                    rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive. backFaceVertex0V + w1Barycentric * geo.primitive. backFaceVertex1V + w2Barycentric * geo.primitive. backFaceVertex2V;
//                  rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive. backFaceVertex0V + w1Barycentric * geo.primitive. backFaceVertex1V + w2Barycentric * geo.primitive. backFaceVertex2V;
                }

                return rayHitResult;
//              return rayHitResult;
            }
            else
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult; // Intersection is behind the ray's origin
//              return rayHitResult; // Intersection is behind the ray's origin
            }
        }
        break;
//      break;


        case GeometryType::IMPLICIT_SURFACE:
//      case GeometryType::IMPLICIT_SURFACE:
        {
            if (!HitAABB(ray, rayT, geo.aabb3d)) { return RayHitResult{ .hitted = false, .materialIndex = geo.materialIndex, }; }
//          if (!HitAABB(ray, rayT, geo.aabb3d)) { return RayHitResult{ .hitted = false, .materialIndex = geo.materialIndex, }; }
            const SDFFunctionPointer& sdf = sdfsDatabase.sdfs[geo.implicitSurface.sdfIndex];
//          const SDFFunctionPointer& sdf = sdfsDatabase.sdfs[geo.implicitSurface.sdfIndex];
            float t = rayT.min;
//          float t = rayT.min;
            for (int i = 0; i < geo.implicitSurface.maxSteps; ++i)
//          for (int i = 0; i < geo.implicitSurface.maxSteps; ++i)
            {
                Point3 currentPoint = ray.Marching(t);
//              Point3 currentPoint = ray.Marching(t);
                SDFHitResult sdfHitResult = sdf(currentPoint);
//              SDFHitResult sdfHitResult = sdf(currentPoint);

                if (sdfHitResult.distance < geo.implicitSurface.hitEpsilon)
//              if (sdfHitResult.distance < geo.implicitSurface.hitEpsilon)
                {
                    RayHitResult rayHitResult{ .materialIndex = sdfHitResult.materialIndex };
//                  RayHitResult rayHitResult{ .materialIndex = sdfHitResult.materialIndex };
                    rayHitResult.hitted = true;
//                  rayHitResult.hitted = true;
                    rayHitResult.minT = t;
//                  rayHitResult.minT = t;
                    rayHitResult.at = currentPoint;
//                  rayHitResult.at = currentPoint;

                    Vec3 outwardNormal = CalculateImplicitNormal(sdf, currentPoint);
//                  Vec3 outwardNormal = CalculateImplicitNormal(sdf, currentPoint);
                    rayHitResult.isFrontFace = Dot(ray.dir, outwardNormal) < 0.0f;
//                  rayHitResult.isFrontFace = Dot(ray.dir, outwardNormal) < 0.0f;
                    if (rayHitResult.isFrontFace)
//                  if (rayHitResult.isFrontFace)
                    {
                        rayHitResult.normal =  outwardNormal;
//                      rayHitResult.normal =  outwardNormal;
                    }
                    else
                    {
                        rayHitResult.normal = -outwardNormal;
//                      rayHitResult.normal = -outwardNormal;
                    }

                    rayHitResult.uSurfaceCoordinate = rayHitResult.at.x;
//                  rayHitResult.uSurfaceCoordinate = rayHitResult.at.x;
                    rayHitResult.vSurfaceCoordinate = rayHitResult.at.y;
//                  rayHitResult.vSurfaceCoordinate = rayHitResult.at.y;

                    SDFHitResult finalSDFHitResult = sdf(rayHitResult.at);
//                  SDFHitResult finalSDFHitResult = sdf(rayHitResult.at);
                    if (finalSDFHitResult.distance < 0.0f)
//                  if (finalSDFHitResult.distance < 0.0f)
                    {
                        rayHitResult.at -= outwardNormal * finalSDFHitResult.distance;
//                      rayHitResult.at -= outwardNormal * finalSDFHitResult.distance;
                    }

                    return rayHitResult;
//                  return rayHitResult;
                }

                t += sdfHitResult.distance;
//              t += sdfHitResult.distance;

                if (t > rayT.max || t > geo.implicitSurface.maxMarchingDistance)
//              if (t > rayT.max || t > geo.implicitSurface.maxMarchingDistance)
                {
                    break;
//                  break;
                }
            }
            return RayHitResult{ .hitted = false, .materialIndex = geo.materialIndex, };
//          return RayHitResult{ .hitted = false, .materialIndex = geo.materialIndex, };
        }
        break;
//      break;


        default:
//      default:
        {
            return { .materialIndex = geo.materialIndex };
//          return { .materialIndex = geo.materialIndex };
        }
        break;
//      break;
    }
}



        inline static RayHitResult RayHit(const std::vector<Geometry>& geometries, const Ray& ray, const Interval& rayT)
//      inline static RayHitResult RayHit(const std::vector<Geometry>& geometries, const Ray& ray, const Interval& rayT)
{
    RayHitResult finalRayHitResult{};
//  RayHitResult finalRayHitResult{};
    float closestTSoFar = rayT.max;
//  float closestTSoFar = rayT.max;
    for (const Geometry& geo : geometries)
//  for (const Geometry& geo : geometries)
    {
        RayHitResult temporaryRayHitResult = std::move(RayHit(geo, ray, Interval { .min = rayT.min, .max = closestTSoFar }));
//      RayHitResult temporaryRayHitResult = std::move(RayHit(geo, ray, Interval { .min = rayT.min, .max = closestTSoFar }));
        if (temporaryRayHitResult.hitted) [[unlikely]]
        {
            finalRayHitResult = std::move(temporaryRayHitResult);
//          finalRayHitResult = std::move(temporaryRayHitResult);
            closestTSoFar = finalRayHitResult.minT;
//          closestTSoFar = finalRayHitResult.minT;
        }
    }
    return finalRayHitResult;
//  return finalRayHitResult;
}










inline
static Color3 RayColor(const Ray& initialRay, const std::vector<Geometry>& geometries, int maxDepth = 50)
{
    Color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
//  Color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
    Ray currentRay = initialRay;
//  Ray currentRay = initialRay;

    for (int depth = 0; depth < maxDepth; ++depth)
//  for (int depth = 0; depth < maxDepth; ++depth)
    {
        const RayHitResult& rayHitResult = RayHit(geometries, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });
//      const RayHitResult& rayHitResult = RayHit(geometries, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });

        if (rayHitResult.hitted) [[unlikely]]
//      if (rayHitResult.hitted) [[unlikely]]
        {
            const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered) [[unlikely]]
//          if (!materialScatteredResult.isScattered) [[unlikely]]
            {
                return Color3{};  // Return black if scattering fails
//              return Color3{};  // Return black if scattering fails
            }

            // Multiply the current color by the attenuation
//          // Multiply the current color by the attenuation
            finalColor = finalColor * materialScatteredResult.attenuation;
//          finalColor = finalColor * materialScatteredResult.attenuation;
            // Update the ray for the next iteration
//          // Update the ray for the next iteration
            currentRay = materialScatteredResult.scatteredRay;
//          currentRay = materialScatteredResult.scatteredRay;
        }
        else
//      else
        {
            // If no hit, calculate background color and return final result
//          // If no hit, calculate background color and return final result
            const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
//          const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
            const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//          const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
            Color3 backgroundColor = BlendLinear(Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, Color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
//          Color3 backgroundColor = BlendLinear(Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, Color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
            return finalColor * backgroundColor;
//          return finalColor * backgroundColor;
        }
    }

    // If we reach max depth, return black
//  // If we reach max depth, return black
    return Color3{};
//  return Color3{};
}




    enum class MediumType : std::uint8_t
//  enum class MediumType : std::uint8_t
{
    SURFACE = 0,
//  SURFACE = 0,
    PARTICIPATING_WITH_CONSTANT_DENSITY = 1,
//  PARTICIPATING_WITH_CONSTANT_DENSITY = 1,
};




    struct BVHNode
//  struct BVHNode
{
    AABB3D aabb3d; int geometryIndex = -1; int childIndexL = -1; int childIndexR = -1;
//  AABB3D aabb3d; int geometryIndex = -1; int childIndexL = -1; int childIndexR = -1;
};
    struct BVHNodeMain
//  struct BVHNodeMain
{
    AABB3D aabb3d; int bvhTreeIndex = -1; int childIndexL = -1; int childIndexR = -1;
//  AABB3D aabb3d; int bvhTreeIndex = -1; int childIndexL = -1; int childIndexR = -1;
};
    struct BVHTree
//  struct BVHTree
{
    std::vector<BVHNode> bvhNodes; std::vector<Geometry> geometries; union { struct MediumSurface {} mediumSurface; struct MediumParticipatingWithConstantDensity { float negativeInverseDensity = - (1.0f / 1.0f); std::uint8_t materialIndex; } mediumParticipatingWithConstantDensity; }; MediumType mediumType = MediumType::SURFACE;
//  std::vector<BVHNode> bvhNodes; std::vector<Geometry> geometries; union { struct MediumSurface {} mediumSurface; struct MediumParticipatingWithConstantDensity { float negativeInverseDensity = - (1.0f / 1.0f); std::uint8_t materialIndex; } mediumParticipatingWithConstantDensity; }; MediumType mediumType = MediumType::SURFACE;
};
    struct BVHTreeMain
//  struct BVHTreeMain
{
    std::vector<BVHNodeMain> bvhNodeMains; std::vector<BVHTree> bvhTrees;
//  std::vector<BVHNodeMain> bvhNodeMains; std::vector<BVHTree> bvhTrees;
};
    inline static RayHitResult RayHit(const BVHTree& bvhTree, int bvhNodeIndex, const Ray& ray, const Interval& rayT)
//  inline static RayHitResult RayHit(const BVHTree& bvhTree, int bvhNodeIndex, const Ray& ray, const Interval& rayT)
{
    const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];
//  const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];

    // Leaf node: test geometry intersection
    // Leaf node: test geometry intersection
    if (bvhNode.geometryIndex != -1)
//  if (bvhNode.geometryIndex != -1)
    {
        return RayHit(bvhTree.geometries[bvhNode.geometryIndex], ray, rayT);
//      return RayHit(bvhTree.geometries[bvhNode.geometryIndex], ray, rayT);
    }

    // Non-leaf node: test AABB first
    // Non-leaf node: test AABB first
    if (!HitAABB(ray, rayT, bvhNode.aabb3d))
//  if (!HitAABB(ray, rayT, bvhNode.aabb3d))
    {
        RayHitResult rayHitResult{};
//      RayHitResult rayHitResult{};
        rayHitResult.hitted = false;
//      rayHitResult.hitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    // Recursively traverse children
    // Recursively traverse children
    RayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray, rayT);
//  RayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray, rayT);
    Interval updatedRayT;
//  Interval updatedRayT;
    if (rayHitResultL.hitted)
//  if (rayHitResultL.hitted)
    {
        updatedRayT = Interval{ .min = rayT.min, .max = rayHitResultL.minT };
//      updatedRayT = Interval{ .min = rayT.min, .max = rayHitResultL.minT };
    }
    else
    {
        updatedRayT = rayT;
//      updatedRayT = rayT;
    }
    RayHitResult rayHitResultR = RayHit(bvhTree, bvhNode.childIndexR, ray, updatedRayT);
//  RayHitResult rayHitResultR = RayHit(bvhTree, bvhNode.childIndexR, ray, updatedRayT);

    // Return the closest hit
    // Return the closest hit
    if (rayHitResultL.hitted
    &&  rayHitResultR.hitted)
    {
        if (rayHitResultL.minT < rayHitResultR.minT)
        {
            return rayHitResultL;
        }
        else
        {
            return rayHitResultR;
        }
    }
    else
    if (rayHitResultL.hitted)
    {
        return rayHitResultL;
    }
    else
//  if (rayHitResultR.hitted)
    {
        return rayHitResultR;
    }
}
    inline static RayHitResult RayHit(const BVHTreeMain& bvhTreeMain, int bvhNodeMainIndex, const Ray& ray, const Interval& rayT)
//  inline static RayHitResult RayHit(const BVHTreeMain& bvhTreeMain, int bvhNodeMainIndex, const Ray& ray, const Interval& rayT)
{
    const BVHNodeMain& bvhNodeMain = bvhTreeMain.bvhNodeMains[bvhNodeMainIndex];
//  const BVHNodeMain& bvhNodeMain = bvhTreeMain.bvhNodeMains[bvhNodeMainIndex];

    // Leaf node: test object-level-BVH-tree intersection
    // Leaf node: test object-level-BVH-tree intersection
    if (bvhNodeMain.bvhTreeIndex != -1)
//  if (bvhNodeMain.bvhTreeIndex != -1)
    {
        const BVHTree& bvhTree = bvhTreeMain.bvhTrees[bvhNodeMain.bvhTreeIndex];
//      const BVHTree& bvhTree = bvhTreeMain.bvhTrees[bvhNodeMain.bvhTreeIndex];
        switch (bvhTree.mediumType)
//      switch (bvhTree.mediumType)
        {
            case MediumType::SURFACE:
//          case MediumType::SURFACE:
            {
                return RayHit(bvhTree, 0, ray, rayT);
//              return RayHit(bvhTree, 0, ray, rayT);
            }
            break;
//          break;


            case MediumType::PARTICIPATING_WITH_CONSTANT_DENSITY:
//          case MediumType::PARTICIPATING_WITH_CONSTANT_DENSITY:
            {
                RayHitResult rayHitResultEntry = RayHit(bvhTree, 0, ray, Interval::universe);
//              RayHitResult rayHitResultEntry = RayHit(bvhTree, 0, ray, Interval::universe);
                if (!rayHitResultEntry.hitted) { return rayHitResultEntry; }
//              if (!rayHitResultEntry.hitted) { return rayHitResultEntry; }
                RayHitResult rayHitResultExits = RayHit(bvhTree, 0, ray, Interval{ .min = rayHitResultEntry.minT + 1e-4f, .max = positiveInfinity, });
//              RayHitResult rayHitResultExits = RayHit(bvhTree, 0, ray, Interval{ .min = rayHitResultEntry.minT + 1e-4f, .max = positiveInfinity, });
                if (!rayHitResultExits.hitted) { return rayHitResultExits; }
//              if (!rayHitResultExits.hitted) { return rayHitResultExits; }
                if (rayHitResultEntry.minT < rayT.min) { rayHitResultEntry.minT = rayT.min; }
//              if (rayHitResultEntry.minT < rayT.min) { rayHitResultEntry.minT = rayT.min; }
                if (rayHitResultExits.minT > rayT.max) { rayHitResultExits.minT = rayT.max; }
//              if (rayHitResultExits.minT > rayT.max) { rayHitResultExits.minT = rayT.max; }
                RayHitResult rayHitResult{};
//              RayHitResult rayHitResult{};
                if (rayHitResultEntry.minT >= rayHitResultExits.minT)
//              if (rayHitResultEntry.minT >= rayHitResultExits.minT)
                {
                    rayHitResult.hitted = false;
//                  rayHitResult.hitted = false;
                    return rayHitResult;
//                  return rayHitResult;
                }
                if (rayHitResultEntry.minT < 0)
//              if (rayHitResultEntry.minT < 0)
                {
                    rayHitResultEntry.minT = 0;
//                  rayHitResultEntry.minT = 0;
                }
                float rayUnitLength = ray.dir.Length();
//              float rayUnitLength = ray.dir.Length();
                float rayTravelingDistanceInsideObjectNoneParticipating = (rayHitResultExits.minT - rayHitResultEntry.minT) * rayUnitLength;
//              float rayTravelingDistanceInsideObjectNoneParticipating = (rayHitResultExits.minT - rayHitResultEntry.minT) * rayUnitLength;
                float rayTravelingDistanceInsideObjectWithParticipating = bvhTree.mediumParticipatingWithConstantDensity.negativeInverseDensity * std::logf(Random());
//              float rayTravelingDistanceInsideObjectWithParticipating = bvhTree.mediumParticipatingWithConstantDensity.negativeInverseDensity * std::logf(Random());
                if (rayTravelingDistanceInsideObjectWithParticipating > rayTravelingDistanceInsideObjectNoneParticipating)
//              if (rayTravelingDistanceInsideObjectWithParticipating > rayTravelingDistanceInsideObjectNoneParticipating)
                {
                    rayHitResult.hitted = false;
//                  rayHitResult.hitted = false;
                    return rayHitResult;
//                  return rayHitResult;
                }
                rayHitResult.hitted = true;
//              rayHitResult.hitted = true;
                rayHitResult.materialIndex = bvhTree.mediumParticipatingWithConstantDensity.materialIndex;
//              rayHitResult.materialIndex = bvhTree.mediumParticipatingWithConstantDensity.materialIndex;
                rayHitResult.minT = rayHitResultEntry.minT + rayTravelingDistanceInsideObjectWithParticipating / rayUnitLength;
//              rayHitResult.minT = rayHitResultEntry.minT + rayTravelingDistanceInsideObjectWithParticipating / rayUnitLength;
                rayHitResult.at = Marching(ray.ori, ray.dir, rayHitResult.minT);
//              rayHitResult.at = Marching(ray.ori, ray.dir, rayHitResult.minT);
                rayHitResult.normal = rayHitResultEntry.normal;
//              rayHitResult.normal = rayHitResultEntry.normal;
                rayHitResult.isFrontFace = rayHitResultEntry.isFrontFace;
//              rayHitResult.isFrontFace = rayHitResultEntry.isFrontFace;
                rayHitResult.uSurfaceCoordinate = rayHitResultEntry.uSurfaceCoordinate;
//              rayHitResult.uSurfaceCoordinate = rayHitResultEntry.uSurfaceCoordinate;
                rayHitResult.vSurfaceCoordinate = rayHitResultEntry.vSurfaceCoordinate;
//              rayHitResult.vSurfaceCoordinate = rayHitResultEntry.vSurfaceCoordinate;
                return rayHitResult;
//              return rayHitResult;
            }
            break;
//          break;


            default:
//          default:
            {
                return RayHit(bvhTree, 0, ray, rayT);
//              return RayHit(bvhTree, 0, ray, rayT);
            }
            break;
//          break;
        }
    }

    // Non-leaf node: test AABB first
    // Non-leaf node: test AABB first
    if (!HitAABB(ray, rayT, bvhNodeMain.aabb3d))
//  if (!HitAABB(ray, rayT, bvhNodeMain.aabb3d))
    {
        RayHitResult rayHitResult{};
//      RayHitResult rayHitResult{};
        rayHitResult.hitted = false;
//      rayHitResult.hitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    // Recursively traverse children
    // Recursively traverse children
    RayHitResult rayHitResultL = RayHit(bvhTreeMain, bvhNodeMain.childIndexL, ray, rayT);
//  RayHitResult rayHitResultL = RayHit(bvhTreeMain, bvhNodeMain.childIndexL, ray, rayT);
    Interval updatedRayT;
//  Interval updatedRayT;
    if (rayHitResultL.hitted)
//  if (rayHitResultL.hitted)
    {
        updatedRayT = Interval{ .min = rayT.min, .max = rayHitResultL.minT };
//      updatedRayT = Interval{ .min = rayT.min, .max = rayHitResultL.minT };
    }
    else
    {
        updatedRayT = rayT;
//      updatedRayT = rayT;
    }
    RayHitResult rayHitResultR = RayHit(bvhTreeMain, bvhNodeMain.childIndexR, ray, updatedRayT);
//  RayHitResult rayHitResultR = RayHit(bvhTreeMain, bvhNodeMain.childIndexR, ray, updatedRayT);

    // Return the closest hit
    // Return the closest hit
    if (rayHitResultL.hitted
    &&  rayHitResultR.hitted)
    {
        if (rayHitResultL.minT < rayHitResultR.minT)
        {
            return rayHitResultL;
        }
        else
        {
            return rayHitResultR;
        }
    }
    else
    if (rayHitResultL.hitted)
    {
        return rayHitResultL;
    }
    else
//  if (rayHitResultR.hitted)
    {
        return rayHitResultR;
    }
}
enum class Axis : std::uint8_t
{
    X = +0,
    Y = +1,
    Z = +2,
    _ = +3,
};
//  Calculate the centroid of a geometry's AABB along a specific axis
//  Calculate the centroid of a geometry's AABB along a specific axis
    static inline float GetCentroid(const Geometry& geo, const Axis& axis)
//  static inline float GetCentroid(const Geometry& geo, const Axis& axis)
{
        switch (axis)
        {
        case Axis::X:
            return (geo.aabb3d.intervalAxisX.min + geo.aabb3d.intervalAxisX.max) / 2.0f;
        case Axis::Y:
            return (geo.aabb3d.intervalAxisY.min + geo.aabb3d.intervalAxisY.max) / 2.0f;
        case Axis::Z:
            return (geo.aabb3d.intervalAxisZ.min + geo.aabb3d.intervalAxisZ.max) / 2.0f;
        default:
            return 0;
        }
}

    static inline float GetCentroid(const BVHTree& bvhTree, const Axis& axis)
//  static inline float GetCentroid(const BVHTree& bvhTree, const Axis& axis)
{
        switch (axis)
        {
        case Axis::X:
            return (bvhTree.bvhNodes[0].aabb3d.intervalAxisX.min + bvhTree.bvhNodes[0].aabb3d.intervalAxisX.max) / 2.0f;
        case Axis::Y:
            return (bvhTree.bvhNodes[0].aabb3d.intervalAxisY.min + bvhTree.bvhNodes[0].aabb3d.intervalAxisY.max) / 2.0f;
        case Axis::Z:
            return (bvhTree.bvhNodes[0].aabb3d.intervalAxisZ.min + bvhTree.bvhNodes[0].aabb3d.intervalAxisZ.max) / 2.0f;
        default:
            return 0;
        }
}

//  Calculate the surface area of an AABB3D
//  Calculate the surface area of an AABB3D
    static inline float SurfaceArea(const AABB3D& aabb3d)
//  static inline float SurfaceArea(const AABB3D& aabb3d)
{
        float  width = aabb3d.intervalAxisX.max - aabb3d.intervalAxisX.min;
//      float  width = aabb3d.intervalAxisX.max - aabb3d.intervalAxisX.min;
        float height = aabb3d.intervalAxisY.max - aabb3d.intervalAxisY.min;
//      float height = aabb3d.intervalAxisY.max - aabb3d.intervalAxisY.min;
        float  depth = aabb3d.intervalAxisZ.max - aabb3d.intervalAxisZ.min;
//      float  depth = aabb3d.intervalAxisZ.max - aabb3d.intervalAxisZ.min;
        return 2.0f * (width * height + width * depth + height * depth);
//      return 2.0f * (width * height + width * depth + height * depth);
}

//  Compute the union of two AABB3Ds
//  Compute the union of two AABB3Ds
    static inline void Union(const AABB3D& aabb3d1, const AABB3D& aabb3d2, AABB3D& aabb3dResult)
//  static inline void Union(const AABB3D& aabb3d1, const AABB3D& aabb3d2, AABB3D& aabb3dResult)
{
        aabb3dResult.intervalAxisX.min = std::fminf(aabb3d1.intervalAxisX.min, aabb3d2.intervalAxisX.min);
        aabb3dResult.intervalAxisX.max = std::fmaxf(aabb3d1.intervalAxisX.max, aabb3d2.intervalAxisX.max);
        aabb3dResult.intervalAxisY.min = std::fminf(aabb3d1.intervalAxisY.min, aabb3d2.intervalAxisY.min);
        aabb3dResult.intervalAxisY.max = std::fmaxf(aabb3d1.intervalAxisY.max, aabb3d2.intervalAxisY.max);
        aabb3dResult.intervalAxisZ.min = std::fminf(aabb3d1.intervalAxisZ.min, aabb3d2.intervalAxisZ.min);
        aabb3dResult.intervalAxisZ.max = std::fmaxf(aabb3d1.intervalAxisZ.max, aabb3d2.intervalAxisZ.max);
}
//  SAH BVH
//  SAH BVH
    inline static int BuildBVHTree(BVHTree& bvhTree, int start, int cease)
//  inline static int BuildBVHTree(BVHTree& bvhTree, int start, int cease)
{
        int objectSpan = cease - start;
//      int objectSpan = cease - start;

        // Base case: create a leaf node with a single geometry
        // Base case: create a leaf node with a single geometry
        if (objectSpan == 1)
//      if (objectSpan == 1)
        {
            int current = static_cast<int>(bvhTree.bvhNodes.size());
//          int current = static_cast<int>(bvhTree.bvhNodes.size());
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .geometryIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .geometryIndex = start, .childIndexL = -1, .childIndexR = -1, });
            return current;
//          return current;
        }

        // Variables to track the best split
        // Variables to track the best split
        float bestCost =  std::numeric_limits<float>::infinity(); // Assuming positiveInfinity is not defined
//      float bestCost =  std::numeric_limits<float>::infinity(); // Assuming positiveInfinity is not defined
        Axis  bestAxis = Axis::_;
//      Axis  bestAxis = Axis::_;
        int   bestIndexToSplit = -1;
//      int   bestIndexToSplit = -1;

        // Evaluate splits along each axis ( x = 0 , y = 1 , z = 2 )
        // Evaluate splits along each axis ( x = 0 , y = 1 , z = 2 )
        for (int axis = 0; axis < 3; ++axis)
//      for (int axis = 0; axis < 3; ++axis)
        {
            // Sort geometries based on centroid along the current axis
            // Sort geometries based on centroid along the current axis
            std::function<bool(const Geometry& geo1, const Geometry& geo2)> comparator = [axis]
//          std::function<bool(const Geometry& geo1, const Geometry& geo2)> comparator = [axis]
                              (const Geometry& geo1, const Geometry& geo2)
//                            (const Geometry& geo1, const Geometry& geo2)
                        ->bool{ return GetCentroid(geo1, Axis(axis))
//                      ->bool{ return GetCentroid(geo1, Axis(axis))
                         <             GetCentroid(geo2, Axis(axis));
//                       <             GetCentroid(geo2, Axis(axis));
                              };
//                            };
            std::sort(std::begin(bvhTree.geometries) + start ,
//          std::sort(std::begin(bvhTree.geometries) + start ,
                      std::begin(bvhTree.geometries) + cease , comparator);
//                    std::begin(bvhTree.geometries) + cease , comparator);

            // Compute cumulative AABB3Ds from the left!
            // Compute cumulative AABB3Ds from the left!
            std::vector<AABB3D> lAABB3Ds(objectSpan);
//          std::vector<AABB3D> lAABB3Ds(objectSpan);
            lAABB3Ds[0].intervalAxisX.min = bvhTree.geometries[start].aabb3d.intervalAxisX.min;
            lAABB3Ds[0].intervalAxisX.max = bvhTree.geometries[start].aabb3d.intervalAxisX.max;
            lAABB3Ds[0].intervalAxisY.min = bvhTree.geometries[start].aabb3d.intervalAxisY.min;
            lAABB3Ds[0].intervalAxisY.max = bvhTree.geometries[start].aabb3d.intervalAxisY.max;
            lAABB3Ds[0].intervalAxisZ.min = bvhTree.geometries[start].aabb3d.intervalAxisZ.min;
            lAABB3Ds[0].intervalAxisZ.max = bvhTree.geometries[start].aabb3d.intervalAxisZ.max;
            for (int i = 1; i < objectSpan; ++i)
//          for (int i = 1; i < objectSpan; ++i)
            {
                Union(lAABB3Ds[static_cast<std::size_t>(i - 1)], bvhTree.geometries[static_cast<std::size_t>(start + i)].aabb3d, lAABB3Ds[i]);
//              Union(lAABB3Ds[static_cast<std::size_t>(i - 1)], bvhTree.geometries[static_cast<std::size_t>(start + i)].aabb3d, lAABB3Ds[i]);
            }

            // Compute cumulative AABB3Ds from the right
            // Compute cumulative AABB3Ds from the right
            std::vector<AABB3D> rAABB3Ds(objectSpan);
//          std::vector<AABB3D> rAABB3Ds(objectSpan);
            int r1 = objectSpan - 1;
//          int r1 = objectSpan - 1;
            int r2 = cease      - 1;
//          int r2 = cease      - 1;
            rAABB3Ds[r1].intervalAxisX.min = bvhTree.geometries[r2].aabb3d.intervalAxisX.min;
            rAABB3Ds[r1].intervalAxisX.max = bvhTree.geometries[r2].aabb3d.intervalAxisX.max;
            rAABB3Ds[r1].intervalAxisY.min = bvhTree.geometries[r2].aabb3d.intervalAxisY.min;
            rAABB3Ds[r1].intervalAxisY.max = bvhTree.geometries[r2].aabb3d.intervalAxisY.max;
            rAABB3Ds[r1].intervalAxisZ.min = bvhTree.geometries[r2].aabb3d.intervalAxisZ.min;
            rAABB3Ds[r1].intervalAxisZ.max = bvhTree.geometries[r2].aabb3d.intervalAxisZ.max;
            for (int i = objectSpan - 2; i >= 0; --i)
//          for (int i = objectSpan - 2; i >= 0; --i)
            {
                Union(bvhTree.geometries[static_cast<std::size_t>(start + i)].aabb3d, rAABB3Ds[static_cast<std::size_t>(i + 1)], rAABB3Ds[i]);
//              Union(bvhTree.geometries[static_cast<std::size_t>(start + i)].aabb3d, rAABB3Ds[static_cast<std::size_t>(i + 1)], rAABB3Ds[i]);
            }

            // Evaluate all possible splits
            // Evaluate all possible splits
            for (int i = 0; i < objectSpan - 1; ++i)
//          for (int i = 0; i < objectSpan - 1; ++i)
            {
                float   cost = SurfaceArea(lAABB3Ds[                         i     ]) * (             i + 1)
                             + SurfaceArea(rAABB3Ds[static_cast<std::size_t>(i + 1)]) * (objectSpan - i - 1);
                if (    cost <
                    bestCost)
                {
                    bestCost =      cost ;
//                  bestCost =      cost ;
                    bestAxis = Axis(axis);
//                  bestAxis = Axis(axis);
                    bestIndexToSplit = i ;
//                  bestIndexToSplit = i ;
                }
            }
        }

        // If no valid split is found (shouldn't happen), create a leaf node as fallback
        // If no valid split is found (shouldn't happen), create a leaf node as fallback
        if (bestAxis == Axis::_)
        {
            int current = static_cast<int>(bvhTree.bvhNodes.size());
//          int current = static_cast<int>(bvhTree.bvhNodes.size());
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .geometryIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .geometryIndex = start, .childIndexL = -1, .childIndexR = -1, });
            return current;
//          return current;
        }

        // Apply the best split
        // Apply the best split
        std::function<bool(const Geometry& geo1, const Geometry& geo2)> bestComparator = [bestAxis]
//      std::function<bool(const Geometry& geo1, const Geometry& geo2)> bestComparator = [bestAxis]
                          (const Geometry& geo1, const Geometry& geo2)
//                        (const Geometry& geo1, const Geometry& geo2)
                    ->bool{      return GetCentroid(geo1, bestAxis)
//                  ->bool{      return GetCentroid(geo1, bestAxis)
                     <                  GetCentroid(geo2, bestAxis);
//                   <                  GetCentroid(geo2, bestAxis);
                          };
//                        };
        std::sort(std::begin(bvhTree.geometries) + start,
//      std::sort(std::begin(bvhTree.geometries) + start,
                  std::begin(bvhTree.geometries) + cease, bestComparator);
//                std::begin(bvhTree.geometries) + cease, bestComparator);
        int mid = start + bestIndexToSplit + 1;
//      int mid = start + bestIndexToSplit + 1;

        // Create an internal node
        // Create an internal node
        int current = static_cast<int>(bvhTree.bvhNodes.size());
//      int current = static_cast<int>(bvhTree.bvhNodes.size());
        bvhTree.bvhNodes.emplace_back(BVHNode{  });
//      bvhTree.bvhNodes.emplace_back(BVHNode{  });

        // Recursively build left and right subtrees
        // Recursively build left and right subtrees
        int childIndexL = BuildBVHTree(bvhTree, start, mid       );
//      int childIndexL = BuildBVHTree(bvhTree, start, mid       );
        int childIndexR = BuildBVHTree(bvhTree,        mid, cease);
//      int childIndexR = BuildBVHTree(bvhTree,        mid, cease);

        // Set up the internal node
        // Set up the internal node
        const BVHNode& bvhNodeL = bvhTree.bvhNodes[childIndexL];
        const BVHNode& bvhNodeR = bvhTree.bvhNodes[childIndexR];
        Union(bvhNodeL.aabb3d, bvhNodeR.aabb3d, bvhTree.bvhNodes[current].aabb3d);
//      Union(bvhNodeL.aabb3d, bvhNodeR.aabb3d, bvhTree.bvhNodes[current].aabb3d);
        bvhTree.bvhNodes[current].geometryIndex = -1;
//      bvhTree.bvhNodes[current].geometryIndex = -1;
        bvhTree.bvhNodes[current].childIndexL = childIndexL;
//      bvhTree.bvhNodes[current].childIndexL = childIndexL;
        bvhTree.bvhNodes[current].childIndexR = childIndexR;
//      bvhTree.bvhNodes[current].childIndexR = childIndexR;

        return current;
//      return current;
}
    inline static int BuildBVHTree(BVHTreeMain& bvhTreeMain, int start, int cease)
//  inline static int BuildBVHTree(BVHTreeMain& bvhTreeMain, int start, int cease)
{
        int objectSpan = cease - start;
//      int objectSpan = cease - start;

        // Base case: create a leaf node with a single object-level-BVH-tree
        // Base case: create a leaf node with a single object-level-BVH-tree
        if (objectSpan == 1)
//      if (objectSpan == 1)
        {
            int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
//          int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
            bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{ .aabb3d = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d, .bvhTreeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{ .aabb3d = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d, .bvhTreeIndex = start, .childIndexL = -1, .childIndexR = -1, });
            return current;
//          return current;
        }

        // Variables to track the best split
        // Variables to track the best split
        float bestCost =  std::numeric_limits<float>::infinity(); // Assuming positiveInfinity is not defined
//      float bestCost =  std::numeric_limits<float>::infinity(); // Assuming positiveInfinity is not defined
        Axis  bestAxis = Axis::_;
//      Axis  bestAxis = Axis::_;
        int   bestIndexToSplit = -1;
//      int   bestIndexToSplit = -1;

        // Evaluate splits along each axis ( x = 0 , y = 1 , z = 2 )
        // Evaluate splits along each axis ( x = 0 , y = 1 , z = 2 )
        for (int axis = 0; axis < 3; ++axis)
//      for (int axis = 0; axis < 3; ++axis)
        {
            // Sort geometries based on centroid along the current axis
            // Sort geometries based on centroid along the current axis
            std::function<bool(const BVHTree& bvhTree1, const BVHTree& bvhTree2)> comparator = [axis]
//          std::function<bool(const BVHTree& bvhTree1, const BVHTree& bvhTree2)> comparator = [axis]
                              (const BVHTree& bvhTree1, const BVHTree& bvhTree2)
//                            (const BVHTree& bvhTree1, const BVHTree& bvhTree2)
                        ->bool{ return GetCentroid(bvhTree1, Axis(axis))
//                      ->bool{ return GetCentroid(bvhTree1, Axis(axis))
                         <             GetCentroid(bvhTree2, Axis(axis));
//                       <             GetCentroid(bvhTree2, Axis(axis));
                              };
//                            };
            std::sort(std::begin(bvhTreeMain.bvhTrees) + start ,
//          std::sort(std::begin(bvhTreeMain.bvhTrees) + start ,
                      std::begin(bvhTreeMain.bvhTrees) + cease , comparator);
//                    std::begin(bvhTreeMain.bvhTrees) + cease , comparator);

            // Compute cumulative AABB3Ds from the left!
            // Compute cumulative AABB3Ds from the left!
            std::vector<AABB3D> lAABB3Ds(objectSpan);
//          std::vector<AABB3D> lAABB3Ds(objectSpan);
            lAABB3Ds[0].intervalAxisX.min = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisX.min;
            lAABB3Ds[0].intervalAxisX.max = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisX.max;
            lAABB3Ds[0].intervalAxisY.min = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisY.min;
            lAABB3Ds[0].intervalAxisY.max = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisY.max;
            lAABB3Ds[0].intervalAxisZ.min = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisZ.min;
            lAABB3Ds[0].intervalAxisZ.max = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d.intervalAxisZ.max;
            for (int i = 1; i < objectSpan; ++i)
//          for (int i = 1; i < objectSpan; ++i)
            {
                Union(lAABB3Ds[static_cast<std::size_t>(i - 1)], bvhTreeMain.bvhTrees[static_cast<std::size_t>(start + i)].bvhNodes[0].aabb3d, lAABB3Ds[i]);
//              Union(lAABB3Ds[static_cast<std::size_t>(i - 1)], bvhTreeMain.bvhTrees[static_cast<std::size_t>(start + i)].bvhNodes[0].aabb3d, lAABB3Ds[i]);
            }

            // Compute cumulative AABB3Ds from the right
            // Compute cumulative AABB3Ds from the right
            std::vector<AABB3D> rAABB3Ds(objectSpan);
//          std::vector<AABB3D> rAABB3Ds(objectSpan);
            int r1 = objectSpan - 1;
//          int r1 = objectSpan - 1;
            int r2 = cease      - 1;
//          int r2 = cease      - 1;
            rAABB3Ds[r1].intervalAxisX.min = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisX.min;
            rAABB3Ds[r1].intervalAxisX.max = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisX.max;
            rAABB3Ds[r1].intervalAxisY.min = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisY.min;
            rAABB3Ds[r1].intervalAxisY.max = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisY.max;
            rAABB3Ds[r1].intervalAxisZ.min = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisZ.min;
            rAABB3Ds[r1].intervalAxisZ.max = bvhTreeMain.bvhTrees[r2].bvhNodes[0].aabb3d.intervalAxisZ.max;
            for (int i = objectSpan - 2; i >= 0; --i)
//          for (int i = objectSpan - 2; i >= 0; --i)
            {
                Union(bvhTreeMain.bvhTrees[static_cast<std::size_t>(start + i)].bvhNodes[0].aabb3d, rAABB3Ds[static_cast<std::size_t>(i + 1)], rAABB3Ds[i]);
//              Union(bvhTreeMain.bvhTrees[static_cast<std::size_t>(start + i)].bvhNodes[0].aabb3d, rAABB3Ds[static_cast<std::size_t>(i + 1)], rAABB3Ds[i]);
            }

            // Evaluate all possible splits
            // Evaluate all possible splits
            for (int i = 0; i < objectSpan - 1; ++i)
//          for (int i = 0; i < objectSpan - 1; ++i)
            {
                float   cost = SurfaceArea(lAABB3Ds[                         i     ]) * (             i + 1)
                             + SurfaceArea(rAABB3Ds[static_cast<std::size_t>(i + 1)]) * (objectSpan - i - 1);
                if (    cost <
                    bestCost)
                {
                    bestCost =      cost ;
//                  bestCost =      cost ;
                    bestAxis = Axis(axis);
//                  bestAxis = Axis(axis);
                    bestIndexToSplit = i ;
//                  bestIndexToSplit = i ;
                }
            }
        }

        // If no valid split is found (shouldn't happen), create a leaf node as fallback
        // If no valid split is found (shouldn't happen), create a leaf node as fallback
        if (bestAxis == Axis::_)
        {
            int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
//          int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
            bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{ .aabb3d = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d, .bvhTreeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{ .aabb3d = bvhTreeMain.bvhTrees[start].bvhNodes[0].aabb3d, .bvhTreeIndex = start, .childIndexL = -1, .childIndexR = -1, });
            return current;
//          return current;
        }

        // Apply the best split
        // Apply the best split
        std::function<bool(const BVHTree& bvhTree1, const BVHTree& bvhTree2)> bestComparator = [bestAxis]
//      std::function<bool(const BVHTree& bvhTree1, const BVHTree& bvhTree2)> bestComparator = [bestAxis]
                          (const BVHTree& bvhTree1, const BVHTree& bvhTree2)
//                        (const BVHTree& bvhTree1, const BVHTree& bvhTree2)
                    ->bool{      return GetCentroid(bvhTree1, bestAxis)
//                  ->bool{      return GetCentroid(bvhTree1, bestAxis)
                     <                  GetCentroid(bvhTree2, bestAxis);
//                   <                  GetCentroid(bvhTree2, bestAxis);
                          };
//                        };
        std::sort(std::begin(bvhTreeMain.bvhTrees) + start,
//      std::sort(std::begin(bvhTreeMain.bvhTrees) + start,
                  std::begin(bvhTreeMain.bvhTrees) + cease, bestComparator);
//                std::begin(bvhTreeMain.bvhTrees) + cease, bestComparator);
        int mid = start + bestIndexToSplit + 1;
//      int mid = start + bestIndexToSplit + 1;

        // Create an internal node
        // Create an internal node
        int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
//      int current = static_cast<int>(bvhTreeMain.bvhNodeMains.size());
        bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{  });
//      bvhTreeMain.bvhNodeMains.emplace_back(BVHNodeMain{  });

        // Recursively build left and right subtrees
        // Recursively build left and right subtrees
        int childIndexL = BuildBVHTree(bvhTreeMain, start, mid       );
//      int childIndexL = BuildBVHTree(bvhTreeMain, start, mid       );
        int childIndexR = BuildBVHTree(bvhTreeMain,        mid, cease);
//      int childIndexR = BuildBVHTree(bvhTreeMain,        mid, cease);

        // Set up the internal node
        // Set up the internal node
        const BVHNodeMain& bvhNodeMainL = bvhTreeMain.bvhNodeMains[childIndexL];
        const BVHNodeMain& bvhNodeMainR = bvhTreeMain.bvhNodeMains[childIndexR];
        Union(bvhNodeMainL.aabb3d, bvhNodeMainR.aabb3d, bvhTreeMain.bvhNodeMains[current].aabb3d);
//      Union(bvhNodeMainL.aabb3d, bvhNodeMainR.aabb3d, bvhTreeMain.bvhNodeMains[current].aabb3d);
        bvhTreeMain.bvhNodeMains[current].bvhTreeIndex = -1;
//      bvhTreeMain.bvhNodeMains[current].bvhTreeIndex = -1;
        bvhTreeMain.bvhNodeMains[current].childIndexL = childIndexL;
        bvhTreeMain.bvhNodeMains[current].childIndexR = childIndexR;

        return current;
//      return current;
}
    inline static Color3 RayColor(const Ray& initialRay, const BVHTree& bvhTree, int maxDepth, BackgroundType backgroundType)
//  inline static Color3 RayColor(const Ray& initialRay, const BVHTree& bvhTree, int maxDepth, BackgroundType backgroundType)
{
        Color3 accumulatedColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
//      Color3 accumulatedColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
        Color3 attenuation      = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
//      Color3 attenuation      = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
        Ray currentRay = initialRay;
//      Ray currentRay = initialRay;

        for (int depth = 0; depth < maxDepth; ++depth)
//      for (int depth = 0; depth < maxDepth; ++depth)
        {
            const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });
//          const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });

            if (!rayHitResult.hitted) [[unlikely]]
//          if (!rayHitResult.hitted) [[unlikely]]
            {
                Color3  backgroundColor;
//              Color3  backgroundColor;
                switch (backgroundType)
//              switch (backgroundType)
                {
                    case BackgroundType::BLUE_LERP_WHITE:
//                  case BackgroundType::BLUE_LERP_WHITE:
                    {
                        const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
//                      const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
                        const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//                      const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
                        backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
//                      backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
                    }
                    break;
//                  break;


                    case BackgroundType::DARK_ROOM_SPACE:
//                  case BackgroundType::DARK_ROOM_SPACE:
                    {
                        backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
//                      backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
                    }
                    break;
//                  break;


                    default:
//                  default:
                    {
                        backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
//                      backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
                    }
                    break;
//                  break;
                }

                accumulatedColor += attenuation * backgroundColor;
//              accumulatedColor += attenuation * backgroundColor;
                break;
//              break;
            }

            const MaterialScatteredResult& scatterResult = Scatter(currentRay, rayHitResult);
//          const MaterialScatteredResult& scatterResult = Scatter(currentRay, rayHitResult);

            accumulatedColor += attenuation * scatterResult.emission;
//          accumulatedColor += attenuation * scatterResult.emission;

            if (!scatterResult.isScattered) [[unlikely]]
//          if (!scatterResult.isScattered) [[unlikely]]
            {
                break;
//              break;
            }

            attenuation *= scatterResult.attenuation ;
//          attenuation *= scatterResult.attenuation ;
            currentRay   = scatterResult.scatteredRay;
//          currentRay   = scatterResult.scatteredRay;
        }

        return accumulatedColor;
//      return accumulatedColor;
}
    #if defined(FOG_HEIGHT)
//  #if defined(FOG_HEIGHT)
    static inline Color3 ApplyFogHeight(const Color3& accumulatedColor, float minT, const Ray& initialRay, const Color3& fogColor, float fogDensity, float fogFalloff)
//  static inline Color3 ApplyFogHeight(const Color3& accumulatedColor, float minT, const Ray& initialRay, const Color3& fogColor, float fogDensity, float fogFalloff)
    {
        float fogAmount;
//      float fogAmount;
        if (std::fabsf(initialRay.dir.y) < 1e-4f)
//      if (std::fabsf(initialRay.dir.y) < 1e-4f)
        {
            fogAmount = fogDensity * std::expf(-initialRay.ori.y * fogFalloff) * minT;
//          fogAmount = fogDensity * std::expf(-initialRay.ori.y * fogFalloff) * minT;
        }
        else
//      else
        {
            fogAmount = (fogDensity / fogFalloff) * (std::expf(-initialRay.ori.y * fogFalloff)) * ((1.0f - std::expf(-minT * initialRay.dir.y * fogFalloff)) / initialRay.dir.y);
//          fogAmount = (fogDensity / fogFalloff) * (std::expf(-initialRay.ori.y * fogFalloff)) * ((1.0f - std::expf(-minT * initialRay.dir.y * fogFalloff)) / initialRay.dir.y);
        }
        // CRITICAL: Clamp the fog amount to [0 ; 1] to ensure the mix is valid. The formula can produce values > 1 for very long distances or certain parameters.
        // CRITICAL: Clamp the fog amount to [0 ; 1] to ensure the mix is valid. The formula can produce values > 1 for very long distances or certain parameters.
        fogAmount = std::clamp(fogAmount, 0.0f, 1.0f);
//      fogAmount = std::clamp(fogAmount, 0.0f, 1.0f);
        return BlendLinear(accumulatedColor, fogColor, fogAmount);
//      return BlendLinear(accumulatedColor, fogColor, fogAmount);
    }
    #endif
//  #endif
    inline static Color3 RayColor(const Ray& initialRay, const BVHTreeMain& bvhTreeMain, int maxDepth, BackgroundType backgroundType, Color3& normal, Color3& albedo)
//  inline static Color3 RayColor(const Ray& initialRay, const BVHTreeMain& bvhTreeMain, int maxDepth, BackgroundType backgroundType, Color3& normal, Color3& albedo)
{
        Color3 accumulatedColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
//      Color3 accumulatedColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
        Color3 attenuation      = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
//      Color3 attenuation      = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
        Ray currentRay = initialRay;
//      Ray currentRay = initialRay;

    #if defined(FOG_SIMPLE) || defined(FOG_HEIGHT)
//  #if defined(FOG_SIMPLE) || defined(FOG_HEIGHT)
        RayHitResult firstRayHitResult{};
//      RayHitResult firstRayHitResult{};
    #else
//  #else
        RayHitResult firstRayHitResult{};
//      RayHitResult firstRayHitResult{};
    #endif
//  #endif

        for (int depth = 0; depth < maxDepth; ++depth)
//      for (int depth = 0; depth < maxDepth; ++depth)
        {
            const RayHitResult& rayHitResult = RayHit(bvhTreeMain, 0, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });
//          const RayHitResult& rayHitResult = RayHit(bvhTreeMain, 0, currentRay, Interval{ .min = 1e-4f, .max = positiveInfinity });

    #if defined(FOG_SIMPLE) || defined(FOG_HEIGHT)
//  #if defined(FOG_SIMPLE) || defined(FOG_HEIGHT)
            if (depth == 0) { firstRayHitResult = rayHitResult; }
//          if (depth == 0) { firstRayHitResult = rayHitResult; }
    #else
//  #else
            if (depth == 0) { firstRayHitResult = rayHitResult; }
//          if (depth == 0) { firstRayHitResult = rayHitResult; }
    #endif
//  #endif

            if (!rayHitResult.hitted) [[unlikely]]
//          if (!rayHitResult.hitted) [[unlikely]]
            {
                Color3  backgroundColor;
//              Color3  backgroundColor;
                switch (backgroundType)
//              switch (backgroundType)
                {
                    case BackgroundType::BLUE_LERP_WHITE:
//                  case BackgroundType::BLUE_LERP_WHITE:
                    {
                        const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
//                      const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
                        const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//                      const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
                        backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
//                      backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
                    }
                    break;
//                  break;


                    case BackgroundType::DARK_ROOM_SPACE:
//                  case BackgroundType::DARK_ROOM_SPACE:
                    {
                        backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
//                      backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
                    }
                    break;
//                  break;


                    case BackgroundType::SKY_BOX:
//                  case BackgroundType::SKY_BOX:
                    {
                        float theta = std::acos (-currentRay.dir.y); // latitude
//                      float theta = std::acos (-currentRay.dir.y); // latitude
                        float phi   = std::atan2(-currentRay.dir.z, currentRay.dir.x) + std::numbers::pi_v<float>; // longitude
//                      float phi   = std::atan2(-currentRay.dir.z, currentRay.dir.x) + std::numbers::pi_v<float>; // longitude

                        float u = phi   / (2.0f * std::numbers::pi_v<float>);
//                      float u = phi   / (2.0f * std::numbers::pi_v<float>);
                        float v = theta /         std::numbers::pi_v<float> ;
//                      float v = theta /         std::numbers::pi_v<float> ;

                        u =        std::clamp(u, 0.0f, 1.0f);
//                      u =        std::clamp(u, 0.0f, 1.0f);
                        v = 1.0f - std::clamp(v, 0.0f, 1.0f);
//                      v = 1.0f - std::clamp(v, 0.0f, 1.0f);

                        float imagePixelX = u * (imagesDatabase.imageBasedLighting.w - 1);
                        float imagePixelY = v * (imagesDatabase.imageBasedLighting.h - 1);

                        backgroundColor = SampleRGB2LinearInterpolation(imagesDatabase.imageBasedLighting.rgbs, imagesDatabase.imageBasedLighting.w, imagesDatabase.imageBasedLighting.h, imagePixelX, imagePixelY);
//                      backgroundColor = SampleRGB2LinearInterpolation(imagesDatabase.imageBasedLighting.rgbs, imagesDatabase.imageBasedLighting.w, imagesDatabase.imageBasedLighting.h, imagePixelX, imagePixelY);
                    }
                    break;
//                  break;


                    default:
//                  default:
                    {
                        backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
//                      backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
                    }
                    break;
//                  break;
                }

                accumulatedColor += attenuation * backgroundColor;
//              accumulatedColor += attenuation * backgroundColor;
                break;
//              break;
            }

            const MaterialScatteredResult& scatterResult = Scatter(currentRay, rayHitResult);
//          const MaterialScatteredResult& scatterResult = Scatter(currentRay, rayHitResult);

            accumulatedColor += attenuation * scatterResult.emission;
//          accumulatedColor += attenuation * scatterResult.emission;

            if (!scatterResult.isScattered) [[unlikely]]
//          if (!scatterResult.isScattered) [[unlikely]]
            {
                break;
//              break;
            }

            attenuation *= scatterResult.attenuation ;
//          attenuation *= scatterResult.attenuation ;
            currentRay   = scatterResult.scatteredRay;
//          currentRay   = scatterResult.scatteredRay;
        }

    #if defined(FOG_SIMPLE)
//  #if defined(FOG_SIMPLE)
        Color3 exColor
//      Color3 exColor
        {
            .x = std::expf(-firstRayHitResult.minT * _EX_SCATTERING_FOG_SIMPLE_DENSITY_.x),
            .y = std::expf(-firstRayHitResult.minT * _EX_SCATTERING_FOG_SIMPLE_DENSITY_.y),
            .z = std::expf(-firstRayHitResult.minT * _EX_SCATTERING_FOG_SIMPLE_DENSITY_.z),
        };
        Color3 inColor
//      Color3 inColor
        {
            .x = std::expf(-firstRayHitResult.minT * _IN_SCATTERING_FOG_SIMPLE_DENSITY_.x),
            .y = std::expf(-firstRayHitResult.minT * _IN_SCATTERING_FOG_SIMPLE_DENSITY_.y),
            .z = std::expf(-firstRayHitResult.minT * _IN_SCATTERING_FOG_SIMPLE_DENSITY_.z),
        };
        return accumulatedColor * exColor + _FOG_SIMPLE_COLOR_ * (1.0f - inColor);
//      return accumulatedColor * exColor + _FOG_SIMPLE_COLOR_ * (1.0f - inColor);
    #elif defined(FOG_HEIGHT)
//  #elif defined(FOG_HEIGHT)
        return ApplyFogHeight(accumulatedColor, firstRayHitResult.minT, initialRay, _FOG_HEIGHT_COLOR_, _FOG_HEIGHT_DENSITY_, _FOG_HEIGHT_FALLOFF_);
//      return ApplyFogHeight(accumulatedColor, firstRayHitResult.minT, initialRay, _FOG_HEIGHT_COLOR_, _FOG_HEIGHT_DENSITY_, _FOG_HEIGHT_FALLOFF_);
    #else
//  #else
        if (firstRayHitResult.hitted)
//      if (firstRayHitResult.hitted)
        {
            normal = firstRayHitResult.normal;
//          normal = firstRayHitResult.normal;
            const MaterialScatteredResult& scatterResult = Scatter(initialRay, firstRayHitResult);
//          const MaterialScatteredResult& scatterResult = Scatter(initialRay, firstRayHitResult);
            if (!scatterResult.isScattered)
//          if (!scatterResult.isScattered)
            {
                albedo = scatterResult.emission;
//              albedo = scatterResult.emission;
            }
            else
//          else
            {
                albedo = scatterResult.attenuation;
//              albedo = scatterResult.attenuation;
            }
        }
        else
//      else
        {
            normal = { 0.0f, 1.0f, 0.0f };
//          normal = { 0.0f, 1.0f, 0.0f };
            Color3  backgroundColor;
//          Color3  backgroundColor;
            switch (backgroundType)
//          switch (backgroundType)
            {
                case BackgroundType::BLUE_LERP_WHITE:
//              case BackgroundType::BLUE_LERP_WHITE:
                {
                    const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
//                  const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
                    const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//                  const float ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
                    backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
//                  backgroundColor = BlendLinear(Color3{ .x = 1.00f, .y = 1.00f, .z = 1.00f }, Color3{ .x = 0.50f, .y = 0.70f, .z = 1.00f }, ratio);
                }
                break;
//              break;
                case BackgroundType::DARK_ROOM_SPACE:
//              case BackgroundType::DARK_ROOM_SPACE:
                {
                    backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
//                  backgroundColor = { .x = 0.00f, .y = 0.00f, .z = 0.00f };
                }
                break;
//              break;
                case BackgroundType::SKY_BOX:
//              case BackgroundType::SKY_BOX:
                {
                    float theta = std::acos (-currentRay.dir.y); // latitude
//                  float theta = std::acos (-currentRay.dir.y); // latitude
                    float phi   = std::atan2(-currentRay.dir.z, currentRay.dir.x) + std::numbers::pi_v<float>; // longitude
//                  float phi   = std::atan2(-currentRay.dir.z, currentRay.dir.x) + std::numbers::pi_v<float>; // longitude
                    float u = phi   / (2.0f * std::numbers::pi_v<float>);
//                  float u = phi   / (2.0f * std::numbers::pi_v<float>);
                    float v = theta /         std::numbers::pi_v<float> ;
//                  float v = theta /         std::numbers::pi_v<float> ;
                    u =        std::clamp(u, 0.0f, 1.0f);
//                  u =        std::clamp(u, 0.0f, 1.0f);
                    v = 1.0f - std::clamp(v, 0.0f, 1.0f);
//                  v = 1.0f - std::clamp(v, 0.0f, 1.0f);
                    float imagePixelX = u * (imagesDatabase.imageBasedLighting.w - 1);
                    float imagePixelY = v * (imagesDatabase.imageBasedLighting.h - 1);
                    backgroundColor = SampleRGB2LinearInterpolation(imagesDatabase.imageBasedLighting.rgbs, imagesDatabase.imageBasedLighting.w, imagesDatabase.imageBasedLighting.h, imagePixelX, imagePixelY);
//                  backgroundColor = SampleRGB2LinearInterpolation(imagesDatabase.imageBasedLighting.rgbs, imagesDatabase.imageBasedLighting.w, imagesDatabase.imageBasedLighting.h, imagePixelX, imagePixelY);
                }
                break;
//              break;
                default:
//              default:
                {
                    backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
//                  backgroundColor = { .x = 1.00f, .y = 1.00f, .z = 1.00f };
                }
                break;
//              break;
            }
            albedo = backgroundColor;
//          albedo = backgroundColor;
        }
        normal = normal * 0.5f + Vec3{ 0.5f, 0.5f, 0.5f };
//      normal = normal * 0.5f + Vec3{ 0.5f, 0.5f, 0.5f };
        return accumulatedColor;
//      return accumulatedColor;
    #endif
//  #endif
}


    static inline void Scale(Geometry& geo, const Vec3& scaleFactor)
//  static inline void Scale(Geometry& geo, const Vec3& scaleFactor)
    {
        switch (geo.geometryType)
//      switch (geo.geometryType)
        {
            case GeometryType::SPHERE:
//          case GeometryType::SPHERE:
            {
                // scaleFactor.x == scaleFactor.y == scaleFactor.z
                // scaleFactor.x == scaleFactor.y == scaleFactor.z
                geo.sphere.radius *= scaleFactor.x;
//              geo.sphere.radius *= scaleFactor.x;
            }
            break;
//          break;


            case GeometryType::PRIMITIVE:
//          case GeometryType::PRIMITIVE:
            {
                // scaleFactor.x == scaleFactor.y == scaleFactor.z
                // scaleFactor.x != scaleFactor.y != scaleFactor.z
                geo.primitive.vertex0.x *= scaleFactor.x;
                geo.primitive.vertex0.y *= scaleFactor.y;
                geo.primitive.vertex0.z *= scaleFactor.z;
                geo.primitive.vertex1.x *= scaleFactor.x;
                geo.primitive.vertex1.y *= scaleFactor.y;
                geo.primitive.vertex1.z *= scaleFactor.z;
                geo.primitive.vertex2.x *= scaleFactor.x;
                geo.primitive.vertex2.y *= scaleFactor.y;
                geo.primitive.vertex2.z *= scaleFactor.z;
            }
            break;
//          break;


            default:
//          default:
            {
            }
            break;
//          break;
        }
    }
    static inline void Move(Geometry& geo, const Vec3& translation)
//  static inline void Move(Geometry& geo, const Vec3& translation)
    {
        switch (geo.geometryType)
//      switch (geo.geometryType)
        {
            case GeometryType::SPHERE:
//          case GeometryType::SPHERE:
            {
                geo.sphere.center += translation;
//              geo.sphere.center += translation;
            }
            break;
//          break;


            case GeometryType::PRIMITIVE:
//          case GeometryType::PRIMITIVE:
            {
                geo.primitive.vertex0 += translation;
                geo.primitive.vertex1 += translation;
                geo.primitive.vertex2 += translation;
            }
            break;
//          break;


            default:
//          default:
            {
            }
            break;
//          break;
        }
    }
    static inline void RotateAroundPivotAndAxis(Vec3& point, const Vec3& pivot, const Vec3& axis, float angleRadians)
//  static inline void RotateAroundPivotAndAxis(Vec3& point, const Vec3& pivot, const Vec3& axis, float angleRadians)
    {
        Vec3 p = point - pivot; // to inner space
//      Vec3 p = point - pivot; // to inner space
        Vec3 k = Normalize(axis);
//      Vec3 k = Normalize(axis);

        float cosTheta = std::cos(angleRadians);
//      float cosTheta = std::cos(angleRadians);
        float sinTheta = std::sin(angleRadians);
//      float sinTheta = std::sin(angleRadians);

        // Rodrigues' rotation formula
//      // Rodrigues' rotation formula
        Vec3 rotated = p * cosTheta + Cross(k, p) * sinTheta + k * Dot(k, p) * (1.0f - cosTheta);
//      Vec3 rotated = p * cosTheta + Cross(k, p) * sinTheta + k * Dot(k, p) * (1.0f - cosTheta);

        point = rotated + pivot; // back to outer space
//      point = rotated + pivot; // back to outer space
    }
    static inline void RotateAroundPivotAndAxis(Geometry& geo, const Vec3& pivot, const Vec3& axis, float angleRadians)
//  static inline void RotateAroundPivotAndAxis(Geometry& geo, const Vec3& pivot, const Vec3& axis, float angleRadians)
    {
        switch (geo.geometryType)
//      switch (geo.geometryType)
        {
            case GeometryType::SPHERE:
//          case GeometryType::SPHERE:
            {
                RotateAroundPivotAndAxis(geo.sphere.center, pivot, axis, angleRadians);
//              RotateAroundPivotAndAxis(geo.sphere.center, pivot, axis, angleRadians);
            }
            break;
//          break;


            case GeometryType::PRIMITIVE:
//          case GeometryType::PRIMITIVE:
            {
                Vec3 p0 = geo.primitive.vertex0 - pivot;
//              Vec3 p0 = geo.primitive.vertex0 - pivot;
                Vec3 p1 = geo.primitive.vertex1 - pivot;
//              Vec3 p1 = geo.primitive.vertex1 - pivot;
                Vec3 p2 = geo.primitive.vertex2 - pivot;
//              Vec3 p2 = geo.primitive.vertex2 - pivot;

                Vec3 k = Normalize(axis);
//              Vec3 k = Normalize(axis);

                float cosTheta = std::cos(angleRadians);
//              float cosTheta = std::cos(angleRadians);
                float sinTheta = std::sin(angleRadians);
//              float sinTheta = std::sin(angleRadians);
                float oneMinusCosTheta = 1.0f - cosTheta;
//              float oneMinusCosTheta = 1.0f - cosTheta;

                // Rodrigues' rotation formula
//              // Rodrigues' rotation formula
                Vec3 rotated0 = p0 * cosTheta + Cross(k, p0) * sinTheta + k * Dot(k, p0) * oneMinusCosTheta;
//              Vec3 rotated0 = p0 * cosTheta + Cross(k, p0) * sinTheta + k * Dot(k, p0) * oneMinusCosTheta;
                Vec3 rotated1 = p1 * cosTheta + Cross(k, p1) * sinTheta + k * Dot(k, p1) * oneMinusCosTheta;
//              Vec3 rotated1 = p1 * cosTheta + Cross(k, p1) * sinTheta + k * Dot(k, p1) * oneMinusCosTheta;
                Vec3 rotated2 = p2 * cosTheta + Cross(k, p2) * sinTheta + k * Dot(k, p2) * oneMinusCosTheta;
//              Vec3 rotated2 = p2 * cosTheta + Cross(k, p2) * sinTheta + k * Dot(k, p2) * oneMinusCosTheta;

                geo.primitive.vertex0 = rotated0 + pivot;
//              geo.primitive.vertex0 = rotated0 + pivot;
                geo.primitive.vertex1 = rotated1 + pivot;
//              geo.primitive.vertex1 = rotated1 + pivot;
                geo.primitive.vertex2 = rotated2 + pivot;
//              geo.primitive.vertex2 = rotated2 + pivot;

                geo.primitive.vertex0FrontFaceNormal = Normalize(geo.primitive.vertex0FrontFaceNormal * cosTheta + Cross(k, geo.primitive.vertex0FrontFaceNormal) * sinTheta + k * Dot(k, geo.primitive.vertex0FrontFaceNormal) * oneMinusCosTheta);
                geo.primitive.vertex1FrontFaceNormal = Normalize(geo.primitive.vertex1FrontFaceNormal * cosTheta + Cross(k, geo.primitive.vertex1FrontFaceNormal) * sinTheta + k * Dot(k, geo.primitive.vertex1FrontFaceNormal) * oneMinusCosTheta);
                geo.primitive.vertex2FrontFaceNormal = Normalize(geo.primitive.vertex2FrontFaceNormal * cosTheta + Cross(k, geo.primitive.vertex2FrontFaceNormal) * sinTheta + k * Dot(k, geo.primitive.vertex2FrontFaceNormal) * oneMinusCosTheta);
            }
            break;
//          break;


            default:
//          default:
            {
            }
            break;
//          break;
        }
    }



    static inline Color2 Saturate(const Color2& v)
//  static inline Color2 Saturate(const Color2& v)
    {
        return
        {
            .x = std::clamp(v.x, 0.0f, 1.0f),
            .y = std::clamp(v.y, 0.0f, 1.0f),
        };

    }
    static inline Color3 Saturate(const Color3& v)
//  static inline Color3 Saturate(const Color3& v)
    {
        return
        {
            .x = std::clamp(v.x, 0.0f, 1.0f),
            .y = std::clamp(v.y, 0.0f, 1.0f),
            .z = std::clamp(v.z, 0.0f, 1.0f),
        };
    }
    static inline Color3 TonemapACES(const Color3& v)
//  static inline Color3 TonemapACES(const Color3& v)
    {
        return Saturate((v * (2.51f * v + Color3{ .x = 0.03f, .y = 0.03f, .z = 0.03f })) / (v * (2.43f * v + Color3{ .x = 0.59f, .y = 0.59f, .z = 0.59f }) + Color3{ .x = 0.14f, .y = 0.14f, .z = 0.14f }));
//      return Saturate((v * (2.51f * v + Color3{ .x = 0.03f, .y = 0.03f, .z = 0.03f })) / (v * (2.43f * v + Color3{ .x = 0.59f, .y = 0.59f, .z = 0.59f }) + Color3{ .x = 0.14f, .y = 0.14f, .z = 0.14f }));
    }
    static inline Color3 TonemapFilmic(const Color3& v)
//  static inline Color3 TonemapFilmic(const Color3& v)
    {
        Color3 m{ .x = std::fmaxf(0.0f, v.x - 0.004f), .y = std::fmaxf(0.0f, v.y - 0.004f), .z = std::fmaxf(0.0f, v.z - 0.004f) };
//      Color3 m{ .x = std::fmaxf(0.0f, v.x - 0.004f), .y = std::fmaxf(0.0f, v.y - 0.004f), .z = std::fmaxf(0.0f, v.z - 0.004f) };
        return (m * (6.2f * m + Color3{ .x = 0.5f, .y = 0.5f, .z = 0.5f })) / (m * (6.2f * m + Color3{ .x = 1.7f, .y = 1.7f, .z = 1.7f }) + Color3{ .x = 0.06f, .y = 0.06f, .z = 0.06f });
//      return (m * (6.2f * m + Color3{ .x = 0.5f, .y = 0.5f, .z = 0.5f })) / (m * (6.2f * m + Color3{ .x = 1.7f, .y = 1.7f, .z = 1.7f }) + Color3{ .x = 0.06f, .y = 0.06f, .z = 0.06f });
    }
    static inline Color3 TonemapReinhard(const Color3& v)
//  static inline Color3 TonemapReinhard(const Color3& v)
    {
        return v / (1.0f + Dot(v, Color3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }));
//      return v / (1.0f + Dot(v, Color3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }));
    }
    static inline Color3 TonemapReinhardJodie(const Color3& v)
//  static inline Color3 TonemapReinhardJodie(const Color3& v)
    {
        float l = Dot(v, Color3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }); Color3 tc = v / (v + Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }); return BlendLinear(v / (l + 1.0f), tc, tc);
//      float l = Dot(v, Color3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }); Color3 tc = v / (v + Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }); return BlendLinear(v / (l + 1.0f), tc, tc);
    }
    static inline Color3 TonemapUncharted2(const Color3& v)
//  static inline Color3 TonemapUncharted2(const Color3& v)
    {
        constexpr float A = 0.15f;
//      constexpr float A = 0.15f;
        constexpr float B = 0.50f;
//      constexpr float B = 0.50f;
        constexpr float C = 0.10f;
//      constexpr float C = 0.10f;
        constexpr float D = 0.20f;
//      constexpr float D = 0.20f;
        constexpr float E = 0.02f;
//      constexpr float E = 0.02f;
        constexpr float F = 0.30f;
//      constexpr float F = 0.30f;
        constexpr Color3 VCB{ .x = C * B, .y = C * B, .z = C * B };
//      constexpr Color3 VCB{ .x = C * B, .y = C * B, .z = C * B };
        constexpr Color3 VB { .x =     B, .y =     B, .z =     B };
//      constexpr Color3 VB { .x =     B, .y =     B, .z =     B };
        constexpr Color3 VDE{ .x = D * E, .y = D * E, .z = D * E };
//      constexpr Color3 VDE{ .x = D * E, .y = D * E, .z = D * E };
        constexpr Color3 VDF{ .x = D * F, .y = D * F, .z = D * F };
//      constexpr Color3 VDF{ .x = D * F, .y = D * F, .z = D * F };
        constexpr Color3 VEF{ .x = E / F, .y = E / F, .z = E / F };
//      constexpr Color3 VEF{ .x = E / F, .y = E / F, .z = E / F };
        return ((v * (A * v + VCB) + VDE) / (v * (A * v + VB) + VDF)) - VEF;
//      return ((v * (A * v + VCB) + VDE) / (v * (A * v + VB) + VDF)) - VEF;
    }
    static inline Color3 TonemapUncharted1(const Color3& v)
//  static inline Color3 TonemapUncharted1(const Color3& v)
    {
        constexpr float W = 11.2f;
//      constexpr float W = 11.2f;
        constexpr float exposureBias = 2.0f;
//      constexpr float exposureBias = 2.0f;
        Color3 curr = TonemapUncharted2(exposureBias * v);
//      Color3 curr = TonemapUncharted2(exposureBias * v);
        Color3 whiteScale = 1.0f / TonemapUncharted2(Color3{ .x = W, .y = W, .z = W });
//      Color3 whiteScale = 1.0f / TonemapUncharted2(Color3{ .x = W, .y = W, .z = W });
        return curr * whiteScale;
//      return curr * whiteScale;
    }
    static inline Color3 TonemapUnreal(const Color3& v)
//  static inline Color3 TonemapUnreal(const Color3& v)
    {
        return v / (v + Color3{ .x = 0.155f, .y = 0.155f, .z = 0.155f }) * 1.019f;
//      return v / (v + Color3{ .x = 0.155f, .y = 0.155f, .z = 0.155f }) * 1.019f;
    }
    static inline Color3 TonemapLinear(const Color3& v)
//  static inline Color3 TonemapLinear(const Color3& v)
    {
        return v;
//      return v;
    }
    static inline Color3 TonemapDrago(const Color3& v)
//  static inline Color3 TonemapDrago(const Color3& v)
    {
        float lum = Dot(v, Color3{ 0.2126f, 0.7152f, 0.0722f });
//      float lum = Dot(v, Color3{ 0.2126f, 0.7152f, 0.0722f });
        float logLum = std::log2f(1.0f + lum);
//      float logLum = std::log2f(1.0f + lum);
        float bias = std::log2f(0.85f);
//      float bias = std::log2f(0.85f);
        float exponent = logLum / bias;
//      float exponent = logLum / bias;
        float scale = std::powf(0.96f, exponent);
//      float scale = std::powf(0.96f, exponent);
        return v * scale;
//      return v * scale;
    }
    static inline Color3 TonemapExponential(const Color3& v)
//  static inline Color3 TonemapExponential(const Color3& v)
    {
        float exposure = 1.0f;
//      float exposure = 1.0f;
        return 1.0f - Exp(-v * exposure);
//      return 1.0f - Exp(-v * exposure);
    }
    static inline Color3 TonemapLogarithmic(const Color3& v)
//  static inline Color3 TonemapLogarithmic(const Color3& v)
    {
        return v / (v + 1.0f);
//      return v / (v + 1.0f);
    }
    static inline Color3 TonemapExtendedReinhard(const Color3& v)
//  static inline Color3 TonemapExtendedReinhard(const Color3& v)
    {
        float whitePoint = 4.0f; // This can be adjusted, or calculated from the scene's max brightness.
//      float whitePoint = 4.0f; // This can be adjusted, or calculated from the scene's max brightness.
        Color3 numerator = v * (1.0f + (v / (whitePoint * whitePoint)));
//      Color3 numerator = v * (1.0f + (v / (whitePoint * whitePoint)));
        return numerator / (1.0f + v);
//      return numerator / (1.0f + v);
    }
    static inline Color3 TonemapHableFilmic(const Color3& v)
//  static inline Color3 TonemapHableFilmic(const Color3& v)
    {
        float A = 0.22f; // Shoulder Strength
//      float A = 0.22f; // Shoulder Strength
        float B = 0.30f; // Linear   Strength
//      float B = 0.30f; // Linear   Strength
        float C = 0.10f; // Linear   Angle
//      float C = 0.10f; // Linear   Angle
        float D = 0.20f; // Toe      Strength
//      float D = 0.20f; // Toe      Strength
        float E = 0.01f; // Toe        Numerator
//      float E = 0.01f; // Toe        Numerator
        float F = 0.30f; // Toe      Denominator
//      float F = 0.30f; // Toe      Denominator
        float exposure = 2.0f;
//      float exposure = 2.0f;

        Color3 curr = ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
//      Color3 curr = ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
        Color3 whitePoint = ((Color3{ 11.2f, 11.2f, 11.2f } * (A * Color3{ 11.2f, 11.2f, 11.2f } + C * B) + D * E) / (Color3{ 11.2f, 11.2f, 11.2f } * (A * Color3{ 11.2f, 11.2f, 11.2f } + B) + D * F)) - E / F;
//      Color3 whitePoint = ((Color3{ 11.2f, 11.2f, 11.2f } * (A * Color3{ 11.2f, 11.2f, 11.2f } + C * B) + D * E) / (Color3{ 11.2f, 11.2f, 11.2f } * (A * Color3{ 11.2f, 11.2f, 11.2f } + B) + D * F)) - E / F;

        return curr * (1.0f / whitePoint);
//      return curr * (1.0f / whitePoint);
    }
    static inline Color3 TonemapACESFitted(const Color3& v)
//  static inline Color3 TonemapACESFitted(const Color3& v)
    {
        // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
        // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
        const float a = 2.51f;
//      const float a = 2.51f;
        const float b = 0.03f;
//      const float b = 0.03f;
        const float c = 2.43f;
//      const float c = 2.43f;
        const float d = 0.59f;
//      const float d = 0.59f;
        const float e = 0.14f;
//      const float e = 0.14f;

        Color3 clamped = Saturate((v * (a * v + b)) / (v * (c * v + d) + e));
//      Color3 clamped = Saturate((v * (a * v + b)) / (v * (c * v + d) + e));
        return clamped;
//      return clamped;
    }
    static inline Color3 TonemapLottes(const Color3& v)
//  static inline Color3 TonemapLottes(const Color3& v)
    {
        // A simpler, more robust curve inspired by Lottes' GDC talks
        // A simpler, more robust curve inspired by Lottes' GDC talks
        const float contrast = 1.60f;
//      const float contrast = 1.60f;
        const float shoulder = 0.98f;
//      const float shoulder = 0.98f;

        // Apply contrast
        // Apply contrast
        Color3 contrasted
//      Color3 contrasted
        {
            .x = std::powf(v.x, contrast),
            .y = std::powf(v.y, contrast),
            .z = std::powf(v.z, contrast),
        };

        // Apply shoulder curve to compress highlights
        // Apply shoulder curve to compress highlights
        Color3 tonemapped
//      Color3 tonemapped
        {
            .x = (1.0f - shoulder) * contrasted.x + shoulder * contrasted.x / (1.0f + contrasted.x),
            .y = (1.0f - shoulder) * contrasted.y + shoulder * contrasted.y / (1.0f + contrasted.y),
            .z = (1.0f - shoulder) * contrasted.z + shoulder * contrasted.z / (1.0f + contrasted.z),
        };

        return tonemapped;
//      return tonemapped;
    }
    static inline Color3 TonemapHejlFilmic(const Color3& v)
//  static inline Color3 TonemapHejlFilmic(const Color3& v)
    {
        Color3 temp
//      Color3 temp
        {
            .x = std::fmaxf(0.0f, v.x - 0.004f),
            .y = std::fmaxf(0.0f, v.y - 0.004f),
            .z = std::fmaxf(0.0f, v.z - 0.004f),
        };

        Color3   numerator = (temp * (6.2f * temp + 0.5f));
//      Color3   numerator = (temp * (6.2f * temp + 0.5f));
        Color3 denominator = (temp * (6.2f * temp + 1.7f)) + 0.06f;
//      Color3 denominator = (temp * (6.2f * temp + 1.7f)) + 0.06f;

        return Color3
//      return Color3
        {
            .x = numerator.x / denominator.x,
            .y = numerator.y / denominator.y,
            .z = numerator.z / denominator.z,
        };
    }
    using TonemapFunctionPointer = Color3(*)(const Color3& color3);
//  using TonemapFunctionPointer = Color3(*)(const Color3& color3);
    static inline struct TonemapsDatabase { TonemapFunctionPointer tonemaps[16] { TonemapACES, TonemapFilmic, TonemapReinhard, TonemapReinhardJodie, TonemapUncharted2, TonemapUncharted1, TonemapUnreal, TonemapLinear, TonemapDrago, TonemapExponential, TonemapLogarithmic, TonemapExtendedReinhard, TonemapHableFilmic, TonemapACESFitted, TonemapLottes, TonemapHejlFilmic, }; std::string tonemapNames[16] { "TonemapACES", "TonemapFilmic", "TonemapReinhard", "TonemapReinhardJodie", "TonemapUncharted2", "TonemapUncharted1", "TonemapUnreal", "TonemapLinear", "TonemapDrago", "TonemapExponential", "TonemapLogarithmic", "TonemapExtendedReinhard", "TonemapHableFilmic", "TonemapACESFitted", "TonemapLottes", "TonemapHejlFilmic", }; } tonemapsDatabase;
//  static inline struct TonemapsDatabase { TonemapFunctionPointer tonemaps[16] { TonemapACES, TonemapFilmic, TonemapReinhard, TonemapReinhardJodie, TonemapUncharted2, TonemapUncharted1, TonemapUnreal, TonemapLinear, TonemapDrago, TonemapExponential, TonemapLogarithmic, TonemapExtendedReinhard, TonemapHableFilmic, TonemapACESFitted, TonemapLottes, TonemapHejlFilmic, }; std::string tonemapNames[16] { "TonemapACES", "TonemapFilmic", "TonemapReinhard", "TonemapReinhardJodie", "TonemapUncharted2", "TonemapUncharted1", "TonemapUnreal", "TonemapLinear", "TonemapDrago", "TonemapExponential", "TonemapLogarithmic", "TonemapExtendedReinhard", "TonemapHableFilmic", "TonemapACESFitted", "TonemapLottes", "TonemapHejlFilmic", }; } tonemapsDatabase;
    template <typename T, size_t N> static inline const constexpr std::size_t GetArraySize(const T(&array)[N]) { return N; }
//  template <typename T, size_t N> static inline const constexpr std::size_t GetArraySize(const T(&array)[N]) { return N; }
    thread_local static inline const constexpr std::size_t tonemapsSize = GetArraySize(tonemapsDatabase.tonemaps);
//  thread_local static inline const constexpr std::size_t tonemapsSize = GetArraySize(tonemapsDatabase.tonemaps);



int main()
{
    noise = new FastNoiseLite();
//  noise = new FastNoiseLite();
    noise->SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
//  noise->SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise->SetFractalType(FastNoiseLite::FractalType_FBm);
//  noise->SetFractalType(FastNoiseLite::FractalType_FBm);
    noise->SetFractalOctaves(6); // 6 layers of detail
//  noise->SetFractalOctaves(6); // 6 layers of detail
    noise->SetFractalLacunarity(2.0); // Each octave is 2.0x higher frequency
//  noise->SetFractalLacunarity(2.0); // Each octave is 2.0x higher frequency
    noise->SetFractalGain(0.5); // Each octave is 0.5x the amplitude
//  noise->SetFractalGain(0.5); // Each octave is 0.5x the amplitude



/*
    noisesDatabase.noisePerlins.reserve(4);
//  noisesDatabase.noisePerlins.reserve(4);
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::BLOCKY          , .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
//  noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::BLOCKY          , .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
//  noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_1     });
//  noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_1     });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_2     });
//  noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_2     });
    for (NoisePerlin& np : noisesDatabase.noisePerlins) Generate(np);
//  for (NoisePerlin& np : noisesDatabase.noisePerlins) Generate(np);
*/



#ifdef SCENE_001
    thread_local static const constexpr BackgroundType backgroundType = BackgroundType::SKY_BOX;
//  thread_local static const constexpr BackgroundType backgroundType = BackgroundType::SKY_BOX;
    imagesDatabase.imageBasedLighting.Load(R"(./assets/scene001/citrus_orchard_road_puresky_16k.exr)");
//  imagesDatabase.imageBasedLighting.Load(R"(./assets/scene001/citrus_orchard_road_puresky_16k.exr)");
    imagesDatabase.images.reserve(1);
//  imagesDatabase.images.reserve(1);
    imagesDatabase.images.emplace_back(R"(./assets/scene001/Nackter_Reiter_C_Nackter_Reiter_O.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene001/Nackter_Reiter_C_Nackter_Reiter_O.png)");
#endif
#ifdef SCENE_002
    thread_local static const constexpr BackgroundType backgroundType = BackgroundType::DARK_ROOM_SPACE;
//  thread_local static const constexpr BackgroundType backgroundType = BackgroundType::DARK_ROOM_SPACE;
    imagesDatabase.images.reserve(1);
//  imagesDatabase.images.reserve(1);
    imagesDatabase.images.emplace_back(R"(./assets/scene002/joseph-kainz/source/Joseph_Kainz_C/Joseph_Kainz_C_Joseph_Kainz_O.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene002/joseph-kainz/source/Joseph_Kainz_C/Joseph_Kainz_C_Joseph_Kainz_O.png)");
#endif
#ifdef SCENE_003
    thread_local static const constexpr BackgroundType backgroundType = BackgroundType::DARK_ROOM_SPACE;
//  thread_local static const constexpr BackgroundType backgroundType = BackgroundType::DARK_ROOM_SPACE;
    imagesDatabase.images.reserve(4);
//  imagesDatabase.images.reserve(4);
    imagesDatabase.images.emplace_back(R"(./assets/scene003/Chiours.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene003/Chiours.png)");
    imagesDatabase.images.emplace_back(R"(./assets/scene003/Owl.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene003/Owl.png)");
    imagesDatabase.images.emplace_back(R"(./assets/scene003/Bob.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene003/Bob.png)");
    imagesDatabase.images.emplace_back(R"(./assets/scene003/Cat.png)");
//  imagesDatabase.images.emplace_back(R"(./assets/scene003/Cat.png)");
#endif



#ifdef SCENE_001
    texturesDatabase.textures.reserve(3);
//  texturesDatabase.textures.reserve(3);
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });

#endif
#ifdef SCENE_002
    texturesDatabase.textures.reserve(6);
//  texturesDatabase.textures.reserve(6);
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 9.9f, 9.9f, 9.9f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 9.9f, 9.9f, 9.9f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.3f, 0.3f, 0.3f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
#endif
#ifdef SCENE_003
    texturesDatabase.textures.reserve(9);
//  texturesDatabase.textures.reserve(9);
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +2, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +2, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +3, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.000f, 000.000f, 000.000f }, .scale = 1.0f, .imageIndex = +3, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 010.000f, 010.000f, 010.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 010.000f, 010.000f, 010.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 001.000f, 001.000f, 001.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 001.000f, 001.000f, 001.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.965f, 000.976f, 000.984f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.965f, 000.976f, 000.984f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.231f, 000.525f, 000.620f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 000.231f, 000.525f, 000.620f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 001.000f, 000.380f, 000.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 001.000f, 000.380f, 000.000f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
#endif



#ifdef SCENE_001
    materialsDatabase.materials.reserve(3);
//  materialsDatabase.materials.reserve(3);
//  materialsDatabase.materials.emplace_back(Material{ .absorption = { 0.1f, 0.4f, 0.6f }, .scattering = { 1.0f, 1.0f, 1.0f }, .asymmetryParameter = 0.85f, .layer1Roughness = 0.5f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::SKIN), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::SubsurfaceRandomWalk2, });
//  materialsDatabase.materials.emplace_back(Material{ .absorption = { 0.1f, 0.4f, 0.6f }, .scattering = { 1.0f, 1.0f, 1.0f }, .asymmetryParameter = 0.85f, .layer1Roughness = 0.5f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::SKIN), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::SubsurfaceRandomWalk2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::FresnelBlendedDielectricGlossyDiffuse2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::Metal, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.1f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::Metal, });

#endif
#ifdef SCENE_002
    materialsDatabase.materials.reserve(6);
//  materialsDatabase.materials.reserve(6);
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::LightDiffuse, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::LightDiffuse, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::Metal, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::Metal, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.0f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::LambertianDiffuseReflectance2, });
#endif
#ifdef SCENE_003
    materialsDatabase.materials.reserve(11);
//  materialsDatabase.materials.reserve(11);
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 0, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 1, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 3, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::LightDiffuse                          , });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 4, .materialType = MaterialType::LightDiffuse                          , });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::GLASS  ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::Dielectric                            , });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::GLASS  ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::Dielectric                            , });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 8, .materialType = MaterialType::LambertianDiffuseReflectance2, });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 0.05f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::AIR    ), .layer1IOR = GetRefractionIndex(MaterialDielectric::MARBLE ), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 8, .materialType = MaterialType::LambertianDiffuseReflectance2, });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::Isotropic2                            , });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::Isotropic2                            , });
    materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::LambertianDiffuseReflectance2         , });
//  materialsDatabase.materials.emplace_back(Material{ .layer1Roughness = 1.00f, .layer1Thickness = 1.0f, .layer0IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer1IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .layer2IOR = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 5, .materialType = MaterialType::LambertianDiffuseReflectance2         , });
#endif



    ThreadPool* threadPool = new ThreadPool();
//  ThreadPool* threadPool = new ThreadPool();
    threadPool->WarmUp(4);
//  threadPool->WarmUp(4);

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    int                              samplesPerPixel = 100 ;
//  int                              samplesPerPixel = 100 ;
    float pixelSamplesScale = 1.0f / samplesPerPixel        ; // 1.0f / (stratifiedSamplesPerPixel * stratifiedSamplesPerPixel) ;
//  float pixelSamplesScale = 1.0f / samplesPerPixel        ; // 1.0f / (stratifiedSamplesPerPixel * stratifiedSamplesPerPixel) ;
    int stratifiedSamplesPerPixel = static_cast<int>(std::sqrtf(static_cast<float>(samplesPerPixel)));
//  int stratifiedSamplesPerPixel = static_cast<int>(std::sqrtf(static_cast<float>(samplesPerPixel)));
    float inverseStratifiedSamplesPerPixel = 1.0f / stratifiedSamplesPerPixel;
//  float inverseStratifiedSamplesPerPixel = 1.0f / stratifiedSamplesPerPixel;



    BVHTreeMain bvhTreeMain;
//  BVHTreeMain bvhTreeMain;
#ifdef SCENE_001
    bvhTreeMain.bvhTrees.reserve(2);
//  bvhTreeMain.bvhTrees.reserve(2);
    for (std::uint8_t i = 0; i < 2; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
//  for (std::uint8_t i = 0; i < 2; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
    bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(2 * 2 - 1));
//  bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(2 * 2 - 1));
    bvhTreeMain.bvhTrees[0].geometries.reserve(2);
//  bvhTreeMain.bvhTrees[0].geometries.reserve(2);
    bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(2 * 2 - 1));
//  bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(2 * 2 - 1));

    //FLOOR
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });



/*
    Geometry metaballs;
//  Geometry metaballs;
    metaballs.geometryType = GeometryType::IMPLICIT_SURFACE;
//  metaballs.geometryType = GeometryType::IMPLICIT_SURFACE;
    metaballs.materialIndex = 1;
//  metaballs.materialIndex = 1;
    metaballs.movingDirection = { 0.0f, 0.0f, 0.0f };
//  metaballs.movingDirection = { 0.0f, 0.0f, 0.0f };

    metaballs.implicitSurface.sdfIndex = 0;
//  metaballs.implicitSurface.sdfIndex = 0;
    metaballs.implicitSurface.maxSteps = 1000;
//  metaballs.implicitSurface.maxSteps = 1000;
    metaballs.implicitSurface.maxMarchingDistance = 1000.0f;
//  metaballs.implicitSurface.maxMarchingDistance = 1000.0f;
    metaballs.implicitSurface.hitEpsilon = 0.00001f; // 1e-5f
//  metaballs.implicitSurface.hitEpsilon = 0.00001f; // 1e-5f

    metaballs.aabb3d.intervalAxisX = { -100.0f, +100.0f };
//  metaballs.aabb3d.intervalAxisX = { -100.0f, +100.0f };
    metaballs.aabb3d.intervalAxisY = { -100.0f, +100.0f };
//  metaballs.aabb3d.intervalAxisY = { -100.0f, +100.0f };
    metaballs.aabb3d.intervalAxisZ = { -100.0f, +100.0f };
//  metaballs.aabb3d.intervalAxisZ = { -100.0f, +100.0f };

    bvhTreeMain.bvhTrees[0].geometries.push_back(metaballs);
//  bvhTreeMain.bvhTrees[0].geometries.push_back(metaballs);
*/



    //MODEL
    std::unique_ptr<Assimp::Importer> importerA = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importerA = std::make_unique<Assimp::Importer>();
    try
    {
        float minModelX = std::numeric_limits<float>::max();
        float maxModelX = std::numeric_limits<float>::lowest();
        float minModelY = std::numeric_limits<float>::max();
        float maxModelY = std::numeric_limits<float>::lowest();
        float minModelZ = std::numeric_limits<float>::max();
        float maxModelZ = std::numeric_limits<float>::lowest();

        aiScene const * const scene = importerA->ReadFile(R"(./assets/scene001/Nackter_Reiter_C.obj)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importerA->ReadFile(R"(./assets/scene001/Nackter_Reiter_C.obj)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    mesh->mVertices[vertexIndex] *= 1.5f;
//                  mesh->mVertices[vertexIndex] *= 1.5f;
                }
            }



            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[1].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[1].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[1].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[1].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
                                        Move(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +04.00f, .y = +00.00f, .z = +00.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +04.00f, .y = +00.00f, .z = +00.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importerA->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importerA->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }



#endif
#ifdef SCENE_002
    bvhTreeMain.bvhTrees.reserve(2);
//  bvhTreeMain.bvhTrees.reserve(2);
    for (std::uint8_t i = 0; i < 2; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
//  for (std::uint8_t i = 0; i < 2; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
    bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(2 * 2 - 1));
//  bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(2 * 2 - 1));
    bvhTreeMain.bvhTrees[0].geometries.reserve(12);
//  bvhTreeMain.bvhTrees[0].geometries.reserve(12);
    bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(12 * 2 - 1));
//  bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(12 * 2 - 1));


    //B
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 5, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 5, .geometryType = GeometryType::PRIMITIVE, });

    //T
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });

    //L
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });

    //R
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });

    //B
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });

    //F
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 1, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 1, .geometryType = GeometryType::PRIMITIVE, });



    float minModelX = std::numeric_limits<float>::max();
    float maxModelX = std::numeric_limits<float>::lowest();
    float minModelY = std::numeric_limits<float>::max();
    float maxModelY = std::numeric_limits<float>::lowest();
    float minModelZ = std::numeric_limits<float>::max();
    float maxModelZ = std::numeric_limits<float>::lowest();


    std::unique_ptr<Assimp::Importer> importer = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importer = std::make_unique<Assimp::Importer>();
    try
    {
        aiScene const * const scene = importer->ReadFile(R"(./assets/scene002/joseph-kainz/source/Joseph_Kainz_C/Joseph_Kainz_C.obj)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importer->ReadFile(R"(./assets/scene002/joseph-kainz/source/Joseph_Kainz_C/Joseph_Kainz_C.obj)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[1].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[1].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[1].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[1].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f , .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f , .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
//                                      Move(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f , .y = +00.00f, .z = -10.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[1].geometries[currentIndex], { .x = +00.00f , .y = +00.00f, .z = -10.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importer->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importer->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }


#endif
#ifdef SCENE_003
    bvhTreeMain.bvhTrees.reserve(6);
//  bvhTreeMain.bvhTrees.reserve(6);
    for (std::uint8_t i = 0; i < 6; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
//  for (std::uint8_t i = 0; i < 6; ++i) bvhTreeMain.bvhTrees.emplace_back(BVHTree{});
    bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(6 * 2 - 1));
//  bvhTreeMain.bvhNodeMains.reserve(static_cast<std::size_t>(6 * 2 - 1));
    bvhTreeMain.bvhTrees[0].geometries.reserve(44);
//  bvhTreeMain.bvhTrees[0].geometries.reserve(44);
    bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(44 * 2 - 1));
//  bvhTreeMain.bvhTrees[0].bvhNodes.reserve(static_cast<std::size_t>(44 * 2 - 1));
    bvhTreeMain.bvhTrees[1].geometries.reserve(12);
//  bvhTreeMain.bvhTrees[1].geometries.reserve(12);
    bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(12 * 2 - 1));
//  bvhTreeMain.bvhTrees[1].bvhNodes.reserve(static_cast<std::size_t>(12 * 2 - 1));



    //BO - WA
    //BO - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });


    //TO - LI - TO - WA
    //TO - LI - TO - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +30.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +30.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = +30.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +30.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +30.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = +30.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });


    //TO - LI - FR - WA
    //TO - LI - FR - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +30.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +30.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +30.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });


    //TO - LI - BA - WA
    //TO - LI - BA - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +30.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +30.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +30.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });


    //TO - LI - LE - WA
    //TO - LI - LE - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +30.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +30.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = +30.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });


    //TO - LI - RI - WA
    //TO - LI - RI - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +30.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +30.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = +30.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 4, .geometryType = GeometryType::PRIMITIVE, });


    //TO - GL - TO - RO
    //TO - GL - TO - RO
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = +00.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +00.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = +00.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +00.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +00.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +00.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });


    //TO - GL - BO - RO
    //TO - GL - BO - RO
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = +00.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +00.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +00.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +00.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = +00.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = +00.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });


    //TO - GL - LE - CO
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = +00.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = +00.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = +00.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +00.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = +00.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +00.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });


    //TO - GL - RI - CO
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = +00.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = +00.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = +00.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });

    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = +00.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +00.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = +00.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });


    //TO - GL - @CENTER
    //TO - GL - @CENTER
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .vertex1 = { .x = -10.00f, .y = +20.00f, .z = -10.00f }, .vertex2 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 5, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +10.00f, .y = +20.00f, .z = -10.00f }, .vertex1 = { .x = +10.00f, .y = +20.00f, .z = +10.00f }, .vertex2 = { .x = -10.00f, .y = +20.00f, .z = +10.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 5, .geometryType = GeometryType::PRIMITIVE, });


    //LE - WA
    //LE - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = +20.00f }, .vertex2 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 8, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 8, .geometryType = GeometryType::PRIMITIVE, });


    //RI - WA
    //RI - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 7, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = +20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = +20.00f }, .vertex2 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 7, .geometryType = GeometryType::PRIMITIVE, });


    //BA - WA
    //BA - WA
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .vertex1 = { .x = -20.00f, .y = -20.00f, .z = -20.00f }, .vertex2 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[0].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +20.00f, .y = -20.00f, .z = -20.00f }, .vertex1 = { .x = +20.00f, .y = +20.00f, .z = -20.00f }, .vertex2 = { .x = -20.00f, .y = +20.00f, .z = -20.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 6, .geometryType = GeometryType::PRIMITIVE, });



    bvhTreeMain.bvhTrees[1].mediumType = MediumType::PARTICIPATING_WITH_CONSTANT_DENSITY;
//  bvhTreeMain.bvhTrees[1].mediumType = MediumType::PARTICIPATING_WITH_CONSTANT_DENSITY;
    bvhTreeMain.bvhTrees[1].mediumParticipatingWithConstantDensity.negativeInverseDensity = - (1.0f / 0.02f);
//  bvhTreeMain.bvhTrees[1].mediumParticipatingWithConstantDensity.negativeInverseDensity = - (1.0f / 0.02f);
    bvhTreeMain.bvhTrees[1].mediumParticipatingWithConstantDensity.materialIndex = 9;
//  bvhTreeMain.bvhTrees[1].mediumParticipatingWithConstantDensity.materialIndex = 9;



    //FOG - BO
    //FOG - BO
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = -0.00f, .z = -0.00f }, .vertex1 = { .x = -0.00f, .y = -0.00f, .z = +0.00f }, .vertex2 = { .x = +0.00f, .y = -0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = -0.00f, .z = +0.00f }, .vertex1 = { .x = +0.00f, .y = -0.00f, .z = -0.00f }, .vertex2 = { .x = -0.00f, .y = -0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });

    //FOG - TO
    //FOG - TO
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = +0.00f, .z = +0.00f }, .vertex1 = { .x = -0.00f, .y = +0.00f, .z = -0.00f }, .vertex2 = { .x = +0.00f, .y = +0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = +0.00f, .z = -0.00f }, .vertex1 = { .x = +0.00f, .y = +0.00f, .z = +0.00f }, .vertex2 = { .x = -0.00f, .y = +0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });

    //FOG - LE
    //FOG - LE
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = +0.00f, .z = +0.00f }, .vertex1 = { .x = -0.00f, .y = -0.00f, .z = +0.00f }, .vertex2 = { .x = -0.00f, .y = -0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = -0.00f, .z = -0.00f }, .vertex1 = { .x = -0.00f, .y = +0.00f, .z = -0.00f }, .vertex2 = { .x = -0.00f, .y = +0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });

    //FOG - RI
    //FOG - RI
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = +0.00f, .z = -0.00f }, .vertex1 = { .x = +0.00f, .y = -0.00f, .z = -0.00f }, .vertex2 = { .x = +0.00f, .y = -0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = -0.00f, .z = +0.00f }, .vertex1 = { .x = +0.00f, .y = +0.00f, .z = +0.00f }, .vertex2 = { .x = +0.00f, .y = +0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });

    //FOG - BA
    //FOG - BA
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = +0.00f, .z = -0.00f }, .vertex1 = { .x = -0.00f, .y = -0.00f, .z = -0.00f }, .vertex2 = { .x = +0.00f, .y = -0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = -0.00f, .z = -0.00f }, .vertex1 = { .x = +0.00f, .y = +0.00f, .z = -0.00f }, .vertex2 = { .x = -0.00f, .y = +0.00f, .z = -0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });

    //FOG - FR
    //FOG - FR
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.00f, .y = +0.00f, .z = +0.00f }, .vertex1 = { .x = +0.00f, .y = -0.00f, .z = +0.00f }, .vertex2 = { .x = -0.00f, .y = -0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });
    bvhTreeMain.bvhTrees[1].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -0.00f, .y = -0.00f, .z = +0.00f }, .vertex1 = { .x = -0.00f, .y = +0.00f, .z = +0.00f }, .vertex2 = { .x = +0.00f, .y = +0.00f, .z = +0.00f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, .perVertexFrontFaceNormalAvailable = false, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 10, .geometryType = GeometryType::PRIMITIVE, });



    std::unique_ptr<Assimp::Importer> importerA = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importerA = std::make_unique<Assimp::Importer>();
    try
    {
        float minModelX = std::numeric_limits<float>::max();
        float maxModelX = std::numeric_limits<float>::lowest();
        float minModelY = std::numeric_limits<float>::max();
        float maxModelY = std::numeric_limits<float>::lowest();
        float minModelZ = std::numeric_limits<float>::max();
        float maxModelZ = std::numeric_limits<float>::lowest();

        aiScene const * const scene = importerA->ReadFile(R"(./assets/scene003/Chiours.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importerA->ReadFile(R"(./assets/scene003/Chiours.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    mesh->mVertices[vertexIndex] *= 9.0f;
//                  mesh->mVertices[vertexIndex] *= 9.0f;
                }
            }



            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[2].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[2].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[2].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[2].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[2].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[2].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 0, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[2].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[2].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[2].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[2].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
                                        Move(bvhTreeMain.bvhTrees[2].geometries[currentIndex], { .x = +07.50f, .y = +00.00f, .z = -10.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[2].geometries[currentIndex], { .x = +07.50f, .y = +00.00f, .z = -10.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importerA->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importerA->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }



    std::unique_ptr<Assimp::Importer> importerB = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importerB = std::make_unique<Assimp::Importer>();
    try
    {
        float minModelX = std::numeric_limits<float>::max();
        float maxModelX = std::numeric_limits<float>::lowest();
        float minModelY = std::numeric_limits<float>::max();
        float maxModelY = std::numeric_limits<float>::lowest();
        float minModelZ = std::numeric_limits<float>::max();
        float maxModelZ = std::numeric_limits<float>::lowest();

        aiScene const * const scene = importerB->ReadFile(R"(./assets/scene003/Owl.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importerB->ReadFile(R"(./assets/scene003/Owl.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    mesh->mVertices[vertexIndex] *= 9.0f;
//                  mesh->mVertices[vertexIndex] *= 9.0f;
                }
            }



            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[3].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[3].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[3].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[3].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[3].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 1, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[3].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 1, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[3].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[3].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[3].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+180.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[3].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+180.000f));
                                        Move(bvhTreeMain.bvhTrees[3].geometries[currentIndex], { .x = -07.50f, .y = +00.00f, .z = -10.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[3].geometries[currentIndex], { .x = -07.50f, .y = +00.00f, .z = -10.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importerB->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importerB->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }



    std::unique_ptr<Assimp::Importer> importerC = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importerC = std::make_unique<Assimp::Importer>();
    try
    {
        float minModelX = std::numeric_limits<float>::max();
        float maxModelX = std::numeric_limits<float>::lowest();
        float minModelY = std::numeric_limits<float>::max();
        float maxModelY = std::numeric_limits<float>::lowest();
        float minModelZ = std::numeric_limits<float>::max();
        float maxModelZ = std::numeric_limits<float>::lowest();

        aiScene const * const scene = importerC->ReadFile(R"(./assets/scene003/Bob.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importerC->ReadFile(R"(./assets/scene003/Bob.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    mesh->mVertices[vertexIndex] *= 9.0f;
//                  mesh->mVertices[vertexIndex] *= 9.0f;
                }
            }



            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[4].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[4].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[4].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[4].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[4].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[4].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 2, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[4].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[4].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[4].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+060.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[4].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+060.000f));
                                        Move(bvhTreeMain.bvhTrees[4].geometries[currentIndex], { .x = -12.00f, .y = +00.00f, .z = +00.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[4].geometries[currentIndex], { .x = -12.00f, .y = +00.00f, .z = +00.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importerC->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importerC->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }



    std::unique_ptr<Assimp::Importer> importerD = std::make_unique<Assimp::Importer>();
//  std::unique_ptr<Assimp::Importer> importerD = std::make_unique<Assimp::Importer>();
    try
    {
        float minModelX = std::numeric_limits<float>::max();
        float maxModelX = std::numeric_limits<float>::lowest();
        float minModelY = std::numeric_limits<float>::max();
        float maxModelY = std::numeric_limits<float>::lowest();
        float minModelZ = std::numeric_limits<float>::max();
        float maxModelZ = std::numeric_limits<float>::lowest();

        aiScene const * const scene = importerD->ReadFile(R"(./assets/scene003/Cat.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
//      aiScene const * const scene = importerD->ReadFile(R"(./assets/scene003/Cat.dae)", aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_GenSmoothNormals | aiPostProcessSteps::aiProcess_FixInfacingNormals | aiPostProcessSteps::aiProcess_OptimizeGraph | aiPostProcessSteps::aiProcess_OptimizeMeshes);
        if (scene)
//      if (scene)
        {
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    mesh->mVertices[vertexIndex] *= 9.0f;
//                  mesh->mVertices[vertexIndex] *= 9.0f;
                }
            }



            std::size_t geometriesCount = 0;
//          std::size_t geometriesCount = 0;
            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
//              geometriesCount += scene->mMeshes[meshIndex]->mNumFaces;
            }
            bvhTreeMain.bvhTrees[5].geometries.reserve(geometriesCount);
//          bvhTreeMain.bvhTrees[5].geometries.reserve(geometriesCount);
            bvhTreeMain.bvhTrees[5].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));
//          bvhTreeMain.bvhTrees[5].bvhNodes.reserve(static_cast<std::size_t>(geometriesCount * 2 - 1));



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh const * const mesh = scene->mMeshes[meshIndex];
//              aiMesh const * const mesh = scene->mMeshes[meshIndex];
                for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
//              for (unsigned int vertexIndex = 0; vertexIndex < mesh->mNumVertices; ++vertexIndex)
                {
                    const aiVector3D& vertex = mesh->mVertices[vertexIndex];
//                  const aiVector3D& vertex = mesh->mVertices[vertexIndex];
                    minModelX = std::fminf(minModelX, vertex.x);
                    maxModelX = std::fmaxf(maxModelX, vertex.x);
                    minModelY = std::fminf(minModelY, vertex.y);
                    maxModelY = std::fmaxf(maxModelY, vertex.y);
                    minModelZ = std::fminf(minModelZ, vertex.z);
                    maxModelZ = std::fmaxf(maxModelZ, vertex.z);
                }
            }
            constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;
//          constexpr float floorY = -20.00f; float offsetY = minModelY - floorY;



            for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
//          for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
            {
                aiMesh* mesh = scene->mMeshes[meshIndex];
//              aiMesh* mesh = scene->mMeshes[meshIndex];
                for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
//              for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
                {
                    const aiFace& face = mesh->mFaces[faceIndex];
//                  const aiFace& face = mesh->mFaces[faceIndex];
                    const aiVector3D& vertex0 = mesh->mVertices[face.mIndices[0]];
                    const aiVector3D& vertex1 = mesh->mVertices[face.mIndices[1]];
                    const aiVector3D& vertex2 = mesh->mVertices[face.mIndices[2]];
                    const aiVector3D& vertex0UV = mesh->mTextureCoords[0][face.mIndices[0]];
                    const aiVector3D& vertex1UV = mesh->mTextureCoords[0][face.mIndices[1]];
                    const aiVector3D& vertex2UV = mesh->mTextureCoords[0][face.mIndices[2]];
                    const aiVector3D& vertex0FrontFaceNormal = mesh->mNormals[face.mIndices[0]];
                    const aiVector3D& vertex1FrontFaceNormal = mesh->mNormals[face.mIndices[1]];
                    const aiVector3D& vertex2FrontFaceNormal = mesh->mNormals[face.mIndices[2]];
                    bvhTreeMain.bvhTrees[5].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });
//                  bvhTreeMain.bvhTrees[5].geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = vertex0.x, .y = vertex0.y - offsetY, .z = vertex0.z }, .vertex1 = { .x = vertex1.x, .y = vertex1.y - offsetY, .z = vertex1.z }, .vertex2 = { .x = vertex2.x, .y = vertex2.y - offsetY, .z = vertex2.z }, .vertex0FrontFaceNormal = { .x = vertex0FrontFaceNormal.x, .y = vertex0FrontFaceNormal.y, .z = vertex0FrontFaceNormal.z }, .vertex1FrontFaceNormal = { .x = vertex1FrontFaceNormal.x, .y = vertex1FrontFaceNormal.y, .z = vertex1FrontFaceNormal.z }, .vertex2FrontFaceNormal = { .x = vertex2FrontFaceNormal.x, .y = vertex2FrontFaceNormal.y, .z = vertex2FrontFaceNormal.z }, .frontFaceVertex0U = vertex0UV.x, .frontFaceVertex0V = vertex0UV.y, .frontFaceVertex1U = vertex1UV.x, .frontFaceVertex1V = vertex1UV.y, .frontFaceVertex2U = vertex2UV.x, .frontFaceVertex2V = vertex2UV.y, .backFaceVertex0U = vertex0UV.x, .backFaceVertex0V = vertex0UV.y, .backFaceVertex1U = vertex1UV.x, .backFaceVertex1V = vertex1UV.y, .backFaceVertex2U = vertex2UV.x, .backFaceVertex2V = vertex2UV.y, .perVertexFrontFaceNormalAvailable = true, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .materialIndex = 3, .geometryType = GeometryType::PRIMITIVE, });

                    std::size_t currentIndex = bvhTreeMain.bvhTrees[5].geometries.size() - 1;
//                  std::size_t currentIndex = bvhTreeMain.bvhTrees[5].geometries.size() - 1;
                    RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[5].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
//                  RotateAroundPivotAndAxis(bvhTreeMain.bvhTrees[5].geometries[currentIndex], { .x = +00.00f, .y = +00.00f, .z = +00.00f }, { .x = +00.00f, .y = +01.00f, .z = +00.00f }, lazy::DegToRad(+000.000f));
                                        Move(bvhTreeMain.bvhTrees[5].geometries[currentIndex], { .x = +12.00f, .y = +00.00f, .z = +00.00f }                                                                         );
//                                      Move(bvhTreeMain.bvhTrees[5].geometries[currentIndex], { .x = +12.00f, .y = +00.00f, .z = +00.00f }                                                                         );
                }
            }
        }
        else
        {
            std::cout << "Assimp error: " << importerD->GetErrorString() << std::endl;
//          std::cout << "Assimp error: " << importerD->GetErrorString() << std::endl;
        }
    }
    catch (const std::exception& e)
//  catch (const std::exception& e)
    {
        std::cout << "General error: " << e.what() << std::endl;
//      std::cout << "General error: " << e.what() << std::endl;
    }



#endif



    for (BVHTree& bvhTree : bvhTreeMain.bvhTrees) for (Geometry& geo : bvhTree.geometries) { CalculateAABB3D(geo); if (geo.geometryType == GeometryType::PRIMITIVE && !geo.primitive.perVertexFrontFaceNormalAvailable) { geo.primitive.vertex0FrontFaceNormal = geo.primitive.vertex1FrontFaceNormal = geo.primitive.vertex2FrontFaceNormal = Normalize(Cross(geo.primitive.vertex1 - geo.primitive.vertex0, geo.primitive.vertex2 - geo.primitive.vertex0)); geo.primitive.perVertexFrontFaceNormalAvailable = true; } }
//  for (BVHTree& bvhTree : bvhTreeMain.bvhTrees) for (Geometry& geo : bvhTree.geometries) { CalculateAABB3D(geo); if (geo.geometryType == GeometryType::PRIMITIVE && !geo.primitive.perVertexFrontFaceNormalAvailable) { geo.primitive.vertex0FrontFaceNormal = geo.primitive.vertex1FrontFaceNormal = geo.primitive.vertex2FrontFaceNormal = Normalize(Cross(geo.primitive.vertex1 - geo.primitive.vertex0, geo.primitive.vertex2 - geo.primitive.vertex0)); geo.primitive.perVertexFrontFaceNormalAvailable = true; } }


//  for (BVHTree& bvhTree : bvhTreeMain.bvhTrees) for (Geometry& geo : bvhTree.geometries)
//  for (BVHTree& bvhTree : bvhTreeMain.bvhTrees) for (Geometry& geo : bvhTree.geometries)
//  {
//      std::cout << "x min:" << geo.aabb3d.intervalAxisX.min << " " << "x max:" << geo.aabb3d.intervalAxisX.max << std::endl;
//      std::cout << "y min:" << geo.aabb3d.intervalAxisY.min << " " << "y max:" << geo.aabb3d.intervalAxisY.max << std::endl;
//      std::cout << "z min:" << geo.aabb3d.intervalAxisZ.min << " " << "z max:" << geo.aabb3d.intervalAxisZ.max << std::endl << std::endl << std::endl;
//  }
//  return 0;



    for (BVHTree& bvhTree : bvhTreeMain.bvhTrees)
//  for (BVHTree& bvhTree : bvhTreeMain.bvhTrees)
    {
        BuildBVHTree(bvhTree, 0, static_cast<int>(bvhTree.geometries.size()));
//      BuildBVHTree(bvhTree, 0, static_cast<int>(bvhTree.geometries.size()));
    }
    BuildBVHTree(bvhTreeMain, 0, static_cast<int>(bvhTreeMain.bvhTrees.size()));
//  BuildBVHTree(bvhTreeMain, 0, static_cast<int>(bvhTreeMain.bvhTrees.size()));



/*
    std::cout << std::endl << std::endl << "<<<Main BVH Tree>>>" << std::endl << std::endl;
//  std::cout << std::endl << std::endl << "<<<Main BVH Tree>>>" << std::endl << std::endl;
    for (const BVHNodeMain& bvhNodeMain : bvhTreeMain.bvhNodeMains)
//  for (const BVHNodeMain& bvhNodeMain : bvhTreeMain.bvhNodeMains)
    {
        std::cout << "x min:" << bvhNodeMain.aabb3d.intervalAxisX.min << " " << "x max:" << bvhNodeMain.aabb3d.intervalAxisX.max << std::endl;
        std::cout << "y min:" << bvhNodeMain.aabb3d.intervalAxisY.min << " " << "y max:" << bvhNodeMain.aabb3d.intervalAxisY.max << std::endl;
        std::cout << "z min:" << bvhNodeMain.aabb3d.intervalAxisZ.min << " " << "z max:" << bvhNodeMain.aabb3d.intervalAxisZ.max << std::endl;
        std::cout << "Sub@ BVH Tree index:" << bvhNodeMain.bvhTreeIndex << "   " << "child L index:" << bvhNodeMain.childIndexL << "   " << "child R index:" << bvhNodeMain.childIndexR << std::endl << std::endl << std::endl;
//      std::cout << "Sub@ BVH Tree index:" << bvhNodeMain.bvhTreeIndex << "   " << "child L index:" << bvhNodeMain.childIndexL << "   " << "child R index:" << bvhNodeMain.childIndexR << std::endl << std::endl << std::endl;
    }
    std::cout << std::endl << std::endl << "<<<Sub@ BVH Tree>>>" << std::endl << std::endl;
//  std::cout << std::endl << std::endl << "<<<Sub@ BVH Tree>>>" << std::endl << std::endl;
    for (const BVHTree& bvhTree : bvhTreeMain.bvhTrees)
//  for (const BVHTree& bvhTree : bvhTreeMain.bvhTrees)
    {
        for (const BVHNode& bvhNode : bvhTree.bvhNodes)
//      for (const BVHNode& bvhNode : bvhTree.bvhNodes)
        {
            std::cout << "x min:" << bvhNode.aabb3d.intervalAxisX.min << " " << "x max:" << bvhNode.aabb3d.intervalAxisX.max << std::endl;
            std::cout << "y min:" << bvhNode.aabb3d.intervalAxisY.min << " " << "y max:" << bvhNode.aabb3d.intervalAxisY.max << std::endl;
            std::cout << "z min:" << bvhNode.aabb3d.intervalAxisZ.min << " " << "z max:" << bvhNode.aabb3d.intervalAxisZ.max << std::endl;
            std::cout << "geometry index:" << bvhNode.geometryIndex << "   " << "child L index:" << bvhNode.childIndexL << "   " << "child R index:" << bvhNode.childIndexR << std::endl << std::endl << std::endl;
//          std::cout << "geometry index:" << bvhNode.geometryIndex << "   " << "child L index:" << bvhNode.childIndexL << "   " << "child R index:" << bvhNode.childIndexR << std::endl << std::endl << std::endl;
        }
    }
    return 0;
*/



    float aspectRatio = 16.0f / 9.0f;
//  float aspectRatio = 16.0f / 9.0f;
    int imgW = 1920 / 1;
//  int imgW = 1920 / 1;
    int imgH = int(imgW / aspectRatio);
//  int imgH = int(imgW / aspectRatio);
    imgH = std::max(imgH, 1);
//  imgH = std::max(imgH, 1);

#ifdef SCENE_001
    constexpr Point3 lookFrom { .x = -028.284f, .y = +000.000f, .z = -028.284f };
//  constexpr Point3 lookFrom { .x = -028.284f, .y = +000.000f, .z = -028.284f };
    constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
//  constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
    constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
//  constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
#endif
#ifdef SCENE_002
    constexpr Point3 lookFrom { .x = +000.000f, .y = +000.000f, .z = +019.990f };
//  constexpr Point3 lookFrom { .x = +000.000f, .y = +000.000f, .z = +019.990f };
    constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
//  constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
    constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
//  constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
#endif
#ifdef SCENE_003
    constexpr Point3 lookFrom { .x = +000.000f, .y = +000.000f, .z = +040.000f };
//  constexpr Point3 lookFrom { .x = +000.000f, .y = +000.000f, .z = +040.000f };
    constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
//  constexpr Point3 lookAt   { .x = +000.000f, .y = +000.000f, .z = +000.000f };
    constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
//  constexpr Point3 viewUp   { .x = +000.000f, .y = +001.000f, .z = +000.000f };
#endif

    Vec3 cameraU; // x
    Vec3 cameraV; // y
    Vec3 cameraW; // z

#ifdef SCENE_001
    float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
    Vec3 defocusDiskRadiusU;
    Vec3 defocusDiskRadiusV;
#endif
#ifdef SCENE_002
    float defocusAngle = 0.02f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.02f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.02f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
//  float defocusAngle = 0.02f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
    Vec3 defocusDiskRadiusU;
    Vec3 defocusDiskRadiusV;
#endif
#ifdef SCENE_003
    float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
    Vec3 defocusDiskRadiusU;
    Vec3 defocusDiskRadiusV;
#endif


    float vFOV = std::numbers::pi_v<float> / 2.5f;
//  float vFOV = std::numbers::pi_v<float> / 2.5f;
    float hFOV = std::numbers::pi_v<float> / 2.5f;
//  float hFOV = std::numbers::pi_v<float> / 2.5f;
    float h = std::tanf(vFOV / 2.0f);
//  float h = std::tanf(vFOV / 2.0f);
    float w = std::tanf(hFOV / 2.0f);
//  float w = std::tanf(hFOV / 2.0f);

    float focalLength = (lookAt - lookFrom).Length();
//  float focalLength = (lookAt - lookFrom).Length();


    cameraW = Normalize(lookFrom - lookAt); cameraU = Normalize(Cross(viewUp, cameraW)); cameraV = Cross(cameraW, cameraU);
//  cameraW = Normalize(lookFrom - lookAt); cameraU = Normalize(Cross(viewUp, cameraW)); cameraV = Cross(cameraW, cameraU);

    float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
//  float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
    defocusDiskRadiusU = cameraU * defocusRadius;
    defocusDiskRadiusV = cameraV * defocusRadius;


    float viewportH = 2.0f * h * focalLength /* focusDistance */;
//  float viewportH = 2.0f * h * focalLength /* focusDistance */;
    float viewportW = viewportH * (float(imgW) / imgH);
//  float viewportW = viewportH * (float(imgW) / imgH);

    Point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;
//  Point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;




    Vec3 viewportU = viewportW *  cameraU;
    Vec3 viewportV = viewportH * -cameraV;


    Vec3 fromPixelToPixelDeltaU = viewportU / float(imgW);
    Vec3 fromPixelToPixelDeltaV = viewportV / float(imgH);



    Point3 viewportTL = cameraCenter - (/* focusDistance */ focalLength * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
//  Point3 viewportTL = cameraCenter - (/* focusDistance */ focalLength * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
    Point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;
//  Point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;



    constexpr int numberOfChannels = 3; // R G B
//  constexpr int numberOfChannels = 3; // R G B
    std::vector<float> rgbs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> rgbs(imgW * imgH * numberOfChannels, 1.0f);
    std::vector<float> normals(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> normals(imgW * imgH * numberOfChannels, 1.0f);
    std::vector<float> albedos(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> albedos(imgW * imgH * numberOfChannels, 1.0f);



    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
#ifdef _DEBUG
    std::clog << "Progress: " << pixelY << "\n";
//  std::clog << "Progress: " << pixelY << "\n";
#endif
    threadPool->Enqueue(
//  threadPool->Enqueue(
    [ pixelY, &imgW, &stratifiedSamplesPerPixel, &inverseStratifiedSamplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, &bvhTreeMain, &rgbs, &albedos, &normals
//  [ pixelY, &imgW, &stratifiedSamplesPerPixel, &inverseStratifiedSamplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, &bvhTreeMain, &rgbs, &albedos, &normals
    ]
    {

    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {



        Color3 pixelColor{};
//      Color3 pixelColor{};
        Color3 albedo{};
//      Color3 albedo{};
        Color3 normal{};
//      Color3 normal{};
        for (int sampleY = 0; sampleY < stratifiedSamplesPerPixel; ++sampleY)
        {
        for (int sampleX = 0; sampleX < stratifiedSamplesPerPixel; ++sampleX)
        {
            Vec3 sampleOffset{ ((sampleX + Random()) * inverseStratifiedSamplesPerPixel) - 0.5f, ((sampleY + Random()) * inverseStratifiedSamplesPerPixel) - 0.5f, 0.0f };
//          Vec3 sampleOffset{ ((sampleX + Random()) * inverseStratifiedSamplesPerPixel) - 0.5f, ((sampleY + Random()) * inverseStratifiedSamplesPerPixel) - 0.5f, 0.0f };
            Point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          Point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          Vec3 rayDirection = pixelSampleCenter - cameraCenter;
//          Vec3 rayDirection = pixelSampleCenter - cameraCenter;
            Vec3 rayOrigin = cameraCenter;
//          Vec3 rayOrigin = cameraCenter;
            if (defocusAngle > 0.0f) [[unlikely]] { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
//          if (defocusAngle > 0.0f) [[unlikely]] { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
            Vec3 rayDirection = pixelSampleCenter - rayOrigin;
//          Vec3 rayDirection = pixelSampleCenter - rayOrigin;
            Ray  ray{ .ori = rayOrigin, .dir = Normalize(rayDirection), .time = Random() };
//          Ray  ray{ .ori = rayOrigin, .dir = Normalize(rayDirection), .time = Random() };
            Color3 ialbedo{};
//          Color3 ialbedo{};
            Color3 inormal{};
//          Color3 inormal{};
            pixelColor += RayColor(ray, bvhTreeMain, 100, backgroundType, inormal, ialbedo);
//          pixelColor += RayColor(ray, bvhTreeMain, 100, backgroundType, inormal, ialbedo);
            albedo += ialbedo;
//          albedo += ialbedo;
            normal += inormal;
//          normal += inormal;
        }
        }
        pixelColor *= pixelSamplesScale;
//      pixelColor *= pixelSamplesScale;
        albedo *= pixelSamplesScale;
//      albedo *= pixelSamplesScale;
        normal *= pixelSamplesScale;
//      normal *= pixelSamplesScale;
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = pixelColor.x;
        rgbs[index + 1] = pixelColor.y;
        rgbs[index + 2] = pixelColor.z;
        albedos[index + 0] = albedo.x;
        albedos[index + 1] = albedo.y;
        albedos[index + 2] = albedo.z;
        normals[index + 0] = normal.x;
        normals[index + 1] = normal.y;
        normals[index + 2] = normal.z;
    }
    });
    }



    #ifdef DISPLAY_WINDOW
//  #ifdef DISPLAY_WINDOW



    rl::InitWindow(imgW, imgH, "Path Tracing");
//  rl::InitWindow(imgW, imgH, "Path Tracing");
    rl::SetTargetFPS(60);
//  rl::SetTargetFPS(60);



    std::size_t tonemapIndex = 0;
//  std::size_t tonemapIndex = 0;
    std::vector<std::uint8_t> rgb8(imgW * imgH * numberOfChannels, 0);
//  std::vector<std::uint8_t> rgb8(imgW * imgH * numberOfChannels, 0);
    rl::Image img { .data = rgb8.data(), .width = imgW, .height = imgH, .mipmaps = 1, .format = rl::PIXELFORMAT_UNCOMPRESSED_R8G8B8 };
//  rl::Image img { .data = rgb8.data(), .width = imgW, .height = imgH, .mipmaps = 1, .format = rl::PIXELFORMAT_UNCOMPRESSED_R8G8B8 };
    rl::Texture2D tex = LoadTextureFromImage(img);
//  rl::Texture2D tex = LoadTextureFromImage(img);
    rl::Rectangle src = { .x = 0.0f, .y = 0.0f, .width = static_cast<float>(imgW), .height = static_cast<float>(imgH) };
//  rl::Rectangle src = { .x = 0.0f, .y = 0.0f, .width = static_cast<float>(imgW), .height = static_cast<float>(imgH) };
    rl::Rectangle dst = { .x = 0.0f, .y = 0.0f, .width = static_cast<float>(imgW), .height = static_cast<float>(imgH) };
//  rl::Rectangle dst = { .x = 0.0f, .y = 0.0f, .width = static_cast<float>(imgW), .height = static_cast<float>(imgH) };



    std::size_t size = static_cast<size_t>(imgW) * imgH;
//  std::size_t size = static_cast<size_t>(imgW) * imgH;
    while (!rl::WindowShouldClose())
//  while (!rl::WindowShouldClose())
    {
        if (rl::IsKeyPressed(rl::KEY_SPACE))
//      if (rl::IsKeyPressed(rl::KEY_SPACE))
        {
            ++tonemapIndex;
//          ++tonemapIndex;
            tonemapIndex %= tonemapsSize;
//          tonemapIndex %= tonemapsSize;
        }
        const TonemapFunctionPointer& tonemap = tonemapsDatabase.tonemaps[tonemapIndex];
//      const TonemapFunctionPointer& tonemap = tonemapsDatabase.tonemaps[tonemapIndex];
        for (std::size_t i = 0; i < size; ++i)
//      for (std::size_t i = 0; i < size; ++i)
        {
            std::size_t ri = 3 * i + 0;
//          std::size_t ri = 3 * i + 0;
            std::size_t gi = 3 * i + 1;
//          std::size_t gi = 3 * i + 1;
            std::size_t bi = 3 * i + 2;
//          std::size_t bi = 3 * i + 2;
            Color3 tonemapColor = tonemap({ .x = rgbs[ri], .y = rgbs[gi], .z = rgbs[bi] });
//          Color3 tonemapColor = tonemap({ .x = rgbs[ri], .y = rgbs[gi], .z = rgbs[bi] });
            rgb8[ri] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.x), 0.0f, 1.0f));
//          rgb8[ri] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.x), 0.0f, 1.0f));
            rgb8[gi] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.y), 0.0f, 1.0f));
//          rgb8[gi] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.y), 0.0f, 1.0f));
            rgb8[bi] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.z), 0.0f, 1.0f));
//          rgb8[bi] = static_cast<std::uint8_t>(255.0f * std::clamp(LinearSpaceToGammasSpace(tonemapColor.z), 0.0f, 1.0f));
        }
        rl::UpdateTexture(tex, rgb8.data());
//      rl::UpdateTexture(tex, rgb8.data());
        rl::BeginDrawing();
//      rl::BeginDrawing();
        rl::ClearBackground(rl::BLACK);
//      rl::ClearBackground(rl::BLACK);
//      dst.width  = static_cast<float>(rl::GetScreenWidth ());
//      dst.width  = static_cast<float>(rl::GetScreenWidth ());
//      dst.height = static_cast<float>(rl::GetScreenHeight());
//      dst.height = static_cast<float>(rl::GetScreenHeight());
        rl::DrawTexturePro(tex, src, dst, { .x = 0.0f, .y = 0.0f, }, 0.0f, rl::WHITE);
//      rl::DrawTexturePro(tex, src, dst, { .x = 0.0f, .y = 0.0f, }, 0.0f, rl::WHITE);
        rl::DrawFPS(10, 10);
//      rl::DrawFPS(10, 10);
        rl::EndDrawing();
//      rl::EndDrawing();
    }



    rl::UnloadTexture(tex);
//  rl::UnloadTexture(tex);
    rl::CloseWindow();
//  rl::CloseWindow();



    #else
//  #else



    threadPool->WaitForAllTasksToBeDone();
//  threadPool->WaitForAllTasksToBeDone();



    std::string fileName = GetCurrentDateTime();
//  std::string fileName = GetCurrentDateTime();
    std::vector<std::ofstream> PPMFiles;
//  std::vector<std::ofstream> PPMFiles;
    PPMFiles.reserve(tonemapsSize);
//  PPMFiles.reserve(tonemapsSize);
    for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
//  for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
    {
        std::string digits = std::to_string(tonemapIndex);
//      std::string digits = std::to_string(tonemapIndex);
        std::size_t digitsSize = digits.size();
//      std::size_t digitsSize = digits.size();
        if (digitsSize == 1)
//      if (digitsSize == 1)
        {
            digits.insert(0, "00");
//          digits.insert(0, "00");
        }
        else
        if (digitsSize == 2)
//      if (digitsSize == 2)
        {
            digits.insert(0, "0");
//          digits.insert(0, "0");
        }
        PPMFiles.emplace_back(((fileName + "_").append(digits).append("_").append(tonemapsDatabase.tonemapNames[tonemapIndex]).append(".ppm")));
//      PPMFiles.emplace_back(((fileName + "_").append(digits).append("_").append(tonemapsDatabase.tonemapNames[tonemapIndex]).append(".ppm")));
        PPMFiles[tonemapIndex] << "P3\n" << imgW << " " << imgH << "\n255\n";
//      PPMFiles[tonemapIndex] << "P3\n" << imgW << " " << imgH << "\n255\n";
    }



#ifdef USE_OIDN
    oidn::DeviceRef oidnDevice = oidn::newDevice();
//  oidn::DeviceRef oidnDevice = oidn::newDevice();
    oidnDevice.commit();
//  oidnDevice.commit();

    std::size_t bufferSize = static_cast<std::size_t>(imgW) * imgH * 3 * sizeof(float);
//  std::size_t bufferSize = static_cast<std::size_t>(imgW) * imgH * 3 * sizeof(float);
    oidn::BufferRef iBuffer = oidnDevice.newBuffer(bufferSize);
//  oidn::BufferRef iBuffer = oidnDevice.newBuffer(bufferSize);
    oidn::BufferRef albedoBuf = oidnDevice.newBuffer(bufferSize);
//  oidn::BufferRef albedoBuf = oidnDevice.newBuffer(bufferSize);
    oidn::BufferRef normalBuf = oidnDevice.newBuffer(bufferSize);
//  oidn::BufferRef normalBuf = oidnDevice.newBuffer(bufferSize);
    oidn::BufferRef oBuffer = oidnDevice.newBuffer(bufferSize);
//  oidn::BufferRef oBuffer = oidnDevice.newBuffer(bufferSize);
    iBuffer.write(0, bufferSize, rgbs.data());
//  iBuffer.write(0, bufferSize, rgbs.data());
    albedoBuf.write(0, bufferSize, albedos.data());
//  albedoBuf.write(0, bufferSize, albedos.data());
    normalBuf.write(0, bufferSize, normals.data());
//  normalBuf.write(0, bufferSize, normals.data());


    oidn::FilterRef oidnFilter = oidnDevice.newFilter("RT");
//  oidn::FilterRef oidnFilter = oidnDevice.newFilter("RT");
    oidnFilter.setImage("color" , iBuffer, oidn::Format::Float3, imgW, imgH);
//  oidnFilter.setImage("color" , iBuffer, oidn::Format::Float3, imgW, imgH);
    oidnFilter.setImage("albedo", albedoBuf, oidn::Format::Float3, imgW, imgH); // auxiliary
//  oidnFilter.setImage("albedo", albedoBuf, oidn::Format::Float3, imgW, imgH); // auxiliary
    oidnFilter.setImage("normal", normalBuf, oidn::Format::Float3, imgW, imgH); // auxiliary
//  oidnFilter.setImage("normal", normalBuf, oidn::Format::Float3, imgW, imgH); // auxiliary
    oidnFilter.setImage("output", oBuffer, oidn::Format::Float3, imgW, imgH);
//  oidnFilter.setImage("output", oBuffer, oidn::Format::Float3, imgW, imgH);
    oidnFilter.set("hdr", true);
//  oidnFilter.set("hdr", true);
    oidnFilter.commit();
//  oidnFilter.commit();

    oidnFilter.execute();
//  oidnFilter.execute();

    const char* oidnErrorMessage = nullptr;
//  const char* oidnErrorMessage = nullptr;
    if (oidnDevice.getError(oidnErrorMessage) != oidn::Error::None)
//  if (oidnDevice.getError(oidnErrorMessage) != oidn::Error::None)
    {
        std::cout << "OIDN Error: " << oidnErrorMessage << std::endl;
//      std::cout << "OIDN Error: " << oidnErrorMessage << std::endl;
    }
    delete [] oidnErrorMessage;
//  delete [] oidnErrorMessage;
    oidnErrorMessage = nullptr;
//  oidnErrorMessage = nullptr;

    oBuffer.read(0, bufferSize, rgbs.data());
//  oBuffer.read(0, bufferSize, rgbs.data());
#endif
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        thread_local static const Interval intensity { 0.000f , 0.999f };
//      thread_local static const Interval intensity { 0.000f , 0.999f };
        for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
//      for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
        {
            Color3 tonemapColor = tonemapsDatabase.tonemaps[tonemapIndex]({ .x = rgbs[index + 0], .y = rgbs[index + 1], .z = rgbs[index + 2], });
//          Color3 tonemapColor = tonemapsDatabase.tonemaps[tonemapIndex]({ .x = rgbs[index + 0], .y = rgbs[index + 1], .z = rgbs[index + 2], });
            PPMFiles[tonemapIndex] << std::setw(3) << static_cast<int>(256 * intensity.Clamp(LinearSpaceToGammasSpace(tonemapColor.x))) << " ";
            PPMFiles[tonemapIndex] << std::setw(3) << static_cast<int>(256 * intensity.Clamp(LinearSpaceToGammasSpace(tonemapColor.y))) << " ";
            PPMFiles[tonemapIndex] << std::setw(3) << static_cast<int>(256 * intensity.Clamp(LinearSpaceToGammasSpace(tonemapColor.z))) << " ";
        }
    }
        for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
//      for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
        {
            PPMFiles[tonemapIndex] << "\n";
//          PPMFiles[tonemapIndex] << "\n";
        }
    }

    for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
//  for (std::size_t tonemapIndex = 0; tonemapIndex < tonemapsSize; ++tonemapIndex)
    {
        PPMFiles[tonemapIndex].close();
//      PPMFiles[tonemapIndex].close();
    }



    #endif
//  #endif

    const std::chrono::steady_clock::time_point& ceaseTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& ceaseTime = std::chrono::high_resolution_clock::now();
    const std::chrono::microseconds& executionDuration = std::chrono::duration_cast<std::chrono::microseconds>(ceaseTime - startTime);
//  const std::chrono::microseconds& executionDuration = std::chrono::duration_cast<std::chrono::microseconds>(ceaseTime - startTime);

    std::cout << executionDuration.count() << " " << "microseconds" << std::endl;
//  std::cout << executionDuration.count() << " " << "microseconds" << std::endl;

    threadPool->Stop();
//  threadPool->Stop();
    delete threadPool;
//  delete threadPool;
    threadPool = nullptr;
//  threadPool = nullptr;

    delete noise;
//  delete noise;
    noise = nullptr;
//  noise = nullptr;

    return 0;
//  return 0;
}


// defocus blur = depth of field
// defocus blur = depth of field


// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO /Gy (/Gw /favor:AMD64 /Zc:inline)
// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO /Gy (/Gw /favor:AMD64 /Zc:inline)
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu





// GAUSSIAN BLUR
// @START
// G(x, y) = (1 / (2 * pi * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma)) | sigma up -> blur up
// G(x, y) = (1 / (2 * pi * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma)) | sigma up -> blur up
// e.g. 3x3 matrix <--- G(x := 1 -> 3, y := 1 -> 3)
// e.g. 3x3 matrix <--- G(x := 1 -> 3, y := 1 -> 3)
// with sigma = 1 then
// with sigma = 1 then
//
// -------------------------
// | 0.075 | 0.124 | 0.075 |
// -------------------------
// | 0.124 | 0.204 | 0.124 |
// -------------------------
// | 0.075 | 0.124 | 0.075 |
// -------------------------
//
// -----------------------------------------
// | 0.005 | 0.021 | 0.034 | 0.021 | 0.005 |
// -----------------------------------------
// | 0.021 | 0.087 | 0.151 | 0.087 | 0.021 |
// -----------------------------------------
// | 0.034 | 0.151 | 0.250 | 0.151 | 0.034 |
// -----------------------------------------
// | 0.021 | 0.087 | 0.151 | 0.087 | 0.021 |
// -----------------------------------------
// | 0.005 | 0.021 | 0.034 | 0.021 | 0.005 |
// -----------------------------------------
//
// ---------------------------------------------------------
// | 0.000 | 0.002 | 0.008 | 0.013 | 0.008 | 0.002 | 0.000 |
// ---------------------------------------------------------
// | 0.002 | 0.013 | 0.054 | 0.089 | 0.054 | 0.013 | 0.002 |
// ---------------------------------------------------------
// | 0.008 | 0.054 | 0.149 | 0.234 | 0.149 | 0.054 | 0.008 |
// ---------------------------------------------------------
// | 0.013 | 0.089 | 0.234 | 0.367 | 0.234 | 0.089 | 0.013 |
// ---------------------------------------------------------
// | 0.008 | 0.054 | 0.149 | 0.234 | 0.149 | 0.054 | 0.008 |
// ---------------------------------------------------------
// | 0.002 | 0.013 | 0.054 | 0.089 | 0.054 | 0.013 | 0.002 |
// ---------------------------------------------------------
// | 0.000 | 0.002 | 0.008 | 0.013 | 0.008 | 0.002 | 0.000 |
// ---------------------------------------------------------
//
// with sigma = 2 then
// with sigma = 2 then
//
// -------------------------
// | 0.102 | 0.115 | 0.102 |
// -------------------------
// | 0.115 | 0.131 | 0.115 |
// -------------------------
// | 0.102 | 0.115 | 0.102 |
// -------------------------
//
// -----------------------------------------
// | 0.041 | 0.052 | 0.059 | 0.052 | 0.041 |
// -----------------------------------------
// | 0.052 | 0.066 | 0.075 | 0.066 | 0.052 |
// -----------------------------------------
// | 0.059 | 0.075 | 0.086 | 0.075 | 0.059 |
// -----------------------------------------
// | 0.052 | 0.066 | 0.075 | 0.066 | 0.052 |
// -----------------------------------------
// | 0.041 | 0.052 | 0.059 | 0.052 | 0.041 |
// -----------------------------------------
//
// ---------------------------------------------------------
// | 0.013 | 0.017 | 0.021 | 0.023 | 0.021 | 0.017 | 0.013 |
// ---------------------------------------------------------
// | 0.017 | 0.021 | 0.026 | 0.029 | 0.026 | 0.021 | 0.017 |
// ---------------------------------------------------------
// | 0.021 | 0.026 | 0.033 | 0.036 | 0.033 | 0.026 | 0.021 |
// ---------------------------------------------------------
// | 0.023 | 0.029 | 0.036 | 0.040 | 0.036 | 0.029 | 0.023 |
// ---------------------------------------------------------
// | 0.021 | 0.026 | 0.033 | 0.036 | 0.033 | 0.026 | 0.021 |
// ---------------------------------------------------------
// | 0.017 | 0.021 | 0.026 | 0.029 | 0.026 | 0.021 | 0.017 |
// ---------------------------------------------------------
// | 0.013 | 0.017 | 0.021 | 0.023 | 0.021 | 0.017 | 0.013 |
// ---------------------------------------------------------
//
// with sigma = 3 then
// with sigma = 3 then
//
// -------------------------
// | 0.106 | 0.112 | 0.106 |
// -------------------------
// | 0.112 | 0.118 | 0.112 |
// -------------------------
// | 0.106 | 0.112 | 0.106 |
// -------------------------
//
// -----------------------------------------
// | 0.054 | 0.059 | 0.061 | 0.059 | 0.054 |
// -----------------------------------------
// | 0.059 | 0.064 | 0.067 | 0.064 | 0.059 |
// -----------------------------------------
// | 0.061 | 0.067 | 0.070 | 0.067 | 0.061 |
// -----------------------------------------
// | 0.059 | 0.064 | 0.067 | 0.064 | 0.059 |
// -----------------------------------------
// | 0.054 | 0.059 | 0.061 | 0.059 | 0.054 |
// -----------------------------------------
//
// ---------------------------------------------------------
// | 0.028 | 0.031 | 0.033 | 0.034 | 0.033 | 0.031 | 0.028 |
// ---------------------------------------------------------
// | 0.031 | 0.034 | 0.036 | 0.037 | 0.036 | 0.034 | 0.031 |
// ---------------------------------------------------------
// | 0.033 | 0.036 | 0.039 | 0.040 | 0.039 | 0.036 | 0.033 |
// ---------------------------------------------------------
// | 0.034 | 0.037 | 0.040 | 0.041 | 0.040 | 0.037 | 0.034 |
// ---------------------------------------------------------
// | 0.033 | 0.036 | 0.039 | 0.040 | 0.039 | 0.036 | 0.033 |
// ---------------------------------------------------------
// | 0.031 | 0.034 | 0.036 | 0.037 | 0.036 | 0.034 | 0.031 |
// ---------------------------------------------------------
// | 0.028 | 0.031 | 0.033 | 0.034 | 0.033 | 0.031 | 0.028 |
// ---------------------------------------------------------
//
// ColorChannel[y x] = ColorChannel[y-1   x-1] * G(y-1 , x-1) + ColorChannel[y-1   x+0] * G(y-1 , x+0) + ColorChannel[y-1   x+1] * G(y-1 , x+1)
//                   + ColorChannel[y+0   x-1] * G(y+0 , x-1) + ColorChannel[y+0   x+0] * G(y+0 , x+0) + ColorChannel[y+0   x+1] * G(y+0 , x+1)
//                   + ColorChannel[y+1   x-1] * G(y+1 , x-1) + ColorChannel[y+1   x+0] * G(y+1 , x+0) + ColorChannel[y+1   x+1] * G(y+1 , x+1)
// @CEASE





//  The current ray tracing implementation is a Monte Carlo-based path tracer with multiple samples per pixel (as opposed to Whitted-style ray tracing). Experiment notes: Consider combining Monte Carlo integration with Whitted - style techniques. Don't forget to experiment with Russian Roulette! Also, explore Next Event Estimation, Multiple Importance Sampling, and Hybrid Bidirectional Path Tracing. Normal Maps. Different types of geometries.
//  The current ray tracing implementation is a Monte Carlo-based path tracer with multiple samples per pixel (as opposed to Whitted-style ray tracing). Experiment notes: Consider combining Monte Carlo integration with Whitted - style techniques. Don't forget to experiment with Russian Roulette! Also, explore Next Event Estimation, Multiple Importance Sampling, and Hybrid Bidirectional Path Tracing. Normal Maps. Different types of geometries.


//  BRDF (Bidirectional Reflectance Distribution Function): Describes how light is reflected from a surface. It relates the amount of light coming from an incoming direction (incident light) to the amount of light going out in an outgoing direction (reflected light), considering only light that stays on the same side of the surface it hit. Think of opaque materials: matte surfaces, metals, plastics, etc. Light hits them and bounces off.
//  BRDF (Bidirectional Reflectance Distribution Function): Describes how light is reflected from a surface. It relates the amount of light coming from an incoming direction (incident light) to the amount of light going out in an outgoing direction (reflected light), considering only light that stays on the same side of the surface it hit. Think of opaque materials: matte surfaces, metals, plastics, etc. Light hits them and bounces off.
//  BTDF (Bidirectional Transmittance Distribution Function): Describes how light is transmitted through a surface. It relates the amount of light coming from an incoming direction to the amount of light going out in an outgoing direction, considering only light that passes through to the opposite side of the surface. Think of transparent or translucent materials: glass, water, thin plastics. Light hits them and goes through (possibly changing direction, i.e., refracting).
//  BTDF (Bidirectional Transmittance Distribution Function): Describes how light is transmitted through a surface. It relates the amount of light coming from an incoming direction to the amount of light going out in an outgoing direction, considering only light that passes through to the opposite side of the surface. Think of transparent or translucent materials: glass, water, thin plastics. Light hits them and goes through (possibly changing direction, i.e., refracting).
//  BSDF (Bidirectional Scattering Distribution Function): This is the general function that describes all the ways light can scatter from a surface interaction point. It encompasses both reflection (BRDF) and transmission (BTDF). BSDF = BRDF + BTDF. It relates the amount of light coming from an incoming direction to the amount of light going out in an outgoing direction, regardless of whether the light stays on the same side (reflects) or passes through to the opposite side (transmits).
//  BSDF (Bidirectional Scattering Distribution Function): This is the general function that describes all the ways light can scatter from a surface interaction point. It encompasses both reflection (BRDF) and transmission (BTDF). BSDF = BRDF + BTDF. It relates the amount of light coming from an incoming direction to the amount of light going out in an outgoing direction, regardless of whether the light stays on the same side (reflects) or passes through to the opposite side (transmits).
//  In short: BRDF handles only reflection. BTDF handles only transmission. BSDF handles both reflection and transmission, making it the more comprehensive term.
//  In short: BRDF handles only reflection. BTDF handles only transmission. BSDF handles both reflection and transmission, making it the more comprehensive term.
//  In modern computer graphics, the term BSDF is often used more generally, even when discussing materials that are purely opaque (where the BTDF component would be zero), because it represents the complete theoretical framework for light interaction at a surface boundary.
//  In modern computer graphics, the term BSDF is often used more generally, even when discussing materials that are purely opaque (where the BTDF component would be zero), because it represents the complete theoretical framework for light interaction at a surface boundary.


//  Forward vs Forward+ vs Deferred pipeline
//  Forward vs Forward+ vs Deferred pipeline
