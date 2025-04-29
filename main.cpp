#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include <string_view>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <array>
#include <iomanip>
#include <cstdint>
#include <numbers>
#include "ThreadPool.h"
#include <chrono>
#include <random>
#include <vector>
#include "Lazy.h"
#include <span>
#include "ImagePNG.h"
#include "ImageJPG.h"
#include "ImageSVG.h"

    enum class BackgroundType : std::uint8_t
//  enum class BackgroundType : std::uint8_t
{
    BLUE_LERP_WHITE = 0,
//  BLUE_LERP_WHITE = 0,
    DARK_ROOM_SPACE = 1,
//  DARK_ROOM_SPACE = 1,
};

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
    float valueLerpVer = (1.0f - deltaY) * valueLerpBot
                       +         deltaY  * valueLerpTop
                       ;

    return valueLerpVer;
//  return valueLerpVer;

    return 0.0f;
}
static inline float Sample3LinearInterpolation(const std::vector<float>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    return 0.0f;
}

  static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { if    (linearSpaceComponent > 0.0f) { return std::sqrt(linearSpaceComponent); } return 0.0f; }
//static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { if    (linearSpaceComponent > 0.0f) { return std::sqrt(linearSpaceComponent); } return 0.0f; }
  static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return gammasSpaceComponent *                          gammasSpaceComponent ;                }
//static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return gammasSpaceComponent *                          gammasSpaceComponent ;                }

    constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
//  constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
    constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();
//  constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();

//static
//inline float Random()
//{
////  thread_local static std::random_device rd;// Non-deterministic seed source
////  thread_local static std::random_device rd;// Non-deterministic seed source
//    thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
////  thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//    thread_local static std::mt19937 generator/*(rd())*/ ;
////  thread_local static std::mt19937 generator/*(rd())*/ ;
//    return distribution(generator);
////  return distribution(generator);
//
////    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//////  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
////    static std::mt19937 generator ;
//////  static std::mt19937 generator ;
////    return distribution(generator);
//////  return distribution(generator);
//}

       thread_local static uint32_t seed = 123456789;
//     thread_local static uint32_t seed = 123456789;
static inline float Random()
{
    seed ^= seed << 13;
//  seed ^= seed << 13;
    seed ^= seed >> 17;
//  seed ^= seed >> 17;
    seed ^= seed << 5 ;
//  seed ^= seed << 5 ;
    return (seed & 0xFFFFFF) / 16777216.0f; // Normalize to [ 0.0f , 1.0f ]
//  return (seed & 0xFFFFFF) / 16777216.0f; // Normalize to [ 0.0f , 1.0f ]
}


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
        std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.ppm", &tm); return std::string(buffer);
//      std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.ppm", &tm); return std::string(buffer);
}


struct Vec3
{
    float x;
    float y;
    float z;

    Vec3  operator- (              ) const { return Vec3 { -x, -y, -z }; }
//  Vec3  operator- (              ) const { return Vec3 { -x, -y, -z }; }
    Vec3& operator+=(const Vec3 & v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3& operator-=(const Vec3 & v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3& operator*=(const Vec3 & v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    Vec3& operator/=(const Vec3 & v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    Vec3& operator*=(const float& v)
    {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }
    Vec3& operator/=(const float& v)
    {
        x /= v;
        y /= v;
        z /= v;
        return *this;
    }

    float Length       () const { return std::sqrt(LengthSquared()); }
//  float Length       () const { return std::sqrt(LengthSquared()); }
    float LengthSquared() const { return x * x
                                       + y * y
                                       + z * z                     ; }

    bool NearZero() const
//  bool NearZero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        // Return true if the vector is close to zero in all dimensions.
        constexpr float s = 1e-8f;
//      constexpr float s = 1e-8f;
        return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
//      return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
    }
};


using Point3 = Vec3;
using Color3 = Vec3;

static inline Vec3 operator+(const Vec3& u, const Vec3& v) { return Vec3 { u.x + v.x, u.y + v.y, u.z + v.z }; }
static inline Vec3 operator-(const Vec3& u, const Vec3& v) { return Vec3 { u.x - v.x, u.y - v.y, u.z - v.z }; }
static inline Vec3 operator*(const Vec3& u, const Vec3& v) { return Vec3 { u.x * v.x, u.y * v.y, u.z * v.z }; }
static inline Vec3 operator/(const Vec3& u, const Vec3& v) { return Vec3 { u.x / v.x, u.y / v.y, u.z / v.z }; }

static inline Vec3 operator*(const Vec3& u, float t) { return Vec3 { u.x * t, u.y * t, u.z * t }; }
static inline Vec3 operator/(const Vec3& u, float t) { return Vec3 { u.x / t, u.y / t, u.z / t }; }

static inline Vec3 operator*(float t, const Vec3& u) { return Vec3 { t * u.x, t * u.y, t * u.z }; }
static inline Vec3 operator/(float t, const Vec3& u) { return Vec3 { t / u.x, t / u.y, t / u.z }; }


    // 4x4 matrix: a flat array of 16 floats (row-major order)
//  // 4x4 matrix: a flat array of 16 floats (row-major order)
    static inline Vec3 operator*(const float* m, const Vec3& v)
//  static inline Vec3 operator*(const float* m, const Vec3& v)
{
    float x = m[ 0] * v.x + m[ 1] * v.y + m[ 2] * v.z + m[ 3];
//  float x = m[ 0] * v.x + m[ 1] * v.y + m[ 2] * v.z + m[ 3];
    float y = m[ 4] * v.x + m[ 5] * v.y + m[ 6] * v.z + m[ 7];
//  float y = m[ 4] * v.x + m[ 5] * v.y + m[ 6] * v.z + m[ 7];
    float z = m[ 8] * v.x + m[ 9] * v.y + m[10] * v.z + m[11];
//  float z = m[ 8] * v.x + m[ 9] * v.y + m[10] * v.z + m[11];
    float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
//  float w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];

    if (w != 0.0f
    &&  w != 1.0f)
    {
        float invW = 1.0f / w;
        x *=  invW           ;
        y *=  invW           ;
        z *=  invW           ;
    }

    return { x, y, z };
//  return { x, y, z };
}


    static inline float Dot(const Vec3& u, const Vec3& v)
//  static inline float Dot(const Vec3& u, const Vec3& v)
{
    return u.x * v.x
         + u.y * v.y
         + u.z * v.z
         ;
}

static
inline Vec3  Cross(const Vec3& u, const Vec3& v)
{
return Vec3 { u.y * v.z - u.z * v.y,
              u.z * v.x - u.x * v.z,
              u.x * v.y - u.y * v.x,
            };
}

  inline static Vec3 Normalize(const Vec3& v) { return v / v.Length(); }
//inline static Vec3 Normalize(const Vec3& v) { return v / v.Length(); }






struct Vec2
{
    float x;
    float y;

    Vec2  operator- (             ) const { return Vec2 { -x, -y }; }
//  Vec2  operator- (             ) const { return Vec2 { -x, -y }; }
    Vec2& operator+=(const Vec2 & v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }
    Vec2& operator-=(const Vec2 & v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }
    Vec2& operator*=(const Vec2 & v)
    {
        x *= v.x;
        y *= v.y;
        return *this;
    }
    Vec2& operator/=(const Vec2 & v)
    {
        x /= v.x;
        y /= v.y;
        return *this;
    }
    Vec2& operator*=(const float& v)
    {
        x *= v;
        y *= v;
        return *this;
    }
    Vec2& operator/=(const float& v)
    {
        x /= v;
        y /= v;
        return *this;
    }

    float Length       () const { return std::sqrt(LengthSquared()); }
//  float Length       () const { return std::sqrt(LengthSquared()); }
    float LengthSquared() const { return x * x
                                       + y * y                     ; }

    bool NearZero() const
//  bool NearZero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        // Return true if the vector is close to zero in all dimensions.
        constexpr float s = 1e-8f;
//      constexpr float s = 1e-8f;
        return (std::fabs(x) < s) && (std::fabs(y) < s);
//      return (std::fabs(x) < s) && (std::fabs(y) < s);
    }
};


using Point2 = Vec2;
using Color2 = Vec2;

static inline Vec2 operator+(const Vec2& u, const Vec2& v) { return Vec2 { u.x + v.x, u.y + v.y }; }
static inline Vec2 operator-(const Vec2& u, const Vec2& v) { return Vec2 { u.x - v.x, u.y - v.y }; }
static inline Vec2 operator*(const Vec2& u, const Vec2& v) { return Vec2 { u.x * v.x, u.y * v.y }; }
static inline Vec2 operator/(const Vec2& u, const Vec2& v) { return Vec2 { u.x / v.x, u.y / v.y }; }

static inline Vec2 operator*(const Vec2& u, float t) { return Vec2 { u.x * t, u.y * t }; }
static inline Vec2 operator/(const Vec2& u, float t) { return Vec2 { u.x / t, u.y / t }; }

static inline Vec2 operator*(float t, const Vec2& u) { return Vec2 { t * u.x, t * u.y }; }
static inline Vec2 operator/(float t, const Vec2& u) { return Vec2 { t / u.x, t / u.y }; }


    static inline float Dot(const Vec2& u, const Vec2& v)
//  static inline float Dot(const Vec2& u, const Vec2& v)
{
    return u.x * v.x
         + u.y * v.y
         ;
}

    static inline Vec2 Cross(const Vec2& u, const Vec2& v)
//  static inline Vec2 Cross(const Vec2& u, const Vec2& v)
{
    return Vec2 { u.y * v.x - u.x * v.y,
                  u.x * v.y - u.y * v.x,
                };
}

  inline static Vec2 Normalize(const Vec2& v) { return v / v.Length(); }
//inline static Vec2 Normalize(const Vec2& v) { return v / v.Length(); }






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
    float valueLerpVerR = (1.0f - deltaY) * valueLerpBotR
                        +         deltaY  * valueLerpTopR
                        ;

    float valueLerpTopG = (1.0f - deltaX) * rgbs[indexOfTLPixel + 1] + deltaX * rgbs[indexOfTRPixel + 1];
    float valueLerpBotG = (1.0f - deltaX) * rgbs[indexOfBLPixel + 1] + deltaX * rgbs[indexOfBRPixel + 1];
    float valueLerpVerG = (1.0f - deltaY) * valueLerpBotG
                        +         deltaY  * valueLerpTopG
                        ;

    float valueLerpTopB = (1.0f - deltaX) * rgbs[indexOfTLPixel + 2] + deltaX * rgbs[indexOfTRPixel + 2];
    float valueLerpBotB = (1.0f - deltaX) * rgbs[indexOfBLPixel + 2] + deltaX * rgbs[indexOfBRPixel + 2];
    float valueLerpVerB = (1.0f - deltaY) * valueLerpBotB
                        +         deltaY  * valueLerpTopB
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
    IMAGE_TEXTURE_PNG = 3,
//  IMAGE_TEXTURE_PNG = 3,
    IMAGE_TEXTURE_JPG = 4,
//  IMAGE_TEXTURE_JPG = 4,
    IMAGE_TEXTURE_SVG = 5,
//  IMAGE_TEXTURE_SVG = 5,
    NOISE_PERLIN = 6,
//  NOISE_PERLIN = 6,
};


    struct Texture
//  struct Texture
{
    Color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  Color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
    float scale = 1.0;
//  float scale = 1.0;
    int imageIndex = -1; // png(s) jpg(s) svg(s)
//  int imageIndex = -1; // png(s) jpg(s) svg(s)
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

    static inline struct ImagesDatabase { std::vector<ImagePNG> pngs; std::vector<ImageJPG> jpgs; std::vector<ImageSVG> svgs; } imagesDatabase;
//  static inline struct ImagesDatabase { std::vector<ImagePNG> pngs; std::vector<ImageJPG> jpgs; std::vector<ImageSVG> svgs; } imagesDatabase;

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


        case TextureType::IMAGE_TEXTURE_PNG:
//      case TextureType::IMAGE_TEXTURE_PNG:
        {
            const ImagePNG& imagePNG = imagesDatabase.pngs[texture.imageIndex];
//          const ImagePNG& imagePNG = imagesDatabase.pngs[texture.imageIndex];

            Interval rgbRange{ 0.0f, 1.0f };
//          Interval rgbRange{ 0.0f, 1.0f };
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
//          uTextureCoordinate =        rgbRange.Clamp(std::fmod(uTextureCoordinate + 0.5f, 1.0f));
            uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
//          uTextureCoordinate =        rgbRange.Clamp(uTextureCoordinate);
            vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);
//          vTextureCoordinate = 1.0f - rgbRange.Clamp(vTextureCoordinate);

            std::uint16_t imagePixelX = static_cast<std::uint16_t>(uTextureCoordinate * (imagePNG.w - 1));
            std::uint16_t imagePixelY = static_cast<std::uint16_t>(vTextureCoordinate * (imagePNG.h - 1));

            size_t imagePixelIndex = (static_cast<size_t>(imagePixelY) * imagePNG.w + imagePixelX) * 3 /* number of color channels */;
//          size_t imagePixelIndex = (static_cast<size_t>(imagePixelY) * imagePNG.w + imagePixelX) * 3 /* number of color channels */;

            return Color3{ .x = imagePNG.rgbs[imagePixelIndex + 0],
                           .y = imagePNG.rgbs[imagePixelIndex + 1],
                           .z = imagePNG.rgbs[imagePixelIndex + 2],
                         };
        }
        break;
//      break;


        case TextureType::IMAGE_TEXTURE_JPG:
//      case TextureType::IMAGE_TEXTURE_JPG:
        {
            return {};
//          return {};
        }
        break;
//      break;


        case TextureType::IMAGE_TEXTURE_SVG:
//      case TextureType::IMAGE_TEXTURE_SVG:
        {
            return {};
//          return {};
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
        while (true)
        {
            const Vec3& p = GenRandom(-1.0f, +1.0f);
//          const Vec3& p = GenRandom(-1.0f, +1.0f);
            const float& pLengthSquared = p.LengthSquared();
//          const float& pLengthSquared = p.LengthSquared();
            if (pLengthSquared <= 1.0000f
            &&  pLengthSquared >  1e-160f)
            {
                return p / std::sqrt(pLengthSquared);
            }
        }
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

inline
static float  BlendLinear(      float startValue,       float ceaseValue,       float ratio)
{
return (1.0f - ratio) * startValue
             + ratio  * ceaseValue;
}
inline
static Vec3   BlendLinear(const Vec3& startValue, const Vec3& ceaseValue,       float ratio)
{
return Vec3 {
              BlendLinear(startValue.x, ceaseValue.x, ratio),
              BlendLinear(startValue.y, ceaseValue.y, ratio),
              BlendLinear(startValue.z, ceaseValue.z, ratio),
            };
}
inline
static Vec3   BlendLinear(const Vec3& startValue, const Vec3& ceaseValue, const Vec3& ratio)
{
return Vec3 {
              BlendLinear(startValue.x, ceaseValue.x, ratio.x),
              BlendLinear(startValue.y, ceaseValue.y, ratio.y),
              BlendLinear(startValue.z, ceaseValue.z, ratio.z),
            };
}


    enum class MaterialType       : std::uint8_t
//  enum class MaterialType       : std::uint8_t
{
    LambertianDiffuseReflectance1 = 0,
//  LambertianDiffuseReflectance1 = 0,
    LambertianDiffuseReflectance2 = 1,
//  LambertianDiffuseReflectance2 = 1,
    Metal = 2,
//  Metal = 2,
    MetalFuzzy1 = 3,
//  MetalFuzzy1 = 3,
    MetalFuzzy2 = 4,
//  MetalFuzzy2 = 4,
    Dielectric  = 5,
//  Dielectric  = 5,
    LightDiffuse = 6,
//  LightDiffuse = 6,
    LightMetalic = 7,
//  LightMetalic = 7,
};


    enum class MaterialDielectric : std::uint8_t
//  enum class MaterialDielectric : std::uint8_t
{
    GLASS = 0,
//  GLASS = 0,
    WATER = 1,
//  WATER = 1,
    AIR = 2,
//  AIR = 2,
    DIAMOND = 3,
//  DIAMOND = 3,
    NOTHING = 4,
//  NOTHING = 4,
};


constexpr inline static float GetRefractionIndex(MaterialDielectric materialDielectric)
{
    switch ( materialDielectric )
    {
        case MaterialDielectric::GLASS  : return 1.500000f; break;
//      case MaterialDielectric::GLASS  : return 1.500000f; break;
        case MaterialDielectric::WATER  : return 1.333000f; break;
//      case MaterialDielectric::WATER  : return 1.333000f; break;
        case MaterialDielectric::AIR    : return 1.000293f; break;
//      case MaterialDielectric::AIR    : return 1.000293f; break;
        case MaterialDielectric::DIAMOND: return 2.400000f; break;
//      case MaterialDielectric::DIAMOND: return 2.400000f; break;
                                 default: return 0.000000f; break;
//                               default: return 0.000000f; break;
    }
}


    struct Material
//  struct Material
{
    float scatteredProbability; float fuzz; float refractionIndex; std::uint8_t textureIndex; MaterialType materialType;
//  float scatteredProbability; float fuzz; float refractionIndex; std::uint8_t textureIndex; MaterialType materialType;
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
};

    struct Geometry
//  struct Geometry
{
    union { struct Sphere { Point3 center; float radius; } sphere; struct Primitive { Point3 vertex0; Point3 vertex1; Point3 vertex2; Vec3 frontFaceNormal; float frontFaceVertex0U; float frontFaceVertex0V; float frontFaceVertex1U; float frontFaceVertex1V; float frontFaceVertex2U; float frontFaceVertex2V; float backFaceVertex0U; float backFaceVertex0V; float backFaceVertex1U; float backFaceVertex1V; float backFaceVertex2U; float backFaceVertex2V; } primitive; }; AABB3D aabb3d; Material material; Vec3 movingDirection; GeometryType geometryType;
//  union { struct Sphere { Point3 center; float radius; } sphere; struct Primitive { Point3 vertex0; Point3 vertex1; Point3 vertex2; Vec3 frontFaceNormal; float frontFaceVertex0U; float frontFaceVertex0V; float frontFaceVertex1U; float frontFaceVertex1V; float frontFaceVertex2U; float frontFaceVertex2V; float backFaceVertex0U; float backFaceVertex0V; float backFaceVertex1U; float backFaceVertex1V; float backFaceVertex2U; float backFaceVertex2V; } primitive; }; AABB3D aabb3d; Material material; Vec3 movingDirection; GeometryType geometryType;

};


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
                constexpr float padding = 1e-3f; // COULD BE LOWER
//              constexpr float padding = 1e-3f; // COULD BE LOWER
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
                constexpr float padding = 1e-3f; // COULD BE LOWER
//              constexpr float padding = 1e-3f; // COULD BE LOWER
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
    Material material; Point3 at; Vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace;
//  Material material; Point3 at; Vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace;
};






inline static MaterialScatteredResult Scatter(const Ray& rayIn, const RayHitResult& rayHitResult)
{
    MaterialScatteredResult materialScatteredResult {};
//  MaterialScatteredResult materialScatteredResult {};
    switch (rayHitResult.material.materialType)
//  switch (rayHitResult.material.materialType)
    {

    case MaterialType::LambertianDiffuseReflectance1:
//  case MaterialType::LambertianDiffuseReflectance1:
        {
            Vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
//          Vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            if (scatteredDirection.NearZero()) _UNLIKELY
//          if (scatteredDirection.NearZero()) _UNLIKELY
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
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            Vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
//          Vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            if (scatteredDirection.NearZero()) _UNLIKELY
//          if (scatteredDirection.NearZero()) _UNLIKELY
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
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            reflectionScatteredDirection = Normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
//          reflectionScatteredDirection = Normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            reflectionScatteredDirection = Normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          reflectionScatteredDirection = Normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
//          float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
            if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
//          if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
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

            if ( notAbleToRefract )
            {
                 scatteredRayDirection = Reflect(normalizedIncomingRayDirection, rayHitResult.normal);
//               scatteredRayDirection = Reflect(normalizedIncomingRayDirection, rayHitResult.normal);
            }
            else
            {
                 scatteredRayDirection = Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat);
//               scatteredRayDirection = Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat);
            }

            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* color3 { 1.0f, 1.0f, 1.0f }  */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* color3 { 1.0f, 1.0f, 1.0f }  */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//          materialScatteredResult.emission = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
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
            materialScatteredResult.emission = Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.emission = Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
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
            materialScatteredResult.emission = Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.emission = Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            materialScatteredResult.isScattered = false;
//          materialScatteredResult.isScattered = false;
        }
        break;
//      break;



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
            RayHitResult rayHitResult { .material = geo.material };
//          RayHitResult rayHitResult { .material = geo.material };
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

            RayHitResult rayHitResult{ .material = geo.material };
//          RayHitResult rayHitResult{ .material = geo.material };

            // Small epsilon to handle floating-point inaccuracies
//          // Small epsilon to handle floating-point inaccuracies
            constexpr float EPSILON = 1e-8f;
//          constexpr float EPSILON = 1e-8f;

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

            // Compute the w0 barycentric coordinate
//          // Compute the w0 barycentric coordinate
            float w0Barycentric = inverseDeterminant * Dot(vectorFromPrimitiveVertex0ToRayOrigin, rayDirectionCrossPrimitiveEdge2);
//          float w0Barycentric = inverseDeterminant * Dot(vectorFromPrimitiveVertex0ToRayOrigin, rayDirectionCrossPrimitiveEdge2);
            if  ( w0Barycentric < 0.0f
            ||    w0Barycentric > 1.0f )
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult; // Intersection lies outside the primitive
//              return rayHitResult; // Intersection lies outside the primitive
            }

            // Compute the w1 barycentric coordinate
//          // Compute the w1 barycentric coordinate
            Vec3 rayOriginCrossPrimitiveEdge1 = Cross(vectorFromPrimitiveVertex0ToRayOrigin, primitiveEdge1);
//          Vec3 rayOriginCrossPrimitiveEdge1 = Cross(vectorFromPrimitiveVertex0ToRayOrigin, primitiveEdge1);
            float w1Barycentric = inverseDeterminant * Dot(ray.dir, rayOriginCrossPrimitiveEdge1);
//          float w1Barycentric = inverseDeterminant * Dot(ray.dir, rayOriginCrossPrimitiveEdge1);
            if  ( w1Barycentric < 0.0f
            ||    w0Barycentric +
                  w1Barycentric > 1.0f )
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
// 
                rayHitResult.isFrontFace = Dot(ray.dir, geo.primitive.frontFaceNormal) < 0.0f;
//              rayHitResult.isFrontFace = Dot(ray.dir, geo.primitive.frontFaceNormal) < 0.0f;
                float w2Barycentric = 1.0f - w0Barycentric - w1Barycentric;
//              float w2Barycentric = 1.0f - w0Barycentric - w1Barycentric;
                if (rayHitResult.isFrontFace)
//              if (rayHitResult.isFrontFace)
                {
                    rayHitResult.normal =  geo.primitive.frontFaceNormal;
//                  rayHitResult.normal =  geo.primitive.frontFaceNormal;

                    rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0U + w1Barycentric * geo.primitive.frontFaceVertex1U + w2Barycentric * geo.primitive.frontFaceVertex2U;
//                  rayHitResult.uSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0U + w1Barycentric * geo.primitive.frontFaceVertex1U + w2Barycentric * geo.primitive.frontFaceVertex2U;
                    rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0V + w1Barycentric * geo.primitive.frontFaceVertex1V + w2Barycentric * geo.primitive.frontFaceVertex2V;
//                  rayHitResult.vSurfaceCoordinate = w0Barycentric * geo.primitive.frontFaceVertex0V + w1Barycentric * geo.primitive.frontFaceVertex1V + w2Barycentric * geo.primitive.frontFaceVertex2V;
                }
                else
//              else
                {
                    rayHitResult.normal = -geo.primitive.frontFaceNormal;
//                  rayHitResult.normal = -geo.primitive.frontFaceNormal;

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


        default:
//      default:
        {
            return { .material = geo.material };
//          return { .material = geo.material };
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
        if (temporaryRayHitResult.hitted) _UNLIKELY
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
static Color3 RayColor(const Ray& ray)
{
    Geometry sphere{ .sphere = { .center = { .x = 0.0f, .y = 0.0f, .z = -1.0f }, .radius = 0.5f, }, .geometryType = GeometryType::SPHERE };
//  Geometry sphere{ .sphere = { .center = { .x = 0.0f, .y = 0.0f, .z = -1.0f }, .radius = 0.5f, }, .geometryType = GeometryType::SPHERE };
    const RayHitResult& rayHitResult = RayHit(sphere, ray, Interval { .min = -10.0f, .max = +10.0f });
//  const RayHitResult& rayHitResult = RayHit(sphere, ray, Interval { .min = -10.0f, .max = +10.0f });
    if (rayHitResult.hitted) _UNLIKELY
//  if (rayHitResult.hitted) _UNLIKELY
    {
        return Color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
//      return Color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
    }

    const Vec3& normalizedRayDirection = Normalize(ray.dir);
//  const Vec3& normalizedRayDirection = Normalize(ray.dir);
    const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//  const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
    return BlendLinear(Color3{ 1.0f, 1.0f, 1.0f, }, Color3{ 0.5f, 0.7f, 1.0f, }, ratio);
//  return BlendLinear(Color3{ 1.0f, 1.0f, 1.0f, }, Color3{ 0.5f, 0.7f, 1.0f, }, ratio);
}



/*
inline
static Color3 RayColor(const Ray& ray, const std::vector<Geometry>& geometries, int recursiveDepth = 50)
{
    if (recursiveDepth <= 0.0f)
    {
        return Color3 {};
//      return Color3 {};
    }
    const RayHitResult& rayHitResult = RayHit(geometries, ray, Interval { .min = 0.001f, .max = positiveInfinity });
//  const RayHitResult& rayHitResult = RayHit(geometries, ray, Interval { .min = 0.001f, .max = positiveInfinity });
    if (rayHitResult.hitted)
//  if (rayHitResult.hitted)
    {
        const MaterialScatteredResult& materialScatteredResult = Scatter(ray, rayHitResult);
//      const MaterialScatteredResult& materialScatteredResult = Scatter(ray, rayHitResult);

        if (!materialScatteredResult.isScattered)
//      if (!materialScatteredResult.isScattered)
        {
            return Color3 {};
//          return Color3 {};
        }

        return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, geometries, --recursiveDepth);
//      return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, geometries, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, geometries, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, geometries, --recursiveDepth);
//      return Color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
//      return Color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
    }

    const Vec3& normalizedRayDirection = Normalize(ray.dir);
//  const Vec3& normalizedRayDirection = Normalize(ray.dir);
    const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//  const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
    return BlendLinear(Color3{ 1.0f, 1.0f, 1.0f, }, Color3{ 0.5f, 0.7f, 1.0f, }, ratio);
//  return BlendLinear(Color3{ 1.0f, 1.0f, 1.0f, }, Color3{ 0.5f, 0.7f, 1.0f, }, ratio);
}
*/



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
        const RayHitResult& rayHitResult = RayHit(geometries, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });
//      const RayHitResult& rayHitResult = RayHit(geometries, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });

        if (rayHitResult.hitted) _UNLIKELY
//      if (rayHitResult.hitted) _UNLIKELY
        {
            const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered) _UNLIKELY
//          if (!materialScatteredResult.isScattered) _UNLIKELY
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




struct BVHNode
{
    AABB3D  aabb3d      ;
    int shapeIndex  = -1;
    int childIndexL = -1;
    int childIndexR = -1;
};
struct BVHTree
{
    std::vector<BVHNode> bvhNodes; std::vector<Geometry> geometries;
//  std::vector<BVHNode> bvhNodes; std::vector<Geometry> geometries;
};
inline static RayHitResult RayHit(const BVHTree& bvhTree, int bvhNodeIndex, const Ray& ray, const Interval& rayT)
{
    const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];
//  const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];

    // Leaf node: test geometry intersection
    // Leaf node: test geometry intersection
    if (bvhNode.shapeIndex != -1)
//  if (bvhNode.shapeIndex != -1)
    {
        return RayHit(bvhTree.geometries[bvhNode.shapeIndex], ray, rayT);
//      return RayHit(bvhTree.geometries[bvhNode.shapeIndex], ray, rayT);
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
    RayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray,        rayT);
//  RayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray,        rayT);
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
/*
static inline bool AABB3DCompareAxisX(const Geometry& g1, const Geometry& g2) { return g1.aabb3d.intervalAxisX.min < g2.aabb3d.intervalAxisX.min; }
static inline bool AABB3DCompareAxisY(const Geometry& g1, const Geometry& g2) { return g1.aabb3d.intervalAxisY.min < g2.aabb3d.intervalAxisY.min; }
static inline bool AABB3DCompareAxisZ(const Geometry& g1, const Geometry& g2) { return g1.aabb3d.intervalAxisZ.min < g2.aabb3d.intervalAxisZ.min; }
inline static int  BuildBVHTree(BVHTree& bvhTree, int start, int cease)
{
    int objectSpan = cease - start;
//  int objectSpan = cease - start;
    if (objectSpan == 1)
    {
        // Single geometry: create a leaf node directly
        // Single geometry: create a leaf node directly
        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1 });
//      bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1 });
        return current;
    }
    else
    if (objectSpan == 2)
    {
        // Two geometries: create a parent with two leaf children
        // Two geometries: create a parent with two leaf children
        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
        // Reserve space for parent node
        // Reserve space for parent node
        bvhTree.bvhNodes.emplace_back(BVHNode{});
//      bvhTree.bvhNodes.emplace_back(BVHNode{});

        // Create left and right leaf nodes
        // Create left and right leaf nodes
        int childIndexL = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start + 0].aabb3d, .shapeIndex = start + 0, .childIndexL = -1, .childIndexR = -1 });
        int childIndexR = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start + 1].aabb3d, .shapeIndex = start + 1, .childIndexL = -1, .childIndexR = -1 });

        // Compute combined AABB for parent
        // Compute combined AABB for parent
        // Update parent node
        // Update parent node
        const BVHNode& bvhNodeL = bvhTree.bvhNodes[childIndexL];
        const BVHNode& bvhNodeR = bvhTree.bvhNodes[childIndexR];
        bvhTree.bvhNodes[current].aabb3d.intervalAxisX.min = std::fminf(bvhNodeL.aabb3d.intervalAxisX.min, bvhNodeR.aabb3d.intervalAxisX.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisX.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisX.max, bvhNodeR.aabb3d.intervalAxisX.max);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisY.min = std::fminf(bvhNodeL.aabb3d.intervalAxisY.min, bvhNodeR.aabb3d.intervalAxisY.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisY.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisY.max, bvhNodeR.aabb3d.intervalAxisY.max);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisZ.min = std::fminf(bvhNodeL.aabb3d.intervalAxisZ.min, bvhNodeR.aabb3d.intervalAxisZ.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisZ.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisZ.max, bvhNodeR.aabb3d.intervalAxisZ.max);
        bvhTree.bvhNodes[current].shapeIndex  = -1;
        bvhTree.bvhNodes[current].childIndexL = childIndexL;
        bvhTree.bvhNodes[current].childIndexR = childIndexR;
        
        return current;
    }
    else
    {
        // Multiple geometries: recursive split
        // Multiple geometries: recursive split
        int axis = RandomInt(0, 2);
//      int axis = RandomInt(0, 2);
        std::function<bool(const Geometry&, const Geometry&)> comparator;
//      std::function<bool(const Geometry&, const Geometry&)> comparator;
        if (axis == 0)
        {
            comparator = AABB3DCompareAxisX;
        }
        else
        if (axis == 1)
        {
            comparator = AABB3DCompareAxisY;
        }
        else
        if (axis == 2)
        {
            comparator = AABB3DCompareAxisZ;
        }

        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{}); // Placeholder for parent
//      bvhTree.bvhNodes.emplace_back(BVHNode{}); // Placeholder for parent

        // Sort and split
        // Sort and split
        std::sort(std::begin(bvhTree.geometries) + start,
                  std::begin(bvhTree.geometries) + cease, comparator);
        int mid = start + objectSpan / 2;
//      int mid = start + objectSpan / 2;

        // Recursively build subtrees
        // Recursively build subtrees
        int childIndexL = BuildBVHTree(bvhTree, start, mid       );
//      int childIndexL = BuildBVHTree(bvhTree, start, mid       );
        int childIndexR = BuildBVHTree(bvhTree,        mid, cease);
//      int childIndexR = BuildBVHTree(bvhTree,        mid, cease);

        // Combine AABBs from children
        // Combine AABBs from children
        // Update parent node
        // Update parent node
        const BVHNode& bvhNodeL = bvhTree.bvhNodes[childIndexL];
        const BVHNode& bvhNodeR = bvhTree.bvhNodes[childIndexR];
        bvhTree.bvhNodes[current].aabb3d.intervalAxisX.min = std::fminf(bvhNodeL.aabb3d.intervalAxisX.min, bvhNodeR.aabb3d.intervalAxisX.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisX.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisX.max, bvhNodeR.aabb3d.intervalAxisX.max);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisY.min = std::fminf(bvhNodeL.aabb3d.intervalAxisY.min, bvhNodeR.aabb3d.intervalAxisY.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisY.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisY.max, bvhNodeR.aabb3d.intervalAxisY.max);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisZ.min = std::fminf(bvhNodeL.aabb3d.intervalAxisZ.min, bvhNodeR.aabb3d.intervalAxisZ.min);
        bvhTree.bvhNodes[current].aabb3d.intervalAxisZ.max = std::fmaxf(bvhNodeL.aabb3d.intervalAxisZ.max, bvhNodeR.aabb3d.intervalAxisZ.max);
        bvhTree.bvhNodes[current].shapeIndex  = -1;
        bvhTree.bvhNodes[current].childIndexL = childIndexL;
        bvhTree.bvhNodes[current].childIndexR = childIndexR;

        return current;
    }
}
*/
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
            int current = (int)bvhTree.bvhNodes.size();
//          int current = (int)bvhTree.bvhNodes.size();
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
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
                Union(lAABB3Ds[i - 1], bvhTree.geometries[start + i].aabb3d, lAABB3Ds[i]);
//              Union(lAABB3Ds[i - 1], bvhTree.geometries[start + i].aabb3d, lAABB3Ds[i]);
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
                Union(bvhTree.geometries[start + i].aabb3d, rAABB3Ds[i + 1], rAABB3Ds[i]);
//              Union(bvhTree.geometries[start + i].aabb3d, rAABB3Ds[i + 1], rAABB3Ds[i]);
            }

            // Evaluate all possible splits
            // Evaluate all possible splits
            for (int i = 0; i < objectSpan - 1; ++i)
//          for (int i = 0; i < objectSpan - 1; ++i)
            {
                float   cost = SurfaceArea(lAABB3Ds[i    ]) * (             i + 1)
                             + SurfaceArea(rAABB3Ds[i + 1]) * (objectSpan - i - 1);
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
            int current = (int)bvhTree.bvhNodes.size();
//          int current = (int)bvhTree.bvhNodes.size();
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.geometries[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
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
        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
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
        Union(bvhNodeL         .aabb3d ,
              bvhNodeR         .aabb3d ,
      bvhTree.bvhNodes[current].aabb3d);
      bvhTree.bvhNodes[current].shapeIndex  = -1         ;
      bvhTree.bvhNodes[current].childIndexL = childIndexL;
      bvhTree.bvhNodes[current].childIndexR = childIndexR;

      return current;
//    return current;
}
/*
    inline static Color3 RayColor(const Ray& initialRay, const BVHTree& bvhTree, int maxDepth, BackgroundType backgroundType) // maxDepth = RAYS BOUNCING DEPTH
//  inline static Color3 RayColor(const Ray& initialRay, const BVHTree& bvhTree, int maxDepth, BackgroundType backgroundType) // maxDepth = RAYS BOUNCING DEPTH
{
    Color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
//  Color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
    Ray currentRay = initialRay;
//  Ray currentRay = initialRay;

    for (int depth = 0; depth < maxDepth; ++depth)
//  for (int depth = 0; depth < maxDepth; ++depth)
    {
        const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });
//      const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });

        if (rayHitResult.hitted) _UNLIKELY
//      if (rayHitResult.hitted) _UNLIKELY
        {
            const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const MaterialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered) _UNLIKELY // Notes: Non-scattering materials are mostly emissive
//          if (!materialScatteredResult.isScattered) _UNLIKELY // Notes: Non-scattering materials are mostly emissive
            {
                // L_out = emission
//              // L_out = emission
                return materialScatteredResult.emission; // Notes: When a light ray hits a non-scattering material - it will be absorbed by the material - meaning the path tracing will end with only the color emitted by the material
//              return materialScatteredResult.emission; // Notes: When a light ray hits a non-scattering material - it will be absorbed by the material - meaning the path tracing will end with only the color emitted by the material
            }

            // Multiply the current color by the attenuation
//          // Multiply the current color by the attenuation
            // L_out = (L_in * attenuation) + emission
//          // L_out = (L_in * attenuation) + emission
            finalColor = (finalColor * materialScatteredResult.attenuation) + materialScatteredResult.emission; // Notes: When a light ray hits a scattering material - it will continue bouncing - carrying both its attenuated color and any emission from the material
//          finalColor = (finalColor * materialScatteredResult.attenuation) + materialScatteredResult.emission; // Notes: When a light ray hits a scattering material - it will continue bouncing - carrying both its attenuated color and any emission from the material
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
            Color3 backgroundColor;
//          Color3 backgroundColor;
            switch (backgroundType)
//          switch (backgroundType)
            {
                case BackgroundType::BLUE_LERP_WHITE:
//              case BackgroundType::BLUE_LERP_WHITE:
                {
                    const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
//                  const Vec3& normalizedRayDirection = Normalize(currentRay.dir);
                    const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//                  const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
                    backgroundColor = BlendLinear(Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, Color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
//                  backgroundColor = BlendLinear(Color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, Color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
                }
                break;
//              break;


                case BackgroundType::DARK_ROOM_SPACE:
//              case BackgroundType::DARK_ROOM_SPACE:
                {
                    backgroundColor = { .x = 0.05f, .y = 0.05f, .z = 0.05f };
//                  backgroundColor = { .x = 0.05f, .y = 0.05f, .z = 0.05f };
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
            return finalColor * backgroundColor;
//          return finalColor * backgroundColor;
        }
    }

    // If we reach max depth, return black
//  // If we reach max depth, return black
    return Color3{};
//  return Color3{};
}
*/
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
            const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });
//          const RayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, Interval{ .min = 0.001f, .max = positiveInfinity });

            if (!rayHitResult.hitted) _UNLIKELY
//          if (!rayHitResult.hitted) _UNLIKELY
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
                        backgroundColor = { .x = 0.05f, .y = 0.05f, .z = 0.05f };
//                      backgroundColor = { .x = 0.05f, .y = 0.05f, .z = 0.05f };
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

            if (!scatterResult.isScattered) _UNLIKELY
//          if (!scatterResult.isScattered) _UNLIKELY
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



    static inline Vec2 Saturate(const Vec2& v)
//  static inline Vec2 Saturate(const Vec2& v)
    {
        return
        {
            .x = std::clamp(v.x, 0.0f, 1.0f),
            .y = std::clamp(v.y, 0.0f, 1.0f),
        };

    }
    static inline Vec3 Saturate(const Vec3& v)
//  static inline Vec3 Saturate(const Vec3& v)
    {
        return
        {
            .x = std::clamp(v.x, 0.0f, 1.0f),
            .y = std::clamp(v.y, 0.0f, 1.0f),
            .z = std::clamp(v.z, 0.0f, 1.0f),
        };
    }
    static inline Vec3 TonemapACES(const Vec3& v)
//  static inline Vec3 TonemapACES(const Vec3& v)
    {
        return Saturate((v * (2.51f * v + Vec3{ .x = 0.03f, .y = 0.03f, .z = 0.03f })) / (v * (2.43f * v + Vec3{ .x = 0.59f, .y = 0.59f, .z = 0.59f }) + Vec3{ .x = 0.14f, .y = 0.14f, .z = 0.14f }));
//      return Saturate((v * (2.51f * v + Vec3{ .x = 0.03f, .y = 0.03f, .z = 0.03f })) / (v * (2.43f * v + Vec3{ .x = 0.59f, .y = 0.59f, .z = 0.59f }) + Vec3{ .x = 0.14f, .y = 0.14f, .z = 0.14f }));
    }
    static inline Vec3 TonemapFilmic(const Vec3& v)
//  static inline Vec3 TonemapFilmic(const Vec3& v)
    {
        Vec3 m{ .x = std::fmaxf(0.0f, v.x - 0.004f), .y = std::fmaxf(0.0f, v.y - 0.004f), .z = std::fmaxf(0.0f, v.z - 0.004f) };
//      Vec3 m{ .x = std::fmaxf(0.0f, v.x - 0.004f), .y = std::fmaxf(0.0f, v.y - 0.004f), .z = std::fmaxf(0.0f, v.z - 0.004f) };
        return (m * (6.2f * m + Vec3{ .x = 0.5f, .y = 0.5f, .z = 0.5f })) / (m * (6.2f * m + Vec3{ .x = 1.7f, .y = 1.7f, .z = 1.7f }) + Vec3{ .x = 0.06f, .y = 0.06f, .z = 0.06f });
//      return (m * (6.2f * m + Vec3{ .x = 0.5f, .y = 0.5f, .z = 0.5f })) / (m * (6.2f * m + Vec3{ .x = 1.7f, .y = 1.7f, .z = 1.7f }) + Vec3{ .x = 0.06f, .y = 0.06f, .z = 0.06f });
    }
    static inline Vec3 TonemapReinhard(const Vec3& v)
//  static inline Vec3 TonemapReinhard(const Vec3& v)
    {
        return v / (1.0f + Dot(v, Vec3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }));
//      return v / (1.0f + Dot(v, Vec3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }));
    }
    static inline Vec3 TonemapReinhardJodie(const Vec3& v)
//  static inline Vec3 TonemapReinhardJodie(const Vec3& v)
    {
        float l = Dot(v, Vec3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }); Vec3 tc = v / (v + Vec3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }); return BlendLinear(v / (l + 1.0f), tc, tc);
//      float l = Dot(v, Vec3{ .x = 0.21250175f, .y = 0.71537574f, .z = 0.07212251f }); Vec3 tc = v / (v + Vec3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }); return BlendLinear(v / (l + 1.0f), tc, tc);
    }
    static inline Vec3 TonemapUncharted2(const Vec3& v)
//  static inline Vec3 TonemapUncharted2(const Vec3& v)
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
        constexpr Vec3 VCB{ .x = C * B, .y = C * B, .z = C * B };
        constexpr Vec3 VB { .x =     B, .y =     B, .z =     B };
        constexpr Vec3 VDE{ .x = D * E, .y = D * E, .z = D * E };
        constexpr Vec3 VDF{ .x = D * F, .y = D * F, .z = D * F };
        constexpr Vec3 VEF{ .x = E / F, .y = E / F, .z = E / F };
        return ((v * (A * v + VCB) + VDE) / (v * (A * v + VB) + VDF)) - VEF;
//      return ((v * (A * v + VCB) + VDE) / (v * (A * v + VB) + VDF)) - VEF;
    }
    static inline Vec3 TonemapUncharted1(const Vec3& v)
//  static inline Vec3 TonemapUncharted1(const Vec3& v)
    {
        constexpr float W = 11.2f;
//      constexpr float W = 11.2f;
        constexpr float exposureBias = 2.0f;
//      constexpr float exposureBias = 2.0f;
        Vec3 curr = TonemapUncharted2(exposureBias * v);
//      Vec3 curr = TonemapUncharted2(exposureBias * v);
        Vec3 whiteScale = 1.0f / TonemapUncharted2(Vec3{ .x = W, .y = W, .z = W });
//      Vec3 whiteScale = 1.0f / TonemapUncharted2(Vec3{ .x = W, .y = W, .z = W });
        return curr * whiteScale;
//      return curr * whiteScale;
    }
    static inline Vec3 TonemapUnreal(const Vec3& v)
//  static inline Vec3 TonemapUnreal(const Vec3& v)
    {
        return v / (v + Vec3{ .x = 0.155f, .y = 0.155f, .z = 0.155f }) * 1.019f;
//      return v / (v + Vec3{ .x = 0.155f, .y = 0.155f, .z = 0.155f }) * 1.019f;
    }



int main()
{
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::BLOCKY          , .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::NOISE_NORMALIZED });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_1     });
    noisesDatabase.noisePerlins.emplace_back(NoisePerlin{ .noisePerlinType = NoisePerlinType::SMOOTH_SHIFT_OFF, .noisePerlinProcedureType = NoisePerlinProcedureType::TURBULENCE_2     });
    for (NoisePerlin& np : noisesDatabase.noisePerlins) Generate(np);
//  for (NoisePerlin& np : noisesDatabase.noisePerlins) Generate(np);



    imagesDatabase.pngs.emplace_back("smile-face-001.png");
    imagesDatabase.pngs.emplace_back("smile-face-002.png");
//  imagesDatabase.pngs.emplace_back("example-001.png");
//  imagesDatabase.pngs.emplace_back("example-002.png");
//  imagesDatabase.jpgs.emplace_back("example-001.jpg");
//  imagesDatabase.jpgs.emplace_back("example-002.jpg");
//  imagesDatabase.svgs.emplace_back("example-001.svg");
//  imagesDatabase.svgs.emplace_back("example-002.svg");



    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 1.0f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 0.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.2f, 0.2f, 0.2f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.5f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +3, .eTileTextureIndex = +4, .type = TextureType::CHECKER_TEXTURE_1, });
//  texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.5f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +3, .eTileTextureIndex = +4, .type = TextureType::CHECKER_TEXTURE_1, });

    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.5f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 0.5f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });

    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.1f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +6, .eTileTextureIndex = +7, .type = TextureType::CHECKER_TEXTURE_1, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.1f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +6, .eTileTextureIndex = +7, .type = TextureType::CHECKER_TEXTURE_2, });

    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_PNG, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_PNG, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_JPG, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_JPG, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_SVG, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::IMAGE_TEXTURE_SVG, });

    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 5.0f, .imageIndex = -1, .noiseIndex = +0, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::NOISE_PERLIN, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 5.0f, .imageIndex = -1, .noiseIndex = +1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::NOISE_PERLIN, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 5.0f, .imageIndex = -1, .noiseIndex = +2, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::NOISE_PERLIN, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 5.0f, .imageIndex = -1, .noiseIndex = +3, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::NOISE_PERLIN, });

    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 1.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 0.5f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 1.0f, 0.5f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(Texture{ .albedo = { 0.5f, 1.0f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = TextureType::SOLID_COLOR, });

    

    ThreadPool* threadPool = new ThreadPool();
//  ThreadPool* threadPool = new ThreadPool();
    threadPool->WarmUp(255);
//  threadPool->WarmUp(255);

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    int                              samplesPerPixel = 1 ;
//  int                              samplesPerPixel = 1 ;
    float pixelSamplesScale = 1.0f / samplesPerPixel        ;
//  float pixelSamplesScale = 1.0f / samplesPerPixel        ;



//  std::vector<Geometry> geometries;
//  std::vector<Geometry> geometries;
    BVHTree bvhTree;
//  BVHTree bvhTree;
/*
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y = -5000.5000f, .z =  0000.0000f }, .radius = 5000.0000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 3, .materialType = MaterialType::MetalFuzzy1, }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
//  bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y = -5000.5000f, .z =  0000.0000f }, .radius = 5000.0000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 3, .materialType = MaterialType::MetalFuzzy1, }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 2, .materialType = MaterialType::Dielectric , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y =  0001.0000f, .z =  0000.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 2, .materialType = MaterialType::Dielectric , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .radius = 0000.4000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::AIR    ) / GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 2, .materialType = MaterialType::Dielectric , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x =  0000.0000f, .y =  0001.0000f, .z =  0000.0000f }, .radius = 0000.4000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::AIR    ) / GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 2, .materialType = MaterialType::Dielectric , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    

    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0000.0000f, .y = +0000.0000f, .z = +0004.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 2, .materialType = MaterialType::Metal      , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
//  bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0000.0000f, .y = +0000.0000f, .z = +0004.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 2, .materialType = MaterialType::Metal      , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -0004.0000f, .y = +0000.0000f, .z = +0000.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 2, .materialType = MaterialType::Metal      , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });
//  bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -0004.0000f, .y = +0000.0000f, .z = +0000.0000f }, .radius = 0000.5000f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 2, .materialType = MaterialType::Metal      , }, .movingDirection = { .x =  0000.0000f, .y =  0000.0000f, .z =  0000.0000f }, .geometryType = GeometryType::SPHERE, });


    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = -1.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = -3.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = -1.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = -3.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +1.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = -1.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +0.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +1.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = -1.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +3.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = +1.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +3.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = +1.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +4.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +5.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = +3.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +4.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = +5.0f, .y = -0.5f, .z = +2.0f }, .vertex2 = { .x = +3.0f, .y = -0.5f, .z = +2.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 6, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = +3.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = +1.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +2.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = +3.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = +1.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +0.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = +1.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -1.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = +0.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = +1.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -1.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = -2.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = -1.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -3.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = -2.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = -1.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -3.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 2, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = -4.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = -3.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
//  bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -2.0f, .y = +1.0f, .z = -4.0f }, .vertex1 = { .x = -2.0f, .y = -0.5f, .z = -3.0f }, .vertex2 = { .x = -2.0f, .y = -0.5f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 7, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
*/
    //B
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = -5.0f, .z = -5.0f }, .vertex1 = { .x = -5.0f, .y = -5.0f, .z = +5.0f }, .vertex2 = { .x = +5.0f, .y = -5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = -5.0f, .z = +5.0f }, .vertex1 = { .x = +5.0f, .y = -5.0f, .z = -5.0f }, .vertex2 = { .x = -5.0f, .y = -5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //T
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = +5.0f, .z = +5.0f }, .vertex1 = { .x = -5.0f, .y = +5.0f, .z = -5.0f }, .vertex2 = { .x = +5.0f, .y = +5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = +5.0f, .z = -5.0f }, .vertex1 = { .x = +5.0f, .y = +5.0f, .z = +5.0f }, .vertex2 = { .x = -5.0f, .y = +5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //L
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = +5.0f, .z = +5.0f }, .vertex1 = { .x = -5.0f, .y = -5.0f, .z = +5.0f }, .vertex2 = { .x = -5.0f, .y = -5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = -5.0f, .z = -5.0f }, .vertex1 = { .x = -5.0f, .y = +5.0f, .z = -5.0f }, .vertex2 = { .x = -5.0f, .y = +5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //R
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = +5.0f, .z = -5.0f }, .vertex1 = { .x = +5.0f, .y = -5.0f, .z = -5.0f }, .vertex2 = { .x = +5.0f, .y = -5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = -5.0f, .z = +5.0f }, .vertex1 = { .x = +5.0f, .y = +5.0f, .z = +5.0f }, .vertex2 = { .x = +5.0f, .y = +5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LambertianDiffuseReflectance2, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //B
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = +5.0f, .z = -5.0f }, .vertex1 = { .x = -5.0f, .y = -5.0f, .z = -5.0f }, .vertex2 = { .x = +5.0f, .y = -5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 21, .materialType = MaterialType::Metal                        , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = -5.0f, .z = -5.0f }, .vertex1 = { .x = +5.0f, .y = +5.0f, .z = -5.0f }, .vertex2 = { .x = -5.0f, .y = +5.0f, .z = -5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 21, .materialType = MaterialType::Metal                        , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //F
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = +5.0f, .y = +5.0f, .z = +5.0f }, .vertex1 = { .x = +5.0f, .y = -5.0f, .z = +5.0f }, .vertex2 = { .x = -5.0f, .y = -5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 21, .materialType = MaterialType::Metal                        , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });
    bvhTree.geometries.emplace_back(Geometry{ .primitive = { .vertex0 = { .x = -5.0f, .y = -5.0f, .z = +5.0f }, .vertex1 = { .x = -5.0f, .y = +5.0f, .z = +5.0f }, .vertex2 = { .x = +5.0f, .y = +5.0f, .z = +5.0f }, .frontFaceVertex0U = 0.0f, .frontFaceVertex0V = 0.0f, .frontFaceVertex1U = 1.0f, .frontFaceVertex1V = 0.0f, .frontFaceVertex2U = 0.5f, .frontFaceVertex2V = 1.0, .backFaceVertex0U = 0.0f, .backFaceVertex0V = 1.0f, .backFaceVertex1U = 1.0f, .backFaceVertex1V = 1.0f, .backFaceVertex2U = 0.5f, .backFaceVertex2V = 0.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 21, .materialType = MaterialType::Metal                        , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::PRIMITIVE, });

    //LU
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0.0f, .y = +5.0f, .z = +0.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +4.0f, .y = +5.0f, .z = +4.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 22, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +2.0f, .y = +5.0f, .z = +2.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 23, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -2.0f, .y = +5.0f, .z = -2.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 24, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -4.0f, .y = +5.0f, .z = -4.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 25, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });

    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0.0f, .y = +5.0f, .z = +0.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +4.0f, .y = +5.0f, .z = +4.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +2.0f, .y = +5.0f, .z = +2.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -2.0f, .y = +5.0f, .z = -2.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -4.0f, .y = +5.0f, .z = -4.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    
    //LD
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0.0f, .y = -5.0f, .z = +0.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 20, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +4.0f, .y = -5.0f, .z = -4.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 22, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +2.0f, .y = -5.0f, .z = -2.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 23, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -2.0f, .y = -5.0f, .z = +2.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 24, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -4.0f, .y = -5.0f, .z = +4.0f }, .radius = 0.5f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING), .textureIndex = 25, .materialType = MaterialType::LightDiffuse, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });

    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +0.0f, .y = -5.0f, .z = +0.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +4.0f, .y = -5.0f, .z = -4.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +2.0f, .y = -5.0f, .z = -2.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -2.0f, .y = -5.0f, .z = +2.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -4.0f, .y = -5.0f, .z = +4.0f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::GLASS  ), .textureIndex = 20, .materialType = MaterialType::Dielectric  , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });

    //S
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -3.0f, .y = +1.5f, .z = -1.5f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 20, .materialType = MaterialType::Dielectric, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -3.0f, .y = +1.5f, .z = -1.5f }, .radius = 0.8f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::AIR    ) / GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 20, .materialType = MaterialType::Dielectric, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -3.0f, .y = -1.5f, .z = +1.5f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 20, .materialType = MaterialType::Dielectric, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = -3.0f, .y = -1.5f, .z = +1.5f }, .radius = 0.8f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::AIR    ) / GetRefractionIndex(MaterialDielectric::GLASS), .textureIndex = 20, .materialType = MaterialType::Dielectric, }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +3.0f, .y = +1.5f, .z = -1.5f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 20, .materialType = MaterialType::Metal     , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });
    bvhTree.geometries.emplace_back(Geometry{ .sphere = { .center = { .x = +3.0f, .y = -1.5f, .z = +1.5f }, .radius = 1.0f, }, .material = { .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(MaterialDielectric::NOTHING)                                                , .textureIndex = 20, .materialType = MaterialType::Metal     , }, .movingDirection = { .x = +0.0f, .y = +0.0f, .z = +0.0f }, .geometryType = GeometryType::SPHERE, });



//  RotateAroundPivotAndAxis(bvhTree.geometries[8], { .x = 0.0f, .y = 0.0f, .z = 0.0f }, { .x = 0.0f, .y = 1.0f, .z = 0.0f }, lazy::DegToRad(-45.0f));
//  RotateAroundPivotAndAxis(bvhTree.geometries[8], { .x = 0.0f, .y = 0.0f, .z = 0.0f }, { .x = 0.0f, .y = 1.0f, .z = 0.0f }, lazy::DegToRad(-45.0f));
//  RotateAroundPivotAndAxis(bvhTree.geometries[8], { .x = 0.0f, .y = 0.0f, .z = 0.0f }, { .x = 0.0f, .y = 1.0f, .z = 0.0f }, lazy::DegToRad(+45.0f));
//  RotateAroundPivotAndAxis(bvhTree.geometries[8], { .x = 0.0f, .y = 0.0f, .z = 0.0f }, { .x = 0.0f, .y = 1.0f, .z = 0.0f }, lazy::DegToRad(+45.0f));



    for (Geometry& geo : bvhTree.geometries) { CalculateAABB3D(geo); if (geo.geometryType == GeometryType::PRIMITIVE) { geo.primitive.frontFaceNormal = Normalize(Cross(geo.primitive.vertex1 - geo.primitive.vertex0, geo.primitive.vertex2 - geo.primitive.vertex0)); } }
//  for (Geometry& geo : bvhTree.geometries) { CalculateAABB3D(geo); if (geo.geometryType == GeometryType::PRIMITIVE) { geo.primitive.frontFaceNormal = Normalize(Cross(geo.primitive.vertex1 - geo.primitive.vertex0, geo.primitive.vertex2 - geo.primitive.vertex0)); } }
    
//  for (Geometry& geo : bvhTree.geometries)
//  {
//      std::cout << "x min:" << geo.aabb3d.intervalAxisX.min << " " << "x max:" << geo.aabb3d.intervalAxisX.max << std::endl;
//      std::cout << "y min:" << geo.aabb3d.intervalAxisY.min << " " << "y max:" << geo.aabb3d.intervalAxisY.max << std::endl;
//      std::cout << "z min:" << geo.aabb3d.intervalAxisZ.min << " " << "z max:" << geo.aabb3d.intervalAxisZ.max << std::endl << std::endl << std::endl;
//  }
//  return 0;

    bvhTree.bvhNodes.reserve(2 * bvhTree.geometries.size() - 1);
//  bvhTree.bvhNodes.reserve(2 * bvhTree.geometries.size() - 1);
    BuildBVHTree(bvhTree, 0,(int)bvhTree.geometries.size());
//  BuildBVHTree(bvhTree, 0,(int)bvhTree.geometries.size());

    //for (const BVHNode& bvhNode : bvhTree.bvhNodes)
    //{
    //    std::cout << "x min:" << bvhNode.aabb3d.intervalAxisX.min << " " << "x max:" << bvhNode.aabb3d.intervalAxisX.max << std::endl;
    //    std::cout << "y min:" << bvhNode.aabb3d.intervalAxisY.min << " " << "y max:" << bvhNode.aabb3d.intervalAxisY.max << std::endl;
    //    std::cout << "z min:" << bvhNode.aabb3d.intervalAxisZ.min << " " << "z max:" << bvhNode.aabb3d.intervalAxisZ.max << std::endl;
    //    std::cout << "shape   index:" << bvhNode.shapeIndex  << "  "
    //              << "child L index:" << bvhNode.childIndexL << "  "
    //              << "child R index:" << bvhNode.childIndexR << std::endl << std::endl << std::endl;
    //}
    //return 0;

    float aspectRatio = 16.0f / 9.0f;
//  float aspectRatio = 16.0f / 9.0f;
    int imgW = 2000;
//  int imgW = 2000;
    int imgH = int(imgW / aspectRatio);
//  int imgH = int(imgW / aspectRatio);
    imgH = std::max(imgH, 1);
//  imgH = std::max(imgH, 1);

    constexpr Point3 lookFrom { .x = +0.0f, .y = +0.0f, .z = +4.9f };
    constexpr Point3 lookAt   { .x = +0.0f, .y = +0.0f, .z = +0.0f };
    constexpr Point3 viewUp   { .x = +0.0f, .y = +1.0f, .z = +0.0f };

    Vec3 cameraU; // x
    Vec3 cameraV; // y
    Vec3 cameraW; // z

    float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = (lookAt - lookFrom).Length();
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
//  float defocusAngle = 0.00f * std::numbers::pi_v<float>; float focusDistance = 10.0f;
    Vec3 defocusDiskRadiusU;
    Vec3 defocusDiskRadiusV;


    float vFOV = std::numbers::pi_v<float> / 2.0f;
    float hFOV = std::numbers::pi_v<float> / 2.0f;
    float h = std::tanf(vFOV / 2.0f);
    float w = std::tanf(hFOV / 2.0f);

    float focalLength = (lookAt - lookFrom).Length();
//  float focalLength = (lookAt - lookFrom).Length();


    cameraW = Normalize(lookFrom - lookAt); cameraU = Normalize(Cross(viewUp, cameraW)); cameraV = Cross(cameraW, cameraU);
//  cameraW = Normalize(lookFrom - lookAt); cameraU = Normalize(Cross(viewUp, cameraW)); cameraV = Cross(cameraW, cameraU);

    float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
//  float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
    defocusDiskRadiusU = cameraU * defocusRadius;
    defocusDiskRadiusV = cameraV * defocusRadius;


    float viewportH = 2.0f * h * /* focalLength */ focusDistance;
//  float viewportH = 2.0f * h * /* focalLength */ focusDistance;
    float viewportW = viewportH * (float(imgW) / imgH);
//  float viewportW = viewportH * (float(imgW) / imgH);

    Point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;
//  Point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;



//  vec3 viewportU { };
//  viewportU.x = +viewportW;
//  viewportU.y = 0.0f;
//  viewportU.z = 0.0f;
//  vec3 viewportV { };
//  viewportV.x = 0.0f;
//  viewportV.y = -viewportH;
//  viewportV.z = 0.0f;

    Vec3 viewportU = viewportW *  cameraU;
    Vec3 viewportV = viewportH * -cameraV;


    Vec3 fromPixelToPixelDeltaU = viewportU / float(imgW);
    Vec3 fromPixelToPixelDeltaV = viewportV / float(imgH);



    Point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
//  Point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
    Point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;
//  Point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;


    std::ofstream PPMFile(GetCurrentDateTime());
//  std::ofstream PPMFile(GetCurrentDateTime());
    PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";
//  PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";

    constexpr int numberOfChannels = 3; // R G B
//  constexpr int numberOfChannels = 3; // R G B
    std::vector<float> rgbs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> rgbs(imgW * imgH * numberOfChannels, 1.0f);

//  std::vector<std::thread> threads;
//  std::vector<std::thread> threads;

    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
#ifdef _DEBUG
    std::clog << "Progress: " << pixelY << "\n";
//  std::clog << "Progress: " << pixelY << "\n";
#endif
    threadPool->Enqueue(
//  threadPool->Enqueue(
//  threads.emplace_back(
//  threads.emplace_back(
    [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, /* &geometries */ &bvhTree, &rgbs
//  [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, /* &geometries */ &bvhTree, &rgbs
    ]
    {

    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {


//      Point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
//      Point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
//      Vec3 rayDirection = pixelCenter - cameraCenter;
//      Vec3 rayDirection = pixelCenter - cameraCenter;
//      Ray  ray  { cameraCenter, rayDirection };
//      Ray  ray  { cameraCenter, rayDirection };

//      float r = float(pixelX) / float(imgW - 1);
//      float g = float(pixelY) / float(imgH - 1);
//      float b = 0.00f;
//      Color3 pixelColor { r, g, b, };
//      Color3 pixelColor { r, g, b, };

//      Color3 pixelColor = RayColor(ray);
//      Color3 pixelColor = RayColor(ray);
        
//      Color3 pixelColor = RayColor(ray, geometries);
//      Color3 pixelColor = RayColor(ray, geometries);

        Color3 pixelColor{};
//      Color3 pixelColor{};
        for (int sample = 0; sample < samplesPerPixel; ++sample)
        {
            Vec3 sampleOffset{ Random() - 0.5f, Random() - 0.5f, 0.0f };
//          Vec3 sampleOffset{ Random() - 0.5f, Random() - 0.5f, 0.0f };
            Point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          Point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          Vec3 rayDirection = pixelSampleCenter - cameraCenter;
//          Vec3 rayDirection = pixelSampleCenter - cameraCenter;
            Vec3 rayOrigin = cameraCenter;
//          Vec3 rayOrigin = cameraCenter;
            if (defocusAngle > 0.0f) _UNLIKELY { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
//          if (defocusAngle > 0.0f) _UNLIKELY { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
            Vec3 rayDirection = pixelSampleCenter - rayOrigin;
//          Vec3 rayDirection = pixelSampleCenter - rayOrigin;
            Ray  ray{ .ori = rayOrigin, .dir = rayDirection, .time = Random() };
//          Ray  ray{ .ori = rayOrigin, .dir = rayDirection, .time = Random() };
            pixelColor += RayColor(ray, bvhTree, 1000, BackgroundType::DARK_ROOM_SPACE);
//          pixelColor += RayColor(ray, bvhTree, 1000, BackgroundType::DARK_ROOM_SPACE);
//          pixelColor += RayColor(ray, geometries);
//          pixelColor += RayColor(ray, geometries);
        }
        pixelColor *= pixelSamplesScale;
//      pixelColor *= pixelSamplesScale;
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = pixelColor.x;
        rgbs[index + 1] = pixelColor.y;
        rgbs[index + 2] = pixelColor.z;
    }
    });
    }

    threadPool->WaitForAllTasksToBeDone();
//  threadPool->WaitForAllTasksToBeDone();

//  for (std::thread& t : threads) { t.join(); }
//  for (std::thread& t : threads) { t.join(); }


//  DENOISE 001
//  DENOISE 001
/*
    std::vector<float> gaussianBlurRGBs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> gaussianBlurRGBs(imgW * imgH * numberOfChannels, 1.0f);
    for (int pixelY = 1; pixelY < imgH - 1; ++pixelY)
    {
    for (int pixelX = 1; pixelX < imgW - 1; ++pixelX)
    {
        size_t indexLT = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t indexJT = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX + 0)) * numberOfChannels;
        size_t indexRT = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;

        size_t indexLC = (static_cast<size_t>(pixelY + 0) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t indexCC = (static_cast<size_t>(pixelY + 0) * imgW + static_cast<size_t>(pixelX + 0)) * numberOfChannels;
        size_t indexRC = (static_cast<size_t>(pixelY + 0) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;

        size_t indexLB = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t indexJB = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX + 0)) * numberOfChannels;
        size_t indexRB = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;


        gaussianBlurRGBs[indexCC + 0] = rgbs[indexLT + 0] * 0.075f + rgbs[indexJT + 0] * 0.124f + rgbs[indexRT + 0] * 0.075f + rgbs[indexLC + 0] * 0.124f + rgbs[indexCC + 0] * 0.204f + rgbs[indexRC + 0] * 0.124f + rgbs[indexLB + 0] * 0.075f + rgbs[indexJB + 0] * 0.124f + rgbs[indexRB + 0] * 0.075f;
        gaussianBlurRGBs[indexCC + 1] = rgbs[indexLT + 1] * 0.075f + rgbs[indexJT + 1] * 0.124f + rgbs[indexRT + 1] * 0.075f + rgbs[indexLC + 1] * 0.124f + rgbs[indexCC + 1] * 0.204f + rgbs[indexRC + 1] * 0.124f + rgbs[indexLB + 1] * 0.075f + rgbs[indexJB + 1] * 0.124f + rgbs[indexRB + 1] * 0.075f;
        gaussianBlurRGBs[indexCC + 2] = rgbs[indexLT + 2] * 0.075f + rgbs[indexJT + 2] * 0.124f + rgbs[indexRT + 2] * 0.075f + rgbs[indexLC + 2] * 0.124f + rgbs[indexCC + 2] * 0.204f + rgbs[indexRC + 2] * 0.124f + rgbs[indexLB + 2] * 0.075f + rgbs[indexJB + 2] * 0.124f + rgbs[indexRB + 2] * 0.075f;
    }
    }
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = gaussianBlurRGBs[index + 0];
        rgbs[index + 1] = gaussianBlurRGBs[index + 1];
        rgbs[index + 2] = gaussianBlurRGBs[index + 2];
    }
    }
*/


//  DENOISE 002
//  DENOISE 002
/*
    std::vector<float> gaussianBlurRGBs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> gaussianBlurRGBs(imgW * imgH * numberOfChannels, 1.0f);
    for (int pixelY = 2; pixelY < imgH - 2; ++pixelY)
    {
    for (int pixelX = 2; pixelX < imgW - 2; ++pixelX)
    {
        size_t index00 = (static_cast<size_t>(pixelY - 2) * imgW + static_cast<size_t>(pixelX - 2)) * numberOfChannels;
        size_t index01 = (static_cast<size_t>(pixelY - 2) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t index02 = (static_cast<size_t>(pixelY - 2) * imgW + static_cast<size_t>(pixelX    )) * numberOfChannels;
        size_t index03 = (static_cast<size_t>(pixelY - 2) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;
        size_t index04 = (static_cast<size_t>(pixelY - 2) * imgW + static_cast<size_t>(pixelX + 2)) * numberOfChannels;

        size_t index10 = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX - 2)) * numberOfChannels;
        size_t index11 = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t index12 = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX    )) * numberOfChannels;
        size_t index13 = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;
        size_t index14 = (static_cast<size_t>(pixelY - 1) * imgW + static_cast<size_t>(pixelX + 2)) * numberOfChannels;

        size_t index20 = (static_cast<size_t>(pixelY    ) * imgW + static_cast<size_t>(pixelX - 2)) * numberOfChannels;
        size_t index21 = (static_cast<size_t>(pixelY    ) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t index22 = (static_cast<size_t>(pixelY    ) * imgW + static_cast<size_t>(pixelX    )) * numberOfChannels;
        size_t index23 = (static_cast<size_t>(pixelY    ) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;
        size_t index24 = (static_cast<size_t>(pixelY    ) * imgW + static_cast<size_t>(pixelX + 2)) * numberOfChannels;

        size_t index30 = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX - 2)) * numberOfChannels;
        size_t index31 = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t index32 = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX    )) * numberOfChannels;
        size_t index33 = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;
        size_t index34 = (static_cast<size_t>(pixelY + 1) * imgW + static_cast<size_t>(pixelX + 2)) * numberOfChannels;

        size_t index40 = (static_cast<size_t>(pixelY + 2) * imgW + static_cast<size_t>(pixelX - 2)) * numberOfChannels;
        size_t index41 = (static_cast<size_t>(pixelY + 2) * imgW + static_cast<size_t>(pixelX - 1)) * numberOfChannels;
        size_t index42 = (static_cast<size_t>(pixelY + 2) * imgW + static_cast<size_t>(pixelX    )) * numberOfChannels;
        size_t index43 = (static_cast<size_t>(pixelY + 2) * imgW + static_cast<size_t>(pixelX + 1)) * numberOfChannels;
        size_t index44 = (static_cast<size_t>(pixelY + 2) * imgW + static_cast<size_t>(pixelX + 2)) * numberOfChannels;

        gaussianBlurRGBs[index22 + 0] = rgbs[index00 + 0] * lazy::G_5X5_S3_0_0 + rgbs[index01 + 0] * lazy::G_5X5_S3_1_0 + rgbs[index02 + 0] * lazy::G_5X5_S3_0_2 + rgbs[index03 + 0] * lazy::G_5X5_S3_0_3 + rgbs[index04 + 0] * lazy::G_5X5_S3_0_4
                                      + rgbs[index10 + 0] * lazy::G_5X5_S3_1_0 + rgbs[index11 + 0] * lazy::G_5X5_S3_1_1 + rgbs[index12 + 0] * lazy::G_5X5_S3_1_2 + rgbs[index13 + 0] * lazy::G_5X5_S3_1_3 + rgbs[index14 + 0] * lazy::G_5X5_S3_1_4
                                      + rgbs[index20 + 0] * lazy::G_5X5_S3_2_0 + rgbs[index21 + 0] * lazy::G_5X5_S3_1_2 + rgbs[index22 + 0] * lazy::G_5X5_S3_2_2 + rgbs[index23 + 0] * lazy::G_5X5_S3_2_3 + rgbs[index24 + 0] * lazy::G_5X5_S3_2_4
                                      + rgbs[index30 + 0] * lazy::G_5X5_S3_3_0 + rgbs[index31 + 0] * lazy::G_5X5_S3_1_3 + rgbs[index32 + 0] * lazy::G_5X5_S3_3_2 + rgbs[index33 + 0] * lazy::G_5X5_S3_3_3 + rgbs[index34 + 0] * lazy::G_5X5_S3_3_4
                                      + rgbs[index40 + 0] * lazy::G_5X5_S3_4_0 + rgbs[index41 + 0] * lazy::G_5X5_S3_1_4 + rgbs[index42 + 0] * lazy::G_5X5_S3_4_2 + rgbs[index43 + 0] * lazy::G_5X5_S3_4_3 + rgbs[index44 + 0] * lazy::G_5X5_S3_4_4
                                      ;
        gaussianBlurRGBs[index22 + 1] = rgbs[index00 + 1] * lazy::G_5X5_S3_0_0 + rgbs[index01 + 1] * lazy::G_5X5_S3_1_0 + rgbs[index02 + 1] * lazy::G_5X5_S3_0_2 + rgbs[index03 + 1] * lazy::G_5X5_S3_0_3 + rgbs[index04 + 1] * lazy::G_5X5_S3_0_4
                                      + rgbs[index10 + 1] * lazy::G_5X5_S3_1_0 + rgbs[index11 + 1] * lazy::G_5X5_S3_1_1 + rgbs[index12 + 1] * lazy::G_5X5_S3_1_2 + rgbs[index13 + 1] * lazy::G_5X5_S3_1_3 + rgbs[index14 + 1] * lazy::G_5X5_S3_1_4
                                      + rgbs[index20 + 1] * lazy::G_5X5_S3_2_0 + rgbs[index21 + 1] * lazy::G_5X5_S3_1_2 + rgbs[index22 + 1] * lazy::G_5X5_S3_2_2 + rgbs[index23 + 1] * lazy::G_5X5_S3_2_3 + rgbs[index24 + 1] * lazy::G_5X5_S3_2_4
                                      + rgbs[index30 + 1] * lazy::G_5X5_S3_3_0 + rgbs[index31 + 1] * lazy::G_5X5_S3_1_3 + rgbs[index32 + 1] * lazy::G_5X5_S3_3_2 + rgbs[index33 + 1] * lazy::G_5X5_S3_3_3 + rgbs[index34 + 1] * lazy::G_5X5_S3_3_4
                                      + rgbs[index40 + 1] * lazy::G_5X5_S3_4_0 + rgbs[index41 + 1] * lazy::G_5X5_S3_1_4 + rgbs[index42 + 1] * lazy::G_5X5_S3_4_2 + rgbs[index43 + 1] * lazy::G_5X5_S3_4_3 + rgbs[index44 + 1] * lazy::G_5X5_S3_4_4
                                      ;
        gaussianBlurRGBs[index22 + 2] = rgbs[index00 + 2] * lazy::G_5X5_S3_0_0 + rgbs[index01 + 2] * lazy::G_5X5_S3_1_0 + rgbs[index02 + 2] * lazy::G_5X5_S3_0_2 + rgbs[index03 + 2] * lazy::G_5X5_S3_0_3 + rgbs[index04 + 2] * lazy::G_5X5_S3_0_4
                                      + rgbs[index10 + 2] * lazy::G_5X5_S3_1_0 + rgbs[index11 + 2] * lazy::G_5X5_S3_1_1 + rgbs[index12 + 2] * lazy::G_5X5_S3_1_2 + rgbs[index13 + 2] * lazy::G_5X5_S3_1_3 + rgbs[index14 + 2] * lazy::G_5X5_S3_1_4
                                      + rgbs[index20 + 2] * lazy::G_5X5_S3_2_0 + rgbs[index21 + 2] * lazy::G_5X5_S3_1_2 + rgbs[index22 + 2] * lazy::G_5X5_S3_2_2 + rgbs[index23 + 2] * lazy::G_5X5_S3_2_3 + rgbs[index24 + 2] * lazy::G_5X5_S3_2_4
                                      + rgbs[index30 + 2] * lazy::G_5X5_S3_3_0 + rgbs[index31 + 2] * lazy::G_5X5_S3_1_3 + rgbs[index32 + 2] * lazy::G_5X5_S3_3_2 + rgbs[index33 + 2] * lazy::G_5X5_S3_3_3 + rgbs[index34 + 2] * lazy::G_5X5_S3_3_4
                                      + rgbs[index40 + 2] * lazy::G_5X5_S3_4_0 + rgbs[index41 + 2] * lazy::G_5X5_S3_1_4 + rgbs[index42 + 2] * lazy::G_5X5_S3_4_2 + rgbs[index43 + 2] * lazy::G_5X5_S3_4_3 + rgbs[index44 + 2] * lazy::G_5X5_S3_4_4
                                      ;
    }
    }
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = gaussianBlurRGBs[index + 0];
        rgbs[index + 1] = gaussianBlurRGBs[index + 1];
        rgbs[index + 2] = gaussianBlurRGBs[index + 2];
    }
    }
*/


// CHROMATIC ABERRATION
// CHROMATIC ABERRATION
/*
    std::vector<float> chromaticAberrationRGBs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> chromaticAberrationRGBs(imgW * imgH * numberOfChannels, 1.0f);
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        chromaticAberrationRGBs[index + 0] = Sample2LinearInterpolation(rgbs, imgW, imgH, pixelX + 2.0f, pixelY + 2.0f, 0, 3);
//      chromaticAberrationRGBs[index + 0] = Sample2LinearInterpolation(rgbs, imgW, imgH, pixelX + 2.0f, pixelY + 2.0f, 0, 3);
//      chromaticAberrationRGBs[index + 0] = rgbs[index + 0];
        chromaticAberrationRGBs[index + 1] = rgbs[index + 1];
        chromaticAberrationRGBs[index + 2] = rgbs[index + 2];
    }
    }
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = chromaticAberrationRGBs[index + 0];
        rgbs[index + 1] = chromaticAberrationRGBs[index + 1];
        rgbs[index + 2] = chromaticAberrationRGBs[index + 2];
    }
    }
*/


//  BAYER MATRIX DITHERING
//  BAYER MATRIX DITHERING
/*
    std::vector<float> bayerMatrixDitheringRGBs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> bayerMatrixDitheringRGBs(imgW * imgH * numberOfChannels, 1.0f);
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        float intensity = 0.30f * rgbs[index + 0] + 0.59f * rgbs[index + 1] + 0.11f * rgbs[index + 2];
//      float intensity = 0.30f * rgbs[index + 0] + 0.59f * rgbs[index + 1] + 0.11f * rgbs[index + 2];
        float threshold = lazy::GetValueFromBayer16x16(pixelX, pixelY);
//      float threshold = lazy::GetValueFromBayer16x16(pixelX, pixelY);
        float ditheringOutput;
//      float ditheringOutput;
        if (intensity >= threshold)
//      if (intensity >= threshold)
        {
            ditheringOutput = 1.0f;
//          ditheringOutput = 1.0f;
        }
        else
        {
            ditheringOutput = 0.0f;
//          ditheringOutput = 0.0f;
        }
        bayerMatrixDitheringRGBs[index + 0] = rgbs[index + 0] * ditheringOutput;
        bayerMatrixDitheringRGBs[index + 1] = rgbs[index + 1] * ditheringOutput;
        bayerMatrixDitheringRGBs[index + 2] = rgbs[index + 2] * ditheringOutput;
    }
    }
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = bayerMatrixDitheringRGBs[index + 0];
        rgbs[index + 1] = bayerMatrixDitheringRGBs[index + 1];
        rgbs[index + 2] = bayerMatrixDitheringRGBs[index + 2];
    }
    }
*/


//  BILATERAL FILTERING
//  BILATERAL FILTERING
/*
    constexpr int kernelRadius = 5; // odds
//  constexpr int kernelRadius = 5; // odds
    constexpr float sigmaSpatial = 2.5f;
    constexpr float sigmaRanging = 0.1f;
    constexpr float twoSigmaSpatialSquared = 2.0f * sigmaSpatial * sigmaSpatial;
    constexpr float twoSigmaRangingSquared = 2.0f * sigmaRanging * sigmaRanging;
    std::vector<float> bilateralFilteringRGBs(imgW * imgH * numberOfChannels, 1.0f);
//  std::vector<float> bilateralFilteringRGBs(imgW * imgH * numberOfChannels, 1.0f);
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        float totalWeight  = 0.0f;
//      float totalWeight  = 0.0f;
        float weightedSumR = 0.0f;
        float weightedSumG = 0.0f;
        float weightedSumB = 0.0f;

        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        float centerR = rgbs[index + 0];
        float centerG = rgbs[index + 1];
        float centerB = rgbs[index + 2];

        for (int deltaY = -kernelRadius; deltaY <= +kernelRadius; ++deltaY)
        {
        for (int deltaX = -kernelRadius; deltaX <= +kernelRadius; ++deltaX)
        {
            int neighborPixelX = pixelX + deltaX;
            int neighborPixelY = pixelY + deltaY;

            if (neighborPixelX >= 0
            &&  neighborPixelX < imgW
            &&  neighborPixelY >= 0
            &&  neighborPixelY < imgH)
            {
                size_t neighborIndex = (static_cast<size_t>(neighborPixelY) * imgW + neighborPixelX) * numberOfChannels;
//              size_t neighborIndex = (static_cast<size_t>(neighborPixelY) * imgW + neighborPixelX) * numberOfChannels;
                float neighborR = rgbs[neighborIndex + 0];
                float neighborG = rgbs[neighborIndex + 1];
                float neighborB = rgbs[neighborIndex + 2];

                float distanceSquaredSpatial = static_cast<float>(deltaX * deltaX + deltaY * deltaY);
//              float distanceSquaredSpatial = static_cast<float>(deltaX * deltaX + deltaY * deltaY);
                float weightedSpatial = std::exp(-distanceSquaredSpatial / twoSigmaSpatialSquared);
//              float weightedSpatial = std::exp(-distanceSquaredSpatial / twoSigmaSpatialSquared);

                float diffR = neighborR - centerR;
                float diffG = neighborG - centerG;
                float diffB = neighborB - centerB;
                float distanceSquaredRanging = diffR * diffR + diffG * diffG + diffB * diffB;
//              float distanceSquaredRanging = diffR * diffR + diffG * diffG + diffB * diffB;
                float weightedRanging = std::exp(-distanceSquaredRanging / twoSigmaRangingSquared);
//              float weightedRanging = std::exp(-distanceSquaredRanging / twoSigmaRangingSquared);

                float weightedCombined = weightedSpatial * weightedRanging;
//              float weightedCombined = weightedSpatial * weightedRanging;

                totalWeight  += weightedCombined;
//              totalWeight  += weightedCombined;
                weightedSumR += weightedCombined * neighborR;
                weightedSumG += weightedCombined * neighborG;
                weightedSumB += weightedCombined * neighborB;
            }
        }
        }
        float filteredR = centerR;
        float filteredG = centerG;
        float filteredB = centerB;
        if (totalWeight > 0.0f)
//      if (totalWeight > 0.0f)
        {
            filteredR = weightedSumR / totalWeight;
            filteredG = weightedSumG / totalWeight;
            filteredB = weightedSumB / totalWeight;
        }
        bilateralFilteringRGBs[index + 0] = filteredR;
        bilateralFilteringRGBs[index + 1] = filteredG;
        bilateralFilteringRGBs[index + 2] = filteredB;
    }
    }
    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = bilateralFilteringRGBs[index + 0];
        rgbs[index + 1] = bilateralFilteringRGBs[index + 1];
        rgbs[index + 2] = bilateralFilteringRGBs[index + 2];
    }
    }
*/


    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        thread_local static const Interval intensity { 0.000f , 0.999f };
//      thread_local static const Interval intensity { 0.000f , 0.999f };
        int ir = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(rgbs[index + 0])));
        int ig = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(rgbs[index + 1])));
        int ib = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(rgbs[index + 2])));
        PPMFile << std::setw(3) << ir << " ";
        PPMFile << std::setw(3) << ig << " ";
        PPMFile << std::setw(3) << ib << " ";
    }
        PPMFile << "\n";
//      PPMFile << "\n";
    }

    PPMFile.close();
//  PPMFile.close();

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

    return 0;
//  return 0;
}


// defocus blur = depth of field
// defocus blur = depth of field


// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO
// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO
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
