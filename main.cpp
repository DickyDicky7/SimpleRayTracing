#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define _USE_MATH_DEFINES

#include <string_view>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <array>
#include <iomanip>
#include <cstdint>
#include "ThreadPool.h"
#include <chrono>
#include <random>
#include <vector>
#include <span>
#include "ImagePNG.h"
#include "ImageJPG.h"
#include "ImageSVG.h"

static inline int Sample1LinearInterpolation(const std::vector<int>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    return 000;
}
static inline int Sample2LinearInterpolation(const std::vector<int>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    int pixelX = static_cast<int>(std::floor(x));
    int pixelY = static_cast<int>(std::floor(y));
    
    float deltaX = x - pixelX;
    float deltaY = y - pixelY;

    int currPixelX = std::clamp(pixelX + 0, 0, imgW - 1);
    int nextPixelX = std::clamp(pixelX + 1, 0, imgW - 1);
    int currPixelY = std::clamp(pixelY + 0, 0, imgH - 1);
    int nextPixelY = std::clamp(pixelY + 1, 0, imgH - 1);

    size_t indexOfTLPixel = (static_cast<size_t>(currPixelY) * imgW + currPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfTRPixel = (static_cast<size_t>(currPixelY) * imgW + nextPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfBLPixel = (static_cast<size_t>(nextPixelY) * imgW + currPixelX) * numberOfColorChannels + colorChannel;
    size_t indexOfBRPixel = (static_cast<size_t>(nextPixelY) * imgW + nextPixelX) * numberOfColorChannels + colorChannel;

    float valueAtColorChannelOfTLPixel = static_cast<float>(rgbs[indexOfTLPixel]);
    float valueAtColorChannelOfTRPixel = static_cast<float>(rgbs[indexOfTRPixel]);
    float valueAtColorChannelOfBLPixel = static_cast<float>(rgbs[indexOfBLPixel]);
    float valueAtColorChannelOfBRPixel = static_cast<float>(rgbs[indexOfBRPixel]);

    float valueLerpTop = (1.0f - deltaX) * valueAtColorChannelOfTLPixel + deltaX * valueAtColorChannelOfTRPixel;
    float valueLerpBot = (1.0f - deltaX) * valueAtColorChannelOfBLPixel + deltaX * valueAtColorChannelOfBRPixel;
    float valueLerpVer = (1.0f - deltaY) * valueLerpBot
                       +         deltaY  * valueLerpTop;

    return static_cast<int>(std::round(valueLerpVer));
//  return static_cast<int>(std::round(valueLerpVer));

    return 000;
}
static inline int Sample3LinearInterpolation(const std::vector<int>& rgbs, int imgW, int imgH, float x, float y, std::uint8_t colorChannel, std::uint8_t numberOfColorChannels)
{
    return 000;
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

struct interval
{
    float min = positiveInfinity;
    float max = negativeInfinity;

    static const interval empty;
//  static const interval empty;
    static const interval universe;
//  static const interval universe;

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

    interval Expand(float delta) const { float padding = delta / 2.0f; return interval{ .min = min - padding, .max = max + padding }; }
//  interval Expand(float delta) const { float padding = delta / 2.0f; return interval{ .min = min - padding, .max = max + padding }; }
};

    const interval interval::empty    { positiveInfinity, negativeInfinity };
//  const interval interval::empty    { positiveInfinity, negativeInfinity };
    const interval interval::universe { negativeInfinity, positiveInfinity };
//  const interval interval::universe { negativeInfinity, positiveInfinity };


struct AABB2D
{
    interval intervalAxisX;
    interval intervalAxisY;
//  interval intervalAxisZ;
};
struct AABB3D
{
    interval intervalAxisX;
    interval intervalAxisY;
    interval intervalAxisZ;
};


    inline static std::string GetCurrentDateTime()
//  inline static std::string GetCurrentDateTime()
{
        std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.ppm", &tm); return std::string(buffer);
//      std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.ppm", &tm); return std::string(buffer);
}


struct vec3
{
    float x;
    float y;
    float z;

    vec3  operator- (             ) const { return vec3 { -x, -y, -z }; }
    vec3& operator+=(const vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    vec3& operator-=(const vec3& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    vec3& operator*=(const vec3& v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    vec3& operator/=(const vec3& v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }
    vec3& operator*=(const float& v)
    {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }
    vec3& operator/=(const float& v)
    {
        x /= v;
        y /= v;
        z /= v;
        return *this;
    }

    float length        () const { return std::sqrt(length_squared()); }
    float length_squared() const { return x * x
                                        + y * y
                                        + z * z                      ; }

    bool NearZero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        // Return true if the vector is close to zero in all dimensions.
        float s = 1e-8;
//      float s = 1e-8;
        return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
//      return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
    }
};


using point3 = vec3;
using color3 = vec3;

static inline vec3 operator+(const vec3& u, const vec3& v) { return vec3 { u.x + v.x, u.y + v.y, u.z + v.z }; }
static inline vec3 operator-(const vec3& u, const vec3& v) { return vec3 { u.x - v.x, u.y - v.y, u.z - v.z }; }
static inline vec3 operator*(const vec3& u, const vec3& v) { return vec3 { u.x * v.x, u.y * v.y, u.z * v.z }; }
static inline vec3 operator/(const vec3& u, const vec3& v) { return vec3 { u.x / v.x, u.y / v.y, u.z / v.z }; }

static inline vec3 operator*(const vec3& u, float t) { return vec3 { u.x * t, u.y * t, u.z * t }; }
static inline vec3 operator/(const vec3& u, float t) { return vec3 { u.x / t, u.y / t, u.z / t }; }

static inline vec3 operator*(float t, const vec3& u) { return vec3 { u.x * t, u.y * t, u.z * t }; }
static inline vec3 operator/(float t, const vec3& u) { return vec3 { u.x / t, u.y / t, u.z / t }; }


static
inline float dot  (const vec3& u, const vec3& v)
{
return u.x * v.x
     + u.y * v.y
     + u.z * v.z;
}

static
inline vec3  cross(const vec3& u, const vec3& v)
{
return vec3 { u.y * v.z - u.z * v.y,
              u.z * v.x - u.x * v.z,
              u.x * v.y - u.y * v.x,
            };
}

  inline static vec3 normalize(const vec3& v) { return v / v.length(); }
//inline static vec3 normalize(const vec3& v) { return v / v.length(); }





  struct noisePerlinOrigin
  {
      std::array<float, 256> randomFloats ;
      std::array<int  , 256> permutationsX;
      std::array<int  , 256> permutationsY;
      std::array<int  , 256> permutationsZ;
  };

  struct noisePerlinSmooth
  {
  };

  inline static void Generate(noisePerlinOrigin& npo)
  {
      for (float& randomFloat :
              npo.randomFloats)
      {
                  randomFloat = Random();
      }
      for (int i = 000; i < 256; ++i)
      {
          npo.permutationsX[i] = i;
          npo.permutationsY[i] = i;
          npo.permutationsZ[i] = i;
      }
      for (int i = 255; i > 000; --i)
      {
          int targetX = RandomInt(0, i);
          int targetY = RandomInt(0, i);
          int targetZ = RandomInt(0, i);
          std::swap(npo.permutationsX[i], npo.permutationsX[targetX]);
          std::swap(npo.permutationsY[i], npo.permutationsY[targetY]);
          std::swap(npo.permutationsZ[i], npo.permutationsZ[targetZ]);
      }
  }

  inline static void Generate(noisePerlinSmooth& nps)
  {
  }

  inline static float GetNoiseValue(const noisePerlinOrigin& npo, const point3& p)
  {
      int i = static_cast<int>(4 * p.x) & 255;
      int j = static_cast<int>(4 * p.y) & 255;
      int k = static_cast<int>(4 * p.z) & 255;
      return npo.randomFloats[npo.permutationsX[i] ^
                              npo.permutationsY[j] ^
                              npo.permutationsZ[k]];
  }


  inline static float GetNoiseValue(const noisePerlinSmooth& nps, const point3& p)
  {
      return 0.0f;
  }


    enum class textureType : std::int8_t
//  enum class textureType : std::int8_t
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
    NOISE_PERLIN_ORIGIN = 6,
//  NOISE_PERLIN_ORIGIN = 6,
    NOISE_PERLIN_SMOOTH = 7,
//  NOISE_PERLIN_SMOOTH = 7,
};


    struct texture
//  struct texture
{
    color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
//  color3 albedo = { .x = 0.0f, .y = 0.0f, .z = 0.0f };
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
    textureType type = textureType::SOLID_COLOR;
//  textureType type = textureType::SOLID_COLOR;
};


    static inline struct texturesDatabase { std::vector<texture> textures; } texturesDatabase;
//  static inline struct texturesDatabase { std::vector<texture> textures; } texturesDatabase;

    static inline struct imagesDatabase { std::vector<ImagePNG> pngs; std::vector<ImageJPG> jpgs; std::vector<ImageSVG> svgs; } imagesDatabase;
//  static inline struct imagesDatabase { std::vector<ImagePNG> pngs; std::vector<ImageJPG> jpgs; std::vector<ImageSVG> svgs; } imagesDatabase;

    static inline struct noisesDatabase { std::vector<noisePerlinOrigin> noisePerlinOrigins; std::vector<noisePerlinSmooth> noisePerlinSmooths; } noisesDatabase;
//  static inline struct noisesDatabase { std::vector<noisePerlinOrigin> noisePerlinOrigins; std::vector<noisePerlinSmooth> noisePerlinSmooths; } noisesDatabase;


    static inline color3 Value(int textureIndex, float uTextureCoordinate, float vTextureCoordinate, const point3& point)
//  static inline color3 Value(int textureIndex, float uTextureCoordinate, float vTextureCoordinate, const point3& point)
{
    const texture& texture = texturesDatabase.textures[textureIndex];
//  const texture& texture = texturesDatabase.textures[textureIndex];
    switch (texture.type)
//  switch (texture.type)
    {
        case textureType::SOLID_COLOR:
//      case textureType::SOLID_COLOR:
        {
            return texture.albedo;
//          return texture.albedo;
        }
        break;
//      break;


        case textureType::CHECKER_TEXTURE_1:
//      case textureType::CHECKER_TEXTURE_1:
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


        case textureType::CHECKER_TEXTURE_2:
//      case textureType::CHECKER_TEXTURE_2:
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


        case textureType::IMAGE_TEXTURE_PNG:
//      case textureType::IMAGE_TEXTURE_PNG:
        {
            const ImagePNG& imagePNG = imagesDatabase.pngs[texture.imageIndex];
//          const ImagePNG& imagePNG = imagesDatabase.pngs[texture.imageIndex];

            interval rgbRange{ 0.0f, 1.0f };
//          interval rgbRange{ 0.0f, 1.0f };
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

            return color3{ .x = imagePNG.rgbs[imagePixelIndex + 0],
                           .y = imagePNG.rgbs[imagePixelIndex + 1],
                           .z = imagePNG.rgbs[imagePixelIndex + 2],
                         };
        }
        break;
//      break;


        case textureType::IMAGE_TEXTURE_JPG:
//      case textureType::IMAGE_TEXTURE_JPG:
        {
            return {};
//          return {};
        }
        break;
//      break;


        case textureType::IMAGE_TEXTURE_SVG:
//      case textureType::IMAGE_TEXTURE_SVG:
        {
            return {};
//          return {};
        }
        break;
//      break;


        case textureType::NOISE_PERLIN_ORIGIN:
//      case textureType::NOISE_PERLIN_ORIGIN:
        {
            const noisePerlinOrigin& npo = noisesDatabase.noisePerlinOrigins[texture.noiseIndex];
//          const noisePerlinOrigin& npo = noisesDatabase.noisePerlinOrigins[texture.noiseIndex];
            return color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(npo, point);
//          return color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(npo, point);
        }
        break;
//      break;


        case textureType::NOISE_PERLIN_SMOOTH:
//      case textureType::NOISE_PERLIN_SMOOTH:
        {
            const noisePerlinSmooth& nps = noisesDatabase.noisePerlinSmooths[texture.noiseIndex];
//          const noisePerlinSmooth& nps = noisesDatabase.noisePerlinSmooths[texture.noiseIndex];
            return color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(nps, point);
//          return color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f } * GetNoiseValue(nps, point);
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


struct ray
{
    vec3 ori;
    vec3 dir;
    float time;
//  float time;

    point3 Marching(float t) const { return ori + dir * t; }
//  point3 Marching(float t) const { return ori + dir * t; }
};



static bool HitAABB(const ray& ray, interval rayT, const AABB2D& aabb2d)
{
    return true;
}
static bool HitAABB(const ray& ray, interval rayT, const AABB3D& aabb3d)
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



    inline static vec3 GenRandom(                    ) { return vec3 { Random(        ), Random(        ), Random(        ) }; }
    inline static vec3 GenRandom(float min, float max) { return vec3 { Random(min, max), Random(min, max), Random(min, max) }; }
    inline static vec3 GenRandomUnitVector()
//  inline static vec3 GenRandomUnitVector()
    {
        while (true)
        {
            const vec3& p = GenRandom(-1.0f, +1.0f);
//          const vec3& p = GenRandom(-1.0f, +1.0f);
            const float& pLengthSquared = p.length_squared();
//          const float& pLengthSquared = p.length_squared();
            if (pLengthSquared <= 1.0000f
            &&  pLengthSquared >  1e-160f)
            {
                return p / std::sqrt(pLengthSquared);
            }
        }
    }
    inline static vec3 GenRandomUnitVectorOnHemisphere(const vec3& normal)
//  inline static vec3 GenRandomUnitVectorOnHemisphere(const vec3& normal)
    {
        const vec3& randomUnitVector = GenRandomUnitVector();
//      const vec3& randomUnitVector = GenRandomUnitVector();
        if (dot(randomUnitVector, normal) > 0.0f)
//      if (dot(randomUnitVector, normal) > 0.0f)
        {
            return  randomUnitVector;
        }
        else
        {
            return -randomUnitVector;
        }
    }
    inline static vec3 GenRandomPointInsideNormalizedDisk()
    {
        while (true)
//      while (true)
        {
            point3 point { .x = Random(-1.0f , +1.0f), .y = Random(-1.0f , +1.0f), .z = 0.0f };
//          point3 point { .x = Random(-1.0f , +1.0f), .y = Random(-1.0f , +1.0f), .z = 0.0f };
            if (point.length_squared() < 1.0f)
//          if (point.length_squared() < 1.0f)
            {
                return point;
//              return point;
            }
        }
    }
    inline static vec3 DefocusDiskSample(const point3& diskCenter, const vec3& defocusDiskRadiusU, const vec3& defocusDiskRadiusV)
//  inline static vec3 DefocusDiskSample(const point3& diskCenter, const vec3& defocusDiskRadiusU, const vec3& defocusDiskRadiusV)
    {
        point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
//      point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
        return diskCenter + randomPointInsideNormalizedDisk.x * defocusDiskRadiusU
                          + randomPointInsideNormalizedDisk.y * defocusDiskRadiusV;
    }

    inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0f * dot(incomingVector, normal) * normal; }
//  inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0f * dot(incomingVector, normal) * normal; }

    inline static vec3 Refract(const vec3& incomingVector, const vec3& normal, float ratioOfEtaiOverEtat)
    {
        const float& cosTheta = std::fminf(dot(-incomingVector, normal), 1.0f);
//      const float& cosTheta = std::fminf(dot(-incomingVector, normal), 1.0f);
        const vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
//      const vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
        const vec3& refractedRayParallel = -std::sqrtf(std::fabsf(1.0f - refractedRayPerpendicular.length_squared())) * normal;
//      const vec3& refractedRayParallel = -std::sqrtf(std::fabsf(1.0f - refractedRayPerpendicular.length_squared())) * normal;
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
static vec3   BlendLinear(const vec3& startValue, const vec3& ceaseValue,       float ratio)
{
return vec3 {
              BlendLinear(startValue.x, ceaseValue.x, ratio),
              BlendLinear(startValue.y, ceaseValue.y, ratio),
              BlendLinear(startValue.z, ceaseValue.z, ratio),
            };
}
inline
static vec3   BlendLinear(const vec3& startValue, const vec3& ceaseValue, const vec3& ratio)
{
return vec3 {
              BlendLinear(startValue.x, ceaseValue.x, ratio.x),
              BlendLinear(startValue.y, ceaseValue.y, ratio.y),
              BlendLinear(startValue.z, ceaseValue.z, ratio.z),
            };
}


    enum class materialType       : std::uint8_t
//  enum class materialType       : std::uint8_t
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
};


    enum class materialDielectric : std::uint8_t
//  enum class materialDielectric : std::uint8_t
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


constexpr inline static float GetRefractionIndex(materialDielectric materialDielectric)
{
    switch ( materialDielectric )
    {
        case materialDielectric::GLASS  : return 1.500000f; break;
//      case materialDielectric::GLASS  : return 1.500000f; break;
        case materialDielectric::WATER  : return 1.333000f; break;
//      case materialDielectric::WATER  : return 1.333000f; break;
        case materialDielectric::AIR    : return 1.000293f; break;
//      case materialDielectric::AIR    : return 1.000293f; break;
        case materialDielectric::DIAMOND: return 2.400000f; break;
//      case materialDielectric::DIAMOND: return 2.400000f; break;
                                 default: return 0.000000f; break;
//                               default: return 0.000000f; break;
    }
}


struct material
{
    /* color3 albedo; */ float scatteredProbability; float fuzz; float refractionIndex; int textureIndex; materialType materialType;
//  /* color3 albedo; */ float scatteredProbability; float fuzz; float refractionIndex; int textureIndex; materialType materialType;    
};

struct materialScatteredResult
{
    ray scatteredRay; color3 attenuation; bool isScattered;
//  ray scatteredRay; color3 attenuation; bool isScattered;
};

enum class geometryType : std::int8_t
{
    SPHERE = 0,
//  SPHERE = 0,
};

struct geometry
{
    material material; /* point3 */ ray center; AABB3D aabb3d; float radius; geometryType geometryType;
//  material material; /* point3 */ ray center; AABB3D aabb3d; float radius; geometryType geometryType;

};


inline static bool IsStationary   (geometry& g) { return g.center.dir.x == 0.0f
                                                      && g.center.dir.y == 0.0f
                                                      && g.center.dir.z == 0.0f;
                                                }

inline static void CalculateAABB3D(geometry& g)
{
    if (IsStationary(g))
//  if (IsStationary(g))
    {
        switch (g.geometryType)
//      switch (g.geometryType)
        {
        case geometryType::SPHERE:
//      case geometryType::SPHERE:
            {
                g.aabb3d.intervalAxisX.min = g.center.ori.x - g.radius;
                g.aabb3d.intervalAxisX.max = g.center.ori.x + g.radius;
                g.aabb3d.intervalAxisY.min = g.center.ori.y - g.radius;
                g.aabb3d.intervalAxisY.max = g.center.ori.y + g.radius;
                g.aabb3d.intervalAxisZ.min = g.center.ori.z - g.radius;
                g.aabb3d.intervalAxisZ.max = g.center.ori.z + g.radius;
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
        case geometryType::SPHERE:
//      case geometryType::SPHERE:
            {
                const point3& destinationPoint3 = g.center.Marching(1.0f);
//              const point3& destinationPoint3 = g.center.Marching(1.0f);
                g.aabb3d.intervalAxisX.min = std::fminf(g.center.ori.x, destinationPoint3.x) - g.radius;
                g.aabb3d.intervalAxisX.max = std::fmaxf(g.center.ori.x, destinationPoint3.x) + g.radius;
                g.aabb3d.intervalAxisY.min = std::fminf(g.center.ori.y, destinationPoint3.y) - g.radius;
                g.aabb3d.intervalAxisY.max = std::fmaxf(g.center.ori.y, destinationPoint3.y) + g.radius;
                g.aabb3d.intervalAxisZ.min = std::fminf(g.center.ori.z, destinationPoint3.z) - g.radius;
                g.aabb3d.intervalAxisZ.max = std::fmaxf(g.center.ori.z, destinationPoint3.z) + g.radius;
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

inline static void CalculateAABB3D(std::vector<geometry>& geometries, AABB3D& aabb3d)
{
    for (geometry& g : geometries)
//  for (geometry& g : geometries)
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

struct rayHitResult
{
    material material; point3 at; vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace;
//  material material; point3 at; vec3 normal; float minT; float uSurfaceCoordinate; float vSurfaceCoordinate; bool hitted; bool isFrontFace;

    void SetFaceNormal(const ray& ray, const vec3& outwardNormal)
//  void SetFaceNormal(const ray& ray, const vec3& outwardNormal)
    {
            isFrontFace = dot(ray.dir, outwardNormal) < 0.0f;
        if (isFrontFace)
        {
            normal =  outwardNormal;
        }
        else
        {
            normal = -outwardNormal;
        }
    }
};





    inline static void GetUVSurfaceCoordinates(const geometry& geo, const point3& normalizedSurfacePoint, float& uSurfaceCoordinate, float& vSurfaceCoordinate)
//  inline static void GetUVSurfaceCoordinates(const geometry& geo, const point3& normalizedSurfacePoint, float& uSurfaceCoordinate, float& vSurfaceCoordinate)
{
    switch (geo.geometryType)
//  switch (geo.geometryType)
    {
        case geometryType::SPHERE:
//      case geometryType::SPHERE:
        {
            // GEOGRAPHIC COORDINATE
            // GEOGRAPHIC COORDINATE
            // normalizedSurfacePoint: a given point on the sphere of radius one and centered at the origin <0 0 0>
            // normalizedSurfacePoint: a given point on the sphere of radius one and centered at the origin <0 0 0>
            // uSurfaceCoordinate: returned value [0,1] of angle around the Y axis from X=-1
            // uSurfaceCoordinate: returned value [0,1] of angle around the Y axis from X=-1
            // vSurfaceCoordinate: returned value [0,1] of angle                   from Y=-1 to Y=+1
            // vSurfaceCoordinate: returned value [0,1] of angle                   from Y=-1 to Y=+1
            // <+1 +0 +0> yields <+0.50 +0.50> | <-1 +0 +0> yields <0.00 0.50>
            // <+0 +1 +0> yields <+0.50 +1.00> | <+0 -1 +0> yields <0.50 0.00>
            // <+0 +0 +1> yields <+0.25 +0.50> | <+0 +0 -1> yields <0.75 0.50>

            float theta = std::acos (-normalizedSurfacePoint.y); // latitude
//          float theta = std::acos (-normalizedSurfacePoint.y); // latitude
            float phi   = std::atan2(-normalizedSurfacePoint.z, normalizedSurfacePoint.x) + static_cast<float>(M_PI); // longitude
//          float phi   = std::atan2(-normalizedSurfacePoint.z, normalizedSurfacePoint.x) + static_cast<float>(M_PI); // longitude

            uSurfaceCoordinate = phi   / (2.0f * static_cast<float>(M_PI));
//          uSurfaceCoordinate = phi   / (2.0f * static_cast<float>(M_PI));
            vSurfaceCoordinate = theta /         static_cast<float>(M_PI) ;
//          vSurfaceCoordinate = theta /         static_cast<float>(M_PI) ;
        }
        break;
//      break;


        default:
//      default:
        {

        }
        break;
//      break;
    }
}

inline static materialScatteredResult Scatter(const ray& rayIn, const rayHitResult& rayHitResult)
{
    materialScatteredResult materialScatteredResult {};
//  materialScatteredResult materialScatteredResult {};
    switch (rayHitResult.material.materialType)
//  switch (rayHitResult.material.materialType)
    {

    case materialType::LambertianDiffuseReflectance1:
//  case materialType::LambertianDiffuseReflectance1:
        {
            vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
//          vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
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
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::LambertianDiffuseReflectance2:
//  case materialType::LambertianDiffuseReflectance2:
        {
            vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
//          vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
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
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::Metal:
//  case materialType::Metal:
        {
            vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::MetalFuzzy1:
//  case materialType::MetalFuzzy1:
        {
            vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
//          reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//          materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case materialType::MetalFuzzy2:
//  case materialType::MetalFuzzy2:
        {
            vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//          vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
            reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//          reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
            materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//          materialScatteredResult.scatteredRay.ori = rayHitResult.at;
            materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//          materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
            materialScatteredResult.scatteredRay.time = rayIn.time;
//          materialScatteredResult.scatteredRay.time = rayIn.time;
            materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* rayHitResult.material.albedo */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//          materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case materialType::Dielectric:
        {
            materialScatteredResult.attenuation = /* color3 { 1.0f, 1.0f, 1.0f }  */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
//          materialScatteredResult.attenuation = /* color3 { 1.0f, 1.0f, 1.0f }  */ Value(rayHitResult.material.textureIndex, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate, rayHitResult.at) / rayHitResult.material.scatteredProbability;
            float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
//          float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
            if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
//          if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
            vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);
//          vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);

            float cosTheta = std::fminf(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
//          float cosTheta = std::fminf(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
            float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
//          float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
            bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
//          bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
            vec3 scatteredRayDirection;
//          vec3 scatteredRayDirection;

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
            materialScatteredResult.isScattered = true;
//          materialScatteredResult.isScattered = true;
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
static rayHitResult RayHit(const geometry& geo
                          ,const ray     & ray
                          ,const interval& rayT
//                        ,const interval& rayT
                          )
{
    switch (geo.geometryType)
//  switch (geo.geometryType)
    {
        case geometryType::SPHERE:
//      case geometryType::SPHERE:
        {
            const point3& currentSphereCenterByIncomingRayTime = geo.center.Marching(ray.time);
//          const point3& currentSphereCenterByIncomingRayTime = geo.center.Marching(ray.time);
            const vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
//          const vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
            const float& a = ray.dir.length_squared();
//          const float& a = ray.dir.length_squared();
            const float& h = dot(ray.dir, fromSphereCenterToRayOrigin);
//          const float& h = dot(ray.dir, fromSphereCenterToRayOrigin);
            const float& c = fromSphereCenterToRayOrigin.length_squared() - geo.radius * geo.radius;
//          const float& c = fromSphereCenterToRayOrigin.length_squared() - geo.radius * geo.radius;
            const float& discriminant = h * h - a * c;
//          const float& discriminant = h * h - a * c;
            rayHitResult rayHitResult { .material = geo.material };
//          rayHitResult rayHitResult { .material = geo.material };
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

                const vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / geo.radius;
//              const vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / geo.radius;

                rayHitResult.SetFaceNormal(ray, outwardNormal);
//              rayHitResult.SetFaceNormal(ray, outwardNormal);

                GetUVSurfaceCoordinates(geo, outwardNormal, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate);
//              GetUVSurfaceCoordinates(geo, outwardNormal, rayHitResult.uSurfaceCoordinate, rayHitResult.vSurfaceCoordinate);
            }
    
            return rayHitResult;
//          return rayHitResult;
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



        inline static rayHitResult RayHit(const std::vector<geometry>& geometries, const ray& ray, const interval& rayT)
//      inline static rayHitResult RayHit(const std::vector<geometry>& geometries, const ray& ray, const interval& rayT)
{
    rayHitResult finalRayHitResult{};
//  rayHitResult finalRayHitResult{};
    float closestTSoFar = rayT.max;
//  float closestTSoFar = rayT.max;
    for (const geometry& geo : geometries)
//  for (const geometry& geo : geometries)
    {
        rayHitResult temporaryRayHitResult = std::move(RayHit(geo, ray, interval { .min = rayT.min, .max = closestTSoFar }));
//      rayHitResult temporaryRayHitResult = std::move(RayHit(geo, ray, interval { .min = rayT.min, .max = closestTSoFar }));
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
static color3 RayColor(const ray& ray)
{
    geometry sphere{ .center = { .ori = { 0.0f, 0.0f, -1.0f } }, .radius = 0.5f, .geometryType = geometryType::SPHERE };
//  geometry sphere{ .center = { .ori = { 0.0f, 0.0f, -1.0f } }, .radius = 0.5f, .geometryType = geometryType::SPHERE };
    const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { .min = -10.0f, .max = +10.0f });
//  const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { .min = -10.0f, .max = +10.0f });
    if (rayHitResult.hitted) _UNLIKELY
//  if (rayHitResult.hitted) _UNLIKELY
    {
        return color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
//      return color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
    }

    const vec3& normalizedRayDirection = normalize(ray.dir);
//  const vec3& normalizedRayDirection = normalize(ray.dir);
    const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//  const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
    return BlendLinear(color3{ 1.0f, 1.0f, 1.0f, }, color3{ 0.5f, 0.7f, 1.0f, }, ratio);
//  return BlendLinear(color3{ 1.0f, 1.0f, 1.0f, }, color3{ 0.5f, 0.7f, 1.0f, }, ratio);
}



/*
inline
static color3 RayColor(const ray& ray, const std::vector<geometry>& geometries, int recursiveDepth = 50)
{
    if (recursiveDepth <= 0.0f)
    {
        return color3 {};
//      return color3 {};
    }
    const rayHitResult& rayHitResult = RayHit(geometries, ray, interval { .min = 0.001f, .max = positiveInfinity });
//  const rayHitResult& rayHitResult = RayHit(geometries, ray, interval { .min = 0.001f, .max = positiveInfinity });
    if (rayHitResult.hitted)
//  if (rayHitResult.hitted)
    {
        const materialScatteredResult& materialScatteredResult = Scatter(ray, rayHitResult);
//      const materialScatteredResult& materialScatteredResult = Scatter(ray, rayHitResult);

        if (!materialScatteredResult.isScattered)
//      if (!materialScatteredResult.isScattered)
        {
            return color3 {};
//          return color3 {};
        }

        return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, geometries, --recursiveDepth);
//      return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, geometries, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, geometries, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, geometries, --recursiveDepth);
//      return color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
//      return color3 { rayHitResult.normal.x + 1.0f, rayHitResult.normal.y + 1.0f, rayHitResult.normal.z + 1.0f, } * 0.5f;
    }

    const vec3& normalizedRayDirection = normalize(ray.dir);
//  const vec3& normalizedRayDirection = normalize(ray.dir);
    const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//  const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
    return BlendLinear(color3{ 1.0f, 1.0f, 1.0f, }, color3{ 0.5f, 0.7f, 1.0f, }, ratio);
//  return BlendLinear(color3{ 1.0f, 1.0f, 1.0f, }, color3{ 0.5f, 0.7f, 1.0f, }, ratio);
}
*/



inline
static color3 RayColor(const ray& initialRay, const std::vector<geometry>& geometries, int maxDepth = 50)
{
    color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
//  color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
    ray currentRay = initialRay;
//  ray currentRay = initialRay;

    for (int depth = 0; depth < maxDepth; ++depth)
//  for (int depth = 0; depth < maxDepth; ++depth)
    {
        const rayHitResult& rayHitResult = RayHit(geometries, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });
//      const rayHitResult& rayHitResult = RayHit(geometries, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });

        if (rayHitResult.hitted) _UNLIKELY
//      if (rayHitResult.hitted) _UNLIKELY
        {
            const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered) _UNLIKELY
//          if (!materialScatteredResult.isScattered) _UNLIKELY
            {
                return color3{};  // Return black if scattering fails
//              return color3{};  // Return black if scattering fails
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
            const vec3& normalizedRayDirection = normalize(currentRay.dir);
//          const vec3& normalizedRayDirection = normalize(currentRay.dir);
            const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//          const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
            color3 backgroundColor = BlendLinear(color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
//          color3 backgroundColor = BlendLinear(color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
            return finalColor * backgroundColor;
//          return finalColor * backgroundColor;
        }
    }

    // If we reach max depth, return black
//  // If we reach max depth, return black
    return color3{};
//  return color3{};
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
    std::vector<BVHNode> bvhNodes; std::vector<geometry> geometries;
//  std::vector<BVHNode> bvhNodes; std::vector<geometry> geometries;
};
inline static rayHitResult RayHit(const BVHTree& bvhTree, int bvhNodeIndex, const ray& ray, const interval& rayT)
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
        rayHitResult rayHitResult{};
//      rayHitResult rayHitResult{};
        rayHitResult.hitted = false;
//      rayHitResult.hitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    // Recursively traverse children
    // Recursively traverse children
    rayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray,        rayT);
//  rayHitResult rayHitResultL = RayHit(bvhTree, bvhNode.childIndexL, ray,        rayT);
    interval updatedRayT;
//  interval updatedRayT;
    if (rayHitResultL.hitted)
//  if (rayHitResultL.hitted)
    {
        updatedRayT = interval{ .min = rayT.min, .max = rayHitResultL.minT };
//      updatedRayT = interval{ .min = rayT.min, .max = rayHitResultL.minT };
    }
    else
    {
        updatedRayT = rayT;
//      updatedRayT = rayT;
    }
    rayHitResult rayHitResultR = RayHit(bvhTree, bvhNode.childIndexR, ray, updatedRayT);
//  rayHitResult rayHitResultR = RayHit(bvhTree, bvhNode.childIndexR, ray, updatedRayT);

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
    {
        return rayHitResultR;
    }
}
/*
static inline bool AABB3DCompareAxisX(const geometry& g1, const geometry& g2) { return g1.aabb3d.intervalAxisX.min < g2.aabb3d.intervalAxisX.min; }
static inline bool AABB3DCompareAxisY(const geometry& g1, const geometry& g2) { return g1.aabb3d.intervalAxisY.min < g2.aabb3d.intervalAxisY.min; }
static inline bool AABB3DCompareAxisZ(const geometry& g1, const geometry& g2) { return g1.aabb3d.intervalAxisZ.min < g2.aabb3d.intervalAxisZ.min; }
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
        std::function<bool(const geometry&, const geometry&)> comparator;
//      std::function<bool(const geometry&, const geometry&)> comparator;
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
    static inline float GetCentroid(const geometry& geo, const Axis& axis)
//  static inline float GetCentroid(const geometry& geo, const Axis& axis)
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
            std::function<bool(const geometry& geo1, const geometry& geo2)> comparator = [axis]
//          std::function<bool(const geometry& geo1, const geometry& geo2)> comparator = [axis]
                              (const geometry& geo1, const geometry& geo2)
//                            (const geometry& geo1, const geometry& geo2)
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
        std::function<bool(const geometry& geo1, const geometry& geo2)> bestComparator = [bestAxis]
//      std::function<bool(const geometry& geo1, const geometry& geo2)> bestComparator = [bestAxis]
                          (const geometry& geo1, const geometry& geo2)
//                        (const geometry& geo1, const geometry& geo2)
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
inline
static color3 RayColor(const ray& initialRay, const BVHTree& bvhTree, int maxDepth = 50)
{
    color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
//  color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
    ray currentRay = initialRay;
//  ray currentRay = initialRay;

    for (int depth = 0; depth < maxDepth; ++depth)
//  for (int depth = 0; depth < maxDepth; ++depth)
    {
        const rayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });
//      const rayHitResult& rayHitResult = RayHit(bvhTree, 0, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });

        if (rayHitResult.hitted) _UNLIKELY
//      if (rayHitResult.hitted) _UNLIKELY
        {
            const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered) _UNLIKELY
//          if (!materialScatteredResult.isScattered) _UNLIKELY
            {
                return color3{};  // Return black if scattering fails
//              return color3{};  // Return black if scattering fails
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
            const vec3& normalizedRayDirection = normalize(currentRay.dir);
//          const vec3& normalizedRayDirection = normalize(currentRay.dir);
            const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
//          const float& ratio = 0.5f * (normalizedRayDirection.y + 1.0f);
            color3 backgroundColor = BlendLinear(color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
//          color3 backgroundColor = BlendLinear(color3{ .x = 1.0f, .y = 1.0f, .z = 1.0f }, color3{ .x = 0.5f, .y = 0.7f, .z = 1.0f }, ratio);
            return finalColor * backgroundColor;
//          return finalColor * backgroundColor;
        }
    }

    // If we reach max depth, return black
//  // If we reach max depth, return black
    return color3{};
//  return color3{};
}



int main()
{
    noisesDatabase.noisePerlinOrigins.emplace_back();
    noisesDatabase.noisePerlinSmooths.emplace_back();
    for (noisePerlinOrigin& npo : noisesDatabase.noisePerlinOrigins) Generate(npo);
    for (noisePerlinSmooth& nps : noisesDatabase.noisePerlinSmooths) Generate(nps);



    imagesDatabase.pngs.emplace_back("example.png");
//  imagesDatabase.jpgs.emplace_back("example.jpg");
//  imagesDatabase.svgs.emplace_back("example.svg");



    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 1.0f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 1.0f, 0.0f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.8f, 0.8f, 0.8f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.5f, 0.5f, 0.5f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.2f, 0.2f, 0.2f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.5f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +3, .eTileTextureIndex = +4, .type = textureType::CHECKER_TEXTURE_1, });
//  texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.5f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +3, .eTileTextureIndex = +4, .type = textureType::CHECKER_TEXTURE_1, });

    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.5f, 1.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 1.0f, 0.5f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::SOLID_COLOR, });

    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.1f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +6, .eTileTextureIndex = +7, .type = textureType::CHECKER_TEXTURE_1, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 0.1f, .imageIndex = -1, .noiseIndex = -1, .oTileTextureIndex = +6, .eTileTextureIndex = +7, .type = textureType::CHECKER_TEXTURE_2, });

    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::IMAGE_TEXTURE_PNG, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::IMAGE_TEXTURE_JPG, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = +0, .noiseIndex = -1, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::IMAGE_TEXTURE_SVG, });

    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = +0, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::NOISE_PERLIN_ORIGIN, });
    texturesDatabase.textures.emplace_back(texture{ .albedo = { 0.0f, 0.0f, 0.0f }, .scale = 1.0f, .imageIndex = -1, .noiseIndex = +0, .oTileTextureIndex = -1, .eTileTextureIndex = -1, .type = textureType::NOISE_PERLIN_SMOOTH, });



    ThreadPool* threadPool = new ThreadPool(225);
//  ThreadPool* threadPool = new ThreadPool(225);

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    int                              samplesPerPixel = 100 ;
//  int                              samplesPerPixel = 100 ;
    float pixelSamplesScale = 1.0f / samplesPerPixel       ;
//  float pixelSamplesScale = 1.0f / samplesPerPixel       ;



//  std::vector<geometry> geometries;
//  std::vector<geometry> geometries;
    BVHTree bvhTree;
//  BVHTree bvhTree;
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 1.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   0, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +000.600f,  000.000f, -001.000f }, .dir = { +000.600f,  000.000f, +001.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 1.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   0, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 1.0f, 0.0f, 1.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   1, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { -000.600f,  000.000f, -002.500f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 1.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   2, .materialType = materialType::Metal                         },  .center = { .ori = { +000.600f,  000.000f, -003.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 1.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   2, .materialType = materialType::Metal                         },  .center = { .ori = { +000.600f,  000.000f, -003.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.8f, 0.8f, 0.8f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .textureIndex =   2, .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.8f, 0.8f, 0.8f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(materialDielectric::GLASS), .textureIndex =   2, .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.8f, 0.8f, 0.8f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .textureIndex =   2, .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.400f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.5f, 0.5f, 0.5f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   5, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = {  000.000f, -100.500f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 100.000f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 0.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   8, .materialType = materialType::Metal                         },  .center = { .ori = { -001.800f,  000.000f, -003.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 0.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =   9, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { -001.800f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
    bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 0.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =  10, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +001.800f,  000.000f, -002.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 0.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =  11, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +001.800f,  000.000f, -002.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { 0.0f, 0.0f, 0.0f }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .textureIndex =  12, .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +001.800f,  000.000f, -002.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f, .geometryType = geometryType::SPHERE,  });



//  for (int i = -5; i < 5; ++i)
//  for (int j = -5; j < 5; ++j)
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { Random(0.0f, 1.0f), Random(0.0f, 1.0f), Random(0.0f, 1.0f) }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = 0.0f, .textureIndex = 0, .materialType = materialType::LambertianDiffuseReflectance1 }, .center = { .ori = { float(i) * 0.5f, -0.4f, float(j) * 0.5f - 1.0f }, .dir = { 0.0f, 0.0f, 0.0f }, .time = 0.0f }, .radius = 0.1f, .geometryType = geometryType::SPHERE,  });
//  bvhTree.geometries.emplace_back(geometry{  .material = { /* .albedo = { Random(0.0f, 1.0f), Random(0.0f, 1.0f), Random(0.0f, 1.0f) }, */ .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = 0.0f, .textureIndex = 0, .materialType = materialType::LambertianDiffuseReflectance1 }, .center = { .ori = { float(i) * 0.5f, -0.4f, float(j) * 0.5f - 1.0f }, .dir = { 0.0f, 0.0f, 0.0f }, .time = 0.0f }, .radius = 0.1f, .geometryType = geometryType::SPHERE,  });

    for (geometry& geo : bvhTree.geometries) CalculateAABB3D(geo);
//  for (geometry& geo : bvhTree.geometries) CalculateAABB3D(geo);
    
//  for (geometry& geo : bvhTree.geometries)
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
    int imgW = 400;
//  int imgW = 400;
    int imgH = int(imgW / aspectRatio);
//  int imgH = int(imgW / aspectRatio);
    imgH = std::max(imgH, 1);
//  imgH = std::max(imgH, 1);

    const point3 lookFrom { .x = -2.0f, .y = +2.0f, .z = +1.0f };
    const point3 lookAt   { .x = -0.6f, .y = +0.0f, .z = -1.0f };
//  const point3 lookAt   { .x = +0.0f, .y = +0.0f, .z = -1.0f };
    const point3 viewUp   { .x = +0.0f, .y = +1.0f, .z = +0.0f };

    vec3 cameraU; // x
    vec3 cameraV; // y
    vec3 cameraW; // z

    float defocusAngle = 0.00f * M_PI; float focusDistance = (lookAt - lookFrom).length();
//  float defocusAngle = 0.00f * M_PI; float focusDistance = (lookAt - lookFrom).length();
//  float defocusAngle = 0.00f * M_PI; float focusDistance = 10.0f;
//  float defocusAngle = 0.00f * M_PI; float focusDistance = 10.0f;
    vec3 defocusDiskRadiusU;
    vec3 defocusDiskRadiusV;


    float vFOV = M_PI / 4.0f;
    float hFOV = M_PI / 2.0f;
    float h = std::tanf(vFOV / 2.0f);
    float w = std::tanf(hFOV / 2.0f);

    float focalLength = (lookAt - lookFrom).length();
//  float focalLength = (lookAt - lookFrom).length();


    cameraW = normalize(lookFrom - lookAt); cameraU = normalize(cross(viewUp, cameraW)); cameraV = cross(cameraW, cameraU);
//  cameraW = normalize(lookFrom - lookAt); cameraU = normalize(cross(viewUp, cameraW)); cameraV = cross(cameraW, cameraU);

    float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
//  float defocusRadius = focusDistance * std::tanf(defocusAngle / 2.0f);
    defocusDiskRadiusU = cameraU * defocusRadius;
    defocusDiskRadiusV = cameraV * defocusRadius;


    float viewportH = 2.0f * h * /* focalLength */ focusDistance;
//  float viewportH = 2.0f * h * /* focalLength */ focusDistance;
    float viewportW = viewportH * (float(imgW) / imgH);
//  float viewportW = viewportH * (float(imgW) / imgH);

    point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;
//  point3 cameraCenter /* { 0.0f, 0.0f, 0.0f, } */ = lookFrom;



//  vec3 viewportU { };
//  viewportU.x = +viewportW;
//  viewportU.y = 0.0f;
//  viewportU.z = 0.0f;
//  vec3 viewportV { };
//  viewportV.x = 0.0f;
//  viewportV.y = -viewportH;
//  viewportV.z = 0.0f;

    vec3 viewportU = viewportW *  cameraU;
    vec3 viewportV = viewportH * -cameraV;


    vec3 fromPixelToPixelDeltaU = viewportU / float(imgW);
    vec3 fromPixelToPixelDeltaV = viewportV / float(imgH);



    point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
//  point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0f - viewportV / 2.0f;
    point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;
//  point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5f + fromPixelToPixelDeltaV * 0.5f;


    std::ofstream PPMFile(GetCurrentDateTime());
//  std::ofstream PPMFile(GetCurrentDateTime());
    PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";
//  PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";

    constexpr int numberOfChannels = 3; // R G B
//  constexpr int numberOfChannels = 3; // R G B
    std::vector<int> rgbs(imgW * imgH * numberOfChannels, 255);
//  std::vector<int> rgbs(imgW * imgH * numberOfChannels, 255);

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


//      point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
//      point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
//      vec3 rayDirection = pixelCenter - cameraCenter;
//      vec3 rayDirection = pixelCenter - cameraCenter;
//      ray  ray  { cameraCenter, rayDirection };
//      ray  ray  { cameraCenter, rayDirection };

//      float r = float(pixelX) / float(imgW - 1);
//      float g = float(pixelY) / float(imgH - 1);
//      float b = 0.00f;
//      color3 pixelColor { r, g, b, };
//      color3 pixelColor { r, g, b, };

//      color3 pixelColor = RayColor(ray);
//      color3 pixelColor = RayColor(ray);
        
//      color3 pixelColor = RayColor(ray, geometries);
//      color3 pixelColor = RayColor(ray, geometries);

        color3 pixelColor{};
//      color3 pixelColor{};
        for (int sample = 0; sample < samplesPerPixel; ++sample)
        {
            vec3 sampleOffset{ Random() - 0.5f, Random() - 0.5f, 0.0f };
//          vec3 sampleOffset{ Random() - 0.5f, Random() - 0.5f, 0.0f };
            point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          vec3 rayDirection = pixelSampleCenter - cameraCenter;
//          vec3 rayDirection = pixelSampleCenter - cameraCenter;
            vec3 rayOrigin = cameraCenter;
//          vec3 rayOrigin = cameraCenter;
            if (defocusAngle > 0.0f) _UNLIKELY { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
//          if (defocusAngle > 0.0f) _UNLIKELY { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
            vec3 rayDirection = pixelSampleCenter - rayOrigin;
//          vec3 rayDirection = pixelSampleCenter - rayOrigin;
            ray  ray{ .ori = rayOrigin, .dir = rayDirection, .time = Random() };
//          ray  ray{ .ori = rayOrigin, .dir = rayDirection, .time = Random() };
            pixelColor += RayColor(ray, bvhTree);
//          pixelColor += RayColor(ray, bvhTree);
//          pixelColor += RayColor(ray, geometries);
//          pixelColor += RayColor(ray, geometries);
        }
        pixelColor *= pixelSamplesScale;
//      pixelColor *= pixelSamplesScale;

        static const interval intensity { 0.000f , 0.999f };
//      static const interval intensity { 0.000f , 0.999f };
        int ir = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.x)));
        int ig = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.y)));
        int ib = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.z)));

        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        rgbs[index + 0] = ir;
        rgbs[index + 1] = ig;
        rgbs[index + 2] = ib;
    }
    });
    }

    delete threadPool;
//  delete threadPool;
//  threadPool = nullptr;
//  threadPool = nullptr;

//  for (std::thread& t : threads) { t.join(); }
//  for (std::thread& t : threads) { t.join(); }

//  DENOISE 001
//  DENOISE 001
/*
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


        rgbs[indexCC + 0] = int(rgbs[indexLT + 0] * 0.075f + rgbs[indexJT + 0] * 0.124f + rgbs[indexRT + 0] * 0.075f + rgbs[indexLC + 0] * 0.124f + rgbs[indexCC + 0] * 0.204f + rgbs[indexRC + 0] * 0.124f + rgbs[indexLB + 0] * 0.075f + rgbs[indexJB + 0] * 0.124f + rgbs[indexRB + 0] * 0.075f);
        rgbs[indexCC + 1] = int(rgbs[indexLT + 1] * 0.075f + rgbs[indexJT + 1] * 0.124f + rgbs[indexRT + 1] * 0.075f + rgbs[indexLC + 1] * 0.124f + rgbs[indexCC + 1] * 0.204f + rgbs[indexRC + 1] * 0.124f + rgbs[indexLB + 1] * 0.075f + rgbs[indexJB + 1] * 0.124f + rgbs[indexRB + 1] * 0.075f);
        rgbs[indexCC + 2] = int(rgbs[indexLT + 2] * 0.075f + rgbs[indexJT + 2] * 0.124f + rgbs[indexRT + 2] * 0.075f + rgbs[indexLC + 2] * 0.124f + rgbs[indexCC + 2] * 0.204f + rgbs[indexRC + 2] * 0.124f + rgbs[indexLB + 2] * 0.075f + rgbs[indexJB + 2] * 0.124f + rgbs[indexRB + 2] * 0.075f);
    }
    }
*/

// CHROMATIC ABERRATION
// CHROMATIC ABERRATION
/*
    std::vector<int> chromaticAberrationRGBs(imgW* imgH* numberOfChannels, 255);
//  std::vector<int> chromaticAberrationRGBs(imgW* imgH* numberOfChannels, 255);
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

    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
    for (int pixelX = 0; pixelX < imgW; ++pixelX)
    {
        size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
//      size_t index = (static_cast<size_t>(pixelY) * imgW + pixelX) * numberOfChannels;
        PPMFile << std::setw(3) << rgbs[index + 0] << " ";
        PPMFile << std::setw(3) << rgbs[index + 1] << " ";
        PPMFile << std::setw(3) << rgbs[index + 2] << " ";
    }
        PPMFile << "\n";
    }

    PPMFile.close();

    const std::chrono::steady_clock::time_point& ceaseTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& ceaseTime = std::chrono::high_resolution_clock::now();
    const std::chrono::microseconds& executionDuration = std::chrono::duration_cast<std::chrono::microseconds>(ceaseTime - startTime);
//  const std::chrono::microseconds& executionDuration = std::chrono::duration_cast<std::chrono::microseconds>(ceaseTime - startTime);

    std::cout << executionDuration.count() << " " << "microseconds" << std::endl;
//  std::cout << executionDuration.count() << " " << "microseconds" << std::endl;

    return 0       ;
//  return 0       ;
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
// -------------------------
// | 0.075 | 0.124 | 0.075 |
// -------------------------
// | 0.124 | 0.204 | 0.124 |
// -------------------------
// | 0.075 | 0.124 | 0.075 |
// -------------------------
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
































