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


static std::string GetCurrentDateTime() {
       std::time_t t = std::time(nullptr); std::tm tm = *std::localtime(&t); char buffer[30]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S.ppm", &tm); return std::string(buffer);
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


enum class materialType : std::uint8_t
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


enum class materialDielectric : std::int8_t
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
    switch (materialDielectric)
    {
    case materialDielectric::GLASS  : return 1.500000f; break;
    case materialDielectric::WATER  : return 1.333000f; break;
    case materialDielectric::AIR    : return 1.000293f; break;
    case materialDielectric::DIAMOND: return 2.400000f; break;
                             default: return 0.000000f; break;
    }
}


struct material
{
    color3 albedo; float scatteredProbability; float fuzz; float refractionIndex; materialType materialType;
//  color3 albedo; float scatteredProbability; float fuzz; float refractionIndex; materialType materialType;    
};

struct materialScatteredResult
{
    ray scatteredRay; color3 attenuation; bool isScattered;
//  ray scatteredRay; color3 attenuation; bool isScattered;
};


struct sphere
{
    material material; /* point3 */ ray center; AABB3D aabb3d; float radius;
//  material material; /* point3 */ ray center; AABB3D aabb3d; float radius;

};


inline static bool IsStationary   (sphere& sphere) { return sphere.center.dir.x == 0.0f && sphere.center.dir.y == 0.0f && sphere.center.dir.z == 0.0f; }
inline static bool IsStationary   (              ) { return false;                                                                                     }

inline static void CalculateAABB3D(sphere& sphere)
{
    if (IsStationary(sphere))
    {
        sphere.aabb3d.intervalAxisX.min = sphere.center.ori.x - sphere.radius;
        sphere.aabb3d.intervalAxisX.max = sphere.center.ori.x + sphere.radius;
        sphere.aabb3d.intervalAxisY.min = sphere.center.ori.y - sphere.radius;
        sphere.aabb3d.intervalAxisY.max = sphere.center.ori.y + sphere.radius;
        sphere.aabb3d.intervalAxisZ.min = sphere.center.ori.z - sphere.radius;
        sphere.aabb3d.intervalAxisZ.max = sphere.center.ori.z + sphere.radius;
    }
    else
    {
        const point3& destinationPoint3 = sphere.center.Marching(1.0f);
        sphere.aabb3d.intervalAxisX.min = std::fminf(sphere.center.ori.x, destinationPoint3.x) - sphere.radius;
        sphere.aabb3d.intervalAxisX.max = std::fmaxf(sphere.center.ori.x, destinationPoint3.x) + sphere.radius;
        sphere.aabb3d.intervalAxisY.min = std::fminf(sphere.center.ori.y, destinationPoint3.y) - sphere.radius;
        sphere.aabb3d.intervalAxisY.max = std::fmaxf(sphere.center.ori.y, destinationPoint3.y) + sphere.radius;
        sphere.aabb3d.intervalAxisZ.min = std::fminf(sphere.center.ori.z, destinationPoint3.z) - sphere.radius;
        sphere.aabb3d.intervalAxisZ.max = std::fmaxf(sphere.center.ori.z, destinationPoint3.z) + sphere.radius;
    }
}

inline static void CalculateAABB3D(std::vector<sphere>& spheres, AABB3D& aabb3d)
{
    for (sphere& sphere : spheres)
//  for (sphere& sphere : spheres)
    {
        CalculateAABB3D(sphere);
//      CalculateAABB3D(sphere);
        aabb3d.intervalAxisX.min = std::fminf(sphere.aabb3d.intervalAxisX.min, aabb3d.intervalAxisX.min);
        aabb3d.intervalAxisX.max = std::fmaxf(sphere.aabb3d.intervalAxisX.max, aabb3d.intervalAxisX.max);
        aabb3d.intervalAxisY.min = std::fminf(sphere.aabb3d.intervalAxisY.min, aabb3d.intervalAxisY.min);
        aabb3d.intervalAxisY.max = std::fmaxf(sphere.aabb3d.intervalAxisY.max, aabb3d.intervalAxisY.max);
        aabb3d.intervalAxisZ.min = std::fminf(sphere.aabb3d.intervalAxisZ.min, aabb3d.intervalAxisZ.min);
        aabb3d.intervalAxisZ.max = std::fmaxf(sphere.aabb3d.intervalAxisZ.max, aabb3d.intervalAxisZ.max);
    }
}

struct rayHitResult
{
    material material; point3 at; vec3 normal; float minT; bool hitted; bool isFrontFace;
//  material material; point3 at; vec3 normal; float minT; bool hitted; bool isFrontFace;

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
//      vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal);
        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        if (scatteredDirection.NearZero()) _UNLIKELY
//      if (scatteredDirection.NearZero()) _UNLIKELY
        {
            materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//          materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
        }
        else
        {
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
        }
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
//      materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::LambertianDiffuseReflectance2:
//  case materialType::LambertianDiffuseReflectance2:
        {
        vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
//      vec3 scatteredDirection = rayHitResult.normal + GenRandomUnitVector();
        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        if (scatteredDirection.NearZero()) _UNLIKELY
//      if (scatteredDirection.NearZero()) _UNLIKELY
        {
            materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//          materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
        }
        else
        {
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
        }
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
//      materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::Metal:
//  case materialType::Metal:
        {
        vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//      vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//      materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
//      materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
        }
        break;
//      break;



    case materialType::MetalFuzzy1:
//  case materialType::MetalFuzzy1:
        {
        vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//      vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
        reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
//      reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVector());
        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//      materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
//      materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
        materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//      materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case materialType::MetalFuzzy2:
//  case materialType::MetalFuzzy2:
        {
        vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
//      vec3 reflectionScatteredDirection = Reflect(rayIn.dir, rayHitResult.normal);
        reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
//      reflectionScatteredDirection = normalize(reflectionScatteredDirection) + (rayHitResult.material.fuzz * GenRandomUnitVectorOnHemisphere(rayHitResult.normal));
        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
//      materialScatteredResult.scatteredRay.dir = reflectionScatteredDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
//      materialScatteredResult.attenuation = rayHitResult.material.albedo / rayHitResult.material.scatteredProbability;
        materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
//      materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0.0f;
        }
        break;
//      break;



    case materialType::Dielectric:
        {
        materialScatteredResult.attenuation = color3 { 1.0f, 1.0f, 1.0f };
//      materialScatteredResult.attenuation = color3 { 1.0f, 1.0f, 1.0f };
        float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
//      float ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
        if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
//      if (rayHitResult.isFrontFace) _LIKELY { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
        vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);
//      vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);

        float cosTheta = std::fminf(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
//      float cosTheta = std::fminf(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0f);
        float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
//      float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);
        bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
//      bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0f || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
        vec3 scatteredRayDirection;
//      vec3 scatteredRayDirection;

        if ( notAbleToRefract )
        {
             scatteredRayDirection = Reflect(normalizedIncomingRayDirection, rayHitResult.normal);
//           scatteredRayDirection = Reflect(normalizedIncomingRayDirection, rayHitResult.normal);
        }
        else
        {
             scatteredRayDirection = Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat);
//           scatteredRayDirection = Refract(normalizedIncomingRayDirection, rayHitResult.normal, ratioOfEtaiOverEtat);
        }

        materialScatteredResult.scatteredRay.ori = rayHitResult.at;
//      materialScatteredResult.scatteredRay.ori = rayHitResult.at;
        materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
//      materialScatteredResult.scatteredRay.dir = scatteredRayDirection;
        materialScatteredResult.scatteredRay.time = rayIn.time;
//      materialScatteredResult.scatteredRay.time = rayIn.time;
        materialScatteredResult.isScattered = true;
//      materialScatteredResult.isScattered = true;
        }
        break;



    default:
//  default:
        break;
//      break;
    }
    return materialScatteredResult;
//  return materialScatteredResult;
}




inline
static rayHitResult RayHit(const sphere& sphere
                          ,const ray   & ray
                          ,const interval& rayT
//                        ,const interval& rayT
                          )
{
    const point3& currentSphereCenterByIncomingRayTime = sphere.center.Marching(ray.time);
//  const point3& currentSphereCenterByIncomingRayTime = sphere.center.Marching(ray.time);
    const vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
//  const vec3& fromSphereCenterToRayOrigin = currentSphereCenterByIncomingRayTime - ray.ori;
    const float& a = ray.dir.length_squared();
//  const float& a = ray.dir.length_squared();
    const float& h = dot(ray.dir, fromSphereCenterToRayOrigin);
//  const float& h = dot(ray.dir, fromSphereCenterToRayOrigin);
    const float& c = fromSphereCenterToRayOrigin.length_squared() - sphere.radius * sphere.radius;
//  const float& c = fromSphereCenterToRayOrigin.length_squared() - sphere.radius * sphere.radius;
    const float& discriminant = h * h - a * c;
//  const float& discriminant = h * h - a * c;
    rayHitResult
    rayHitResult { sphere.material };
//  rayHitResult.hitted = discriminant >= 0.0f;
//  rayHitResult.hitted = discriminant >= 0.0f;
    if (discriminant < 0.0f)
    {
        rayHitResult.hitted = false;
//      rayHitResult.hitted = false;
    }
    else
    {
        float sqrtDiscriminant = std::sqrt(discriminant);
//      float sqrtDiscriminant = std::sqrt(discriminant);

        float t = (h - sqrtDiscriminant) / a;
//      float t = (h - sqrtDiscriminant) / a;

        if (!rayT.Surrounds(t))
//      if (!rayT.Surrounds(t))
        {
            t = (h + sqrtDiscriminant) / a;
//          t = (h + sqrtDiscriminant) / a;

            if (!rayT.Surrounds(t))
//          if (!rayT.Surrounds(t))
            {
                rayHitResult.hitted = false;
//              rayHitResult.hitted = false;
                return rayHitResult;
//              return rayHitResult;
            }
        }

        rayHitResult.hitted = true;
//      rayHitResult.hitted = true;

        rayHitResult.minT = t;
//      rayHitResult.minT = t;

        rayHitResult.at = ray.Marching(rayHitResult.minT);
//      rayHitResult.at = ray.Marching(rayHitResult.minT);

        const vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / sphere.radius;
//      const vec3& outwardNormal = (rayHitResult.at - currentSphereCenterByIncomingRayTime) / sphere.radius;

        rayHitResult.SetFaceNormal(ray, outwardNormal);
//      rayHitResult.SetFaceNormal(ray, outwardNormal);
    }
    
    return rayHitResult;
//  return rayHitResult;
}



        inline static rayHitResult RayHit(const std::vector<sphere>& spheres, const ray& ray, const interval& rayT)
//      inline static rayHitResult RayHit(const std::vector<sphere>& spheres, const ray& ray, const interval& rayT)
{
    rayHitResult finalRayHitResult{};
//  rayHitResult finalRayHitResult{};
    float closestTSoFar = rayT.max;
//  float closestTSoFar = rayT.max;
    for (const sphere& sphere : spheres)
//  for (const sphere& sphere : spheres)
    {
        rayHitResult temporaryRayHitResult = std::move(RayHit(sphere, ray, interval { .min = rayT.min, .max = closestTSoFar }));
//      rayHitResult temporaryRayHitResult = std::move(RayHit(sphere, ray, interval { .min = rayT.min, .max = closestTSoFar }));
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
    sphere sphere{ { 0.0f, 0.0f, -1.0f }, 0.5f, };
//  sphere sphere{ { 0.0f, 0.0f, -1.0f }, 0.5f, };
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
static color3 RayColor(const ray& ray, const std::vector<sphere>& spheres, int recursiveDepth = 50)
{
    if (recursiveDepth <= 0.0f)
    {
        return color3 {};
//      return color3 {};
    }
    const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { .min = 0.001f, .max = positiveInfinity });
//  const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { .min = 0.001f, .max = positiveInfinity });
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

        return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, spheres, --recursiveDepth);
//      return materialScatteredResult.attenuation * RayColor(materialScatteredResult.scatteredRay, spheres, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, spheres, --recursiveDepth);
//      return 0.5f * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, spheres, --recursiveDepth);
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
static color3 RayColor(const ray& initialRay, const std::vector<sphere>& spheres, int maxDepth = 50)
{
    color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
//  color3 finalColor = { .x = 1.0f, .y = 1.0f, .z = 1.0f };  // Initial color multiplier
    ray currentRay = initialRay;
//  ray currentRay = initialRay;

    for (int depth = 0; depth < maxDepth; ++depth)
//  for (int depth = 0; depth < maxDepth; ++depth)
    {
        const rayHitResult& rayHitResult = RayHit(spheres, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });
//      const rayHitResult& rayHitResult = RayHit(spheres, currentRay, interval{ .min = 0.001f, .max = positiveInfinity });

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
    std::vector<BVHNode> bvhNodes; std::vector<sphere> spheres;
//  std::vector<BVHNode> bvhNodes; std::vector<sphere> spheres;
};
inline static rayHitResult RayHit(const BVHTree& bvhTree, int bvhNodeIndex, const ray& ray, const interval& rayT)
{
    const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];
//  const BVHNode& bvhNode = bvhTree.bvhNodes[bvhNodeIndex];

    // Leaf node: test sphere intersection
    // Leaf node: test sphere intersection
    if (bvhNode.shapeIndex != -1)
//  if (bvhNode.shapeIndex != -1)
    {
        return RayHit(bvhTree.spheres[bvhNode.shapeIndex], ray, rayT);
//      return RayHit(bvhTree.spheres[bvhNode.shapeIndex], ray, rayT);
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
static inline bool AABB3DCompareAxisX(const sphere& sphere1, const sphere& sphere2) { return sphere1.aabb3d.intervalAxisX.min < sphere2.aabb3d.intervalAxisX.min; }
static inline bool AABB3DCompareAxisY(const sphere& sphere1, const sphere& sphere2) { return sphere1.aabb3d.intervalAxisY.min < sphere2.aabb3d.intervalAxisY.min; }
static inline bool AABB3DCompareAxisZ(const sphere& sphere1, const sphere& sphere2) { return sphere1.aabb3d.intervalAxisZ.min < sphere2.aabb3d.intervalAxisZ.min; }
inline static int  BuildBVHTree(BVHTree& bvhTree, int start, int cease)
{
    int objectSpan = cease - start;
//  int objectSpan = cease - start;
    if (objectSpan == 1)
    {
        // Single sphere: create a leaf node directly
        // Single sphere: create a leaf node directly
        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1 });
//      bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1 });
        return current;
    }
    else
    if (objectSpan == 2)
    {
        // Two spheres: create a parent with two leaf children
        // Two spheres: create a parent with two leaf children
        int current = (int)bvhTree.bvhNodes.size();
//      int current = (int)bvhTree.bvhNodes.size();
        // Reserve space for parent node
        // Reserve space for parent node
        bvhTree.bvhNodes.emplace_back(BVHNode{});
//      bvhTree.bvhNodes.emplace_back(BVHNode{});

        // Create left and right leaf nodes
        // Create left and right leaf nodes
        int childIndexL = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start + 0].aabb3d, .shapeIndex = start + 0, .childIndexL = -1, .childIndexR = -1 });
        int childIndexR = (int)bvhTree.bvhNodes.size();
        bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start + 1].aabb3d, .shapeIndex = start + 1, .childIndexL = -1, .childIndexR = -1 });

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
        // Multiple spheres: recursive split
        // Multiple spheres: recursive split
        int axis = RandomInt(0, 2);
//      int axis = RandomInt(0, 2);
        std::function<bool(const sphere&, const sphere&)> comparator;
//      std::function<bool(const sphere&, const sphere&)> comparator;
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
        std::sort(std::begin(bvhTree.spheres) + start,
                  std::begin(bvhTree.spheres) + cease, comparator);
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
//  Calculate the centroid of a sphere's AABB along a specific axis
//  Calculate the centroid of a sphere's AABB along a specific axis
    static inline float GetCentroid(const sphere& sphere, const Axis& axis)
//  static inline float GetCentroid(const sphere& sphere, const Axis& axis)
{
        switch (axis)
        {
        case Axis::X:
            return (sphere.aabb3d.intervalAxisX.min + sphere.aabb3d.intervalAxisX.max) / 2.0f;
        case Axis::Y:
            return (sphere.aabb3d.intervalAxisY.min + sphere.aabb3d.intervalAxisY.max) / 2.0f;
        case Axis::Z:
            return (sphere.aabb3d.intervalAxisZ.min + sphere.aabb3d.intervalAxisZ.max) / 2.0f;
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

        // Base case: create a leaf node with a single sphere
        // Base case: create a leaf node with a single sphere
        if (objectSpan == 1)
//      if (objectSpan == 1)
        {
            int current = (int)bvhTree.bvhNodes.size();
//          int current = (int)bvhTree.bvhNodes.size();
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
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
            // Sort spheres based on centroid along the current axis
            // Sort spheres based on centroid along the current axis
            std::function<bool(const sphere&        , const sphere&        )> comparator = [axis]
                              (const sphere& sphere1, const sphere& sphere2)
                        ->bool{ return GetCentroid(sphere1, Axis(axis))
                         <             GetCentroid(sphere2, Axis(axis));
                              };
            std::sort(std::begin(bvhTree.spheres) + start ,
                      std::begin(bvhTree.spheres) + cease , comparator);

            // Compute cumulative AABB3Ds from the left!
            // Compute cumulative AABB3Ds from the left!
            std::vector<AABB3D> lAABB3Ds(objectSpan);
//          std::vector<AABB3D> lAABB3Ds(objectSpan);
            lAABB3Ds[0].intervalAxisX.min = bvhTree.spheres[start].aabb3d.intervalAxisX.min;
            lAABB3Ds[0].intervalAxisX.max = bvhTree.spheres[start].aabb3d.intervalAxisX.max;
            lAABB3Ds[0].intervalAxisY.min = bvhTree.spheres[start].aabb3d.intervalAxisY.min;
            lAABB3Ds[0].intervalAxisY.max = bvhTree.spheres[start].aabb3d.intervalAxisY.max;
            lAABB3Ds[0].intervalAxisZ.min = bvhTree.spheres[start].aabb3d.intervalAxisZ.min;
            lAABB3Ds[0].intervalAxisZ.max = bvhTree.spheres[start].aabb3d.intervalAxisZ.max;
            for (int i = 1; i < objectSpan; ++i)
//          for (int i = 1; i < objectSpan; ++i)
            {
                Union(lAABB3Ds[i - 1], bvhTree.spheres[start + i].aabb3d, lAABB3Ds[i]);
//              Union(lAABB3Ds[i - 1], bvhTree.spheres[start + i].aabb3d, lAABB3Ds[i]);
            }

            // Compute cumulative AABB3Ds from the right
            // Compute cumulative AABB3Ds from the right
            std::vector<AABB3D> rAABB3Ds(objectSpan);
//          std::vector<AABB3D> rAABB3Ds(objectSpan);
            int r1 = objectSpan - 1;
//          int r1 = objectSpan - 1;
            int r2 = cease      - 1;
//          int r2 = cease      - 1;
            rAABB3Ds[r1].intervalAxisX.min = bvhTree.spheres[r2].aabb3d.intervalAxisX.min;
            rAABB3Ds[r1].intervalAxisX.max = bvhTree.spheres[r2].aabb3d.intervalAxisX.max;
            rAABB3Ds[r1].intervalAxisY.min = bvhTree.spheres[r2].aabb3d.intervalAxisY.min;
            rAABB3Ds[r1].intervalAxisY.max = bvhTree.spheres[r2].aabb3d.intervalAxisY.max;
            rAABB3Ds[r1].intervalAxisZ.min = bvhTree.spheres[r2].aabb3d.intervalAxisZ.min;
            rAABB3Ds[r1].intervalAxisZ.max = bvhTree.spheres[r2].aabb3d.intervalAxisZ.max;
            for (int i = objectSpan - 2; i >= 0; --i)
//          for (int i = objectSpan - 2; i >= 0; --i)
            {
                Union(bvhTree.spheres[start + i].aabb3d, rAABB3Ds[i + 1], rAABB3Ds[i]);
//              Union(bvhTree.spheres[start + i].aabb3d, rAABB3Ds[i + 1], rAABB3Ds[i]);
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
            bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
//          bvhTree.bvhNodes.emplace_back(BVHNode{ .aabb3d = bvhTree.spheres[start].aabb3d, .shapeIndex = start, .childIndexL = -1, .childIndexR = -1, });
            return current;
//          return current;
        }

        // Apply the best split
        // Apply the best split
        std::function<bool(const sphere&        , const sphere&        )> bestComparator = [bestAxis]
                          (const sphere& sphere1, const sphere& sphere2)
                    ->bool{      return GetCentroid(sphere1, bestAxis)
                     <                  GetCentroid(sphere2, bestAxis);
                          };
        std::sort(std::begin(bvhTree.spheres) + start,
                  std::begin(bvhTree.spheres) + cease, bestComparator);
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
    ThreadPool* threadPool = new ThreadPool(225);
//  ThreadPool* threadPool = new ThreadPool(225);

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    int                              samplesPerPixel = 100 ;
//  int                              samplesPerPixel = 100 ;
    float pixelSamplesScale = 1.0f / samplesPerPixel       ;
//  float pixelSamplesScale = 1.0f / samplesPerPixel       ;



//  std::vector<sphere> spheres;
//  std::vector<sphere> spheres;
    BVHTree bvhTree;
//  BVHTree bvhTree;
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.0f, 1.0f, 0.0f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +000.600f,  000.000f, -001.000f }, .dir = { +000.600f,  000.000f, +001.000f }, .time = 0.0f, }, .radius = 000.500f,  });
//  bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.0f, 1.0f, 0.0f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { +000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f,  });
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 1.0f, 0.0f, 1.0f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = { -000.600f,  000.000f, -002.500f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f,  });
//  bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f,  });
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.500f,  });
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    },  .center = { .ori = { -000.600f,  000.000f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 000.400f,  });
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { 0.5f, 0.5f, 0.5f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 },  .center = { .ori = {  000.000f, -100.500f, -001.000f }, .dir = {  000.000f,  000.000f,  000.000f }, .time = 0.0f, }, .radius = 100.000f,  });
    
    for (int i = -5; i < 5; ++i)
    for (int j = -5; j < 5; ++j)
    bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { Random(0.0f, 1.0f), Random(0.0f, 1.0f), Random(0.0f, 1.0f) }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = 0.0f, .materialType = materialType::LambertianDiffuseReflectance1 }, .center = { .ori = { float(i) * 0.5f, -0.4f, float(j) * 0.5f - 1.0f }, .dir = { 0.0f, 0.0f, 0.0f }, .time = 0.0f }, .radius = 0.1f,  });
//  bvhTree.spheres.emplace_back(sphere{  .material = { .albedo = { Random(0.0f, 1.0f), Random(0.0f, 1.0f), Random(0.0f, 1.0f) }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = 0.0f, .materialType = materialType::LambertianDiffuseReflectance1 }, .center = { .ori = { float(i) * 0.5f, -0.4f, float(j) * 0.5f - 1.0f }, .dir = { 0.0f, 0.0f, 0.0f }, .time = 0.0f }, .radius = 0.1f,  });

    for (sphere& sphere : bvhTree.spheres) CalculateAABB3D(sphere);
//  for (sphere& sphere : bvhTree.spheres) CalculateAABB3D(sphere);
    
//  for (sphere& sphere : bvhTree.spheres)
//  {
//      std::cout << "x min:" << sphere.aabb3d.intervalAxisX.min << " " << "x max:" << sphere.aabb3d.intervalAxisX.max << std::endl;
//      std::cout << "y min:" << sphere.aabb3d.intervalAxisY.min << " " << "y max:" << sphere.aabb3d.intervalAxisY.max << std::endl;
//      std::cout << "z min:" << sphere.aabb3d.intervalAxisZ.min << " " << "z max:" << sphere.aabb3d.intervalAxisZ.max << std::endl << std::endl << std::endl;
//  }
//  return 0;

    bvhTree.bvhNodes.reserve(2 * bvhTree.spheres.size() - 1);
//  bvhTree.bvhNodes.reserve(2 * bvhTree.spheres.size() - 1);
    BuildBVHTree(bvhTree, 0,(int)bvhTree.spheres.size());
//  BuildBVHTree(bvhTree, 0,(int)bvhTree.spheres.size());

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
    int imgW = 400     ;
    int imgH = int    (
        imgW /
           aspectRatio);
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
    [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, /* &spheres */ &bvhTree, &rgbs
//  [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, /* &spheres */ &bvhTree, &rgbs
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
        
//      color3 pixelColor = RayColor(ray, spheres);
//      color3 pixelColor = RayColor(ray, spheres);

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
//          pixelColor += RayColor(ray, spheres);
//          pixelColor += RayColor(ray, spheres);
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

    std::cout << executionDuration.count() << "ms" << std::endl;
//  std::cout << executionDuration.count() << "ms" << std::endl;

    return 0       ;
//  return 0       ;
}


// defocus blur = depth of field
// defocus blur = depth of field


// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO
// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast /OPT:ICF /OPT:REF /LTCG /INCREMENTAL:NO
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu