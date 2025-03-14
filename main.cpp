#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#define _USE_MATH_DEFINES

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

  static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { if    (linearSpaceComponent > 0.0f) { return std::sqrt(linearSpaceComponent); } return 0.0f; }
//static inline float LinearSpaceToGammasSpace(float linearSpaceComponent) { if    (linearSpaceComponent > 0.0f) { return std::sqrt(linearSpaceComponent); } return 0.0f; }
  static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return gammasSpaceComponent *                          gammasSpaceComponent ;                }
//static inline float GammasSpaceToLinearSpace(float gammasSpaceComponent) { return gammasSpaceComponent *                          gammasSpaceComponent ;                }

    constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
//  constexpr float positiveInfinity = +std::numeric_limits<float>::infinity();
    constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();
//  constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();

static
inline float Random()
{
    thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//  thread_local static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    thread_local static std::mt19937 generator ;
//  thread_local static std::mt19937 generator ;
    return distribution(generator);
//  return distribution(generator);

//    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
////  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
//    static std::mt19937 generator ;
////  static std::mt19937 generator ;
//    return distribution(generator);
////  return distribution(generator);
}

static
inline float Random(float min, float max)
{
    return min + (max - min) * Random();
//  return min + (max - min) * Random();
}


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
};

    const interval interval::empty    { positiveInfinity, negativeInfinity };
//  const interval interval::empty    { positiveInfinity, negativeInfinity };
    const interval interval::universe { negativeInfinity, positiveInfinity };
//  const interval interval::universe { negativeInfinity, positiveInfinity };


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

    point3 Marching(float t) const { return ori + dir * t; }
//  point3 Marching(float t) const { return ori + dir * t; }
};






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
        return r0 + (1.0f - r0) * std::powf((1.0f - cosine), 5.0f);
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
    point3 center;
    float radius;
    material material;
//  material material;

};


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
        if (scatteredDirection.NearZero())
//      if (scatteredDirection.NearZero())
        {
            materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//          materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
        }
        else
        {
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
        }
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
        if (scatteredDirection.NearZero())
//      if (scatteredDirection.NearZero())
        {
            materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
//          materialScatteredResult.scatteredRay.dir = rayHitResult.normal;
        }
        else
        {
            materialScatteredResult.scatteredRay.dir = scatteredDirection;
//          materialScatteredResult.scatteredRay.dir = scatteredDirection;
        }
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
        if (rayHitResult.isFrontFace) { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
//      if (rayHitResult.isFrontFace) { ratioOfEtaiOverEtat = 1.0f / rayHitResult.material.refractionIndex; }
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
                          ,const interval&& rayT
//                        ,const interval&& rayT
                          )
{
    const vec3& fromSphereCenterToRayOrigin = sphere.center - ray.ori;
//  const vec3& fromSphereCenterToRayOrigin = sphere.center - ray.ori;
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

        const vec3& outwardNormal = (rayHitResult.at - sphere.center) / sphere.radius;
//      const vec3& outwardNormal = (rayHitResult.at - sphere.center) / sphere.radius;

        rayHitResult.SetFaceNormal(ray, outwardNormal);
//      rayHitResult.SetFaceNormal(ray, outwardNormal);
    }
    
    return rayHitResult;
//  return rayHitResult;
}



inline
static rayHitResult RayHit(const std::vector<sphere>& spheres, const ray& ray, const interval&& rayT)
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
        if (temporaryRayHitResult.hitted)
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
    if (rayHitResult.hitted)
//  if (rayHitResult.hitted)
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

        if (rayHitResult.hitted)
//      if (rayHitResult.hitted)
        {
            const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);
//          const materialScatteredResult& materialScatteredResult = Scatter(currentRay, rayHitResult);

            if (!materialScatteredResult.isScattered)
//          if (!materialScatteredResult.isScattered)
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
//  ThreadPool threadPool;
//  ThreadPool threadPool;

    int                              samplesPerPixel = 1000;
    float pixelSamplesScale = 1.0f / samplesPerPixel       ;

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    std::vector<sphere> spheres;
    spheres.emplace_back(sphere{ .center = { +000.600f,  000.000f, -001.000f }, .radius = 000.500f, .material = { .albedo = { 0.0f, 1.0f, 0.0f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });
    spheres.emplace_back(sphere{ .center = { -000.600f,  000.000f, -002.500f }, .radius = 000.500f, .material = { .albedo = { 0.0f, 0.0f, 1.0f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });
//  spheres.emplace_back(sphere{ .center = { -000.600f,  000.000f, -001.000f }, .radius = 000.500f, .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = { -000.600f,  000.000f, -001.000f }, .radius = 000.500f, .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex =                                                   GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = { -000.600f,  000.000f, -001.000f }, .radius = 000.400f, .material = { .albedo = { 0.8f, 0.8f, 0.8f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = {  000.000f, -100.500f, -001.000f }, .radius = 100.000f, .material = { .albedo = { 0.5f, 0.5f, 0.5f }, .scatteredProbability = 1.0f, .fuzz = 1.0f, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });

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

    float defocusAngle = 0.05f * M_PI; float focusDistance = (lookAt - lookFrom).length();
//  float defocusAngle = 0.05f * M_PI; float focusDistance = (lookAt - lookFrom).length();
//  float defocusAngle = 0.00f * M_PI; float focusDistance = 10.0f;
//  float defocusAngle = 0.00f * M_PI; float focusDistance = 10.0f;
    vec3 defocusDiskRadiusU;
    vec3 defocusDiskRadiusV;


    float vFOV = M_PI / 8.0f;
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
    std::vector<std::thread> threads;
//  std::vector<std::thread> threads;

    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
#ifdef _DEBUG
    std::clog << "Progress: " << pixelY << "\n";
//  std::clog << "Progress: " << pixelY << "\n";
#endif
    threads.emplace_back(
//  threads.emplace_back(
    [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, &spheres, &rgbs
//  [ pixelY, &imgW, &samplesPerPixel, &pixel00Coord, &fromPixelToPixelDeltaU, &fromPixelToPixelDeltaV, &cameraCenter, &defocusAngle, &defocusDiskRadiusU, &defocusDiskRadiusV, &pixelSamplesScale, &spheres, &rgbs
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
            if (defocusAngle > 0.0f) { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
//          if (defocusAngle > 0.0f) { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
            vec3 rayDirection = pixelSampleCenter - rayOrigin;
//          vec3 rayDirection = pixelSampleCenter - rayOrigin;
            ray  ray{ rayOrigin, rayDirection };
//          ray  ray{ rayOrigin, rayDirection };
            pixelColor += RayColor(ray, spheres);
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


    for (std::thread& t : threads) { t.join(); }
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


// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast
// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL /fp:fast
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu