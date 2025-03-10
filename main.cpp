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

inline double LinearSpaceToGammasSpace(double linearSpaceComponent) { if    (linearSpaceComponent > 0) { return std::sqrt(linearSpaceComponent); } return 0.0; }
inline double GammasSpaceToLinearSpace(double gammasSpaceComponent) { return gammasSpaceComponent *                       gammasSpaceComponent ;               }

    constexpr double positiveInfinity = +std::numeric_limits<double>::infinity();
//  constexpr double positiveInfinity = +std::numeric_limits<double>::infinity();
    constexpr double negativeInfinity = -std::numeric_limits<double>::infinity();
//  constexpr double negativeInfinity = -std::numeric_limits<double>::infinity();

inline double Random()
{
    thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
//  thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local static std::mt19937 generator ;
//  thread_local static std::mt19937 generator ;
    return distribution(generator);
//  return distribution(generator);

//    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
////  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
//    static std::mt19937 generator ;
////  static std::mt19937 generator ;
//    return distribution(generator);
////  return distribution(generator);
}

inline double Random(double min, double max)
{
    return min + (max - min) * Random();
//  return min + (max - min) * Random();
}


struct interval
{
    double min = positiveInfinity;
    double max = negativeInfinity;

    static const interval empty;
//  static const interval empty;
    static const interval universe;
//  static const interval universe;

    bool Contains (double x) const { return min <= x && x <= max; }
//  bool Contains (double x) const { return min <= x && x <= max; }
    bool Surrounds(double x) const { return min <  x && x <  max; }
//  bool Surrounds(double x) const { return min <  x && x <  max; }

    double Clamp(double x) const
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
    double x;
    double y;
    double z;

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
    vec3& operator*=(const double& v)
    {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }
    vec3& operator/=(const double& v)
    {
        x /= v;
        y /= v;
        z /= v;
        return *this;
    }

    double length        () const { return std::sqrt(length_squared()); }
    double length_squared() const { return x * x
                                         + y * y
                                         + z * z                      ; }

    bool NearZero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        // Return true if the vector is close to zero in all dimensions.
        double s = 1e-8;
//      double s = 1e-8;
        return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
//      return (std::fabs(x) < s) && (std::fabs(y) < s) && (std::fabs(z) < s);
    }
};


using point3 = vec3;
using color3 = vec3;

inline vec3 operator+(const vec3& u, const vec3& v) { return vec3 { u.x + v.x, u.y + v.y, u.z + v.z }; }
inline vec3 operator-(const vec3& u, const vec3& v) { return vec3 { u.x - v.x, u.y - v.y, u.z - v.z }; }
inline vec3 operator*(const vec3& u, const vec3& v) { return vec3 { u.x * v.x, u.y * v.y, u.z * v.z }; }
inline vec3 operator/(const vec3& u, const vec3& v) { return vec3 { u.x / v.x, u.y / v.y, u.z / v.z }; }

inline vec3 operator*(const vec3& u, double t) { return vec3 { u.x * t, u.y * t, u.z * t }; }
inline vec3 operator/(const vec3& u, double t) { return vec3 { u.x / t, u.y / t, u.z / t }; }

inline vec3 operator*(double t, const vec3& u) { return vec3 { u.x * t, u.y * t, u.z * t }; }
inline vec3 operator/(double t, const vec3& u) { return vec3 { u.x / t, u.y / t, u.z / t }; }


inline double dot  (const vec3& u, const vec3& v)
{
return u.x * v.x
     + u.y * v.y
     + u.z * v.z;
}

inline vec3   cross(const vec3& u, const vec3& v)
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

    point3 Marching(double t) const { return ori + dir * t; }
//  point3 Marching(double t) const { return ori + dir * t; }
};






    static vec3 GenRandom(                      ) { return vec3 { Random(        ), Random(        ), Random(        ) }; }
    static vec3 GenRandom(double min, double max) { return vec3 { Random(min, max), Random(min, max), Random(min, max) }; }
    static vec3 GenRandomUnitVector()
//  static vec3 GenRandomUnitVector()
    {
        while (true)
        {
            const vec3& p = GenRandom(-1.0, +1.0);
//          const vec3& p = GenRandom(-1.0, +1.0);
            const double& pLengthSquared = p.length_squared();
//          const double& pLengthSquared = p.length_squared();
            if (pLengthSquared <= 1
            &&  pLengthSquared >  1e-160)
            {
                return p / std::sqrt(pLengthSquared);
            }
        }
    }
    static vec3 GenRandomUnitVectorOnHemisphere(const vec3& normal)
//  static vec3 GenRandomUnitVectorOnHemisphere(const vec3& normal)
    {
        const vec3& randomUnitVector = GenRandomUnitVector();
//      const vec3& randomUnitVector = GenRandomUnitVector();
        if (dot(randomUnitVector, normal) > 0.0)
//      if (dot(randomUnitVector, normal) > 0.0)
        {
            return  randomUnitVector;
        }
        else
        {
            return -randomUnitVector;
        }
    }
    static vec3 GenRandomPointInsideNormalizedDisk()
    {
        while (true)
//      while (true)
        {
            point3 point { .x = Random(-1.0 , +1.0), .y = Random(-1.0 , +1.0), .z = 0.0 };
//          point3 point { .x = Random(-1.0 , +1.0), .y = Random(-1.0 , +1.0), .z = 0.0 };
            if (point.length_squared() < 1)
//          if (point.length_squared() < 1)
            {
                return point;
//              return point;
            }
        }
    }
    static vec3 DefocusDiskSample(const point3& diskCenter, const vec3& defocusDiskRadiusU, const vec3& defocusDiskRadiusV)
//  static vec3 DefocusDiskSample(const point3& diskCenter, const vec3& defocusDiskRadiusU, const vec3& defocusDiskRadiusV)
    {
        point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
//      point3 randomPointInsideNormalizedDisk = GenRandomPointInsideNormalizedDisk();
        return diskCenter + randomPointInsideNormalizedDisk.x * defocusDiskRadiusU
                          + randomPointInsideNormalizedDisk.y * defocusDiskRadiusV;
    }

    inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }
//  inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }

    inline static vec3 Refract(const vec3& incomingVector, const vec3& normal, double ratioOfEtaiOverEtat)
    {
        const double& cosTheta = std::fmin(dot(-incomingVector, normal), 1.0);
//      const double& cosTheta = std::fmin(dot(-incomingVector, normal), 1.0);
        const vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
//      const vec3& refractedRayPerpendicular = ratioOfEtaiOverEtat * (incomingVector + cosTheta * normal);
        const vec3& refractedRayParallel = -std::sqrt(std::fabs(1.0 - refractedRayPerpendicular.length_squared())) * normal;
//      const vec3& refractedRayParallel = -std::sqrt(std::fabs(1.0 - refractedRayPerpendicular.length_squared())) * normal;
        return refractedRayPerpendicular + refractedRayParallel;
//      return refractedRayPerpendicular + refractedRayParallel;
    }

    inline static double Reflectance(double cosine, double ratioOfEtaiOverEtat)
    {
        // Use Schlick's approximation for reflectance.
        // Use Schlick's approximation for reflectance.
        double r0 = (1.0 - ratioOfEtaiOverEtat) / (1.0 + ratioOfEtaiOverEtat);
//      double r0 = (1.0 - ratioOfEtaiOverEtat) / (1.0 + ratioOfEtaiOverEtat);
        r0 = r0 * r0;
//      r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
//      return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
    }

static double BlendLinear(      double  startValue,       double  ceaseValue,       double  ratio)
{
return (1.0 - ratio) * startValue
            + ratio  * ceaseValue;
}
static vec3   BlendLinear(const vec3  & startValue, const vec3  & ceaseValue,       double  ratio)
{
return vec3 {
              BlendLinear(startValue.x, ceaseValue.x, ratio),
              BlendLinear(startValue.y, ceaseValue.y, ratio),
              BlendLinear(startValue.z, ceaseValue.z, ratio),
            };
}
static vec3   BlendLinear(const vec3  & startValue, const vec3  & ceaseValue, const vec3  & ratio)
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


constexpr static double GetRefractionIndex(materialDielectric materialDielectric)
{
    switch (materialDielectric)
    {
    case materialDielectric::GLASS  : return 1.500000; break;
    case materialDielectric::WATER  : return 1.333000; break;
    case materialDielectric::AIR    : return 1.000293; break;
    case materialDielectric::DIAMOND: return 2.400000; break;
                             default: return 0.000000; break;
    }
}


struct material
{
    color3 albedo; double scatteredProbability; double fuzz; double refractionIndex; materialType materialType;
//  color3 albedo; double scatteredProbability; double fuzz; double refractionIndex; materialType materialType;    
};

struct materialScatteredResult
{
    ray scatteredRay; color3 attenuation; bool isScattered;
//  ray scatteredRay; color3 attenuation; bool isScattered;
};


struct sphere
{
    point3 center;
    double radius;
    material material;
//  material material;

};


struct rayHitResult
{
    material material; point3 at; vec3 normal; double minT; bool hitted; bool isFrontFace;
//  material material; point3 at; vec3 normal; double minT; bool hitted; bool isFrontFace;

    void SetFaceNormal(const ray& ray, const vec3& outwardNormal)
//  void SetFaceNormal(const ray& ray, const vec3& outwardNormal)
    {
            isFrontFace = dot(ray.dir, outwardNormal) < 0;
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





static materialScatteredResult Scatter(const ray& rayIn, const rayHitResult& rayHitResult)
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
        materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0;
//      materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0;
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
        materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0;
//      materialScatteredResult.isScattered = dot(reflectionScatteredDirection, rayHitResult.normal) > 0;
        }
        break;
//      break;



    case materialType::Dielectric:
        {
        materialScatteredResult.attenuation = color3 { 1.0, 1.0, 1.0 };
//      materialScatteredResult.attenuation = color3 { 1.0, 1.0, 1.0 };
        double ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
//      double ratioOfEtaiOverEtat = rayHitResult.material.refractionIndex;
        if (rayHitResult.isFrontFace) { ratioOfEtaiOverEtat = 1.0 / rayHitResult.material.refractionIndex; }
//      if (rayHitResult.isFrontFace) { ratioOfEtaiOverEtat = 1.0 / rayHitResult.material.refractionIndex; }
        vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);
//      vec3 normalizedIncomingRayDirection = normalize(rayIn.dir);

        double cosTheta = std::fmin(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0);
//      double cosTheta = std::fmin(dot(-normalizedIncomingRayDirection, rayHitResult.normal), 1.0);
        double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
//      double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
        bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0 || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
//      bool notAbleToRefract = sinTheta * ratioOfEtaiOverEtat > 1.0 || Reflectance(cosTheta, ratioOfEtaiOverEtat) > Random();
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




static rayHitResult RayHit(const sphere& sphere
                          ,const ray   & ray
                          ,const interval&& rayT
//                        ,const interval&& rayT
                          )
{
    const vec3& fromSphereCenterToRayOrigin = sphere.center - ray.ori;
//  const vec3& fromSphereCenterToRayOrigin = sphere.center - ray.ori;
    const double& a = ray.dir.length_squared();
//  const double& a = ray.dir.length_squared();
    const double& h = dot(ray.dir, fromSphereCenterToRayOrigin);
//  const double& h = dot(ray.dir, fromSphereCenterToRayOrigin);
    const double& c = fromSphereCenterToRayOrigin.length_squared() - sphere.radius * sphere.radius;
//  const double& c = fromSphereCenterToRayOrigin.length_squared() - sphere.radius * sphere.radius;
    const double& discriminant = h * h - a * c;
//  const double& discriminant = h * h - a * c;
    rayHitResult
    rayHitResult { sphere.material };
//  rayHitResult.hitted = discriminant >= 0.0;
//  rayHitResult.hitted = discriminant >= 0.0;
    if (discriminant < 0)
    {
        rayHitResult.hitted = false;
//      rayHitResult.hitted = false;
    }
    else
    {
        double sqrtDiscriminant = std::sqrt(discriminant);
//      double sqrtDiscriminant = std::sqrt(discriminant);

        double t = (h - sqrtDiscriminant) / a;
//      double t = (h - sqrtDiscriminant) / a;

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



static rayHitResult RayHit(const std::vector<sphere>& spheres, const ray& ray, const interval&& rayT)
{
    rayHitResult finalRayHitResult{};
//  rayHitResult finalRayHitResult{};
    double closestTSoFar = rayT.max;
//  double closestTSoFar = rayT.max;
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

static color3 RayColor(const ray& ray)
{
    sphere sphere{ { 0.0, 0.0, -1.0 }, 0.5, };
//  sphere sphere{ { 0.0, 0.0, -1.0 }, 0.5, };
    const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { .min = -10.0, .max = +10.0 });
//  const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { .min = -10.0, .max = +10.0 });
    if (rayHitResult.hitted)
//  if (rayHitResult.hitted)
    {
        return color3 { rayHitResult.normal.x + 1.0, rayHitResult.normal.y + 1.0, rayHitResult.normal.z + 1.0, } * 0.5;
//      return color3 { rayHitResult.normal.x + 1.0, rayHitResult.normal.y + 1.0, rayHitResult.normal.z + 1.0, } * 0.5;
    }

    const vec3& normalizedRayDirection = normalize(ray.dir);
//  const vec3& normalizedRayDirection = normalize(ray.dir);
    const double& ratio = 0.5 * (normalizedRayDirection.y + 1.0);
//  const double& ratio = 0.5 * (normalizedRayDirection.y + 1.0);
    return BlendLinear(color3{ 1.0, 1.0, 1.0, }, color3{ 0.5, 0.7, 1.0, }, ratio);
//  return BlendLinear(color3{ 1.0, 1.0, 1.0, }, color3{ 0.5, 0.7, 1.0, }, ratio);
}





static color3 RayColor(const ray& ray, const std::vector<sphere>& spheres, int recursiveDepth = 50)
{
    if (recursiveDepth <= 0)
    {
        return color3 {};
//      return color3 {};
    }
    const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { .min = 0.001, .max = positiveInfinity });
//  const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { .min = 0.001, .max = positiveInfinity });
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
//      return 0.5 * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, spheres, --recursiveDepth);
//      return 0.5 * RayColor({ rayHitResult.at, rayHitResult.normal + GenRandomUnitVectorOnHemisphere(rayHitResult.normal), }, spheres, --recursiveDepth);
//      return color3 { rayHitResult.normal.x + 1.0, rayHitResult.normal.y + 1.0, rayHitResult.normal.z + 1.0, } * 0.5;
//      return color3 { rayHitResult.normal.x + 1.0, rayHitResult.normal.y + 1.0, rayHitResult.normal.z + 1.0, } * 0.5;
    }

    const vec3& normalizedRayDirection = normalize(ray.dir);
//  const vec3& normalizedRayDirection = normalize(ray.dir);
    const double& ratio = 0.5 * (normalizedRayDirection.y + 1.0);
//  const double& ratio = 0.5 * (normalizedRayDirection.y + 1.0);
    return BlendLinear(color3{ 1.0, 1.0, 1.0, }, color3{ 0.5, 0.7, 1.0, }, ratio);
//  return BlendLinear(color3{ 1.0, 1.0, 1.0, }, color3{ 0.5, 0.7, 1.0, }, ratio);
}

int main()
{
//  ThreadPool threadPool;
//  ThreadPool threadPool;

    int                              samplesPerPixel = 1000;
    double pixelSamplesScale = 1.0 / samplesPerPixel       ;

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

    std::vector<sphere> spheres;
    spheres.emplace_back(sphere{ .center = { +000.600,  000.000, -001.000 }, .radius = 000.500, .material = { .albedo = { 0.0, 1.0, 0.0 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });
    spheres.emplace_back(sphere{ .center = { -000.600,  000.000, -002.500 }, .radius = 000.500, .material = { .albedo = { 0.0, 0.0, 1.0 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });
//  spheres.emplace_back(sphere{ .center = { -000.600,  000.000, -001.000 }, .radius = 000.500, .material = { .albedo = { 0.8, 0.8, 0.8 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = { -000.600,  000.000, -001.000 }, .radius = 000.500, .material = { .albedo = { 0.8, 0.8, 0.8 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex =                                                   GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = { -000.600,  000.000, -001.000 }, .radius = 000.400, .material = { .albedo = { 0.8, 0.8, 0.8 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex = GetRefractionIndex(materialDielectric::AIR    ) / GetRefractionIndex(materialDielectric::GLASS), .materialType = materialType::Dielectric                    } });
    spheres.emplace_back(sphere{ .center = {  000.000, -100.500, -001.000 }, .radius = 100.000, .material = { .albedo = { 0.5, 0.5, 0.5 }, .scatteredProbability = 1.0, .fuzz = 1.0, .refractionIndex = GetRefractionIndex(materialDielectric::NOTHING)                                                , .materialType = materialType::LambertianDiffuseReflectance1 } });

    double aspectRatio = 16.0 / 9.0;
    int imgW = 400     ;
    int imgH = int    (
        imgW /
           aspectRatio);
    imgH = std::max(imgH, 1);
//  imgH = std::max(imgH, 1);

    const point3 lookFrom { .x = -2.0, .y = +2.0, .z = +1.0 };
    const point3 lookAt   { .x = -0.6, .y = +0.0, .z = -1.0 };
//  const point3 lookAt   { .x = +0.0, .y = +0.0, .z = -1.0 };
    const point3 viewUp   { .x = +0.0, .y = +1.0, .z = +0.0 };

    vec3 cameraU; // x
    vec3 cameraV; // y
    vec3 cameraW; // z

    double defocusAngle = 0.05 * M_PI; double focusDistance = (lookAt - lookFrom).length();
//  double defocusAngle = 0.05 * M_PI; double focusDistance = (lookAt - lookFrom).length();
//  double defocusAngle = 0.00 * M_PI; double focusDistance = 10.0;
//  double defocusAngle = 0.00 * M_PI; double focusDistance = 10.0;
    vec3 defocusDiskRadiusU;
    vec3 defocusDiskRadiusV;


    double vFOV = M_PI / 8.0;
    double hFOV = M_PI / 2.0;
    double h = std::tan(vFOV / 2.0);
    double w = std::tan(hFOV / 2.0);

    double focalLength = (lookAt - lookFrom).length();
//  double focalLength = (lookAt - lookFrom).length();


    cameraW = normalize(lookFrom - lookAt); cameraU = normalize(cross(viewUp, cameraW)); cameraV = cross(cameraW, cameraU);
//  cameraW = normalize(lookFrom - lookAt); cameraU = normalize(cross(viewUp, cameraW)); cameraV = cross(cameraW, cameraU);

    double defocusRadius = focusDistance * std::tan(defocusAngle / 2.0);
//  double defocusRadius = focusDistance * std::tan(defocusAngle / 2.0);
    defocusDiskRadiusU = cameraU * defocusRadius;
    defocusDiskRadiusV = cameraV * defocusRadius;


    double viewportH = 2.0 * h * /* focalLength */ focusDistance;
//  double viewportH = 2.0 * h * /* focalLength */ focusDistance;
    double viewportW = viewportH * (double(imgW) / imgH);
//  double viewportW = viewportH * (double(imgW) / imgH);

    point3 cameraCenter /* { 0, 0, 0, } */ = lookFrom;
//  point3 cameraCenter /* { 0, 0, 0, } */ = lookFrom;



//  vec3 viewportU {};
//  viewportU.x = +viewportW;
//  viewportU.y = 0.0;
//  viewportU.z = 0.0;
//  vec3 viewportV {};
//  viewportV.x = 0.0;
//  viewportV.y = -viewportH;
//  viewportV.z = 0.0;

    vec3 viewportU = viewportW *  cameraU;
    vec3 viewportV = viewportH * -cameraV;


    vec3 fromPixelToPixelDeltaU = viewportU / imgW;
    vec3 fromPixelToPixelDeltaV = viewportV / imgH;



    point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0 - viewportV / 2.0;
//  point3 viewportTL = cameraCenter - (focusDistance /* focalLength */ * cameraW) - viewportU / 2.0 - viewportV / 2.0;
    point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5 + fromPixelToPixelDeltaV * 0.5;
//  point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5 + fromPixelToPixelDeltaV * 0.5;


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

//      double r = double(pixelX) / (imgW - 1);
//      double g = double(pixelY) / (imgH - 1);
//      double b = 0.00;
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
            vec3 sampleOffset{ Random() - 0.5, Random() - 0.5, 0.0 };
//          vec3 sampleOffset{ Random() - 0.5, Random() - 0.5, 0.0 };
            point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          point3 pixelSampleCenter = pixel00Coord + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//          vec3 rayDirection = pixelSampleCenter - cameraCenter;
//          vec3 rayDirection = pixelSampleCenter - cameraCenter;
            vec3 rayOrigin = cameraCenter;
//          vec3 rayOrigin = cameraCenter;
            if (defocusAngle > 0.0) { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
//          if (defocusAngle > 0.0) { rayOrigin = DefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV); }
            vec3 rayDirection = pixelSampleCenter - rayOrigin;
//          vec3 rayDirection = pixelSampleCenter - rayOrigin;
            ray  ray{ rayOrigin, rayDirection };
//          ray  ray{ rayOrigin, rayDirection };
            pixelColor += RayColor(ray, spheres);
//          pixelColor += RayColor(ray, spheres);
        }
        pixelColor *= pixelSamplesScale;
//      pixelColor *= pixelSamplesScale;

        static const interval intensity { 0.000 , 0.999 };
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


// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL
// @ON: /O2 /Ob2 /Oi /Ot /Oy /GT /GL
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu
// OFF: /Z7 /Zi /Zl /RTC1 /RTCsu /RTCs /RTCu