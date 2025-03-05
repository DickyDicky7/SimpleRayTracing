#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include "ThreadPool.h"
#include <chrono>
#include <random>

inline double LinearSpaceToGammasSpace(double linearSpaceComponent) { if    (linearSpaceComponent > 0) { return std::sqrt(linearSpaceComponent); } return 0.0; }
inline double GammasSpaceToLinearSpace(double gammasSpaceComponent) { return gammasSpaceComponent *                       gammasSpaceComponent ;               }

    constexpr double positiveInfinity = +std::numeric_limits<double>::infinity();
//  constexpr double positiveInfinity = +std::numeric_limits<double>::infinity();
    constexpr double negativeInfinity = -std::numeric_limits<double>::infinity();
//  constexpr double negativeInfinity = -std::numeric_limits<double>::infinity();

inline double Random()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
//  static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator ;
//  static std::mt19937 generator ;
    return distribution(generator);
//  return distribution(generator);
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

    inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }
//  inline static vec3 Reflect(const vec3& incomingVector, const vec3& normal) { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }



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
    LambertianDiffuseReflectance = 0,
//  LambertianDiffuseReflectance = 0,
    Metal = 1,
//  Metal = 1,
};


struct material
{
    color3 albedo; float scatteredProbability = 1.0; materialType materialType;
//  color3 albedo; float scatteredProbability = 1.0; materialType materialType;    
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

    case materialType::LambertianDiffuseReflectance:
//  case materialType::LambertianDiffuseReflectance:
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
        break;
//      break;

    case materialType::Metal:
//  case materialType::Metal:
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
        rayHitResult temporaryRayHitResult = std::move(RayHit(sphere, ray, interval { rayT.min, closestTSoFar }));
//      rayHitResult temporaryRayHitResult = std::move(RayHit(sphere, ray, interval { rayT.min, closestTSoFar }));
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
    const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { -10.0, +10.0 });
//  const rayHitResult& rayHitResult = RayHit(sphere, ray, interval { -10.0, +10.0 });
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
    const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { 0.001, positiveInfinity });
//  const rayHitResult& rayHitResult = RayHit(spheres, ray, interval { 0.001, positiveInfinity });
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
    spheres.emplace_back(sphere{ { +000.600,  000.000, -001.000 }, 000.500, { { 0.0, 1.0, 0.0 }, 1.0, materialType::LambertianDiffuseReflectance } });
    spheres.emplace_back(sphere{ { -000.600,  000.000, -001.000 }, 000.500, { { 0.8, 0.8, 0.8 }, 1.0, materialType::Metal                        } });
    spheres.emplace_back(sphere{ {  000.000, -100.500, -001.000 }, 100.000, { { 0.5, 0.5, 0.5 }, 1.0, materialType::LambertianDiffuseReflectance } });

    double aspectRatio = 16.0 / 9.0;
    int imgW = 400     ;
    int imgH = int    (
        imgW /
           aspectRatio);
    imgH = std::max(imgH, 1);
//  imgH = std::max(imgH, 1);


    double viewportH = 2.0;
    double viewportW = viewportH * (double(imgW) / imgH);
//  double viewportW = viewportH * (double(imgW) / imgH);
    double focalLength = 1.0;
//  double focalLength = 1.0;
    point3 cameraCenter { 0, 0, 0, };
//  point3 cameraCenter { 0, 0, 0, };


    vec3 viewportU {};
    viewportU.x = +viewportW;
    viewportU.y = 0.0;
    viewportU.z = 0.0;
    vec3 viewportV {};
    viewportV.x = 0.0;
    viewportV.y = -viewportH;
    viewportV.z = 0.0;


    vec3 fromPixelToPixelDeltaU = viewportU / imgW;
    vec3 fromPixelToPixelDeltaV = viewportV / imgH;



    point3 viewportTL = cameraCenter - vec3 { 0.0, 0.0, focalLength } - viewportU / 2.0 - viewportV / 2.0;
//  point3 viewportTL = cameraCenter - vec3 { 0.0, 0.0, focalLength } - viewportU / 2.0 - viewportV / 2.0;
    point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5 + fromPixelToPixelDeltaV * 0.5;
//  point3 pixel00Coord = viewportTL + fromPixelToPixelDeltaU * 0.5 + fromPixelToPixelDeltaV * 0.5;


    std::ofstream PPMFile(GetCurrentDateTime());
//  std::ofstream PPMFile(GetCurrentDateTime());
    PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";
//  PPMFile << "P3\n" << imgW << " " << imgH << "\n255\n";

    for (int pixelY = 0; pixelY < imgH; ++pixelY)
    {
#ifdef _DEBUG
    std::clog << "Progress: " << pixelY << "\n";
//  std::clog << "Progress: " << pixelY << "\n";
#endif
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
            vec3 rayDirection = pixelSampleCenter - cameraCenter;
//          vec3 rayDirection = pixelSampleCenter - cameraCenter;
            ray  ray{ cameraCenter, rayDirection };
//          ray  ray{ cameraCenter, rayDirection };
            pixelColor += RayColor(ray, spheres);
//          pixelColor += RayColor(ray, spheres);
        }
        pixelColor *= pixelSamplesScale;
//      pixelColor *= pixelSamplesScale;

        static const interval intensity { 0.000 , 0.999 };
        int ir = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.x)));
        int ig = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.y)));
        int ib = int(256 * intensity.Clamp(LinearSpaceToGammasSpace(pixelColor.z)));
        
        PPMFile << std::setw(3) << ir << " ";
        PPMFile << std::setw(3) << ig << " ";
        PPMFile << std::setw(3) << ib << " ";
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


