#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include "ThreadPool.h"
#include <chrono>

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
    double length_squared() const { return std::pow (x , 2)
                                         + std::pow (y , 2)
                                         + std::pow (z , 2)           ; }
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



struct RayHitSphereResult
{
    double minT; bool hitted;
//  double minT; bool hitted;
};


static RayHitSphereResult RayHitSphere(const point3& sphereCenter
                                      ,      double  sphereRadius
                                      ,const ray   & ray
                                      )
{
    const vec3& fromSphereCenterToRayOrigin = sphereCenter - ray.ori;
//  const vec3& fromSphereCenterToRayOrigin = sphereCenter - ray.ori;
    const double& a = dot(ray.dir, ray.dir);
//  const double& a = dot(ray.dir, ray.dir);
    const double& b = -2.0 * dot(ray.dir, fromSphereCenterToRayOrigin);
//  const double& b = -2.0 * dot(ray.dir, fromSphereCenterToRayOrigin);
    const double& c = dot(fromSphereCenterToRayOrigin, fromSphereCenterToRayOrigin) - std::pow(sphereRadius, 2.0);
//  const double& c = dot(fromSphereCenterToRayOrigin, fromSphereCenterToRayOrigin) - std::pow(sphereRadius, 2.0);
    const double& discriminant = std::pow(b, 2.0) - 4.0 * a * c;
//  const double& discriminant = std::pow(b, 2.0) - 4.0 * a * c;
    RayHitSphereResult
    rayHitSphereResult {};
    rayHitSphereResult.hitted = discriminant >= 0.0;
    if (discriminant < 0)
    {
        rayHitSphereResult.minT = -1.0;
//      rayHitSphereResult.minT = -1.0;
    }
    else
    {
        rayHitSphereResult.minT = (-b - std::sqrt(discriminant)) / (2.0 * a);
//      rayHitSphereResult.minT = (-b - std::sqrt(discriminant)) / (2.0 * a);
    }
    
    return rayHitSphereResult;
//  return rayHitSphereResult;
}



static color3 RayColor(const ray& ray)
{
    point3 sphereCenter { 0.0, 0.0, -1.0 };
//  point3 sphereCenter { 0.0, 0.0, -1.0 };
    double sphereRadius =            0.5  ;
//  double sphereRadius =            0.5  ;
    const RayHitSphereResult& rayHitSphereResult = RayHitSphere(sphereCenter , sphereRadius , ray);
//  const RayHitSphereResult& rayHitSphereResult = RayHitSphere(sphereCenter , sphereRadius , ray);
    if (rayHitSphereResult.hitted)
//  if (rayHitSphereResult.hitted)
    {
        const vec3& normal = normalize(ray.Marching(rayHitSphereResult.minT) - sphereCenter);
//      const vec3& normal = normalize(ray.Marching(rayHitSphereResult.minT) - sphereCenter);
        return color3 { normal.x + 1.0, normal.y + 1.0, normal.z + 1.0, } * 0.5;
//      return color3 { normal.x + 1.0, normal.y + 1.0, normal.z + 1.0, } * 0.5;
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

    const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();
//  const std::chrono::steady_clock::time_point& startTime = std::chrono::high_resolution_clock::now();

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
        point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
//      point3 pixelCenter = pixel00Coord + fromPixelToPixelDeltaU * pixelX + fromPixelToPixelDeltaV * pixelY;
        vec3 rayDirection = pixelCenter - cameraCenter;
//      vec3 rayDirection = pixelCenter - cameraCenter;
        ray  ray  { pixelCenter, rayDirection };
//      ray  ray  { pixelCenter, rayDirection };

//      double r = double(pixelX) / (imgW - 1);
//      double g = double(pixelY) / (imgH - 1);
//      double b = 0.00;
//      color3 pixelColor { r, g, b, };
//      color3 pixelColor { r, g, b, };

        color3 pixelColor = RayColor(ray);
//      color3 pixelColor = RayColor(ray);
        
        int ir = int(255.999 * pixelColor.x);
        int ig = int(255.999 * pixelColor.y);
        int ib = int(255.999 * pixelColor.z);
        
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


