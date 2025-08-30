#pragma once
#ifndef VEC_3_H
#define VEC_3_H


    #include <cmath>
//  #include <cmath>
    #include <algorithm>
//  #include <algorithm>
struct Vec2;


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
    Vec3& operator+=(const float& v)
    {
        x += v;
        y += v;
        z += v;
        return *this;
    }
    Vec3& operator-=(const float& v)
    {
        x -= v;
        y -= v;
        z -= v;
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


    template <char C>
//  template <char C>
    constexpr float Get() const
//  constexpr float Get() const
    {
        if constexpr (C == 'x') return x;
        if constexpr (C == 'y') return y;
        if constexpr (C == 'z') return z;
        // This static_assert will give a clean compile error for bad characters
        // This static_assert will give a clean compile error for bad characters
        static_assert(C == 'x' || C == 'y' || C == 'z', "Vec3 Get   () Invalid swizzle character.");
//      static_assert(C == 'x' || C == 'y' || C == 'z', "Vec3 Get   () Invalid swizzle character.");
        return 0.0f; // Should not be reached
//      return 0.0f; // Should not be reached
    }


/*
    template <char C>
//  template <char C>
    constexpr float& GetRef() const
//  constexpr float& GetRef() const
    {
        if constexpr (C == 'x') return *x;
        if constexpr (C == 'y') return *y;
        if constexpr (C == 'z') return *z;
        // This static_assert will give a clean compile error for bad characters
        // This static_assert will give a clean compile error for bad characters
        static_assert(C == 'x' || C == 'y' || C == 'z', "Vec3 GetRef() Invalid swizzle character.");
//      static_assert(C == 'x' || C == 'y' || C == 'z', "Vec3 GetRef() Invalid swizzle character.");
        return 0.0f; // Should not be reached
//      return 0.0f; // Should not be reached
    }
*/


    template<char C1, char C2         > constexpr Vec2 Swizzle() const { return Vec2{ Get<C1>(), Get<C2>(),           }; }
//  template<char C1, char C2         > constexpr Vec2 Swizzle() const { return Vec2{ Get<C1>(), Get<C2>(),           }; }
    template<char C1, char C2, char C3> constexpr Vec3 Swizzle() const { return Vec3{ Get<C1>(), Get<C2>(), Get<C3>() }; }
//  template<char C1, char C2, char C3> constexpr Vec3 Swizzle() const { return Vec3{ Get<C1>(), Get<C2>(), Get<C3>() }; }


/*
    template<char C1, char C2> Vec3& operator+=(const Vec2& rhs) { GetRef<C1>() += rhs.x; GetRef<C2>() += rhs.y; return *this; }
    template<char C1, char C2> Vec3& operator-=(const Vec2& rhs) { GetRef<C1>() -= rhs.x; GetRef<C2>() -= rhs.y; return *this; }
    template<char C1, char C2> Vec3& operator*=(const Vec2& rhs) { GetRef<C1>() *= rhs.x; GetRef<C2>() *= rhs.y; return *this; }
    template<char C1, char C2> Vec3& operator/=(const Vec2& rhs) { GetRef<C1>() /= rhs.x; GetRef<C2>() /= rhs.y; return *this; }
*/
};


using Point3 = Vec3;
using Color3 = Vec3;


static inline Vec3 operator+(const Vec3& u, const Vec3& v) { return Vec3 { u.x + v.x, u.y + v.y, u.z + v.z }; }
static inline Vec3 operator-(const Vec3& u, const Vec3& v) { return Vec3 { u.x - v.x, u.y - v.y, u.z - v.z }; }
static inline Vec3 operator*(const Vec3& u, const Vec3& v) { return Vec3 { u.x * v.x, u.y * v.y, u.z * v.z }; }
static inline Vec3 operator/(const Vec3& u, const Vec3& v) { return Vec3 { u.x / v.x, u.y / v.y, u.z / v.z }; }


static inline Vec3 operator+(const Vec3& u, float t) { return Vec3 { u.x + t, u.y + t, u.z + t }; }
static inline Vec3 operator-(const Vec3& u, float t) { return Vec3 { u.x - t, u.y - t, u.z - t }; }
static inline Vec3 operator*(const Vec3& u, float t) { return Vec3 { u.x * t, u.y * t, u.z * t }; }
static inline Vec3 operator/(const Vec3& u, float t) { return Vec3 { u.x / t, u.y / t, u.z / t }; }


static inline Vec3 operator+(float t, const Vec3& u) { return Vec3 { t + u.x, t + u.y, t + u.z }; }
static inline Vec3 operator-(float t, const Vec3& u) { return Vec3 { t - u.x, t - u.y, t - u.z }; }
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


  inline static Vec3 Abs(const Vec3& v) { return { std::fabsf(v.x), std::fabsf(v.y), std::fabsf(v.z) }; }
//inline static Vec3 Abs(const Vec3& v) { return { std::fabsf(v.x), std::fabsf(v.y), std::fabsf(v.z) }; }


  inline static Vec3 Exp(const Vec3& v) { return { std::expf(v.x), std::expf(v.y), std::expf(v.z) }; }
//inline static Vec3 Exp(const Vec3& v) { return { std::expf(v.x), std::expf(v.y), std::expf(v.z) }; }


  inline static Vec3 Min(const Vec3& v1, const Vec3& v2) { return { std::fminf(v1.x, v2.x), std::fminf(v1.y, v2.y), std::fminf(v1.z, v2.z) }; }
//inline static Vec3 Min(const Vec3& v1, const Vec3& v2) { return { std::fminf(v1.x, v2.x), std::fminf(v1.y, v2.y), std::fminf(v1.z, v2.z) }; }


  inline static Vec3 Max(const Vec3& v1, const Vec3& v2) { return { std::fmaxf(v1.x, v2.x), std::fmaxf(v1.y, v2.y), std::fmaxf(v1.z, v2.z) }; }
//inline static Vec3 Max(const Vec3& v1, const Vec3& v2) { return { std::fmaxf(v1.x, v2.x), std::fmaxf(v1.y, v2.y), std::fmaxf(v1.z, v2.z) }; }


  inline static Vec3 Min(const Vec3& v, float f) { return { std::fminf(v.x, f), std::fminf(v.y, f), std::fminf(v.z, f) }; }
//inline static Vec3 Min(const Vec3& v, float f) { return { std::fminf(v.x, f), std::fminf(v.y, f), std::fminf(v.z, f) }; }


  inline static Vec3 Max(const Vec3& v, float f) { return { std::fmaxf(v.x, f), std::fmaxf(v.y, f), std::fmaxf(v.z, f) }; }
//inline static Vec3 Max(const Vec3& v, float f) { return { std::fmaxf(v.x, f), std::fmaxf(v.y, f), std::fmaxf(v.z, f) }; }


  inline static Vec3 Min(float f, const Vec3& v) { return { std::fminf(f, v.x), std::fminf(f, v.y), std::fminf(f, v.z) }; }
//inline static Vec3 Min(float f, const Vec3& v) { return { std::fminf(f, v.x), std::fminf(f, v.y), std::fminf(f, v.z) }; }


  inline static Vec3 Max(float f, const Vec3& v) { return { std::fmaxf(f, v.x), std::fmaxf(f, v.y), std::fmaxf(f, v.z) }; }
//inline static Vec3 Max(float f, const Vec3& v) { return { std::fmaxf(f, v.x), std::fmaxf(f, v.y), std::fmaxf(f, v.z) }; }


  inline static Vec3 Pow(const Vec3& v, float f) { return { std::powf(v.x, f), std::powf(v.y, f), std::powf(v.z, f) }; }
//inline static Vec3 Pow(const Vec3& v, float f) { return { std::powf(v.x, f), std::powf(v.y, f), std::powf(v.z, f) }; }


  inline static float Length(const Vec3& v) { return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
//inline static float Length(const Vec3& v) { return std::sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }


  inline static float LengthSquared(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
//inline static float LengthSquared(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }


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


    inline static float Length2(const Vec3& v) { Vec3 p = v;                p = p * p;                       return std::sqrtf(p.x + p.y + p.z             ); }
//  inline static float Length2(const Vec3& v) { Vec3 p = v;                p = p * p;                       return std::sqrtf(p.x + p.y + p.z             ); }
    inline static float Length6(const Vec3& v) { Vec3 p = v; p = p * p * p; p = p * p;                       return std:: powf(p.x + p.y + p.z, 1.0f / 6.0f); }
//  inline static float Length6(const Vec3& v) { Vec3 p = v; p = p * p * p; p = p * p;                       return std:: powf(p.x + p.y + p.z, 1.0f / 6.0f); }
    inline static float Length8(const Vec3& v) { Vec3 p = v;                p = p * p; p = p * p; p = p * p; return std:: powf(p.x + p.y + p.z, 1.0f / 8.0f); }
//  inline static float Length8(const Vec3& v) { Vec3 p = v;                p = p * p; p = p * p; p = p * p; return std:: powf(p.x + p.y + p.z, 1.0f / 8.0f); }


    inline static Vec3 Round(const Vec3& v) { return { std::roundf(v.x), std::roundf(v.y), std::roundf(v.z), }; }
//  inline static Vec3 Round(const Vec3& v) { return { std::roundf(v.x), std::roundf(v.y), std::roundf(v.z), }; }


    inline static Vec3 Clamp(const Vec3& v, float min, float max) { return { std::clamp<float>(v.x, min, max), std::clamp<float>(v.y, min, max), std::clamp<float>(v.z, min, max), }; }
//  inline static Vec3 Clamp(const Vec3& v, float min, float max) { return { std::clamp<float>(v.x, min, max), std::clamp<float>(v.y, min, max), std::clamp<float>(v.z, min, max), }; }


    inline static Vec3 Clamp(const Vec3& v, const Vec3& min, const Vec3& max) { return { std::clamp<float>(v.x, min.x, max.x), std::clamp<float>(v.y, min.y, max.y), std::clamp<float>(v.z, min.z, max.z), }; }
//  inline static Vec3 Clamp(const Vec3& v, const Vec3& min, const Vec3& max) { return { std::clamp<float>(v.x, min.x, max.x), std::clamp<float>(v.y, min.y, max.y), std::clamp<float>(v.z, min.z, max.z), }; }


#endif
