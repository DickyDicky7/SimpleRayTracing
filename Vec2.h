#pragma once
#ifndef VEC_2_H
#define VEC_2_H


    #include <cmath>
//  #include <cmath>
    #include <algorithm>
//  #include <algorithm>
struct Vec3;


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
    Vec2& operator+=(const float& v)
    {
        x += v;
        y += v;
        return *this;
    }
    Vec2& operator-=(const float& v)
    {
        x -= v;
        y -= v;
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


    template <char C>
//  template <char C>
    constexpr float Get() const
//  constexpr float Get() const
    {
        if constexpr (C == 'x') return x;
        if constexpr (C == 'y') return y;
        // This static_assert will give a clean compile error for bad characters
        // This static_assert will give a clean compile error for bad characters
        static_assert(C == 'x' || C == 'y', "Vec2 Invalid swizzle character.");
//      static_assert(C == 'x' || C == 'y', "Vec2 Invalid swizzle character.");
        return 0.0f; // Should not be reached
//      return 0.0f; // Should not be reached
    }


    template<char C1, char C2         > constexpr Vec2 Swizzle() const { return Vec2{ Get<C1>(), Get<C2>(),           }; }
//  template<char C1, char C2         > constexpr Vec2 Swizzle() const { return Vec2{ Get<C1>(), Get<C2>(),           }; }
    template<char C1, char C2, char C3> constexpr Vec3 Swizzle() const { return Vec3{ Get<C1>(), Get<C2>(), Get<C3>() }; }
//  template<char C1, char C2, char C3> constexpr Vec3 Swizzle() const { return Vec3{ Get<C1>(), Get<C2>(), Get<C3>() }; }
};


using Point2 = Vec2;
using Color2 = Vec2;


static inline Vec2 operator+(const Vec2& u, const Vec2& v) { return Vec2 { u.x + v.x, u.y + v.y }; }
static inline Vec2 operator-(const Vec2& u, const Vec2& v) { return Vec2 { u.x - v.x, u.y - v.y }; }
static inline Vec2 operator*(const Vec2& u, const Vec2& v) { return Vec2 { u.x * v.x, u.y * v.y }; }
static inline Vec2 operator/(const Vec2& u, const Vec2& v) { return Vec2 { u.x / v.x, u.y / v.y }; }


static inline Vec2 operator+(const Vec2& u, float t) { return Vec2 { u.x + t, u.y + t }; }
static inline Vec2 operator-(const Vec2& u, float t) { return Vec2 { u.x - t, u.y - t }; }
static inline Vec2 operator*(const Vec2& u, float t) { return Vec2 { u.x * t, u.y * t }; }
static inline Vec2 operator/(const Vec2& u, float t) { return Vec2 { u.x / t, u.y / t }; }


static inline Vec2 operator+(float t, const Vec2& u) { return Vec2 { t + u.x, t + u.y }; }
static inline Vec2 operator-(float t, const Vec2& u) { return Vec2 { t - u.x, t - u.y }; }
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


  inline static Vec2 Abs(const Vec2& v) { return { std::fabsf(v.x), std::fabsf(v.y), }; }
//inline static Vec2 Abs(const Vec2& v) { return { std::fabsf(v.x), std::fabsf(v.y), }; }


  inline static Vec2 Min(const Vec2& v1, const Vec2& v2) { return { std::fminf(v1.x, v2.x), std::fminf(v1.y, v2.y), }; }
//inline static Vec2 Min(const Vec2& v1, const Vec2& v2) { return { std::fminf(v1.x, v2.x), std::fminf(v1.y, v2.y), }; }


  inline static Vec2 Max(const Vec2& v1, const Vec2& v2) { return { std::fmaxf(v1.x, v2.x), std::fmaxf(v1.y, v2.y), }; }
//inline static Vec2 Max(const Vec2& v1, const Vec2& v2) { return { std::fmaxf(v1.x, v2.x), std::fmaxf(v1.y, v2.y), }; }


  inline static Vec2 Min(const Vec2& v, float f) { return { std::fminf(v.x, f), std::fminf(v.y, f), }; }
//inline static Vec2 Min(const Vec2& v, float f) { return { std::fminf(v.x, f), std::fminf(v.y, f), }; }


  inline static Vec2 Max(const Vec2& v, float f) { return { std::fmaxf(v.x, f), std::fmaxf(v.y, f), }; }
//inline static Vec2 Max(const Vec2& v, float f) { return { std::fmaxf(v.x, f), std::fmaxf(v.y, f), }; }


  inline static Vec2 Min(float f, const Vec2& v) { return { std::fminf(f, v.x), std::fminf(f, v.y), }; }
//inline static Vec2 Min(float f, const Vec2& v) { return { std::fminf(f, v.x), std::fminf(f, v.y), }; }


  inline static Vec2 Max(float f, const Vec2& v) { return { std::fmaxf(f, v.x), std::fmaxf(f, v.y), }; }
//inline static Vec2 Max(float f, const Vec2& v) { return { std::fmaxf(f, v.x), std::fmaxf(f, v.y), }; }


  inline static float Length(const Vec2& v) { return std::sqrtf(v.x * v.x + v.y * v.y); }
//inline static float Length(const Vec2& v) { return std::sqrtf(v.x * v.x + v.y * v.y); }


  inline static float LengthSquared(const Vec2& v) { return v.x * v.x + v.y * v.y; }
//inline static float LengthSquared(const Vec2& v) { return v.x * v.x + v.y * v.y; }


inline static float BlendLinear(      float startValue,       float ceaseValue,       float ratio);
inline static Vec2  BlendLinear(const Vec2& startValue, const Vec2& ceaseValue,       float ratio)
{
       return Vec2
       {
                    BlendLinear(startValue.x, ceaseValue.x, ratio  ),
                    BlendLinear(startValue.y, ceaseValue.y, ratio  ),
       };
}
inline static Vec2  BlendLinear(const Vec2& startValue, const Vec2& ceaseValue, const Vec2& ratio)
{
       return Vec2
       {
                    BlendLinear(startValue.x, ceaseValue.x, ratio.x),
                    BlendLinear(startValue.y, ceaseValue.y, ratio.y),       
       };
}


    inline static Vec2 Round(const Vec2& v) { return { std::roundf(v.x), std::roundf(v.y), }; }
//  inline static Vec2 Round(const Vec2& v) { return { std::roundf(v.x), std::roundf(v.y), }; }


    inline static Vec2 Clamp(const Vec2& v, float min, float max) { return { std::clamp<float>(v.x, min, max), std::clamp<float>(v.y, min, max), }; }
//  inline static Vec2 Clamp(const Vec2& v, float min, float max) { return { std::clamp<float>(v.x, min, max), std::clamp<float>(v.y, min, max), }; }


    inline static Vec2 Clamp(const Vec2& v, const Vec2& min, const Vec2& max) { return { std::clamp<float>(v.x, min.x, max.x), std::clamp<float>(v.y, min.y, max.y), }; }
//  inline static Vec2 Clamp(const Vec2& v, const Vec2& min, const Vec2& max) { return { std::clamp<float>(v.x, min.x, max.x), std::clamp<float>(v.y, min.y, max.y), }; }


#endif
