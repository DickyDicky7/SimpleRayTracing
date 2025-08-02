// MIT License
// MIT License
//
// Copyright(c) 2023 Jordan Peck (jordan.me2@gmail.com)
// Copyright(c) 2023 Jordan Peck (jordan.me2@gmail.com)
// Copyright(c) 2023 Contributors
// Copyright(c) 2023 Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// SOFTWARE.
//
// VERSION: 1.1.1
// VERSION: 1.1.1
// https://github.com/Auburn/FastNoiseLite
// https://github.com/Auburn/FastNoiseLite

    #pragma once
//  #pragma once
    #ifndef FASTNOISELITE_H
//  #ifndef FASTNOISELITE_H
    #define FASTNOISELITE_H
//  #define FASTNOISELITE_H

    #include <cmath>
//  #include <cmath>

    class FastNoiseLite
//  class FastNoiseLite
{
    public:
//  public:
    enum NoiseType
//  enum NoiseType
    {
        NoiseType_OpenSimplex2,
//      NoiseType_OpenSimplex2,
        NoiseType_OpenSimplex2S,
//      NoiseType_OpenSimplex2S,
        NoiseType_Cellular,
//      NoiseType_Cellular,
        NoiseType_Perlin,
//      NoiseType_Perlin,
        NoiseType_ValueCubic,
//      NoiseType_ValueCubic,
        NoiseType_Value,
//      NoiseType_Value,
    };

    enum RotationType3D
//  enum RotationType3D
    {
        RotationType3D_None,
//      RotationType3D_None,
        RotationType3D_ImproveXYPlanes,
//      RotationType3D_ImproveXYPlanes,
        RotationType3D_ImproveXZPlanes,
//      RotationType3D_ImproveXZPlanes,
    };

    enum FractalType
//  enum FractalType
    {
        FractalType_None,
//      FractalType_None,
        FractalType_FBm,
//      FractalType_FBm,
        FractalType_Ridged,
//      FractalType_Ridged,
        FractalType_PingPong,
//      FractalType_PingPong,
        FractalType_DomainWarpProgressive,
//      FractalType_DomainWarpProgressive,
        FractalType_DomainWarpIndependent,
//      FractalType_DomainWarpIndependent,
    };

    enum CellularDistanceFunction
//  enum CellularDistanceFunction
    {
        CellularDistanceFunction_Euclidean,
//      CellularDistanceFunction_Euclidean,
        CellularDistanceFunction_EuclideanSq,
//      CellularDistanceFunction_EuclideanSq,
        CellularDistanceFunction_Manhattan,
//      CellularDistanceFunction_Manhattan,
        CellularDistanceFunction_Hybrid,
//      CellularDistanceFunction_Hybrid,
    };

    enum CellularReturnType
//  enum CellularReturnType
    {
        CellularReturnType_CellValue,
//      CellularReturnType_CellValue,
        CellularReturnType_Distance,
//      CellularReturnType_Distance,
        CellularReturnType_Distance2,
//      CellularReturnType_Distance2,
        CellularReturnType_Distance2Add,
//      CellularReturnType_Distance2Add,
        CellularReturnType_Distance2Sub,
//      CellularReturnType_Distance2Sub,
        CellularReturnType_Distance2Mul,
//      CellularReturnType_Distance2Mul,
        CellularReturnType_Distance2Div,
//      CellularReturnType_Distance2Div,
    };

    enum DomainWarpType
//  enum DomainWarpType
    {
        DomainWarpType_OpenSimplex2,
//      DomainWarpType_OpenSimplex2,
        DomainWarpType_OpenSimplex2Reduced,
//      DomainWarpType_OpenSimplex2Reduced,
        DomainWarpType_BasicGrid,
//      DomainWarpType_BasicGrid,
    };

    /// <summary> Create new FastNoise object with optional seed </summary>
    /// <summary> Create new FastNoise object with optional seed </summary>
    FastNoiseLite(int seed = 1337)
//  FastNoiseLite(int seed = 1337)
    {
        mSeed = seed;
//      mSeed = seed;
        mFrequency = 0.01f;
//      mFrequency = 0.01f;
        mNoiseType = NoiseType_OpenSimplex2;
//      mNoiseType = NoiseType_OpenSimplex2;
        mRotationType3D = RotationType3D_None;
//      mRotationType3D = RotationType3D_None;
        mTransformType3D = TransformType3D_DefaultOpenSimplex2;
//      mTransformType3D = TransformType3D_DefaultOpenSimplex2;

        mFractalType = FractalType_None;
//      mFractalType = FractalType_None;
        mOctaves = 3;
//      mOctaves = 3;
        mLacunarity = 2.0f;
//      mLacunarity = 2.0f;
        mGain = 0.5f;
//      mGain = 0.5f;
        mWeightedStrength = 0.0f;
//      mWeightedStrength = 0.0f;
        mPingPongStrength = 2.0f;
//      mPingPongStrength = 2.0f;

        mFractalBounding = 1 / 1.75f;
//      mFractalBounding = 1 / 1.75f;

        mCellularDistanceFunction = CellularDistanceFunction_EuclideanSq;
//      mCellularDistanceFunction = CellularDistanceFunction_EuclideanSq;
        mCellularReturnType = CellularReturnType_Distance;
//      mCellularReturnType = CellularReturnType_Distance;
        mCellularJitterModifier = 1.0f;
//      mCellularJitterModifier = 1.0f;

        mDomainWarpType = DomainWarpType_OpenSimplex2;
//      mDomainWarpType = DomainWarpType_OpenSimplex2;
        mWarpTransformType3D = TransformType3D_DefaultOpenSimplex2;
//      mWarpTransformType3D = TransformType3D_DefaultOpenSimplex2;
        mDomainWarpAmp = 1.0f;
//      mDomainWarpAmp = 1.0f;
    }

    /// <summary> Sets seed used for all noise types </summary> <remarks> Default: 1337 </remarks>
    /// <summary> Sets seed used for all noise types </summary> <remarks> Default: 1337 </remarks>
    void SetSeed(int seed) { mSeed = seed; }
//  void SetSeed(int seed) { mSeed = seed; }

    /// <summary> Sets frequency for all noise types </summary> <remarks> Default: 0.01 </remarks>
    /// <summary> Sets frequency for all noise types </summary> <remarks> Default: 0.01 </remarks>
    void SetFrequency(float frequency) { mFrequency = frequency; }
//  void SetFrequency(float frequency) { mFrequency = frequency; }

    /// <summary> Sets noise algorithm used for GetNoise(...) </summary> <remarks> Default: OpenSimplex2 </remarks>
    /// <summary> Sets noise algorithm used for GetNoise(...) </summary> <remarks> Default: OpenSimplex2 </remarks>
    void SetNoiseType(NoiseType noiseType)
//  void SetNoiseType(NoiseType noiseType)
    {
        mNoiseType = noiseType;
//      mNoiseType = noiseType;
        UpdateTransformType3D();
//      UpdateTransformType3D();
    }

    /// <summary> Sets domain rotation type for 3D Noise and 3D DomainWarp. Can aid in reducing directional artifacts when sampling a 2D plane in 3D </summary> <remarks> Default: None </remarks>
    /// <summary> Sets domain rotation type for 3D Noise and 3D DomainWarp. Can aid in reducing directional artifacts when sampling a 2D plane in 3D </summary> <remarks> Default: None </remarks>
    void SetRotationType3D(RotationType3D rotationType3D)
//  void SetRotationType3D(RotationType3D rotationType3D)
    {
        mRotationType3D = rotationType3D;
//      mRotationType3D = rotationType3D;
        UpdateTransformType3D();
//      UpdateTransformType3D();
        UpdateWarpTransformType3D();
//      UpdateWarpTransformType3D();
    }

    /// <summary> Sets method for combining octaves in all fractal noise types </summary> <remarks> Default: None Note: FractalType_DomainWarp... only affects DomainWarp(...) </remarks>
    /// <summary> Sets method for combining octaves in all fractal noise types </summary> <remarks> Default: None Note: FractalType_DomainWarp... only affects DomainWarp(...) </remarks>
    void SetFractalType(FractalType fractalType) { mFractalType = fractalType; }
//  void SetFractalType(FractalType fractalType) { mFractalType = fractalType; }

    /// <summary> Sets octave count for all fractal noise types  </summary> <remarks> Default: 3 </remarks>
    /// <summary> Sets octave count for all fractal noise types  </summary> <remarks> Default: 3 </remarks>
    void SetFractalOctaves(int octaves)
//  void SetFractalOctaves(int octaves)
    {
        mOctaves = octaves;
//      mOctaves = octaves;
        CalculateFractalBounding();
//      CalculateFractalBounding();
    }

    /// <summary> Sets octave lacunarity for all fractal noise types </summary> <remarks> Default: 2.0 </remarks>
    /// <summary> Sets octave lacunarity for all fractal noise types </summary> <remarks> Default: 2.0 </remarks>
    void SetFractalLacunarity(float lacunarity) { mLacunarity = lacunarity; }
//  void SetFractalLacunarity(float lacunarity) { mLacunarity = lacunarity; }

    /// <summary> Sets octave gain for all fractal noise types </summary> <remarks> Default: 0.5 </remarks>
    /// <summary> Sets octave gain for all fractal noise types </summary> <remarks> Default: 0.5 </remarks>
    void SetFractalGain(float gain)
//  void SetFractalGain(float gain)
    {
        mGain = gain;
//      mGain = gain;
        CalculateFractalBounding();
//      CalculateFractalBounding();
    }

    /// <summary> Sets octave weighting for all none DomainWarp fratal types </summary> <remarks> Default: 0.0 Note: Keep between 0...1 to maintain -1...1 output bounding </remarks>
    /// <summary> Sets octave weighting for all none DomainWarp fratal types </summary> <remarks> Default: 0.0 Note: Keep between 0...1 to maintain -1...1 output bounding </remarks>
    void SetFractalWeightedStrength(float weightedStrength) { mWeightedStrength = weightedStrength; }
//  void SetFractalWeightedStrength(float weightedStrength) { mWeightedStrength = weightedStrength; }

    /// <summary> Sets strength of the fractal ping pong effect </summary> <remarks> Default: 2.0 </remarks>
    /// <summary> Sets strength of the fractal ping pong effect </summary> <remarks> Default: 2.0 </remarks>
    void SetFractalPingPongStrength(float pingPongStrength) { mPingPongStrength = pingPongStrength; }
//  void SetFractalPingPongStrength(float pingPongStrength) { mPingPongStrength = pingPongStrength; }


    /// <summary> Sets distance function used in cellular noise calculations </summary> <remarks> Default: Distance </remarks>
    /// <summary> Sets distance function used in cellular noise calculations </summary> <remarks> Default: Distance </remarks>
    void SetCellularDistanceFunction(CellularDistanceFunction cellularDistanceFunction) { mCellularDistanceFunction = cellularDistanceFunction; }
//  void SetCellularDistanceFunction(CellularDistanceFunction cellularDistanceFunction) { mCellularDistanceFunction = cellularDistanceFunction; }

    /// <summary> Sets return type from cellular noise calculations </summary> <remarks> Default: EuclideanSq </remarks>
    /// <summary> Sets return type from cellular noise calculations </summary> <remarks> Default: EuclideanSq </remarks>
    void SetCellularReturnType(CellularReturnType cellularReturnType) { mCellularReturnType = cellularReturnType; }
//  void SetCellularReturnType(CellularReturnType cellularReturnType) { mCellularReturnType = cellularReturnType; }

    /// <summary> Sets the maximum distance a cellular point can move from it's grid position </summary> <remarks> Default: 1.0 Note: Setting this higher than 1 will cause artifacts </remarks> 
    /// <summary> Sets the maximum distance a cellular point can move from it's grid position </summary> <remarks> Default: 1.0 Note: Setting this higher than 1 will cause artifacts </remarks> 
    void SetCellularJitter(float cellularJitter) { mCellularJitterModifier = cellularJitter; }
//  void SetCellularJitter(float cellularJitter) { mCellularJitterModifier = cellularJitter; }


    /// <summary> Sets the warp algorithm when using DomainWarp(...) </summary> <remarks> Default: OpenSimplex2 </remarks>
    /// <summary> Sets the warp algorithm when using DomainWarp(...) </summary> <remarks> Default: OpenSimplex2 </remarks>
    void SetDomainWarpType(DomainWarpType domainWarpType)
//  void SetDomainWarpType(DomainWarpType domainWarpType)
    {
        mDomainWarpType = domainWarpType;
//      mDomainWarpType = domainWarpType;
        UpdateWarpTransformType3D();
//      UpdateWarpTransformType3D();
    }


    /// <summary> Sets the maximum warp distance from original position when using DomainWarp(...) </summary> <remarks> Default: 1.0 </remarks>
    /// <summary> Sets the maximum warp distance from original position when using DomainWarp(...) </summary> <remarks> Default: 1.0 </remarks>
    void SetDomainWarpAmp(float domainWarpAmp) { mDomainWarpAmp = domainWarpAmp; }
//  void SetDomainWarpAmp(float domainWarpAmp) { mDomainWarpAmp = domainWarpAmp; }


    /// <summary> 2D noise at given position using current settings </summary> <returns> Noise output bounded between -1...1 </returns>
    /// <summary> 2D noise at given position using current settings </summary> <returns> Noise output bounded between -1...1 </returns>
    template <typename FNfloat>
//  template <typename FNfloat>
    float GetNoise(FNfloat x, FNfloat y) const
//  float GetNoise(FNfloat x, FNfloat y) const
    {
        Arguments_must_be_floating_point_values<FNfloat>();
//      Arguments_must_be_floating_point_values<FNfloat>();

        TransformNoiseCoordinate(x, y);
//      TransformNoiseCoordinate(x, y);

        switch (mFractalType)
//      switch (mFractalType)
        {
        default:
//      default:
            {
                return GenNoiseSingle(mSeed, x, y);
//              return GenNoiseSingle(mSeed, x, y);
            }
        case FractalType_FBm:
//      case FractalType_FBm:
            {
                return GenFractalFBm(x, y);
//              return GenFractalFBm(x, y);
            }
        case FractalType_Ridged:
//      case FractalType_Ridged:
            {
                return GenFractalRidged(x, y);
//              return GenFractalRidged(x, y);
            }
        case FractalType_PingPong:
//      case FractalType_PingPong:
            {
                return GenFractalPingPong(x, y);
//              return GenFractalPingPong(x, y);
            }
        }
    }

    /// <summary> 3D noise at given position using current settings </summary> <returns> Noise output bounded between -1...1 </returns>
    /// <summary> 3D noise at given position using current settings </summary> <returns> Noise output bounded between -1...1 </returns>
    template <typename FNfloat>
//  template <typename FNfloat>
    float GetNoise(FNfloat x, FNfloat y, FNfloat z) const
//  float GetNoise(FNfloat x, FNfloat y, FNfloat z) const
    {
        Arguments_must_be_floating_point_values<FNfloat>();
//      Arguments_must_be_floating_point_values<FNfloat>();

        TransformNoiseCoordinate(x, y, z);
//      TransformNoiseCoordinate(x, y, z);

        switch (mFractalType)
//      switch (mFractalType)
        {
        default:
//      default:
            {
                return GenNoiseSingle(mSeed, x, y, z);
//              return GenNoiseSingle(mSeed, x, y, z);
            }
        case FractalType_FBm:
//      case FractalType_FBm:
            {
                return GenFractalFBm(x, y, z);
//              return GenFractalFBm(x, y, z);
            }
        case FractalType_Ridged:
//      case FractalType_Ridged:
            {
                return GenFractalRidged(x, y, z);
//              return GenFractalRidged(x, y, z);
            }
        case FractalType_PingPong:
//      case FractalType_PingPong:
            {
                return GenFractalPingPong(x, y, z);
//              return GenFractalPingPong(x, y, z);
            }
        }
    }


    /// <summary> 2D warps the input position using current domain warp settings </summary> <example> Example usage with GetNoise <code>DomainWarp(x, y) noise = GetNoise(x, y)</code> </example>
    /// <summary> 2D warps the input position using current domain warp settings </summary> <example> Example usage with GetNoise <code>DomainWarp(x, y) noise = GetNoise(x, y)</code> </example>
    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarp(FNfloat& x, FNfloat& y) const
//  void DomainWarp(FNfloat& x, FNfloat& y) const
    {
        Arguments_must_be_floating_point_values<FNfloat>();
//      Arguments_must_be_floating_point_values<FNfloat>();

        switch (mFractalType)
//      switch (mFractalType)
        {
        default:
//      default:
            {
                DomainWarpSingle(x, y);
//              DomainWarpSingle(x, y);
            }
            break;
//          break;
        case FractalType_DomainWarpProgressive:
//      case FractalType_DomainWarpProgressive:
            {
                DomainWarpFractalProgressive(x, y);
//              DomainWarpFractalProgressive(x, y);
            }
            break;
//          break;
        case FractalType_DomainWarpIndependent:
//      case FractalType_DomainWarpIndependent:
            {
                DomainWarpFractalIndependent(x, y);
//              DomainWarpFractalIndependent(x, y);
            }
            break;
//          break;
        }
    }

    /// <summary> 3D warps the input position using current domain warp settings </summary> <example> Example usage with GetNoise <code>DomainWarp(x, y, z) noise = GetNoise(x, y, z)</code> </example>
    /// <summary> 3D warps the input position using current domain warp settings </summary> <example> Example usage with GetNoise <code>DomainWarp(x, y, z) noise = GetNoise(x, y, z)</code> </example>
    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarp(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void DomainWarp(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        Arguments_must_be_floating_point_values<FNfloat>();
//      Arguments_must_be_floating_point_values<FNfloat>();

        switch (mFractalType)
//      switch (mFractalType)
        {
        default:
//      default:
            {
                DomainWarpSingle(x, y, z);
//              DomainWarpSingle(x, y, z);
            }
            break;
//          break;
        case FractalType_DomainWarpProgressive:
//      case FractalType_DomainWarpProgressive:
            {
                DomainWarpFractalProgressive(x, y, z);
//              DomainWarpFractalProgressive(x, y, z);
            }
            break;
//          break;
        case FractalType_DomainWarpIndependent:
//      case FractalType_DomainWarpIndependent:
            {
                DomainWarpFractalIndependent(x, y, z);
//              DomainWarpFractalIndependent(x, y, z);
            }
            break;
//          break;
        }
    }

    private:
//  private:
    template <typename T>
//  template <typename T>
    struct Arguments_must_be_floating_point_values;
//  struct Arguments_must_be_floating_point_values;

    enum TransformType3D
//  enum TransformType3D
    {
        TransformType3D_None,
//      TransformType3D_None,
        TransformType3D_ImproveXYPlanes,
//      TransformType3D_ImproveXYPlanes,
        TransformType3D_ImproveXZPlanes,
//      TransformType3D_ImproveXZPlanes,
        TransformType3D_DefaultOpenSimplex2,
//      TransformType3D_DefaultOpenSimplex2,
    };

    int mSeed;
//  int mSeed;
    float mFrequency;
//  float mFrequency;
    NoiseType mNoiseType;
//  NoiseType mNoiseType;
    RotationType3D mRotationType3D;
//  RotationType3D mRotationType3D;
    TransformType3D mTransformType3D;
//  TransformType3D mTransformType3D;

    FractalType mFractalType;
//  FractalType mFractalType;
    int mOctaves;
//  int mOctaves;
    float mLacunarity;
//  float mLacunarity;
    float mGain;
//  float mGain;
    float mWeightedStrength;
//  float mWeightedStrength;
    float mPingPongStrength;
//  float mPingPongStrength;

    float mFractalBounding;
//  float mFractalBounding;

    CellularDistanceFunction mCellularDistanceFunction;
//  CellularDistanceFunction mCellularDistanceFunction;
    CellularReturnType mCellularReturnType;
//  CellularReturnType mCellularReturnType;
    float mCellularJitterModifier;
//  float mCellularJitterModifier;

    DomainWarpType mDomainWarpType;
//  DomainWarpType mDomainWarpType;
    TransformType3D mWarpTransformType3D;
//  TransformType3D mWarpTransformType3D;
    float mDomainWarpAmp;
//  float mDomainWarpAmp;


    template <typename T>
//  template <typename T>
    struct Lookup
//  struct Lookup
    {
        static const T Gradients2D[];
//      static const T Gradients2D[];
        static const T Gradients3D[];
//      static const T Gradients3D[];
        static const T RandVecs2D[];
//      static const T RandVecs2D[];
        static const T RandVecs3D[];
//      static const T RandVecs3D[];
    };

    static float FastMin(float a, float b) { return a < b ? a : b; }
//  static float FastMin(float a, float b) { return a < b ? a : b; }

    static float FastMax(float a, float b) { return a > b ? a : b; }
//  static float FastMax(float a, float b) { return a > b ? a : b; }

    static float FastAbs(float f) { return f < 0 ? -f : f; }
//  static float FastAbs(float f) { return f < 0 ? -f : f; }

    static float FastSqrt(float f) { return sqrtf(f); }
//  static float FastSqrt(float f) { return sqrtf(f); }

    template <typename FNfloat>
//  template <typename FNfloat>
    static int FastFloor(FNfloat f) { return f >= 0 ? (int)f : (int)f - 1; }
//  static int FastFloor(FNfloat f) { return f >= 0 ? (int)f : (int)f - 1; }

    template <typename FNfloat>
//  template <typename FNfloat>
    static int FastRound(FNfloat f) { return f >= 0 ? (int)(f + 0.5f) : (int)(f - 0.5f); }
//  static int FastRound(FNfloat f) { return f >= 0 ? (int)(f + 0.5f) : (int)(f - 0.5f); }

    static float Lerp(float a, float b, float t) { return a + t * (b - a); }
//  static float Lerp(float a, float b, float t) { return a + t * (b - a); }

    static float InterpHermite(float t) { return t * t * (3 - 2 * t); }
//  static float InterpHermite(float t) { return t * t * (3 - 2 * t); }

    static float InterpQuintic(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
//  static float InterpQuintic(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    static float CubicLerp(float a, float b, float c, float d, float t)
//  static float CubicLerp(float a, float b, float c, float d, float t)
    {
        float p = (d - c) - (a - b);
//      float p = (d - c) - (a - b);
        return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b;
//      return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b;
    }

    static float PingPong(float t)
//  static float PingPong(float t)
    {
        t -= (int)(t * 0.5f) * 2;
//      t -= (int)(t * 0.5f) * 2;
        return t < 1 ? t : 2 - t;
//      return t < 1 ? t : 2 - t;
    }

    void CalculateFractalBounding()
//  void CalculateFractalBounding()
    {
        float gain = FastAbs(mGain);
//      float gain = FastAbs(mGain);
        float amp = gain;
//      float amp = gain;
        float ampFractal = 1.0f;
//      float ampFractal = 1.0f;
        for (int i = 1; i < mOctaves; i++)
//      for (int i = 1; i < mOctaves; i++)
        {
            ampFractal += amp;
//          ampFractal += amp;
            amp *= gain;
//          amp *= gain;
        }
        mFractalBounding = 1 / ampFractal;
//      mFractalBounding = 1 / ampFractal;
    }

    // Hashing
    // Hashing
    static const int PrimeX = 501125321;
//  static const int PrimeX = 501125321;
    static const int PrimeY = 1136930381;
//  static const int PrimeY = 1136930381;
    static const int PrimeZ = 1720413743;
//  static const int PrimeZ = 1720413743;

    static int Hash(int seed, int xPrimed, int yPrimed)
//  static int Hash(int seed, int xPrimed, int yPrimed)
    {
        int hash = seed ^ xPrimed ^ yPrimed;
//      int hash = seed ^ xPrimed ^ yPrimed;

        hash *= 0x27d4eb2d;
//      hash *= 0x27d4eb2d;
        return hash;
//      return hash;
    }


    static int Hash(int seed, int xPrimed, int yPrimed, int zPrimed)
//  static int Hash(int seed, int xPrimed, int yPrimed, int zPrimed)
    {
        int hash = seed ^ xPrimed ^ yPrimed ^ zPrimed;
//      int hash = seed ^ xPrimed ^ yPrimed ^ zPrimed;

        hash *= 0x27d4eb2d;
//      hash *= 0x27d4eb2d;
        return hash;
//      return hash;
    }


    static float ValCoord(int seed, int xPrimed, int yPrimed)
//  static float ValCoord(int seed, int xPrimed, int yPrimed)
    {
        int hash = Hash(seed, xPrimed, yPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed);

        hash *= hash;
//      hash *= hash;
        hash ^= hash << 19;
//      hash ^= hash << 19;
        return hash * (1 / 2147483648.0f);
//      return hash * (1 / 2147483648.0f);
    }


    static float ValCoord(int seed, int xPrimed, int yPrimed, int zPrimed)
//  static float ValCoord(int seed, int xPrimed, int yPrimed, int zPrimed)
    {
        int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed, zPrimed);

        hash *= hash;
//      hash *= hash;
        hash ^= hash << 19;
//      hash ^= hash << 19;
        return hash * (1 / 2147483648.0f);
//      return hash * (1 / 2147483648.0f);
    }


    float GradCoord(int seed, int xPrimed, int yPrimed, float xd, float yd) const
//  float GradCoord(int seed, int xPrimed, int yPrimed, float xd, float yd) const
    {
        int hash = Hash(seed, xPrimed, yPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed);
        hash ^= hash >> 15;
//      hash ^= hash >> 15;
        hash &= 127 << 1;
//      hash &= 127 << 1;

        float xg = Lookup<float>::Gradients2D[hash];
//      float xg = Lookup<float>::Gradients2D[hash];
        float yg = Lookup<float>::Gradients2D[hash | 1];
//      float yg = Lookup<float>::Gradients2D[hash | 1];

        return xd * xg + yd * yg;
//      return xd * xg + yd * yg;
    }


    float GradCoord(int seed, int xPrimed, int yPrimed, int zPrimed, float xd, float yd, float zd) const
//  float GradCoord(int seed, int xPrimed, int yPrimed, int zPrimed, float xd, float yd, float zd) const
    {
        int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
        hash ^= hash >> 15;
//      hash ^= hash >> 15;
        hash &= 63 << 2;
//      hash &= 63 << 2;

        float xg = Lookup<float>::Gradients3D[hash];
//      float xg = Lookup<float>::Gradients3D[hash];
        float yg = Lookup<float>::Gradients3D[hash | 1];
//      float yg = Lookup<float>::Gradients3D[hash | 1];
        float zg = Lookup<float>::Gradients3D[hash | 2];
//      float zg = Lookup<float>::Gradients3D[hash | 2];

        return xd * xg + yd * yg + zd * zg;
//      return xd * xg + yd * yg + zd * zg;
    }


    void GradCoordOut(int seed, int xPrimed, int yPrimed, float& xo, float& yo) const
//  void GradCoordOut(int seed, int xPrimed, int yPrimed, float& xo, float& yo) const
    {
        int hash = Hash(seed, xPrimed, yPrimed) & (255 << 1);
//      int hash = Hash(seed, xPrimed, yPrimed) & (255 << 1);

        xo = Lookup<float>::RandVecs2D[hash];
//      xo = Lookup<float>::RandVecs2D[hash];
        yo = Lookup<float>::RandVecs2D[hash | 1];
//      yo = Lookup<float>::RandVecs2D[hash | 1];
    }


    void GradCoordOut(int seed, int xPrimed, int yPrimed, int zPrimed, float& xo, float& yo, float& zo) const
//  void GradCoordOut(int seed, int xPrimed, int yPrimed, int zPrimed, float& xo, float& yo, float& zo) const
    {
        int hash = Hash(seed, xPrimed, yPrimed, zPrimed) & (255 << 2);
//      int hash = Hash(seed, xPrimed, yPrimed, zPrimed) & (255 << 2);

        xo = Lookup<float>::RandVecs3D[hash];
//      xo = Lookup<float>::RandVecs3D[hash];
        yo = Lookup<float>::RandVecs3D[hash | 1];
//      yo = Lookup<float>::RandVecs3D[hash | 1];
        zo = Lookup<float>::RandVecs3D[hash | 2];
//      zo = Lookup<float>::RandVecs3D[hash | 2];
    }


    void GradCoordDual(int seed, int xPrimed, int yPrimed, float xd, float yd, float& xo, float& yo) const
//  void GradCoordDual(int seed, int xPrimed, int yPrimed, float xd, float yd, float& xo, float& yo) const
    {
        int hash = Hash(seed, xPrimed, yPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed);
        int index1 = hash & (127 << 1);
//      int index1 = hash & (127 << 1);
        int index2 = (hash >> 7) & (255 << 1);
//      int index2 = (hash >> 7) & (255 << 1);

        float xg = Lookup<float>::Gradients2D[index1];
//      float xg = Lookup<float>::Gradients2D[index1];
        float yg = Lookup<float>::Gradients2D[index1 | 1];
//      float yg = Lookup<float>::Gradients2D[index1 | 1];
        float value = xd * xg + yd * yg;
//      float value = xd * xg + yd * yg;

        float xgo = Lookup<float>::RandVecs2D[index2];
//      float xgo = Lookup<float>::RandVecs2D[index2];
        float ygo = Lookup<float>::RandVecs2D[index2 | 1];
//      float ygo = Lookup<float>::RandVecs2D[index2 | 1];

        xo = value * xgo;
//      xo = value * xgo;
        yo = value * ygo;
//      yo = value * ygo;
    }


    void GradCoordDual(int seed, int xPrimed, int yPrimed, int zPrimed, float xd, float yd, float zd, float& xo, float& yo, float& zo) const
//  void GradCoordDual(int seed, int xPrimed, int yPrimed, int zPrimed, float xd, float yd, float zd, float& xo, float& yo, float& zo) const
    {
        int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//      int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
        int index1 = hash & (63 << 2);
//      int index1 = hash & (63 << 2);
        int index2 = (hash >> 6) & (255 << 2);
//      int index2 = (hash >> 6) & (255 << 2);

        float xg = Lookup<float>::Gradients3D[index1];
//      float xg = Lookup<float>::Gradients3D[index1];
        float yg = Lookup<float>::Gradients3D[index1 | 1];
//      float yg = Lookup<float>::Gradients3D[index1 | 1];
        float zg = Lookup<float>::Gradients3D[index1 | 2];
//      float zg = Lookup<float>::Gradients3D[index1 | 2];
        float value = xd * xg + yd * yg + zd * zg;
//      float value = xd * xg + yd * yg + zd * zg;

        float xgo = Lookup<float>::RandVecs3D[index2];
//      float xgo = Lookup<float>::RandVecs3D[index2];
        float ygo = Lookup<float>::RandVecs3D[index2 | 1];
//      float ygo = Lookup<float>::RandVecs3D[index2 | 1];
        float zgo = Lookup<float>::RandVecs3D[index2 | 2];
//      float zgo = Lookup<float>::RandVecs3D[index2 | 2];

        xo = value * xgo;
//      xo = value * xgo;
        yo = value * ygo;
//      yo = value * ygo;
        zo = value * zgo;
//      zo = value * zgo;
    }


    // Generic noise gen
    // Generic noise gen

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenNoiseSingle(int seed, FNfloat x, FNfloat y) const
//  float GenNoiseSingle(int seed, FNfloat x, FNfloat y) const
    {
        switch (mNoiseType)
//      switch (mNoiseType)
        {
        case NoiseType_OpenSimplex2:
//      case NoiseType_OpenSimplex2:
            {
                return SingleSimplex(seed, x, y);
//              return SingleSimplex(seed, x, y);
            }
        case NoiseType_OpenSimplex2S:
//      case NoiseType_OpenSimplex2S:
            {
                return SingleOpenSimplex2S(seed, x, y);
//              return SingleOpenSimplex2S(seed, x, y);
            }
        case NoiseType_Cellular:
//      case NoiseType_Cellular:
            {
                return SingleCellular(seed, x, y);
//              return SingleCellular(seed, x, y);
            }
        case NoiseType_Perlin:
//      case NoiseType_Perlin:
            {
                return SinglePerlin(seed, x, y);
//              return SinglePerlin(seed, x, y);
            }
        case NoiseType_ValueCubic:
//      case NoiseType_ValueCubic:
            {
                return SingleValueCubic(seed, x, y);
//              return SingleValueCubic(seed, x, y);
            }
        case NoiseType_Value:
//      case NoiseType_Value:
            {
                return SingleValue(seed, x, y);
//              return SingleValue(seed, x, y);
            }
        default:
//      default:
            {
                return 0;
//              return 0;
            }
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenNoiseSingle(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float GenNoiseSingle(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        switch (mNoiseType)
//      switch (mNoiseType)
        {
        case NoiseType_OpenSimplex2:
//      case NoiseType_OpenSimplex2:
            {
                return SingleOpenSimplex2(seed, x, y, z);
//              return SingleOpenSimplex2(seed, x, y, z);
            }
        case NoiseType_OpenSimplex2S:
//      case NoiseType_OpenSimplex2S:
            {
                return SingleOpenSimplex2S(seed, x, y, z);
//              return SingleOpenSimplex2S(seed, x, y, z);
            }
        case NoiseType_Cellular:
//      case NoiseType_Cellular:
            {
                return SingleCellular(seed, x, y, z);
//              return SingleCellular(seed, x, y, z);
            }
        case NoiseType_Perlin:
//      case NoiseType_Perlin:
            {
                return SinglePerlin(seed, x, y, z);
//              return SinglePerlin(seed, x, y, z);
            }
        case NoiseType_ValueCubic:
//      case NoiseType_ValueCubic:
            {
                return SingleValueCubic(seed, x, y, z);
//              return SingleValueCubic(seed, x, y, z);
            }
        case NoiseType_Value:
//      case NoiseType_Value:
            {
                return SingleValue(seed, x, y, z);
//              return SingleValue(seed, x, y, z);
            }
        default:
//      default:
            {
                return 0;
//              return 0;
            }
        }
    }


    // Noise Coordinate Transforms (frequency, and possible skew or rotation)
    // Noise Coordinate Transforms (frequency, and possible skew or rotation)

    template <typename FNfloat>
//  template <typename FNfloat>
    void TransformNoiseCoordinate(FNfloat& x, FNfloat& y) const
//  void TransformNoiseCoordinate(FNfloat& x, FNfloat& y) const
    {
        x *= mFrequency;
//      x *= mFrequency;
        y *= mFrequency;
//      y *= mFrequency;

        switch (mNoiseType)
//      switch (mNoiseType)
        {
        case NoiseType_OpenSimplex2:
//      case NoiseType_OpenSimplex2:
        case NoiseType_OpenSimplex2S:
//      case NoiseType_OpenSimplex2S:
            {
                const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
//              const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
                const FNfloat F2 = 0.5f * (SQRT3 - 1);
//              const FNfloat F2 = 0.5f * (SQRT3 - 1);
                FNfloat t = (x + y) * F2;
//              FNfloat t = (x + y) * F2;
                x += t;
//              x += t;
                y += t;
//              y += t;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void TransformNoiseCoordinate(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void TransformNoiseCoordinate(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        x *= mFrequency;
//      x *= mFrequency;
        y *= mFrequency;
//      y *= mFrequency;
        z *= mFrequency;
//      z *= mFrequency;

        switch (mTransformType3D)
//      switch (mTransformType3D)
        {
        case TransformType3D_ImproveXYPlanes:
//      case TransformType3D_ImproveXYPlanes:
            {
                FNfloat xy = x + y;
//              FNfloat xy = x + y;
                FNfloat s2 = xy * -(FNfloat)0.211324865405187;
//              FNfloat s2 = xy * -(FNfloat)0.211324865405187;
                z *= (FNfloat)0.577350269189626;
//              z *= (FNfloat)0.577350269189626;
                x += s2 - z;
//              x += s2 - z;
                y = y + s2 - z;
//              y = y + s2 - z;
                z += xy * (FNfloat)0.577350269189626;
//              z += xy * (FNfloat)0.577350269189626;
            }
            break;
//          break;
        case TransformType3D_ImproveXZPlanes:
//      case TransformType3D_ImproveXZPlanes:
            {
                FNfloat xz = x + z;
//              FNfloat xz = x + z;
                FNfloat s2 = xz * -(FNfloat)0.211324865405187;
//              FNfloat s2 = xz * -(FNfloat)0.211324865405187;
                y *= (FNfloat)0.577350269189626;
//              y *= (FNfloat)0.577350269189626;
                x += s2 - y;
//              x += s2 - y;
                z += s2 - y;
//              z += s2 - y;
                y += xz * (FNfloat)0.577350269189626;
//              y += xz * (FNfloat)0.577350269189626;
            }
            break;
//          break;
        case TransformType3D_DefaultOpenSimplex2:
//      case TransformType3D_DefaultOpenSimplex2:
            {
                const FNfloat R3 = (FNfloat)(2.0 / 3.0);
//              const FNfloat R3 = (FNfloat)(2.0 / 3.0);
                FNfloat r = (x + y + z) * R3; // Rotation, not skew
//              FNfloat r = (x + y + z) * R3; // Rotation, not skew
                x = r - x;
//              x = r - x;
                y = r - y;
//              y = r - y;
                z = r - z;
//              z = r - z;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }

    void UpdateTransformType3D()
//  void UpdateTransformType3D()
    {
        switch (mRotationType3D)
//      switch (mRotationType3D)
        {
        case RotationType3D_ImproveXYPlanes:
//      case RotationType3D_ImproveXYPlanes:
            {
                mTransformType3D = TransformType3D_ImproveXYPlanes;
//              mTransformType3D = TransformType3D_ImproveXYPlanes;
            }
            break;
//          break;
        case RotationType3D_ImproveXZPlanes:
//      case RotationType3D_ImproveXZPlanes:
            {
                mTransformType3D = TransformType3D_ImproveXZPlanes;
//              mTransformType3D = TransformType3D_ImproveXZPlanes;
            }
            break;
//          break;
        default:
//      default:
            {
                switch (mNoiseType)
//              switch (mNoiseType)
                {
                case NoiseType_OpenSimplex2:
//              case NoiseType_OpenSimplex2:
                case NoiseType_OpenSimplex2S:
//              case NoiseType_OpenSimplex2S:
                    {
                        mTransformType3D = TransformType3D_DefaultOpenSimplex2;
//                      mTransformType3D = TransformType3D_DefaultOpenSimplex2;
                    }
                    break;
//                  break;
                default:
//              default:
                    {
                        mTransformType3D = TransformType3D_None;
//                      mTransformType3D = TransformType3D_None;
                    }
                    break;
//                  break;
                }
            }
            break;
//          break;
        }
    }


    // Domain Warp Coordinate Transforms
    // Domain Warp Coordinate Transforms

    template <typename FNfloat>
//  template <typename FNfloat>
    void TransformDomainWarpCoordinate(FNfloat& x, FNfloat& y) const
//  void TransformDomainWarpCoordinate(FNfloat& x, FNfloat& y) const
    {
        switch (mDomainWarpType)
//      switch (mDomainWarpType)
        {
        case DomainWarpType_OpenSimplex2:
//      case DomainWarpType_OpenSimplex2:
        case DomainWarpType_OpenSimplex2Reduced:
//      case DomainWarpType_OpenSimplex2Reduced:
            {
                const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
//              const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
                const FNfloat F2 = 0.5f * (SQRT3 - 1);
//              const FNfloat F2 = 0.5f * (SQRT3 - 1);
                FNfloat t = (x + y) * F2;
//              FNfloat t = (x + y) * F2;
                x += t;
//              x += t;
                y += t;
//              y += t;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void TransformDomainWarpCoordinate(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void TransformDomainWarpCoordinate(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        switch (mWarpTransformType3D)
//      switch (mWarpTransformType3D)
        {
        case TransformType3D_ImproveXYPlanes:
//      case TransformType3D_ImproveXYPlanes:
            {
                FNfloat xy = x + y;
//              FNfloat xy = x + y;
                FNfloat s2 = xy * -(FNfloat)0.211324865405187;
//              FNfloat s2 = xy * -(FNfloat)0.211324865405187;
                z *= (FNfloat)0.577350269189626;
//              z *= (FNfloat)0.577350269189626;
                x += s2 - z;
//              x += s2 - z;
                y = y + s2 - z;
//              y = y + s2 - z;
                z += xy * (FNfloat)0.577350269189626;
//              z += xy * (FNfloat)0.577350269189626;
            }
            break;
//          break;
        case TransformType3D_ImproveXZPlanes:
//      case TransformType3D_ImproveXZPlanes:
            {
                FNfloat xz = x + z;
//              FNfloat xz = x + z;
                FNfloat s2 = xz * -(FNfloat)0.211324865405187;
//              FNfloat s2 = xz * -(FNfloat)0.211324865405187;
                y *= (FNfloat)0.577350269189626;
//              y *= (FNfloat)0.577350269189626;
                x += s2 - y;
//              x += s2 - y;
                z += s2 - y;
//              z += s2 - y;
                y += xz * (FNfloat)0.577350269189626;
//              y += xz * (FNfloat)0.577350269189626;
            }
            break;
//          break;
        case TransformType3D_DefaultOpenSimplex2:
//      case TransformType3D_DefaultOpenSimplex2:
            {
                const FNfloat R3 = (FNfloat)(2.0 / 3.0);
//              const FNfloat R3 = (FNfloat)(2.0 / 3.0);
                FNfloat r = (x + y + z) * R3; // Rotation, not skew
//              FNfloat r = (x + y + z) * R3; // Rotation, not skew
                x = r - x;
//              x = r - x;
                y = r - y;
//              y = r - y;
                z = r - z;
//              z = r - z;
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }
    }

    void UpdateWarpTransformType3D()
//  void UpdateWarpTransformType3D()
    {
        switch (mRotationType3D)
//      switch (mRotationType3D)
        {
        case RotationType3D_ImproveXYPlanes:
//      case RotationType3D_ImproveXYPlanes:
            {
                mWarpTransformType3D = TransformType3D_ImproveXYPlanes;
//              mWarpTransformType3D = TransformType3D_ImproveXYPlanes;
            }
            break;
//          break;
        case RotationType3D_ImproveXZPlanes:
//      case RotationType3D_ImproveXZPlanes:
            {
                mWarpTransformType3D = TransformType3D_ImproveXZPlanes;
//              mWarpTransformType3D = TransformType3D_ImproveXZPlanes;
            }
            break;
//          break;
        default:
//      default:
            {
                switch (mDomainWarpType)
//              switch (mDomainWarpType)
                {
                case DomainWarpType_OpenSimplex2:
//              case DomainWarpType_OpenSimplex2:
                case DomainWarpType_OpenSimplex2Reduced:
//              case DomainWarpType_OpenSimplex2Reduced:
                    {
                        mWarpTransformType3D = TransformType3D_DefaultOpenSimplex2;
//                      mWarpTransformType3D = TransformType3D_DefaultOpenSimplex2;
                    }
                    break;
//                  break;
                default:
//              default:
                    {
                        mWarpTransformType3D = TransformType3D_None;
//                      mWarpTransformType3D = TransformType3D_None;
                    }
                    break;
//                  break;
                }
            }
            break;
//          break;
        }
    }


    // Fractal FBm
    // Fractal FBm

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalFBm(FNfloat x, FNfloat y) const
//  float GenFractalFBm(FNfloat x, FNfloat y) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = GenNoiseSingle(seed++, x, y);
//          float noise = GenNoiseSingle(seed++, x, y);
            sum += noise * amp;
//          sum += noise * amp;
            amp *= Lerp(1.0f, FastMin(noise + 1, 2) * 0.5f, mWeightedStrength);
//          amp *= Lerp(1.0f, FastMin(noise + 1, 2) * 0.5f, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalFBm(FNfloat x, FNfloat y, FNfloat z) const
//  float GenFractalFBm(FNfloat x, FNfloat y, FNfloat z) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = GenNoiseSingle(seed++, x, y, z);
//          float noise = GenNoiseSingle(seed++, x, y, z);
            sum += noise * amp;
//          sum += noise * amp;
            amp *= Lerp(1.0f, (noise + 1) * 0.5f, mWeightedStrength);
//          amp *= Lerp(1.0f, (noise + 1) * 0.5f, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            z *= mLacunarity;
//          z *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }


    // Fractal Ridged
    // Fractal Ridged

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalRidged(FNfloat x, FNfloat y) const
//  float GenFractalRidged(FNfloat x, FNfloat y) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = FastAbs(GenNoiseSingle(seed++, x, y));
//          float noise = FastAbs(GenNoiseSingle(seed++, x, y));
            sum += (noise * -2 + 1) * amp;
//          sum += (noise * -2 + 1) * amp;
            amp *= Lerp(1.0f, 1 - noise, mWeightedStrength);
//          amp *= Lerp(1.0f, 1 - noise, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalRidged(FNfloat x, FNfloat y, FNfloat z) const
//  float GenFractalRidged(FNfloat x, FNfloat y, FNfloat z) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = FastAbs(GenNoiseSingle(seed++, x, y, z));
//          float noise = FastAbs(GenNoiseSingle(seed++, x, y, z));
            sum += (noise * -2 + 1) * amp;
//          sum += (noise * -2 + 1) * amp;
            amp *= Lerp(1.0f, 1 - noise, mWeightedStrength);
//          amp *= Lerp(1.0f, 1 - noise, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            z *= mLacunarity;
//          z *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }


    // Fractal PingPong
    // Fractal PingPong

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalPingPong(FNfloat x, FNfloat y) const
//  float GenFractalPingPong(FNfloat x, FNfloat y) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = PingPong((GenNoiseSingle(seed++, x, y) + 1) * mPingPongStrength);
//          float noise = PingPong((GenNoiseSingle(seed++, x, y) + 1) * mPingPongStrength);
            sum += (noise - 0.5f) * 2 * amp;
//          sum += (noise - 0.5f) * 2 * amp;
            amp *= Lerp(1.0f, noise, mWeightedStrength);
//          amp *= Lerp(1.0f, noise, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float GenFractalPingPong(FNfloat x, FNfloat y, FNfloat z) const
//  float GenFractalPingPong(FNfloat x, FNfloat y, FNfloat z) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float sum = 0;
//      float sum = 0;
        float amp = mFractalBounding;
//      float amp = mFractalBounding;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            float noise = PingPong((GenNoiseSingle(seed++, x, y, z) + 1) * mPingPongStrength);
//          float noise = PingPong((GenNoiseSingle(seed++, x, y, z) + 1) * mPingPongStrength);
            sum += (noise - 0.5f) * 2 * amp;
//          sum += (noise - 0.5f) * 2 * amp;
            amp *= Lerp(1.0f, noise, mWeightedStrength);
//          amp *= Lerp(1.0f, noise, mWeightedStrength);

            x *= mLacunarity;
//          x *= mLacunarity;
            y *= mLacunarity;
//          y *= mLacunarity;
            z *= mLacunarity;
//          z *= mLacunarity;
            amp *= mGain;
//          amp *= mGain;
        }

        return sum;
//      return sum;
    }


    // Simplex/OpenSimplex2 Noise
    // Simplex/OpenSimplex2 Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleSimplex(int seed, FNfloat x, FNfloat y) const
//  float SingleSimplex(int seed, FNfloat x, FNfloat y) const
    {
        // 2D OpenSimplex2 case uses the same algorithm as ordinary Simplex.
        // 2D OpenSimplex2 case uses the same algorithm as ordinary Simplex.

        const float SQRT3 = 1.7320508075688772935274463415059f;
//      const float SQRT3 = 1.7320508075688772935274463415059f;
        const float G2 = (3 - SQRT3) / 6;
//      const float G2 = (3 - SQRT3) / 6;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
         * --- Skew moved to TransformNoiseCoordinate method ---
         * const FNfloat F2 = 0.5f * (SQRT3 - 1);
         * const FNfloat F2 = 0.5f * (SQRT3 - 1);
         * FNfloat s = (x + y) * F2;
         * FNfloat s = (x + y) * F2;
         * x += s; y += s;
         * x += s; y += s;
        */

        int i = FastFloor(x);
//      int i = FastFloor(x);
        int j = FastFloor(y);
//      int j = FastFloor(y);
        float xi = (float)(x - i);
//      float xi = (float)(x - i);
        float yi = (float)(y - j);
//      float yi = (float)(y - j);

        float t = (xi + yi) * G2;
//      float t = (xi + yi) * G2;
        float x0 = (float)(xi - t);
//      float x0 = (float)(xi - t);
        float y0 = (float)(yi - t);
//      float y0 = (float)(yi - t);

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;

        float n0, n1, n2;
//      float n0, n1, n2;

        float a = 0.5f - x0 * x0 - y0 * y0;
//      float a = 0.5f - x0 * x0 - y0 * y0;
        if (a <= 0)
//      if (a <= 0)
        {
            n0 = 0;
//          n0 = 0;
        }
        else
        {
            n0 = (a * a) * (a * a) * GradCoord(seed, i, j, x0, y0);
//          n0 = (a * a) * (a * a) * GradCoord(seed, i, j, x0, y0);
        }

        float c = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a);
//      float c = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a);
        if (c <= 0)
//      if (c <= 0)
        {
            n2 = 0;
//          n2 = 0;
        }
        else
        {
            float x2 = x0 + (2 * (float)G2 - 1);
//          float x2 = x0 + (2 * (float)G2 - 1);
            float y2 = y0 + (2 * (float)G2 - 1);
//          float y2 = y0 + (2 * (float)G2 - 1);
            n2 = (c * c) * (c * c) * GradCoord(seed, i + PrimeX, j + PrimeY, x2, y2);
//          n2 = (c * c) * (c * c) * GradCoord(seed, i + PrimeX, j + PrimeY, x2, y2);
        }

        if (y0 > x0)
//      if (y0 > x0)
        {
            float x1 = x0 + (float)G2;
//          float x1 = x0 + (float)G2;
            float y1 = y0 + ((float)G2 - 1);
//          float y1 = y0 + ((float)G2 - 1);
            float b = 0.5f - x1 * x1 - y1 * y1;
//          float b = 0.5f - x1 * x1 - y1 * y1;
            if (b <= 0)
//          if (b <= 0)
            {
                n1 = 0;
//              n1 = 0;
            }
            else
            {
                n1 = (b * b) * (b * b) * GradCoord(seed, i, j + PrimeY, x1, y1);
//              n1 = (b * b) * (b * b) * GradCoord(seed, i, j + PrimeY, x1, y1);
            }
        }
        else
        {
            float x1 = x0 + ((float)G2 - 1);
//          float x1 = x0 + ((float)G2 - 1);
            float y1 = y0 + (float)G2;
//          float y1 = y0 + (float)G2;
            float b = 0.5f - x1 * x1 - y1 * y1;
//          float b = 0.5f - x1 * x1 - y1 * y1;
            if (b <= 0)
//          if (b <= 0)
            {
                n1 = 0;
//              n1 = 0;
            }
            else
            {
                n1 = (b * b) * (b * b) * GradCoord(seed, i + PrimeX, j, x1, y1);
//              n1 = (b * b) * (b * b) * GradCoord(seed, i + PrimeX, j, x1, y1);
            }
        }

        return (n0 + n1 + n2) * 99.83685446303647f;
//      return (n0 + n1 + n2) * 99.83685446303647f;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleOpenSimplex2(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SingleOpenSimplex2(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        // 3D OpenSimplex2 case uses two offset rotated cube grids.
        // 3D OpenSimplex2 case uses two offset rotated cube grids.

        /*
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         * x = r - x; y = r - y; z = r - z;
        */

        int i = FastRound(x);
//      int i = FastRound(x);
        int j = FastRound(y);
//      int j = FastRound(y);
        int k = FastRound(z);
//      int k = FastRound(z);
        float x0 = (float)(x - i);
//      float x0 = (float)(x - i);
        float y0 = (float)(y - j);
//      float y0 = (float)(y - j);
        float z0 = (float)(z - k);
//      float z0 = (float)(z - k);

        int xNSign = (int)(-1.0f - x0) | 1;
//      int xNSign = (int)(-1.0f - x0) | 1;
        int yNSign = (int)(-1.0f - y0) | 1;
//      int yNSign = (int)(-1.0f - y0) | 1;
        int zNSign = (int)(-1.0f - z0) | 1;
//      int zNSign = (int)(-1.0f - z0) | 1;

        float ax0 = xNSign * -x0;
//      float ax0 = xNSign * -x0;
        float ay0 = yNSign * -y0;
//      float ay0 = yNSign * -y0;
        float az0 = zNSign * -z0;
//      float az0 = zNSign * -z0;

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;
        k *= PrimeZ;
//      k *= PrimeZ;

        float value = 0;
//      float value = 0;
        float a = (0.6f - x0 * x0) - (y0 * y0 + z0 * z0);
//      float a = (0.6f - x0 * x0) - (y0 * y0 + z0 * z0);

        for (int l = 0; ; l++)
//      for (int l = 0; ; l++)
        {
            if (a > 0)
//          if (a > 0)
            {
                value += (a * a) * (a * a) * GradCoord(seed, i, j, k, x0, y0, z0);
//              value += (a * a) * (a * a) * GradCoord(seed, i, j, k, x0, y0, z0);
            }

            float b = a + 1;
//          float b = a + 1;
            int i1 = i;
//          int i1 = i;
            int j1 = j;
//          int j1 = j;
            int k1 = k;
//          int k1 = k;
            float x1 = x0;
//          float x1 = x0;
            float y1 = y0;
//          float y1 = y0;
            float z1 = z0;
//          float z1 = z0;

            if (ax0 >= ay0 && ax0 >= az0)
//          if (ax0 >= ay0 && ax0 >= az0)
            {
                x1 += xNSign;
//              x1 += xNSign;
                b  -= xNSign * 2 * x1;
//              b  -= xNSign * 2 * x1;
                i1 -= xNSign * PrimeX;
//              i1 -= xNSign * PrimeX;
            }
            else
            if (ay0 >  ax0 && ay0 >= az0)
//          if (ay0 >  ax0 && ay0 >= az0)
            {
                y1 += yNSign;
//              y1 += yNSign;
                b  -= yNSign * 2 * y1;
//              b  -= yNSign * 2 * y1;
                j1 -= yNSign * PrimeY;
//              j1 -= yNSign * PrimeY;
            }
            else
            {
                z1 += zNSign;
//              z1 += zNSign;
                b  -= zNSign * 2 * z1;
//              b  -= zNSign * 2 * z1;
                k1 -= zNSign * PrimeZ;
//              k1 -= zNSign * PrimeZ;
            }

            if (b > 0)
//          if (b > 0)
            {
                value += (b * b) * (b * b) * GradCoord(seed, i1, j1, k1, x1, y1, z1);
//              value += (b * b) * (b * b) * GradCoord(seed, i1, j1, k1, x1, y1, z1);
            }

            if (l == 1) break;
//          if (l == 1) break;

            ax0 = 0.5f - ax0;
//          ax0 = 0.5f - ax0;
            ay0 = 0.5f - ay0;
//          ay0 = 0.5f - ay0;
            az0 = 0.5f - az0;
//          az0 = 0.5f - az0;

            x0 = xNSign * ax0;
//          x0 = xNSign * ax0;
            y0 = yNSign * ay0;
//          y0 = yNSign * ay0;
            z0 = zNSign * az0;
//          z0 = zNSign * az0;

            a += (0.75f - ax0) - (ay0 + az0);
//          a += (0.75f - ax0) - (ay0 + az0);

            i += (xNSign >> 1) & PrimeX;
//          i += (xNSign >> 1) & PrimeX;
            j += (yNSign >> 1) & PrimeY;
//          j += (yNSign >> 1) & PrimeY;
            k += (zNSign >> 1) & PrimeZ;
//          k += (zNSign >> 1) & PrimeZ;

            xNSign = -xNSign;
//          xNSign = -xNSign;
            yNSign = -yNSign;
//          yNSign = -yNSign;
            zNSign = -zNSign;
//          zNSign = -zNSign;

            seed = ~seed;
//          seed = ~seed;
        }

        return value * 32.69428253173828125f;
//      return value * 32.69428253173828125f;
    }


    // OpenSimplex2S Noise
    // OpenSimplex2S Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleOpenSimplex2S(int seed, FNfloat x, FNfloat y) const
//  float SingleOpenSimplex2S(int seed, FNfloat x, FNfloat y) const
    {
        // 2D OpenSimplex2S case is a modified 2D simplex noise.
        // 2D OpenSimplex2S case is a modified 2D simplex noise.

        const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
//      const FNfloat SQRT3 = (FNfloat)1.7320508075688772935274463415059;
        const FNfloat G2 = (3 - SQRT3) / 6;
//      const FNfloat G2 = (3 - SQRT3) / 6;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
         * --- Skew moved to TransformNoiseCoordinate method ---
         * const FNfloat F2 = 0.5f * (SQRT3 - 1);
         * const FNfloat F2 = 0.5f * (SQRT3 - 1);
         * FNfloat s = (x + y) * F2;
         * FNfloat s = (x + y) * F2;
         * x += s; y += s;
         * x += s; y += s;
        */

        int i = FastFloor(x);
//      int i = FastFloor(x);
        int j = FastFloor(y);
//      int j = FastFloor(y);
        float xi = (float)(x - i);
//      float xi = (float)(x - i);
        float yi = (float)(y - j);
//      float yi = (float)(y - j);

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;
        int i1 = i + PrimeX;
//      int i1 = i + PrimeX;
        int j1 = j + PrimeY;
//      int j1 = j + PrimeY;

        float t = (xi + yi) * (float)G2;
//      float t = (xi + yi) * (float)G2;
        float x0 = xi - t;
//      float x0 = xi - t;
        float y0 = yi - t;
//      float y0 = yi - t;

        float a0 = (2.0f / 3.0f) - x0 * x0 - y0 * y0;
//      float a0 = (2.0f / 3.0f) - x0 * x0 - y0 * y0;
        float value = (a0 * a0) * (a0 * a0) * GradCoord(seed, i, j, x0, y0);
//      float value = (a0 * a0) * (a0 * a0) * GradCoord(seed, i, j, x0, y0);

        float a1 = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a0);
//      float a1 = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a0);
        float x1 = x0 - (float)(1 - 2 * G2);
//      float x1 = x0 - (float)(1 - 2 * G2);
        float y1 = y0 - (float)(1 - 2 * G2);
//      float y1 = y0 - (float)(1 - 2 * G2);
        value += (a1 * a1) * (a1 * a1) * GradCoord(seed, i1, j1, x1, y1);
//      value += (a1 * a1) * (a1 * a1) * GradCoord(seed, i1, j1, x1, y1);

        // Nested conditionals were faster than compact bit logic/arithmetic.
        // Nested conditionals were faster than compact bit logic/arithmetic.
        float xmyi = xi - yi;
//      float xmyi = xi - yi;
        if (t > G2)
//      if (t > G2)
        {
            if (xi + xmyi > 1)
//          if (xi + xmyi > 1)
            {
                float x2 = x0 + (float)(3 * G2 - 2);
//              float x2 = x0 + (float)(3 * G2 - 2);
                float y2 = y0 + (float)(3 * G2 - 1);
//              float y2 = y0 + (float)(3 * G2 - 1);
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + (PrimeX << 1), j + PrimeY, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + (PrimeX << 1), j + PrimeY, x2, y2);
                }
            }
            else
            {
                float x2 = x0 + (float)G2;
//              float x2 = x0 + (float)G2;
                float y2 = y0 + (float)(G2 - 1);
//              float y2 = y0 + (float)(G2 - 1);
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j + PrimeY, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j + PrimeY, x2, y2);
                }
            }

            if (yi - xmyi > 1)
//          if (yi - xmyi > 1)
            {
                float x3 = x0 + (float)(3 * G2 - 1);
//              float x3 = x0 + (float)(3 * G2 - 1);
                float y3 = y0 + (float)(3 * G2 - 2);
//              float y3 = y0 + (float)(3 * G2 - 2);
                float a3 = (2.0f / 3.0f) - x3 * x3 - y3 * y3;
//              float a3 = (2.0f / 3.0f) - x3 * x3 - y3 * y3;
                if (a3 > 0)
//              if (a3 > 0)
                {
                    value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + PrimeX, j + (PrimeY << 1), x3, y3);
//                  value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + PrimeX, j + (PrimeY << 1), x3, y3);
                }
            }
            else
            {
                float x3 = x0 + (float)(G2 - 1);
//              float x3 = x0 + (float)(G2 - 1);
                float y3 = y0 + (float)G2;
//              float y3 = y0 + (float)G2;
                float a3 = (2.0f / 3.0f) - x3 * x3 - y3 * y3;
//              float a3 = (2.0f / 3.0f) - x3 * x3 - y3 * y3;
                if (a3 > 0)
//              if (a3 > 0)
                {
                    value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + PrimeX, j, x3, y3);
//                  value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + PrimeX, j, x3, y3);
                }
            }
        }
        else
        {
            if (xi + xmyi < 0)
//          if (xi + xmyi < 0)
            {
                float x2 = x0 + (float)(1 - G2);
//              float x2 = x0 + (float)(1 - G2);
                float y2 = y0 - (float)G2;
//              float y2 = y0 - (float)G2;
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i - PrimeX, j, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i - PrimeX, j, x2, y2);
                }
            }
            else
            {
                float x2 = x0 + (float)(G2 - 1);
//              float x2 = x0 + (float)(G2 - 1);
                float y2 = y0 + (float)G2;
//              float y2 = y0 + (float)G2;
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + PrimeX, j, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + PrimeX, j, x2, y2);
                }
            }

            if (yi < xmyi)
//          if (yi < xmyi)
            {
                float x2 = x0 - (float)G2;
//              float x2 = x0 - (float)G2;
                float y2 = y0 - (float)(G2 - 1);
//              float y2 = y0 - (float)(G2 - 1);
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j - PrimeY, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j - PrimeY, x2, y2);
                }
            }
            else
            {
                float x2 = x0 + (float)G2;
//              float x2 = x0 + (float)G2;
                float y2 = y0 + (float)(G2 - 1);
//              float y2 = y0 + (float)(G2 - 1);
                float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
//              float a2 = (2.0f / 3.0f) - x2 * x2 - y2 * y2;
                if (a2 > 0)
//              if (a2 > 0)
                {
                    value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j + PrimeY, x2, y2);
//                  value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i, j + PrimeY, x2, y2);
                }
            }
        }

        return value * 18.24196194486065f;
//      return value * 18.24196194486065f;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleOpenSimplex2S(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SingleOpenSimplex2S(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        // 3D OpenSimplex2S case uses two offset rotated cube grids.
        // 3D OpenSimplex2S case uses two offset rotated cube grids.

        /*
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         * x = r - x; y = r - y; z = r - z;
        */

        int i = FastFloor(x);
//      int i = FastFloor(x);
        int j = FastFloor(y);
//      int j = FastFloor(y);
        int k = FastFloor(z);
//      int k = FastFloor(z);
        float xi = (float)(x - i);
//      float xi = (float)(x - i);
        float yi = (float)(y - j);
//      float yi = (float)(y - j);
        float zi = (float)(z - k);
//      float zi = (float)(z - k);

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;
        k *= PrimeZ;
//      k *= PrimeZ;
        int seed2 = seed + 1293373;
//      int seed2 = seed + 1293373;

        int xNMask = (int)(-0.5f - xi);
//      int xNMask = (int)(-0.5f - xi);
        int yNMask = (int)(-0.5f - yi);
//      int yNMask = (int)(-0.5f - yi);
        int zNMask = (int)(-0.5f - zi);
//      int zNMask = (int)(-0.5f - zi);

        float x0 = xi + xNMask;
//      float x0 = xi + xNMask;
        float y0 = yi + yNMask;
//      float y0 = yi + yNMask;
        float z0 = zi + zNMask;
//      float z0 = zi + zNMask;
        float a0 = 0.75f - x0 * x0 - y0 * y0 - z0 * z0;
//      float a0 = 0.75f - x0 * x0 - y0 * y0 - z0 * z0;
        float value = (a0 * a0) * (a0 * a0) * GradCoord(seed, i + (xNMask & PrimeX), j + (yNMask & PrimeY), k + (zNMask & PrimeZ), x0, y0, z0);
//      float value = (a0 * a0) * (a0 * a0) * GradCoord(seed, i + (xNMask & PrimeX), j + (yNMask & PrimeY), k + (zNMask & PrimeZ), x0, y0, z0);

        float x1 = xi - 0.5f;
//      float x1 = xi - 0.5f;
        float y1 = yi - 0.5f;
//      float y1 = yi - 0.5f;
        float z1 = zi - 0.5f;
//      float z1 = zi - 0.5f;
        float a1 = 0.75f - x1 * x1 - y1 * y1 - z1 * z1;
//      float a1 = 0.75f - x1 * x1 - y1 * y1 - z1 * z1;
        value += (a1 * a1) * (a1 * a1) * GradCoord(seed2, i + PrimeX, j + PrimeY, k + PrimeZ, x1, y1, z1);
//      value += (a1 * a1) * (a1 * a1) * GradCoord(seed2, i + PrimeX, j + PrimeY, k + PrimeZ, x1, y1, z1);

        float xAFlipMask0 = ((xNMask | 1) << 1) * x1;
//      float xAFlipMask0 = ((xNMask | 1) << 1) * x1;
        float yAFlipMask0 = ((yNMask | 1) << 1) * y1;
//      float yAFlipMask0 = ((yNMask | 1) << 1) * y1;
        float zAFlipMask0 = ((zNMask | 1) << 1) * z1;
//      float zAFlipMask0 = ((zNMask | 1) << 1) * z1;
        float xAFlipMask1 = (-2 - (xNMask << 2)) * x1 - 1.0f;
//      float xAFlipMask1 = (-2 - (xNMask << 2)) * x1 - 1.0f;
        float yAFlipMask1 = (-2 - (yNMask << 2)) * y1 - 1.0f;
//      float yAFlipMask1 = (-2 - (yNMask << 2)) * y1 - 1.0f;
        float zAFlipMask1 = (-2 - (zNMask << 2)) * z1 - 1.0f;
//      float zAFlipMask1 = (-2 - (zNMask << 2)) * z1 - 1.0f;

        bool skip5 = false;
//      bool skip5 = false;
        float a2 = xAFlipMask0 + a0;
//      float a2 = xAFlipMask0 + a0;
        if (a2 > 0)
//      if (a2 > 0)
        {
            float x2 = x0 - (xNMask | 1);
//          float x2 = x0 - (xNMask | 1);
            float y2 = y0;
//          float y2 = y0;
            float z2 = z0;
//          float z2 = z0;
            value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + (~xNMask & PrimeX), j + (yNMask & PrimeY), k + (zNMask & PrimeZ), x2, y2, z2);
//          value += (a2 * a2) * (a2 * a2) * GradCoord(seed, i + (~xNMask & PrimeX), j + (yNMask & PrimeY), k + (zNMask & PrimeZ), x2, y2, z2);
        }
        else
        {
            float a3 = yAFlipMask0 + zAFlipMask0 + a0;
//          float a3 = yAFlipMask0 + zAFlipMask0 + a0;
            if (a3 > 0)
//          if (a3 > 0)
            {
                float x3 = x0;
//              float x3 = x0;
                float y3 = y0 - (yNMask | 1);
//              float y3 = y0 - (yNMask | 1);
                float z3 = z0 - (zNMask | 1);
//              float z3 = z0 - (zNMask | 1);
                value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + (xNMask & PrimeX), j + (~yNMask & PrimeY), k + (~zNMask & PrimeZ), x3, y3, z3);
//              value += (a3 * a3) * (a3 * a3) * GradCoord(seed, i + (xNMask & PrimeX), j + (~yNMask & PrimeY), k + (~zNMask & PrimeZ), x3, y3, z3);
            }

            float a4 = xAFlipMask1 + a1;
//          float a4 = xAFlipMask1 + a1;
            if (a4 > 0)
//          if (a4 > 0)
            {
                float x4 = (xNMask | 1) + x1;
//              float x4 = (xNMask | 1) + x1;
                float y4 = y1;
//              float y4 = y1;
                float z4 = z1;
//              float z4 = z1;
                value += (a4 * a4) * (a4 * a4) * GradCoord(seed2, i + (xNMask & (PrimeX * 2)), j + PrimeY, k + PrimeZ, x4, y4, z4);
//              value += (a4 * a4) * (a4 * a4) * GradCoord(seed2, i + (xNMask & (PrimeX * 2)), j + PrimeY, k + PrimeZ, x4, y4, z4);
                skip5 = true;
//              skip5 = true;
            }
        }

        bool skip9 = false;
//      bool skip9 = false;
        float a6 = yAFlipMask0 + a0;
//      float a6 = yAFlipMask0 + a0;
        if (a6 > 0)
//      if (a6 > 0)
        {
            float x6 = x0;
//          float x6 = x0;
            float y6 = y0 - (yNMask | 1);
//          float y6 = y0 - (yNMask | 1);
            float z6 = z0;
//          float z6 = z0;
            value += (a6 * a6) * (a6 * a6) * GradCoord(seed, i + (xNMask & PrimeX), j + (~yNMask & PrimeY), k + (zNMask & PrimeZ), x6, y6, z6);
//          value += (a6 * a6) * (a6 * a6) * GradCoord(seed, i + (xNMask & PrimeX), j + (~yNMask & PrimeY), k + (zNMask & PrimeZ), x6, y6, z6);
        }
        else
        {
            float a7 = xAFlipMask0 + zAFlipMask0 + a0;
//          float a7 = xAFlipMask0 + zAFlipMask0 + a0;
            if (a7 > 0)
//          if (a7 > 0)
            {
                float x7 = x0 - (xNMask | 1);
//              float x7 = x0 - (xNMask | 1);
                float y7 = y0;
//              float y7 = y0;
                float z7 = z0 - (zNMask | 1);
//              float z7 = z0 - (zNMask | 1);
                value += (a7 * a7) * (a7 * a7) * GradCoord(seed, i + (~xNMask & PrimeX), j + (yNMask & PrimeY), k + (~zNMask & PrimeZ), x7, y7, z7);
//              value += (a7 * a7) * (a7 * a7) * GradCoord(seed, i + (~xNMask & PrimeX), j + (yNMask & PrimeY), k + (~zNMask & PrimeZ), x7, y7, z7);
            }

            float a8 = yAFlipMask1 + a1;
//          float a8 = yAFlipMask1 + a1;
            if (a8 > 0)
//          if (a8 > 0)
            {
                float x8 = x1;
//              float x8 = x1;
                float y8 = (yNMask | 1) + y1;
//              float y8 = (yNMask | 1) + y1;
                float z8 = z1;
//              float z8 = z1;
                value += (a8 * a8) * (a8 * a8) * GradCoord(seed2, i + PrimeX, j + (yNMask & (PrimeY << 1)), k + PrimeZ, x8, y8, z8);
//              value += (a8 * a8) * (a8 * a8) * GradCoord(seed2, i + PrimeX, j + (yNMask & (PrimeY << 1)), k + PrimeZ, x8, y8, z8);
                skip9 = true;
//              skip9 = true;
            }
        }

        bool skipD = false;
//      bool skipD = false;
        float aA = zAFlipMask0 + a0;
//      float aA = zAFlipMask0 + a0;
        if (aA > 0)
//      if (aA > 0)
        {
            float xA = x0;
//          float xA = x0;
            float yA = y0;
//          float yA = y0;
            float zA = z0 - (zNMask | 1);
//          float zA = z0 - (zNMask | 1);
            value += (aA * aA) * (aA * aA) * GradCoord(seed, i + (xNMask & PrimeX), j + (yNMask & PrimeY), k + (~zNMask & PrimeZ), xA, yA, zA);
//          value += (aA * aA) * (aA * aA) * GradCoord(seed, i + (xNMask & PrimeX), j + (yNMask & PrimeY), k + (~zNMask & PrimeZ), xA, yA, zA);
        }
        else
        {
            float aB = xAFlipMask0 + yAFlipMask0 + a0;
//          float aB = xAFlipMask0 + yAFlipMask0 + a0;
            if (aB > 0)
//          if (aB > 0)
            {
                float xB = x0 - (xNMask | 1);
//              float xB = x0 - (xNMask | 1);
                float yB = y0 - (yNMask | 1);
//              float yB = y0 - (yNMask | 1);
                float zB = z0;
//              float zB = z0;
                value += (aB * aB) * (aB * aB) * GradCoord(seed, i + (~xNMask & PrimeX), j + (~yNMask & PrimeY), k + (zNMask & PrimeZ), xB, yB, zB);
//              value += (aB * aB) * (aB * aB) * GradCoord(seed, i + (~xNMask & PrimeX), j + (~yNMask & PrimeY), k + (zNMask & PrimeZ), xB, yB, zB);
            }

            float aC = zAFlipMask1 + a1;
//          float aC = zAFlipMask1 + a1;
            if (aC > 0)
//          if (aC > 0)
            {
                float xC = x1;
//              float xC = x1;
                float yC = y1;
//              float yC = y1;
                float zC = (zNMask | 1) + z1;
//              float zC = (zNMask | 1) + z1;
                value += (aC * aC) * (aC * aC) * GradCoord(seed2, i + PrimeX, j + PrimeY, k + (zNMask & (PrimeZ << 1)), xC, yC, zC);
//              value += (aC * aC) * (aC * aC) * GradCoord(seed2, i + PrimeX, j + PrimeY, k + (zNMask & (PrimeZ << 1)), xC, yC, zC);
                skipD = true;
//              skipD = true;
            }
        }

        if (!skip5)
//      if (!skip5)
        {
            float a5 = yAFlipMask1 + zAFlipMask1 + a1;
//          float a5 = yAFlipMask1 + zAFlipMask1 + a1;
            if (a5 > 0)
//          if (a5 > 0)
            {
                float x5 = x1;
//              float x5 = x1;
                float y5 = (yNMask | 1) + y1;
//              float y5 = (yNMask | 1) + y1;
                float z5 = (zNMask | 1) + z1;
//              float z5 = (zNMask | 1) + z1;
                value += (a5 * a5) * (a5 * a5) * GradCoord(seed2, i + PrimeX, j + (yNMask & (PrimeY << 1)), k + (zNMask & (PrimeZ << 1)), x5, y5, z5);
//              value += (a5 * a5) * (a5 * a5) * GradCoord(seed2, i + PrimeX, j + (yNMask & (PrimeY << 1)), k + (zNMask & (PrimeZ << 1)), x5, y5, z5);
            }
        }

        if (!skip9)
//      if (!skip9)
        {
            float a9 = xAFlipMask1 + zAFlipMask1 + a1;
//          float a9 = xAFlipMask1 + zAFlipMask1 + a1;
            if (a9 > 0)
//          if (a9 > 0)
            {
                float x9 = (xNMask | 1) + x1;
//              float x9 = (xNMask | 1) + x1;
                float y9 = y1;
//              float y9 = y1;
                float z9 = (zNMask | 1) + z1;
//              float z9 = (zNMask | 1) + z1;
                value += (a9 * a9) * (a9 * a9) * GradCoord(seed2, i + (xNMask & (PrimeX * 2)), j + PrimeY, k + (zNMask & (PrimeZ << 1)), x9, y9, z9);
//              value += (a9 * a9) * (a9 * a9) * GradCoord(seed2, i + (xNMask & (PrimeX * 2)), j + PrimeY, k + (zNMask & (PrimeZ << 1)), x9, y9, z9);
            }
        }

        if (!skipD)
//      if (!skipD)
        {
            float aD = xAFlipMask1 + yAFlipMask1 + a1;
//          float aD = xAFlipMask1 + yAFlipMask1 + a1;
            if (aD > 0)
//          if (aD > 0)
            {
                float xD = (xNMask | 1) + x1;
//              float xD = (xNMask | 1) + x1;
                float yD = (yNMask | 1) + y1;
//              float yD = (yNMask | 1) + y1;
                float zD = z1;
//              float zD = z1;
                value += (aD * aD) * (aD * aD) * GradCoord(seed2, i + (xNMask & (PrimeX << 1)), j + (yNMask & (PrimeY << 1)), k + PrimeZ, xD, yD, zD);
//              value += (aD * aD) * (aD * aD) * GradCoord(seed2, i + (xNMask & (PrimeX << 1)), j + (yNMask & (PrimeY << 1)), k + PrimeZ, xD, yD, zD);
            }
        }

        return value * 9.046026385208288f;
//      return value * 9.046026385208288f;
    }


    // Cellular Noise
    // Cellular Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleCellular(int seed, FNfloat x, FNfloat y) const
//  float SingleCellular(int seed, FNfloat x, FNfloat y) const
    {
        int xr = FastRound(x);
//      int xr = FastRound(x);
        int yr = FastRound(y);
//      int yr = FastRound(y);

        float distance0 = 1e10f;
//      float distance0 = 1e10f;
        float distance1 = 1e10f;
//      float distance1 = 1e10f;
        int closestHash = 0;
//      int closestHash = 0;

        float cellularJitter = 0.43701595f * mCellularJitterModifier;
//      float cellularJitter = 0.43701595f * mCellularJitterModifier;

        int xPrimed     = (xr - 1) * PrimeX;
//      int xPrimed     = (xr - 1) * PrimeX;
        int yPrimedBase = (yr - 1) * PrimeY;
//      int yPrimedBase = (yr - 1) * PrimeY;

        switch (mCellularDistanceFunction)
//      switch (mCellularDistanceFunction)
        {
        default:
//      default:
        case CellularDistanceFunction_Euclidean:
//      case CellularDistanceFunction_Euclidean:
        case CellularDistanceFunction_EuclideanSq:
//      case CellularDistanceFunction_EuclideanSq:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int hash = Hash(seed, xPrimed, yPrimed);
//                      int hash = Hash(seed, xPrimed, yPrimed);
                        int idx  = hash & (255 << 1);
//                      int idx  = hash & (255 << 1);

                        float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
//                      float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
                        float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;
//                      float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;

                        float newDistance = vecX * vecX + vecY * vecY;
//                      float newDistance = vecX * vecX + vecY * vecY;

                        distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                      distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                        if (newDistance < distance0)
//                      if (newDistance < distance0)
                        {
                            distance0 = newDistance;
//                          distance0 = newDistance;
                            closestHash = hash;
//                          closestHash = hash;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        case CellularDistanceFunction_Manhattan:
//      case CellularDistanceFunction_Manhattan:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int hash = Hash(seed, xPrimed, yPrimed);
//                      int hash = Hash(seed, xPrimed, yPrimed);
                        int idx  = hash & (255 << 1);
//                      int idx  = hash & (255 << 1);

                        float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
//                      float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
                        float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;
//                      float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;

                        float newDistance = FastAbs(vecX) + FastAbs(vecY);
//                      float newDistance = FastAbs(vecX) + FastAbs(vecY);

                        distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                      distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                        if (newDistance < distance0)
//                      if (newDistance < distance0)
                        {
                            distance0 = newDistance;
//                          distance0 = newDistance;
                            closestHash = hash;
//                          closestHash = hash;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        case CellularDistanceFunction_Hybrid:
//      case CellularDistanceFunction_Hybrid:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int hash = Hash(seed, xPrimed, yPrimed);
//                      int hash = Hash(seed, xPrimed, yPrimed);
                        int idx  = hash & (255 << 1);
//                      int idx  = hash & (255 << 1);

                        float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
//                      float vecX = (float)(xi - x) + Lookup<float>::RandVecs2D[idx    ] * cellularJitter;
                        float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;
//                      float vecY = (float)(yi - y) + Lookup<float>::RandVecs2D[idx | 1] * cellularJitter;

                        float newDistance = (FastAbs(vecX) + FastAbs(vecY)) + (vecX * vecX + vecY * vecY);
//                      float newDistance = (FastAbs(vecX) + FastAbs(vecY)) + (vecX * vecX + vecY * vecY);

                        distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                      distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                        if (newDistance < distance0)
//                      if (newDistance < distance0)
                        {
                            distance0 = newDistance;
//                          distance0 = newDistance;
                            closestHash = hash;
//                          closestHash = hash;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        }

        if (mCellularDistanceFunction == CellularDistanceFunction_Euclidean && mCellularReturnType >= CellularReturnType_Distance)
//      if (mCellularDistanceFunction == CellularDistanceFunction_Euclidean && mCellularReturnType >= CellularReturnType_Distance)
        {
            distance0 = FastSqrt(distance0);
//          distance0 = FastSqrt(distance0);

            if (mCellularReturnType >= CellularReturnType_Distance2)
//          if (mCellularReturnType >= CellularReturnType_Distance2)
            {
                distance1 = FastSqrt(distance1);
//              distance1 = FastSqrt(distance1);
            }
        }

        switch (mCellularReturnType)
//      switch (mCellularReturnType)
        {
        case CellularReturnType_CellValue:
//      case CellularReturnType_CellValue:
            {
                return closestHash * (1 / 2147483648.0f);
//              return closestHash * (1 / 2147483648.0f);
            }
        case CellularReturnType_Distance:
//      case CellularReturnType_Distance:
            {
                return distance0 - 1;
//              return distance0 - 1;
            }
        case CellularReturnType_Distance2:
//      case CellularReturnType_Distance2:
            {
                return distance1 - 1;
//              return distance1 - 1;
            }
        case CellularReturnType_Distance2Add:
//      case CellularReturnType_Distance2Add:
            {
                return (distance1 + distance0) * 0.5f - 1;
//              return (distance1 + distance0) * 0.5f - 1;
            }
        case CellularReturnType_Distance2Sub:
//      case CellularReturnType_Distance2Sub:
            {
                return distance1 - distance0 - 1;
//              return distance1 - distance0 - 1;
            }
        case CellularReturnType_Distance2Mul:
//      case CellularReturnType_Distance2Mul:
            {
                return distance1 * distance0 * 0.5f - 1;
//              return distance1 * distance0 * 0.5f - 1;
            }
        case CellularReturnType_Distance2Div:
//      case CellularReturnType_Distance2Div:
            {
                return distance0 / distance1 - 1;
//              return distance0 / distance1 - 1;
            }
        default:
//      default:
            {
                return 0;
//              return 0;
            }
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleCellular(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SingleCellular(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        int xr = FastRound(x);
//      int xr = FastRound(x);
        int yr = FastRound(y);
//      int yr = FastRound(y);
        int zr = FastRound(z);
//      int zr = FastRound(z);

        float distance0 = 1e10f;
//      float distance0 = 1e10f;
        float distance1 = 1e10f;
//      float distance1 = 1e10f;
        int closestHash = 0;
//      int closestHash = 0;

        float cellularJitter = 0.39614353f * mCellularJitterModifier;
//      float cellularJitter = 0.39614353f * mCellularJitterModifier;

        int xPrimed     = (xr - 1) * PrimeX;
//      int xPrimed     = (xr - 1) * PrimeX;
        int yPrimedBase = (yr - 1) * PrimeY;
//      int yPrimedBase = (yr - 1) * PrimeY;
        int zPrimedBase = (zr - 1) * PrimeZ;
//      int zPrimedBase = (zr - 1) * PrimeZ;

        switch (mCellularDistanceFunction)
//      switch (mCellularDistanceFunction)
        {
        case CellularDistanceFunction_Euclidean:
//      case CellularDistanceFunction_Euclidean:
        case CellularDistanceFunction_EuclideanSq:
//      case CellularDistanceFunction_EuclideanSq:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int zPrimed = zPrimedBase;
//                      int zPrimed = zPrimedBase;

                        for (int zi = zr - 1; zi <= zr + 1; zi++)
//                      for (int zi = zr - 1; zi <= zr + 1; zi++)
                        {
                            int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//                          int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
                            int idx  = hash & (255 << 2);
//                          int idx  = hash & (255 << 2);

                            float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
//                          float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
                            float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
//                          float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
                            float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;
//                          float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;

                            float newDistance = vecX * vecX + vecY * vecY + vecZ * vecZ;
//                          float newDistance = vecX * vecX + vecY * vecY + vecZ * vecZ;

                            distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                          distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                            if (newDistance < distance0)
//                          if (newDistance < distance0)
                            {
                                distance0 = newDistance;
//                              distance0 = newDistance;
                                closestHash = hash;
//                              closestHash = hash;
                            }
                            zPrimed += PrimeZ;
//                          zPrimed += PrimeZ;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        case CellularDistanceFunction_Manhattan:
//      case CellularDistanceFunction_Manhattan:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int zPrimed = zPrimedBase;
//                      int zPrimed = zPrimedBase;

                        for (int zi = zr - 1; zi <= zr + 1; zi++)
//                      for (int zi = zr - 1; zi <= zr + 1; zi++)
                        {
                            int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//                          int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
                            int idx  = hash & (255 << 2);
//                          int idx  = hash & (255 << 2);

                            float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
//                          float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
                            float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
//                          float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
                            float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;
//                          float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;

                            float newDistance = FastAbs(vecX) + FastAbs(vecY) + FastAbs(vecZ);
//                          float newDistance = FastAbs(vecX) + FastAbs(vecY) + FastAbs(vecZ);

                            distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                          distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                            if (newDistance < distance0)
//                          if (newDistance < distance0)
                            {
                                distance0 = newDistance;
//                              distance0 = newDistance;
                                closestHash = hash;
//                              closestHash = hash;
                            }
                            zPrimed += PrimeZ;
//                          zPrimed += PrimeZ;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        case CellularDistanceFunction_Hybrid:
//      case CellularDistanceFunction_Hybrid:
            {
                for (int xi = xr - 1; xi <= xr + 1; xi++)
//              for (int xi = xr - 1; xi <= xr + 1; xi++)
                {
                    int yPrimed = yPrimedBase;
//                  int yPrimed = yPrimedBase;

                    for (int yi = yr - 1; yi <= yr + 1; yi++)
//                  for (int yi = yr - 1; yi <= yr + 1; yi++)
                    {
                        int zPrimed = zPrimedBase;
//                      int zPrimed = zPrimedBase;

                        for (int zi = zr - 1; zi <= zr + 1; zi++)
//                      for (int zi = zr - 1; zi <= zr + 1; zi++)
                        {
                            int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
//                          int hash = Hash(seed, xPrimed, yPrimed, zPrimed);
                            int idx  = hash & (255 << 2);
//                          int idx  = hash & (255 << 2);

                            float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
//                          float vecX = (float)(xi - x) + Lookup<float>::RandVecs3D[idx    ] * cellularJitter;
                            float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
//                          float vecY = (float)(yi - y) + Lookup<float>::RandVecs3D[idx | 1] * cellularJitter;
                            float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;
//                          float vecZ = (float)(zi - z) + Lookup<float>::RandVecs3D[idx | 2] * cellularJitter;

                            float newDistance = (FastAbs(vecX) + FastAbs(vecY) + FastAbs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ);
//                          float newDistance = (FastAbs(vecX) + FastAbs(vecY) + FastAbs(vecZ)) + (vecX * vecX + vecY * vecY + vecZ * vecZ);

                            distance1 = FastMax(FastMin(distance1, newDistance), distance0);
//                          distance1 = FastMax(FastMin(distance1, newDistance), distance0);
                            if (newDistance < distance0)
//                          if (newDistance < distance0)
                            {
                                distance0 = newDistance;
//                              distance0 = newDistance;
                                closestHash = hash;
//                              closestHash = hash;
                            }
                            zPrimed += PrimeZ;
//                          zPrimed += PrimeZ;
                        }
                        yPrimed += PrimeY;
//                      yPrimed += PrimeY;
                    }
                    xPrimed += PrimeX;
//                  xPrimed += PrimeX;
                }
            }
            break;
//          break;
        default:
//      default:
            break;
//          break;
        }

        if (mCellularDistanceFunction == CellularDistanceFunction_Euclidean && mCellularReturnType >= CellularReturnType_Distance)
//      if (mCellularDistanceFunction == CellularDistanceFunction_Euclidean && mCellularReturnType >= CellularReturnType_Distance)
        {
            distance0 = FastSqrt(distance0);
//          distance0 = FastSqrt(distance0);

            if (mCellularReturnType >= CellularReturnType_Distance2)
//          if (mCellularReturnType >= CellularReturnType_Distance2)
            {
                distance1 = FastSqrt(distance1);
//              distance1 = FastSqrt(distance1);
            }
        }

        switch (mCellularReturnType)
//      switch (mCellularReturnType)
        {
        case CellularReturnType_CellValue:
//      case CellularReturnType_CellValue:
            {
                return closestHash * (1 / 2147483648.0f);
//              return closestHash * (1 / 2147483648.0f);
            }
        case CellularReturnType_Distance:
//      case CellularReturnType_Distance:
            {
                return distance0 - 1;
//              return distance0 - 1;
            }
        case CellularReturnType_Distance2:
//      case CellularReturnType_Distance2:
            {
                return distance1 - 1;
//              return distance1 - 1;
            }
        case CellularReturnType_Distance2Add:
//      case CellularReturnType_Distance2Add:
            {
                return (distance1 + distance0) * 0.5f - 1;
//              return (distance1 + distance0) * 0.5f - 1;
            }
        case CellularReturnType_Distance2Sub:
//      case CellularReturnType_Distance2Sub:
            {
                return distance1 - distance0 - 1;
//              return distance1 - distance0 - 1;
            }
        case CellularReturnType_Distance2Mul:
//      case CellularReturnType_Distance2Mul:
            {
                return distance1 * distance0 * 0.5f - 1;
//              return distance1 * distance0 * 0.5f - 1;
            }
        case CellularReturnType_Distance2Div:
//      case CellularReturnType_Distance2Div:
            {
                return distance0 / distance1 - 1;
//              return distance0 / distance1 - 1;
            }
        default:
//      default:
            {
                return 0;
//              return 0;
            }
        }
    }


    // Perlin Noise
    // Perlin Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SinglePerlin(int seed, FNfloat x, FNfloat y) const
//  float SinglePerlin(int seed, FNfloat x, FNfloat y) const
    {
        int x0 = FastFloor(x);
        int y0 = FastFloor(y);

        float xd0 = (float)(x - x0);
        float yd0 = (float)(y - y0);
        float xd1 = xd0 - 1;
        float yd1 = yd0 - 1;

        float xs = InterpQuintic(xd0);
        float ys = InterpQuintic(yd0);

        x0 *= PrimeX;
        y0 *= PrimeY;
        int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;

        float xf0 = Lerp(GradCoord(seed, x0, y0, xd0, yd0), GradCoord(seed, x1, y0, xd1, yd0), xs);
//      float xf0 = Lerp(GradCoord(seed, x0, y0, xd0, yd0), GradCoord(seed, x1, y0, xd1, yd0), xs);
        float xf1 = Lerp(GradCoord(seed, x0, y1, xd0, yd1), GradCoord(seed, x1, y1, xd1, yd1), xs);
//      float xf1 = Lerp(GradCoord(seed, x0, y1, xd0, yd1), GradCoord(seed, x1, y1, xd1, yd1), xs);

        return Lerp(xf0, xf1, ys) * 1.4247691104677813f;
//      return Lerp(xf0, xf1, ys) * 1.4247691104677813f;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SinglePerlin(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SinglePerlin(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        int x0 = FastFloor(x);
        int y0 = FastFloor(y);
        int z0 = FastFloor(z);

        float xd0 = (float)(x - x0);
        float yd0 = (float)(y - y0);
        float zd0 = (float)(z - z0);
        float xd1 = xd0 - 1;
        float yd1 = yd0 - 1;
        float zd1 = zd0 - 1;

        float xs = InterpQuintic(xd0);
        float ys = InterpQuintic(yd0);
        float zs = InterpQuintic(zd0);

        x0 *= PrimeX;
        y0 *= PrimeY;
        z0 *= PrimeZ;
        int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;
        int z1 = z0 + PrimeZ;

        float xf00 = Lerp(GradCoord(seed, x0, y0, z0, xd0, yd0, zd0), GradCoord(seed, x1, y0, z0, xd1, yd0, zd0), xs);
//      float xf00 = Lerp(GradCoord(seed, x0, y0, z0, xd0, yd0, zd0), GradCoord(seed, x1, y0, z0, xd1, yd0, zd0), xs);
        float xf10 = Lerp(GradCoord(seed, x0, y1, z0, xd0, yd1, zd0), GradCoord(seed, x1, y1, z0, xd1, yd1, zd0), xs);
//      float xf10 = Lerp(GradCoord(seed, x0, y1, z0, xd0, yd1, zd0), GradCoord(seed, x1, y1, z0, xd1, yd1, zd0), xs);
        float xf01 = Lerp(GradCoord(seed, x0, y0, z1, xd0, yd0, zd1), GradCoord(seed, x1, y0, z1, xd1, yd0, zd1), xs);
//      float xf01 = Lerp(GradCoord(seed, x0, y0, z1, xd0, yd0, zd1), GradCoord(seed, x1, y0, z1, xd1, yd0, zd1), xs);
        float xf11 = Lerp(GradCoord(seed, x0, y1, z1, xd0, yd1, zd1), GradCoord(seed, x1, y1, z1, xd1, yd1, zd1), xs);
//      float xf11 = Lerp(GradCoord(seed, x0, y1, z1, xd0, yd1, zd1), GradCoord(seed, x1, y1, z1, xd1, yd1, zd1), xs);

        float yf0 = Lerp(xf00, xf10, ys);
//      float yf0 = Lerp(xf00, xf10, ys);
        float yf1 = Lerp(xf01, xf11, ys);
//      float yf1 = Lerp(xf01, xf11, ys);

        return Lerp(yf0, yf1, zs) * 0.964921414852142333984375f;
//      return Lerp(yf0, yf1, zs) * 0.964921414852142333984375f;
    }


    // Value Cubic Noise
    // Value Cubic Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleValueCubic(int seed, FNfloat x, FNfloat y) const
//  float SingleValueCubic(int seed, FNfloat x, FNfloat y) const
    {
        int x1 = FastFloor(x);
        int y1 = FastFloor(y);

        float xs = (float)(x - x1);
        float ys = (float)(y - y1);

        x1 *= PrimeX;
        y1 *= PrimeY;
        int x0 = x1 - PrimeX;
        int y0 = y1 - PrimeY;
        int x2 = x1 + PrimeX;
        int y2 = y1 + PrimeY;
        int x3 = x1 + (int)((long)PrimeX << 1);
        int y3 = y1 + (int)((long)PrimeY << 1);

        return CubicLerp(
            CubicLerp(ValCoord(seed, x0, y0), ValCoord(seed, x1, y0), ValCoord(seed, x2, y0), ValCoord(seed, x3, y0),
                      xs),
            CubicLerp(ValCoord(seed, x0, y1), ValCoord(seed, x1, y1), ValCoord(seed, x2, y1), ValCoord(seed, x3, y1),
                      xs),
            CubicLerp(ValCoord(seed, x0, y2), ValCoord(seed, x1, y2), ValCoord(seed, x2, y2), ValCoord(seed, x3, y2),
                      xs),
            CubicLerp(ValCoord(seed, x0, y3), ValCoord(seed, x1, y3), ValCoord(seed, x2, y3), ValCoord(seed, x3, y3),
                      xs),
            ys) * (1 / (1.5f * 1.5f));
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleValueCubic(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SingleValueCubic(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        int x1 = FastFloor(x);
        int y1 = FastFloor(y);
        int z1 = FastFloor(z);

        float xs = (float)(x - x1);
        float ys = (float)(y - y1);
        float zs = (float)(z - z1);

        x1 *= PrimeX;
        y1 *= PrimeY;
        z1 *= PrimeZ;

        int x0 = x1 - PrimeX;
        int y0 = y1 - PrimeY;
        int z0 = z1 - PrimeZ;
        int x2 = x1 + PrimeX;
        int y2 = y1 + PrimeY;
        int z2 = z1 + PrimeZ;
        int x3 = x1 + (int)((long)PrimeX << 1);
        int y3 = y1 + (int)((long)PrimeY << 1);
        int z3 = z1 + (int)((long)PrimeZ << 1);


        return CubicLerp(
            CubicLerp(
                CubicLerp(ValCoord(seed, x0, y0, z0), ValCoord(seed, x1, y0, z0), ValCoord(seed, x2, y0, z0), ValCoord(seed, x3, y0, z0), xs),
                CubicLerp(ValCoord(seed, x0, y1, z0), ValCoord(seed, x1, y1, z0), ValCoord(seed, x2, y1, z0), ValCoord(seed, x3, y1, z0), xs),
                CubicLerp(ValCoord(seed, x0, y2, z0), ValCoord(seed, x1, y2, z0), ValCoord(seed, x2, y2, z0), ValCoord(seed, x3, y2, z0), xs),
                CubicLerp(ValCoord(seed, x0, y3, z0), ValCoord(seed, x1, y3, z0), ValCoord(seed, x2, y3, z0), ValCoord(seed, x3, y3, z0), xs),
                ys),
            CubicLerp(
                CubicLerp(ValCoord(seed, x0, y0, z1), ValCoord(seed, x1, y0, z1), ValCoord(seed, x2, y0, z1), ValCoord(seed, x3, y0, z1), xs),
                CubicLerp(ValCoord(seed, x0, y1, z1), ValCoord(seed, x1, y1, z1), ValCoord(seed, x2, y1, z1), ValCoord(seed, x3, y1, z1), xs),
                CubicLerp(ValCoord(seed, x0, y2, z1), ValCoord(seed, x1, y2, z1), ValCoord(seed, x2, y2, z1), ValCoord(seed, x3, y2, z1), xs),
                CubicLerp(ValCoord(seed, x0, y3, z1), ValCoord(seed, x1, y3, z1), ValCoord(seed, x2, y3, z1), ValCoord(seed, x3, y3, z1), xs),
                ys),
            CubicLerp(
                CubicLerp(ValCoord(seed, x0, y0, z2), ValCoord(seed, x1, y0, z2), ValCoord(seed, x2, y0, z2), ValCoord(seed, x3, y0, z2), xs),
                CubicLerp(ValCoord(seed, x0, y1, z2), ValCoord(seed, x1, y1, z2), ValCoord(seed, x2, y1, z2), ValCoord(seed, x3, y1, z2), xs),
                CubicLerp(ValCoord(seed, x0, y2, z2), ValCoord(seed, x1, y2, z2), ValCoord(seed, x2, y2, z2), ValCoord(seed, x3, y2, z2), xs),
                CubicLerp(ValCoord(seed, x0, y3, z2), ValCoord(seed, x1, y3, z2), ValCoord(seed, x2, y3, z2), ValCoord(seed, x3, y3, z2), xs),
                ys),
            CubicLerp(
                CubicLerp(ValCoord(seed, x0, y0, z3), ValCoord(seed, x1, y0, z3), ValCoord(seed, x2, y0, z3), ValCoord(seed, x3, y0, z3), xs),
                CubicLerp(ValCoord(seed, x0, y1, z3), ValCoord(seed, x1, y1, z3), ValCoord(seed, x2, y1, z3), ValCoord(seed, x3, y1, z3), xs),
                CubicLerp(ValCoord(seed, x0, y2, z3), ValCoord(seed, x1, y2, z3), ValCoord(seed, x2, y2, z3), ValCoord(seed, x3, y2, z3), xs),
                CubicLerp(ValCoord(seed, x0, y3, z3), ValCoord(seed, x1, y3, z3), ValCoord(seed, x2, y3, z3), ValCoord(seed, x3, y3, z3), xs),
                ys),
            zs) * (1 / (1.5f * 1.5f * 1.5f));
    }


    // Value Noise
    // Value Noise

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleValue(int seed, FNfloat x, FNfloat y) const
//  float SingleValue(int seed, FNfloat x, FNfloat y) const
    {
        int x0 = FastFloor(x);
//      int x0 = FastFloor(x);
        int y0 = FastFloor(y);
//      int y0 = FastFloor(y);

        float xs = InterpHermite((float)(x - x0));
//      float xs = InterpHermite((float)(x - x0));
        float ys = InterpHermite((float)(y - y0));
//      float ys = InterpHermite((float)(y - y0));

        x0 *= PrimeX;
//      x0 *= PrimeX;
        y0 *= PrimeY;
//      y0 *= PrimeY;
        int x1 = x0 + PrimeX;
//      int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;
//      int y1 = y0 + PrimeY;

        float xf0 = Lerp(ValCoord(seed, x0, y0), ValCoord(seed, x1, y0), xs);
//      float xf0 = Lerp(ValCoord(seed, x0, y0), ValCoord(seed, x1, y0), xs);
        float xf1 = Lerp(ValCoord(seed, x0, y1), ValCoord(seed, x1, y1), xs);
//      float xf1 = Lerp(ValCoord(seed, x0, y1), ValCoord(seed, x1, y1), xs);

        return Lerp(xf0, xf1, ys);
//      return Lerp(xf0, xf1, ys);
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    float SingleValue(int seed, FNfloat x, FNfloat y, FNfloat z) const
//  float SingleValue(int seed, FNfloat x, FNfloat y, FNfloat z) const
    {
        int x0 = FastFloor(x);
//      int x0 = FastFloor(x);
        int y0 = FastFloor(y);
//      int y0 = FastFloor(y);
        int z0 = FastFloor(z);
//      int z0 = FastFloor(z);

        float xs = InterpHermite((float)(x - x0));
//      float xs = InterpHermite((float)(x - x0));
        float ys = InterpHermite((float)(y - y0));
//      float ys = InterpHermite((float)(y - y0));
        float zs = InterpHermite((float)(z - z0));
//      float zs = InterpHermite((float)(z - z0));

        x0 *= PrimeX;
//      x0 *= PrimeX;
        y0 *= PrimeY;
//      y0 *= PrimeY;
        z0 *= PrimeZ;
//      z0 *= PrimeZ;
        int x1 = x0 + PrimeX;
//      int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;
//      int y1 = y0 + PrimeY;
        int z1 = z0 + PrimeZ;
//      int z1 = z0 + PrimeZ;

        float xf00 = Lerp(ValCoord(seed, x0, y0, z0), ValCoord(seed, x1, y0, z0), xs);
//      float xf00 = Lerp(ValCoord(seed, x0, y0, z0), ValCoord(seed, x1, y0, z0), xs);
        float xf10 = Lerp(ValCoord(seed, x0, y1, z0), ValCoord(seed, x1, y1, z0), xs);
//      float xf10 = Lerp(ValCoord(seed, x0, y1, z0), ValCoord(seed, x1, y1, z0), xs);
        float xf01 = Lerp(ValCoord(seed, x0, y0, z1), ValCoord(seed, x1, y0, z1), xs);
//      float xf01 = Lerp(ValCoord(seed, x0, y0, z1), ValCoord(seed, x1, y0, z1), xs);
        float xf11 = Lerp(ValCoord(seed, x0, y1, z1), ValCoord(seed, x1, y1, z1), xs);
//      float xf11 = Lerp(ValCoord(seed, x0, y1, z1), ValCoord(seed, x1, y1, z1), xs);

        float yf0 = Lerp(xf00, xf10, ys);
//      float yf0 = Lerp(xf00, xf10, ys);
        float yf1 = Lerp(xf01, xf11, ys);
//      float yf1 = Lerp(xf01, xf11, ys);

        return Lerp(yf0, yf1, zs);
//      return Lerp(yf0, yf1, zs);
    }


    // Domain Warp
    // Domain Warp

    template <typename FNfloat>
//  template <typename FNfloat>
    void DoSingleDomainWarp(int seed, float amp, float freq, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr) const
//  void DoSingleDomainWarp(int seed, float amp, float freq, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr) const
    {
        switch (mDomainWarpType)
//      switch (mDomainWarpType)
        {
        case DomainWarpType_OpenSimplex2:
//      case DomainWarpType_OpenSimplex2:
            {
                SingleDomainWarpSimplexGradient(seed, amp * 38.283687591552734375f, freq, x, y, xr, yr, false);
//              SingleDomainWarpSimplexGradient(seed, amp * 38.283687591552734375f, freq, x, y, xr, yr, false);
            }
            break;
//          break;
        case DomainWarpType_OpenSimplex2Reduced:
//      case DomainWarpType_OpenSimplex2Reduced:
            {
                SingleDomainWarpSimplexGradient(seed, amp * 16.0f, freq, x, y, xr, yr, true);
//              SingleDomainWarpSimplexGradient(seed, amp * 16.0f, freq, x, y, xr, yr, true);
            }
            break;
//          break;
        case DomainWarpType_BasicGrid:
//      case DomainWarpType_BasicGrid:
            {
                SingleDomainWarpBasicGrid(seed, amp, freq, x, y, xr, yr);
//              SingleDomainWarpBasicGrid(seed, amp, freq, x, y, xr, yr);
            }
            break;
//          break;
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void DoSingleDomainWarp(int seed, float amp, float freq, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr) const
//  void DoSingleDomainWarp(int seed, float amp, float freq, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr) const
    {
        switch (mDomainWarpType)
//      switch (mDomainWarpType)
        {
        case DomainWarpType_OpenSimplex2:
//      case DomainWarpType_OpenSimplex2:
            {
                SingleDomainWarpOpenSimplex2Gradient(seed, amp * 32.69428253173828125f, freq, x, y, z, xr, yr, zr, false);
//              SingleDomainWarpOpenSimplex2Gradient(seed, amp * 32.69428253173828125f, freq, x, y, z, xr, yr, zr, false);
            }
            break;
//          break;
        case DomainWarpType_OpenSimplex2Reduced:
//      case DomainWarpType_OpenSimplex2Reduced:
            {
                SingleDomainWarpOpenSimplex2Gradient(seed, amp * 7.71604938271605f, freq, x, y, z, xr, yr, zr, true);
//              SingleDomainWarpOpenSimplex2Gradient(seed, amp * 7.71604938271605f, freq, x, y, z, xr, yr, zr, true);
            }
            break;
//          break;
        case DomainWarpType_BasicGrid:
//      case DomainWarpType_BasicGrid:
            {
                SingleDomainWarpBasicGrid(seed, amp, freq, x, y, z, xr, yr, zr);
//              SingleDomainWarpBasicGrid(seed, amp, freq, x, y, z, xr, yr, zr);
            }
            break;
//          break;
        }
    }


    // Domain Warp Single Wrapper
    // Domain Warp Single Wrapper

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpSingle(FNfloat& x, FNfloat& y) const
//  void DomainWarpSingle(FNfloat& x, FNfloat& y) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        FNfloat xs = x;
//      FNfloat xs = x;
        FNfloat ys = y;
//      FNfloat ys = y;
        TransformDomainWarpCoordinate(xs, ys);
//      TransformDomainWarpCoordinate(xs, ys);

        DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);
//      DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpSingle(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void DomainWarpSingle(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        FNfloat xs = x;
//      FNfloat xs = x;
        FNfloat ys = y;
//      FNfloat ys = y;
        FNfloat zs = z;
//      FNfloat zs = z;
        TransformDomainWarpCoordinate(xs, ys, zs);
//      TransformDomainWarpCoordinate(xs, ys, zs);

        DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);
//      DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);
    }


    // Domain Warp Fractal Progressive
    // Domain Warp Fractal Progressive

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpFractalProgressive(FNfloat& x, FNfloat& y) const
//  void DomainWarpFractalProgressive(FNfloat& x, FNfloat& y) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            FNfloat xs = x;
//          FNfloat xs = x;
            FNfloat ys = y;
//          FNfloat ys = y;
            TransformDomainWarpCoordinate(xs, ys);
//          TransformDomainWarpCoordinate(xs, ys);

            DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);
//          DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);

            seed++;
//          seed++;
            amp *= mGain;
//          amp *= mGain;
            freq *= mLacunarity;
//          freq *= mLacunarity;
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpFractalProgressive(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void DomainWarpFractalProgressive(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            FNfloat xs = x;
//          FNfloat xs = x;
            FNfloat ys = y;
//          FNfloat ys = y;
            FNfloat zs = z;
//          FNfloat zs = z;
            TransformDomainWarpCoordinate(xs, ys, zs);
//          TransformDomainWarpCoordinate(xs, ys, zs);

            DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);
//          DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);

            seed++;
//          seed++;
            amp *= mGain;
//          amp *= mGain;
            freq *= mLacunarity;
//          freq *= mLacunarity;
        }
    }


    // Domain Warp Fractal Independant
    // Domain Warp Fractal Independant

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpFractalIndependent(FNfloat& x, FNfloat& y) const
//  void DomainWarpFractalIndependent(FNfloat& x, FNfloat& y) const
    {
        FNfloat xs = x;
//      FNfloat xs = x;
        FNfloat ys = y;
//      FNfloat ys = y;
        TransformDomainWarpCoordinate(xs, ys);
//      TransformDomainWarpCoordinate(xs, ys);

        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);
//          DoSingleDomainWarp(seed, amp, freq, xs, ys, x, y);

            seed++;
//          seed++;
            amp *= mGain;
//          amp *= mGain;
            freq *= mLacunarity;
//          freq *= mLacunarity;
        }
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void DomainWarpFractalIndependent(FNfloat& x, FNfloat& y, FNfloat& z) const
//  void DomainWarpFractalIndependent(FNfloat& x, FNfloat& y, FNfloat& z) const
    {
        FNfloat xs = x;
//      FNfloat xs = x;
        FNfloat ys = y;
//      FNfloat ys = y;
        FNfloat zs = z;
//      FNfloat zs = z;
        TransformDomainWarpCoordinate(xs, ys, zs);
//      TransformDomainWarpCoordinate(xs, ys, zs);

        int seed = mSeed;
//      int seed = mSeed;
        float amp = mDomainWarpAmp * mFractalBounding;
//      float amp = mDomainWarpAmp * mFractalBounding;
        float freq = mFrequency;
//      float freq = mFrequency;

        for (int i = 0; i < mOctaves; i++)
//      for (int i = 0; i < mOctaves; i++)
        {
            DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);
//          DoSingleDomainWarp(seed, amp, freq, xs, ys, zs, x, y, z);

            seed++;
//          seed++;
            amp *= mGain;
//          amp *= mGain;
            freq *= mLacunarity;
//          freq *= mLacunarity;
        }
    }


    // Domain Warp Basic Grid
    // Domain Warp Basic Grid

    template <typename FNfloat>
//  template <typename FNfloat>
    void SingleDomainWarpBasicGrid(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr) const
//  void SingleDomainWarpBasicGrid(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr) const
    {
        FNfloat xf = x * frequency;
//      FNfloat xf = x * frequency;
        FNfloat yf = y * frequency;
//      FNfloat yf = y * frequency;

        int x0 = FastFloor(xf);
//      int x0 = FastFloor(xf);
        int y0 = FastFloor(yf);
//      int y0 = FastFloor(yf);

        float xs = InterpHermite((float)(xf - x0));
//      float xs = InterpHermite((float)(xf - x0));
        float ys = InterpHermite((float)(yf - y0));
//      float ys = InterpHermite((float)(yf - y0));

        x0 *= PrimeX;
//      x0 *= PrimeX;
        y0 *= PrimeY;
//      y0 *= PrimeY;
        int x1 = x0 + PrimeX;
//      int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;
//      int y1 = y0 + PrimeY;

        int hash0 = Hash(seed, x0, y0) & (255 << 1);
//      int hash0 = Hash(seed, x0, y0) & (255 << 1);
        int hash1 = Hash(seed, x1, y0) & (255 << 1);
//      int hash1 = Hash(seed, x1, y0) & (255 << 1);

        float lx0x = Lerp(Lookup<float>::RandVecs2D[hash0    ], Lookup<float>::RandVecs2D[hash1    ], xs);
//      float lx0x = Lerp(Lookup<float>::RandVecs2D[hash0    ], Lookup<float>::RandVecs2D[hash1    ], xs);
        float ly0x = Lerp(Lookup<float>::RandVecs2D[hash0 | 1], Lookup<float>::RandVecs2D[hash1 | 1], xs);
//      float ly0x = Lerp(Lookup<float>::RandVecs2D[hash0 | 1], Lookup<float>::RandVecs2D[hash1 | 1], xs);

        hash0 = Hash(seed, x0, y1) & (255 << 1);
//      hash0 = Hash(seed, x0, y1) & (255 << 1);
        hash1 = Hash(seed, x1, y1) & (255 << 1);
//      hash1 = Hash(seed, x1, y1) & (255 << 1);

        float lx1x = Lerp(Lookup<float>::RandVecs2D[hash0    ], Lookup<float>::RandVecs2D[hash1    ], xs);
//      float lx1x = Lerp(Lookup<float>::RandVecs2D[hash0    ], Lookup<float>::RandVecs2D[hash1    ], xs);
        float ly1x = Lerp(Lookup<float>::RandVecs2D[hash0 | 1], Lookup<float>::RandVecs2D[hash1 | 1], xs);
//      float ly1x = Lerp(Lookup<float>::RandVecs2D[hash0 | 1], Lookup<float>::RandVecs2D[hash1 | 1], xs);

        xr += Lerp(lx0x, lx1x, ys) * warpAmp;
//      xr += Lerp(lx0x, lx1x, ys) * warpAmp;
        yr += Lerp(ly0x, ly1x, ys) * warpAmp;
//      yr += Lerp(ly0x, ly1x, ys) * warpAmp;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void SingleDomainWarpBasicGrid(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr) const
//  void SingleDomainWarpBasicGrid(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr) const
    {
        FNfloat xf = x * frequency;
//      FNfloat xf = x * frequency;
        FNfloat yf = y * frequency;
//      FNfloat yf = y * frequency;
        FNfloat zf = z * frequency;
//      FNfloat zf = z * frequency;

        int x0 = FastFloor(xf);
//      int x0 = FastFloor(xf);
        int y0 = FastFloor(yf);
//      int y0 = FastFloor(yf);
        int z0 = FastFloor(zf);
//      int z0 = FastFloor(zf);

        float xs = InterpHermite((float)(xf - x0));
//      float xs = InterpHermite((float)(xf - x0));
        float ys = InterpHermite((float)(yf - y0));
//      float ys = InterpHermite((float)(yf - y0));
        float zs = InterpHermite((float)(zf - z0));
//      float zs = InterpHermite((float)(zf - z0));

        x0 *= PrimeX;
//      x0 *= PrimeX;
        y0 *= PrimeY;
//      y0 *= PrimeY;
        z0 *= PrimeZ;
//      z0 *= PrimeZ;
        int x1 = x0 + PrimeX;
//      int x1 = x0 + PrimeX;
        int y1 = y0 + PrimeY;
//      int y1 = y0 + PrimeY;
        int z1 = z0 + PrimeZ;
//      int z1 = z0 + PrimeZ;

        int hash0 = Hash(seed, x0, y0, z0) & (255 << 2);
//      int hash0 = Hash(seed, x0, y0, z0) & (255 << 2);
        int hash1 = Hash(seed, x1, y0, z0) & (255 << 2);
//      int hash1 = Hash(seed, x1, y0, z0) & (255 << 2);

        float lx0x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
//      float lx0x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
        float ly0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
//      float ly0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
        float lz0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);
//      float lz0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);

        hash0 = Hash(seed, x0, y1, z0) & (255 << 2);
//      hash0 = Hash(seed, x0, y1, z0) & (255 << 2);
        hash1 = Hash(seed, x1, y1, z0) & (255 << 2);
//      hash1 = Hash(seed, x1, y1, z0) & (255 << 2);

        float lx1x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
//      float lx1x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
        float ly1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
//      float ly1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
        float lz1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);
//      float lz1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);

        float lx0y = Lerp(lx0x, lx1x, ys);
//      float lx0y = Lerp(lx0x, lx1x, ys);
        float ly0y = Lerp(ly0x, ly1x, ys);
//      float ly0y = Lerp(ly0x, ly1x, ys);
        float lz0y = Lerp(lz0x, lz1x, ys);
//      float lz0y = Lerp(lz0x, lz1x, ys);

        hash0 = Hash(seed, x0, y0, z1) & (255 << 2);
//      hash0 = Hash(seed, x0, y0, z1) & (255 << 2);
        hash1 = Hash(seed, x1, y0, z1) & (255 << 2);
//      hash1 = Hash(seed, x1, y0, z1) & (255 << 2);

        lx0x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
//      lx0x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
        ly0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
//      ly0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
        lz0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);
//      lz0x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);

        hash0 = Hash(seed, x0, y1, z1) & (255 << 2);
//      hash0 = Hash(seed, x0, y1, z1) & (255 << 2);
        hash1 = Hash(seed, x1, y1, z1) & (255 << 2);
//      hash1 = Hash(seed, x1, y1, z1) & (255 << 2);

        lx1x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
//      lx1x = Lerp(Lookup<float>::RandVecs3D[hash0    ], Lookup<float>::RandVecs3D[hash1    ], xs);
        ly1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
//      ly1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 1], Lookup<float>::RandVecs3D[hash1 | 1], xs);
        lz1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);
//      lz1x = Lerp(Lookup<float>::RandVecs3D[hash0 | 2], Lookup<float>::RandVecs3D[hash1 | 2], xs);

        xr += Lerp(lx0y, Lerp(lx0x, lx1x, ys), zs) * warpAmp;
//      xr += Lerp(lx0y, Lerp(lx0x, lx1x, ys), zs) * warpAmp;
        yr += Lerp(ly0y, Lerp(ly0x, ly1x, ys), zs) * warpAmp;
//      yr += Lerp(ly0y, Lerp(ly0x, ly1x, ys), zs) * warpAmp;
        zr += Lerp(lz0y, Lerp(lz0x, lz1x, ys), zs) * warpAmp;
//      zr += Lerp(lz0y, Lerp(lz0x, lz1x, ys), zs) * warpAmp;
    }


    // Domain Warp Simplex/OpenSimplex2
    // Domain Warp Simplex/OpenSimplex2

    template <typename FNfloat>
//  template <typename FNfloat>
    void SingleDomainWarpSimplexGradient(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr, bool outGradOnly) const
//  void SingleDomainWarpSimplexGradient(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat& xr, FNfloat& yr, bool outGradOnly) const
    {
        const float SQRT3 = 1.7320508075688772935274463415059f;
//      const float SQRT3 = 1.7320508075688772935274463415059f;
        const float G2 = (3 - SQRT3) / 6;
//      const float G2 = (3 - SQRT3) / 6;

        x *= frequency;
//      x *= frequency;
        y *= frequency;
//      y *= frequency;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
//       * --- Skew moved to TransformNoiseCoordinate method ---
         * const FNfloat F2 = 0.5f * (SQRT3 - 1);
//       * const FNfloat F2 = 0.5f * (SQRT3 - 1);
         * FNfloat s = (x + y) * F2;
//       * FNfloat s = (x + y) * F2;
         * x += s; y += s;
//       * x += s; y += s;
        */

        int i = FastFloor(x);
//      int i = FastFloor(x);
        int j = FastFloor(y);
//      int j = FastFloor(y);
        float xi = (float)(x - i);
//      float xi = (float)(x - i);
        float yi = (float)(y - j);
//      float yi = (float)(y - j);

        float t = (xi + yi) * G2;
//      float t = (xi + yi) * G2;
        float x0 = (float)(xi - t);
//      float x0 = (float)(xi - t);
        float y0 = (float)(yi - t);
//      float y0 = (float)(yi - t);

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;

        float vx, vy;
//      float vx, vy;
        vx = vy = 0;
//      vx = vy = 0;

        float a = 0.5f - x0 * x0 - y0 * y0;
//      float a = 0.5f - x0 * x0 - y0 * y0;
        if (a > 0)
//      if (a > 0)
        {
            float aaaa = (a * a) * (a * a);
//          float aaaa = (a * a) * (a * a);
            float xo, yo;
//          float xo, yo;
            if (outGradOnly)
//          if (outGradOnly)
            {
                GradCoordOut(seed, i, j, xo, yo);
//              GradCoordOut(seed, i, j, xo, yo);
            }
            else
//          else
            {
                GradCoordDual(seed, i, j, x0, y0, xo, yo);
//              GradCoordDual(seed, i, j, x0, y0, xo, yo);
            }
            vx += aaaa * xo;
//          vx += aaaa * xo;
            vy += aaaa * yo;
//          vy += aaaa * yo;
        }

        float c = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a);
//      float c = (float)(2 * (1 - 2 * G2) * (1 / G2 - 2)) * t + ((float)(-2 * (1 - 2 * G2) * (1 - 2 * G2)) + a);
        if (c > 0)
//      if (c > 0)
        {
            float x2 = x0 + (2 * (float)G2 - 1);
//          float x2 = x0 + (2 * (float)G2 - 1);
            float y2 = y0 + (2 * (float)G2 - 1);
//          float y2 = y0 + (2 * (float)G2 - 1);
            float cccc = (c * c) * (c * c);
//          float cccc = (c * c) * (c * c);
            float xo, yo;
//          float xo, yo;
            if (outGradOnly)
//          if (outGradOnly)
            {
                GradCoordOut(seed, i + PrimeX, j + PrimeY, xo, yo);
//              GradCoordOut(seed, i + PrimeX, j + PrimeY, xo, yo);
            }
            else
//          else
            {
                GradCoordDual(seed, i + PrimeX, j + PrimeY, x2, y2, xo, yo);
//              GradCoordDual(seed, i + PrimeX, j + PrimeY, x2, y2, xo, yo);
            }
            vx += cccc * xo;
//          vx += cccc * xo;
            vy += cccc * yo;
//          vy += cccc * yo;
        }

        if (y0 > x0)
//      if (y0 > x0)
        {
            float x1 = x0 + (float)G2;
//          float x1 = x0 + (float)G2;
            float y1 = y0 + ((float)G2 - 1);
//          float y1 = y0 + ((float)G2 - 1);
            float b = 0.5f - x1 * x1 - y1 * y1;
//          float b = 0.5f - x1 * x1 - y1 * y1;
            if (b > 0)
//          if (b > 0)
            {
                float bbbb = (b * b) * (b * b);
//              float bbbb = (b * b) * (b * b);
                float xo, yo;
//              float xo, yo;
                if (outGradOnly)
//              if (outGradOnly)
                {
                    GradCoordOut(seed, i, j + PrimeY, xo, yo);
//                  GradCoordOut(seed, i, j + PrimeY, xo, yo);
                }
                else
//              else
                {
                    GradCoordDual(seed, i, j + PrimeY, x1, y1, xo, yo);
//                  GradCoordDual(seed, i, j + PrimeY, x1, y1, xo, yo);
                }
                vx += bbbb * xo;
//              vx += bbbb * xo;
                vy += bbbb * yo;
//              vy += bbbb * yo;
            }
        }
        else
//      else
        {
            float x1 = x0 + ((float)G2 - 1);
//          float x1 = x0 + ((float)G2 - 1);
            float y1 = y0 + (float)G2;
//          float y1 = y0 + (float)G2;
            float b = 0.5f - x1 * x1 - y1 * y1;
//          float b = 0.5f - x1 * x1 - y1 * y1;
            if (b > 0)
//          if (b > 0)
            {
                float bbbb = (b * b) * (b * b);
//              float bbbb = (b * b) * (b * b);
                float xo, yo;
//              float xo, yo;
                if (outGradOnly)
//              if (outGradOnly)
                {
                    GradCoordOut(seed, i + PrimeX, j, xo, yo);
//                  GradCoordOut(seed, i + PrimeX, j, xo, yo);
                }
                else
//              else
                {
                    GradCoordDual(seed, i + PrimeX, j, x1, y1, xo, yo);
//                  GradCoordDual(seed, i + PrimeX, j, x1, y1, xo, yo);
                }
                vx += bbbb * xo;
//              vx += bbbb * xo;
                vy += bbbb * yo;
//              vy += bbbb * yo;
            }
        }

        xr += vx * warpAmp;
//      xr += vx * warpAmp;
        yr += vy * warpAmp;
//      yr += vy * warpAmp;
    }

    template <typename FNfloat>
//  template <typename FNfloat>
    void SingleDomainWarpOpenSimplex2Gradient(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr, bool outGradOnly) const
//  void SingleDomainWarpOpenSimplex2Gradient(int seed, float warpAmp, float frequency, FNfloat x, FNfloat y, FNfloat z, FNfloat& xr, FNfloat& yr, FNfloat& zr, bool outGradOnly) const
    {
        x *= frequency;
//      x *= frequency;
        y *= frequency;
//      y *= frequency;
        z *= frequency;
//      z *= frequency;

        /*
         * --- Rotation moved to TransformDomainWarpCoordinate method ---
         * --- Rotation moved to TransformDomainWarpCoordinate method ---
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * const FNfloat R3 = (FNfloat)(2.0 / 3.0);
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * FNfloat r = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         * x = r - x; y = r - y; z = r - z;
        */

        int i = FastRound(x);
//      int i = FastRound(x);
        int j = FastRound(y);
//      int j = FastRound(y);
        int k = FastRound(z);
//      int k = FastRound(z);
        float x0 = (float)x - i;
//      float x0 = (float)x - i;
        float y0 = (float)y - j;
//      float y0 = (float)y - j;
        float z0 = (float)z - k;
//      float z0 = (float)z - k;

        int xNSign = (int)(-x0 - 1.0f) | 1;
//      int xNSign = (int)(-x0 - 1.0f) | 1;
        int yNSign = (int)(-y0 - 1.0f) | 1;
//      int yNSign = (int)(-y0 - 1.0f) | 1;
        int zNSign = (int)(-z0 - 1.0f) | 1;
//      int zNSign = (int)(-z0 - 1.0f) | 1;

        float ax0 = xNSign * -x0;
//      float ax0 = xNSign * -x0;
        float ay0 = yNSign * -y0;
//      float ay0 = yNSign * -y0;
        float az0 = zNSign * -z0;
//      float az0 = zNSign * -z0;

        i *= PrimeX;
//      i *= PrimeX;
        j *= PrimeY;
//      j *= PrimeY;
        k *= PrimeZ;
//      k *= PrimeZ;

        float vx, vy, vz;
//      float vx, vy, vz;
        vx = vy = vz = 0;
//      vx = vy = vz = 0;

        float a = (0.6f - x0 * x0) - (y0 * y0 + z0 * z0);
//      float a = (0.6f - x0 * x0) - (y0 * y0 + z0 * z0);
        for (int l = 0; l < 2; l++)
//      for (int l = 0; l < 2; l++)
        {
            if (a > 0)
//          if (a > 0)
            {
                float aaaa = (a * a) * (a * a);
//              float aaaa = (a * a) * (a * a);
                float xo, yo, zo;
//              float xo, yo, zo;
                if (outGradOnly)
//              if (outGradOnly)
                {
                    GradCoordOut(seed, i, j, k, xo, yo, zo);
//                  GradCoordOut(seed, i, j, k, xo, yo, zo);
                }
                else
//              else
                {
                    GradCoordDual(seed, i, j, k, x0, y0, z0, xo, yo, zo);
//                  GradCoordDual(seed, i, j, k, x0, y0, z0, xo, yo, zo);
                }
                vx += aaaa * xo;
//              vx += aaaa * xo;
                vy += aaaa * yo;
//              vy += aaaa * yo;
                vz += aaaa * zo;
//              vz += aaaa * zo;
            }

            float b = a + 1;
//          float b = a + 1;
            int i1 = i;
//          int i1 = i;
            int j1 = j;
//          int j1 = j;
            int k1 = k;
//          int k1 = k;
            float x1 = x0;
//          float x1 = x0;
            float y1 = y0;
//          float y1 = y0;
            float z1 = z0;
//          float z1 = z0;

            if (ax0 >= ay0 && ax0 >= az0)
//          if (ax0 >= ay0 && ax0 >= az0)
            {
                x1 += xNSign;
//              x1 += xNSign;
                b  -= xNSign * 2 * x1;
//              b  -= xNSign * 2 * x1;
                i1 -= xNSign * PrimeX;
//              i1 -= xNSign * PrimeX;
            }
            else
//          else
            if (ay0 >  ax0 && ay0 >= az0)
//          if (ay0 >  ax0 && ay0 >= az0)
            {
                y1 += yNSign;
//              y1 += yNSign;
                b  -= yNSign * 2 * y1;
//              b  -= yNSign * 2 * y1;
                j1 -= yNSign * PrimeY;
//              j1 -= yNSign * PrimeY;
            }
            else
//          else
            {
                z1 += zNSign;
//              z1 += zNSign;
                b  -= zNSign * 2 * z1;
//              b  -= zNSign * 2 * z1;
                k1 -= zNSign * PrimeZ;
//              k1 -= zNSign * PrimeZ;
            }

            if (b > 0)
//          if (b > 0)
            {
                float bbbb = (b * b) * (b * b);
//              float bbbb = (b * b) * (b * b);
                float xo, yo, zo;
//              float xo, yo, zo;
                if (outGradOnly)
//              if (outGradOnly)
                {
                    GradCoordOut(seed, i1, j1, k1, xo, yo, zo);
//                  GradCoordOut(seed, i1, j1, k1, xo, yo, zo);
                }
                else
//              else
                {
                    GradCoordDual(seed, i1, j1, k1, x1, y1, z1, xo, yo, zo);
//                  GradCoordDual(seed, i1, j1, k1, x1, y1, z1, xo, yo, zo);
                }
                vx += bbbb * xo;
//              vx += bbbb * xo;
                vy += bbbb * yo;
//              vy += bbbb * yo;
                vz += bbbb * zo;
//              vz += bbbb * zo;
            }

            if (l == 1) break;
//          if (l == 1) break;

            ax0 = 0.5f - ax0;
//          ax0 = 0.5f - ax0;
            ay0 = 0.5f - ay0;
//          ay0 = 0.5f - ay0;
            az0 = 0.5f - az0;
//          az0 = 0.5f - az0;

            x0 = xNSign * ax0;
//          x0 = xNSign * ax0;
            y0 = yNSign * ay0;
//          y0 = yNSign * ay0;
            z0 = zNSign * az0;
//          z0 = zNSign * az0;

            a += (0.75f - ax0) - (ay0 + az0);
//          a += (0.75f - ax0) - (ay0 + az0);

            i += (xNSign >> 1) & PrimeX;
//          i += (xNSign >> 1) & PrimeX;
            j += (yNSign >> 1) & PrimeY;
//          j += (yNSign >> 1) & PrimeY;
            k += (zNSign >> 1) & PrimeZ;
//          k += (zNSign >> 1) & PrimeZ;

            xNSign = -xNSign;
//          xNSign = -xNSign;
            yNSign = -yNSign;
//          yNSign = -yNSign;
            zNSign = -zNSign;
//          zNSign = -zNSign;

            seed += 1293373;
//          seed += 1293373;
        }

        xr += vx * warpAmp;
//      xr += vx * warpAmp;
        yr += vy * warpAmp;
//      yr += vy * warpAmp;
        zr += vz * warpAmp;
//      zr += vz * warpAmp;
    }
};

    template <>
//  template <>
    struct FastNoiseLite::Arguments_must_be_floating_point_values<float> {};
//  struct FastNoiseLite::Arguments_must_be_floating_point_values<float> {};
    template <>
//  template <>
    struct FastNoiseLite::Arguments_must_be_floating_point_values<double> {};
//  struct FastNoiseLite::Arguments_must_be_floating_point_values<double> {};
    template <>
//  template <>
    struct FastNoiseLite::Arguments_must_be_floating_point_values<long double> {};
//  struct FastNoiseLite::Arguments_must_be_floating_point_values<long double> {};

    template <typename T>
//  template <typename T>
    const T FastNoiseLite::Lookup<T>::Gradients2D[] =
//  const T FastNoiseLite::Lookup<T>::Gradients2D[] =
{
    +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
//  +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
    +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
//  +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
    +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
//  +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
    -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
//  -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
    -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
//  -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
    -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
//  -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
    +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
//  +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
    +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
//  +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
    +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
//  +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
    -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
//  -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
    -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
//  -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
    -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
//  -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
    +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
//  +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
    +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
//  +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
    +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
//  +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
    -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
//  -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
    -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
//  -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
    -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
//  -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
    +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
//  +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
    +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
//  +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
    +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
//  +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
    -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
//  -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
    -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
//  -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
    -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
//  -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
    +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
//  +0.130526192220052f, +0.991444861373810f, +0.382683432365090f, +0.923879532511287f, +0.608761429008721f, +0.793353340291235f, +0.793353340291235f, +0.608761429008721f,
    +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
//  +0.923879532511287f, +0.382683432365090f, +0.991444861373810f, +0.130526192220051f, +0.991444861373810f, -0.130526192220051f, +0.923879532511287f, -0.382683432365090f,
    +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
//  +0.793353340291235f, -0.608761429008720f, +0.608761429008721f, -0.793353340291235f, +0.382683432365090f, -0.923879532511287f, +0.130526192220052f, -0.991444861373810f,
    -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
//  -0.130526192220052f, -0.991444861373810f, -0.382683432365090f, -0.923879532511287f, -0.608761429008721f, -0.793353340291235f, -0.793353340291235f, -0.608761429008721f,
    -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
//  -0.923879532511287f, -0.382683432365090f, -0.991444861373810f, -0.130526192220052f, -0.991444861373810f, +0.130526192220051f, -0.923879532511287f, +0.382683432365090f,
    -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
//  -0.793353340291235f, +0.608761429008721f, -0.608761429008721f, +0.793353340291235f, -0.382683432365090f, +0.923879532511287f, -0.130526192220052f, +0.991444861373810f,
    +0.382683432365090f, +0.923879532511287f, +0.923879532511287f, +0.382683432365090f, +0.923879532511287f, -0.382683432365090f, +0.382683432365090f, -0.923879532511287f,
//  +0.382683432365090f, +0.923879532511287f, +0.923879532511287f, +0.382683432365090f, +0.923879532511287f, -0.382683432365090f, +0.382683432365090f, -0.923879532511287f,
    -0.382683432365090f, -0.923879532511287f, -0.923879532511287f, -0.382683432365090f, -0.923879532511287f, +0.382683432365090f, -0.382683432365090f, +0.923879532511287f,
//  -0.382683432365090f, -0.923879532511287f, -0.923879532511287f, -0.382683432365090f, -0.923879532511287f, +0.382683432365090f, -0.382683432365090f, +0.923879532511287f,
};

    template <typename T>
//  template <typename T>
    const T FastNoiseLite::Lookup<T>::RandVecs2D[] =
//  const T FastNoiseLite::Lookup<T>::RandVecs2D[] =
{
    -0.270022219800f, -0.962854091100f, +0.386309262700f, -0.922369315200f, +0.044448590060f, -0.999011673000f, -0.599252315800f, -0.800560217600f, -0.781928028800f, +0.623368717400f, +0.946467227100f, +0.322799919600f, -0.651414679700f, -0.758721895700f, +0.937847228900f, +0.347048376000f,
//  -0.270022219800f, -0.962854091100f, +0.386309262700f, -0.922369315200f, +0.044448590060f, -0.999011673000f, -0.599252315800f, -0.800560217600f, -0.781928028800f, +0.623368717400f, +0.946467227100f, +0.322799919600f, -0.651414679700f, -0.758721895700f, +0.937847228900f, +0.347048376000f,
    -0.849787595700f, -0.527125262300f, -0.879042592000f, +0.476743244700f, -0.892300288000f, -0.451442350800f, -0.379844434000f, -0.925050380200f, -0.995165083200f, +0.098216378900f, +0.772439780800f, -0.635088013600f, +0.757328332200f, -0.653034300200f, -0.992800452500f, -0.119780055000f,
//  -0.849787595700f, -0.527125262300f, -0.879042592000f, +0.476743244700f, -0.892300288000f, -0.451442350800f, -0.379844434000f, -0.925050380200f, -0.995165083200f, +0.098216378900f, +0.772439780800f, -0.635088013600f, +0.757328332200f, -0.653034300200f, -0.992800452500f, -0.119780055000f,
    -0.053266571300f, +0.998580328500f, +0.975425372600f, -0.220330076200f, -0.766501816300f, +0.642242139400f, +0.991636706000f, +0.129060618400f, -0.994696838000f, +0.102850378800f, -0.537920551300f, -0.842995540000f, +0.502281547100f, -0.864704138700f, +0.455982146100f, -0.889988922600f,
//  -0.053266571300f, +0.998580328500f, +0.975425372600f, -0.220330076200f, -0.766501816300f, +0.642242139400f, +0.991636706000f, +0.129060618400f, -0.994696838000f, +0.102850378800f, -0.537920551300f, -0.842995540000f, +0.502281547100f, -0.864704138700f, +0.455982146100f, -0.889988922600f,
    -0.865913122400f, -0.500194426600f, +0.087945840700f, -0.996125257700f, -0.505168498300f, +0.863020734600f, +0.775318522600f, -0.631570414600f, -0.692194461200f, +0.721711041800f, -0.519165944900f, -0.854673459100f, +0.897862288200f, -0.440276403500f, -0.170677410700f, +0.985326961700f,
//  -0.865913122400f, -0.500194426600f, +0.087945840700f, -0.996125257700f, -0.505168498300f, +0.863020734600f, +0.775318522600f, -0.631570414600f, -0.692194461200f, +0.721711041800f, -0.519165944900f, -0.854673459100f, +0.897862288200f, -0.440276403500f, -0.170677410700f, +0.985326961700f,
    -0.935343010600f, -0.353742070500f, -0.999240479800f, +0.038967467940f, -0.288206402100f, -0.957568310800f, -0.966381132900f, +0.257113799500f, -0.875971423800f, -0.482363000900f, -0.830312301800f, -0.557298377500f, +0.051101337550f, -0.998693473100f, -0.855837328100f, -0.517245075200f,
//  -0.935343010600f, -0.353742070500f, -0.999240479800f, +0.038967467940f, -0.288206402100f, -0.957568310800f, -0.966381132900f, +0.257113799500f, -0.875971423800f, -0.482363000900f, -0.830312301800f, -0.557298377500f, +0.051101337550f, -0.998693473100f, -0.855837328100f, -0.517245075200f,
    +0.098870252820f, +0.995100333200f, +0.918901608700f, +0.394486797600f, -0.243937589200f, -0.969790932400f, -0.812140938700f, -0.583461306100f, -0.991043136300f, +0.133542135500f, +0.849242398500f, -0.528003170900f, -0.971783899400f, -0.235872959100f, +0.994945720700f, +0.100414206800f,
//  +0.098870252820f, +0.995100333200f, +0.918901608700f, +0.394486797600f, -0.243937589200f, -0.969790932400f, -0.812140938700f, -0.583461306100f, -0.991043136300f, +0.133542135500f, +0.849242398500f, -0.528003170900f, -0.971783899400f, -0.235872959100f, +0.994945720700f, +0.100414206800f,
    +0.624106550800f, -0.781339243400f, +0.662910307000f, +0.748698821200f, -0.719741817600f, +0.694241828200f, -0.814337077500f, -0.580392215800f, +0.104521054000f, -0.994522674100f, -0.106592611300f, -0.994302778400f, +0.445799684000f, -0.895132750900f, +0.105547406000f, +0.994414272400f,
//  +0.624106550800f, -0.781339243400f, +0.662910307000f, +0.748698821200f, -0.719741817600f, +0.694241828200f, -0.814337077500f, -0.580392215800f, +0.104521054000f, -0.994522674100f, -0.106592611300f, -0.994302778400f, +0.445799684000f, -0.895132750900f, +0.105547406000f, +0.994414272400f,
    -0.992790267000f, +0.119864447700f, -0.833436640800f, +0.552615025000f, +0.911556156300f, -0.411175599900f, +0.828554490900f, -0.559908435100f, +0.721709765400f, -0.692195792100f, +0.494049267700f, -0.869433908400f, -0.365232127200f, -0.930916480300f, -0.969660675800f, +0.244454850100f,
//  -0.992790267000f, +0.119864447700f, -0.833436640800f, +0.552615025000f, +0.911556156300f, -0.411175599900f, +0.828554490900f, -0.559908435100f, +0.721709765400f, -0.692195792100f, +0.494049267700f, -0.869433908400f, -0.365232127200f, -0.930916480300f, -0.969660675800f, +0.244454850100f,
    +0.089255097310f, -0.996008799000f, +0.535407127600f, -0.844594108300f, -0.105357618600f, +0.994434398100f, -0.989028458600f, +0.147725110100f, +0.004856104961f, +0.999988209100f, +0.988559847800f, +0.150829133100f, +0.928612956200f, -0.371049831600f, -0.583239386300f, -0.812300325200f,
//  +0.089255097310f, -0.996008799000f, +0.535407127600f, -0.844594108300f, -0.105357618600f, +0.994434398100f, -0.989028458600f, +0.147725110100f, +0.004856104961f, +0.999988209100f, +0.988559847800f, +0.150829133100f, +0.928612956200f, -0.371049831600f, -0.583239386300f, -0.812300325200f,
    +0.301520750900f, +0.953459614600f, -0.957511052800f, +0.288396573800f, +0.971580215400f, -0.236710551100f, +0.229981792000f, +0.973194931800f, +0.955763816000f, -0.294135220700f, +0.740956116000f, +0.671553448500f, -0.997151378700f, -0.075426307640f, +0.690571066300f, -0.723264545200f,
//  +0.301520750900f, +0.953459614600f, -0.957511052800f, +0.288396573800f, +0.971580215400f, -0.236710551100f, +0.229981792000f, +0.973194931800f, +0.955763816000f, -0.294135220700f, +0.740956116000f, +0.671553448500f, -0.997151378700f, -0.075426307640f, +0.690571066300f, -0.723264545200f,
    -0.290713703000f, -0.956810087200f, +0.591277779100f, -0.806467970800f, -0.945459221200f, -0.325740481000f, +0.666445568100f, +0.745553690000f, +0.623613491200f, +0.781732827500f, +0.912699385100f, -0.408631658700f, -0.819176201100f, +0.573541935300f, -0.881274575900f, -0.472604614700f,
//  -0.290713703000f, -0.956810087200f, +0.591277779100f, -0.806467970800f, -0.945459221200f, -0.325740481000f, +0.666445568100f, +0.745553690000f, +0.623613491200f, +0.781732827500f, +0.912699385100f, -0.408631658700f, -0.819176201100f, +0.573541935300f, -0.881274575900f, -0.472604614700f,
    +0.995331362700f, +0.096516726510f, +0.985565084600f, -0.169296969900f, -0.849598088700f, +0.527430647200f, +0.617485394600f, -0.786582346300f, +0.850815637100f, +0.525464320000f, +0.998503245100f, -0.054692499260f, +0.197137156300f, -0.980375918500f, +0.660785574800f, -0.750574729200f,
//  +0.995331362700f, +0.096516726510f, +0.985565084600f, -0.169296969900f, -0.849598088700f, +0.527430647200f, +0.617485394600f, -0.786582346300f, +0.850815637100f, +0.525464320000f, +0.998503245100f, -0.054692499260f, +0.197137156300f, -0.980375918500f, +0.660785574800f, -0.750574729200f,
    -0.030974940630f, +0.999520161400f, -0.673166080100f, +0.739491331000f, -0.719501836200f, -0.694490538300f, +0.972751168900f, +0.231851597900f, +0.999705908800f, -0.024250690700f, +0.442178742900f, -0.896926953200f, +0.998135096100f, -0.061043673000f, -0.917366079900f, -0.398044564800f,
//  -0.030974940630f, +0.999520161400f, -0.673166080100f, +0.739491331000f, -0.719501836200f, -0.694490538300f, +0.972751168900f, +0.231851597900f, +0.999705908800f, -0.024250690700f, +0.442178742900f, -0.896926953200f, +0.998135096100f, -0.061043673000f, -0.917366079900f, -0.398044564800f,
    -0.815005663500f, -0.579452990700f, -0.878933130400f, +0.476945020200f, +0.015860582900f, +0.999874213000f, -0.809546447400f, +0.587055831700f, -0.916589890700f, -0.399828678600f, -0.802354256500f, +0.596848093800f, -0.517673791700f, +0.855578076700f, -0.815440730700f, -0.578840577900f,
//  -0.815005663500f, -0.579452990700f, -0.878933130400f, +0.476945020200f, +0.015860582900f, +0.999874213000f, -0.809546447400f, +0.587055831700f, -0.916589890700f, -0.399828678600f, -0.802354256500f, +0.596848093800f, -0.517673791700f, +0.855578076700f, -0.815440730700f, -0.578840577900f,
    +0.402201034700f, -0.915551379100f, -0.905255686800f, -0.424867204500f, +0.731744561900f, +0.681578972800f, -0.564763220100f, -0.825252994700f, -0.840327633500f, -0.542078839700f, -0.931428152700f, +0.363925262000f, +0.523819847200f, +0.851829071900f, +0.743280386900f, -0.668980019500f,
//  +0.402201034700f, -0.915551379100f, -0.905255686800f, -0.424867204500f, +0.731744561900f, +0.681578972800f, -0.564763220100f, -0.825252994700f, -0.840327633500f, -0.542078839700f, -0.931428152700f, +0.363925262000f, +0.523819847200f, +0.851829071900f, +0.743280386900f, -0.668980019500f,
    -0.985371561000f, -0.170419736900f, +0.460146873100f, +0.887842810000f, +0.825855404000f, +0.563881948300f, +0.618236609900f, +0.785992044600f, +0.833150286300f, -0.553046653000f, +0.150030750600f, +0.988681330800f, -0.662330369000f, -0.749211907500f, -0.668598664000f, +0.743623444000f,
//  -0.985371561000f, -0.170419736900f, +0.460146873100f, +0.887842810000f, +0.825855404000f, +0.563881948300f, +0.618236609900f, +0.785992044600f, +0.833150286300f, -0.553046653000f, +0.150030750600f, +0.988681330800f, -0.662330369000f, -0.749211907500f, -0.668598664000f, +0.743623444000f,
    +0.702560627800f, +0.711623892400f, -0.541938976300f, -0.840417840100f, -0.338861645600f, +0.940836215900f, +0.833153031500f, +0.553042517400f, -0.298972066200f, -0.954261863200f, +0.263852299300f, +0.964563094900f, +0.124108739000f, -0.992268623400f, -0.728264930800f, -0.685295695700f,
//  +0.702560627800f, +0.711623892400f, -0.541938976300f, -0.840417840100f, -0.338861645600f, +0.940836215900f, +0.833153031500f, +0.553042517400f, -0.298972066200f, -0.954261863200f, +0.263852299300f, +0.964563094900f, +0.124108739000f, -0.992268623400f, -0.728264930800f, -0.685295695700f,
    +0.696250014900f, +0.717799356900f, -0.918353536800f, +0.395761015600f, -0.632610227400f, -0.774470335200f, -0.933189185900f, -0.359385508000f, -0.115377935700f, -0.993321665900f, +0.951497478800f, -0.307656542100f, -0.089879774450f, -0.995952622400f, +0.667849691600f, +0.744296170500f,
//  +0.696250014900f, +0.717799356900f, -0.918353536800f, +0.395761015600f, -0.632610227400f, -0.774470335200f, -0.933189185900f, -0.359385508000f, -0.115377935700f, -0.993321665900f, +0.951497478800f, -0.307656542100f, -0.089879774450f, -0.995952622400f, +0.667849691600f, +0.744296170500f,
    +0.795240039300f, -0.606294713800f, -0.646200740200f, -0.763167480500f, -0.273359875300f, +0.961911835100f, +0.966959022600f, -0.254931851000f, -0.979289459500f, +0.202465193400f, -0.536950299500f, -0.843613878400f, -0.270036471000f, -0.962850094400f, -0.640027713100f, +0.768351824700f,
//  +0.795240039300f, -0.606294713800f, -0.646200740200f, -0.763167480500f, -0.273359875300f, +0.961911835100f, +0.966959022600f, -0.254931851000f, -0.979289459500f, +0.202465193400f, -0.536950299500f, -0.843613878400f, -0.270036471000f, -0.962850094400f, -0.640027713100f, +0.768351824700f,
    -0.785453749300f, -0.618920356600f, +0.060059053830f, -0.998194825700f, -0.024557703780f, +0.999698414100f, -0.659836230000f, +0.751409442000f, -0.625389446600f, -0.780312783500f, -0.621040885100f, -0.783778169500f, +0.834888849100f, +0.550418576800f, -0.159227524500f, +0.987241913300f,
//  -0.785453749300f, -0.618920356600f, +0.060059053830f, -0.998194825700f, -0.024557703780f, +0.999698414100f, -0.659836230000f, +0.751409442000f, -0.625389446600f, -0.780312783500f, -0.621040885100f, -0.783778169500f, +0.834888849100f, +0.550418576800f, -0.159227524500f, +0.987241913300f,
    +0.836762248800f, +0.547566378600f, -0.867575391600f, -0.497305680600f, -0.202266262800f, -0.979330566700f, +0.939918993700f, +0.341397547200f, +0.987740480700f, -0.156104909300f, -0.903445565600f, +0.428702822400f, +0.126980421800f, -0.991905223500f, -0.381960085400f, +0.924178821000f,
//  +0.836762248800f, +0.547566378600f, -0.867575391600f, -0.497305680600f, -0.202266262800f, -0.979330566700f, +0.939918993700f, +0.341397547200f, +0.987740480700f, -0.156104909300f, -0.903445565600f, +0.428702822400f, +0.126980421800f, -0.991905223500f, -0.381960085400f, +0.924178821000f,
    +0.975462589400f, +0.220165248600f, -0.320401585600f, -0.947281808100f, -0.987476088400f, +0.157768738700f, +0.025353484740f, -0.999678548700f, +0.483513079400f, -0.875337136200f, -0.285079992500f, -0.958503728700f, -0.068055160060f, -0.997681560000f, -0.788524404500f, -0.615003466300f,
//  +0.975462589400f, +0.220165248600f, -0.320401585600f, -0.947281808100f, -0.987476088400f, +0.157768738700f, +0.025353484740f, -0.999678548700f, +0.483513079400f, -0.875337136200f, -0.285079992500f, -0.958503728700f, -0.068055160060f, -0.997681560000f, -0.788524404500f, -0.615003466300f,
    +0.318539212700f, -0.947909684500f, +0.888004308900f, +0.459835130600f, +0.647692148800f, -0.761902146200f, +0.982024129900f, +0.188755419400f, +0.935727512800f, -0.352723718700f, -0.889489541400f, +0.456955529300f, +0.792279130200f, +0.610158815300f, +0.748381826100f, +0.663268152600f,
//  +0.318539212700f, -0.947909684500f, +0.888004308900f, +0.459835130600f, +0.647692148800f, -0.761902146200f, +0.982024129900f, +0.188755419400f, +0.935727512800f, -0.352723718700f, -0.889489541400f, +0.456955529300f, +0.792279130200f, +0.610158815300f, +0.748381826100f, +0.663268152600f,
    -0.728892975500f, -0.684627658100f, +0.872903278300f, -0.487893294400f, +0.828834578400f, +0.559493736900f, +0.080745670770f, +0.996734737400f, +0.979914821600f, -0.199416504800f, -0.580730673000f, -0.814095747100f, -0.470004979100f, -0.882663763600f, +0.240949297900f, +0.970537704500f,
//  -0.728892975500f, -0.684627658100f, +0.872903278300f, -0.487893294400f, +0.828834578400f, +0.559493736900f, +0.080745670770f, +0.996734737400f, +0.979914821600f, -0.199416504800f, -0.580730673000f, -0.814095747100f, -0.470004979100f, -0.882663763600f, +0.240949297900f, +0.970537704500f,
    +0.943781675700f, -0.330569430800f, -0.892799863800f, -0.450453552800f, -0.806962230400f, +0.590603046700f, +0.062589731660f, +0.998039340700f, -0.931259746900f, +0.364355984900f, +0.577744978500f, +0.816217336200f, -0.336009585500f, -0.941858566000f, +0.697932075000f, -0.716163960700f,
//  +0.943781675700f, -0.330569430800f, -0.892799863800f, -0.450453552800f, -0.806962230400f, +0.590603046700f, +0.062589731660f, +0.998039340700f, -0.931259746900f, +0.364355984900f, +0.577744978500f, +0.816217336200f, -0.336009585500f, -0.941858566000f, +0.697932075000f, -0.716163960700f,
    -0.002008157227f, -0.999997983700f, -0.182729431200f, -0.983163239200f, -0.652391172200f, +0.757882417300f, -0.430262691100f, -0.902703725800f, -0.998512628900f, -0.054520912510f, -0.010281021720f, -0.999947148900f, -0.494607112900f, +0.869116680200f, -0.299935019400f, +0.953959634400f,
//  -0.002008157227f, -0.999997983700f, -0.182729431200f, -0.983163239200f, -0.652391172200f, +0.757882417300f, -0.430262691100f, -0.902703725800f, -0.998512628900f, -0.054520912510f, -0.010281021720f, -0.999947148900f, -0.494607112900f, +0.869116680200f, -0.299935019400f, +0.953959634400f,
    +0.816547196100f, +0.577278681900f, +0.269746047500f, +0.962931498000f, -0.730628739100f, -0.682774959700f, -0.759095206400f, -0.650979621600f, -0.907053853000f, +0.421014617100f, -0.510486106400f, -0.859886001300f, +0.861335059700f, +0.508037316500f, +0.500788159500f, -0.865569881200f,
//  +0.816547196100f, +0.577278681900f, +0.269746047500f, +0.962931498000f, -0.730628739100f, -0.682774959700f, -0.759095206400f, -0.650979621600f, -0.907053853000f, +0.421014617100f, -0.510486106400f, -0.859886001300f, +0.861335059700f, +0.508037316500f, +0.500788159500f, -0.865569881200f,
    -0.654158152000f, +0.756357793800f, -0.838275531100f, -0.545246856000f, +0.694007083400f, +0.719968171700f, +0.069509360310f, +0.997581299400f, +0.170294218500f, -0.985393261200f, +0.269597327400f, +0.962973146600f, +0.551961219200f, -0.833869781500f, +0.225657487000f, -0.974206702200f,
//  -0.654158152000f, +0.756357793800f, -0.838275531100f, -0.545246856000f, +0.694007083400f, +0.719968171700f, +0.069509360310f, +0.997581299400f, +0.170294218500f, -0.985393261200f, +0.269597327400f, +0.962973146600f, +0.551961219200f, -0.833869781500f, +0.225657487000f, -0.974206702200f,
    +0.421526285500f, -0.906816183500f, +0.488187330500f, -0.872738867200f, -0.368385499600f, -0.929673127300f, -0.982539057800f, +0.186056442700f, +0.812564710000f, +0.582870990900f, +0.319646093300f, -0.947537004600f, +0.957091385900f, +0.289786264300f, -0.687665549700f, -0.726027610900f,
//  +0.421526285500f, -0.906816183500f, +0.488187330500f, -0.872738867200f, -0.368385499600f, -0.929673127300f, -0.982539057800f, +0.186056442700f, +0.812564710000f, +0.582870990900f, +0.319646093300f, -0.947537004600f, +0.957091385900f, +0.289786264300f, -0.687665549700f, -0.726027610900f,
    -0.998877092200f, -0.047376731000f, -0.125017902700f, +0.992154486000f, -0.828013361700f, +0.560708367000f, +0.932486376900f, -0.361205145100f, +0.639465318300f, +0.768819944200f, -0.016238470640f, -0.999868147300f, -0.995501466600f, -0.094746134580f, -0.814533150000f, +0.580117012000f,
//  -0.998877092200f, -0.047376731000f, -0.125017902700f, +0.992154486000f, -0.828013361700f, +0.560708367000f, +0.932486376900f, -0.361205145100f, +0.639465318300f, +0.768819944200f, -0.016238470640f, -0.999868147300f, -0.995501466600f, -0.094746134580f, -0.814533150000f, +0.580117012000f,
    +0.403732797800f, -0.914876946900f, +0.994426337100f, +0.105433676600f, -0.162471165400f, +0.986713291900f, -0.994948781400f, -0.100383875000f, -0.699530256400f, +0.714602980900f, +0.526341492200f, -0.850273270000f, -0.539522147900f, +0.841971408000f, +0.657937031800f, +0.753072946200f,
//  +0.403732797800f, -0.914876946900f, +0.994426337100f, +0.105433676600f, -0.162471165400f, +0.986713291900f, -0.994948781400f, -0.100383875000f, -0.699530256400f, +0.714602980900f, +0.526341492200f, -0.850273270000f, -0.539522147900f, +0.841971408000f, +0.657937031800f, +0.753072946200f,
    +0.014267588470f, -0.999898212800f, -0.673438399100f, +0.739243344700f, +0.639412098000f, -0.768864207100f, +0.921157142100f, +0.389190852300f, -0.146637214000f, -0.989190339400f, -0.782318098000f, +0.622879116300f, -0.503961083900f, -0.863726360500f, -0.774312019100f, -0.632803995700f,
//  +0.014267588470f, -0.999898212800f, -0.673438399100f, +0.739243344700f, +0.639412098000f, -0.768864207100f, +0.921157142100f, +0.389190852300f, -0.146637214000f, -0.989190339400f, -0.782318098000f, +0.622879116300f, -0.503961083900f, -0.863726360500f, -0.774312019100f, -0.632803995700f,
};

    template <typename T>
//  template <typename T>
    const T FastNoiseLite::Lookup<T>::Gradients3D[] =
//  const T FastNoiseLite::Lookup<T>::Gradients3D[] =
{
    0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
//  0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
    1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
//  1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
    1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
//  1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
    0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
//  0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
    1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
//  1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
    1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
//  1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
    0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
//  0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
    1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
//  1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
    1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
//  1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
    0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
//  0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
    1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
//  1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
    1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
//  1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
    0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
//  0, 1, 1, 0,  0,-1, 1, 0,  0, 1,-1, 0,  0,-1,-1, 0,
    1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
//  1, 0, 1, 0, -1, 0, 1, 0,  1, 0,-1, 0, -1, 0,-1, 0,
    1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
//  1, 1, 0, 0, -1, 1, 0, 0,  1,-1, 0, 0, -1,-1, 0, 0,
    1, 1, 0, 0,  0,-1, 1, 0, -1, 1, 0, 0,  0,-1,-1, 0,
//  1, 1, 0, 0,  0,-1, 1, 0, -1, 1, 0, 0,  0,-1,-1, 0,
};

    template <typename T>
//  template <typename T>
    const T FastNoiseLite::Lookup<T>::RandVecs3D[] =
//  const T FastNoiseLite::Lookup<T>::RandVecs3D[] =
{
    -0.7292736885000f, -0.6618439697000f, +0.1735581948000f, 0, +0.7902920810000f, -0.5480887466000f, -0.2739291014000f, 0, +0.7217578935000f, +0.6226212466000f, -0.3023380997000f, 0, +0.5656831370000f, -0.8208298145000f, -0.0790000257000f, 0, +0.7600490340000f, -0.5555979497000f, -0.3370999617000f, 0, +0.3713945616000f, +0.5011264475000f, +0.7816254623000f, 0, -0.1277062463000f, -0.4254438999000f, -0.8959289049000f, 0, -0.2881560924000f, -0.5815838982000f, +0.7607405838000f, 0,
//  -0.7292736885000f, -0.6618439697000f, +0.1735581948000f, 0, +0.7902920810000f, -0.5480887466000f, -0.2739291014000f, 0, +0.7217578935000f, +0.6226212466000f, -0.3023380997000f, 0, +0.5656831370000f, -0.8208298145000f, -0.0790000257000f, 0, +0.7600490340000f, -0.5555979497000f, -0.3370999617000f, 0, +0.3713945616000f, +0.5011264475000f, +0.7816254623000f, 0, -0.1277062463000f, -0.4254438999000f, -0.8959289049000f, 0, -0.2881560924000f, -0.5815838982000f, +0.7607405838000f, 0,
    +0.5849561111000f, -0.6628202390000f, -0.4674352136000f, 0, +0.3307171178000f, +0.0391653737000f, +0.9429168900000f, 0, +0.8712121778000f, -0.4113374369000f, -0.2679381538000f, 0, +0.5809810150000f, +0.7021915846000f, +0.4115677815000f, 0, +0.5037568730000f, +0.6330056931000f, -0.5878203852000f, 0, +0.4493712205000f, +0.6013901950000f, +0.6606022552000f, 0, -0.6878403724000f, +0.0901889080700f, -0.7202371714000f, 0, -0.5958956522000f, -0.6469350577000f, +0.4757976490000f, 0,
//  +0.5849561111000f, -0.6628202390000f, -0.4674352136000f, 0, +0.3307171178000f, +0.0391653737000f, +0.9429168900000f, 0, +0.8712121778000f, -0.4113374369000f, -0.2679381538000f, 0, +0.5809810150000f, +0.7021915846000f, +0.4115677815000f, 0, +0.5037568730000f, +0.6330056931000f, -0.5878203852000f, 0, +0.4493712205000f, +0.6013901950000f, +0.6606022552000f, 0, -0.6878403724000f, +0.0901889080700f, -0.7202371714000f, 0, -0.5958956522000f, -0.6469350577000f, +0.4757976490000f, 0,
    -0.5127052122000f, +0.1946921978000f, -0.8361987284000f, 0, -0.9911507142000f, -0.0541027646600f, -0.1212153153000f, 0, -0.2149721042000f, +0.9720882117000f, -0.0939760774900f, 0, -0.7518650936000f, -0.5428057603000f, +0.3742469607000f, 0, +0.5237068895000f, +0.8516377189000f, -0.0210781783400f, 0, +0.6333504779000f, +0.1926167129000f, -0.7495104896000f, 0, -0.0678824160600f, +0.3998305789000f, +0.9140719259000f, 0, -0.5538628599000f, -0.4729896695000f, -0.6852128902000f, 0,
//  -0.5127052122000f, +0.1946921978000f, -0.8361987284000f, 0, -0.9911507142000f, -0.0541027646600f, -0.1212153153000f, 0, -0.2149721042000f, +0.9720882117000f, -0.0939760774900f, 0, -0.7518650936000f, -0.5428057603000f, +0.3742469607000f, 0, +0.5237068895000f, +0.8516377189000f, -0.0210781783400f, 0, +0.6333504779000f, +0.1926167129000f, -0.7495104896000f, 0, -0.0678824160600f, +0.3998305789000f, +0.9140719259000f, 0, -0.5538628599000f, -0.4729896695000f, -0.6852128902000f, 0,
    -0.7261455366000f, -0.5911990757000f, +0.3509933228000f, 0, -0.9229274737000f, -0.1782808786000f, +0.3412049336000f, 0, -0.6968815002000f, +0.6511274338000f, +0.3006480328000f, 0, +0.9608044783000f, -0.2098363234000f, -0.1811724921000f, 0, +0.0681714606200f, -0.9743405129000f, +0.2145069156000f, 0, -0.3577285196000f, -0.6697087264000f, -0.6507845481000f, 0, -0.1868621131000f, +0.7648617052000f, -0.6164974636000f, 0, -0.6541697588000f, +0.3967914832000f, +0.6439087246000f, 0,
//  -0.7261455366000f, -0.5911990757000f, +0.3509933228000f, 0, -0.9229274737000f, -0.1782808786000f, +0.3412049336000f, 0, -0.6968815002000f, +0.6511274338000f, +0.3006480328000f, 0, +0.9608044783000f, -0.2098363234000f, -0.1811724921000f, 0, +0.0681714606200f, -0.9743405129000f, +0.2145069156000f, 0, -0.3577285196000f, -0.6697087264000f, -0.6507845481000f, 0, -0.1868621131000f, +0.7648617052000f, -0.6164974636000f, 0, -0.6541697588000f, +0.3967914832000f, +0.6439087246000f, 0,
    +0.6993340405000f, -0.6164538506000f, +0.3618239211000f, 0, -0.1546665739000f, +0.6291283928000f, +0.7617583057000f, 0, -0.6841612949000f, -0.2580482182000f, -0.6821542638000f, 0, +0.5383980957000f, +0.4258654885000f, +0.7271630328000f, 0, -0.5026987823000f, -0.7939832935000f, -0.3418836993000f, 0, +0.3202971715000f, +0.2834415347000f, +0.9039195862000f, 0, +0.8683227101000f, -0.0003762656404f, -0.4959995258000f, 0, +0.7911200310000f, -0.0851104574500f, +0.6057105799000f, 0,
//  +0.6993340405000f, -0.6164538506000f, +0.3618239211000f, 0, -0.1546665739000f, +0.6291283928000f, +0.7617583057000f, 0, -0.6841612949000f, -0.2580482182000f, -0.6821542638000f, 0, +0.5383980957000f, +0.4258654885000f, +0.7271630328000f, 0, -0.5026987823000f, -0.7939832935000f, -0.3418836993000f, 0, +0.3202971715000f, +0.2834415347000f, +0.9039195862000f, 0, +0.8683227101000f, -0.0003762656404f, -0.4959995258000f, 0, +0.7911200310000f, -0.0851104574500f, +0.6057105799000f, 0,
    -0.0401101605200f, -0.4397248749000f, +0.8972364289000f, 0, +0.9145119872000f, +0.3579346169000f, -0.1885487608000f, 0, -0.9612039066000f, -0.2756484276000f, +0.0102466692900f, 0, +0.6510361721000f, -0.2877799159000f, -0.7023778346000f, 0, -0.2041786351000f, +0.7365237271000f, +0.6448595850000f, 0, -0.7718263711000f, +0.3790626912000f, +0.5104855816000f, 0, -0.3060082741000f, -0.7692987727000f, +0.5608371729000f, 0, +0.4540073410000f, -0.5024843065000f, +0.7357899537000f, 0,
//  -0.0401101605200f, -0.4397248749000f, +0.8972364289000f, 0, +0.9145119872000f, +0.3579346169000f, -0.1885487608000f, 0, -0.9612039066000f, -0.2756484276000f, +0.0102466692900f, 0, +0.6510361721000f, -0.2877799159000f, -0.7023778346000f, 0, -0.2041786351000f, +0.7365237271000f, +0.6448595850000f, 0, -0.7718263711000f, +0.3790626912000f, +0.5104855816000f, 0, -0.3060082741000f, -0.7692987727000f, +0.5608371729000f, 0, +0.4540073410000f, -0.5024843065000f, +0.7357899537000f, 0,
    +0.4816795475000f, +0.6021208291000f, -0.6367380315000f, 0, +0.6961980369000f, -0.3222197429000f, +0.6414691970000f, 0, -0.6532160499000f, -0.6781148932000f, +0.3368515753000f, 0, +0.5089301236000f, -0.6154662304000f, -0.6018234363000f, 0, -0.1635919754000f, -0.9133604627000f, -0.3728408920000f, 0, +0.5240801900000f, -0.8437664109000f, +0.1157505864000f, 0, +0.5902587356000f, +0.4983817807000f, -0.6349883666000f, 0, +0.5863227872000f, +0.4947647450000f, +0.6414307729000f, 0,
//  +0.4816795475000f, +0.6021208291000f, -0.6367380315000f, 0, +0.6961980369000f, -0.3222197429000f, +0.6414691970000f, 0, -0.6532160499000f, -0.6781148932000f, +0.3368515753000f, 0, +0.5089301236000f, -0.6154662304000f, -0.6018234363000f, 0, -0.1635919754000f, -0.9133604627000f, -0.3728408920000f, 0, +0.5240801900000f, -0.8437664109000f, +0.1157505864000f, 0, +0.5902587356000f, +0.4983817807000f, -0.6349883666000f, 0, +0.5863227872000f, +0.4947647450000f, +0.6414307729000f, 0,
    +0.6779335087000f, +0.2341345225000f, +0.6968408593000f, 0, +0.7177054546000f, -0.6858979348000f, +0.1201786310000f, 0, -0.5328819713000f, -0.5205125012000f, +0.6671608058000f, 0, -0.8654874251000f, -0.0700727088000f, -0.4960053754000f, 0, -0.2861810166000f, +0.7952089234000f, +0.5345495242000f, 0, -0.0484952963400f, +0.9810836427000f, -0.1874115585000f, 0, -0.6358521667000f, +0.6058348682000f, +0.4781800233000f, 0, +0.6254794696000f, -0.2861619734000f, +0.7258696564000f, 0,
//  +0.6779335087000f, +0.2341345225000f, +0.6968408593000f, 0, +0.7177054546000f, -0.6858979348000f, +0.1201786310000f, 0, -0.5328819713000f, -0.5205125012000f, +0.6671608058000f, 0, -0.8654874251000f, -0.0700727088000f, -0.4960053754000f, 0, -0.2861810166000f, +0.7952089234000f, +0.5345495242000f, 0, -0.0484952963400f, +0.9810836427000f, -0.1874115585000f, 0, -0.6358521667000f, +0.6058348682000f, +0.4781800233000f, 0, +0.6254794696000f, -0.2861619734000f, +0.7258696564000f, 0,
    -0.2585259868000f, +0.5061949264000f, -0.8227581726000f, 0, +0.0213630678100f, +0.5064016808000f, -0.8620330371000f, 0, +0.2001117730000f, +0.8599263484000f, +0.4695550591000f, 0, +0.4743561372000f, +0.6014985084000f, -0.6427953014000f, 0, +0.6622993731000f, -0.5202474575000f, -0.5391679918000f, 0, +0.0808497281800f, -0.6532720452000f, +0.7527940996000f, 0, -0.6893687501000f, +0.0592860349000f, +0.7219805347000f, 0, -0.1121887082000f, -0.9673185067000f, +0.2273952515000f, 0,
//  -0.2585259868000f, +0.5061949264000f, -0.8227581726000f, 0, +0.0213630678100f, +0.5064016808000f, -0.8620330371000f, 0, +0.2001117730000f, +0.8599263484000f, +0.4695550591000f, 0, +0.4743561372000f, +0.6014985084000f, -0.6427953014000f, 0, +0.6622993731000f, -0.5202474575000f, -0.5391679918000f, 0, +0.0808497281800f, -0.6532720452000f, +0.7527940996000f, 0, -0.6893687501000f, +0.0592860349000f, +0.7219805347000f, 0, -0.1121887082000f, -0.9673185067000f, +0.2273952515000f, 0,
    +0.7344116094000f, +0.5979668656000f, -0.3210532909000f, 0, +0.5789393465000f, -0.2488849713000f, +0.7764570201000f, 0, +0.6988182827000f, +0.3557169806000f, -0.6205791146000f, 0, -0.8636845529000f, -0.2748771249000f, -0.4224826141000f, 0, -0.4247027957000f, -0.4640880967000f, +0.7773350460000f, 0, +0.5257722489000f, -0.8427017621000f, +0.1158329937000f, 0, +0.9343830603000f, +0.3163024720000f, -0.1639543925000f, 0, -0.1016836419000f, -0.8057303073000f, -0.5834887393000f, 0,
//  +0.7344116094000f, +0.5979668656000f, -0.3210532909000f, 0, +0.5789393465000f, -0.2488849713000f, +0.7764570201000f, 0, +0.6988182827000f, +0.3557169806000f, -0.6205791146000f, 0, -0.8636845529000f, -0.2748771249000f, -0.4224826141000f, 0, -0.4247027957000f, -0.4640880967000f, +0.7773350460000f, 0, +0.5257722489000f, -0.8427017621000f, +0.1158329937000f, 0, +0.9343830603000f, +0.3163024720000f, -0.1639543925000f, 0, -0.1016836419000f, -0.8057303073000f, -0.5834887393000f, 0,
    -0.6529238969000f, +0.5060212600000f, -0.5635892736000f, 0, -0.2465286165000f, -0.9668205684000f, -0.0669449749400f, 0, -0.9776897119000f, -0.2099250524000f, -0.0073688253440f, 0, +0.7736893337000f, +0.5734244712000f, +0.2694238123000f, 0, -0.6095087895000f, +0.4995678998000f, +0.6155736747000f, 0, +0.5794535482000f, +0.7434546771000f, +0.3339292269000f, 0, -0.8226211154000f, +0.0814258185500f, +0.5627293636000f, 0, -0.5103854830000f, +0.4703667658000f, +0.7199039967000f, 0,
//  -0.6529238969000f, +0.5060212600000f, -0.5635892736000f, 0, -0.2465286165000f, -0.9668205684000f, -0.0669449749400f, 0, -0.9776897119000f, -0.2099250524000f, -0.0073688253440f, 0, +0.7736893337000f, +0.5734244712000f, +0.2694238123000f, 0, -0.6095087895000f, +0.4995678998000f, +0.6155736747000f, 0, +0.5794535482000f, +0.7434546771000f, +0.3339292269000f, 0, -0.8226211154000f, +0.0814258185500f, +0.5627293636000f, 0, -0.5103854830000f, +0.4703667658000f, +0.7199039967000f, 0,
    -0.5764971849000f, -0.0723165627400f, -0.8138926898000f, 0, +0.7250628871000f, +0.3949971505000f, -0.5641463116000f, 0, -0.1525424005000f, +0.4860840828000f, -0.8604958341000f, 0, -0.5550976208000f, -0.4957820792000f, +0.6678822960000f, 0, -0.1883614327000f, +0.9145869398000f, +0.3578417250000f, 0, +0.7625556724000f, -0.5414408243000f, -0.3540489801000f, 0, -0.5870231946000f, -0.3226498013000f, -0.7424963803000f, 0, +0.3051124198000f, +0.2262544068000f, -0.9250488391000f, 0,
//  -0.5764971849000f, -0.0723165627400f, -0.8138926898000f, 0, +0.7250628871000f, +0.3949971505000f, -0.5641463116000f, 0, -0.1525424005000f, +0.4860840828000f, -0.8604958341000f, 0, -0.5550976208000f, -0.4957820792000f, +0.6678822960000f, 0, -0.1883614327000f, +0.9145869398000f, +0.3578417250000f, 0, +0.7625556724000f, -0.5414408243000f, -0.3540489801000f, 0, -0.5870231946000f, -0.3226498013000f, -0.7424963803000f, 0, +0.3051124198000f, +0.2262544068000f, -0.9250488391000f, 0,
    +0.6379576059000f, +0.5772424240000f, -0.5097070502000f, 0, -0.5966775796000f, +0.1454852398000f, -0.7891830656000f, 0, -0.6583305730000f, +0.6555487542000f, -0.3699414651000f, 0, +0.7434892426000f, +0.2351084581000f, +0.6260573129000f, 0, +0.5562114096000f, +0.8264360377000f, -0.0873632843000f, 0, -0.3028940016000f, -0.8251527185000f, +0.4768419182000f, 0, +0.1129343818000f, -0.9858884390000f, -0.1235710781000f, 0, +0.5937652891000f, -0.5896813806000f, +0.5474656618000f, 0,
//  +0.6379576059000f, +0.5772424240000f, -0.5097070502000f, 0, -0.5966775796000f, +0.1454852398000f, -0.7891830656000f, 0, -0.6583305730000f, +0.6555487542000f, -0.3699414651000f, 0, +0.7434892426000f, +0.2351084581000f, +0.6260573129000f, 0, +0.5562114096000f, +0.8264360377000f, -0.0873632843000f, 0, -0.3028940016000f, -0.8251527185000f, +0.4768419182000f, 0, +0.1129343818000f, -0.9858884390000f, -0.1235710781000f, 0, +0.5937652891000f, -0.5896813806000f, +0.5474656618000f, 0,
    +0.6757964092000f, -0.5835758614000f, -0.4502648413000f, 0, +0.7242302609000f, -0.1152719764000f, +0.6798550586000f, 0, -0.9511914166000f, +0.0753623979000f, -0.2992580792000f, 0, +0.2539470961000f, -0.1886339355000f, +0.9486454084000f, 0, +0.5714336210000f, -0.1679450851000f, -0.8032795685000f, 0, -0.0677823497900f, +0.3978269256000f, +0.9149531629000f, 0, +0.6074972649000f, +0.7330600240000f, -0.3058922593000f, 0, -0.5435478392000f, +0.1675822484000f, +0.8224791405000f, 0,
//  +0.6757964092000f, -0.5835758614000f, -0.4502648413000f, 0, +0.7242302609000f, -0.1152719764000f, +0.6798550586000f, 0, -0.9511914166000f, +0.0753623979000f, -0.2992580792000f, 0, +0.2539470961000f, -0.1886339355000f, +0.9486454084000f, 0, +0.5714336210000f, -0.1679450851000f, -0.8032795685000f, 0, -0.0677823497900f, +0.3978269256000f, +0.9149531629000f, 0, +0.6074972649000f, +0.7330600240000f, -0.3058922593000f, 0, -0.5435478392000f, +0.1675822484000f, +0.8224791405000f, 0,
    -0.5876678086000f, -0.3380045064000f, -0.7351186982000f, 0, -0.7967562402000f, +0.0409782270600f, -0.6029098428000f, 0, -0.1996350917000f, +0.8706294745000f, +0.4496111079000f, 0, -0.0278766033600f, -0.9106232682000f, -0.4122962022000f, 0, -0.7797625996000f, -0.6257634692000f, +0.0197577558100f, 0, -0.5211232846000f, +0.7401644346000f, -0.4249554471000f, 0, +0.8575424857000f, +0.4053272873000f, -0.3167501783000f, 0, +0.1045223322000f, +0.8390195772000f, -0.5339674439000f, 0,
//  -0.5876678086000f, -0.3380045064000f, -0.7351186982000f, 0, -0.7967562402000f, +0.0409782270600f, -0.6029098428000f, 0, -0.1996350917000f, +0.8706294745000f, +0.4496111079000f, 0, -0.0278766033600f, -0.9106232682000f, -0.4122962022000f, 0, -0.7797625996000f, -0.6257634692000f, +0.0197577558100f, 0, -0.5211232846000f, +0.7401644346000f, -0.4249554471000f, 0, +0.8575424857000f, +0.4053272873000f, -0.3167501783000f, 0, +0.1045223322000f, +0.8390195772000f, -0.5339674439000f, 0,
    +0.3501822831000f, +0.9242524096000f, -0.1520850155000f, 0, +0.1987849858000f, +0.0764761326600f, +0.9770547224000f, 0, +0.7845996363000f, +0.6066256811000f, -0.1280964233000f, 0, +0.0900673743600f, -0.9750989929000f, -0.2026569073000f, 0, -0.8274343547000f, -0.5422995590000f, +0.1458203587000f, 0, -0.3485797732000f, -0.4158022770000f, +0.8400003620000f, 0, -0.2471778936000f, -0.7304819962000f, -0.6366310879000f, 0, -0.3700154943000f, +0.8577948156000f, +0.3567584454000f, 0,
//  +0.3501822831000f, +0.9242524096000f, -0.1520850155000f, 0, +0.1987849858000f, +0.0764761326600f, +0.9770547224000f, 0, +0.7845996363000f, +0.6066256811000f, -0.1280964233000f, 0, +0.0900673743600f, -0.9750989929000f, -0.2026569073000f, 0, -0.8274343547000f, -0.5422995590000f, +0.1458203587000f, 0, -0.3485797732000f, -0.4158022770000f, +0.8400003620000f, 0, -0.2471778936000f, -0.7304819962000f, -0.6366310879000f, 0, -0.3700154943000f, +0.8577948156000f, +0.3567584454000f, 0,
    +0.5913394901000f, -0.5483119670000f, -0.5913303597000f, 0, +0.1204873514000f, -0.7626472379000f, -0.6354935001000f, 0, +0.6169592650000f, +0.0307964792800f, +0.7863922953000f, 0, +0.1258156836000f, -0.6640829889000f, -0.7369967419000f, 0, -0.6477565124000f, -0.1740147258000f, -0.7417077429000f, 0, +0.6217889313000f, -0.7804430448000f, -0.0654765507600f, 0, +0.6589943422000f, -0.6096987708000f, +0.4404473475000f, 0, -0.2689837504000f, -0.6732403169000f, -0.6887635427000f, 0,
//  +0.5913394901000f, -0.5483119670000f, -0.5913303597000f, 0, +0.1204873514000f, -0.7626472379000f, -0.6354935001000f, 0, +0.6169592650000f, +0.0307964792800f, +0.7863922953000f, 0, +0.1258156836000f, -0.6640829889000f, -0.7369967419000f, 0, -0.6477565124000f, -0.1740147258000f, -0.7417077429000f, 0, +0.6217889313000f, -0.7804430448000f, -0.0654765507600f, 0, +0.6589943422000f, -0.6096987708000f, +0.4404473475000f, 0, -0.2689837504000f, -0.6732403169000f, -0.6887635427000f, 0,
    -0.3849775103000f, +0.5676542638000f, +0.7277093879000f, 0, +0.5754444408000f, +0.8110471154000f, -0.1051963504000f, 0, +0.9141593684000f, +0.3832947817000f, +0.1319005670000f, 0, -0.1079253190000f, +0.9245493968000f, +0.3654593525000f, 0, +0.3779770890000f, +0.3043148782000f, +0.8743716458000f, 0, -0.2142885215000f, -0.8259286236000f, +0.5214617324000f, 0, +0.5802544474000f, +0.4148098596000f, -0.7008834116000f, 0, -0.1982660881000f, +0.8567161266000f, -0.4761596756000f, 0,
//  -0.3849775103000f, +0.5676542638000f, +0.7277093879000f, 0, +0.5754444408000f, +0.8110471154000f, -0.1051963504000f, 0, +0.9141593684000f, +0.3832947817000f, +0.1319005670000f, 0, -0.1079253190000f, +0.9245493968000f, +0.3654593525000f, 0, +0.3779770890000f, +0.3043148782000f, +0.8743716458000f, 0, -0.2142885215000f, -0.8259286236000f, +0.5214617324000f, 0, +0.5802544474000f, +0.4148098596000f, -0.7008834116000f, 0, -0.1982660881000f, +0.8567161266000f, -0.4761596756000f, 0,
    -0.0338155370400f, +0.3773180787000f, -0.9254661404000f, 0, -0.6867922841000f, -0.6656597827000f, +0.2919133642000f, 0, +0.7731742607000f, -0.2875793547000f, -0.5652430251000f, 0, -0.0965594192800f, +0.9193708367000f, -0.3813575004000f, 0, +0.2715702457000f, -0.9577909544000f, -0.0942660558100f, 0, +0.2451015704000f, -0.6917998565000f, -0.6792188003000f, 0, +0.9777007820000f, -0.1753855374000f, +0.1155036542000f, 0, -0.5224739938000f, +0.8521606816000f, +0.0290361594500f, 0,
//  -0.0338155370400f, +0.3773180787000f, -0.9254661404000f, 0, -0.6867922841000f, -0.6656597827000f, +0.2919133642000f, 0, +0.7731742607000f, -0.2875793547000f, -0.5652430251000f, 0, -0.0965594192800f, +0.9193708367000f, -0.3813575004000f, 0, +0.2715702457000f, -0.9577909544000f, -0.0942660558100f, 0, +0.2451015704000f, -0.6917998565000f, -0.6792188003000f, 0, +0.9777007820000f, -0.1753855374000f, +0.1155036542000f, 0, -0.5224739938000f, +0.8521606816000f, +0.0290361594500f, 0,
    -0.7734880599000f, -0.5261292347000f, +0.3534179531000f, 0, -0.7134492443000f, -0.2695472430000f, +0.6467878011000f, 0, +0.1644037271000f, +0.5105846203000f, -0.8439637196000f, 0, +0.6494635788000f, +0.0558561129600f, +0.7583384168000f, 0, -0.4711970882000f, +0.5017280509000f, -0.7254255765000f, 0, -0.6335764307000f, -0.2381686273000f, -0.7361091029000f, 0, -0.9021533097000f, -0.2709478030000f, -0.3357181763000f, 0, -0.3793711033000f, +0.8722581170000f, +0.3086152025000f, 0,
//  -0.7734880599000f, -0.5261292347000f, +0.3534179531000f, 0, -0.7134492443000f, -0.2695472430000f, +0.6467878011000f, 0, +0.1644037271000f, +0.5105846203000f, -0.8439637196000f, 0, +0.6494635788000f, +0.0558561129600f, +0.7583384168000f, 0, -0.4711970882000f, +0.5017280509000f, -0.7254255765000f, 0, -0.6335764307000f, -0.2381686273000f, -0.7361091029000f, 0, -0.9021533097000f, -0.2709478030000f, -0.3357181763000f, 0, -0.3793711033000f, +0.8722581170000f, +0.3086152025000f, 0,
    -0.6855598966000f, -0.3250143309000f, +0.6514394162000f, 0, +0.2900942212000f, -0.7799057743000f, -0.5546100667000f, 0, -0.2098319339000f, +0.8503707300000f, +0.4825351604000f, 0, -0.4592603758000f, +0.6598504336000f, -0.5947077538000f, 0, +0.8715945488000f, +0.0961636540600f, -0.4807031248000f, 0, -0.6776666319000f, +0.7118504878000f, -0.1844907016000f, 0, +0.7044377633000f, +0.3124275970000f, +0.6373040360000f, 0, -0.7052318886000f, -0.2401093292000f, -0.6670798253000f, 0,
//  -0.6855598966000f, -0.3250143309000f, +0.6514394162000f, 0, +0.2900942212000f, -0.7799057743000f, -0.5546100667000f, 0, -0.2098319339000f, +0.8503707300000f, +0.4825351604000f, 0, -0.4592603758000f, +0.6598504336000f, -0.5947077538000f, 0, +0.8715945488000f, +0.0961636540600f, -0.4807031248000f, 0, -0.6776666319000f, +0.7118504878000f, -0.1844907016000f, 0, +0.7044377633000f, +0.3124275970000f, +0.6373040360000f, 0, -0.7052318886000f, -0.2401093292000f, -0.6670798253000f, 0,
    +0.0819210070000f, -0.7207336136000f, -0.6883545647000f, 0, -0.6993680906000f, -0.5875763221000f, -0.4069869034000f, 0, -0.1281454481000f, +0.6419895885000f, +0.7559286424000f, 0, -0.6337388239000f, -0.6785471501000f, -0.3714146849000f, 0, +0.5565051903000f, -0.2168887573000f, -0.8020356851000f, 0, -0.5791554484000f, +0.7244372011000f, -0.3738578718000f, 0, +0.1175779076000f, -0.7096451073000f, +0.6946792478000f, 0, -0.6134619607000f, +0.1323631078000f, +0.7785527795000f, 0,
//  +0.0819210070000f, -0.7207336136000f, -0.6883545647000f, 0, -0.6993680906000f, -0.5875763221000f, -0.4069869034000f, 0, -0.1281454481000f, +0.6419895885000f, +0.7559286424000f, 0, -0.6337388239000f, -0.6785471501000f, -0.3714146849000f, 0, +0.5565051903000f, -0.2168887573000f, -0.8020356851000f, 0, -0.5791554484000f, +0.7244372011000f, -0.3738578718000f, 0, +0.1175779076000f, -0.7096451073000f, +0.6946792478000f, 0, -0.6134619607000f, +0.1323631078000f, +0.7785527795000f, 0,
    +0.6984635305000f, -0.0298051623700f, -0.7150247190000f, 0, +0.8318082963000f, -0.3930171956000f, +0.3919597455000f, 0, +0.1469576422000f, +0.0554165171700f, -0.9875892167000f, 0, +0.7088685750000f, -0.2690503865000f, +0.6520101478000f, 0, +0.2726053183000f, +0.6736976600000f, -0.6868899500000f, 0, -0.6591295371000f, +0.3035458599000f, -0.6880466294000f, 0, +0.4815131379000f, -0.7528270071000f, +0.4487723203000f, 0, +0.9430009463000f, +0.1675647412000f, -0.2875261255000f, 0,
//  +0.6984635305000f, -0.0298051623700f, -0.7150247190000f, 0, +0.8318082963000f, -0.3930171956000f, +0.3919597455000f, 0, +0.1469576422000f, +0.0554165171700f, -0.9875892167000f, 0, +0.7088685750000f, -0.2690503865000f, +0.6520101478000f, 0, +0.2726053183000f, +0.6736976600000f, -0.6868899500000f, 0, -0.6591295371000f, +0.3035458599000f, -0.6880466294000f, 0, +0.4815131379000f, -0.7528270071000f, +0.4487723203000f, 0, +0.9430009463000f, +0.1675647412000f, -0.2875261255000f, 0,
    +0.4348029570000f, +0.7695304522000f, -0.4677277752000f, 0, +0.3931996188000f, +0.5944736250000f, +0.7014236729000f, 0, +0.7254336655000f, -0.6039256540000f, +0.3301814672000f, 0, +0.7590235227000f, -0.6506083235000f, +0.0243331320700f, 0, -0.8552768592000f, -0.3430042733000f, +0.3883935666000f, 0, -0.6139746835000f, +0.6981725247000f, +0.3682257648000f, 0, -0.7465905486000f, -0.5752009504000f, +0.3342849376000f, 0, +0.5730065677000f, +0.8105555370000f, -0.1210916791000f, 0,
//  +0.4348029570000f, +0.7695304522000f, -0.4677277752000f, 0, +0.3931996188000f, +0.5944736250000f, +0.7014236729000f, 0, +0.7254336655000f, -0.6039256540000f, +0.3301814672000f, 0, +0.7590235227000f, -0.6506083235000f, +0.0243331320700f, 0, -0.8552768592000f, -0.3430042733000f, +0.3883935666000f, 0, -0.6139746835000f, +0.6981725247000f, +0.3682257648000f, 0, -0.7465905486000f, -0.5752009504000f, +0.3342849376000f, 0, +0.5730065677000f, +0.8105555370000f, -0.1210916791000f, 0,
    -0.9225877367000f, -0.3475211012000f, -0.1675140360000f, 0, -0.7105816789000f, -0.4719692027000f, -0.5218416899000f, 0, -0.0856460971700f, +0.3583001386000f, +0.9296697030000f, 0, -0.8279697606000f, -0.2043157126000f, +0.5222271202000f, 0, +0.4279440230000f, +0.2781659940000f, +0.8599346446000f, 0, +0.5399079671000f, -0.7857120652000f, -0.3019204161000f, 0, +0.5678404253000f, -0.5495413974000f, -0.6128307303000f, 0, -0.9896071041000f, +0.1365639107000f, -0.0450341842800f, 0,
//  -0.9225877367000f, -0.3475211012000f, -0.1675140360000f, 0, -0.7105816789000f, -0.4719692027000f, -0.5218416899000f, 0, -0.0856460971700f, +0.3583001386000f, +0.9296697030000f, 0, -0.8279697606000f, -0.2043157126000f, +0.5222271202000f, 0, +0.4279440230000f, +0.2781659940000f, +0.8599346446000f, 0, +0.5399079671000f, -0.7857120652000f, -0.3019204161000f, 0, +0.5678404253000f, -0.5495413974000f, -0.6128307303000f, 0, -0.9896071041000f, +0.1365639107000f, -0.0450341842800f, 0,
    -0.6154342638000f, -0.6440875597000f, +0.4543037336000f, 0, +0.1074204368000f, -0.7946340692000f, +0.5975094525000f, 0, -0.3595449969000f, -0.8885529948000f, +0.2849578400000f, 0, -0.2180405296000f, +0.1529888965000f, +0.9638738118000f, 0, -0.7277432317000f, -0.6164050508000f, -0.3007234646000f, 0, +0.7249729114000f, -0.0066971948400f, +0.6887448187000f, 0, -0.5553659455000f, -0.5336586252000f, +0.6377908264000f, 0, +0.5137558015000f, +0.7976208196000f, -0.3160000073000f, 0,
//  -0.6154342638000f, -0.6440875597000f, +0.4543037336000f, 0, +0.1074204368000f, -0.7946340692000f, +0.5975094525000f, 0, -0.3595449969000f, -0.8885529948000f, +0.2849578400000f, 0, -0.2180405296000f, +0.1529888965000f, +0.9638738118000f, 0, -0.7277432317000f, -0.6164050508000f, -0.3007234646000f, 0, +0.7249729114000f, -0.0066971948400f, +0.6887448187000f, 0, -0.5553659455000f, -0.5336586252000f, +0.6377908264000f, 0, +0.5137558015000f, +0.7976208196000f, -0.3160000073000f, 0,
    -0.3794024848000f, +0.9245608561000f, -0.0352275149400f, 0, +0.8229248658000f, +0.2745365933000f, -0.4974176556000f, 0, -0.5404114394000f, +0.6091141441000f, +0.5804613989000f, 0, +0.8036581901000f, -0.2703029469000f, +0.5301601931000f, 0, +0.6044318879000f, +0.6832968393000f, +0.4095943388000f, 0, +0.0638998881700f, +0.9658208605000f, -0.2512108074000f, 0, +0.1087113286000f, +0.7402471173000f, -0.6634877936000f, 0, -0.7134277120000f, -0.6926784018000f, +0.1059128479000f, 0,
//  -0.3794024848000f, +0.9245608561000f, -0.0352275149400f, 0, +0.8229248658000f, +0.2745365933000f, -0.4974176556000f, 0, -0.5404114394000f, +0.6091141441000f, +0.5804613989000f, 0, +0.8036581901000f, -0.2703029469000f, +0.5301601931000f, 0, +0.6044318879000f, +0.6832968393000f, +0.4095943388000f, 0, +0.0638998881700f, +0.9658208605000f, -0.2512108074000f, 0, +0.1087113286000f, +0.7402471173000f, -0.6634877936000f, 0, -0.7134277120000f, -0.6926784018000f, +0.1059128479000f, 0,
    +0.6458897819000f, -0.5724548511000f, -0.5050958653000f, 0, -0.6553931414000f, +0.7381471625000f, +0.1599956150000f, 0, +0.3910961323000f, +0.9188871375000f, -0.0518675599800f, 0, -0.4879022471000f, -0.5904376907000f, +0.6429111375000f, 0, +0.6014790094000f, +0.7707441366000f, -0.2101820095000f, 0, -0.5677173047000f, +0.7511360995000f, +0.3368851762000f, 0, +0.7858573506000f, +0.2266746650000f, +0.5753666838000f, 0, -0.4520345543000f, -0.6042226860000f, -0.6561857263000f, 0,
//  +0.6458897819000f, -0.5724548511000f, -0.5050958653000f, 0, -0.6553931414000f, +0.7381471625000f, +0.1599956150000f, 0, +0.3910961323000f, +0.9188871375000f, -0.0518675599800f, 0, -0.4879022471000f, -0.5904376907000f, +0.6429111375000f, 0, +0.6014790094000f, +0.7707441366000f, -0.2101820095000f, 0, -0.5677173047000f, +0.7511360995000f, +0.3368851762000f, 0, +0.7858573506000f, +0.2266746650000f, +0.5753666838000f, 0, -0.4520345543000f, -0.6042226860000f, -0.6561857263000f, 0,
    +0.0022721163450f, +0.4132844051000f, -0.9105991643000f, 0, -0.5815751419000f, -0.5162925989000f, +0.6286591339000f, 0, -0.0370370478500f, +0.8273785755000f, +0.5604221175000f, 0, -0.5119692504000f, +0.7953543429000f, -0.3244980058000f, 0, -0.2682417366000f, -0.9572290247000f, -0.1084387619000f, 0, -0.2322482736000f, -0.9679131102000f, -0.0959424332400f, 0, +0.3554328906000f, -0.8881505545000f, +0.2913006227000f, 0, +0.7346520519000f, -0.4371373164000f, +0.5188422971000f, 0,
//  +0.0022721163450f, +0.4132844051000f, -0.9105991643000f, 0, -0.5815751419000f, -0.5162925989000f, +0.6286591339000f, 0, -0.0370370478500f, +0.8273785755000f, +0.5604221175000f, 0, -0.5119692504000f, +0.7953543429000f, -0.3244980058000f, 0, -0.2682417366000f, -0.9572290247000f, -0.1084387619000f, 0, -0.2322482736000f, -0.9679131102000f, -0.0959424332400f, 0, +0.3554328906000f, -0.8881505545000f, +0.2913006227000f, 0, +0.7346520519000f, -0.4371373164000f, +0.5188422971000f, 0,
    +0.9985120116000f, +0.0465901116100f, -0.0283394457700f, 0, -0.3727687496000f, -0.9082481361000f, +0.1900757285000f, 0, +0.9173737700000f, -0.3483642108000f, +0.1925298489000f, 0, +0.2714911074000f, +0.4147529736000f, -0.8684886582000f, 0, +0.5131763485000f, -0.7116334161000f, +0.4798207128000f, 0, -0.8737353606000f, +0.1888699200000f, -0.4482350644000f, 0, +0.8460043821000f, -0.3725217914000f, +0.3814499973000f, 0, +0.8978727456000f, -0.1780209141000f, -0.4026575304000f, 0,
//  +0.9985120116000f, +0.0465901116100f, -0.0283394457700f, 0, -0.3727687496000f, -0.9082481361000f, +0.1900757285000f, 0, +0.9173737700000f, -0.3483642108000f, +0.1925298489000f, 0, +0.2714911074000f, +0.4147529736000f, -0.8684886582000f, 0, +0.5131763485000f, -0.7116334161000f, +0.4798207128000f, 0, -0.8737353606000f, +0.1888699200000f, -0.4482350644000f, 0, +0.8460043821000f, -0.3725217914000f, +0.3814499973000f, 0, +0.8978727456000f, -0.1780209141000f, -0.4026575304000f, 0,
    +0.2178065647000f, -0.9698322841000f, -0.1094789531000f, 0, -0.1518031304000f, -0.7788918132000f, -0.6085091231000f, 0, -0.2600384876000f, -0.4755398075000f, -0.8403819825000f, 0, +0.5723135090000f, -0.7474340931000f, -0.3373418503000f, 0, -0.7174141009000f, +0.1699017182000f, -0.6756111411000f, 0, -0.6841807840000f, +0.0214570759300f, -0.7289967412000f, 0, -0.2007447902000f, +0.0655560578900f, -0.9774476623000f, 0, -0.1148803697000f, -0.8044887315000f, +0.5827524187000f, 0,
//  +0.2178065647000f, -0.9698322841000f, -0.1094789531000f, 0, -0.1518031304000f, -0.7788918132000f, -0.6085091231000f, 0, -0.2600384876000f, -0.4755398075000f, -0.8403819825000f, 0, +0.5723135090000f, -0.7474340931000f, -0.3373418503000f, 0, -0.7174141009000f, +0.1699017182000f, -0.6756111411000f, 0, -0.6841807840000f, +0.0214570759300f, -0.7289967412000f, 0, -0.2007447902000f, +0.0655560578900f, -0.9774476623000f, 0, -0.1148803697000f, -0.8044887315000f, +0.5827524187000f, 0,
    -0.7870349638000f, +0.0344748923100f, +0.6159443543000f, 0, -0.2015596421000f, +0.6859872284000f, +0.6991389226000f, 0, -0.0858108251200f, -0.1092083600000f, -0.9903080513000f, 0, +0.5532693395000f, +0.7325250401000f, -0.3966107710000f, 0, -0.1842489331000f, -0.9777375055000f, -0.1004076743000f, 0, +0.0775473789000f, -0.9111505856000f, +0.4047110257000f, 0, +0.1399838409000f, +0.7601631212000f, -0.6344734459000f, 0, +0.4484419361000f, -0.8452892480000f, +0.2904925424000f, 0,
//  -0.7870349638000f, +0.0344748923100f, +0.6159443543000f, 0, -0.2015596421000f, +0.6859872284000f, +0.6991389226000f, 0, -0.0858108251200f, -0.1092083600000f, -0.9903080513000f, 0, +0.5532693395000f, +0.7325250401000f, -0.3966107710000f, 0, -0.1842489331000f, -0.9777375055000f, -0.1004076743000f, 0, +0.0775473789000f, -0.9111505856000f, +0.4047110257000f, 0, +0.1399838409000f, +0.7601631212000f, -0.6344734459000f, 0, +0.4484419361000f, -0.8452892480000f, +0.2904925424000f, 0,
};

#endif
