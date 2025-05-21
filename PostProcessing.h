#pragma once
#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H



    #include "lazy.h"
//  #include "lazy.h"
    #include <vector>
//  #include <vector>
    #include <cmath>
//  #include <cmath>
    #include <algorithm>
//  #include <algorithm>



    namespace post_processing
//  namespace post_processing
{



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



    static inline const constexpr void GaussianBlur001(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
//  static inline const constexpr void GaussianBlur001(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
{
//  DENOISE 001
//  DENOISE 001
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
}



    static inline const constexpr void GaussianBlur002(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
//  static inline const constexpr void GaussianBlur002(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
{
//  DENOISE 002
//  DENOISE 002
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
}



    static inline const constexpr void ChromaticAberration(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
//  static inline const constexpr void ChromaticAberration(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
{
// CHROMATIC ABERRATION
// CHROMATIC ABERRATION
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
}



    static inline const constexpr void BayerMatrixDithering(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
//  static inline const constexpr void BayerMatrixDithering(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
{
//  BAYER MATRIX DITHERING
//  BAYER MATRIX DITHERING
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
}



    static inline const constexpr void BilateralFiltering(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
//  static inline const constexpr void BilateralFiltering(std::vector<float>& rgbs, std::uint16_t imgW, std::uint16_t imgH, std::uint8_t numberOfChannels)
{
//  BILATERAL FILTERING
//  BILATERAL FILTERING
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
}



}



#endif
