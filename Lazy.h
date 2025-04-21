#pragma once
#ifndef LAZY_H
#define LAZY_H


#include <numbers>


namespace lazy
{
    thread_local static inline const constexpr float G_3X3_S1_0_0 = 0.075f;
    thread_local static inline const constexpr float G_3X3_S1_0_1 = 0.124f;
    thread_local static inline const constexpr float G_3X3_S1_0_2 = 0.075f;
    thread_local static inline const constexpr float G_3X3_S1_1_0 = 0.124f;
    thread_local static inline const constexpr float G_3X3_S1_1_1 = 0.204f;
    thread_local static inline const constexpr float G_3X3_S1_1_2 = 0.124f;
    thread_local static inline const constexpr float G_3X3_S1_2_0 = 0.075f;
    thread_local static inline const constexpr float G_3X3_S1_2_1 = 0.124f;
    thread_local static inline const constexpr float G_3X3_S1_2_2 = 0.075f;


    thread_local static inline const constexpr float G_5X5_S1_0_0 = 0.005f;
    thread_local static inline const constexpr float G_5X5_S1_0_1 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_0_2 = 0.034f;
    thread_local static inline const constexpr float G_5X5_S1_0_3 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_0_4 = 0.005f;
    thread_local static inline const constexpr float G_5X5_S1_1_0 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_1_1 = 0.087f;
    thread_local static inline const constexpr float G_5X5_S1_1_2 = 0.151f;
    thread_local static inline const constexpr float G_5X5_S1_1_3 = 0.087f;
    thread_local static inline const constexpr float G_5X5_S1_1_4 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_2_0 = 0.034f;
    thread_local static inline const constexpr float G_5X5_S1_2_1 = 0.151f;
    thread_local static inline const constexpr float G_5X5_S1_2_2 = 0.250f;
    thread_local static inline const constexpr float G_5X5_S1_2_3 = 0.151f;
    thread_local static inline const constexpr float G_5X5_S1_2_4 = 0.034f;
    thread_local static inline const constexpr float G_5X5_S1_3_0 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_3_1 = 0.087f;
    thread_local static inline const constexpr float G_5X5_S1_3_2 = 0.151f;
    thread_local static inline const constexpr float G_5X5_S1_3_3 = 0.087f;
    thread_local static inline const constexpr float G_5X5_S1_3_4 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_4_0 = 0.005f;
    thread_local static inline const constexpr float G_5X5_S1_4_1 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_4_2 = 0.034f;
    thread_local static inline const constexpr float G_5X5_S1_4_3 = 0.021f;
    thread_local static inline const constexpr float G_5X5_S1_4_4 = 0.005f;


    thread_local static inline const constexpr float G_7X7_S1_0_0 = 0.000f;
    thread_local static inline const constexpr float G_7X7_S1_0_1 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_0_2 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_0_3 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_0_4 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_0_5 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_0_6 = 0.000f;
    thread_local static inline const constexpr float G_7X7_S1_1_0 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_1_1 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_1_2 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_1_3 = 0.089f;
    thread_local static inline const constexpr float G_7X7_S1_1_4 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_1_5 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_1_6 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_2_0 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_2_1 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_2_2 = 0.149f;
    thread_local static inline const constexpr float G_7X7_S1_2_3 = 0.234f;
    thread_local static inline const constexpr float G_7X7_S1_2_4 = 0.149f;
    thread_local static inline const constexpr float G_7X7_S1_2_5 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_2_6 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_3_0 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_3_1 = 0.089f;
    thread_local static inline const constexpr float G_7X7_S1_3_2 = 0.234f;
    thread_local static inline const constexpr float G_7X7_S1_3_3 = 0.367f;
    thread_local static inline const constexpr float G_7X7_S1_3_4 = 0.234f;
    thread_local static inline const constexpr float G_7X7_S1_3_5 = 0.089f;
    thread_local static inline const constexpr float G_7X7_S1_3_6 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_4_0 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_4_1 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_4_2 = 0.149f;
    thread_local static inline const constexpr float G_7X7_S1_4_3 = 0.234f;
    thread_local static inline const constexpr float G_7X7_S1_4_4 = 0.149f;
    thread_local static inline const constexpr float G_7X7_S1_4_5 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_4_6 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_5_0 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_5_1 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_5_2 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_5_3 = 0.089f;
    thread_local static inline const constexpr float G_7X7_S1_5_4 = 0.054f;
    thread_local static inline const constexpr float G_7X7_S1_5_5 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_5_6 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_6_0 = 0.000f;
    thread_local static inline const constexpr float G_7X7_S1_6_1 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_6_2 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_6_3 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S1_6_4 = 0.008f;
    thread_local static inline const constexpr float G_7X7_S1_6_5 = 0.002f;
    thread_local static inline const constexpr float G_7X7_S1_6_6 = 0.000f;






    thread_local static inline const constexpr float G_3X3_S2_0_0 = 0.102f;
    thread_local static inline const constexpr float G_3X3_S2_0_1 = 0.115f;
    thread_local static inline const constexpr float G_3X3_S2_0_2 = 0.102f;
    thread_local static inline const constexpr float G_3X3_S2_1_0 = 0.115f;
    thread_local static inline const constexpr float G_3X3_S2_1_1 = 0.131f;
    thread_local static inline const constexpr float G_3X3_S2_1_2 = 0.115f;
    thread_local static inline const constexpr float G_3X3_S2_2_0 = 0.102f;
    thread_local static inline const constexpr float G_3X3_S2_2_1 = 0.115f;
    thread_local static inline const constexpr float G_3X3_S2_2_2 = 0.102f;


    thread_local static inline const constexpr float G_5X5_S2_0_0 = 0.041f;
    thread_local static inline const constexpr float G_5X5_S2_0_1 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_0_2 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S2_0_3 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_0_4 = 0.041f;
    thread_local static inline const constexpr float G_5X5_S2_1_0 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_1_1 = 0.066f;
    thread_local static inline const constexpr float G_5X5_S2_1_2 = 0.075f;
    thread_local static inline const constexpr float G_5X5_S2_1_3 = 0.066f;
    thread_local static inline const constexpr float G_5X5_S2_1_4 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_2_0 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S2_2_1 = 0.075f;
    thread_local static inline const constexpr float G_5X5_S2_2_2 = 0.086f;
    thread_local static inline const constexpr float G_5X5_S2_2_3 = 0.075f;
    thread_local static inline const constexpr float G_5X5_S2_2_4 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S2_3_0 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_3_1 = 0.066f;
    thread_local static inline const constexpr float G_5X5_S2_3_2 = 0.075f;
    thread_local static inline const constexpr float G_5X5_S2_3_3 = 0.066f;
    thread_local static inline const constexpr float G_5X5_S2_3_4 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_4_0 = 0.041f;
    thread_local static inline const constexpr float G_5X5_S2_4_1 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_4_2 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S2_4_3 = 0.052f;
    thread_local static inline const constexpr float G_5X5_S2_4_4 = 0.041f;


    thread_local static inline const constexpr float G_7X7_S2_0_0 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S2_0_1 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_0_2 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_0_3 = 0.023f;
    thread_local static inline const constexpr float G_7X7_S2_0_4 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_0_5 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_0_6 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S2_1_0 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_1_1 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_1_2 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_1_3 = 0.029f;
    thread_local static inline const constexpr float G_7X7_S2_1_4 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_1_5 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_1_6 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_2_0 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_2_1 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_2_2 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S2_2_3 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S2_2_4 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S2_2_5 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_2_6 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_3_0 = 0.023f;
    thread_local static inline const constexpr float G_7X7_S2_3_1 = 0.029f;
    thread_local static inline const constexpr float G_7X7_S2_3_2 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S2_3_3 = 0.040f;
    thread_local static inline const constexpr float G_7X7_S2_3_4 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S2_3_5 = 0.029f;
    thread_local static inline const constexpr float G_7X7_S2_3_6 = 0.023f;
    thread_local static inline const constexpr float G_7X7_S2_4_0 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_4_1 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_4_2 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S2_4_3 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S2_4_4 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S2_4_5 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_4_6 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_5_0 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_5_1 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_5_2 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_5_3 = 0.029f;
    thread_local static inline const constexpr float G_7X7_S2_5_4 = 0.026f;
    thread_local static inline const constexpr float G_7X7_S2_5_5 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_5_6 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_6_0 = 0.013f;
    thread_local static inline const constexpr float G_7X7_S2_6_1 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_6_2 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_6_3 = 0.023f;
    thread_local static inline const constexpr float G_7X7_S2_6_4 = 0.021f;
    thread_local static inline const constexpr float G_7X7_S2_6_5 = 0.017f;
    thread_local static inline const constexpr float G_7X7_S2_6_6 = 0.013f;






    thread_local static inline const constexpr float G_3X3_S3_0_0 = 0.106f;
    thread_local static inline const constexpr float G_3X3_S3_0_1 = 0.112f;
    thread_local static inline const constexpr float G_3X3_S3_0_2 = 0.106f;
    thread_local static inline const constexpr float G_3X3_S3_1_0 = 0.112f;
    thread_local static inline const constexpr float G_3X3_S3_1_1 = 0.118f;
    thread_local static inline const constexpr float G_3X3_S3_1_2 = 0.112f;
    thread_local static inline const constexpr float G_3X3_S3_2_0 = 0.106f;
    thread_local static inline const constexpr float G_3X3_S3_2_1 = 0.112f;
    thread_local static inline const constexpr float G_3X3_S3_2_2 = 0.106f;


    thread_local static inline const constexpr float G_5X5_S3_0_0 = 0.054f;
    thread_local static inline const constexpr float G_5X5_S3_0_1 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_0_2 = 0.061f;
    thread_local static inline const constexpr float G_5X5_S3_0_3 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_0_4 = 0.054f;
    thread_local static inline const constexpr float G_5X5_S3_1_0 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_1_1 = 0.064f;
    thread_local static inline const constexpr float G_5X5_S3_1_2 = 0.067f;
    thread_local static inline const constexpr float G_5X5_S3_1_3 = 0.064f;
    thread_local static inline const constexpr float G_5X5_S3_1_4 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_2_0 = 0.061f;
    thread_local static inline const constexpr float G_5X5_S3_2_1 = 0.067f;
    thread_local static inline const constexpr float G_5X5_S3_2_2 = 0.070f;
    thread_local static inline const constexpr float G_5X5_S3_2_3 = 0.067f;
    thread_local static inline const constexpr float G_5X5_S3_2_4 = 0.061f;
    thread_local static inline const constexpr float G_5X5_S3_3_0 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_3_1 = 0.064f;
    thread_local static inline const constexpr float G_5X5_S3_3_2 = 0.067f;
    thread_local static inline const constexpr float G_5X5_S3_3_3 = 0.064f;
    thread_local static inline const constexpr float G_5X5_S3_3_4 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_4_0 = 0.054f;
    thread_local static inline const constexpr float G_5X5_S3_4_1 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_4_2 = 0.061f;
    thread_local static inline const constexpr float G_5X5_S3_4_3 = 0.059f;
    thread_local static inline const constexpr float G_5X5_S3_4_4 = 0.054f;


    thread_local static inline const constexpr float G_7X7_S3_0_0 = 0.028f;
    thread_local static inline const constexpr float G_7X7_S3_0_1 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_0_2 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_0_3 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_0_4 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_0_5 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_0_6 = 0.028f;
    thread_local static inline const constexpr float G_7X7_S3_1_0 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_1_1 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_1_2 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_1_3 = 0.037f;
    thread_local static inline const constexpr float G_7X7_S3_1_4 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_1_5 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_1_6 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_2_0 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_2_1 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_2_2 = 0.039f;
    thread_local static inline const constexpr float G_7X7_S3_2_3 = 0.040f;
    thread_local static inline const constexpr float G_7X7_S3_2_4 = 0.039f;
    thread_local static inline const constexpr float G_7X7_S3_2_5 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_2_6 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_3_0 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_3_1 = 0.037f;
    thread_local static inline const constexpr float G_7X7_S3_3_2 = 0.040f;
    thread_local static inline const constexpr float G_7X7_S3_3_3 = 0.041f;
    thread_local static inline const constexpr float G_7X7_S3_3_4 = 0.040f;
    thread_local static inline const constexpr float G_7X7_S3_3_5 = 0.037f;
    thread_local static inline const constexpr float G_7X7_S3_3_6 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_4_0 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_4_1 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_4_2 = 0.039f;
    thread_local static inline const constexpr float G_7X7_S3_4_3 = 0.040f;
    thread_local static inline const constexpr float G_7X7_S3_4_4 = 0.039f;
    thread_local static inline const constexpr float G_7X7_S3_4_5 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_4_6 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_5_0 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_5_1 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_5_2 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_5_3 = 0.037f;
    thread_local static inline const constexpr float G_7X7_S3_5_4 = 0.036f;
    thread_local static inline const constexpr float G_7X7_S3_5_5 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_5_6 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_6_0 = 0.028f;
    thread_local static inline const constexpr float G_7X7_S3_6_1 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_6_2 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_6_3 = 0.034f;
    thread_local static inline const constexpr float G_7X7_S3_6_4 = 0.033f;
    thread_local static inline const constexpr float G_7X7_S3_6_5 = 0.031f;
    thread_local static inline const constexpr float G_7X7_S3_6_6 = 0.028f;






    constexpr std::array<float, 4> GenerateBayer2x2()
//  constexpr std::array<float, 4> GenerateBayer2x2()
    {
        return std::array<float, 4>
//      return std::array<float, 4>
        {
            0.0f / 4.0f, 2.0f / 4.0f, // Row 0
//          0.0f / 4.0f, 2.0f / 4.0f, // Row 0
            3.0f / 4.0f, 1.0f / 4.0f, // Row 1
//          3.0f / 4.0f, 1.0f / 4.0f, // Row 1
        };
    }

    constexpr std::array<float, 16> GenerateBayer4x4()
//  constexpr std::array<float, 16> GenerateBayer4x4()
    {
        std::array<float, 16> result{};
//      std::array<float, 16> result{};
        constexpr std::array<float, 4> b2 = GenerateBayer2x2();
//      constexpr std::array<float, 4> b2 = GenerateBayer2x2();
        constexpr std::size_t n = 4;
//      constexpr std::size_t n = 4;
        for (std::size_t y = 0; y < n; ++y)
//      for (std::size_t y = 0; y < n; ++y)
        {
            for (std::size_t x = 0; x < n; ++x)
//          for (std::size_t x = 0; x < n; ++x)
            {
                std::size_t idx = x + y * n;
//              std::size_t idx = x + y * n;
                // Scale 2x2 matrix by 4 and add offset based on position
//              // Scale 2x2 matrix by 4 and add offset based on position
                float base = b2[(x / 2) + (y / 2) * 2] * 4.0f;
//              float base = b2[(x / 2) + (y / 2) * 2] * 4.0f;
                float offset = static_cast<float>((x % 2) + (y % 2) * 2); // Pattern: 0, 1, 2, 3
//              float offset = static_cast<float>((x % 2) + (y % 2) * 2); // Pattern: 0, 1, 2, 3
                result[idx] = (base + offset) / 16.0f;
//              result[idx] = (base + offset) / 16.0f;
            }
        }
        return result;
//      return result;
    }

    constexpr std::array<float, 64> GenerateBayer8x8()
//  constexpr std::array<float, 64> GenerateBayer8x8()
    {
        std::array<float, 64> result{};
//      std::array<float, 64> result{};
        constexpr std::array<float, 16> b4 = GenerateBayer4x4();
//      constexpr std::array<float, 16> b4 = GenerateBayer4x4();
        constexpr std::size_t n = 8;
//      constexpr std::size_t n = 8;
        for (std::size_t y = 0; y < n; ++y)
//      for (std::size_t y = 0; y < n; ++y)
        {
            for (std::size_t x = 0; x < n; ++x)
//          for (std::size_t x = 0; x < n; ++x)
            {
                std::size_t idx = x + y * n;
//              std::size_t idx = x + y * n;
                // Scale 4x4 matrix by 4 and add offset based on position
//              // Scale 4x4 matrix by 4 and add offset based on position
                float base = b4[(x / 2) + (y / 2) * 4] * 4.0f;
//              float base = b4[(x / 2) + (y / 2) * 4] * 4.0f;
                float offset = static_cast<float>((x % 2) + (y % 2) * 2);
//              float offset = static_cast<float>((x % 2) + (y % 2) * 2);
                result[idx] = (base + offset) / 64.0f;
//              result[idx] = (base + offset) / 64.0f;
            }
        }
        return result;
//      return result;
    }

    constexpr std::array<float, 256> GenerateBayer16x16()
//  constexpr std::array<float, 256> GenerateBayer16x16()
    {
        std::array<float, 256> result{};
//      std::array<float, 256> result{};
        constexpr std::array<float, 64> b8 = GenerateBayer8x8();
//      constexpr std::array<float, 64> b8 = GenerateBayer8x8();
        constexpr std::size_t n = 16;
//      constexpr std::size_t n = 16;
        for (std::size_t y = 0; y < n; ++y)
//      for (std::size_t y = 0; y < n; ++y)
        {
            for (std::size_t x = 0; x < n; ++x)
//          for (std::size_t x = 0; x < n; ++x)
            {
                std::size_t idx = x + y * n;
//              std::size_t idx = x + y * n;
                // Scale 8x8 matrix by 4 and add offset based on position
//              // Scale 8x8 matrix by 4 and add offset based on position
                float base = b8[(x / 2) + (y / 2) * 8] * 4.0f;
//              float base = b8[(x / 2) + (y / 2) * 8] * 4.0f;
                float offset = static_cast<float>((x % 2) + (y % 2) * 2);
//              float offset = static_cast<float>((x % 2) + (y % 2) * 2);
                result[idx] = (base + offset) / 256.0f;
//              result[idx] = (base + offset) / 256.0f;
            }
        }
        return result;
//      return result;
    }

    static inline const constexpr std::array<float, 4> bayer2x2 = GenerateBayer2x2();
//  static inline const constexpr std::array<float, 4> bayer2x2 = GenerateBayer2x2();
    static inline const constexpr std::array<float, 16> bayer4x4 = GenerateBayer4x4();
//  static inline const constexpr std::array<float, 16> bayer4x4 = GenerateBayer4x4();
    static inline const constexpr std::array<float, 64> bayer8x8 = GenerateBayer8x8();
//  static inline const constexpr std::array<float, 64> bayer8x8 = GenerateBayer8x8();
    static inline const constexpr std::array<float, 256> bayer16x16 = GenerateBayer16x16();
//  static inline const constexpr std::array<float, 256> bayer16x16 = GenerateBayer16x16();

    static inline const constexpr float GetValueFromBayer2x2(std::size_t x, std::size_t y)
//  static inline const constexpr float GetValueFromBayer2x2(std::size_t x, std::size_t y)
    {
        return bayer2x2[x % 2 + y % 2 * 2];
//      return bayer2x2[x % 2 + y % 2 * 2];
    }

    static inline const constexpr float GetValueFromBayer4x4(std::size_t x, std::size_t y)
//  static inline const constexpr float GetValueFromBayer4x4(std::size_t x, std::size_t y)
    {
        return bayer4x4[x % 4 + y % 4 * 4];
//      return bayer4x4[x % 4 + y % 4 * 4];
    }

    static inline const constexpr float GetValueFromBayer8x8(std::size_t x, std::size_t y)
//  static inline const constexpr float GetValueFromBayer8x8(std::size_t x, std::size_t y)
    {
        return bayer8x8[x % 8 + y % 8 * 8];
//      return bayer8x8[x % 8 + y % 8 * 8];
    }

    static inline const constexpr float GetValueFromBayer16x16(std::size_t x, std::size_t y)
//  static inline const constexpr float GetValueFromBayer16x16(std::size_t x, std::size_t y)
    {
        return bayer16x16[x % 16 + y % 16 * 16];
//      return bayer16x16[x % 16 + y % 16 * 16];
    }






    static inline const constexpr float DegToRad(float deg) { return deg * std::numbers::pi_v<float> / 180.0f; }
//  static inline const constexpr float DegToRad(float deg) { return deg * std::numbers::pi_v<float> / 180.0f; }

    static inline const constexpr float RadToDeg(float rad) { return rad * 180.0f / std::numbers::pi_v<float>; }
//  static inline const constexpr float RadToDeg(float rad) { return rad * 180.0f / std::numbers::pi_v<float>; }
}


#endif
