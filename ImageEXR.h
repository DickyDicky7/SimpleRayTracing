#pragma once
#ifndef IMAGE_EXR_H
#define IMAGE_EXR_H



#ifdef PNG_Z_DEFAULT_COMPRESSION
#undef PNG_Z_DEFAULT_COMPRESSION
#endif



    #include <FreeImage.h>
//  #include <FreeImage.h>
    #include <iostream>
//  #include <iostream>
    #include <vector>
//  #include <vector>



class ImageEXR
{
    public:
//  public:
    std::vector<float> rgbs; // RGB float buffer (HDR)
//  std::vector<float> rgbs; // RGB float buffer (HDR)
    std::uint16_t w;
//  std::uint16_t w;
    std::uint16_t h;
//  std::uint16_t h;



    ImageEXR(                    ) : w(0), h(0) {                       }
//  ImageEXR(                    ) : w(0), h(0) {                       }
    ImageEXR(const char* fileName) : w(0), h(0) { this->Load(fileName); }
//  ImageEXR(const char* fileName) : w(0), h(0) { this->Load(fileName); }



    void Load(const char* fileName)
//  void Load(const char* fileName)
    {
        FREE_IMAGE_FORMAT freeImageFormat = FreeImage_GetFileType(fileName, 0);
//      FREE_IMAGE_FORMAT freeImageFormat = FreeImage_GetFileType(fileName, 0);


        if (freeImageFormat == FREE_IMAGE_FORMAT::FIF_UNKNOWN)
//      if (freeImageFormat == FREE_IMAGE_FORMAT::FIF_UNKNOWN)
        {
            freeImageFormat = FreeImage_GetFIFFromFilename(fileName);
//          freeImageFormat = FreeImage_GetFIFFromFilename(fileName);
        }


        if (freeImageFormat == FREE_IMAGE_FORMAT::FIF_UNKNOWN || !FreeImage_FIFSupportsReading(freeImageFormat))
//      if (freeImageFormat == FREE_IMAGE_FORMAT::FIF_UNKNOWN || !FreeImage_FIFSupportsReading(freeImageFormat))
        {
            std::cerr << "#== " << "ImageEXR: unsupported or unknown image format: " << fileName << " ==#" << std::endl;
//          std::cerr << "#== " << "ImageEXR: unsupported or unknown image format: " << fileName << " ==#" << std::endl;
            return;
//          return;
        }


        FIBITMAP* freeImageBitmap = FreeImage_Load(freeImageFormat, fileName, 0);
//      FIBITMAP* freeImageBitmap = FreeImage_Load(freeImageFormat, fileName, 0);
        if (!freeImageBitmap)
//      if (!freeImageBitmap)
        {
            std::cerr << "#== " << "ImageEXR: failed to load EXR image: " << fileName << " ==#" << std::endl;
//          std::cerr << "#== " << "ImageEXR: failed to load EXR image: " << fileName << " ==#" << std::endl;
            return;
//          return;
        }


        FIBITMAP* freeImageBitmapRGBF = FreeImage_ConvertToRGBF(freeImageBitmap);
//      FIBITMAP* freeImageBitmapRGBF = FreeImage_ConvertToRGBF(freeImageBitmap);
        FreeImage_Unload(freeImageBitmap);
//      FreeImage_Unload(freeImageBitmap);


        if (!freeImageBitmapRGBF)
//      if (!freeImageBitmapRGBF)
        {
            std::cerr << "#== " << "ImageEXR: conversion to float RGB failed: " << fileName << " ==#" << std::endl;
//          std::cerr << "#== " << "ImageEXR: conversion to float RGB failed: " << fileName << " ==#" << std::endl;
            return;
//          return;
        }


        FreeImage_FlipVertical (freeImageBitmapRGBF);
//      FreeImage_FlipVertical (freeImageBitmapRGBF);


        w = FreeImage_GetWidth (freeImageBitmapRGBF);
//      w = FreeImage_GetWidth (freeImageBitmapRGBF);
        h = FreeImage_GetHeight(freeImageBitmapRGBF);
//      h = FreeImage_GetHeight(freeImageBitmapRGBF);


        rgbs.reserve(static_cast<std::size_t>(w) * h * 3);
//      rgbs.reserve(static_cast<std::size_t>(w) * h * 3);


        for (int y = 0; y < h; ++y)
//      for (int y = 0; y < h; ++y)
        {
            FIRGBF* pixel = (FIRGBF*)FreeImage_GetScanLine(freeImageBitmapRGBF, y);
//          FIRGBF* pixel = (FIRGBF*)FreeImage_GetScanLine(freeImageBitmapRGBF, y);
        for (int x = 0; x < w; ++x)
//      for (int x = 0; x < w; ++x)
        {
            rgbs.emplace_back(pixel[x].red  );
//          rgbs.emplace_back(pixel[x].red  );
            rgbs.emplace_back(pixel[x].green);
//          rgbs.emplace_back(pixel[x].green);
            rgbs.emplace_back(pixel[x].blue );
//          rgbs.emplace_back(pixel[x].blue );
        }
        }


        FreeImage_Unload(freeImageBitmapRGBF);
//      FreeImage_Unload(freeImageBitmapRGBF);
    }
};



#endif
