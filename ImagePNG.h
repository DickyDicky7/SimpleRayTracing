#pragma once
#ifndef IMAGE_PNG_H
#define IMAGE_PNG_H


// NUGET: <package id="libpng_static" version="1.6.39.1" targetFramework="native" />
// NUGET: <package id="libpng_static" version="1.6.39.1" targetFramework="native" />
// NUGET: <package id="  zlib_static" version="1.2.11.9" targetFramework="native" />
// NUGET: <package id="  zlib_static" version="1.2.11.9" targetFramework="native" />
// project Properties -> All Configurations/All Platforms -> C/C++  -> General -> Additional Include Directories -> D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\include;D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\include;%(AdditionalIncludeDirectories)
// project Properties -> All Configurations/All Platforms -> C/C++  -> General -> Additional Include Directories -> D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\include;D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\include;%(AdditionalIncludeDirectories)
// project Properties -> All Configurations/All Platforms -> Linker -> Input   -> Additional Dependencies        -> libpng.lib;zlib.lib;%(AdditionalDependencies)
// project Properties -> All Configurations/All Platforms -> Linker -> Input   -> Additional Dependencies        -> libpng.lib;zlib.lib;%(AdditionalDependencies)
// project Properties ->              Debug/All Platforms -> Linker -> General -> Additional Library Directories -> D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\lib\x64\v143\Debug  \MultiThreadedDebugDLL;D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\lib\x64\v142\Debug  \MultiThreadedDebugDLL;%(AdditionalLibraryDirectories)
// project Properties ->              Debug/All Platforms -> Linker -> General -> Additional Library Directories -> D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\lib\x64\v143\Debug  \MultiThreadedDebugDLL;D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\lib\x64\v142\Debug  \MultiThreadedDebugDLL;%(AdditionalLibraryDirectories)
// project Properties ->            Release/All Platforms -> Linker -> General -> Additional Library Directories -> D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\lib\x64\v143\Release\MultiThreadedDLL     ;D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\lib\x64\v142\Release\MultiThreadedDLL     ;%(AdditionalLibraryDirectories)
// project Properties ->            Release/All Platforms -> Linker -> General -> Additional Library Directories -> D:\Workspace\SimpleRayTracingLocal\packages\libpng_static.1.6.39.1\build\native\lib\x64\v143\Release\MultiThreadedDLL     ;D:\Workspace\SimpleRayTracingLocal\packages\zlib_static.1.2.11.9\build\native\lib\x64\v142\Release\MultiThreadedDLL     ;%(AdditionalLibraryDirectories)



    #include <png.h>
//  #include <png.h>



    class ImagePNG
//  class ImagePNG
{
    public:
//  public:
    std::vector<float> rgbs; // RGB float buffer (0.0f to 1.0f)
//  std::vector<float> rgbs; // RGB float buffer (0.0f to 1.0f)
    std::uint16_t w;
//  std::uint16_t w;
    std::uint16_t h;
//  std::uint16_t h;



    ImagePNG(                    ) : w(0), h(0) {                       }
//  ImagePNG(                    ) : w(0), h(0) {                       }
    ImagePNG(const char* fileName) : w(0), h(0) { this->Load(fileName); }
//  ImagePNG(const char* fileName) : w(0), h(0) { this->Load(fileName); }



    void Load(const char* fileName)
//  void Load(const char* fileName)
    {
        FILE* fp = fopen(fileName, "rb");
//      FILE* fp = fopen(fileName, "rb");
        if (!fp)
//      if (!fp)
        {
            throw std::runtime_error("Failed to open PNG file");
//          throw std::runtime_error("Failed to open PNG file");
        }


        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
//      png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png)
//      if (!png)
        {
            fclose(fp);
//          fclose(fp);
            throw std::runtime_error("Failed to create PNG read struct");
//          throw std::runtime_error("Failed to create PNG read struct");
        }


        png_infop info = png_create_info_struct(png);
//      png_infop info = png_create_info_struct(png);
        if (!info)
//      if (!info)
        {
            png_destroy_read_struct(&png, nullptr, nullptr);
//          png_destroy_read_struct(&png, nullptr, nullptr);
            fclose(fp);
//          fclose(fp);
            throw std::runtime_error("Failed to create PNG info struct");
//          throw std::runtime_error("Failed to create PNG info struct");
        }


        if (setjmp(png_jmpbuf(png)))
//      if (setjmp(png_jmpbuf(png)))
        {
            png_destroy_read_struct(&png, &info, nullptr);
//          png_destroy_read_struct(&png, &info, nullptr);
            fclose(fp);
//          fclose(fp);
            throw std::runtime_error("Error during PNG reading");
//          throw std::runtime_error("Error during PNG reading");
        }


        png_init_io  (png, fp  );
//      png_init_io  (png, fp  );
        png_read_info(png, info);
//      png_read_info(png, info);


        w = png_get_image_width (png, info);
//      w = png_get_image_width (png, info);
        h = png_get_image_height(png, info);
//      h = png_get_image_height(png, info);
        png_byte colorType = png_get_color_type(png, info);
//      png_byte colorType = png_get_color_type(png, info);
        png_byte bitDepth  = png_get_bit_depth (png, info);
//      png_byte bitDepth  = png_get_bit_depth (png, info);


        if (bitDepth == 16)
//      if (bitDepth == 16)
        {
            png_set_strip_16(png);
//          png_set_strip_16(png);
        }
        if (colorType == PNG_COLOR_TYPE_PALETTE)
//      if (colorType == PNG_COLOR_TYPE_PALETTE)
        {
            png_set_palette_to_rgb(png);
//          png_set_palette_to_rgb(png);
        }
        if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
//      if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        {
            png_set_expand_gray_1_2_4_to_8(png);
//          png_set_expand_gray_1_2_4_to_8(png);
        }
        if (png_get_valid(png, info, PNG_INFO_tRNS))
//      if (png_get_valid(png, info, PNG_INFO_tRNS))
        {
            png_set_tRNS_to_alpha(png);
//          png_set_tRNS_to_alpha(png);
        }
        if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
//      if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        {
            png_set_gray_to_rgb(png);
//          png_set_gray_to_rgb(png);
        }
        if (colorType & PNG_COLOR_MASK_ALPHA)
//      if (colorType & PNG_COLOR_MASK_ALPHA)
        {
            png_set_strip_alpha(png);
//          png_set_strip_alpha(png);
        }


        // Add gamma correction to output linear values
        // Add gamma correction to output linear values
//      png_set_gamma(png, 1.0, PNG_DEFAULT_sRGB); // Output linear (1.0), assume input is sRGB
//      png_set_gamma(png, 1.0, PNG_DEFAULT_sRGB); // Output linear (1.0), assume input is sRGB
        // Note: If the PNG has a gAMA chunk, libpng will use it; otherwise, it assumes sRGB (2.2)
        // Note: If the PNG has a gAMA chunk, libpng will use it; otherwise, it assumes sRGB (2.2)


        png_read_update_info(png, info);
//      png_read_update_info(png, info);


        std::vector<png_bytep> rowPointers(h);
//      std::vector<png_bytep> rowPointers(h);
        size_t rowBytes = png_get_rowbytes(png, info);
//      size_t rowBytes = png_get_rowbytes(png, info);
        std::vector<png_byte> imageData(rowBytes * h);
//      std::vector<png_byte> imageData(rowBytes * h);


        for (int y = 0; y < h; y++)
//      for (int y = 0; y < h; y++)
        {
            rowPointers[y] = &imageData[y * rowBytes];
//          rowPointers[y] = &imageData[y * rowBytes];
        }


        png_read_image(png, rowPointers.data());
//      png_read_image(png, rowPointers.data());


        png_destroy_read_struct(&png, &info, nullptr);
//      png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
//      fclose(fp);


        rgbs.resize(static_cast<std::size_t>(w) * h * 3);
//      rgbs.resize(static_cast<std::size_t>(w) * h * 3);
        for (int y = 0; y < h; y++)
//      for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
//          for (int x = 0; x < w; x++)
            {
                size_t srcIdx = (static_cast<std::size_t>(y) * w + x) * 3;
//              size_t srcIdx = (static_cast<std::size_t>(y) * w + x) * 3;
                size_t dstIdx = srcIdx;
//              size_t dstIdx = srcIdx;
                rgbs[dstIdx + 0] = imageData[srcIdx + 0] / 255.0f; // R
//              rgbs[dstIdx + 0] = imageData[srcIdx + 0] / 255.0f; // R
                rgbs[dstIdx + 1] = imageData[srcIdx + 1] / 255.0f; // G
//              rgbs[dstIdx + 1] = imageData[srcIdx + 1] / 255.0f; // G
                rgbs[dstIdx + 2] = imageData[srcIdx + 2] / 255.0f; // B
//              rgbs[dstIdx + 2] = imageData[srcIdx + 2] / 255.0f; // B
            }
        }
    }
};

#endif
