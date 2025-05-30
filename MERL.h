#pragma once
#ifndef MERL_H
#define MERL_H


    #include <math.h>
//  #include <math.h>
    #include <stdio.h>
//  #include <stdio.h>
    #include <array>
//  #include <array>
    #include <vector>
//  #include <vector>
    #include <numbers>
//  #include <numbers>
    #include <stdlib.h>
//  #include <stdlib.h>


    namespace merl
//  namespace merl
{
    constexpr int BRDF_SAMPLING_RES_THETA_H =  90;
//  constexpr int BRDF_SAMPLING_RES_THETA_H =  90;
    constexpr int BRDF_SAMPLING_RES_THETA_D =  90;
//  constexpr int BRDF_SAMPLING_RES_THETA_D =  90;
    constexpr int BRDF_SAMPLING_RES_PHI_D   = 360;
//  constexpr int BRDF_SAMPLING_RES_PHI_D   = 360;


    constexpr double R_SCALE = 1.00 / 1500.0;
//  constexpr double R_SCALE = 1.00 / 1500.0;
    constexpr double G_SCALE = 1.15 / 1500.0;
//  constexpr double G_SCALE = 1.15 / 1500.0;
    constexpr double B_SCALE = 1.66 / 1500.0;
//  constexpr double B_SCALE = 1.66 / 1500.0;


    // cross product of two vectors
    // cross product of two vectors
    inline static void cross_product(double* v1, double* v2, double* out)
//  inline static void cross_product(double* v1, double* v2, double* out)
    {
        out[0] = v1[1] * v2[2] - v1[2] * v2[1];
//      out[0] = v1[1] * v2[2] - v1[2] * v2[1];
        out[1] = v1[2] * v2[0] - v1[0] * v2[2];
//      out[1] = v1[2] * v2[0] - v1[0] * v2[2];
        out[2] = v1[0] * v2[1] - v1[1] * v2[0];
//      out[2] = v1[0] * v2[1] - v1[1] * v2[0];
    }


    // normalize vector
    // normalize vector
    inline static void normalize(double* v)
//  inline static void normalize(double* v)
    {
        // normalize
        // normalize
        double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
//      double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] = v[0] / len;
        v[1] = v[1] / len;
        v[2] = v[2] / len;
    }


    // rotate vector along one axis
    // rotate vector along one axis
    inline static void rotate_vector(double* vector, double* axis, double angle, double* out)
//  inline static void rotate_vector(double* vector, double* axis, double angle, double* out)
    {
        double temp;
//      double temp;
        double cross[3];
//      double cross[3];
        double cos_ang = cos(angle);
        double sin_ang = sin(angle);

        out[0] = vector[0] * cos_ang;
        out[1] = vector[1] * cos_ang;
        out[2] = vector[2] * cos_ang;

        temp = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
//      temp = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
        temp = temp * (1.0 - cos_ang);
//      temp = temp * (1.0 - cos_ang);

        out[0] += axis[0] * temp;
        out[1] += axis[1] * temp;
        out[2] += axis[2] * temp;

        cross_product(axis, vector, cross);
//      cross_product(axis, vector, cross);

        out[0] += cross[0] * sin_ang;
        out[1] += cross[1] * sin_ang;
        out[2] += cross[2] * sin_ang;
    }


    // convert standard coordinates to half vector/difference vector coordinates
    // convert standard coordinates to half vector/difference vector coordinates
    inline static void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out, double& theta_half, double& fi_half, double& theta_diff, double& fi_diff)
//  inline static void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out, double& theta_half, double& fi_half, double& theta_diff, double& fi_diff)
    {
        // compute in vector
        // compute in vector
        double in_vec_z = cos(theta_in);
//      double in_vec_z = cos(theta_in);
        double proj_in_vec = sin(theta_in);
//      double proj_in_vec = sin(theta_in);
        double in_vec_x = proj_in_vec * cos(fi_in);
//      double in_vec_x = proj_in_vec * cos(fi_in);
        double in_vec_y = proj_in_vec * sin(fi_in);
//      double in_vec_y = proj_in_vec * sin(fi_in);
        double in[3] = { in_vec_x, in_vec_y, in_vec_z };
//      double in[3] = { in_vec_x, in_vec_y, in_vec_z };
        normalize(in);
//      normalize(in);


        // compute out vector
        // compute out vector
        double out_vec_z = cos(theta_out);
//      double out_vec_z = cos(theta_out);
        double proj_out_vec = sin(theta_out);
//      double proj_out_vec = sin(theta_out);
        double out_vec_x = proj_out_vec * cos(fi_out);
//      double out_vec_x = proj_out_vec * cos(fi_out);
        double out_vec_y = proj_out_vec * sin(fi_out);
//      double out_vec_y = proj_out_vec * sin(fi_out);
        double out[3] = { out_vec_x, out_vec_y, out_vec_z };
//      double out[3] = { out_vec_x, out_vec_y, out_vec_z };
        normalize(out);
//      normalize(out);


        // compute halfway vector
        // compute halfway vector
        double half_x = (in_vec_x + out_vec_x) / 2.0f;
        double half_y = (in_vec_y + out_vec_y) / 2.0f;
        double half_z = (in_vec_z + out_vec_z) / 2.0f;
        double half[3] = { half_x, half_y, half_z };
//      double half[3] = { half_x, half_y, half_z };
        normalize(half);
//      normalize(half);


        // compute theta_half, fi_half
        // compute theta_half, fi_half
        theta_half = acos(half[2]);
//      theta_half = acos(half[2]);
        fi_half = atan2(half[1], half[0]);
//      fi_half = atan2(half[1], half[0]);


        double bi_normal[3] = { 0.0, 1.0, 0.0 };
//      double bi_normal[3] = { 0.0, 1.0, 0.0 };
        double    normal[3] = { 0.0, 0.0, 1.0 };
//      double    normal[3] = { 0.0, 0.0, 1.0 };
        double temp[3];
//      double temp[3];
        double diff[3];
//      double diff[3];


        // compute diff vector
        // compute diff vector
        rotate_vector(in  ,    normal, -   fi_half, temp);
//      rotate_vector(in  ,    normal, -   fi_half, temp);
        rotate_vector(temp, bi_normal, -theta_half, diff);
//      rotate_vector(temp, bi_normal, -theta_half, diff);


        // compute theta_diff, fi_diff
        // compute theta_diff, fi_diff
        theta_diff = acos(diff[2]);
//      theta_diff = acos(diff[2]);
        fi_diff = atan2(diff[1], diff[0]);
//      fi_diff = atan2(diff[1], diff[0]);
    }


    // Lookup theta_half index
    // Lookup theta_half index
    // This is a non-linear mapping!
    // This is a non-linear mapping!
    // In : [0 .. pi/2]
    // In : [0 .. pi/2]
    // Out: [0 .. 89  ]
    // Out: [0 .. 89  ]
    inline static int theta_half_index(double theta_half)
//  inline static int theta_half_index(double theta_half)
    {
        if (theta_half <= 0.0)
//      if (theta_half <= 0.0)
        {
            return 0;
//          return 0;
        }
        double        theta_half_deg = ((theta_half / (std::numbers::pi / 2.0)) * BRDF_SAMPLING_RES_THETA_H);
//      double        theta_half_deg = ((theta_half / (std::numbers::pi / 2.0)) * BRDF_SAMPLING_RES_THETA_H);
        double temp = theta_half_deg                                            * BRDF_SAMPLING_RES_THETA_H ;
//      double temp = theta_half_deg                                            * BRDF_SAMPLING_RES_THETA_H ;
        temp = sqrt(temp);
//      temp = sqrt(temp);
        int ret_val = (int)temp;
//      int ret_val = (int)temp;
        if (ret_val < 0)
//      if (ret_val < 0)
        {
            ret_val = 0;
//          ret_val = 0;
        }
        if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
//      if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
        {
            ret_val =  BRDF_SAMPLING_RES_THETA_H - 1;
//          ret_val =  BRDF_SAMPLING_RES_THETA_H - 1;
        }
        return ret_val;
//      return ret_val;
    }


    // Lookup theta_diff index
    // Lookup theta_diff index
    // In : [0 .. pi/2]
    // In : [0 .. pi/2]
    // Out: [0 .. 89  ]
    // Out: [0 .. 89  ]
    inline static int theta_diff_index(double theta_diff)
//  inline static int theta_diff_index(double theta_diff)
    {
        int tmp  = int(theta_diff / (std::numbers::pi * 0.5) * BRDF_SAMPLING_RES_THETA_D);
//      int tmp  = int(theta_diff / (std::numbers::pi * 0.5) * BRDF_SAMPLING_RES_THETA_D);
        if (tmp  < 0)
//      if (tmp  < 0)
        {
            return 0;
//          return 0;
        }
        else
        if (tmp  < BRDF_SAMPLING_RES_THETA_D - 1)
//      if (tmp  < BRDF_SAMPLING_RES_THETA_D - 1)
        {
            return tmp;
//          return tmp;
        }
        else
        {
            return BRDF_SAMPLING_RES_THETA_D - 1;
//          return BRDF_SAMPLING_RES_THETA_D - 1;
        }
    }


    // Lookup phi_diff index
    // Lookup phi_diff index
    inline static int phi_diff_index(double phi_diff)
//  inline static int phi_diff_index(double phi_diff)
    {
        // Because of reciprocity, the BRDF is unchanged under
        // Because of reciprocity, the BRDF is unchanged under
        // phi_diff -> phi_diff + std::numbers::pi
        // phi_diff -> phi_diff + std::numbers::pi
        if (phi_diff < 0.0)
//      if (phi_diff < 0.0)
        {
            phi_diff += std::numbers::pi;
//          phi_diff += std::numbers::pi;
        }

        // In : phi_diff in [0 .. pi ]
        // In : phi_diff in [0 .. pi ]
        // Out: tmp      in [0 .. 179]
        // Out: tmp      in [0 .. 179]
        int tmp  = int(phi_diff / std::numbers::pi * BRDF_SAMPLING_RES_PHI_D / 2);
//      int tmp  = int(phi_diff / std::numbers::pi * BRDF_SAMPLING_RES_PHI_D / 2);
        if (tmp  < 0)
//      if (tmp  < 0)
        {
            return 0;
//          return 0;
        }
        else
        if (tmp  < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
//      if (tmp  < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
        {
            return tmp;
//          return tmp;
        }
        else
        {
            return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
//          return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
        }
    }


    // Given a pair of incoming/outgoing angles, look up the BRDF.
    // Given a pair of incoming/outgoing angles, look up the BRDF.
    inline static void lookup_brdf_val(double* brdf, double theta_in, double fi_in, double theta_out, double fi_out, double& r, double& g, double& b)
//  inline static void lookup_brdf_val(double* brdf, double theta_in, double fi_in, double theta_out, double fi_out, double& r, double& g, double& b)
    {
        // Convert to halfangle / difference angle coordinates
        // Convert to halfangle / difference angle coordinates
        double theta_half, fi_half, theta_diff, fi_diff;
//      double theta_half, fi_half, theta_diff, fi_diff;


        std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out, theta_half, fi_half, theta_diff, fi_diff);
//      std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out, theta_half, fi_half, theta_diff, fi_diff);


        // Find index.
        // Find index.
        // Note that phi_half is ignored, since isotropic BRDFs are assumed
        // Note that phi_half is ignored, since isotropic BRDFs are assumed
        int ind = phi_diff_index(fi_diff) + theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 + theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D;
//      int ind = phi_diff_index(fi_diff) + theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 + theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D;


        r = brdf[ind                                                                                      ] * R_SCALE;
//      r = brdf[ind                                                                                      ] * R_SCALE;
        g = brdf[ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2] * G_SCALE;
//      g = brdf[ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2] * G_SCALE;
        b = brdf[ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D    ] * B_SCALE;
//      b = brdf[ind + BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D    ] * B_SCALE;


        if (r < 0.0
        ||  g < 0.0
        ||  b < 0.0)
        {
            fprintf(stderr, "Below horizon.\n");
//          fprintf(stderr, "Below horizon.\n");
        }


        r *= 10.0;
        g *= 10.0;
        b *= 10.0;
    }


    // Read BRDF data
    // Read BRDF data
    inline static bool read_brdf(const char* filename, double*& brdf)
//  inline static bool read_brdf(const char* filename, double*& brdf)
    {
        FILE* f = fopen(filename, "rb");
//      FILE* f = fopen(filename, "rb");
        if  (!f)
//      if  (!f)
        {
            return false;
//          return false;
        }

        int dims[3];
//      int dims[3];
        fread(dims, sizeof(int), 3, f);
//      fread(dims, sizeof(int), 3, f);
        int n = dims[0] * dims[1] * dims[2];
//      int n = dims[0] * dims[1] * dims[2];
        if (n != BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2)
//      if (n != BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2)
        {
            fprintf(stderr, "Dimensions don't match\n");
//          fprintf(stderr, "Dimensions don't match\n");
            fclose(f);
//          fclose(f);
            return false;
//          return false;
        }

        brdf = (double*)malloc(sizeof(double) * 3 * n);
//      brdf = (double*)malloc(sizeof(double) * 3 * n);
        fread(brdf, sizeof(double), 3 * static_cast<std::size_t>(n), f);
//      fread(brdf, sizeof(double), 3 * static_cast<std::size_t>(n), f);

        fclose(f);
//      fclose(f);
        return true;
//      return true;
    }


    inline static void test(const char* filename)
//  inline static void test(const char* filename)
    {
        double* brdf;
//      double* brdf;


        // read brdf
        // read brdf
        if (!read_brdf(filename, brdf))
//      if (!read_brdf(filename, brdf))
        {
            fprintf(stderr, "Error reading %s\n", filename);
//          fprintf(stderr, "Error reading %s\n", filename);
            exit(1);
//          exit(1);
        }


        // print out a 16x64x16x64 table table of BRDF values
        // print out a 16x64x16x64 table table of BRDF values
        const int n = 16;
//      const int n = 16;
        for (int i = 0; i < n; i++)
//      for (int i = 0; i < n; i++)
        {
            double theta_in = i * 0.5 * std::numbers::pi / n;
//          double theta_in = i * 0.5 * std::numbers::pi / n;
            for (int j = 0; j < 4 * n; j++)
//          for (int j = 0; j < 4 * n; j++)
            {
                double phi_in = j * 2.0 * std::numbers::pi / (4 * n);
//              double phi_in = j * 2.0 * std::numbers::pi / (4 * n);
                for (int k = 0; k < n; k++)
//              for (int k = 0; k < n; k++)
                {
                    double theta_out = k * 0.5 * std::numbers::pi / n;
//                  double theta_out = k * 0.5 * std::numbers::pi / n;
                    for (int l = 0; l < 4 * n; l++)
//                  for (int l = 0; l < 4 * n; l++)
                    {
                        double phi_out = l * 2.0 * std::numbers::pi / (4 * n);
//                      double phi_out = l * 2.0 * std::numbers::pi / (4 * n);
                        double r, g, b;
//                      double r, g, b;
                        lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out, r, g, b);
//                      lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out, r, g, b);
                        printf("%f %f %f\n", (float)r, (float)g, (float)b);
//                      printf("%f %f %f\n", (float)r, (float)g, (float)b);
                    }
                }
            }
        }
    }
}


#endif
