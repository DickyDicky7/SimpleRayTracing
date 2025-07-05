#pragma once
#ifndef SDF_H
#define SDF_H


    #include "Vec2.h"
//  #include "Vec2.h"
    #include "Vec3.h"
//  #include "Vec3.h"
    #include <concepts>
//  #include <concepts>


    namespace sdf3d
//  namespace sdf3d
{
        template<typename T> concept Number = std::integral<T> || std::floating_point<T>;
//      template<typename T> concept Number = std::integral<T> || std::floating_point<T>;
        template<Number T> static inline T Sign(T x) { return static_cast<T>((x > 0) - (x < 0)); }
//      template<Number T> static inline T Sign(T x) { return static_cast<T>((x > 0) - (x < 0)); }
        static inline float Dot2(Vec2 v) { return Dot(v, v); }
//      static inline float Dot2(Vec2 v) { return Dot(v, v); }
        static inline float Dot2(Vec3 v) { return Dot(v, v); }
//      static inline float Dot2(Vec3 v) { return Dot(v, v); }
        static inline float NDot(Vec2 a, Vec2 b) { return a.x * b.x - a.y * b.y; }
//      static inline float NDot(Vec2 a, Vec2 b) { return a.x * b.x - a.y * b.y; }



        // Calculates the signed distance from a point to the surface of a sphere.
        // Calculates the signed distance from a point to the surface of a sphere.
        static inline float SDFSphere(const Point3& samplePoint, float radius)
//      static inline float SDFSphere(const Point3& samplePoint, float radius)
        {
            // The distance from the origin (center of the sphere) minus the radius.
            // The distance from the origin (center of the sphere) minus the radius.
            return Length(samplePoint) - radius;
//          return Length(samplePoint) - radius;
        }



        // Calculates the signed distance from a point to the surface of an axis-aligned box.
        // Calculates the signed distance from a point to the surface of an axis-aligned box.
        static inline float SDFBox(const Point3& samplePoint, const Vec3& boxHalfSize)
//      static inline float SDFBox(const Point3& samplePoint, const Vec3& boxHalfSize)
        {
            // Fold the point into the first octant (all coordinates positive)
            // Fold the point into the first octant (all coordinates positive)
            // then calculate the vector from the box's surface to the point.
            // then calculate the vector from the box's surface to the point.
            Vec3 offsetFromBox = Abs(samplePoint) - boxHalfSize;
//          Vec3 offsetFromBox = Abs(samplePoint) - boxHalfSize;

            // The distance is composed of two parts:
            // The distance is composed of two parts:
            // 1. The length of the positive components of the offset (distance from the corner/edge/face).
            // 1. The length of the positive components of the offset (distance from the corner/edge/face).
            // 2. A correction term for points inside the box, which is negative.
            // 2. A correction term for points inside the box, which is negative.
            float outsideDistance = Length(Max(offsetFromBox, 0.0f));
//          float outsideDistance = Length(Max(offsetFromBox, 0.0f));
            float insideDistance = std::fminf(std::fmaxf(offsetFromBox.x, std::fmaxf(offsetFromBox.y, offsetFromBox.z)), 0.0f);
//          float insideDistance = std::fminf(std::fmaxf(offsetFromBox.x, std::fmaxf(offsetFromBox.y, offsetFromBox.z)), 0.0f);

            return outsideDistance + insideDistance;
//          return outsideDistance + insideDistance;
        }



        // Calculates the signed distance from a point to a box with rounded corners.
        // Calculates the signed distance from a point to a box with rounded corners.
        static inline float SDFRoundBox(const Point3& samplePoint, const Vec3& boxHalfSize, float cornerRadius)
//      static inline float SDFRoundBox(const Point3& samplePoint, const Vec3& boxHalfSize, float cornerRadius)
        {
            // By subtracting the halfSize and then adding the cornerRadius,
            // By subtracting the halfSize and then adding the cornerRadius,
            // we are calculating the offset from an inner, shrunken box.
            // we are calculating the offset from an inner, shrunken box.
            Vec3 offsetFromInnerBox = Abs(samplePoint) - boxHalfSize + cornerRadius;
//          Vec3 offsetFromInnerBox = Abs(samplePoint) - boxHalfSize + cornerRadius;

            // Calculate the distance to the shrunken box, then subtract the
            // Calculate the distance to the shrunken box, then subtract the
            // corner radius to effectively "round" the corners.
            // corner radius to effectively "round" the corners.
            float outsideDistance = Length(Max(offsetFromInnerBox, 0.0f));
//          float outsideDistance = Length(Max(offsetFromInnerBox, 0.0f));
            float insideDistance = std::fminf(std::fmaxf(offsetFromInnerBox.x, std::fmaxf(offsetFromInnerBox.y, offsetFromInnerBox.z)), 0.0f);
//          float insideDistance = std::fminf(std::fmaxf(offsetFromInnerBox.x, std::fmaxf(offsetFromInnerBox.y, offsetFromInnerBox.z)), 0.0f);

            return outsideDistance + insideDistance - cornerRadius;
//          return outsideDistance + insideDistance - cornerRadius;
        }



        // Calculates the signed distance from a point to the frame of a box.
        // Calculates the signed distance from a point to the frame of a box.
        static inline float SDFBoxFrame(const Point3& samplePoint, const Vec3& outerHalfSize, float frameThickness)
//      static inline float SDFBoxFrame(const Point3& samplePoint, const Vec3& outerHalfSize, float frameThickness)
        {
            // Fold the point into the first octant and find its position relative to the outer box.
            // Fold the point into the first octant and find its position relative to the outer box.
            Vec3 offsetFromOuterBox = Abs(samplePoint) - outerHalfSize;
//          Vec3 offsetFromOuterBox = Abs(samplePoint) - outerHalfSize;

            // This calculation effectively creates an inner "rounded" boundary.
            // This calculation effectively creates an inner "rounded" boundary.
            Vec3 innerOffset = Abs(offsetFromOuterBox + frameThickness) - frameThickness;
//          Vec3 innerOffset = Abs(offsetFromOuterBox + frameThickness) - frameThickness;

            // The frame is constructed by finding the distance to three "slabs"
            // The frame is constructed by finding the distance to three "slabs"
            // (box shapes infinite in one dimension) and taking the minimum.
            // (box shapes infinite in one dimension) and taking the minimum.
            // This is equivalent to the union of these three shapes.
            // This is equivalent to the union of these three shapes.
            // Each part is an SDFBox calculation, but with mixed components from the outer and inner offsets.
            // Each part is an SDFBox calculation, but with mixed components from the outer and inner offsets.

            // Slab 1: Using outer offset for x, inner for y and z.
            // Slab 1: Using outer offset for x, inner for y and z.
            float distToSlab1 = Length(Max(Vec3{ offsetFromOuterBox.x, innerOffset.y, innerOffset.z }, 0.0f)) + std::fminf(std::fmaxf(offsetFromOuterBox.x, std::fmaxf(innerOffset.y, innerOffset.z)), 0.0f);
//          float distToSlab1 = Length(Max(Vec3{ offsetFromOuterBox.x, innerOffset.y, innerOffset.z }, 0.0f)) + std::fminf(std::fmaxf(offsetFromOuterBox.x, std::fmaxf(innerOffset.y, innerOffset.z)), 0.0f);

            // Slab 2: Using outer offset for y, inner for x and z.
            // Slab 2: Using outer offset for y, inner for x and z.
            float distToSlab2 = Length(Max(Vec3{ innerOffset.x, offsetFromOuterBox.y, innerOffset.z }, 0.0f)) + std::fminf(std::fmaxf(innerOffset.x, std::fmaxf(offsetFromOuterBox.y, innerOffset.z)), 0.0f);
//          float distToSlab2 = Length(Max(Vec3{ innerOffset.x, offsetFromOuterBox.y, innerOffset.z }, 0.0f)) + std::fminf(std::fmaxf(innerOffset.x, std::fmaxf(offsetFromOuterBox.y, innerOffset.z)), 0.0f);

            // Slab 3: Using outer offset for z, inner for x and y.
            // Slab 3: Using outer offset for z, inner for x and y.
            float distToSlab3 = Length(Max(Vec3{ innerOffset.x, innerOffset.y, offsetFromOuterBox.z }, 0.0f)) + std::fminf(std::fmaxf(innerOffset.x, std::fmaxf(innerOffset.y, offsetFromOuterBox.z)), 0.0f);
//          float distToSlab3 = Length(Max(Vec3{ innerOffset.x, innerOffset.y, offsetFromOuterBox.z }, 0.0f)) + std::fminf(std::fmaxf(innerOffset.x, std::fmaxf(innerOffset.y, offsetFromOuterBox.z)), 0.0f);

            return std::fminf(std::fminf(distToSlab1, distToSlab2), distToSlab3);
//          return std::fminf(std::fminf(distToSlab1, distToSlab2), distToSlab3);
        }



        // Calculates the signed distance from a point to a torus.
        // Calculates the signed distance from a point to a torus.
        static inline float SDFTorus(const Point3& samplePoint, const Vec2& torusRadii)
//      static inline float SDFTorus(const Point3& samplePoint, const Vec2& torusRadii)
        {
            // torusRadii.x is the major radius (distance from center to the tube's center)
            // torusRadii.x is the major radius (distance from center to the tube's center)
            // torusRadii.y is the minor radius (the radius of the tube itself)
            // torusRadii.y is the minor radius (the radius of the tube itself)

            // Project the 3D sample point onto a 2D plane representing the torus cross-section.
            // Project the 3D sample point onto a 2D plane representing the torus cross-section.
            // The new x-component is the distance from the center of the tube.
            // The new x-component is the distance from the center of the tube.
            // The new y-component is the original height.
            // The new y-component is the original height.
            Vec2 pointInCrossSection = Vec2{ Length(Vec2{samplePoint.x, samplePoint.z}) - torusRadii.x, samplePoint.y };
//          Vec2 pointInCrossSection = Vec2{ Length(Vec2{samplePoint.x, samplePoint.z}) - torusRadii.x, samplePoint.y };

            // The final distance is the SDF of a 2D circle in that cross-section plane.
            // The final distance is the SDF of a 2D circle in that cross-section plane.
            return Length(pointInCrossSection) - torusRadii.y;
//          return Length(pointInCrossSection) - torusRadii.y;
        }



        // Calculates the signed distance to a torus capped with a cone/sphere section.
        // Calculates the signed distance to a torus capped with a cone/sphere section.
        static inline float SDFCappedTorus(const Point3& samplePoint, const Vec2& capNormal, float majorRadius, float tubeRadius)
//      static inline float SDFCappedTorus(const Point3& samplePoint, const Vec2& capNormal, float majorRadius, float tubeRadius)
        {
            // Fold the space so we only need to consider the positive x-quadrant.
            // Fold the space so we only need to consider the positive x-quadrant.
            Point3 foldedPoint = samplePoint;
//          Point3 foldedPoint = samplePoint;
            foldedPoint.x = abs(foldedPoint.x);
//          foldedPoint.x = abs(foldedPoint.x);

            float projectionDistance;
//          float projectionDistance;

            // The cap is a combination of a spherical part and a conical part.
            // The cap is a combination of a spherical part and a conical part.
            // This logic determines which part of the shape the sample point is closest to
            // This logic determines which part of the shape the sample point is closest to
            // by comparing its position relative to the line defined by the cap normal.
            // by comparing its position relative to the line defined by the cap normal.
            if (capNormal.y * foldedPoint.x > capNormal.x * foldedPoint.y)
//          if (capNormal.y * foldedPoint.x > capNormal.x * foldedPoint.y)
            {
                // If the point is in the "cone" region, the effective distance to the
                // If the point is in the "cone" region, the effective distance to the
                // torus centerline is the dot product with the cap normal.
                // torus centerline is the dot product with the cap normal.
                projectionDistance = Dot(Vec2{ foldedPoint.x, foldedPoint.y }, capNormal);
//              projectionDistance = Dot(Vec2{ foldedPoint.x, foldedPoint.y }, capNormal);
            }
            else
            {
                // Otherwise, it's in the "sphere" region, and we use the 2D radial distance.
                // Otherwise, it's in the "sphere" region, and we use the 2D radial distance.
                projectionDistance = Length(Vec2{ foldedPoint.x, foldedPoint.y });
//              projectionDistance = Length(Vec2{ foldedPoint.x, foldedPoint.y });
            }

            // This is the core formula for the distance to the surface of the capped torus's skeleton.
            // This is the core formula for the distance to the surface of the capped torus's skeleton.
            float skeletonDistance = std::sqrtf(Dot(foldedPoint, foldedPoint) + majorRadius * majorRadius - 2.0f * majorRadius * projectionDistance);
//          float skeletonDistance = std::sqrtf(Dot(foldedPoint, foldedPoint) + majorRadius * majorRadius - 2.0f * majorRadius * projectionDistance);

            // Subtract the tube's radius to get the final signed distance.
            // Subtract the tube's radius to get the final signed distance.
            return skeletonDistance - tubeRadius;
//          return skeletonDistance - tubeRadius;
        }



        // Calculates the signed distance to a link shape (two toroid halves connected by cylinders).
        // Calculates the signed distance to a link shape (two toroid halves connected by cylinders).
        static inline float SDFLink(const Point3& samplePoint, float linkHalfLength, float toroidMajorRadius, float tubeRadius)
//      static inline float SDFLink(const Point3& samplePoint, float linkHalfLength, float toroidMajorRadius, float tubeRadius)
        {
            // This transformation effectively collapses the straight central part of the link
            // This transformation effectively collapses the straight central part of the link
            // onto the x-z plane.
            // onto the x-z plane.
            // If the point is in the straight section (|y| < linkHalfLength), its y is treated as 0.
            // If the point is in the straight section (|y| < linkHalfLength), its y is treated as 0.
            // If it's in one of the toroid ends, its y is the distance from the straight section.
            // If it's in one of the toroid ends, its y is the distance from the straight section.
            Point3 clampedPoint { samplePoint.x, std::fmaxf(abs(samplePoint.y) - linkHalfLength, 0.0f), samplePoint.z };
//          Point3 clampedPoint { samplePoint.x, std::fmaxf(abs(samplePoint.y) - linkHalfLength, 0.0f), samplePoint.z };

            // Now, calculate the distance as if it's a simple torus, using the clamped point.
            // Now, calculate the distance as if it's a simple torus, using the clamped point.
            // This is the SDF for a torus centered at the origin, but evaluated in a warped space.
            // This is the SDF for a torus centered at the origin, but evaluated in a warped space.
            Vec2 pointInCrossSection = Vec2(Length(Vec2{ clampedPoint.x, clampedPoint.y }) - toroidMajorRadius, clampedPoint.z);
//          Vec2 pointInCrossSection = Vec2(Length(Vec2{ clampedPoint.x, clampedPoint.y }) - toroidMajorRadius, clampedPoint.z);

            return Length(pointInCrossSection) - tubeRadius;
//          return Length(pointInCrossSection) - tubeRadius;
        }



        // Note: Many of these functions operate in a reduced or "folded" space. For example, by taking Abs(p), we only need to solve the problem in the first octant/quadrant, which simplifies the geometry. Other transformations project the 3D problem into a 2D plane for easier calculation.
        // Note: Many of these functions operate in a reduced or "folded" space. For example, by taking Abs(p), we only need to solve the problem in the first octant/quadrant, which simplifies the geometry. Other transformations project the 3D problem into a 2D plane for easier calculation.



        // Calculates the signed distance to an infinite cylinder aligned with the Y-axis.
        // Calculates the signed distance to an infinite cylinder aligned with the Y-axis.
        static inline float SDFCylinder(const Point3& samplePoint, const Vec3& centerXYAndRadius)
//      static inline float SDFCylinder(const Point3& samplePoint, const Vec3& centerXYAndRadius)
        {
            // Projects the point onto the XZ plane and finds the distance to the cylinder's 2D center, then subtracts the radius. centerXY_and_radius.xy is the center, .z is the radius.
            // Projects the point onto the XZ plane and finds the distance to the cylinder's 2D center, then subtracts the radius. centerXY_and_radius.xy is the center, .z is the radius.
            float  distanceToCenter = Length(samplePoint.Swizzle<'x', 'z'>() - centerXYAndRadius.Swizzle<'x', 'y'>());
//          float  distanceToCenter = Length(samplePoint.Swizzle<'x', 'z'>() - centerXYAndRadius.Swizzle<'x', 'y'>());
            return distanceToCenter -                                          centerXYAndRadius.z                   ;
//          return distanceToCenter -                                          centerXYAndRadius.z                   ;
        }



        // Calculates the exact signed distance to a finite cone aligned with the Y-axis.
        // Calculates the exact signed distance to a finite cone aligned with the Y-axis.
        static inline float SDFConeExact(const Point3& samplePoint, const Vec2& coneAngleSinCos, float height)
        {
            // coneAngleSinCos contains the sine and cosine of the cone's half-angle.
            // coneAngleSinCos contains the sine and cosine of the cone's half-angle.

            // A 2D point on the cone's slope at the base.
            // A 2D point on the cone's slope at the base.
            Vec2 baseSlopePoint = height * Vec2{ coneAngleSinCos.x / coneAngleSinCos.y, -1.0f };
//          Vec2 baseSlopePoint = height * Vec2{ coneAngleSinCos.x / coneAngleSinCos.y, -1.0f };

            // Project the 3D sample point into a 2D plane (radial distance vs. height).
            // Project the 3D sample point into a 2D plane (radial distance vs. height).
            Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            // Calculate vectors from the 2D point to the cone's outline (the tip-to-base line segment).
            // Calculate vectors from the 2D point to the cone's outline (the tip-to-base line segment).
            Vec2 vecToSlopeSegment = pointIn2D - baseSlopePoint * std::clamp(Dot(pointIn2D, baseSlopePoint) / Dot(baseSlopePoint, baseSlopePoint), 0.0f, 1.0f);
//          Vec2 vecToSlopeSegment = pointIn2D - baseSlopePoint * std::clamp(Dot(pointIn2D, baseSlopePoint) / Dot(baseSlopePoint, baseSlopePoint), 0.0f, 1.0f);
            Vec2 vecToBaseEdge = pointIn2D - baseSlopePoint * Vec2{ std::clamp(pointIn2D.x / baseSlopePoint.x, 0.0f, 1.0f), 1.0f };
//          Vec2 vecToBaseEdge = pointIn2D - baseSlopePoint * Vec2{ std::clamp(pointIn2D.x / baseSlopePoint.x, 0.0f, 1.0f), 1.0f };

            // Find the minimum squared distance to the cone's boundary in 2D.
            // Find the minimum squared distance to the cone's boundary in 2D.
            float minSquaredDistance = std::fminf(Dot(vecToSlopeSegment, vecToSlopeSegment), Dot(vecToBaseEdge, vecToBaseEdge));
//          float minSquaredDistance = std::fminf(Dot(vecToSlopeSegment, vecToSlopeSegment), Dot(vecToBaseEdge, vecToBaseEdge));

            // Determine if the point is inside or outside using a 2D cross-product-like test.
            // Determine if the point is inside or outside using a 2D cross-product-like test.
            float signDirection = std::fmaxf(Sign(baseSlopePoint.y) * (pointIn2D.x * baseSlopePoint.y - pointIn2D.y * baseSlopePoint.x), Sign(baseSlopePoint.y) * (pointIn2D.y - baseSlopePoint.y));
//          float signDirection = std::fmaxf(Sign(baseSlopePoint.y) * (pointIn2D.x * baseSlopePoint.y - pointIn2D.y * baseSlopePoint.x), Sign(baseSlopePoint.y) * (pointIn2D.y - baseSlopePoint.y));

            return std::sqrtf(minSquaredDistance) * Sign(signDirection);
//          return std::sqrtf(minSquaredDistance) * Sign(signDirection);
        }



        // Calculates a fast, approximate ("bounding") signed distance to a cone.
        // Calculates a fast, approximate ("bounding") signed distance to a cone.
        static inline float SDFConeBound(const Point3& samplePoint, const Vec2& coneNormal, float height)
//      static inline float SDFConeBound(const Point3& samplePoint, const Vec2& coneNormal, float height)
        {
            // This method uses the intersection of two planes (one for the slope, one for the cap)
            // This method uses the intersection of two planes (one for the slope, one for the cap)
            // as a cheap approximation of the cone's distance.
            // as a cheap approximation of the cone's distance.
            float radialDistance = Length(samplePoint.Swizzle<'x', 'z'>());
//          float radialDistance = Length(samplePoint.Swizzle<'x', 'z'>());

            // Dot product with coneNormal gives distance to the infinite cone slope.
            // Dot product with coneNormal gives distance to the infinite cone slope.
            float distanceToSlope = Dot(coneNormal.Swizzle<'x', 'y'>(), Vec2{ radialDistance, samplePoint.y });
//          float distanceToSlope = Dot(coneNormal.Swizzle<'x', 'y'>(), Vec2{ radialDistance, samplePoint.y });

            // Distance to the top capping plane.
            // Distance to the top capping plane.
            float distanceToCap = -height - samplePoint.y;
//          float distanceToCap = -height - samplePoint.y;

            return std::fmaxf(distanceToSlope, distanceToCap);
//          return std::fmaxf(distanceToSlope, distanceToCap);
        }



        // Calculates the exact signed distance to an infinite cone.
        // Calculates the exact signed distance to an infinite cone.
        static inline float SDFConeExactInfinite(const Point3& samplePoint, const Vec2& coneAngleSinCos)
//      static inline float SDFConeExactInfinite(const Point3& samplePoint, const Vec2& coneAngleSinCos)
        {
            // Project the 3D point into a 2D plane (radial distance vs. negative height).
            // Project the 3D point into a 2D plane (radial distance vs. negative height).
            Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), -samplePoint.y };
//          Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), -samplePoint.y };

            // Project pointIn2D onto the cone's slope direction vector.
            // Project pointIn2D onto the cone's slope direction vector.
            Vec2 projectedPoint = coneAngleSinCos * std::fmaxf(Dot(pointIn2D, coneAngleSinCos), 0.0f);
//          Vec2 projectedPoint = coneAngleSinCos * std::fmaxf(Dot(pointIn2D, coneAngleSinCos), 0.0f);
            float distance = Length(pointIn2D - projectedPoint);
//          float distance = Length(pointIn2D - projectedPoint);

            // Use a 2D cross-product to determine if the point is inside or outside the cone.
            // Use a 2D cross-product to determine if the point is inside or outside the cone.
            float sign;
//          float sign;
            if (pointIn2D.x * coneAngleSinCos.y - pointIn2D.y * coneAngleSinCos.x < 0.0f)
//          if (pointIn2D.x * coneAngleSinCos.y - pointIn2D.y * coneAngleSinCos.x < 0.0f)
            {
                sign = -1.0f;
//              sign = -1.0f;
            }
            else
            {
                sign = +1.0f;
//              sign = +1.0f;
            }

            return distance * sign;
//          return distance * sign;
        }



        // Calculates the signed distance to an infinite plane.
        // Calculates the signed distance to an infinite plane.
        static inline float SDFPlane(const Point3& samplePoint, const Vec3& planeNormal, float distanceFromOrigin)
//      static inline float SDFPlane(const Point3& samplePoint, const Vec3& planeNormal, float distanceFromOrigin)
        {
            // The plane normal must be a unit vector. Dot product gives the signed distance to the plane passing through the origin. The offset h shifts the plane along its normal.
            // The plane normal must be a unit vector. Dot product gives the signed distance to the plane passing through the origin. The offset h shifts the plane along its normal.
            return Dot(samplePoint, planeNormal) + distanceFromOrigin;
//          return Dot(samplePoint, planeNormal) + distanceFromOrigin;
        }



        // Calculates the signed distance to a hexagonal prism.
        // Calculates the signed distance to a hexagonal prism.
        static inline float SDFHexPrism(const Point3& samplePoint, const Vec2& hexRadiusAndHalfHeight)
//      static inline float SDFHexPrism(const Point3& samplePoint, const Vec2& hexRadiusAndHalfHeight)
        {
            // Constants for hexagonal grid calculations.
            // Constants for hexagonal grid calculations.
            // k.xy = (cos(30), sin(30)), k.z = tan(30)
            // k.xy = (cos(30), sin(30)), k.z = tan(30)
            const Vec3 hexConstants = { -0.8660254f, 0.5f, 0.57735f };
//          const Vec3 hexConstants = { -0.8660254f, 0.5f, 0.57735f };

            // Fold the point into the first octant.
            // Fold the point into the first octant.
            Vec3 foldedPoint = Abs(samplePoint);
//          Vec3 foldedPoint = Abs(samplePoint);

            // This projects the 2D point onto a line to fold the 120-degree hex sector into a 60-degree one.
            // This projects the 2D point onto a line to fold the 120-degree hex sector into a 60-degree one.
            Vec2 projection = 2.0f * std::fminf(Dot(hexConstants.Swizzle<'x', 'y'>(), foldedPoint.Swizzle<'x', 'y'>()), 0.0f) * hexConstants.Swizzle<'x', 'y'>();
//          Vec2 projection = 2.0f * std::fminf(Dot(hexConstants.Swizzle<'x', 'y'>(), foldedPoint.Swizzle<'x', 'y'>()), 0.0f) * hexConstants.Swizzle<'x', 'y'>();
            foldedPoint.x -= projection.x;
            foldedPoint.y -= projection.y;

            // Now calculate distance as if it's a box in this warped space.
            // Now calculate distance as if it's a box in this warped space.
            // d.x is the signed distance to the hexagon's edge.
            // d.x is the signed distance to the hexagon's edge.
            // d.y is the signed distance to the @prism@'s cap@.
            // d.y is the signed distance to the @prism@'s cap@.
            Vec2 distances{ Length(foldedPoint.Swizzle<'x','y'>() - Vec2{std::clamp(foldedPoint.x, -hexConstants.z * hexRadiusAndHalfHeight.x, hexConstants.z * hexRadiusAndHalfHeight.x), hexRadiusAndHalfHeight.x}) * Sign(foldedPoint.y - hexRadiusAndHalfHeight.x), foldedPoint.z - hexRadiusAndHalfHeight.y };
//          Vec2 distances{ Length(foldedPoint.Swizzle<'x','y'>() - Vec2{std::clamp(foldedPoint.x, -hexConstants.z * hexRadiusAndHalfHeight.x, hexConstants.z * hexRadiusAndHalfHeight.x), hexRadiusAndHalfHeight.x}) * Sign(foldedPoint.y - hexRadiusAndHalfHeight.x), foldedPoint.z - hexRadiusAndHalfHeight.y };

            // Standard SDF for a 2D box (using the two distances).
            // Standard SDF for a 2D box (using the two distances).
            return std::fminf(std::fmaxf(distances.x, distances.y), 0.0f) + Length(Max(distances, 0.0f));
//          return std::fminf(std::fmaxf(distances.x, distances.y), 0.0f) + Length(Max(distances, 0.0f));
        }



        // Calculates the signed distance to a triangular prism.
        // Calculates the signed distance to a triangular prism.
        static inline float SDFTriPrism(const Point3& samplePoint, const Vec2& triRadiusAndHalfHeight)
//      static inline float SDFTriPrism(const Point3& samplePoint, const Vec2& triRadiusAndHalfHeight)
        {
            // Fold point into the first octant.
            // Fold point into the first octant.
            Vec3 foldedPoint = Abs(samplePoint);
//          Vec3 foldedPoint = Abs(samplePoint);

            // The distance is the maximum of the distances to the three planes that bound the prism's 60-degree sector.
            // The distance is the maximum of the distances to the three planes that bound the prism's 60-degree sector.
            // 1. Distance to the top/bottom face.
            // 1. Distance to the top/bottom face.
            float distanceToCap = foldedPoint.z - triRadiusAndHalfHeight.y;
//          float distanceToCap = foldedPoint.z - triRadiusAndHalfHeight.y;

            // 2. Distances to the two side faces (after folding).
            // 2. Distances to the two side faces (after folding).
            // The numbers 0.866... and 0.5 are sin/cos of 60 degrees.
            // The numbers 0.866... and 0.5 are sin/cos of 60 degrees.
            float distanceToSideFaces = std::fmaxf(foldedPoint.x * 0.866025f + samplePoint.y * 0.5f, -samplePoint.y) - triRadiusAndHalfHeight.x * 0.5f;
//          float distanceToSideFaces = std::fmaxf(foldedPoint.x * 0.866025f + samplePoint.y * 0.5f, -samplePoint.y) - triRadiusAndHalfHeight.x * 0.5f;

            return std::fmaxf(distanceToCap, distanceToSideFaces);
//          return std::fmaxf(distanceToCap, distanceToSideFaces);
        }



        // Calculates the signed distance to a capsule defined by a line segment and radius.
        // Calculates the signed distance to a capsule defined by a line segment and radius.
        static inline float SDFCapsule(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float radius)
//      static inline float SDFCapsule(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float radius)
        {
            Vec3 vecToStart =  samplePoint - segmentStart;
//          Vec3 vecToStart =  samplePoint - segmentStart;
            Vec3 vecToCease = segmentCease - segmentStart;
//          Vec3 vecToCease = segmentCease - segmentStart;

            // Project the sample point onto the capsule's line segment and clamp it between the end points.
            // Project the sample point onto the capsule's line segment and clamp it between the end points.
            float projectionFactor = std::clamp(Dot(vecToStart, vecToCease) / Dot(vecToCease, vecToCease), 0.0f, 1.0f);
//          float projectionFactor = std::clamp(Dot(vecToStart, vecToCease) / Dot(vecToCease, vecToCease), 0.0f, 1.0f);

            // Find the closest point on the segment to the sample point.
            // Find the closest point on the segment to the sample point.
            Vec3 closestPointOnSegment = segmentStart + vecToCease * projectionFactor;
//          Vec3 closestPointOnSegment = segmentStart + vecToCease * projectionFactor;

            // The distance is the distance to this closest point, minus the capsule's radius.
            // The distance is the distance to this closest point, minus the capsule's radius.
            return Length(samplePoint - closestPointOnSegment) - radius;
//          return Length(samplePoint - closestPointOnSegment) - radius;
        }



        // Calculates the signed distance to a capsule standing vertically on the Y-axis.
        // Calculates the signed distance to a capsule standing vertically on the Y-axis.
        static inline float SDFVerticalCapsule(const Point3& samplePoint, float height, float radius)
//      static inline float SDFVerticalCapsule(const Point3& samplePoint, float height, float radius)
        {
            Vec3 offsetPoint = samplePoint;
//          Vec3 offsetPoint = samplePoint;

            // Find the closest y-coordinate on the capsule's core segment [0, h].
            // Find the closest y-coordinate on the capsule's core segment [0, h].
            offsetPoint.y -= std::clamp(offsetPoint.y, 0.0f, height);
//          offsetPoint.y -= std::clamp(offsetPoint.y, 0.0f, height);

            // The distance is the length of the resulting vector minus the radius.
            // The distance is the length of the resulting vector minus the radius.
            return Length(offsetPoint) - radius;
//          return Length(offsetPoint) - radius;
        }



        // Calculates the signed distance to a cylinder with flat caps, aligned with the Y-axis.
        // Calculates the signed distance to a cylinder with flat caps, aligned with the Y-axis.
        static inline float SDFVerticalCappedCylinder(const Point3& samplePoint, float halfHeight, float radius)
//      static inline float SDFVerticalCappedCylinder(const Point3& samplePoint, float halfHeight, float radius)
        {
            // Project the 3D point to a 2D space (radial distance vs. height).
            // Project the 3D point to a 2D space (radial distance vs. height).
            Vec2 pointIn2D = Abs(Vec2{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y });
//          Vec2 pointIn2D = Abs(Vec2{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y });

            // Calculate the offset from the boundary of a 2D rectangle.
            // Calculate the offset from the boundary of a 2D rectangle.
            Vec2 offsetFromBounds = pointIn2D - Vec2{ radius, halfHeight };
//          Vec2 offsetFromBounds = pointIn2D - Vec2{ radius, halfHeight };

            // Use the 2D box SDF formula to get the final distance.
            // Use the 2D box SDF formula to get the final distance.
            return std::fminf(std::fmaxf(offsetFromBounds.x, offsetFromBounds.y), 0.0f) + Length(Max(offsetFromBounds, 0.0f));
//          return std::fminf(std::fmaxf(offsetFromBounds.x, offsetFromBounds.y), 0.0f) + Length(Max(offsetFromBounds, 0.0f));
        }



        // Calculates the signed distance to a capped cylinder between two arbitrary points.
        // Calculates the signed distance to a capped cylinder between two arbitrary points.
        static inline float SDFArbitraryCappedCylinder(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float radius)
//      static inline float SDFArbitraryCappedCylinder(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float radius)
        {
            Vec3 segmentVec = segmentCease - segmentStart; Vec3 vecToStart = samplePoint - segmentStart;
//          Vec3 segmentVec = segmentCease - segmentStart; Vec3 vecToStart = samplePoint - segmentStart;

            float segmentLenSq  = Dot(segmentVec, segmentVec);
//          float segmentLenSq  = Dot(segmentVec, segmentVec);
            float projectionDot = Dot(vecToStart, segmentVec);
//          float projectionDot = Dot(vecToStart, segmentVec);

            // `distToAxisSq `: squared distance from the sample point to the cylinder's infinite axis. It's derived from the rejection of vecToStart from segmentVec.
            // `distToAxisSq `: squared distance from the sample point to the cylinder's infinite axis. It's derived from the rejection of vecToStart from segmentVec.
            float distToAxisSq  = Dot(vecToStart, vecToStart) - projectionDot * projectionDot / segmentLenSq;
//          float distToAxisSq  = Dot(vecToStart, vecToStart) - projectionDot * projectionDot / segmentLenSq;

            // `distAlongAxis`: signed distance from the center of the segment along the axis.
            // `distAlongAxis`: signed distance from the center of the segment along the axis.
            float distAlongAxis = projectionDot - segmentLenSq * 0.5f;
//          float distAlongAxis = projectionDot - segmentLenSq * 0.5f;

            float d;
//          float d;
            // Condition checks if the closest point is on the cylindrical part or the spherical caps.
            // Condition checks if the closest point is on the cylindrical part or the spherical caps.
            if (std::fabs(distAlongAxis) < segmentLenSq * 0.5f && distToAxisSq < radius * radius)
//          if (std::fabs(distAlongAxis) < segmentLenSq * 0.5f && distToAxisSq < radius * radius)
            {
                // Point is inside the central cylinder part. We find the smaller distance to either the side wall or the cap.
                // Point is inside the central cylinder part. We find the smaller distance to either the side wall or the cap.
                float distToWall =              radius - std::sqrt(distToAxisSq );
//              float distToWall =              radius - std::sqrt(distToAxisSq );
                float distToCap  = segmentLenSq * 0.5f - std::fabs(distAlongAxis);
//              float distToCap  = segmentLenSq * 0.5f - std::fabs(distAlongAxis);
                d = -std::fminf(distToWall * distToWall, distToCap * distToCap);
//              d = -std::fminf(distToWall * distToWall, distToCap * distToCap);
            }
            else
            {
                // Point is outside, closer to the rounded caps. Calculate squared distance to the "corner" of the shape's 2D profile.
                // Point is outside, closer to the rounded caps. Calculate squared distance to the "corner" of the shape's 2D profile.
                float dx = std::sqrt(distToAxisSq ) - radius;
//              float dx = std::sqrt(distToAxisSq ) - radius;
                float dy = std::fabs(distAlongAxis) - segmentLenSq * 0.5f;
//              float dy = std::fabs(distAlongAxis) - segmentLenSq * 0.5f;

                float outsideDistSq = 0.0f;
//              float outsideDistSq = 0.0f;
                if (dx > 0.0f) { outsideDistSq += dx * dx; }
//              if (dx > 0.0f) { outsideDistSq += dx * dx; }
                if (dy > 0.0f) { outsideDistSq += dy * dy; }
//              if (dy > 0.0f) { outsideDistSq += dy * dy; }
                d = outsideDistSq;
//              d = outsideDistSq;
            }

            // The final distance must be scaled correctly.
            // The final distance must be scaled correctly.
            return Sign(d) * std::sqrtf(std::fabsf(d));
//          return Sign(d) * std::sqrtf(std::fabsf(d));
        }



        // Calculates the signed distance to a cylinder with rounded edges.
        // Calculates the signed distance to a cylinder with rounded edges.
        static inline float SDFRoundedCylinder(const Point3& samplePoint, float radius, float edgeRadius, float halfHeight)
//      static inline float SDFRoundedCylinder(const Point3& samplePoint, float radius, float edgeRadius, float halfHeight)
        {
            // Project to 2D (radial distance vs. height) and offset to an inner rectangle.
            // Project to 2D (radial distance vs. height) and offset to an inner rectangle.
            Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()) - radius + edgeRadius, std::fabsf(samplePoint.y) - halfHeight };
//          Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()) - radius + edgeRadius, std::fabsf(samplePoint.y) - halfHeight };

            // Use the 2D rounded box SDF formula.
            // Use the 2D rounded box SDF formula.
            return std::fminf(std::fmaxf(pointIn2D.x, pointIn2D.y), 0.0f) + Length(Max(pointIn2D, 0.0f)) - edgeRadius;
//          return std::fminf(std::fmaxf(pointIn2D.x, pointIn2D.y), 0.0f) + Length(Max(pointIn2D, 0.0f)) - edgeRadius;
        }



        // Calculates the signed distance to a capped cone (frustum).
        // Calculates the signed distance to a capped cone (frustum).
        static inline float SDFCappedCone1(const Point3& samplePoint, float height, float bottomRadius, float topRadius)
//      static inline float SDFCappedCone1(const Point3& samplePoint, float height, float bottomRadius, float topRadius)
        {
            Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            Vec2 slopeVec1{ topRadius               ,        height };
//          Vec2 slopeVec1{ topRadius               ,        height };
            Vec2 slopeVec2{ topRadius - bottomRadius, 2.0f * height };
//          Vec2 slopeVec2{ topRadius - bottomRadius, 2.0f * height };

            float activeRadius;
//          float activeRadius;
            if (pointIn2D.y < 0.0f)
//          if (pointIn2D.y < 0.0f)
            {
                activeRadius = bottomRadius;
//              activeRadius = bottomRadius;
            }
            else
            {
                activeRadius =    topRadius;
//              activeRadius =    topRadius;
            }
            Vec2 vecToCapps{ pointIn2D.x - std::fminf(pointIn2D.x, activeRadius), std::fabsf(pointIn2D.y) - height };
//          Vec2 vecToCapps{ pointIn2D.x - std::fminf(pointIn2D.x, activeRadius), std::fabsf(pointIn2D.y) - height };

            Vec2 vecToSlope = pointIn2D - slopeVec1 + slopeVec2 * std::clamp(Dot(slopeVec1 - pointIn2D, slopeVec2) / Dot2(slopeVec2), 0.0f, 1.0f);
//          Vec2 vecToSlope = pointIn2D - slopeVec1 + slopeVec2 * std::clamp(Dot(slopeVec1 - pointIn2D, slopeVec2) / Dot2(slopeVec2), 0.0f, 1.0f);

            float sign;
//          float sign;
            if (vecToSlope.x < 0.0f
            &&  vecToCapps.y < 0.0f)
            {
                sign = -1.0f;
//              sign = -1.0f;
            }
            else
            {
                sign = +1.0f;
//              sign = +1.0f;
            }

            return sign * std::sqrtf(std::fminf(Dot2(vecToCapps), Dot2(vecToSlope)));
//          return sign * std::sqrtf(std::fminf(Dot2(vecToCapps), Dot2(vecToSlope)));
        }



        // Calculates the signed distance to a capped cone between two arbitrary points.
        // Calculates the signed distance to a capped cone between two arbitrary points.
        static inline float SDFCappedCone2(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float startRadius, float ceaseRadius)
//      static inline float SDFCappedCone2(const Point3& samplePoint, const Vec3& segmentStart, const Vec3& segmentCease, float startRadius, float ceaseRadius)
        {
            float radiusDiff = ceaseRadius - startRadius;
//          float radiusDiff = ceaseRadius - startRadius;
            float      segmentLenSq = Dot(segmentCease - segmentStart, segmentCease - segmentStart);
//          float      segmentLenSq = Dot(segmentCease - segmentStart, segmentCease - segmentStart);
            float startToPointLenSq = Dot(samplePoint  - segmentStart, samplePoint  - segmentStart);
//          float startToPointLenSq = Dot(samplePoint  - segmentStart, samplePoint  - segmentStart);
            float projectionFactor  = Dot(samplePoint  - segmentStart, segmentCease - segmentStart) / segmentLenSq;
//          float projectionFactor  = Dot(samplePoint  - segmentStart, segmentCease - segmentStart) / segmentLenSq;

            float distToAxisSq = startToPointLenSq - projectionFactor * projectionFactor * segmentLenSq ;
//          float distToAxisSq = startToPointLenSq - projectionFactor * projectionFactor * segmentLenSq ;
            float distToAxis   =                                                std::sqrtf(distToAxisSq);
//          float distToAxis   =                                                std::sqrtf(distToAxisSq);

            float activeRadius;
//          float activeRadius;
            if (projectionFactor < 0.5f)
//          if (projectionFactor < 0.5f)
            {
                activeRadius = startRadius;
//              activeRadius = startRadius;
            }
            else
            {
                activeRadius = ceaseRadius;
//              activeRadius = ceaseRadius;
            }

            Vec2 vecToCapps{ std::fmaxf(0.0f, distToAxis - activeRadius), std::fabsf(projectionFactor - 0.5f) - 0.5f };
//          Vec2 vecToCapps{ std::fmaxf(0.0f, distToAxis - activeRadius), std::fabsf(projectionFactor - 0.5f) - 0.5f };

            float slopeFactor = radiusDiff * radiusDiff + segmentLenSq;
//          float slopeFactor = radiusDiff * radiusDiff + segmentLenSq;
            float projectionOnSlope = std::clamp((radiusDiff * (distToAxis - startRadius) + projectionFactor * segmentLenSq) / slopeFactor, 0.0f, 1.0f);
//          float projectionOnSlope = std::clamp((radiusDiff * (distToAxis - startRadius) + projectionFactor * segmentLenSq) / slopeFactor, 0.0f, 1.0f);

            Vec2 vecToSlope{ distToAxis - startRadius - projectionOnSlope * radiusDiff, projectionFactor - projectionOnSlope };
//          Vec2 vecToSlope{ distToAxis - startRadius - projectionOnSlope * radiusDiff, projectionFactor - projectionOnSlope };

            float sign;
//          float sign;
            if (vecToSlope.x < 0.0f
            &&  vecToCapps.y < 0.0f)
            {
                sign = -1.0f;
//              sign = -1.0f;
            }
            else
            {
                sign = +1.0f;
//              sign = +1.0f;
            }

            float distToCappsSq = vecToCapps.x * vecToCapps.x + vecToCapps.y * vecToCapps.y * segmentLenSq;
//          float distToCappsSq = vecToCapps.x * vecToCapps.x + vecToCapps.y * vecToCapps.y * segmentLenSq;
            float distToSlopeSq = vecToSlope.x * vecToSlope.x + vecToSlope.y * vecToSlope.y * segmentLenSq;
//          float distToSlopeSq = vecToSlope.x * vecToSlope.x + vecToSlope.y * vecToSlope.y * segmentLenSq;

            return sign * std::sqrtf(std::fminf(distToCappsSq, distToSlopeSq));
//          return sign * std::sqrtf(std::fminf(distToCappsSq, distToSlopeSq));
        }



        // Calculates the signed distance to a solid angle shape (sphere with a cone removed/added).
        // Calculates the signed distance to a solid angle shape (sphere with a cone removed/added).
        static inline float SDFSolidAngle(const Point3& samplePoint, const Vec2& angleSinCos, float sphereRadius)
//      static inline float SDFSolidAngle(const Point3& samplePoint, const Vec2& angleSinCos, float sphereRadius)
        {
            Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D { Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            // Distance to the sphere part.
            // Distance to the sphere part.
            float distanceToSphere = Length(pointIn2D) - sphereRadius;
//          float distanceToSphere = Length(pointIn2D) - sphereRadius;

            // Distance to the cone   part.
            // Distance to the cone   part.
            float distanceToCone   = Length(pointIn2D - angleSinCos * std::clamp(Dot(pointIn2D, angleSinCos), 0.0f, sphereRadius));
//          float distanceToCone   = Length(pointIn2D - angleSinCos * std::clamp(Dot(pointIn2D, angleSinCos), 0.0f, sphereRadius));

            // The sign of the 2D cross product determines if we are inside or outside the cone's angle.
            // The sign of the 2D cross product determines if we are inside or outside the cone's angle.
            return std::fmaxf(distanceToSphere, distanceToCone * Sign(angleSinCos.y * pointIn2D.x - angleSinCos.x * pointIn2D.y));
//          return std::fmaxf(distanceToSphere, distanceToCone * Sign(angleSinCos.y * pointIn2D.x - angleSinCos.x * pointIn2D.y));
        }



        // Calculates the signed distance to a sphere with its top cut off by a plane.
        // Calculates the signed distance to a sphere with its top cut off by a plane.
        static inline float SDFCutSphere(const Point3& samplePoint, float sphereRadius, float cutHeight)
//      static inline float SDFCutSphere(const Point3& samplePoint, float sphereRadius, float cutHeight)
        {
            // The radius of the circular cut.
            // The radius of the circular cut.
            float cutRadius = std::sqrtf(sphereRadius * sphereRadius - cutHeight * cutHeight);
//          float cutRadius = std::sqrtf(sphereRadius * sphereRadius - cutHeight * cutHeight);

            // Project to 2D (radial distance vs. height).
            // Project to 2D (radial distance vs. height).
            Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            // This check determines which feature is closest: the spherical part, the flat cap, or the edge.
            // This check determines which feature is closest: the spherical part, the flat cap, or the edge.
            float edgeCheck = (cutHeight - sphereRadius) * pointIn2D.x * pointIn2D.x + cutRadius * cutRadius * (cutHeight + sphereRadius - 2.0f * pointIn2D.y);
//          float edgeCheck = (cutHeight - sphereRadius) * pointIn2D.x * pointIn2D.x + cutRadius * cutRadius * (cutHeight + sphereRadius - 2.0f * pointIn2D.y);

            if (edgeCheck < 0.0f)
//          if (edgeCheck < 0.0f)
            {
                // Closest to the spherical surface.
                // Closest to the spherical surface.
                return Length(pointIn2D) - sphereRadius;
//              return Length(pointIn2D) - sphereRadius;
            }
            else
            {
                if (pointIn2D.x < cutRadius)
//              if (pointIn2D.x < cutRadius)
                {
                    // Closest to the flat cutting plane.
                    // Closest to the flat cutting plane.
                    return cutHeight - pointIn2D.y;
//                  return cutHeight - pointIn2D.y;
                }
                else
                {
                    // Closest to the circular edge.
                    // Closest to the circular edge.
                    return Length(pointIn2D - Vec2{ cutRadius, cutHeight });
//                  return Length(pointIn2D - Vec2{ cutRadius, cutHeight });
                }
            }
        }



        // Calculates the signed distance to a hollow sphere with its top cut off.
        // Calculates the signed distance to a hollow sphere with its top cut off.
        static inline float SDFCutHollowSphere(const Point3& samplePoint, float sphereRadius, float cutHeight, float thickness)
//      static inline float SDFCutHollowSphere(const Point3& samplePoint, float sphereRadius, float cutHeight, float thickness)
        {
            float cutRadius = std::sqrtf(sphereRadius * sphereRadius - cutHeight * cutHeight);
//          float cutRadius = std::sqrtf(sphereRadius * sphereRadius - cutHeight * cutHeight);
            Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            // This is a Voronoi test to see if the point is closer to the circular edge or the spherical surface.
            // This is a Voronoi test to see if the point is closer to the circular edge or the spherical surface.
            float distanceToSurface;
//          float distanceToSurface;
            if (cutHeight * pointIn2D.x < cutRadius * pointIn2D.y)
//          if (cutHeight * pointIn2D.x < cutRadius * pointIn2D.y)
            {
                // Closer to the cut edge.
                // Closer to the cut edge.
                distanceToSurface = Length(pointIn2D - Vec2{ cutRadius, cutHeight });
//              distanceToSurface = Length(pointIn2D - Vec2{ cutRadius, cutHeight });
            }
            else
            {
                // Closer to the main sphere surface.
                // Closer to the main sphere surface.
                distanceToSurface = std::fabsf(Length(pointIn2D) - sphereRadius);
//              distanceToSurface = std::fabsf(Length(pointIn2D) - sphereRadius);
            }

            return distanceToSurface - thickness;
//          return distanceToSurface - thickness;
        }



        // Calculates the signed distance to a "Death Star" shape (sphere with a spherical crater).
        // Calculates the signed distance to a "Death Star" shape (sphere with a spherical crater).
        static inline float SDFDeathStar(const Point3& samplePoint, float mainSphereRadius, float craterSphereRadius, float craterOffset)
//      static inline float SDFDeathStar(const Point3& samplePoint, float mainSphereRadius, float craterSphereRadius, float craterOffset)
        {
            // `a` and `b` define the circle of intersection between the two spheres.
            // `a` and `b` define the circle of intersection between the two spheres.
            float a =                      (mainSphereRadius * mainSphereRadius - craterSphereRadius * craterSphereRadius + craterOffset * craterOffset) / (2.0f * craterOffset);
//          float a =                      (mainSphereRadius * mainSphereRadius - craterSphereRadius * craterSphereRadius + craterOffset * craterOffset) / (2.0f * craterOffset);
            float b = std::sqrtf(std::fmaxf(mainSphereRadius * mainSphereRadius - a                  * a                                                 ,  0.0f));
//          float b = std::sqrtf(std::fmaxf(mainSphereRadius * mainSphereRadius - a                  * a                                                 ,  0.0f));

            // Reduce to a 2D problem.
            // Reduce to a 2D problem.
            Vec2 pointIn2D{ samplePoint.x, Length(samplePoint.Swizzle<'y', 'z'>()) };
//          Vec2 pointIn2D{ samplePoint.x, Length(samplePoint.Swizzle<'y', 'z'>()) };

            float distance;
//          float distance;
            // This condition checks if the point is closer to the intersection edge or one of the sphere surfaces.
            // This condition checks if the point is closer to the intersection edge or one of the sphere surfaces.
            if (pointIn2D.x * b - pointIn2D.y * a > craterOffset * std::fmaxf(b - pointIn2D.y, 0.0f))
//          if (pointIn2D.x * b - pointIn2D.y * a > craterOffset * std::fmaxf(b - pointIn2D.y, 0.0f))
            {
                // Closest to the intersection rim.
                // Closest to the intersection rim.
                distance = Length(pointIn2D - Vec2{ a, b });
//              distance = Length(pointIn2D - Vec2{ a, b });
            }
            else
            {
                // The shape is the intersection of the main sphere (distance > 0)
                // The shape is the intersection of the main sphere (distance > 0)
                // and the exterior of the crater sphere (distance > 0). The fmax achieves this.
                // and the exterior of the crater sphere (distance > 0). The fmax achieves this.
                float distanceToMainSphere     =   Length(pointIn2D)                              -   mainSphereRadius ;
//              float distanceToMainSphere     =   Length(pointIn2D)                              -   mainSphereRadius ;
                float distanceToCraterExterior = -(Length(pointIn2D - Vec2{ craterOffset, 0.0f }) - craterSphereRadius);
//              float distanceToCraterExterior = -(Length(pointIn2D - Vec2{ craterOffset, 0.0f }) - craterSphereRadius);
                distance = std::fmaxf(distanceToMainSphere, distanceToCraterExterior);
//              distance = std::fmaxf(distanceToMainSphere, distanceToCraterExterior);
            }

            return distance;
//          return distance;
        }



        // Calculates the signed distance to a cone with rounded caps.
        // Calculates the signed distance to a cone with rounded caps.
        static inline float SDFRoundCone1(const Point3& samplePoint, float bottomRadius, float topRadius, float height)
//      static inline float SDFRoundCone1(const Point3& samplePoint, float bottomRadius, float topRadius, float height)
        {
            // Parameters for the cone's slope.
            // Parameters for the cone's slope.
            float radiusDiff = bottomRadius - topRadius; float slopeParamB = radiusDiff / height; float slopeParamA = std::sqrtf(1.0f - slopeParamB * slopeParamB);
//          float radiusDiff = bottomRadius - topRadius; float slopeParamB = radiusDiff / height; float slopeParamA = std::sqrtf(1.0f - slopeParamB * slopeParamB);

            Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };
//          Vec2 pointIn2D{ Length(samplePoint.Swizzle<'x', 'z'>()), samplePoint.y };

            // Project the 2D point onto the normal of the cone's slope.
            // Project the 2D point onto the normal of the cone's slope.
            float projectionOnNormal = Dot(pointIn2D, Vec2{ -slopeParamB, slopeParamA });
//          float projectionOnNormal = Dot(pointIn2D, Vec2{ -slopeParamB, slopeParamA });

            if (projectionOnNormal < 0.0f)
//          if (projectionOnNormal < 0.0f)
            {
                // Closest to the bottom round cap.
                // Closest to the bottom round cap.
                return Length(pointIn2D) - bottomRadius;
//              return Length(pointIn2D) - bottomRadius;
            }
            else
            {
                if (projectionOnNormal > slopeParamA * height)
//              if (projectionOnNormal > slopeParamA * height)
                {
                    // Closest to the top round cap.
                    // Closest to the top round cap.
                    return Length(pointIn2D - Vec2{ 0.0f, height }) - topRadius;
//                  return Length(pointIn2D - Vec2{ 0.0f, height }) - topRadius;
                }
                else
                {
                    // Closest to the slanted body of the cone.
                    // Closest to the slanted body of the cone.
                    return Dot(pointIn2D, Vec2{ slopeParamA, slopeParamB }) - bottomRadius;
//                  return Dot(pointIn2D, Vec2{ slopeParamA, slopeParamB }) - bottomRadius;
                }
            }
        }



        // Calculates the signed distance to an arbitrarily-oriented cone with rounded caps.
        // Calculates the signed distance to an arbitrarily-oriented cone with rounded caps.
        static inline float SDFRoundCone2(const Point3& samplePoint, const Vec3& startPoint, const Vec3& ceasePoint, float startRadius, float ceaseRadius)
//      static inline float SDFRoundCone2(const Point3& samplePoint, const Vec3& startPoint, const Vec3& ceasePoint, float startRadius, float ceaseRadius)
        {
            Vec3 segmentVec = ceasePoint - startPoint;
//          Vec3 segmentVec = ceasePoint - startPoint;
            float segmentLenSq = Dot(segmentVec, segmentVec);
//          float segmentLenSq = Dot(segmentVec, segmentVec);
            float radiusDiff = startRadius - ceaseRadius;
//          float radiusDiff = startRadius - ceaseRadius;
            float slopeFactorSq = segmentLenSq - radiusDiff * radiusDiff;
//          float slopeFactorSq = segmentLenSq - radiusDiff * radiusDiff;
            float invSegmentLenSq = 1.0f / segmentLenSq;
//          float invSegmentLenSq = 1.0f / segmentLenSq;

            Vec3 vecToStart = samplePoint - startPoint;
//          Vec3 vecToStart = samplePoint - startPoint;
            float yProjection = Dot(vecToStart, segmentVec);
//          float yProjection = Dot(vecToStart, segmentVec);
            float zProjection = yProjection - segmentLenSq;
//          float zProjection = yProjection - segmentLenSq;
            float xSquared = Dot2(vecToStart * segmentLenSq - segmentVec * yProjection);
//          float xSquared = Dot2(vecToStart * segmentLenSq - segmentVec * yProjection);

            // This logic determines which feature (start cap, end cap, or sloped side) is closest.
            // This logic determines which feature (start cap, end cap, or sloped side) is closest.
            float k = Sign(radiusDiff) * radiusDiff * radiusDiff * xSquared;
//          float k = Sign(radiusDiff) * radiusDiff * radiusDiff * xSquared;
            if (Sign(zProjection) * slopeFactorSq * zProjection * zProjection > k) { return std::sqrtf(xSquared + zProjection * zProjection) * invSegmentLenSq - ceaseRadius; }
//          if (Sign(zProjection) * slopeFactorSq * zProjection * zProjection > k) { return std::sqrtf(xSquared + zProjection * zProjection) * invSegmentLenSq - ceaseRadius; }
            if (Sign(yProjection) * slopeFactorSq * yProjection * yProjection < k) { return std::sqrtf(xSquared + yProjection * yProjection) * invSegmentLenSq - startRadius; }
//          if (Sign(yProjection) * slopeFactorSq * yProjection * yProjection < k) { return std::sqrtf(xSquared + yProjection * yProjection) * invSegmentLenSq - startRadius; }
            return (std::sqrtf(xSquared * slopeFactorSq * invSegmentLenSq) + yProjection * radiusDiff) * invSegmentLenSq - startRadius;
//          return (std::sqrtf(xSquared * slopeFactorSq * invSegmentLenSq) + yProjection * radiusDiff) * invSegmentLenSq - startRadius;
        }



        // Calculates an approximate signed distance to an ellipsoid.
        // Calculates an approximate signed distance to an ellipsoid.
        static inline float SDFEllipsoid(const Point3& samplePoint, const Vec3& radii)
//      static inline float SDFEllipsoid(const Point3& samplePoint, const Vec3& radii)
        {
            // This is a known, good approximation for the ellipsoid distance.
            // This is a known, good approximation for the ellipsoid distance.
            float k0 = Length(samplePoint /  radii         );
//          float k0 = Length(samplePoint /  radii         );
            float k1 = Length(samplePoint / (radii * radii));
//          float k1 = Length(samplePoint / (radii * radii));
            return k0 * (k0 - 1.0f) / k1;
//          return k0 * (k0 - 1.0f) / k1;
        }



        // Calculates the signed distance for a vesica segment (intersection of two spheres).
        // Calculates the signed distance for a vesica segment (intersection of two spheres).
        static inline float SDFVesicaSegment(const Point3& samplePoint, const Vec3& center1, const Vec3& center2, float overlap)
//      static inline float SDFVesicaSegment(const Point3& samplePoint, const Vec3& center1, const Vec3& center2, float overlap)
        {
            Vec3  midpoint = (center1 + center2) * 0.5f;
//          Vec3  midpoint = (center1 + center2) * 0.5f;
            float distanceBetweenCenters = Length(center2 - center1);
//          float distanceBetweenCenters = Length(center2 - center1);
            Vec3  axisVect = (center2 - center1) / distanceBetweenCenters;
//          Vec3  axisVect = (center2 - center1) / distanceBetweenCenters;

            float yProjection = Dot(samplePoint - midpoint, axisVect);
//          float yProjection = Dot(samplePoint - midpoint, axisVect);
            Vec2  pointIn2D{ Length(samplePoint - midpoint - yProjection * axisVect), std::fabsf(yProjection) };
//          Vec2  pointIn2D{ Length(samplePoint - midpoint - yProjection * axisVect), std::fabsf(yProjection) };

            float radius = 0.5f * distanceBetweenCenters;
//          float radius = 0.5f * distanceBetweenCenters;
            float offset = 0.5f * (radius * radius - overlap * overlap) / overlap;
//          float offset = 0.5f * (radius * radius - overlap * overlap) / overlap;

            Vec3 helper;
//          Vec3 helper;
            if (radius * pointIn2D.x < offset * (pointIn2D.y - radius))
//          if (radius * pointIn2D.x < offset * (pointIn2D.y - radius))
            {
                helper = Vec3{ 0.0f, radius, 0.0f };
//              helper = Vec3{ 0.0f, radius, 0.0f };
            }
            else
//          else
            {
                helper = Vec3{ -offset, 0.0f, offset + overlap };
//              helper = Vec3{ -offset, 0.0f, offset + overlap };
            }

            return Length(pointIn2D - helper.Swizzle<'x', 'y'>()) - helper.z;
//          return Length(pointIn2D - helper.Swizzle<'x', 'y'>()) - helper.z;
        }



        // Calculates the signed distance to a rhombus with rounded edges.
        // Calculates the signed distance to a rhombus with rounded edges.
        static inline float SDFRhombus(const Point3& samplePoint, float halfDiagonalA, float halfDiagonalB, float halfHeight, float edgeRadius)
//      static inline float SDFRhombus(const Point3& samplePoint, float halfDiagonalA, float halfDiagonalB, float halfHeight, float edgeRadius)
        {
            Vec3 foldedPoint = Abs(samplePoint);
//          Vec3 foldedPoint = Abs(samplePoint);
            Vec2 diagonals{ halfDiagonalA, halfDiagonalB };
//          Vec2 diagonals{ halfDiagonalA, halfDiagonalB };

            // `projectionFactor` finds the closest point on the rhombus edge.
            // `projectionFactor` finds the closest point on the rhombus edge.
            float projectionFactor = std::clamp((Dot(diagonals, diagonals - 2.0f * foldedPoint.Swizzle<'x', 'z'>())) / Dot(diagonals, diagonals), -1.0f, 1.0f);
//          float projectionFactor = std::clamp((Dot(diagonals, diagonals - 2.0f * foldedPoint.Swizzle<'x', 'z'>())) / Dot(diagonals, diagonals), -1.0f, 1.0f);
            Vec2 projectedEdgePoint = 0.5f * diagonals * Vec2{ 1.0f - projectionFactor, 1.0f + projectionFactor };
//          Vec2 projectedEdgePoint = 0.5f * diagonals * Vec2{ 1.0f - projectionFactor, 1.0f + projectionFactor };

            // `offsetFromEdge` is the 2D vector used for the rounded box SDF.
            // `offsetFromEdge` is the 2D vector used for the rounded box SDF.
            Vec2 offsetFromEdge{ Length(foldedPoint.Swizzle<'x', 'z'>() - projectedEdgePoint) * Sign(foldedPoint.x * diagonals.y + foldedPoint.z * diagonals.x - diagonals.x * diagonals.y) - edgeRadius, foldedPoint.y - halfHeight };
//          Vec2 offsetFromEdge{ Length(foldedPoint.Swizzle<'x', 'z'>() - projectedEdgePoint) * Sign(foldedPoint.x * diagonals.y + foldedPoint.z * diagonals.x - diagonals.x * diagonals.y) - edgeRadius, foldedPoint.y - halfHeight };

            return std::fminf(std::fmaxf(offsetFromEdge.x, offsetFromEdge.y), 0.0f) + Length(Max(offsetFromEdge, 0.0f));
//          return std::fminf(std::fmaxf(offsetFromEdge.x, offsetFromEdge.y), 0.0f) + Length(Max(offsetFromEdge, 0.0f));
        }



        // Calculates the exact signed distance to an octahedron.
        // Calculates the exact signed distance to an octahedron.
        static inline float SDFOctahedronExact(const Point3& samplePoint, float size)
//      static inline float SDFOctahedronExact(const Point3& samplePoint, float size)
        {
            Vec3                  foldedPoint   =             Abs(samplePoint)        ;
//          Vec3                  foldedPoint   =             Abs(samplePoint)        ;
            float planeDistance = foldedPoint.x + foldedPoint.y + foldedPoint.z - size;
//          float planeDistance = foldedPoint.x + foldedPoint.y + foldedPoint.z - size;

            Vec3 q;
//          Vec3 q;
                 if (3.0f * foldedPoint.x < planeDistance) q = foldedPoint.Swizzle<'x', 'y', 'z'>();
//               if (3.0f * foldedPoint.x < planeDistance) q = foldedPoint.Swizzle<'x', 'y', 'z'>();
            else if (3.0f * foldedPoint.y < planeDistance) q = foldedPoint.Swizzle<'y', 'z', 'x'>();
//          else if (3.0f * foldedPoint.y < planeDistance) q = foldedPoint.Swizzle<'y', 'z', 'x'>();
            else if (3.0f * foldedPoint.z < planeDistance) q = foldedPoint.Swizzle<'z', 'x', 'y'>();
//          else if (3.0f * foldedPoint.z < planeDistance) q = foldedPoint.Swizzle<'z', 'x', 'y'>();
            else return planeDistance * 0.57735027f; // 1/sqrt(3)
//          else return planeDistance * 0.57735027f; // 1/sqrt(3)

            float k = std::clamp(0.5f * (q.z - q.y + size), 0.0f, size);
//          float k = std::clamp(0.5f * (q.z - q.y + size), 0.0f, size);
            return Length(Vec3{ q.x, q.y - size + k, q.z - k });
//          return Length(Vec3{ q.x, q.y - size + k, q.z - k });
        }



        // Calculates a fast, approximate ("bounding") signed distance to an octahedron.
        // Calculates a fast, approximate ("bounding") signed distance to an octahedron.
        static inline float SDFOctahedronBound(const Point3& samplePoint, float size)
//      static inline float SDFOctahedronBound(const Point3& samplePoint, float size)
        {
            Vec3 foldedPoint = Abs(samplePoint);
//          Vec3 foldedPoint = Abs(samplePoint);
            // This is the distance to the plane x+y+z=s, normalized by 1/sqrt(3).
            // This is the distance to the plane x+y+z=s, normalized by 1/sqrt(3).
            return (foldedPoint.x + foldedPoint.y + foldedPoint.z - size) * 0.57735027f;
//          return (foldedPoint.x + foldedPoint.y + foldedPoint.z - size) * 0.57735027f;
        }



        // Calculates the signed distance to a square-base pyramid.
        // Calculates the signed distance to a square-base pyramid.
        static inline float SDFPyramid(const Point3& samplePoint, float height)
//      static inline float SDFPyramid(const Point3& samplePoint, float height)
        {
            float slopeSquared = height * height + 0.25f;
//          float slopeSquared = height * height + 0.25f;

            // Fold space so we only consider one-quarter of the pyramid base.
            // Fold space so we only consider one-quarter of the pyramid base.
            Vec3 pointTemporarily = samplePoint;
//          Vec3 pointTemporarily = samplePoint;
            Vec2 pointInBasePlane = Abs(pointTemporarily.Swizzle<'x', 'z'>());
//          Vec2 pointInBasePlane = Abs(pointTemporarily.Swizzle<'x', 'z'>());
            if ( pointInBasePlane.y
               > pointInBasePlane.x)
            {
                std::swap(pointInBasePlane.x, pointInBasePlane.y);
//              std::swap(pointInBasePlane.x, pointInBasePlane.y);
            }
            pointInBasePlane -= 0.5f;
//          pointInBasePlane -= 0.5f;

            pointTemporarily.x = pointInBasePlane.x;
//          pointTemporarily.x = pointInBasePlane.x;
            pointTemporarily.z = pointInBasePlane.y;
//          pointTemporarily.z = pointInBasePlane.y;

            // Project into a 2D space related to the pyramid's sloped face.
            // Project into a 2D space related to the pyramid's sloped face.
            Vec3 pointInSlopePlane =
//          Vec3 pointInSlopePlane =
            {
                                                     pointTemporarily.z,
//                                                   pointTemporarily.z,
                height * pointTemporarily.y - 0.5f * pointTemporarily.x,
//              height * pointTemporarily.y - 0.5f * pointTemporarily.x,
                height * pointTemporarily.x + 0.5f * pointTemporarily.y,
//              height * pointTemporarily.x + 0.5f * pointTemporarily.y,
            };

            float s = std::fmaxf(-pointInSlopePlane.x , 0.0f);
//          float s = std::fmaxf(-pointInSlopePlane.x , 0.0f);
            float t = std::clamp((pointInSlopePlane.y - 0.5f * pointTemporarily.z) / (slopeSquared + 0.25f), 0.0f, 1.0f);
//          float t = std::clamp((pointInSlopePlane.y - 0.5f * pointTemporarily.z) / (slopeSquared + 0.25f), 0.0f, 1.0f);

            float distanceSquaredA = slopeSquared * (pointInSlopePlane.x +        s) * (pointInSlopePlane.x +        s) +  pointInSlopePlane.y *                      pointInSlopePlane.y                    ;
//          float distanceSquaredA = slopeSquared * (pointInSlopePlane.x +        s) * (pointInSlopePlane.x +        s) +  pointInSlopePlane.y *                      pointInSlopePlane.y                    ;
            float distanceSquaredB = slopeSquared * (pointInSlopePlane.x + 0.5f * t) * (pointInSlopePlane.x + 0.5f * t) + (pointInSlopePlane.y - slopeSquared * t) * (pointInSlopePlane.y - slopeSquared * t);
//          float distanceSquaredB = slopeSquared * (pointInSlopePlane.x + 0.5f * t) * (pointInSlopePlane.x + 0.5f * t) + (pointInSlopePlane.y - slopeSquared * t) * (pointInSlopePlane.y - slopeSquared * t);

            float minSquaredDistance;
//          float minSquaredDistance;
            if (pointInSlopePlane.y < 0.0f && -pointInSlopePlane.x * slopeSquared - pointInSlopePlane.y * 0.5f < 0.0f)
//          if (pointInSlopePlane.y < 0.0f && -pointInSlopePlane.x * slopeSquared - pointInSlopePlane.y * 0.5f < 0.0f)
            {
                minSquaredDistance = 0.0f;
//              minSquaredDistance = 0.0f;
            }
            else
            {
                minSquaredDistance = std::fminf(distanceSquaredA, distanceSquaredB);
//              minSquaredDistance = std::fminf(distanceSquaredA, distanceSquaredB);
            }

            return std::sqrtf((minSquaredDistance + pointInSlopePlane.z * pointInSlopePlane.z) / slopeSquared) * Sign(std::fmaxf(pointInSlopePlane.z, -pointTemporarily.y));
//          return std::sqrtf((minSquaredDistance + pointInSlopePlane.z * pointInSlopePlane.z) / slopeSquared) * Sign(std::fmaxf(pointInSlopePlane.z, -pointTemporarily.y));
        }



        // Calculates the signed distance to a triangle in 3D space.
        // Calculates the signed distance to a triangle in 3D space.
        static inline float SDFTriangle(const Point3& samplePoint, const Point3& pointA, const Point3& pointB, const Point3& pointC)
//      static inline float SDFTriangle(const Point3& samplePoint, const Point3& pointA, const Point3& pointB, const Point3& pointC)
        {
            Vec3 edgeAB = pointB - pointA; Vec3 pointAToSamplePoint = samplePoint - pointA;
//          Vec3 edgeAB = pointB - pointA; Vec3 pointAToSamplePoint = samplePoint - pointA;
            Vec3 edgeBC = pointC - pointB; Vec3 pointBToSamplePoint = samplePoint - pointB;
//          Vec3 edgeBC = pointC - pointB; Vec3 pointBToSamplePoint = samplePoint - pointB;
            Vec3 edgeCA = pointA - pointC; Vec3 pointCToSamplePoint = samplePoint - pointC;
//          Vec3 edgeCA = pointA - pointC; Vec3 pointCToSamplePoint = samplePoint - pointC;
            Vec3 normal = Cross(edgeAB, edgeCA);
//          Vec3 normal = Cross(edgeAB, edgeCA);

            // This check uses the sign of the dot product between the normal of an edge plane and the vector to the point. If the sum is less than 2, the point's projection is outside the triangle.
            // This check uses the sign of the dot product between the normal of an edge plane and the vector to the point. If the sum is less than 2, the point's projection is outside the triangle.
            bool isOutside =
//          bool isOutside =
                Sign(Dot(Cross(edgeAB, normal), pointAToSamplePoint)) +
//              Sign(Dot(Cross(edgeAB, normal), pointAToSamplePoint)) +
                Sign(Dot(Cross(edgeBC, normal), pointBToSamplePoint)) +
//              Sign(Dot(Cross(edgeBC, normal), pointBToSamplePoint)) +
                Sign(Dot(Cross(edgeCA, normal), pointCToSamplePoint)) < 2.0f;
//              Sign(Dot(Cross(edgeCA, normal), pointCToSamplePoint)) < 2.0f;

            float distanceSquared;
//          float distanceSquared;
            if (isOutside)
//          if (isOutside)
            {
                // If outside, the distance is the distance to the closest edge.
                // If outside, the distance is the distance to the closest edge.
                distanceSquared = std::fminf(std::fminf(
//              distanceSquared = std::fminf(std::fminf(
                    Dot2(edgeAB * std::clamp(Dot(edgeAB, pointAToSamplePoint) / Dot2(edgeAB), 0.0f, 1.0f) - pointAToSamplePoint) ,
//                  Dot2(edgeAB * std::clamp(Dot(edgeAB, pointAToSamplePoint) / Dot2(edgeAB), 0.0f, 1.0f) - pointAToSamplePoint) ,
                    Dot2(edgeBC * std::clamp(Dot(edgeBC, pointBToSamplePoint) / Dot2(edgeBC), 0.0f, 1.0f) - pointBToSamplePoint)),
//                  Dot2(edgeBC * std::clamp(Dot(edgeBC, pointBToSamplePoint) / Dot2(edgeBC), 0.0f, 1.0f) - pointBToSamplePoint)),
                    Dot2(edgeCA * std::clamp(Dot(edgeCA, pointCToSamplePoint) / Dot2(edgeCA), 0.0f, 1.0f) - pointCToSamplePoint));
//                  Dot2(edgeCA * std::clamp(Dot(edgeCA, pointCToSamplePoint) / Dot2(edgeCA), 0.0f, 1.0f) - pointCToSamplePoint));
            }
            else
//          else
            {
                // If inside, the distance is the distance to the triangle's plane.
                // If inside, the distance is the distance to the triangle's plane.
                distanceSquared = Dot(normal, pointAToSamplePoint) * Dot(normal, pointAToSamplePoint) / Dot2(normal);
//              distanceSquared = Dot(normal, pointAToSamplePoint) * Dot(normal, pointAToSamplePoint) / Dot2(normal);
            }

            return std::sqrtf(distanceSquared);
//          return std::sqrtf(distanceSquared);
        }



        // Calculates the signed distance to a quadrilateral in 3D space.
        // Calculates the signed distance to a quadrilateral in 3D space.
        static inline float SDFQuad(const Point3& samplePoint, const Point3& pointA, const Point3& pointB, const Point3& pointC, const Point3& pointD)
//      static inline float SDFQuad(const Point3& samplePoint, const Point3& pointA, const Point3& pointB, const Point3& pointC, const Point3& pointD)
        {
            Vec3 edgeAB = pointB - pointA; Vec3 pointAToSamplePoint = samplePoint - pointA;
//          Vec3 edgeAB = pointB - pointA; Vec3 pointAToSamplePoint = samplePoint - pointA;
            Vec3 edgeBC = pointC - pointB; Vec3 pointBToSamplePoint = samplePoint - pointB;
//          Vec3 edgeBC = pointC - pointB; Vec3 pointBToSamplePoint = samplePoint - pointB;
            Vec3 edgeCD = pointD - pointC; Vec3 pointCToSamplePoint = samplePoint - pointC;
//          Vec3 edgeCD = pointD - pointC; Vec3 pointCToSamplePoint = samplePoint - pointC;
            Vec3 edgeDA = pointA - pointD; Vec3 pointDToSamplePoint = samplePoint - pointD;
//          Vec3 edgeDA = pointA - pointD; Vec3 pointDToSamplePoint = samplePoint - pointD;
            Vec3 normal = Cross(edgeAB, edgeDA);
//          Vec3 normal = Cross(edgeAB, edgeDA);

            // Similar to the triangle, check if the point's projection is inside the quad.
            // Similar to the triangle, check if the point's projection is inside the quad.
            bool isOutside =
//          bool isOutside =
                Sign(Dot(Cross(edgeAB, normal), pointAToSamplePoint)) +
//              Sign(Dot(Cross(edgeAB, normal), pointAToSamplePoint)) +
                Sign(Dot(Cross(edgeBC, normal), pointBToSamplePoint)) +
//              Sign(Dot(Cross(edgeBC, normal), pointBToSamplePoint)) +
                Sign(Dot(Cross(edgeCD, normal), pointCToSamplePoint)) +
//              Sign(Dot(Cross(edgeCD, normal), pointCToSamplePoint)) +
                Sign(Dot(Cross(edgeDA, normal), pointDToSamplePoint)) < 3.0f;
//              Sign(Dot(Cross(edgeDA, normal), pointDToSamplePoint)) < 3.0f;

            float distanceSquared;
//          float distanceSquared;
            if (isOutside)
//          if (isOutside)
            {
                // If outside, find the distance to the closest of the four edges.
                // If outside, find the distance to the closest of the four edges.
                distanceSquared = std::fminf(std::fminf(std::fminf(
//              distanceSquared = std::fminf(std::fminf(std::fminf(
                    Dot2(edgeAB * std::clamp(Dot(edgeAB, pointAToSamplePoint) / Dot2(edgeAB), 0.0f, 1.0f) - pointAToSamplePoint),
//                  Dot2(edgeAB * std::clamp(Dot(edgeAB, pointAToSamplePoint) / Dot2(edgeAB), 0.0f, 1.0f) - pointAToSamplePoint),
                    Dot2(edgeBC * std::clamp(Dot(edgeBC, pointBToSamplePoint) / Dot2(edgeBC), 0.0f, 1.0f) - pointBToSamplePoint)),
//                  Dot2(edgeBC * std::clamp(Dot(edgeBC, pointBToSamplePoint) / Dot2(edgeBC), 0.0f, 1.0f) - pointBToSamplePoint)),
                    Dot2(edgeCD * std::clamp(Dot(edgeCD, pointCToSamplePoint) / Dot2(edgeCD), 0.0f, 1.0f) - pointCToSamplePoint)),
//                  Dot2(edgeCD * std::clamp(Dot(edgeCD, pointCToSamplePoint) / Dot2(edgeCD), 0.0f, 1.0f) - pointCToSamplePoint)),
                    Dot2(edgeDA * std::clamp(Dot(edgeDA, pointDToSamplePoint) / Dot2(edgeDA), 0.0f, 1.0f) - pointDToSamplePoint));
//                  Dot2(edgeDA * std::clamp(Dot(edgeDA, pointDToSamplePoint) / Dot2(edgeDA), 0.0f, 1.0f) - pointDToSamplePoint));
            }
            else
//          else
            {
                // If inside, find the distance to the quad's plane.
                // If inside, find the distance to the quad's plane.
                distanceSquared = Dot(normal, pointAToSamplePoint) * Dot(normal, pointAToSamplePoint) / Dot2(normal);
//              distanceSquared = Dot(normal, pointAToSamplePoint) * Dot(normal, pointAToSamplePoint) / Dot2(normal);
            }

            return std::sqrtf(distanceSquared);
//          return std::sqrtf(distanceSquared);
        }



        // --- Domain Operators ---
        // --- Domain Operators ---



        // Revolves a 2D shape around the Y-axis. 'sdf2D' is the distance function for the 2D profile shape in the XY plane. 'offset' is the distance from the Y-axis to revolve around.
        // Revolves a 2D shape around the Y-axis. 'sdf2D' is the distance function for the 2D profile shape in the XY plane. 'offset' is the distance from the Y-axis to revolve around.
        static inline float OpRevolution(const Point3& samplePoint, float offset, float(*sdf2D)(const Point2& p))
//      static inline float OpRevolution(const Point3& samplePoint, float offset, float(*sdf2D)(const Point2& p))
        {
            Point2 pointInProfile{ Length(samplePoint.Swizzle<'x', 'z'>()) - offset, samplePoint.y };
//          Point2 pointInProfile{ Length(samplePoint.Swizzle<'x', 'z'>()) - offset, samplePoint.y };
            return sdf2D(pointInProfile);
//          return sdf2D(pointInProfile);
        }



        // Extrudes a 2D shape along the Z-axis. 'sdf2D' is the distance function for the 2D shape in the XY plane. 'halfHeight' is half the total extrusion depth.
        // Extrudes a 2D shape along the Z-axis. 'sdf2D' is the distance function for the 2D shape in the XY plane. 'halfHeight' is half the total extrusion depth.
        static inline float OpExtrusion(const Point3& samplePoint, float halfHeight, float(*sdf2D)(const Point2& p))
//      static inline float OpExtrusion(const Point3& samplePoint, float halfHeight, float(*sdf2D)(const Point2& p))
        {
            float distanceIn2D = sdf2D(samplePoint.Swizzle<'x', 'y'>());
//          float distanceIn2D = sdf2D(samplePoint.Swizzle<'x', 'y'>());
            Vec2 distances{ distanceIn2D, std::abs(samplePoint.z) - halfHeight };
//          Vec2 distances{ distanceIn2D, std::abs(samplePoint.z) - halfHeight };

            // This is the SDF for a 2D box, which correctly bounds the extrusion.
            // This is the SDF for a 2D box, which correctly bounds the extrusion.
            float outsideDistance = Length(Max(distances, 0.0f));
//          float outsideDistance = Length(Max(distances, 0.0f));
            float  insideDistance = std::fminf(std::fmaxf(distances.x, distances.y), 0.0f);
//          float  insideDistance = std::fminf(std::fmaxf(distances.x, distances.y), 0.0f);

            return outsideDistance + insideDistance;
//          return outsideDistance + insideDistance;
        }



        // Elongates a 3D shape, creating spherical/rounded ends.
        // Elongates a 3D shape, creating spherical/rounded ends.
        static inline float OpElongateRound(const Point3& samplePoint, const Vec3& elongationBounds, float(*sdf3D)(const Point3& p))
//      static inline float OpElongateRound(const Point3& samplePoint, const Vec3& elongationBounds, float(*sdf3D)(const Point3& p))
        {
            Vec3 offsetPoint = samplePoint - Clamp(samplePoint, -elongationBounds, elongationBounds);
//          Vec3 offsetPoint = samplePoint - Clamp(samplePoint, -elongationBounds, elongationBounds);
            return sdf3D(offsetPoint);
//          return sdf3D(offsetPoint);
        }



        // Elongates a 3D shape, creating flat@@@@@/sharped ends.
        // Elongates a 3D shape, creating flat@@@@@/sharped ends.
        static inline float OpElongateSharp(const Point3& samplePoint, const Vec3& elongationBounds, float(*sdf3D)(const Point3& p))
//      static inline float OpElongateSharp(const Point3& samplePoint, const Vec3& elongationBounds, float(*sdf3D)(const Point3& p))
        {
            Vec3 offsetPoint = Abs(samplePoint) - elongationBounds;
//          Vec3 offsetPoint = Abs(samplePoint) - elongationBounds;
            float outsideDistance = sdf3D(Max(offsetPoint, 0.0f));
//          float outsideDistance = sdf3D(Max(offsetPoint, 0.0f));
            float  insideDistance = std::fminf(std::fmaxf(offsetPoint.x, std::fmaxf(offsetPoint.y, offsetPoint.z)), 0.0f);
//          float  insideDistance = std::fminf(std::fmaxf(offsetPoint.x, std::fmaxf(offsetPoint.y, offsetPoint.z)), 0.0f);
            return outsideDistance + insideDistance;
//          return outsideDistance + insideDistance;
        }



        // --- Morphological Operators ---
        // --- Morphological Operators ---



        // Pushes the surface of a shape outwards (or inwards) by a radius.
        // Pushes the surface of a shape outwards (or inwards) by a radius.
        static inline float OpRound(float distance, float radius   ) { return     distance  - radius   ; }
//      static inline float OpRound(float distance, float radius   ) { return     distance  - radius   ; }
        static inline float OpOnion(float distance, float thickness) { return abs(distance) - thickness; }
//      static inline float OpOnion(float distance, float thickness) { return abs(distance) - thickness; }



        // --- Boolean Operators (@Hard@) ---
        // --- Boolean Operators (@Hard@) ---



        static inline float OpUnion(float distanceA, float distanceB)
//      static inline float OpUnion(float distanceA, float distanceB)
        {
            return std::fminf(distanceA, distanceB);
//          return std::fminf(distanceA, distanceB);
        }
        static inline float OpSubtraction(float distanceA, float distanceB)
//      static inline float OpSubtraction(float distanceA, float distanceB)
        {
            return std::fmaxf(-distanceA, distanceB);
//          return std::fmaxf(-distanceA, distanceB);
        }
        static inline float OpIntersection(float distanceA, float distanceB)
//      static inline float OpIntersection(float distanceA, float distanceB)
        {
            return std::fmaxf(distanceA, distanceB);
//          return std::fmaxf(distanceA, distanceB);
        }
        static inline float OpXor1(float distanceA, float distanceB)
//      static inline float OpXor1(float distanceA, float distanceB)
        {
            return std::fmaxf(OpSubtraction(distanceA, distanceB), OpSubtraction(distanceB, distanceA));
//          return std::fmaxf(OpSubtraction(distanceA, distanceB), OpSubtraction(distanceB, distanceA));
        }
        static inline float OpXor2(float distanceA, float distanceB)
//      static inline float OpXor2(float distanceA, float distanceB)
        {
            return std::fmaxf(std::fminf(distanceA, distanceB), -std::fmaxf(distanceA, distanceB));
//          return std::fmaxf(std::fminf(distanceA, distanceB), -std::fmaxf(distanceA, distanceB));
        }



        // --- Boolean Operators (Smooth) ---
        // --- Boolean Operators (Smooth) ---



        static inline float OpSmoothUnion       (float distA, float distB, float smoothness) { float h = std::clamp(0.5f + 0.5f * (distB - distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB,  distA, h) - smoothness * h * (1.0f - h); }
//      static inline float OpSmoothUnion       (float distA, float distB, float smoothness) { float h = std::clamp(0.5f + 0.5f * (distB - distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB,  distA, h) - smoothness * h * (1.0f - h); }
        static inline float OpSmoothSubtraction (float distA, float distB, float smoothness) { float h = std::clamp(0.5f - 0.5f * (distB + distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB, -distA, h) + smoothness * h * (1.0f - h); }
//      static inline float OpSmoothSubtraction (float distA, float distB, float smoothness) { float h = std::clamp(0.5f - 0.5f * (distB + distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB, -distA, h) + smoothness * h * (1.0f - h); }
        static inline float OpSmoothIntersection(float distA, float distB, float smoothness) { float h = std::clamp(0.5f - 0.5f * (distB - distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB,  distA, h) + smoothness * h * (1.0f - h); }
//      static inline float OpSmoothIntersection(float distA, float distB, float smoothness) { float h = std::clamp(0.5f - 0.5f * (distB - distA) / smoothness, 0.0f, 1.0f); return BlendLinear(distB,  distA, h) + smoothness * h * (1.0f - h); }



        // --- Transformation Operators ---
        // --- Transformation Operators ---



        static inline Point3 OpTranslate(const Point3& samplePoint, const Vec3& translation)
//      static inline Point3 OpTranslate(const Point3& samplePoint, const Vec3& translation)
        {
            return samplePoint - translation;
//          return samplePoint - translation;
        }



        // Rotates a shape around a pivot point. To do this, we apply the INVERSE rotation to the sample point.
        // Rotates a shape around a pivot point. To do this, we apply the INVERSE rotation to the sample point.
        static inline Point3 OpRotate(const Point3& samplePoint, const Vec3& pivot, const Vec3& axis, float angleRadians)
//      static inline Point3 OpRotate(const Point3& samplePoint, const Vec3& pivot, const Vec3& axis, float angleRadians)
        {
            // Move the point to be relative to the pivot
            // Move the point to be relative to the pivot
            Vec3 pointToRotate = samplePoint - pivot;
//          Vec3 pointToRotate = samplePoint - pivot;

            Vec3 k = Normalize(axis);
//          Vec3 k = Normalize(axis);
            float cosTheta = std::cos(angleRadians);
//          float cosTheta = std::cos(angleRadians);
            float sinTheta = std::sin(angleRadians);
//          float sinTheta = std::sin(angleRadians);

            // Apply the INVERSE rotation using Rodrigues' formula. The inverse is achieved by rotating by -angle, which means flipping the sign of sin(angle).
            // Apply the INVERSE rotation using Rodrigues' formula. The inverse is achieved by rotating by -angle, which means flipping the sign of sin(angle).
            Vec3 rotatedPoint = pointToRotate * cosTheta - Cross(k, pointToRotate) * sinTheta /* <--The sign is flipped here for inverse rotation */ + k * Dot(k, pointToRotate) * (1.0f - cosTheta);
//          Vec3 rotatedPoint = pointToRotate * cosTheta - Cross(k, pointToRotate) * sinTheta /* <--The sign is flipped here for inverse rotation */ + k * Dot(k, pointToRotate) * (1.0f - cosTheta);

            // Move the point back to world space and sample the original, un-rotated SDF
            // Move the point back to world space and sample the original, un-rotated SDF
            return rotatedPoint + pivot;
//          return rotatedPoint + pivot;
        }



        static inline float OpScale(const Point3& samplePoint, float scaleFactor, float(*shapeSDF)(const Point3& p))
//      static inline float OpScale(const Point3& samplePoint, float scaleFactor, float(*shapeSDF)(const Point3& p))
        {
            return shapeSDF(samplePoint / scaleFactor) * scaleFactor;
//          return shapeSDF(samplePoint / scaleFactor) * scaleFactor;
        }


        
        static inline Point3 OpSymX(const Point3& samplePoint) { Vec3 foldedPoint = samplePoint; foldedPoint.x = std::fabsf(foldedPoint.x); return foldedPoint; }
        static inline Point3 OpSymY(const Point3& samplePoint) { Vec3 foldedPoint = samplePoint; foldedPoint.y = std::fabsf(foldedPoint.y); return foldedPoint; }
        static inline Point3 OpSymZ(const Point3& samplePoint) { Vec3 foldedPoint = samplePoint; foldedPoint.z = std::fabsf(foldedPoint.z); return foldedPoint; }



        static inline Point3 OpSymXY(const Point3& samplePoint)
//      static inline Point3 OpSymXY(const Point3& samplePoint)
        {
            Vec3 foldedPoint = samplePoint;
//          Vec3 foldedPoint = samplePoint;
            Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'x', 'y'>());
//          Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'x', 'y'>());
            foldedPoint.x = foldedPlane.x;
//          foldedPoint.x = foldedPlane.x;
            foldedPoint.y = foldedPlane.y;
//          foldedPoint.y = foldedPlane.y;
            return foldedPoint;
//          return foldedPoint;
        }
        static inline Point3 OpSymXZ(const Point3& samplePoint)
//      static inline Point3 OpSymXZ(const Point3& samplePoint)
        {
            Vec3 foldedPoint = samplePoint;
//          Vec3 foldedPoint = samplePoint;
            Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'x', 'z'>());
//          Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'x', 'z'>());
            foldedPoint.x = foldedPlane.x;
//          foldedPoint.x = foldedPlane.x;
            foldedPoint.z = foldedPlane.y;
//          foldedPoint.z = foldedPlane.y;
            return foldedPoint;
//          return foldedPoint;
        }
        static inline Point3 OpSymYZ(const Point3& samplePoint)
//      static inline Point3 OpSymYZ(const Point3& samplePoint)
        {
            Vec3 foldedPoint = samplePoint;
//          Vec3 foldedPoint = samplePoint;
            Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'y', 'z'>());
//          Vec2 foldedPlane = Abs(foldedPoint.Swizzle<'y', 'z'>());
            foldedPoint.y = foldedPlane.x;
//          foldedPoint.y = foldedPlane.x;
            foldedPoint.z = foldedPlane.y;
//          foldedPoint.z = foldedPlane.y;
            return foldedPoint;
//          return foldedPoint;
        }



        static inline Point3 OpRepetition(const Point3& samplePoint, const Vec3& spacing)
//      static inline Point3 OpRepetition(const Point3& samplePoint, const Vec3& spacing)
        {
            return samplePoint - spacing * Round(samplePoint / spacing);
//          return samplePoint - spacing * Round(samplePoint / spacing);
        }



        static inline Point3 OpLimitedRepetition(const Point3& samplePoint, float spacing, const Vec3& limit)
//      static inline Point3 OpLimitedRepetition(const Point3& samplePoint, float spacing, const Vec3& limit)
        {
            return samplePoint - spacing * Clamp(Round(samplePoint / spacing), -limit, limit);
//          return samplePoint - spacing * Clamp(Round(samplePoint / spacing), -limit, limit);
        }



        static inline Point3 OpLimitedRepetition(const Point3& samplePoint, const Vec3& spacing, const Vec3& limit)
//      static inline Point3 OpLimitedRepetition(const Point3& samplePoint, const Vec3& spacing, const Vec3& limit)
        {
            return samplePoint - spacing * Clamp(Round(samplePoint / spacing), -limit, limit);
//          return samplePoint - spacing * Clamp(Round(samplePoint / spacing), -limit, limit);
        }



        static inline float OpDisplace(const Point3& samplePoint, float(*sdf3D)(const Point3& p), float(*displacement)(const Point3& p))
//      static inline float OpDisplace(const Point3& samplePoint, float(*sdf3D)(const Point3& p), float(*displacement)(const Point3& p))
        {
            float d1 = sdf3D       (samplePoint);
//          float d1 = sdf3D       (samplePoint);
            float d2 = displacement(samplePoint);
//          float d2 = displacement(samplePoint);
            return d1 + d2;
//          return d1 + d2;
        }



        static inline Point3 OpTwist(const Point3& samplePoint, float twistAmount = 0.1f)
//      static inline Point3 OpTwist(const Point3& samplePoint, float twistAmount = 0.1f)
        {
            float c = std::cosf(twistAmount * samplePoint.y);
//          float c = std::cosf(twistAmount * samplePoint.y);
            float s = std::sinf(twistAmount * samplePoint.y);
//          float s = std::sinf(twistAmount * samplePoint.y);
            float x = c * samplePoint.x - s * samplePoint.z ;
//          float x = c * samplePoint.x - s * samplePoint.z ;
            float z = s * samplePoint.x + c * samplePoint.z ;
//          float z = s * samplePoint.x + c * samplePoint.z ;
            Vec3 q{ x, z, samplePoint.y };
//          Vec3 q{ x, z, samplePoint.y };
            return q;
//          return q;
        }



        static inline Point3 OpCheapBend(const Point3& samplePoint, float bendAmount = 0.1f)
//      static inline Point3 OpCheapBend(const Point3& samplePoint, float bendAmount = 0.1f)
        {
            float c = std::cosf( bendAmount * samplePoint.x);
//          float c = std::cosf( bendAmount * samplePoint.x);
            float s = std::sinf( bendAmount * samplePoint.x);
//          float s = std::sinf( bendAmount * samplePoint.x);
            float x = c * samplePoint.x - s * samplePoint.y ;
//          float x = c * samplePoint.x - s * samplePoint.y ;
            float y = s * samplePoint.x + c * samplePoint.y ;
//          float y = s * samplePoint.x + c * samplePoint.y ;
            Vec3 q{ x, y, samplePoint.z };
//          Vec3 q{ x, y, samplePoint.z };
            return q;
//          return q;
        }
}


#endif
