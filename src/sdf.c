// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "core.h"
#include "shape.h"
#include "simd.h"

#include "box3d/collision.h"
#include "box3d/constants.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

typedef struct b3SDFBuildTriangle
{
	b3Vec3 vertices[3];
} b3SDFBuildTriangle;

static int b3CompareSDFPoints( const b3Vec3* a, const b3Vec3* b )
{
	if ( a->x != b->x )
	{
		return a->x < b->x ? -1 : 1;
	}
	if ( a->y != b->y )
	{
		return a->y < b->y ? -1 : 1;
	}
	if ( a->z != b->z )
	{
		return a->z < b->z ? -1 : 1;
	}
	return 0;
}

static void b3CanonicalizeSDFTriangle( b3Vec3 output[3], const b3SDFBuildTriangle* triangle )
{
	output[0] = triangle->vertices[0];
	output[1] = triangle->vertices[1];
	output[2] = triangle->vertices[2];
	for ( int i = 1; i < 3; ++i )
	{
		b3Vec3 point = output[i];
		int j = i;
		while ( j > 0 && b3CompareSDFPoints( output + j - 1, &point ) > 0 )
		{
			output[j] = output[j - 1];
			j -= 1;
		}
		output[j] = point;
	}
}

static bool b3SameSDFTriangleGeometry( const b3SDFBuildTriangle* a, const b3SDFBuildTriangle* b )
{
	b3Vec3 keyA[3], keyB[3];
	b3CanonicalizeSDFTriangle( keyA, a );
	b3CanonicalizeSDFTriangle( keyB, b );
	for ( int i = 0; i < 3; ++i )
	{
		if ( b3CompareSDFPoints( keyA + i, keyB + i ) != 0 )
		{
			return false;
		}
	}
	return true;
}

static int b3SDFIndex( int countX, int countY, int x, int y, int z )
{
	return x + countX * ( y + countY * z );
}

static b3Vec3 b3SDFGridPoint( const b3SDFDef* data, int x, int y, int z )
{
	return b3Add( data->origin, b3Mul( data->spacing, (b3Vec3){ (float)x, (float)y, (float)z } ) );
}

static int b3SDFCellTriangles( const b3Vec3* points, const float* values, b3SDFBuildTriangle* output )
{
	// A six-tetrahedral decomposition avoids the lookup table and ambiguous cases of
	// marching cubes. The diagonal is shared by all six tetrahedra.
	static const int tetrahedra[6][4] = {
		{ 0, 5, 1, 6 }, { 0, 1, 2, 6 }, { 0, 2, 3, 6 }, { 0, 3, 7, 6 }, { 0, 7, 4, 6 }, { 0, 4, 5, 6 },
	};

	static const int edges[6][2] = {
		{ 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 },
	};

	int triangleCount = 0;
	for ( int t = 0; t < 6; ++t )
	{
		b3Vec3 tetraPoints[4];
		float tetraValues[4];
		b3Vec3 positiveCenter = b3Vec3_zero;
		b3Vec3 negativeCenter = b3Vec3_zero;
		int positiveCount = 0;
		for ( int i = 0; i < 4; ++i )
		{
			int index = tetrahedra[t][i];
			tetraPoints[i] = points[index];
			tetraValues[i] = values[index];
			if ( tetraValues[i] >= 0.0f )
			{
				positiveCenter = b3Add( positiveCenter, tetraPoints[i] );
				positiveCount += 1;
			}
			else
			{
				negativeCenter = b3Add( negativeCenter, tetraPoints[i] );
			}
		}

		if ( positiveCount == 0 || positiveCount == 4 )
		{
			continue;
		}

		b3Vec3 intersections[6];
		int intersectionCount = 0;
		for ( int e = 0; e < 6; ++e )
		{
			int i1 = edges[e][0];
			int i2 = edges[e][1];
			bool inside1 = tetraValues[i1] < 0.0f;
			bool inside2 = tetraValues[i2] < 0.0f;
			if ( inside1 == inside2 )
			{
				continue;
			}

			float denominator = tetraValues[i1] - tetraValues[i2];
			float alpha = denominator != 0.0f ? tetraValues[i1] / denominator : 0.5f;
			intersections[intersectionCount++] = b3Lerp( tetraPoints[i1], tetraPoints[i2], alpha );
		}

		if ( intersectionCount < 3 )
		{
			continue;
		}

		b3Vec3 center = b3Vec3_zero;
		for ( int i = 0; i < intersectionCount; ++i )
		{
			center = b3Add( center, intersections[i] );
		}
		center = b3MulSV( 1.0f / (float)intersectionCount, center );
		positiveCenter = b3MulSV( 1.0f / (float)positiveCount, positiveCenter );

	b3Vec3 normalDirection = b3Sub( positiveCenter, center );
	if ( b3LengthSquared( normalDirection ) == 0.0f )
	{
		// All non-negative vertices may lie exactly on the isosurface. In that
		// case, orient away from the negative vertices instead.
		int negativeCount = 4 - positiveCount;
			negativeCenter = b3MulSV( 1.0f / (float)negativeCount, negativeCenter );
			normalDirection = b3Sub( center, negativeCenter );
		}
		normalDirection = b3Normalize( normalDirection );
		if ( b3LengthSquared( normalDirection ) == 0.0f )
		{
			normalDirection = b3Vec3_axisY;
		}

		// Sort the intersection polygon around the positive-side direction.
		b3Vec3 u = b3Perp( normalDirection );
		b3Vec3 v = b3Cross( normalDirection, u );
		float angles[6];
		for ( int i = 0; i < intersectionCount; ++i )
		{
			b3Vec3 d = b3Sub( intersections[i], center );
			angles[i] = atan2f( b3Dot( d, v ), b3Dot( d, u ) );
		}

		for ( int i = 1; i < intersectionCount; ++i )
		{
			b3Vec3 point = intersections[i];
			float angle = angles[i];
			int j = i;
			while ( j > 0 && angles[j - 1] > angle )
			{
				intersections[j] = intersections[j - 1];
				angles[j] = angles[j - 1];
				--j;
			}
			intersections[j] = point;
			angles[j] = angle;
		}

		int fanCount = intersectionCount - 2;
		for ( int i = 0; i < fanCount; ++i )
		{
			b3Vec3 p1 = intersections[0];
			b3Vec3 p2 = intersections[i + 1];
			b3Vec3 p3 = intersections[i + 2];
			b3Vec3 triangleNormal = b3Cross( b3Sub( p2, p1 ), b3Sub( p3, p1 ) );
			float minDoubleArea = 0.02f * B3_LINEAR_SLOP * B3_LINEAR_SLOP;
			if ( b3LengthSquared( triangleNormal ) <= minDoubleArea * minDoubleArea )
			{
				continue;
			}

			if ( b3Dot( triangleNormal, normalDirection ) < 0.0f )
			{
				b3Vec3 swap = p2;
				p2 = p3;
				p3 = swap;
			}

			if ( output != NULL )
			{
				output[triangleCount].vertices[0] = p1;
				output[triangleCount].vertices[1] = p2;
				output[triangleCount].vertices[2] = p3;
			}
			triangleCount += 1;
		}
	}

	return triangleCount;
}

static int b3SDFBuildCell( const b3SDFDef* data, int x, int y, int z, b3SDFBuildTriangle* output )
{
	b3Vec3 points[8];
	float values[8];
	static const int offsets[8][3] = {
		{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 },
	};

	for ( int i = 0; i < 8; ++i )
	{
		int px = x + offsets[i][0];
		int py = y + offsets[i][1];
		int pz = z + offsets[i][2];
		points[i] = b3SDFGridPoint( data, px, py, pz );
		values[i] = data->distances[b3SDFIndex( data->countX, data->countY, px, py, pz )];
	}

	return b3SDFCellTriangles( points, values, output );
}

static float b3SDFSample( const b3SDFData* sdf, b3Vec3 point )
{
	if ( b3AABB_Contains( sdf->aabb, (b3AABB){ point, point } ) == false )
	{
		return FLT_MAX;
	}

	const float* distances = b3GetSDFDistances( sdf );
	b3Vec3 relative = b3Sub( point, sdf->origin );
	b3Vec3 q = { relative.x / sdf->spacing.x, relative.y / sdf->spacing.y, relative.z / sdf->spacing.z };
	int x = b3ClampInt( (int)floorf( q.x ), 0, sdf->countX - 2 );
	int y = b3ClampInt( (int)floorf( q.y ), 0, sdf->countY - 2 );
	int z = b3ClampInt( (int)floorf( q.z ), 0, sdf->countZ - 2 );
	float fx = q.x - (float)x;
	float fy = q.y - (float)y;
	float fz = q.z - (float)z;
	fx = b3ClampFloat( fx, 0.0f, 1.0f );
	fy = b3ClampFloat( fy, 0.0f, 1.0f );
	fz = b3ClampFloat( fz, 0.0f, 1.0f );

	float value[8];
	static const int offsets[8][3] = {
		{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 },
	};
	for ( int i = 0; i < 8; ++i )
	{
		value[i] = distances[b3SDFIndex( sdf->countX, sdf->countY, x + offsets[i][0], y + offsets[i][1], z + offsets[i][2] )];
	}

	float x0 = b3LerpFloat( value[0], value[1], fx );
	float x1 = b3LerpFloat( value[3], value[2], fx );
	float x2 = b3LerpFloat( value[4], value[5], fx );
	float x3 = b3LerpFloat( value[7], value[6], fx );
	float y0 = b3LerpFloat( x0, x1, fy );
	float y1 = b3LerpFloat( x2, x3, fy );
	return b3LerpFloat( y0, y1, fz );
}

b3SDFData* b3CreateSDF( const b3SDFDef* data )
{
	if ( data == NULL || data->distances == NULL || data->countX < 2 || data->countY < 2 || data->countZ < 2 )
	{
		return NULL;
	}

	if ( b3IsValidVec3( data->origin ) == false || b3IsValidVec3( data->spacing ) == false || data->spacing.x <= 0.0f ||
		 data->spacing.y <= 0.0f || data->spacing.z <= 0.0f )
	{
		return NULL;
	}

	size_t sampleCount = (size_t)data->countX * (size_t)data->countY * (size_t)data->countZ;
	size_t cellCount = (size_t)( data->countX - 1 ) * (size_t)( data->countY - 1 ) * (size_t)( data->countZ - 1 );
	if ( sampleCount > (size_t)INT_MAX || cellCount > (size_t)INT_MAX / 12u )
	{
		return NULL;
	}

	int minX = data->countX;
	int minY = data->countY;
	int minZ = data->countZ;
	int maxX = -1;
	int maxY = -1;
	int maxZ = -1;
	for ( int z = 0; z < data->countZ; ++z )
	{
		for ( int y = 0; y < data->countY; ++y )
		{
			for ( int x = 0; x < data->countX; ++x )
			{
				float distance = data->distances[b3SDFIndex( data->countX, data->countY, x, y, z )];
				if ( b3IsValidFloat( distance ) == false )
				{
					return NULL;
				}
				if ( distance <= 0.0f )
				{
					minX = b3MinInt( minX, x );
					minY = b3MinInt( minY, y );
					minZ = b3MinInt( minZ, z );
					maxX = b3MaxInt( maxX, x );
					maxY = b3MaxInt( maxY, y );
					maxZ = b3MaxInt( maxZ, z );
				}
			}
		}
	}

	b3Vec3 gridExtent =
		b3Mul( data->spacing, (b3Vec3){ (float)( data->countX - 1 ), (float)( data->countY - 1 ), (float)( data->countZ - 1 ) } );
	b3Vec3 gridUpper = b3Add( data->origin, gridExtent );
	if ( b3IsValidVec3( gridExtent ) == false || b3IsValidVec3( gridUpper ) == false )
	{
		return NULL;
	}

	size_t byteCount = b3AlignUp8( sizeof( b3SDFData ) );
	int distancesOffset = (int)byteCount;
	byteCount += b3AlignUp8( sampleCount * sizeof( float ) );

	if ( byteCount > (size_t)INT_MAX )
	{
		return NULL;
	}

	b3SDFData* sdf = (b3SDFData*)b3Alloc( byteCount );
	memset( sdf, 0, byteCount );
	sdf->version = B3_SDF_VERSION;
	sdf->byteCount = (int)byteCount;
	sdf->origin = data->origin;
	sdf->spacing = data->spacing;
	sdf->countX = data->countX;
	sdf->countY = data->countY;
	sdf->countZ = data->countZ;
	sdf->distancesOffset = distancesOffset;
	if ( maxX >= 0 )
	{
		b3Vec3 lowerIndex = { (float)b3MaxInt( minX - 1, 0 ), (float)b3MaxInt( minY - 1, 0 ), (float)b3MaxInt( minZ - 1, 0 ) };
		b3Vec3 upperIndex = { (float)b3MinInt( maxX + 1, data->countX - 1 ), (float)b3MinInt( maxY + 1, data->countY - 1 ),
							  (float)b3MinInt( maxZ + 1, data->countZ - 1 ) };
		sdf->aabb.lowerBound = b3Add( data->origin, b3Mul( data->spacing, lowerIndex ) );
		sdf->aabb.upperBound = b3Add( data->origin, b3Mul( data->spacing, upperIndex ) );
	}
	else
	{
		sdf->aabb = (b3AABB){ data->origin, data->origin };
	}

	float* distances = (float*)( (intptr_t)sdf + distancesOffset );
	memcpy( distances, data->distances, sampleCount * sizeof( float ) );

	sdf->hash = 0;
	sdf->hash = b3NonZeroHash( b3Hash( B3_HASH_INIT, (const uint8_t*)sdf, sdf->byteCount ) );
	return sdf;
}

void b3DestroySDF( b3SDFData* sdf )
{
	b3Free( sdf, sdf->byteCount );
}

b3AABB b3ComputeSDFAABB( const b3SDFData* shape, b3Transform transform )
{
	return b3AABB_Transform( transform, shape->aabb );
}

typedef struct b3SDFOverlapContext
{
	b3ShapeProxy proxy;
	b3SimplexCache cache;
	bool hit;
} b3SDFOverlapContext;

static bool b3SDFOverlapFcn( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* rawContext )
{
	B3_UNUSED( triangleIndex );
	b3SDFOverlapContext* context = rawContext;
	b3Vec3 vertices[3] = { a, b, c };
	b3DistanceInput input = {
		.proxyA = { vertices, 3, 0.0f },
		.proxyB = context->proxy,
		.transform = b3Transform_identity,
		.useRadii = true,
	};
	context->cache.count = 0;
	b3DistanceOutput output = b3ShapeDistance( &input, &context->cache, NULL, 0 );
	context->hit = output.distance < 0.1f * B3_LINEAR_SLOP;
	return context->hit == false;
}

typedef struct b3SDFShapeCastContext
{
	const b3ShapeCastInput* input;
	b3CastOutput output;
} b3SDFShapeCastContext;

static bool b3SDFShapeCastFcn( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* rawContext )
{
	b3SDFShapeCastContext* context = rawContext;
	b3Vec3 vertices[3] = { b3Vec3_zero, b3Sub( b, a ), b3Sub( c, a ) };
	b3ShapeCastPairInput pairInput = {
		.proxyA = { vertices, 3, 0.0f },
		.proxyB = context->input->proxy,
		.transform = { b3Neg( a ), b3Quat_identity },
		.translationB = context->input->translation,
		.maxFraction = context->output.fraction,
		.canEncroach = context->input->canEncroach,
	};
	b3CastOutput output = b3ShapeCast( &pairInput );
	if ( output.hit )
	{
		output.point = b3Add( output.point, a );
		output.triangleIndex = triangleIndex;
		output.materialIndex = 0;
		context->output = output;
	}
	return true;
}

typedef struct b3SDFRayContext
{
	const b3RayCastInput* input;
	b3CastOutput output;
} b3SDFRayContext;

static bool b3SDFRayFcn( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* rawContext )
{
	b3SDFRayContext* context = rawContext;
	float fraction = b3IntersectRayTriangle( b3LoadV( &context->input->origin.x ), b3LoadV( &context->input->translation.x ),
											 b3LoadV( &a.x ), b3LoadV( &b.x ), b3LoadV( &c.x ) );
	if ( fraction < context->output.fraction )
	{
		context->output.point = b3MulAdd( context->input->origin, fraction, context->input->translation );
		context->output.normal = b3Normalize( b3Cross( b3Sub( b, a ), b3Sub( c, a ) ) );
		context->output.fraction = fraction;
		context->output.triangleIndex = triangleIndex;
		context->output.materialIndex = 0;
		context->output.hit = true;
	}
	return true;
}

typedef struct b3SDFMoverContext
{
	b3PlaneResult* planes;
	int capacity;
	int count;
	b3DistanceInput input;
	b3SimplexCache cache;
	float radius;
} b3SDFMoverContext;

static bool b3SDFMoverFcn( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* rawContext )
{
	B3_UNUSED( triangleIndex );
	b3SDFMoverContext* context = rawContext;
	b3Vec3 vertices[3] = { a, b, c };
	context->input.proxyA = (b3ShapeProxy){ vertices, 3, 0.0f };
	context->cache.count = 0;
	b3DistanceOutput output = b3ShapeDistance( &context->input, &context->cache, NULL, 0 );
	if ( output.distance > 0.0f && output.distance <= context->radius )
	{
		b3Plane plane = { output.normal, context->radius - output.distance };
		context->planes[context->count++] = (b3PlaneResult){ plane, output.pointA };
	}
	return context->count < context->capacity;
}

bool b3OverlapSDF( const b3SDFData* shape, b3Transform shapeTransform, const b3ShapeProxy* proxy )
{
	B3_ASSERT( proxy->count > 0 );
	b3Vec3 buffer[B3_MAX_SHAPE_CAST_POINTS];
	b3ShapeProxy localProxy = b3MakeLocalProxy( proxy, shapeTransform, buffer );
	b3AABB proxyBounds = b3ComputeProxyAABB( &localProxy );

	// The signed samples make a point/volume query work even when the query shape is
	// completely inside the SDF and therefore does not touch the extracted surface.
	for ( int i = 0; i < localProxy.count; ++i )
	{
		float distance = b3SDFSample( shape, localProxy.points[i] );
		if ( distance <= localProxy.radius + B3_LINEAR_SLOP )
		{
			return true;
		}
	}
	float centerDistance = b3SDFSample( shape, b3AABB_Center( proxyBounds ) );
	if ( centerDistance <= localProxy.radius + B3_LINEAR_SLOP )
	{
		return true;
	}
	b3SDFOverlapContext context = { .proxy = localProxy };
	b3QuerySDF( shape, proxyBounds, b3SDFOverlapFcn, &context );
	return context.hit;
}

b3CastOutput b3ShapeCastSDF( const b3SDFData* shape, const b3ShapeCastInput* input )
{
	b3AABB startBounds = b3MakeAABB( input->proxy.points, input->proxy.count, input->proxy.radius );
	b3AABB endBounds = { b3Add( startBounds.lowerBound, b3MulSV( input->maxFraction, input->translation ) ),
						 b3Add( startBounds.upperBound, b3MulSV( input->maxFraction, input->translation ) ) };
	b3SDFShapeCastContext context = {
		.input = input,
		.output = { .fraction = input->maxFraction, .triangleIndex = B3_NULL_INDEX },
	};
	b3QuerySDF( shape, b3AABB_Union( startBounds, endBounds ), b3SDFShapeCastFcn, &context );
	return context.output;
}

b3CastOutput b3RayCastSDF( const b3SDFData* shape, const b3RayCastInput* input )
{
	b3Vec3 end = b3MulAdd( input->origin, input->maxFraction, input->translation );
	b3SDFRayContext context = {
		.input = input,
		.output = { .fraction = input->maxFraction, .triangleIndex = B3_NULL_INDEX },
	};
	b3QuerySDF( shape, (b3AABB){ b3Min( input->origin, end ), b3Max( input->origin, end ) }, b3SDFRayFcn, &context );
	return context.output;
}

static int b3BuildSDFCellUniqueLocal( const b3SDFDef* source, int x, int y, int z, b3SDFBuildTriangle* triangles )
{
	b3SDFBuildTriangle candidates[12];
	int candidateCount = b3SDFBuildCell( source, x, y, z, candidates );
	int count = 0;
	for ( int i = 0; i < candidateCount; ++i )
	{
		b3SDFBuildTriangle candidate = candidates[i];
		bool duplicate = false;
		for ( int j = 0; j < count; ++j )
		{
			if ( b3SameSDFTriangleGeometry( triangles + j, &candidate ) )
			{
				duplicate = true;
				break;
			}
		}
		if ( duplicate == false )
		{
			triangles[count++] = candidate;
		}
	}
	return count;
}

static bool b3SDFCellHasTriangle( const b3SDFDef* source, int x, int y, int z, const b3SDFBuildTriangle* triangle )
{
	b3SDFBuildTriangle triangles[12];
	int count = b3BuildSDFCellUniqueLocal( source, x, y, z, triangles );
	for ( int i = 0; i < count; ++i )
	{
		if ( b3SameSDFTriangleGeometry( triangles + i, triangle ) )
		{
			return true;
		}
	}
	return false;
}

static int b3BuildSDFCellUnique( const b3SDFData* sdf, int x, int y, int z, b3SDFBuildTriangle* triangles )
{
	b3SDFDef source = {
		.distances = (float*)b3GetSDFDistances( sdf ),
		.origin = sdf->origin,
		.spacing = sdf->spacing,
		.countX = sdf->countX,
		.countY = sdf->countY,
		.countZ = sdf->countZ,
	};
	b3SDFBuildTriangle candidates[12];
	int candidateCount = b3BuildSDFCellUniqueLocal( &source, x, y, z, candidates );
	int count = 0;
	for ( int i = 0; i < candidateCount; ++i )
	{
		b3SDFBuildTriangle candidate = candidates[i];

		// Prefer the adjacent lower-index cell for a surface exactly on a cell
		// boundary, but only if that cell generates the same triangle. Exact-zero
		// samples can make only one of the two cells generate the surface.
		b3Vec3 lower = b3SDFGridPoint( &source, x, y, z );
		bool onLowerX = x > 0 && candidate.vertices[0].x == lower.x && candidate.vertices[1].x == lower.x &&
						candidate.vertices[2].x == lower.x;
		bool onLowerY = y > 0 && candidate.vertices[0].y == lower.y && candidate.vertices[1].y == lower.y &&
						candidate.vertices[2].y == lower.y;
		bool onLowerZ = z > 0 && candidate.vertices[0].z == lower.z && candidate.vertices[1].z == lower.z &&
						candidate.vertices[2].z == lower.z;
		if ( ( onLowerX && b3SDFCellHasTriangle( &source, x - 1, y, z, &candidate ) ) ||
			 ( onLowerY && b3SDFCellHasTriangle( &source, x, y - 1, z, &candidate ) ) ||
			 ( onLowerZ && b3SDFCellHasTriangle( &source, x, y, z - 1, &candidate ) ) )
		{
			continue;
		}

		triangles[count++] = candidate;
	}
	return count;
}

static int b3SDFVertexId( b3Vec3 vertex )
{
	// Canonicalize signed zero so geometrically shared vertices receive the same key.
	vertex.x = vertex.x == 0.0f ? 0.0f : vertex.x;
	vertex.y = vertex.y == 0.0f ? 0.0f : vertex.y;
	vertex.z = vertex.z == 0.0f ? 0.0f : vertex.z;
	uint32_t hash = b3Hash( B3_HASH_INIT, (const uint8_t*)&vertex, sizeof( vertex ) );
	return (int)( hash & INT_MAX );
}

int b3GetSDFCellTriangles( const b3SDFData* sdf, int cellIndex, b3Triangle output[12] )
{
	int cellsX = sdf->countX - 1;
	int cellsY = sdf->countY - 1;
	int cellCount = cellsX * cellsY * ( sdf->countZ - 1 );
	B3_ASSERT( 0 <= cellIndex && cellIndex < cellCount );
	if ( cellIndex < 0 || cellIndex >= cellCount )
	{
		return 0;
	}
	int x = cellIndex % cellsX;
	int yz = cellIndex / cellsX;
	int y = yz % cellsY;
	int z = yz / cellsY;
	b3SDFBuildTriangle triangles[12];
	int count = b3BuildSDFCellUnique( sdf, x, y, z, triangles );
	for ( int i = 0; i < count; ++i )
	{
		b3SDFBuildTriangle triangle = triangles[i];
		output[i] = (b3Triangle){
			.vertices = { triangle.vertices[0], triangle.vertices[1], triangle.vertices[2] },
			.i1 = b3SDFVertexId( triangle.vertices[0] ),
			.i2 = b3SDFVertexId( triangle.vertices[1] ),
			.i3 = b3SDFVertexId( triangle.vertices[2] ),
			.flags = 0,
		};
	}
	return count;
}

void b3QuerySDF( const b3SDFData* sdf, b3AABB bounds, b3MeshQueryFcn* fcn, void* context )
{
	if ( b3AABB_Overlaps( bounds, sdf->aabb ) == false )
	{
		return;
	}

	b3Vec3 d1 = b3Sub( bounds.lowerBound, sdf->origin );
	b3Vec3 d2 = b3Sub( bounds.upperBound, sdf->origin );
	b3Vec3 q1 = { d1.x / sdf->spacing.x, d1.y / sdf->spacing.y, d1.z / sdf->spacing.z };
	b3Vec3 q2 = { d2.x / sdf->spacing.x, d2.y / sdf->spacing.y, d2.z / sdf->spacing.z };
	// Include the cell immediately below the lower bound. A zero surface that
	// lies exactly on a cell boundary is owned by that lower cell.
	int minX = b3ClampInt( (int)floorf( q1.x ) - 1, 0, sdf->countX - 2 );
	int minY = b3ClampInt( (int)floorf( q1.y ) - 1, 0, sdf->countY - 2 );
	int minZ = b3ClampInt( (int)floorf( q1.z ) - 1, 0, sdf->countZ - 2 );
	int maxX = b3ClampInt( (int)floorf( q2.x ), 0, sdf->countX - 2 );
	int maxY = b3ClampInt( (int)floorf( q2.y ), 0, sdf->countY - 2 );
	int maxZ = b3ClampInt( (int)floorf( q2.z ), 0, sdf->countZ - 2 );
	int cellsX = sdf->countX - 1;
	int cellsY = sdf->countY - 1;

	for ( int z = minZ; z <= maxZ; ++z )
	{
		for ( int y = minY; y <= maxY; ++y )
		{
			for ( int x = minX; x <= maxX; ++x )
			{
				b3SDFBuildTriangle triangles[12];
				int count = b3BuildSDFCellUnique( sdf, x, y, z, triangles );
				int cellIndex = x + cellsX * ( y + cellsY * z );
				for ( int i = 0; i < count; ++i )
				{
					b3Vec3 a = triangles[i].vertices[0];
					b3Vec3 b = triangles[i].vertices[1];
					b3Vec3 c = triangles[i].vertices[2];
					b3AABB triangleBounds = { b3Min( a, b3Min( b, c ) ), b3Max( a, b3Max( b, c ) ) };
					if ( b3AABB_Overlaps( bounds, triangleBounds ) && fcn( a, b, c, 12 * cellIndex + i, context ) == false )
					{
						return;
					}
				}
			}
		}
	}
}

int b3CollideMoverAndSDF( b3PlaneResult* planes, int capacity, const b3SDFData* shape, const b3Capsule* mover )
{
	if ( capacity == 0 )
	{
		return 0;
	}

	b3SDFMoverContext context = {
		.planes = planes,
		.capacity = capacity,
		.input =
			{
				.proxyB = { &mover->center1, 2, 0.0f },
				.transform = b3Transform_identity,
				.useRadii = false,
			},
		.radius = mover->radius,
	};
	b3AABB bounds = b3MakeAABB( &mover->center1, 2, mover->radius );
	b3QuerySDF( shape, bounds, b3SDFMoverFcn, &context );
	return context.count;
}
