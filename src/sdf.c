// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "core.h"
#include "shape.h"

#include "box3d/collision.h"
#include "box3d/constants.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
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

static int b3CompareSDFTriangles( const void* contextA, const void* contextB )
{
	const b3SDFBuildTriangle* a = contextA;
	const b3SDFBuildTriangle* b = contextB;
	b3Vec3 keyA[3], keyB[3];
	b3CanonicalizeSDFTriangle( keyA, a );
	b3CanonicalizeSDFTriangle( keyB, b );
	for ( int i = 0; i < 3; ++i )
	{
		int comparison = b3CompareSDFPoints( keyA + i, keyB + i );
		if ( comparison != 0 )
		{
			return comparison;
		}
	}

	// Deterministically order duplicate geometry that uses a different cyclic
	// vertex order. The first triangle retained below is then platform independent.
	for ( int i = 0; i < 3; ++i )
	{
		int comparison = b3CompareSDFPoints( a->vertices + i, b->vertices + i );
		if ( comparison != 0 )
		{
			return comparison;
		}
	}
	return 0;
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
		{ 0, 5, 1, 6 }, { 0, 1, 2, 6 }, { 0, 2, 3, 6 },
		{ 0, 3, 7, 6 }, { 0, 7, 4, 6 }, { 0, 4, 5, 6 },
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

		b3Vec3 normalDirection = b3Normalize( b3Sub( positiveCenter, center ) );
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
		{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 },
		{ 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 },
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
		{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 },
		{ 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 },
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
	if ( sampleCount > (size_t)INT_MAX || cellCount > (size_t)INT_MAX / 36u )
	{
		return NULL;
	}

	for ( size_t i = 0; i < sampleCount; ++i )
	{
		if ( b3IsValidFloat( data->distances[i] ) == false )
		{
			return NULL;
		}
	}

	b3Vec3 gridExtent = b3Mul( data->spacing,
		(b3Vec3){ (float)( data->countX - 1 ), (float)( data->countY - 1 ), (float)( data->countZ - 1 ) } );
	b3Vec3 gridUpper = b3Add( data->origin, gridExtent );
	if ( b3IsValidVec3( gridExtent ) == false || b3IsValidVec3( gridUpper ) == false )
	{
		return NULL;
	}

	int buildTriangleCount = 0;
	b3SDFDef source = *data;
	source.distances = data->distances;
	for ( int z = 0; z < data->countZ - 1; ++z )
	{
		for ( int y = 0; y < data->countY - 1; ++y )
		{
			for ( int x = 0; x < data->countX - 1; ++x )
			{
				buildTriangleCount += b3SDFBuildCell( &source, x, y, z, NULL );
			}
		}
	}

	b3MeshData* surfaceMesh = NULL;
	if ( buildTriangleCount > 0 )
	{
		int buildTriangleCapacity = buildTriangleCount;
		b3SDFBuildTriangle* buildTriangles = b3Alloc( (size_t)buildTriangleCapacity * sizeof( b3SDFBuildTriangle ) );

		int triangleIndex = 0;
		for ( int z = 0; z < data->countZ - 1; ++z )
		{
			for ( int y = 0; y < data->countY - 1; ++y )
			{
				for ( int x = 0; x < data->countX - 1; ++x )
				{
					triangleIndex += b3SDFBuildCell( &source, x, y, z, buildTriangles + triangleIndex );
				}
			}
		}
		B3_ASSERT( triangleIndex == buildTriangleCount );

		qsort( buildTriangles, (size_t)buildTriangleCount, sizeof( b3SDFBuildTriangle ), b3CompareSDFTriangles );
		int uniqueTriangleCount = 1;
		for ( int i = 1; i < buildTriangleCount; ++i )
		{
			if ( b3SameSDFTriangleGeometry( buildTriangles + uniqueTriangleCount - 1, buildTriangles + i ) == false )
			{
				buildTriangles[uniqueTriangleCount] = buildTriangles[i];
				uniqueTriangleCount += 1;
			}
		}
		buildTriangleCount = uniqueTriangleCount;

		int vertexCount = 3 * buildTriangleCount;
		int32_t* indices = b3Alloc( (size_t)vertexCount * sizeof( int32_t ) );
		for ( int i = 0; i < vertexCount; ++i )
		{
			indices[i] = i;
		}

		float minSpacing = b3MinFloat( data->spacing.x, b3MinFloat( data->spacing.y, data->spacing.z ) );
		b3MeshDef meshDef = { 0 };
		meshDef.vertices = buildTriangles[0].vertices;
		meshDef.indices = indices;
		meshDef.weldTolerance = 0.0001f * minSpacing;
		meshDef.vertexCount = vertexCount;
		meshDef.triangleCount = buildTriangleCount;
		meshDef.weldVertices = true;
		meshDef.useMedianSplit = true;
		meshDef.identifyEdges = true;
		surfaceMesh = b3CreateMesh( &meshDef, NULL, 0 );

		b3Free( indices, (size_t)vertexCount * sizeof( int32_t ) );
		b3Free( buildTriangles, (size_t)buildTriangleCapacity * sizeof( b3SDFBuildTriangle ) );
		if ( surfaceMesh == NULL )
		{
			return NULL;
		}
	}

	size_t byteCount = b3AlignUp8( sizeof( b3SDFData ) );
	int distancesOffset = (int)byteCount;
	byteCount += b3AlignUp8( sampleCount * sizeof( float ) );
	int meshOffset = surfaceMesh != NULL ? (int)byteCount : 0;
	if ( surfaceMesh != NULL )
	{
		byteCount += b3AlignUp8( (size_t)surfaceMesh->byteCount );
	}

	if ( byteCount > (size_t)INT_MAX )
	{
		if ( surfaceMesh != NULL )
		{
			b3DestroyMesh( surfaceMesh );
		}
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
	sdf->triangleCount = surfaceMesh != NULL ? surfaceMesh->triangleCount : 0;
	sdf->distancesOffset = distancesOffset;
	sdf->meshOffset = meshOffset;
	sdf->aabb.lowerBound = data->origin;
	sdf->aabb.upperBound = gridUpper;

	float* distances = (float*)( (intptr_t)sdf + distancesOffset );
	memcpy( distances, data->distances, sampleCount * sizeof( float ) );

	if ( surfaceMesh != NULL )
	{
		memcpy( (uint8_t*)sdf + meshOffset, surfaceMesh, (size_t)surfaceMesh->byteCount );
		b3DestroyMesh( surfaceMesh );
	}

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

	const b3MeshData* meshData = b3GetSDFMesh( shape );
	if ( meshData == NULL )
	{
		return false;
	}

	b3Mesh mesh = { meshData, b3Vec3_one };
	return b3OverlapMesh( &mesh, b3Transform_identity, &localProxy );
}

b3CastOutput b3ShapeCastSDF( const b3SDFData* shape, const b3ShapeCastInput* input )
{
	const b3MeshData* meshData = b3GetSDFMesh( shape );
	if ( meshData == NULL )
	{
		return (b3CastOutput){ .fraction = input->maxFraction, .triangleIndex = B3_NULL_INDEX };
	}

	b3Mesh mesh = { meshData, b3Vec3_one };
	return b3ShapeCastMesh( &mesh, input );
}

b3CastOutput b3RayCastSDF( const b3SDFData* shape, const b3RayCastInput* input )
{
	const b3MeshData* meshData = b3GetSDFMesh( shape );
	if ( meshData == NULL )
	{
		return (b3CastOutput){ .fraction = input->maxFraction, .triangleIndex = B3_NULL_INDEX };
	}

	b3Mesh mesh = { meshData, b3Vec3_one };
	return b3RayCastMesh( &mesh, input );
}

void b3QuerySDF( const b3SDFData* sdf, b3AABB bounds, b3MeshQueryFcn* fcn, void* context )
{
	const b3MeshData* meshData = b3GetSDFMesh( sdf );
	if ( meshData == NULL )
	{
		return;
	}

	b3Mesh mesh = { meshData, b3Vec3_one };
	b3QueryMesh( &mesh, bounds, fcn, context );
}

int b3CollideMoverAndSDF( b3PlaneResult* planes, int capacity, const b3SDFData* shape, const b3Capsule* mover )
{
	if ( capacity == 0 )
	{
		return 0;
	}

	const b3MeshData* meshData = b3GetSDFMesh( shape );
	if ( meshData == NULL )
	{
		return 0;
	}

	b3Mesh mesh = { meshData, b3Vec3_one };
	return b3CollideMoverAndMesh( planes, capacity, &mesh, mover );
}
