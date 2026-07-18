// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

typedef struct SDFQueryContext
{
	int count;
	bool nonDegenerate;
	b3Vec3 normalSum;
} SDFQueryContext;

static bool CountSDFTriangle( b3Vec3 a, b3Vec3 b, b3Vec3 c, int triangleIndex, void* rawContext )
{
	(void)triangleIndex;
	SDFQueryContext* context = rawContext;
	b3Vec3 normal = b3Cross( b3Sub( b, a ), b3Sub( c, a ) );
	context->nonDegenerate = context->nonDegenerate && b3LengthSquared( normal ) > FLT_EPSILON * FLT_EPSILON;
	context->normalSum = b3Add( context->normalSum, normal );
	context->count += 1;
	return true;
}

static b3SDFData* MakeBoxSDF( void )
{
	const int count = 9;
	const float spacing = 0.5f;
	const float halfExtent = 0.75f;
	float* distances = (float*)malloc( (size_t)count * count * count * sizeof( float ) );

	for ( int z = 0; z < count; ++z )
	{
		for ( int y = 0; y < count; ++y )
		{
			for ( int x = 0; x < count; ++x )
			{
				b3Vec3 p = { -2.0f + spacing * x, -2.0f + spacing * y, -2.0f + spacing * z };
				b3Vec3 q = { fabsf( p.x ) - halfExtent, fabsf( p.y ) - halfExtent, fabsf( p.z ) - halfExtent };
				b3Vec3 outside = { b3MaxFloat( q.x, 0.0f ), b3MaxFloat( q.y, 0.0f ), b3MaxFloat( q.z, 0.0f ) };
				float outsideDistance = b3Length( outside );
				float insideDistance = b3MinFloat( b3MaxFloat( q.x, b3MaxFloat( q.y, q.z ) ), 0.0f );
				distances[x + count * ( y + count * z )] = outsideDistance + insideDistance;
			}
		}
	}

	b3SDFDef def = { 0 };
	def.distances = distances;
	def.origin = (b3Vec3){ -2.0f, -2.0f, -2.0f };
	def.spacing = (b3Vec3){ spacing, spacing, spacing };
	def.countX = count;
	def.countY = count;
	def.countZ = count;
	b3SDFData* sdf = b3CreateSDF( &def );
	free( distances );
	return sdf;
}

static int SDFCreateAndQuery( void )
{
	b3SDFData* sdf = MakeBoxSDF();
	ENSURE( sdf != NULL );
	ENSURE( sdf->version == B3_SDF_VERSION );
	ENSURE( sdf->byteCount < 4096 );
	SDFQueryContext query = { .nonDegenerate = true };
	b3QuerySDF( sdf, sdf->aabb, CountSDFTriangle, &query );
	ENSURE( query.count > 0 );
	ENSURE( query.nonDegenerate );
	ENSURE_SMALL( sdf->aabb.lowerBound.x + 1.0f, FLT_EPSILON );
	ENSURE_SMALL( sdf->aabb.upperBound.z - 1.0f, FLT_EPSILON );

	b3Vec3 inside = { 0.0f, 0.0f, 0.0f };
	b3ShapeProxy insideProxy = { &inside, 1, 0.0f };
	ENSURE( b3OverlapSDF( sdf, b3Transform_identity, &insideProxy ) );

	b3Vec3 outside = { 0.0f, 1.5f, 0.0f };
	b3ShapeProxy outsideProxy = { &outside, 1, 0.0f };
	ENSURE( b3OverlapSDF( sdf, b3Transform_identity, &outsideProxy ) == false );

	b3RayCastInput ray = { 0 };
	ray.origin = (b3Vec3){ 0.0f, 2.0f, 0.0f };
	ray.translation = (b3Vec3){ 0.0f, -4.0f, 0.0f };
	ray.maxFraction = 1.0f;
	b3CastOutput hit = b3RayCastSDF( sdf, &ray );
	ENSURE( hit.hit );
	ENSURE( hit.fraction > 0.0f && hit.fraction < 1.0f );

	b3Vec3 castPoint = { 0.0f, 2.0f, 0.0f };
	b3ShapeCastInput shapeCast = { 0 };
	shapeCast.proxy = (b3ShapeProxy){ &castPoint, 1, 0.25f };
	shapeCast.translation = (b3Vec3){ 0.0f, -4.0f, 0.0f };
	shapeCast.maxFraction = 1.0f;
	hit = b3ShapeCastSDF( sdf, &shapeCast );
	ENSURE( hit.hit );
	ENSURE( hit.fraction > 0.0f && hit.fraction < 1.0f );

	b3DestroySDF( sdf );
	return 0;
}

static int SDFExactZeroSamples( void )
{
	// A zero sample with both signs in the same cell used to emit repeated
	// intersections and zero-area triangles.
	float distances[8] = { 1.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
	b3SDFDef def = { 0 };
	def.distances = distances;
	def.spacing = b3Vec3_one;
	def.countX = 2;
	def.countY = 2;
	def.countZ = 2;

	b3SDFData* sdf = b3CreateSDF( &def );
	ENSURE( sdf != NULL );
	SDFQueryContext query = { .nonDegenerate = true };
	b3QuerySDF( sdf, sdf->aabb, CountSDFTriangle, &query );
	ENSURE( query.count > 0 );
	ENSURE( query.nonDegenerate );

	b3DestroySDF( sdf );

	// A zero-valued face shared by multiple tetrahedra must be emitted once.
	float faceDistances[8] = { 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 0.0f };
	def.distances = faceDistances;
	sdf = b3CreateSDF( &def );
	ENSURE( sdf != NULL );
	query = (SDFQueryContext){ .nonDegenerate = true };
	b3QuerySDF( sdf, sdf->aabb, CountSDFTriangle, &query );
	ENSURE( query.count == 1 );
	query = (SDFQueryContext){ .nonDegenerate = true };
	b3AABB faceBounds = { { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f } };
	b3QuerySDF( sdf, faceBounds, CountSDFTriangle, &query );
	ENSURE( query.count == 1 );
	b3DestroySDF( sdf );

	// A complete zero-valued face between positive and negative sample planes
	// must not be discarded in favor of the adjacent cell that has no crossing.
	float planeDistances[12];
	for ( int i = 0; i < 4; ++i )
	{
		planeDistances[i] = 1.0f;
		planeDistances[4 + i] = 0.0f;
		planeDistances[8 + i] = -1.0f;
	}
	def.distances = planeDistances;
	def.origin = (b3Vec3){ -1.0f, -1.0f, -1.0f };
	def.countZ = 3;
	sdf = b3CreateSDF( &def );
	ENSURE( sdf != NULL );
	query = (SDFQueryContext){ .nonDegenerate = true };
	b3QuerySDF( sdf, sdf->aabb, CountSDFTriangle, &query );
	ENSURE( query.count == 2 );
	ENSURE( query.nonDegenerate );
	ENSURE( query.normalSum.z < 0.0f );
	b3DestroySDF( sdf );
	return 0;
}

static int SDFStaticOnly( void )
{
	b3SDFData* sdf = MakeBoxSDF();
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId world = b3CreateWorld( &worldDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();

	b3BodyDef dynamicDef = b3DefaultBodyDef();
	dynamicDef.type = b3_dynamicBody;
	b3BodyId dynamicBody = b3CreateBody( world, &dynamicDef );
	b3ShapeId rejected = b3CreateSDFShape( dynamicBody, &shapeDef, sdf );
	ENSURE( rejected.index1 == 0 );

	b3BodyDef staticDef = b3DefaultBodyDef();
	staticDef.type = b3_staticBody;
	b3BodyId staticBody = b3CreateBody( world, &staticDef );
	b3ShapeId shape = b3CreateSDFShape( staticBody, &shapeDef, sdf );
	ENSURE( b3Shape_IsValid( shape ) );
	ENSURE( b3Shape_GetType( shape ) == b3_sdfShape );
	ENSURE( b3Shape_GetSDF( shape ) == sdf );

	b3BodyDef fallingDef = b3DefaultBodyDef();
	fallingDef.type = b3_dynamicBody;
	fallingDef.position = (b3Vec3){ 0.0f, 1.4f, 0.0f };
	b3BodyId fallingBody = b3CreateBody( world, &fallingDef );
	b3ShapeDef fallingShapeDef = b3DefaultShapeDef();
	fallingShapeDef.density = 1.0f;
	b3Sphere sphere = { b3Vec3_zero, 0.25f };
	b3CreateSphereShape( fallingBody, &fallingShapeDef, &sphere );
	for ( int i = 0; i < 60; ++i )
	{
		b3World_Step( world, 1.0f / 60.0f, 4 );
	}
	ENSURE( b3Body_GetPosition( fallingBody ).y > 0.8f );

	b3DestroyWorld( world );
	b3DestroySDF( sdf );
	return 0;
}

int SDFTest( void )
{
	RUN_SUBTEST( SDFCreateAndQuery );
	RUN_SUBTEST( SDFExactZeroSamples );
	RUN_SUBTEST( SDFStaticOnly );
	return 0;
}
