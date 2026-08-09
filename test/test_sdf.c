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

static b3SDFStorageValue EncodeSDFDistance( float distance, float distanceScale )
{
	float encoded = B3_SDF_ISOVALUE - distance / distanceScale;
	int value = (int)roundf( b3ClampFloat( encoded, 0.0f, 255.0f ) );
	value = distance < 0.0f ? b3MaxInt( value, 128 ) : b3MinInt( value, 127 );
	return (b3SDFStorageValue)value;
}

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
	const float distanceScale = 0.02f;
	b3SDFStorageValue* samples = malloc( (size_t)count * count * count * sizeof( b3SDFStorageValue ) );

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
				samples[x + count * ( y + count * z )] = EncodeSDFDistance( outsideDistance + insideDistance, distanceScale );
			}
		}
	}

	b3SDFDef def = { 0 };
	def.samples = samples;
	def.distanceScale = distanceScale;
	def.origin = (b3Vec3){ -2.0f, -2.0f, -2.0f };
	def.spacing = (b3Vec3){ spacing, spacing, spacing };
	def.countX = count;
	def.countY = count;
	def.countZ = count;
	b3SDFData* sdf = b3CreateSDF( &def );
	free( samples );
	return sdf;
}

static int SDFCreateAndQuery( void )
{
	b3SDFData* sdf = MakeBoxSDF();
	ENSURE( sdf != NULL );
	ENSURE( sdf->version == B3_SDF_VERSION );
	ENSURE( sdf->byteCount < 4096 );
	ENSURE( b3GetSDFSamples( sdf ) != NULL );
	ENSURE( sizeof( b3SDFStorageValue ) == 1 );
	ENSURE( sdf->byteCount < 1024 );
	ENSURE_SMALL( b3GetSDFDistanceScale( sdf ) - 0.02f, FLT_EPSILON );
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

static int SDFHalfIntegerIsovalue( void )
{
	// Integer samples cannot equal the half-integer isovalue, avoiding zero-area
	// triangles caused by exact-zero lattice samples.
	b3SDFStorageValue samples[8] = { 126, 127, 128, 128, 128, 128, 128, 128 };
	b3SDFDef def = { 0 };
	def.samples = samples;
	def.distanceScale = 1.0f;
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
	return 0;
}

static int SDFUpdate( void )
{
	b3SDFData* sdf = MakeBoxSDF();
	ENSURE( sdf != NULL );
	uint32_t oldHash = sdf->hash;

	b3SDFStorageValue samples[9 * 9 * 9];
	for ( int i = 0; i < ARRAY_COUNT( samples ); ++i )
	{
		samples[i] = 127;
	}

	b3SDFDef def = {
		.samples = samples,
		.distanceScale = 0.02f,
		.origin = { 3.0f, 4.0f, 5.0f },
		.spacing = { 0.5f, 0.5f, 0.5f },
		.countX = 9,
		.countY = 9,
		.countZ = 9,
	};
	ENSURE( b3UpdateSDF( sdf, &def ) );
	ENSURE( sdf->hash != oldHash );
	ENSURE_SMALL( sdf->aabb.lowerBound.x - def.origin.x, FLT_EPSILON );
	ENSURE_SMALL( sdf->aabb.upperBound.z - def.origin.z, FLT_EPSILON );

	SDFQueryContext query = { .nonDegenerate = true };
	b3QuerySDF( sdf, sdf->aabb, CountSDFTriangle, &query );
	ENSURE( query.count == 0 );

	def.countZ = 8;
	ENSURE( b3UpdateSDF( sdf, &def ) == false );
	ENSURE( sdf->countZ == 9 );

	b3DestroySDF( sdf );
	return 0;
}

static int SDFStorage( void )
{
	b3SDFStorageValue samples[8] = { 0, 31, 127, 128, 129, 192, 224, 255 };
	b3SDFDef def = {
		.samples = samples,
		.distanceScale = 0.25f,
		.spacing = { 1.0f, 1.0f, 1.0f },
		.countX = 2,
		.countY = 2,
		.countZ = 2,
	};
	b3SDFData* sdf = b3CreateSDF( &def );
	ENSURE( sdf != NULL );

	const b3SDFStorageValue* stored = b3GetSDFSamples( sdf );
	for ( int i = 0; i < ARRAY_COUNT( samples ); ++i )
	{
		ENSURE( stored[i] == samples[i] );
	}

	b3SDFStorageValue updatedSamples[8] = { 255, 224, 192, 129, 126, 63, 31, 0 };
	def.samples = updatedSamples;
	def.distanceScale = 0.125f;
	ENSURE( b3UpdateSDF( sdf, &def ) );
	ENSURE_SMALL( b3GetSDFDistanceScale( sdf ) - 0.125f, FLT_EPSILON );
	stored = b3GetSDFSamples( sdf );
	for ( int i = 0; i < ARRAY_COUNT( updatedSamples ); ++i )
	{
		ENSURE( stored[i] == updatedSamples[i] );
	}

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

	b3SDFStorageValue emptySamples[9 * 9 * 9];
	for ( int i = 0; i < ARRAY_COUNT( emptySamples ); ++i )
	{
		emptySamples[i] = 127;
	}
	b3SDFDef emptyDef = {
		.samples = emptySamples,
		.distanceScale = 0.02f,
		.origin = { -2.0f, -2.0f, -2.0f },
		.spacing = { 0.5f, 0.5f, 0.5f },
		.countX = 9,
		.countY = 9,
		.countZ = 9,
	};
	ENSURE( b3UpdateSDF( sdf, &emptyDef ) );
	b3Shape_SetSDF( shape, sdf );
	for ( int i = 0; i < 120; ++i )
	{
		b3World_Step( world, 1.0f / 60.0f, 4 );
	}
	ENSURE( b3Body_GetPosition( fallingBody ).y < -1.0f );

	b3DestroyWorld( world );
	b3DestroySDF( sdf );
	return 0;
}

int SDFTest( void )
{
	RUN_SUBTEST( SDFCreateAndQuery );
	RUN_SUBTEST( SDFHalfIntegerIsovalue );
	RUN_SUBTEST( SDFUpdate );
	RUN_SUBTEST( SDFStorage );
	RUN_SUBTEST( SDFStaticOnly );
	return 0;
}
