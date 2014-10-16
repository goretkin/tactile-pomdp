/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/

#include <Box2D/Dynamics/b2Fixture.h>
#include <Box2D/Dynamics/Contacts/b2Contact.h>
#include <Box2D/Dynamics/b2World.h>
#include <Box2D/Collision/Shapes/b2CircleShape.h>
#include <Box2D/Collision/Shapes/b2EdgeShape.h>
#include <Box2D/Collision/Shapes/b2PolygonShape.h>
#include <Box2D/Collision/Shapes/b2ChainShape.h>
#include <Box2D/Collision/b2BroadPhase.h>
#include <Box2D/Collision/b2Collision.h>
#include <Box2D/Common/b2BlockAllocator.h>

b2Fixture::b2Fixture()
{
	m_userData = NULL;
	m_body = NULL;
	m_next = NULL;
	m_proxies = NULL;
	m_proxyCount = 0;
	m_shape = NULL;
	m_density = 0.0f;
}


bool inside(b2Vec2 cp1, b2Vec2 cp2, b2Vec2 p) {
      return (cp2.x-cp1.x)*(p.y-cp1.y) > (cp2.y-cp1.y)*(p.x-cp1.x);
}
  
b2Vec2 intersection(b2Vec2 cp1, b2Vec2 cp2, b2Vec2 s, b2Vec2 e) {
  b2Vec2 dc( cp1.x - cp2.x, cp1.y - cp2.y );
  b2Vec2 dp( s.x - e.x, s.y - e.y );
  float n1 = cp1.x * cp2.y - cp1.y * cp2.x;
  float n2 = s.x * e.y - s.y * e.x;
  float n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
  return b2Vec2( (n1*dp.x - n2*dc.x) * n3, (n1*dp.y - n2*dc.y) * n3);
}
 

//http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#JavaScript
//Note that this only works when fB is a convex polygon, but we know all 
//fixtures in Box2D are convex, so that will not be a problem
bool b2Fixture::findIntersectionOfFixtures(b2Fixture* fA, b2Fixture* fB, std::vector<b2Vec2>& outputVertices)
{
  //currently this only handles polygon vs polygon
  if ( fA->GetShape()->GetType() != b2Shape::e_polygon ||
       fB->GetShape()->GetType() != b2Shape::e_polygon )
      return false;

  b2PolygonShape* polyA = (b2PolygonShape*)fA->GetShape();
  b2PolygonShape* polyB = (b2PolygonShape*)fB->GetShape();

  //fill subject polygon from fixtureA polygon
  for (int i = 0; i < polyA->GetVertexCount(); i++)
      outputVertices.push_back( fA->GetBody()->GetWorldPoint( polyA->GetVertex(i) ) );

  //fill clip polygon from fixtureB polygon
  std::vector<b2Vec2> clipPolygon;
  for (int i = 0; i < polyB->GetVertexCount(); i++)
      clipPolygon.push_back( fB->GetBody()->GetWorldPoint( polyB->GetVertex(i) ) );

  b2Vec2 cp1 = clipPolygon[clipPolygon.size()-1];
  for (int j = 0; j < clipPolygon.size(); j++) {
      b2Vec2 cp2 = clipPolygon[j];
      if ( outputVertices.empty() )
          return false;
      std::vector<b2Vec2> inputList = outputVertices;
      outputVertices.clear();
      b2Vec2 s = inputList[inputList.size() - 1]; //last on the input list
      for (int i = 0; i < inputList.size(); i++) {
          b2Vec2 e = inputList[i];
          if (inside(cp1, cp2, e)) {
              if (!inside(cp1, cp2, s)) {
                  outputVertices.push_back( intersection(cp1, cp2, s, e) );
              }
              outputVertices.push_back(e);
          }
          else if (inside(cp1, cp2, s)) {
              outputVertices.push_back( intersection(cp1, cp2, s, e) );
          }
          s = e;
      }
      cp1 = cp2;
  }

  return !outputVertices.empty();
}

//Taken from b2PolygonShape.cpp
b2Vec2 b2Fixture::ComputeCentroid(std::vector<b2Vec2> vs, float& area)
{
  int count = (int)vs.size();
  b2Assert(count >= 3);

  b2Vec2 c;
  c.Set(0.0f, 0.0f);
  area = 0.0f;

  // pRef is the reference point for forming triangles.
  // Its location doesnt change the result (except for rounding error).
  b2Vec2 pRef(0.0f, 0.0f);

  const float32 inv3 = 1.0f / 3.0f;

  for (int32 i = 0; i < count; ++i)
  {
      // Triangle vertices.
      b2Vec2 p1 = pRef;
      b2Vec2 p2 = vs[i];
      b2Vec2 p3 = i + 1 < count ? vs[i+1] : vs[0];

      b2Vec2 e1 = p2 - p1;
      b2Vec2 e2 = p3 - p1;

      float32 D = b2Cross(e1, e2);

      float32 triangleArea = 0.5f * D;
      area += triangleArea;

      // Area weighted centroid
      c += triangleArea * inv3 * (p1 + p2 + p3);
  }

  // Centroid
  if (area > b2_epsilon)
      c *= 1.0f / area;
  else
      area = 0;
  return c;
}

b2Vec2 b2Fixture::CenterOfFixtureIntersection(b2Fixture* fA, b2Fixture* fB)
{
	std::vector<b2Vec2> r;
	b2Fixture::findIntersectionOfFixtures(fA,fB,r);
	float area;
	//if r.size()>=3
	return ComputeCentroid(r,area);
}

/*
bool b2Fixture::CenterOfFixtureIntersection(b2Fixture* fA, b2Fixture* fB, b2Vec2& centerOut)
{
	std::vector<b2Vec2> r;
	b2Fixture::findIntersectionOfFixtures(fA,fB,r);
	float area;
	if (r.size()>=3)
	{
		centerOut= ComputeCentroid(r,area);
		return true;
	}
	else
	{
		return false;
	}
	
}
*/

void b2Fixture::Create(b2BlockAllocator* allocator, b2Body* body, const b2FixtureDef* def)
{
	m_userData = def->userData;
	m_friction = def->friction;
	m_restitution = def->restitution;

	m_body = body;
	m_next = NULL;

	m_filter = def->filter;

	m_isSensor = def->isSensor;

	m_shape = def->shape->Clone(allocator);

	// Reserve proxy space
	int32 childCount = m_shape->GetChildCount();
	m_proxies = (b2FixtureProxy*)allocator->Allocate(childCount * sizeof(b2FixtureProxy));
	for (int32 i = 0; i < childCount; ++i)
	{
		m_proxies[i].fixture = NULL;
		m_proxies[i].proxyId = b2BroadPhase::e_nullProxy;
	}
	m_proxyCount = 0;

	m_density = def->density;
}

void b2Fixture::Destroy(b2BlockAllocator* allocator)
{
	// The proxies must be destroyed before calling this.
	b2Assert(m_proxyCount == 0);

	// Free the proxy array.
	int32 childCount = m_shape->GetChildCount();
	allocator->Free(m_proxies, childCount * sizeof(b2FixtureProxy));
	m_proxies = NULL;

	// Free the child shape.
	switch (m_shape->m_type)
	{
	case b2Shape::e_circle:
		{
			b2CircleShape* s = (b2CircleShape*)m_shape;
			s->~b2CircleShape();
			allocator->Free(s, sizeof(b2CircleShape));
		}
		break;

	case b2Shape::e_edge:
		{
			b2EdgeShape* s = (b2EdgeShape*)m_shape;
			s->~b2EdgeShape();
			allocator->Free(s, sizeof(b2EdgeShape));
		}
		break;

	case b2Shape::e_polygon:
		{
			b2PolygonShape* s = (b2PolygonShape*)m_shape;
			s->~b2PolygonShape();
			allocator->Free(s, sizeof(b2PolygonShape));
		}
		break;

	case b2Shape::e_chain:
		{
			b2ChainShape* s = (b2ChainShape*)m_shape;
			s->~b2ChainShape();
			allocator->Free(s, sizeof(b2ChainShape));
		}
		break;

	default:
		b2Assert(false);
		break;
	}

	m_shape = NULL;
}

void b2Fixture::CreateProxies(b2BroadPhase* broadPhase, const b2Transform& xf)
{
	b2Assert(m_proxyCount == 0);

	// Create proxies in the broad-phase.
	m_proxyCount = m_shape->GetChildCount();

	for (int32 i = 0; i < m_proxyCount; ++i)
	{
		b2FixtureProxy* proxy = m_proxies + i;
		m_shape->ComputeAABB(&proxy->aabb, xf, i);
		proxy->proxyId = broadPhase->CreateProxy(proxy->aabb, proxy);
		proxy->fixture = this;
		proxy->childIndex = i;
	}
}

void b2Fixture::DestroyProxies(b2BroadPhase* broadPhase)
{
	// Destroy proxies in the broad-phase.
	for (int32 i = 0; i < m_proxyCount; ++i)
	{
		b2FixtureProxy* proxy = m_proxies + i;
		broadPhase->DestroyProxy(proxy->proxyId);
		proxy->proxyId = b2BroadPhase::e_nullProxy;
	}

	m_proxyCount = 0;
}

void b2Fixture::Synchronize(b2BroadPhase* broadPhase, const b2Transform& transform1, const b2Transform& transform2)
{
	if (m_proxyCount == 0)
	{	
		return;
	}

	for (int32 i = 0; i < m_proxyCount; ++i)
	{
		b2FixtureProxy* proxy = m_proxies + i;

		// Compute an AABB that covers the swept shape (may miss some rotation effect).
		b2AABB aabb1, aabb2;
		m_shape->ComputeAABB(&aabb1, transform1, proxy->childIndex);
		m_shape->ComputeAABB(&aabb2, transform2, proxy->childIndex);
	
		proxy->aabb.Combine(aabb1, aabb2);

		b2Vec2 displacement = transform2.p - transform1.p;

		broadPhase->MoveProxy(proxy->proxyId, proxy->aabb, displacement);
	}
}

void b2Fixture::SetFilterData(const b2Filter& filter)
{
	m_filter = filter;

	Refilter();
}

void b2Fixture::Refilter()
{
	if (m_body == NULL)
	{
		return;
	}

	// Flag associated contacts for filtering.
	b2ContactEdge* edge = m_body->GetContactList();
	while (edge)
	{
		b2Contact* contact = edge->contact;
		b2Fixture* fixtureA = contact->GetFixtureA();
		b2Fixture* fixtureB = contact->GetFixtureB();
		if (fixtureA == this || fixtureB == this)
		{
			contact->FlagForFiltering();
		}

		edge = edge->next;
	}

	b2World* world = m_body->GetWorld();

	if (world == NULL)
	{
		return;
	}

	// Touch each proxy so that new pairs may be created
	b2BroadPhase* broadPhase = &world->m_contactManager.m_broadPhase;
	for (int32 i = 0; i < m_proxyCount; ++i)
	{
		broadPhase->TouchProxy(m_proxies[i].proxyId);
	}
}

void b2Fixture::SetSensor(bool sensor)
{
	if (sensor != m_isSensor)
	{
		m_body->SetAwake(true);
		m_isSensor = sensor;
	}
}

void b2Fixture::Dump(int32 bodyIndex)
{
	b2Log("    b2FixtureDef fd;\n");
	b2Log("    fd.friction = %.15lef;\n", m_friction);
	b2Log("    fd.restitution = %.15lef;\n", m_restitution);
	b2Log("    fd.density = %.15lef;\n", m_density);
	b2Log("    fd.isSensor = bool(%d);\n", m_isSensor);
	b2Log("    fd.filter.categoryBits = uint16(%d);\n", m_filter.categoryBits);
	b2Log("    fd.filter.maskBits = uint16(%d);\n", m_filter.maskBits);
	b2Log("    fd.filter.groupIndex = int16(%d);\n", m_filter.groupIndex);

	switch (m_shape->m_type)
	{
	case b2Shape::e_circle:
		{
			b2CircleShape* s = (b2CircleShape*)m_shape;
			b2Log("    b2CircleShape shape;\n");
			b2Log("    shape.m_radius = %.15lef;\n", s->m_radius);
			b2Log("    shape.m_p.Set(%.15lef, %.15lef);\n", s->m_p.x, s->m_p.y);
		}
		break;

	case b2Shape::e_edge:
		{
			b2EdgeShape* s = (b2EdgeShape*)m_shape;
			b2Log("    b2EdgeShape shape;\n");
			b2Log("    shape.m_radius = %.15lef;\n", s->m_radius);
			b2Log("    shape.m_vertex0.Set(%.15lef, %.15lef);\n", s->m_vertex0.x, s->m_vertex0.y);
			b2Log("    shape.m_vertex1.Set(%.15lef, %.15lef);\n", s->m_vertex1.x, s->m_vertex1.y);
			b2Log("    shape.m_vertex2.Set(%.15lef, %.15lef);\n", s->m_vertex2.x, s->m_vertex2.y);
			b2Log("    shape.m_vertex3.Set(%.15lef, %.15lef);\n", s->m_vertex3.x, s->m_vertex3.y);
			b2Log("    shape.m_hasVertex0 = bool(%d);\n", s->m_hasVertex0);
			b2Log("    shape.m_hasVertex3 = bool(%d);\n", s->m_hasVertex3);
		}
		break;

	case b2Shape::e_polygon:
		{
			b2PolygonShape* s = (b2PolygonShape*)m_shape;
			b2Log("    b2PolygonShape shape;\n");
			b2Log("    b2Vec2 vs[%d];\n", b2_maxPolygonVertices);
			for (int32 i = 0; i < s->m_count; ++i)
			{
				b2Log("    vs[%d].Set(%.15lef, %.15lef);\n", i, s->m_vertices[i].x, s->m_vertices[i].y);
			}
			b2Log("    shape.Set(vs, %d);\n", s->m_count);
		}
		break;

	case b2Shape::e_chain:
		{
			b2ChainShape* s = (b2ChainShape*)m_shape;
			b2Log("    b2ChainShape shape;\n");
			b2Log("    b2Vec2 vs[%d];\n", s->m_count);
			for (int32 i = 0; i < s->m_count; ++i)
			{
				b2Log("    vs[%d].Set(%.15lef, %.15lef);\n", i, s->m_vertices[i].x, s->m_vertices[i].y);
			}
			b2Log("    shape.CreateChain(vs, %d);\n", s->m_count);
			b2Log("    shape.m_prevVertex.Set(%.15lef, %.15lef);\n", s->m_prevVertex.x, s->m_prevVertex.y);
			b2Log("    shape.m_nextVertex.Set(%.15lef, %.15lef);\n", s->m_nextVertex.x, s->m_nextVertex.y);
			b2Log("    shape.m_hasPrevVertex = bool(%d);\n", s->m_hasPrevVertex);
			b2Log("    shape.m_hasNextVertex = bool(%d);\n", s->m_hasNextVertex);
		}
		break;

	default:
		return;
	}

	b2Log("\n");
	b2Log("    fd.shape = &shape;\n");
	b2Log("\n");
	b2Log("    bodies[%d]->CreateFixture(&fd);\n", bodyIndex);
}
