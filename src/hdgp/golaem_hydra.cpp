/*
 * TODO:
 * - FBX support?
 * - materials
 * - UVs
 * - LODs
 * - motion blur
 * - optimization: cache and reuse FileMeshAdapter instances (only
 *   points and normals change for the same LOD)
 */
#include "pxr/imaging/hdGp/generativeProceduralPlugin.h"
#include "pxr/imaging/hdGp/generativeProceduralPluginRegistry.h"

#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/imaging/hd/sceneGlobalsSchema.h"
#include "pxr/imaging/hd/tokens.h"

#include "pxr/usd/usdGeom/tokens.h"

#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/base/tf/staticTokens.h"

#include <glmCrowdGcgCharacter.h>
#include <glmGolaemCharacter.h>
#include <glmIdsFilter.h>
#include <glmSimulationCacheFactory.h>
#include <glmSimulationCacheFactorySimulation.h>
#include "glmUSD.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using glm::GlmString;
using glm::GolaemCharacter;
using glm::crowdio::CachedSimulation;
using glm::crowdio::GlmFrameData;
using glm::crowdio::GlmSimulationData;
using glm::crowdio::SimulationCacheFactory;

namespace
{

TF_DEFINE_PRIVATE_TOKENS(
    golaemTokens,
    (crowdFields)
    (cacheName)
    (cacheDir)
    (characterFiles)
    (entityIds)
    (geometryTag)
    (displayMode)
    (bbox)
    (mesh)
);

/*
 * Arguments (primvars) provided by the USD prim.
 */
struct Args
{
    Args()
        : entityIds("*"),
          geometryTag(0),
          displayMode(golaemTokens->mesh)
        {}

    VtTokenArray crowdFields;
    TfToken cacheName;
    TfToken cacheDir;
    VtTokenArray characterFiles;
    TfToken entityIds;
    short geometryTag;
    TfToken displayMode;
};

/*
 * Class which provides Hydra data sources wrapping the information
 * found in a GlmFileMesh: topology, vertices and normals.
 */
class FileMeshAdapter
{
    using IntArrayDS = HdRetainedTypedSampledDataSource<VtIntArray>;
    using Vec3fArrayDS = HdRetainedTypedSampledDataSource<VtArray<GfVec3f>>;

public:
    FileMeshAdapter(
        const glm::crowdio::GlmFileMesh& fileMesh,
        const glm::Array<glm::Vector3>& deformedVertices,
        const glm::Array<glm::Vector3>& deformedNormals)
    {
        // mesh topology data source (vertices)

        VtIntArray vertexCounts(fileMesh._polygonCount);
        for (int i = 0; i < fileMesh._polygonCount; ++i) {
            vertexCounts[i] = fileMesh._polygonsVertexCount[i];
        }

        VtIntArray vertexIndices(fileMesh._polygonsTotalVertexCount);
        for (int i = 0; i < fileMesh._polygonsTotalVertexCount; ++i) {
            vertexIndices[i] = fileMesh._polygonsVertexIndices[i];
        }

        _meshDataSource = HdMeshSchema::Builder()
            .SetTopology(
                HdMeshTopologySchema::Builder()
                .SetFaceVertexCounts(IntArrayDS::New(vertexCounts))
                .SetFaceVertexIndices(IntArrayDS::New(vertexIndices))
                .Build())
            .SetSubdivisionScheme(
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    UsdGeomTokens->none))
            .Build();

        VtVec3fArray vertices(fileMesh._vertexCount);
        for (int i = 0; i < fileMesh._vertexCount; ++i) {
            vertices[i].Set(deformedVertices[i].getFloatValues());
        }

        HdContainerDataSourceHandle vertexDataSource =
            HdPrimvarSchema::Builder()
            .SetPrimvarValue(Vec3fArrayDS::New(vertices))
            .SetInterpolation(
                HdPrimvarSchema::BuildInterpolationDataSource(
                    HdPrimvarSchemaTokens->vertex))
            .SetRole(
                HdPrimvarSchema::BuildRoleDataSource(
                    HdPrimvarSchemaTokens->point))
            .Build();

        // normal data source, if the mesh contains normals

        HdContainerDataSourceHandle normalDataSource;

        if (fileMesh._normalCount > 0) {
            VtVec3fArray normals(fileMesh._normalCount);
            for (int i = 0; i < fileMesh._normalCount; ++i) {
                normals[i].Set(deformedNormals[i].getFloatValues());
            }

            HdPrimvarSchema::Builder normalBuilder;

            // normals may or may not be indexed

            if (fileMesh._normalMode == glm::crowdio::GLM_NORMAL_PER_POLYGON_VERTEX_INDEXED) {
                VtIntArray indices(fileMesh._polygonsTotalVertexCount);
                for (int i = 0; i < fileMesh._polygonsTotalVertexCount; ++i) {
                    indices[i] = fileMesh._polygonsNormalIndices[i];
                }
                normalBuilder.SetIndexedPrimvarValue(Vec3fArrayDS::New(normals));
                normalBuilder.SetIndices(IntArrayDS::New(indices));
            } else {
                normalBuilder.SetPrimvarValue(Vec3fArrayDS::New(normals));
            }

            // normals may or may not be shared by polygons using the
            // same vertices

            if (fileMesh._normalMode == glm::crowdio::GLM_NORMAL_PER_CONTROL_POINT) {
                normalBuilder.SetInterpolation(
                    HdPrimvarSchema::BuildInterpolationDataSource(
                        HdPrimvarSchemaTokens->vertex));
            } else {
                normalBuilder.SetInterpolation(
                    HdPrimvarSchema::BuildInterpolationDataSource(
                        HdPrimvarSchemaTokens->faceVarying));
            }

            normalBuilder.SetRole(
                HdPrimvarSchema::BuildRoleDataSource(
                    HdPrimvarSchemaTokens->normal));

            normalDataSource = normalBuilder.Build();
        }

        // the final primvars data source contains both the vertices
        // and the normals

        _primvarsDataSource = HdRetainedContainerDataSource::New(
            HdPrimvarsSchemaTokens->points,
            vertexDataSource,
            HdPrimvarsSchemaTokens->normals,
            normalDataSource);
    }

    HdContainerDataSourceHandle GetMeshDataSource() const
    {
        return _meshDataSource;
    }

    HdContainerDataSourceHandle GetPrimvarsDataSource() const
    {
        return _primvarsDataSource;
    }

private:
    HdContainerDataSourceHandle _meshDataSource;
    HdContainerDataSourceHandle _primvarsDataSource;
};

/*
 * Information needed by the renderer for each entity in bbox display
 * mode.
 */
struct BBoxEntityData
{
    GfVec3f extent;
    float scale;
    GfVec3f pos;
    GfQuatf quat;
};

/*
 * Information needed by the renderer for each entity in mesh display
 * mode.
 */
struct MeshEntityData
{
    int crowdFieldIndex;
    int entityIndex;
    std::vector<FileMeshAdapter> meshes;
};

/*
 * Creates a data source which returns the topology of a cube.
 */
HdContainerDataSourceHandle GetCubeMeshDataSource()
{
    static const VtIntArray faceVertexCounts =
        {4, 4, 4, 4, 4, 4};

    static const VtIntArray faceVertexIndices =
        {0, 1, 3, 2, 2, 3, 5, 4, 4, 5, 7, 6, 6, 7, 1, 0, 1,
            7, 5, 3, 6, 0, 2, 4};

    using _IntArrayDs =
        HdRetainedTypedSampledDataSource<VtIntArray>;

    static const _IntArrayDs::Handle fvcDs =
        _IntArrayDs::New(faceVertexCounts);

    static const _IntArrayDs::Handle fviDs =
        _IntArrayDs::New(faceVertexIndices);

    static const HdContainerDataSourceHandle meshDs =
        HdMeshSchema::Builder()
            .SetTopology(HdMeshTopologySchema::Builder()
                .SetFaceVertexCounts(fvcDs)
                .SetFaceVertexIndices(fviDs)
                .Build())
            .Build();

    return meshDs;
}

/*
 * Creates a data source which returns the vertices of a cube.
 */
HdContainerDataSourceHandle GetCubePrimvarsDataSource()
{
    static const VtArray<GfVec3f> points = {
        {-1.0f, -1.0f, 1.0f},
        {1.0f, -1.0f, 1.0f},
        {-1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
        {-1.0f, 1.0f, -1.0f},
        {1.0f, 1.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f},
        {1.0f, -1.0f, -1.0f}};

    using _PointArrayDs =
        HdRetainedTypedSampledDataSource<VtArray<GfVec3f>>;

    static const HdContainerDataSourceHandle primvarsDs =
        HdRetainedContainerDataSource::New(
            HdPrimvarsSchemaTokens->points,
            HdPrimvarSchema::Builder()
                .SetPrimvarValue(_PointArrayDs::New(points))
                .SetInterpolation(HdPrimvarSchema::
                    BuildInterpolationDataSource(
                        HdPrimvarSchemaTokens->vertex))
                .SetRole(HdPrimvarSchema::
                    BuildRoleDataSource(
                        HdPrimvarSchemaTokens->point))
                .Build()
        );

    return primvarsDs;
}

class GolaemProcedural: public HdGpGenerativeProcedural
{
public:
    GolaemProcedural(const SdfPath &proceduralPrimPath)
        : HdGpGenerativeProcedural(proceduralPrimPath)
    {
        glm::usdplugin::init();
        _factory = new SimulationCacheFactory();
        _updateCount = 0;
    }

    virtual ~GolaemProcedural()
    {
        std::cout << "deleting factory... " << std::flush;
        delete _factory;
        std::cout << "done" << std::endl;
        std::cout << "calling finish()... " << std::flush;
        glm::usdplugin::finish();
        std::cout << "done" << std::endl;
    }

    DependencyMap UpdateDependencies(
        const HdSceneIndexBaseRefPtr &inputScene) override
    {
        DependencyMap result;

        // call Update() when the current frame changes

        result[HdSceneGlobalsSchema::GetDefaultPrimPath()] = {
            HdSceneGlobalsSchema::GetCurrentFrameLocator()
        };

        return result;
    }

    ChildPrimTypeMap Update(
        const HdSceneIndexBaseRefPtr &inputScene,
        const ChildPrimTypeMap &previousResult,
        const DependencyMap &dirtiedDependencies,
        HdSceneIndexObserver::DirtiedPrimEntries *outputDirtiedPrims) override
    {
        // fetch arguments (primvars) the first time only (we assume
        // they never change), then (re)populate the scene

        if (_updateCount == 0) {
            _args = GetArgs(inputScene);
            InitCrowd(inputScene);
        }
        ++_updateCount;
        PopulateCrowd(inputScene);

        ChildPrimTypeMap result;
        SdfPath myPath = _GetProceduralPrimPath();
        char buffer[128];

        // bbox display mode

        if (_args.displayMode == golaemTokens->bbox) {

            // generate a prim for each entity in the crowd

            _childIndices.clear();

            for (int i = 0; i < _bboxEntities.size(); ++i) {
                sprintf(buffer, "c%d", i);
                SdfPath childPath = myPath.AppendChild(TfToken(buffer));
                result[childPath] = HdPrimTypeTokens->mesh;
                _childIndices[childPath] = i;

                // if the same path was generated by the previous call
                // to Update(), too, tell Hydra its xform may have
                // changed

                if (previousResult.size() > 0) {
                    outputDirtiedPrims->emplace_back(
                        childPath, HdXformSchema::GetDefaultLocator());
                }
            }
        }

        // mesh display mode

        else {

            // generate a prim for each mesh for each entity

            _childIndexPairs.clear();

            for (int i = 0; i < _meshEntities.size(); ++i) {
                const MeshEntityData& entity = _meshEntities[i];
                for (int j = 0; j < entity.meshes.size(); ++j) {

                    // including the crowd field, entity and mesh in
                    // the path enables us to tell Hydra that, if the
                    // same prim appears in two successive frames,
                    // only the primvars (points and normals) will
                    // have changed

                    sprintf(buffer, "c%d_%d_%d",
                            entity.crowdFieldIndex, entity.entityIndex, j);
                    SdfPath childPath = myPath.AppendChild(TfToken(buffer));
                    result[childPath] = HdPrimTypeTokens->mesh;
                    _childIndexPairs[childPath] = std::pair<int, int>(i, j);

                    if (previousResult.size() > 0) {
                        outputDirtiedPrims->emplace_back(
                            childPath,
                            HdPrimvarsSchema::GetDefaultLocator());
                    }
                }
            }
        }

        return result;
    }

    HdSceneIndexPrim GetChildPrim(
        const HdSceneIndexBaseRefPtr &inputScene,
        const SdfPath &childPrimPath) override
    {
        HdSceneIndexPrim result;

        // bbox display mode

        if (_args.displayMode == golaemTokens->bbox) {
            auto it = _childIndices.find(childPrimPath);
            if (it == _childIndices.end()) {
                return result;
            }
            result.primType = HdPrimTypeTokens->mesh;

            const BBoxEntityData& entity = _bboxEntities[it->second];
            GfMatrix4d mtx;
            mtx.SetScale(entity.extent * entity.scale);
            GfMatrix4d mtx2(GfRotation(entity.quat), entity.pos);
            mtx *= mtx2;

            result.dataSource = HdRetainedContainerDataSource::New(
                HdXformSchemaTokens->xform,
                HdXformSchema::Builder()
                .SetMatrix(
                    HdRetainedTypedSampledDataSource<GfMatrix4d>::New(mtx))
                .Build(),
                HdMeshSchemaTokens->mesh,
                GetCubeMeshDataSource(),
                HdPrimvarsSchemaTokens->primvars,
                GetCubePrimvarsDataSource());
        }

        // mesh display mode

        else {
            auto it = _childIndexPairs.find(childPrimPath);
            if (it == _childIndexPairs.end()) {
                return result;
            }
            int entityIndex = it->second.first;
            int meshIndex = it->second.second;
            result.primType = HdPrimTypeTokens->mesh;
            const FileMeshAdapter& adapter =
                _meshEntities[entityIndex].meshes[meshIndex];
            result.dataSource = HdRetainedContainerDataSource::New(
                HdMeshSchemaTokens->mesh,
                adapter.GetMeshDataSource(),
                HdPrimvarsSchemaTokens->primvars,
                adapter.GetPrimvarsDataSource());
        }

        return result;
    }

private:

    Args GetArgs(const HdSceneIndexBaseRefPtr& inputScene);
    void InitCrowd(const HdSceneIndexBaseRefPtr& inputScene);
    void PopulateCrowd(const HdSceneIndexBaseRefPtr& inputScene);
    std::vector<FileMeshAdapter> GenerateMeshes(
        CachedSimulation& cachedSimulation, double frame,
        int entityIndex);

    using _ChildIndexMap = std::unordered_map<SdfPath, int, TfHash>;
    using _ChildIndexPairMap =
        std::unordered_map<SdfPath, std::pair<int, int>, TfHash>;

    // primvars provided by the procedural prim
    Args _args;

    // in bbox display mode, maps the path of a Hydra prim to an index
    // into _bboxEntities
    _ChildIndexMap _childIndices;

    // in mesh display mode, maps the path of a Hydra prim to a pair
    // of indices: an index into _meshEntities, and an index into that
    // structure's meshes
    _ChildIndexPairMap _childIndexPairs;

    SimulationCacheFactory* _factory;

    // how many times Update() has been called
    int _updateCount;

    // the definition of each displayed entity in bbox display mode
    VtArray<BBoxEntityData> _bboxEntities;

    // the definition of each displayed entity in mesh display mode
    VtArray<MeshEntityData> _meshEntities;
};

/*
 * Fetches the primvar of type T identified by the given token and
 * stores it in result, if found.
 */
template<typename T>
void GetTypedPrimvar(
    HdPrimvarsSchema& primvars, TfToken token, T& result)
{
    HdSampledDataSourceHandle src =
        primvars.GetPrimvar(token).GetPrimvarValue();

    if (src) {
        VtValue v = src->GetValue(0.0f);
        if (v.IsHolding<T>()) {
            result = v.UncheckedGet<T>();
        }
    }
}

/*
 * Fetches a primvar which is a token containing a list of names
 * separated by semicolons. Stores the names found in result.
 */
void GetTokenArrayPrimvar(
    HdPrimvarsSchema& primvars, TfToken token, VtTokenArray& result)
{
    HdSampledDataSourceHandle src =
        primvars.GetPrimvar(token).GetPrimvarValue();

    if (src) {
        VtValue v = src->GetValue(0.0f);
        if (v.IsHolding<TfToken>()) {
            std::string str = v.UncheckedGet<TfToken>();
            std::string::size_type pos, lastpos = 0;
            while ((pos = str.find(';', lastpos)) != std::string::npos) {
                result.push_back(
                    TfToken(str.substr(lastpos, pos - lastpos)));
                lastpos = pos + 1;
            }
            if (result.empty()) {
                result.push_back(v.UncheckedGet<TfToken>());
            } else {
                result.push_back(
                    TfToken(str.substr(lastpos, str.size() - lastpos)));
            }
        }
    }
}

Args GolaemProcedural::GetArgs(
    const HdSceneIndexBaseRefPtr& inputScene)
{
    Args result;

    HdSceneIndexPrim prim = inputScene->GetPrim(_GetProceduralPrimPath());
    HdPrimvarsSchema primvars =
        HdPrimvarsSchema::GetFromParent(prim.dataSource);

    GetTokenArrayPrimvar(
        primvars, golaemTokens->crowdFields, result.crowdFields);
    GetTypedPrimvar(
        primvars, golaemTokens->cacheName, result.cacheName);
    GetTypedPrimvar(
        primvars, golaemTokens->cacheDir, result.cacheDir);
    GetTokenArrayPrimvar(
        primvars, golaemTokens->characterFiles, result.characterFiles);
    GetTypedPrimvar(
        primvars, golaemTokens->entityIds, result.entityIds);
    GetTypedPrimvar(
        primvars, golaemTokens->geometryTag, result.geometryTag);
    GetTypedPrimvar(
        primvars, golaemTokens->displayMode, result.displayMode);

    return result;
}

/*
 * Stuff to do one time only, once the arguments (cache file, crowd
 * field names, etc.) are known. We assume that the arguments never
 * change.
 */
void GolaemProcedural::InitCrowd(
    const HdSceneIndexBaseRefPtr& inputScene)
{
    // load characters

    if (!_args.characterFiles.empty()) {
        GlmString list;
        for (const TfToken& file: _args.characterFiles) {
            if (!list.empty()) {
                list += ";";
            }
            list += file.data();
        }
        _factory->loadGolaemCharacters(list);
    }

    // GolaemForUSD does this; do we need to?
    /*
    for (TfToken fieldName: _args.crowdFields) {
        if (fieldName.IsEmpty()) {
            continue;
        }

        // force creation of simulation data (might change Golaem
        // characters if there is a CreateEntity node)

        CachedSimulation& cachedSimulation =
            _factory->getCachedSimulation(
                _args.cacheDir.data(), _args.cacheName.data(),
                fieldName.data());
        cachedSimulation.getFinalSimulationData();

        // compute assets if needed

        int firstFrame, lastFrame;
        cachedSimulation.getSrcFrameRangeAvailableOnDisk(
            firstFrame, lastFrame);

        const auto& entityAssets =
            cachedSimulation.getFinalEntityAssets(firstFrame);
    }
    */

}

void GolaemProcedural::PopulateCrowd(
    const HdSceneIndexBaseRefPtr& inputScene)
{
    // fetch current frame

    double frame = 0.0;
    const HdSceneGlobalsSchema globals =
        HdSceneGlobalsSchema::GetFromSceneIndex(inputScene);
    if (globals) {
        frame = globals.GetCurrentFrame()->GetTypedValue(0.0);
    }

    // iterate over entities in crowd fields

    _bboxEntities.clear();
    _meshEntities.clear();

    glm::IdsFilter entityIdsFilter(_args.entityIds.data());

    for (int ifield = 0; ifield < _args.crowdFields.size(); ++ifield) {
        TfToken fieldName = _args.crowdFields[ifield];
        if (fieldName.IsEmpty()) {
            continue;
        }

        CachedSimulation& cachedSimulation =
            _factory->getCachedSimulation(
                _args.cacheDir.data(), _args.cacheName.data(),
                fieldName.data());

        const GlmSimulationData *simData =
            cachedSimulation.getFinalSimulationData();

        if (simData == nullptr) {
            continue;
        }

        const GlmFrameData *frameData =
            cachedSimulation.getFinalFrameData(
                frame, UINT32_MAX, true);

        if (frameData == nullptr) {
            continue;
        }

        if (_args.displayMode == golaemTokens->bbox) {
            _bboxEntities.reserve(
                _bboxEntities.size() + simData->_entityCount);
        } else {
            _meshEntities.reserve(
                _meshEntities.size() + simData->_entityCount);
        }

        // TODO: multithread?

        for (int ientity = 0; ientity < simData->_entityCount; ++ientity) {

            // do nothing if the entity has been killed or excluded

            auto id = simData->_entityIds[ientity];
            if (id < 0 || !entityIdsFilter(id)) {
                continue;
            }

            // fetch the corresponding character

            auto entityType = simData->_entityTypes[ientity];
            auto characterIndex = simData->_characterIdx[ientity];
            const GolaemCharacter* character =
                _factory->getGolaemCharacter(characterIndex);

            if (character == nullptr) {
                continue;
            }

            // save data needed for rendering bounding boxes

            if (_args.displayMode == golaemTokens->bbox) {

                // fetch animation data: the root bone's position,
                // orientation and size

                const auto& animData = character->_converterMapping;
                const auto *rootBone =
                    animData._skeletonDescription->getRootBone();
                auto rootBoneIndex = rootBone->getSpecificBoneIndex();
                auto boneCount = simData->_boneCount[entityType];

                int frameDataIndex = rootBoneIndex
                    + simData->_iBoneOffsetPerEntityType[entityType]
                    + simData->_indexInEntityType[ientity] * boneCount;

                // the rendering bounding box is huge, so we use the
                // "perception shape" instead (used for obstacle
                // detection by other characters)

                glm::Vector3 extent;
#if 0
                const glm::GeometryAsset *geoAsset =
                    character->getGeometryAssetFirstLOD(_args.geometryTag);
                if (geoAsset) {
                    extent = geoAsset->_halfExtentsYUp;
                } else {
                    extent.fill(1);
                }
#else
                extent = character->_perceptionShapeExtents;
#endif

                glm::Quaternion quat(
                    frameData->_boneOrientations[frameDataIndex]);

                _bboxEntities.resize(_bboxEntities.size() + 1);
                BBoxEntityData& entity = _bboxEntities.back();

                entity.extent.Set(extent.getFloatValues());
                entity.scale = simData->_scales[ientity];
                entity.pos.Set(
                    frameData->_bonePositions[frameDataIndex]);
                entity.quat.SetReal(quat.w);
                entity.quat.SetImaginary(quat.x, quat.y, quat.z);
            }

            // save data needed for rendering meshes

            else {
                _meshEntities.resize(_meshEntities.size() + 1);
                MeshEntityData& entity = _meshEntities.back();

                entity.crowdFieldIndex = ifield;
                entity.entityIndex = ientity;
                entity.meshes = GenerateMeshes(
                    cachedSimulation, frame, ientity);
            }
        }
    }
}

/*
 * Generates and returns a FileMeshAdapter for each mesh constituting
 * the given entity at the given frame.
 */
std::vector<FileMeshAdapter> GolaemProcedural::GenerateMeshes(
    CachedSimulation& cachedSimulation, double frame, int entityIndex)
{
    std::vector<FileMeshAdapter> adapters;

    const GlmSimulationData *simData =
        cachedSimulation.getFinalSimulationData();
    const GlmFrameData *frameData =
        cachedSimulation.getFinalFrameData(frame, UINT32_MAX, true);

    auto entityType = simData->_entityTypes[entityIndex];
    auto characterIndex = simData->_characterIdx[entityIndex];
    const GolaemCharacter* character =
        _factory->getGolaemCharacter(characterIndex);

    const glm::Array<glm::PODArray<int>>& entityAssets =
        cachedSimulation.getFinalEntityAssets(frame);

    glm::crowdio::InputEntityGeoData inputData;
    inputData._simuData = simData;
    inputData._characterIdx = characterIndex;
    inputData._character = character;
    inputData._assets = &entityAssets[entityIndex];
    inputData._entityIndex = entityIndex;
    inputData._entityToBakeIndex = simData->_entityToBakeIndex[entityIndex];
    inputData._entityId = simData->_entityIds[entityIndex];
    inputData._frames.assign(1, frame);
    inputData._frameDatas.assign(1, frameData);
    inputData._geometryTag = _args.geometryTag;
    inputData._geoFileIndex = 0;

    glm::crowdio::OutputEntityGeoData outputData;
    glm::crowdio::GlmGeometryGenerationStatus geoStatus =
        glm::crowdio::glmPrepareEntityGeometry(&inputData, &outputData);

    if (geoStatus != glm::crowdio::GIO_SUCCESS) {
        std::cerr << "glmPrepareEntityGeometry returned error: "
                  << glmConvertGeometryGenerationStatus(geoStatus)
                  << '\n';
        return adapters;
    }

    if (outputData._geoType != glm::crowdio::GeometryType::GCG) {
        std::cerr << "geoType is not GCG\n";
        return adapters;
    }

    glm::crowdio::CrowdGcgCharacter* gcgCharacter =
        outputData._gcgCharacters[0];
    const glm::crowdio::GlmGeometryFile& geoFile =
        gcgCharacter->getGeometry();

    const auto& meshIndices = outputData._meshAssetNameIndices;
    adapters.reserve(meshIndices.size());

    for (int j = 0; j < meshIndices.size(); ++j) {
        const glm::crowdio::GlmFileMeshTransform& meshXform =
            geoFile._transforms[outputData._transformIndicesInGcgFile[j]];
        const glm::crowdio::GlmFileMesh& fileMesh =
            geoFile._meshes[meshXform._meshIndex];
        adapters.emplace_back(
            fileMesh,
            outputData._deformedVertices[0][j],
            outputData._deformedNormals[0][j]);
    }

    return adapters;
}

}  // anonymous namespace

class GolaemProceduralPlugin: public HdGpGenerativeProceduralPlugin
{
public:
    GolaemProceduralPlugin() = default;

    HdGpGenerativeProcedural *Construct(
        const SdfPath &proceduralPrimPath) override
    {
        return new GolaemProcedural(proceduralPrimPath);
    }
};

TF_REGISTRY_FUNCTION(TfType)
{
    HdGpGenerativeProceduralPluginRegistry::Define<
        GolaemProceduralPlugin,
        HdGpGenerativeProceduralPlugin>();
}
