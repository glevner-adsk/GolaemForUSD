/*
 * TODO:
 * - LODs
 * - FBX support
 * - skeleton display mode
 */
#include <pxr/imaging/hdGp/generativeProceduralPlugin.h>
#include <pxr/imaging/hdGp/generativeProceduralPluginRegistry.h>

#include <pxr/imaging/hd/cameraSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopologySchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderSettingsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneGlobalsSchema.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/xformSchema.h>

#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/tf/debug.h>
#include <pxr/base/tf/staticTokens.h>

#include <glmCrowdGcgCharacter.h>
#include <glmCrowdIOUtils.h>
#include <glmCrowdTerrainMesh.h>
#include <glmGolaemCharacter.h>
#include <glmIdsFilter.h>
#include <glmSimulationCacheFactory.h>
#include <glmSimulationCacheFactorySimulation.h>

#include "glmUSD.h"
#include "fileMeshAdapter.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using glm::GlmString;
using glm::GolaemCharacter;
using glm::PODArray;
using glm::ShaderAssetDataContainer;
using glm::ShaderAttribute;
using glm::ShaderAttributeType;
using glm::crowdio::CachedSimulation;
using glm::crowdio::CrowdGcgCharacter;
using glm::crowdio::GlmFileMesh;
using glm::crowdio::GlmFileMeshTransform;
using glm::crowdio::GlmFrameData;
using glm::crowdio::GlmGeometryFile;
using glm::crowdio::GlmNormalMode;
using glm::crowdio::GlmSimulationData;
using glm::crowdio::GlmUVMode;
using glm::crowdio::SimulationCacheFactory;

using glmhydra::FileMeshAdapter;

TF_DEBUG_CODES(
    GLMHYDRA_DIRTY_PRIMS,
    GLMHYDRA_MOTION_BLUR
);

namespace
{
TF_DEFINE_PRIVATE_TOKENS(
    golaemTokens,
    (crowdFields)
    (cacheName)
    (cacheDir)
    (characterFiles)
    (entityIds)
    (enableLayout)
    (layoutFiles)
    (terrainFile)
    (renderPercent)
    (displayMode)
    (geometryTag)
    (dirmap)
    (materialPath)
    (materialAssignMode)
    (enableMotionBlur)
    (lodMode)
    (bbox)
    (mesh)
    (bySurfaceShader)
    (byShadingGroup)
    (none)
    (disabled)
    ((staticLOD, "static"))
    ((dynamicLOD, "dynamic"))
);

/*
 * We use a hash map to store an entity's custom primvars (name and
 * data source for each), which are generated from shader attributes
 * and PP attributes.
 */
using PrimvarDataSourceMap =
    TfDenseHashMap<TfToken, HdSampledDataSourceHandle, TfHash>;
using PrimvarDataSourceMapRef = std::shared_ptr<PrimvarDataSourceMap>;

/*
 * Arguments (primvars) provided by the USD prim.
 */
struct Args
{
    Args()
        : entityIds("*"),
          enableLayout(true),
          renderPercent(100),
          displayMode(golaemTokens->mesh),
          geometryTag(0),
          materialPath("Materials"),
          materialAssignMode(golaemTokens->byShadingGroup),
          enableMotionBlur(false),
          lodMode(golaemTokens->disabled)
        {}

    VtTokenArray crowdFields;
    TfToken cacheName;
    TfToken cacheDir;
    TfToken characterFiles;
    TfToken entityIds;
    bool enableLayout;
    TfToken layoutFiles;
    TfToken terrainFile;
    float renderPercent;
    TfToken displayMode;
    int geometryTag;
    TfToken dirmap;
    SdfPath materialPath;
    TfToken materialAssignMode;
    bool enableMotionBlur;
    TfToken lodMode;
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
    std::vector<std::shared_ptr<FileMeshAdapter>> meshes;
};

/*
 * This is the actual plugin implementation.
 */
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
        const HdSceneIndexBaseRefPtr& inputScene) override;

    ChildPrimTypeMap Update(
        const HdSceneIndexBaseRefPtr& inputScene,
        const ChildPrimTypeMap& previousResult,
        const DependencyMap& dirtiedDependencies,
        HdSceneIndexObserver::DirtiedPrimEntries *outputDirtiedPrims)
        override;

    HdSceneIndexPrim GetChildPrim(
        const HdSceneIndexBaseRefPtr& /*inputScene*/,
        const SdfPath &childPrimPath) override;

private:
    Args GetArgs(const HdSceneIndexBaseRefPtr& inputScene);
    void InitCrowd(const HdSceneIndexBaseRefPtr& inputScene);
    void PopulateCrowd(const HdSceneIndexBaseRefPtr& inputScene);
    std::vector<std::shared_ptr<FileMeshAdapter>> GenerateMeshes(
        CachedSimulation& cachedSimulation, double frame,
        int entityIndex, bool motionBlur, const GfVec2d& shutter);
    PrimvarDataSourceMapRef GenerateCustomPrimvars(
        const GlmSimulationData *simData,
        const GlmFrameData *frameData,
        const ShaderAssetDataContainer *shaderData,
        const GolaemCharacter *character,
        int entityIndex) const;

    using _ChildIndexMap = std::unordered_map<SdfPath, size_t, TfHash>;
    using _ChildIndexPairMap =
        std::unordered_map<SdfPath, std::pair<size_t, size_t>, TfHash>;

    // primvars provided by the procedural prim
    Args _args;

    // parsed dirmap rules for findDirmappedFile()
    glm::Array<GlmString> _dirmapRules;

    // actual cache directory after applying dirmap rules
    GlmString _mappedCacheDir;

    // in bbox display mode, maps the path of a Hydra prim to an index
    // into _bboxEntities
    _ChildIndexMap _childIndices;

    // in mesh display mode, maps the path of a Hydra prim to a pair
    // of indices: an index into _meshEntities, and an index into that
    // structure's meshes
    _ChildIndexPairMap _childIndexPairs;

    // the Golaem simulation cache factory
    SimulationCacheFactory* _factory;

    // how many times Update() has been called
    int _updateCount;

    // the definition of each displayed entity in bbox display mode
    std::vector<BBoxEntityData> _bboxEntities;

    // the definition of each displayed entity in mesh display mode
    std::vector<MeshEntityData> _meshEntities;
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

    HdSceneIndexPrim prim =
        inputScene->GetPrim(_GetProceduralPrimPath());
    HdPrimvarsSchema primvars =
        HdPrimvarsSchema::GetFromParent(prim.dataSource);

    GetTokenArrayPrimvar(
        primvars, golaemTokens->crowdFields, result.crowdFields);
    GetTypedPrimvar(
        primvars, golaemTokens->cacheName, result.cacheName);
    GetTypedPrimvar(
        primvars, golaemTokens->cacheDir, result.cacheDir);
    GetTypedPrimvar(
        primvars, golaemTokens->characterFiles, result.characterFiles);
    GetTypedPrimvar(
        primvars, golaemTokens->entityIds, result.entityIds);
    GetTypedPrimvar(
        primvars, golaemTokens->enableLayout, result.enableLayout);
    GetTypedPrimvar(
        primvars, golaemTokens->layoutFiles, result.layoutFiles);
    GetTypedPrimvar(
        primvars, golaemTokens->terrainFile, result.terrainFile);
    GetTypedPrimvar(
        primvars, golaemTokens->renderPercent, result.renderPercent);
    GetTypedPrimvar(
        primvars, golaemTokens->displayMode, result.displayMode);
    GetTypedPrimvar(
        primvars, golaemTokens->geometryTag, result.geometryTag);
    GetTypedPrimvar(
        primvars, golaemTokens->dirmap, result.dirmap);
    GetTypedPrimvar(
        primvars, golaemTokens->materialAssignMode,
        result.materialAssignMode);
    GetTypedPrimvar(
        primvars, golaemTokens->enableMotionBlur,
        result.enableMotionBlur);
    GetTypedPrimvar(
        primvars, golaemTokens->lodMode, result.lodMode);

    // a primvar cannot be a relationship, so we convert the
    // materialPath argument (a token) to an SdfPath, which can be
    // relative to the procedural prim

    TfToken matpath;
    GetTypedPrimvar(primvars, golaemTokens->materialPath, matpath);

    if (matpath.IsEmpty()) {
        result.materialPath = _GetProceduralPrimPath();
    } else {
        std::string stdpath = matpath.GetString();
        if (stdpath.back() == '/') {
            stdpath.pop_back();
        }
        result.materialPath = SdfPath(stdpath)
            .MakeAbsolutePath(_GetProceduralPrimPath());
    }

    return result;
}

/*
 * Stuff to do one time only, once the arguments (cache file, crowd
 * field names, etc.) are known. We assume that the arguments never
 * change.
 */
void GolaemProcedural::InitCrowd(
    const HdSceneIndexBaseRefPtr& /*inputScene*/)
{
    // apply dirmap rules to find actual paths of character files and
    // load them

    if (_args.characterFiles.size() > 0) {
        glm::Array<GlmString> fileList;
        split(_args.characterFiles.GetString(), ";", fileList);

        for (int i = 0; i < fileList.size(); ++i) {
            GlmString mappedPath;
            findDirmappedFile(mappedPath, fileList[i], _dirmapRules);
            fileList[i] = mappedPath;
        }

        GlmString characterFiles =
            glm::stringArrayToString(fileList, ";");
        _factory->loadGolaemCharacters(characterFiles);
    }

    // dirmap and load layout and terrain files

    if (_args.enableLayout && !_args.layoutFiles.IsEmpty()) {

        // load layout files

        glm::Array<GlmString> fileList;
        split(_args.layoutFiles.GetString(), ";", fileList);

        for (int i = 0; i < fileList.size(); ++i) {
            GlmString mappedPath;
            findDirmappedFile(mappedPath, fileList[i], _dirmapRules);
            _factory->loadLayoutHistoryFile(
                _factory->getLayoutHistoryCount(), mappedPath.c_str());
        }

        // load terrain files

        glm::crowdio::crowdTerrain::TerrainMesh *srcTerrain = nullptr;
        glm::crowdio::crowdTerrain::TerrainMesh *dstTerrain = nullptr;

        if (_args.crowdFields.size() > 0) {
            GlmString glmpath =
                _mappedCacheDir + "/" + _args.cacheName.GetText() +
                "." + _args.crowdFields[0].GetText() + ".gtg";
            srcTerrain = glm::crowdio::crowdTerrain::loadTerrainAsset(
                glmpath.c_str());
        }

        if (!_args.terrainFile.IsEmpty()) {
            GlmString mappedPath;
            findDirmappedFile(
                mappedPath, _args.terrainFile.GetString(),
                _dirmapRules);
            dstTerrain = glm::crowdio::crowdTerrain::loadTerrainAsset(
                mappedPath.c_str());
        }

        if (dstTerrain == nullptr) {
            dstTerrain = srcTerrain;
        }

        _factory->setTerrainMeshes(srcTerrain, dstTerrain);
    }
}

/*
 * Fetches and returns the current frame number from globals.
 */
double GetCurrentFrame(const HdSceneIndexBaseRefPtr& inputScene)
{
    double frame = 0.0;

    const HdSceneGlobalsSchema globals =
        HdSceneGlobalsSchema::GetFromSceneIndex(inputScene);

    if (globals) {
        frame = globals.GetCurrentFrame()->GetTypedValue(0);
    }

    return frame;
}

/*
 * Fetches and returns the shutter interval for motion blur from the
 * active render settings prim, if there is one. Returns true if there
 * is, false if not.
 */
bool GetShutterFromRenderSettings(
    const HdSceneIndexBaseRefPtr& inputScene, GfVec2d *shutter)
{
    const HdSceneGlobalsSchema globals =
        HdSceneGlobalsSchema::GetFromSceneIndex(inputScene);
    if (!globals) {
        return false;
    }

    HdPathDataSourceHandle rsPathDS = globals.GetActiveRenderSettingsPrim();
    if (!rsPathDS) {
        return false;
    }

    const SdfPath rsPath = rsPathDS->GetTypedValue(0);
    if (rsPath.IsEmpty()) {
        return false;
    }

    const HdSceneIndexPrim rsPrim = inputScene->GetPrim(rsPath);
    HdRenderSettingsSchema rs =
        HdRenderSettingsSchema::GetFromParent(rsPrim.dataSource);
    if (!rs) {
        return false;
    }

    HdVec2dDataSourceHandle shutterDS = rs.GetShutterInterval();
    if (!shutterDS) {
        return false;
    }

    *shutter = shutterDS->GetTypedValue(0);
    return true;
}

/*
 * Fetches and returns the shutter interval from the primary camera
 * prim, if there is one. Returns true if there is, false if not.
 */
bool GetShutterFromCamera(
    const HdSceneIndexBaseRefPtr& inputScene, GfVec2d *shutter)
{
    const HdSceneGlobalsSchema globals =
        HdSceneGlobalsSchema::GetFromSceneIndex(inputScene);
    if (!globals) {
        return false;
    }

    HdPathDataSourceHandle camPathDS = globals.GetPrimaryCameraPrim();
    if (!camPathDS) {
        return false;
    }

    const SdfPath camPath = camPathDS->GetTypedValue(0);
    if (camPath.IsEmpty()) {
        return false;
    }

    const HdSceneIndexPrim camPrim = inputScene->GetPrim(camPath);
    HdCameraSchema cam =
        HdCameraSchema::GetFromParent(camPrim.dataSource);
    if (!cam) {
        return false;
    }

    HdDoubleDataSourceHandle openDS = cam.GetShutterOpen();
    HdDoubleDataSourceHandle closeDS = cam.GetShutterClose();
    if (!(openDS && closeDS)) {
        return false;
    }

    shutter->Set(openDS->GetTypedValue(0), closeDS->GetTypedValue(0));
    return true;
}

void GolaemProcedural::PopulateCrowd(
    const HdSceneIndexBaseRefPtr& inputScene)
{
    // fetch current frame and motion blur settings

    double frame = GetCurrentFrame(inputScene);

    bool motionBlur = false;
    GfVec2d shutter;

    if (_args.enableMotionBlur) {
        if (GetShutterFromRenderSettings(inputScene, &shutter)) {
            TF_DEBUG_MSG(
                GLMHYDRA_MOTION_BLUR,
                "motion blur shutter from render settings: %g %g\n",
                shutter[0], shutter[1]);
            motionBlur = (shutter[0] < shutter[1]);
        } else if (GetShutterFromCamera(inputScene, &shutter)) {
            TF_DEBUG_MSG(
                GLMHYDRA_MOTION_BLUR,
                "motion blur shutter from camera: %g %g\n",
                shutter[0], shutter[1]);
            motionBlur = (shutter[0] < shutter[1]);
        }
    }

    // iterate over entities in crowd fields

    _bboxEntities.clear();
    _meshEntities.clear();

    glm::IdsFilter entityIdsFilter(_args.entityIds.GetText());

    for (int ifield = 0; ifield < _args.crowdFields.size(); ++ifield) {
        TfToken fieldName = _args.crowdFields[ifield];
        if (fieldName.IsEmpty()) {
            continue;
        }

        CachedSimulation& cachedSimulation =
            _factory->getCachedSimulation(
                _mappedCacheDir.c_str(), _args.cacheName.GetText(),
                fieldName.GetText());

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

        int entityCount = simData->_entityCount;
        if (_args.renderPercent < 100.0f) {
            entityCount =
                std::lround(entityCount * _args.renderPercent * 0.01f);
        }
        if (_args.displayMode == golaemTokens->bbox) {
            _bboxEntities.reserve(_bboxEntities.size() + entityCount);
        } else {
            _meshEntities.reserve(_meshEntities.size() + entityCount);
        }

        for (int ientity = 0; ientity < entityCount; ++ientity) {

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
                    character->getGeometryAssetFirstLOD(
                        short(_args.geometryTag));
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
                    cachedSimulation, frame, ientity, motionBlur,
                    shutter);
            }
        }
    }
}

/*
 * Finds all the shader and PP attributes defined for the given entity
 * and generates a Hydra data source of the appropriate type for each.
 * Returns a shared pointer to a hash map containing the name and data
 * source for each. Pass that hash map to GenerateMeshes() so that
 * each of the mesh's entities shares them.
 */
PrimvarDataSourceMapRef GolaemProcedural::GenerateCustomPrimvars(
    const GlmSimulationData *simData,
    const GlmFrameData *frameData,
    const ShaderAssetDataContainer* shaderData,
    const GolaemCharacter *character,
    int entityIndex) const
{
    PrimvarDataSourceMapRef dataSources =
        std::make_shared<PrimvarDataSourceMap>();

    size_t shaderAttrCount = character->_shaderAttributes.size();
    size_t totalCount = shaderAttrCount
        + simData->_ppFloatAttributeCount
        + simData->_ppVectorAttributeCount;

    if (totalCount == 0) {
        return dataSources;
    }

    dataSources->reserve(totalCount);

    auto characterIndex = simData->_characterIdx[entityIndex];
    auto bakeIndex = simData->_entityToBakeIndex[entityIndex];

    // shader attributes (int, float, string, vector)

    const PODArray<int>& intData =
        shaderData->intData[entityIndex];
    const PODArray<float>& floatData =
        shaderData->floatData[entityIndex];
    const glm::Array<glm::Vector3>& vectorData =
        shaderData->vectorData[entityIndex];
    const glm::Array<GlmString>& stringData =
        shaderData->stringData[entityIndex];

    const PODArray<size_t>& globalToSpecificShaderAttrIdx =
        shaderData->globalToSpecificShaderAttrIdxPerChar[characterIndex];

    for (size_t i = 0; i < shaderAttrCount; ++i) {
        const ShaderAttribute& attr = character->_shaderAttributes[i];

        // ensure the attribute name is a valid identifier, and maybe
        // prefix it with "arnold:"

        std::string stdname;
        GlmString glmname = attr._name.c_str();
        GlmString subAttrName;
        glm::crowdio::RendererAttributeType::Value overrideType =
            glm::crowdio::RendererAttributeType::END;
        if (glm::crowdio::parseRendererAttribute(
                "arnold", attr._name, glmname, subAttrName,
                overrideType)) {
            stdname = "arnold:" + TfMakeValidIdentifier(glmname.c_str());
        } else {
            stdname = TfMakeValidIdentifier(glmname.c_str());
        }
        TfToken name(stdname);

        // create a data source that returns the attribute's value

        size_t index = globalToSpecificShaderAttrIdx[i];
        switch (attr._type) {
        case ShaderAttributeType::INT:
            (*dataSources)[name] =
                HdRetainedTypedSampledDataSource<int>::New(
                    intData[index]);
            break;
        case ShaderAttributeType::FLOAT:
            (*dataSources)[name] =
                HdRetainedTypedSampledDataSource<float>::New(
                    floatData[index]);
            break;
        case ShaderAttributeType::STRING:
            (*dataSources)[name] =
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    TfToken(stringData[index].c_str()));
            break;
        case ShaderAttributeType::VECTOR:
            (*dataSources)[name] =
                HdRetainedTypedSampledDataSource<GfVec3f>::New(
                    GfVec3f(vectorData[index].getFloatValues()));
            break;
        }
    }

    // PP attributes (float and vector)

    for (size_t i = 0; i < simData->_ppFloatAttributeCount; ++i) {
        TfToken name(
            TfMakeValidIdentifier(simData->_ppFloatAttributeNames[i]));
        (*dataSources)[name] =
            HdRetainedTypedSampledDataSource<float>::New(
                frameData->_ppFloatAttributeData[i][bakeIndex]);
    }

    for (size_t i = 0; i < simData->_ppVectorAttributeCount; ++i) {
        TfToken name(
            TfMakeValidIdentifier(simData->_ppVectorAttributeNames[i]));
        (*dataSources)[name] =
            HdRetainedTypedSampledDataSource<GfVec3f>::New(
                GfVec3f(frameData->_ppVectorAttributeData[i][bakeIndex]));
    }

    return dataSources;
}

/*
 * Generates and returns a FileMeshAdapter for each mesh constituting
 * the given entity at the given frame.
 */
std::vector<std::shared_ptr<FileMeshAdapter>>
GolaemProcedural::GenerateMeshes(
    CachedSimulation& cachedSimulation, double frame, int entityIndex,
    bool motionBlur, const GfVec2d& shutter)
{
    std::vector<std::shared_ptr<FileMeshAdapter>> adapters;

    // fetch simulation data, frame data and assets, then call
    // glmPrepareEntityGeometry() to generate information about this
    // entity at this frame

    const GlmSimulationData *simData =
        cachedSimulation.getFinalSimulationData();
    const GlmFrameData *frameData =
        cachedSimulation.getFinalFrameData(frame, UINT32_MAX, true);

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
    inputData._dirMapRules = _dirmapRules;
    inputData._geometryTag = short(_args.geometryTag);
    inputData._geoFileIndex = 0;

    glm::Array<float> shutterOffsets;

    if (motionBlur) {
        inputData._frames.reserve(3);
        inputData._frameDatas.reserve(3);
        if (shutter[0] != 0.0) {
            inputData._frames.push_back(shutter[0]);
            inputData._frameDatas.push_back(
                cachedSimulation.getFinalFrameData(
                    frame + shutter[0], UINT32_MAX, true));
            shutterOffsets.push_back(float(shutter[0]));
        }
        if (shutter[0] <= 0.0 && shutter[1] >= 0.0) {
            inputData._frames.push_back(0);
            inputData._frameDatas.push_back(frameData);
            shutterOffsets.push_back(0);
        }
        if (shutter[1] != 0.0) {
            inputData._frames.push_back(shutter[1]);
            inputData._frameDatas.push_back(
                cachedSimulation.getFinalFrameData(
                    frame + shutter[0], UINT32_MAX, true));
            shutterOffsets.push_back(float(shutter[1]));
        }
    } else {
        inputData._frames.assign(1, frame);
        inputData._frameDatas.assign(1, frameData);
    }

    glm::crowdio::OutputEntityGeoData outputData;
    glm::crowdio::GlmGeometryGenerationStatus geoStatus =
        glm::crowdio::glmPrepareEntityGeometry(&inputData, &outputData);

    if (geoStatus != glm::crowdio::GIO_SUCCESS) {
        std::cerr << "glmPrepareEntityGeometry() returned error: "
                  << glmConvertGeometryGenerationStatus(geoStatus)
                  << '\n';
        return adapters;
    }

    if (outputData._geoType != glm::crowdio::GeometryType::GCG) {
        std::cerr << "geometry type is not GCG, ignoring\n";
        return adapters;
    }

    // fetch custom primvars for this entity: shader attributes and PP
    // attributes

    const ShaderAssetDataContainer* shaderData =
        cachedSimulation.getFinalShaderData(frame, UINT32_MAX, true);

    PrimvarDataSourceMapRef customPrimvars = GenerateCustomPrimvars(
        simData, frameData, shaderData, character, entityIndex);

    // fetch the corresponding character, geometry and mesh count

    CrowdGcgCharacter *gcgCharacter = outputData._gcgCharacters[0];
    const GlmGeometryFile& geoFile = gcgCharacter->getGeometry();
    size_t meshCount = outputData._meshAssetNameIndices.size();
    adapters.reserve(meshCount);

    for (size_t imesh = 0; imesh < meshCount; ++imesh) {

        // fetch the mesh itself

        const GlmFileMeshTransform& meshXform =
            geoFile._transforms[
                outputData._transformIndicesInGcgFile[imesh]];
        const GlmFileMesh& fileMesh =
            geoFile._meshes[meshXform._meshIndex];

        // append the name of its material to the material path

        SdfPath material;
        int shGroupIndex = outputData._meshShadingGroups[imesh];

        if (_args.materialAssignMode != golaemTokens->none
            && shGroupIndex >= 0) {
            const glm::ShadingGroup& shGroup =
                inputData._character->_shadingGroups[shGroupIndex];
            std::string matname;

            // assign material by shading group

            if (_args.materialAssignMode == golaemTokens->byShadingGroup) {
                const GlmString& glmname = shGroup._name;
                matname.assign(glmname.c_str(), glmname.size());
            }

            // assign material by surface shader

            else {
                int shAssetIndex =
                    character->findShaderAsset(shGroup, "surface");
                if (shAssetIndex >= 0) {
                    const GlmString& glmname =
                        character->_shaderAssets[shAssetIndex]._name;
                    matname.assign(glmname.c_str(), glmname.size());
                } else {
                    matname = "DefaultGolaemMat";
                }
            }

            material =
                _args.materialPath.AppendElementString(matname);
        }

        // construct a FileMeshAdapter to generate Hydra data sources
        // for the mesh and its material

        std::shared_ptr<FileMeshAdapter> adapter =
            std::make_shared<FileMeshAdapter>(
                fileMesh, material, customPrimvars);

        if (motionBlur) {
            adapter->SetAnimatedData(
                shutterOffsets, outputData._deformedVertices,
                outputData._deformedNormals, imesh);
        } else {
            adapter->SetAnimatedData(
                outputData._deformedVertices[0][imesh],
                outputData._deformedNormals[0][imesh]);
        }

        adapters.emplace_back(adapter);
    }

    return adapters;
}

HdGpGenerativeProcedural::DependencyMap
GolaemProcedural::UpdateDependencies(
    const HdSceneIndexBaseRefPtr &inputScene)
{
    DependencyMap result;

    // always call Update() when the current frame changes

    SdfPath primPath = HdSceneGlobalsSchema::GetDefaultPrimPath();
    result[primPath] = HdSceneGlobalsSchema::GetCurrentFrameLocator();

    // update when the camera changes if motion blur or dynamic LOD is
    // enabled (and note the path of the camera prim for later)

    const HdSceneGlobalsSchema globals =
        HdSceneGlobalsSchema::GetFromSceneIndex(inputScene);

    SdfPath camPath;

    if (_args.enableMotionBlur
        || _args.lodMode == golaemTokens->dynamicLOD) {
        result[primPath].insert(
            HdSceneGlobalsSchema::GetPrimaryCameraPrimLocator());

        if (globals) {
            HdPathDataSourceHandle camPathDS =
                globals.GetPrimaryCameraPrim();
            if (camPathDS) {
                camPath = camPathDS->GetTypedValue(0);
            }
        }
    }

    // update when the camera moves if dynamic LOD is enabled

    if (_args.lodMode == golaemTokens->dynamicLOD && !camPath.IsEmpty()) {
        TF_DEBUG_MSG(
            GLMHYDRA_DIRTY_PRIMS,
            "add dependency on camera xform: %s\n",
            camPath.GetAsString().c_str());
        result[camPath].insert(HdXformSchema::GetDefaultLocator());
    }

    // if motion blur is enabled, update when the render settings
    // change or when the shutter interval changes: if there is an
    // active render settings prim, get the shutter interval from
    // there, otherwise use the primary camera's shutter settings

    if (_args.enableMotionBlur) {
        result[primPath].insert(
            HdSceneGlobalsSchema::GetActiveRenderSettingsPrimLocator());

        if (globals) {
            HdPathDataSourceHandle rsPrimDS =
                globals.GetActiveRenderSettingsPrim();
            if (rsPrimDS) {
                const SdfPath rsPath = rsPrimDS->GetTypedValue(0);
                if (!rsPath.IsEmpty()) {
                    TF_DEBUG_MSG(
                        GLMHYDRA_DIRTY_PRIMS,
                        "add dependency on render settings shutter\n");
                    result[rsPath] = {
                        HdRenderSettingsSchema::GetShutterIntervalLocator()
                    };
                }
            } else if (!camPath.IsEmpty()) {
                TF_DEBUG_MSG(
                    GLMHYDRA_DIRTY_PRIMS,
                    "add dependency on camera shutter: %s\n",
                    camPath.GetAsString().c_str());
                result[camPath].insert(
                    HdCameraSchema::GetShutterOpenLocator());
                result[camPath].insert(
                    HdCameraSchema::GetShutterCloseLocator());
            }
        }
    }

    return result;
}

HdGpGenerativeProcedural::ChildPrimTypeMap GolaemProcedural::Update(
    const HdSceneIndexBaseRefPtr& inputScene,
    const ChildPrimTypeMap& previousResult,
    const DependencyMap& dirtiedDependencies,
    HdSceneIndexObserver::DirtiedPrimEntries *outputDirtiedPrims)
{
    if (TfDebug::IsEnabled(GLMHYDRA_DIRTY_PRIMS)) {
        std::ostringstream strm;
        for (auto it = dirtiedDependencies.begin();
             it != dirtiedDependencies.end(); ++it) {
            strm << "dirty prim: " << it->first << " "
                 << it->second << '\n';
        }
        TF_DEBUG_MSG(GLMHYDRA_DIRTY_PRIMS, strm.str());
    }

    // fetch arguments (primvars) the first time only (we assume they
    // never change), then (re)populate the scene

    if (_updateCount == 0) {
        _args = GetArgs(inputScene);
        _dirmapRules = glm::stringToStringArray(
            _args.dirmap.GetString(), ";");
        findDirmappedFile(
            _mappedCacheDir, _args.cacheDir.GetString(), _dirmapRules);
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

        for (size_t i = 0; i < _bboxEntities.size(); ++i) {
            sprintf_s(buffer, "c%zu", i);
            SdfPath childPath = myPath.AppendChild(TfToken(buffer));
            result[childPath] = HdPrimTypeTokens->mesh;
            _childIndices[childPath] = i;

            // if the same path was generated by the previous call to
            // Update(), too, tell Hydra its xform may have changed

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

        for (size_t i = 0; i < _meshEntities.size(); ++i) {
            const MeshEntityData& entity = _meshEntities[i];
            for (size_t j = 0; j < entity.meshes.size(); ++j) {

                // including the crowd field, entity and mesh in the
                // path enables us to tell Hydra that, if the same
                // prim appears in two successive frames, only the
                // points and normals will have changed

                sprintf_s(buffer, "c%d_%d_%zu",
                        entity.crowdFieldIndex, entity.entityIndex, j);
                SdfPath childPath = myPath.AppendChild(TfToken(buffer));
                result[childPath] = HdPrimTypeTokens->mesh;
                _childIndexPairs[childPath] = {i, j};

                if (previousResult.size() > 0) {
                    outputDirtiedPrims->emplace_back(
                        childPath, HdDataSourceLocatorSet({
                                HdPrimvarsSchema::GetPointsLocator(),
                                HdPrimvarsSchema::GetNormalsLocator()
                            }));
                }
            }
        }
    }

    return result;
}

/*
 * Returns a data source which returns the topology of a cube.
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
 * Returns a data source which returns the vertices of a cube.
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

HdSceneIndexPrim GolaemProcedural::GetChildPrim(
    const HdSceneIndexBaseRefPtr &/*inputScene*/,
    const SdfPath &childPrimPath)
{
    HdSceneIndexPrim result;

    // bbox display mode

    if (_args.displayMode == golaemTokens->bbox) {
        auto it = _childIndices.find(childPrimPath);
        if (it == _childIndices.end()) {
            return result;
        }

        const BBoxEntityData& entity = _bboxEntities[it->second];
        GfMatrix4d mtx;
        mtx.SetScale(entity.extent * entity.scale);
        GfMatrix4d mtx2(GfRotation(entity.quat), entity.pos);
        mtx *= mtx2;

        result.primType = HdPrimTypeTokens->mesh;
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
        static const HdContainerDataSourceHandle identityXform =
            HdXformSchema::Builder()
            .SetMatrix(
                HdRetainedTypedSampledDataSource<GfMatrix4d>::New(
                    GfMatrix4d(1.0)))
            .Build();

        auto it = _childIndexPairs.find(childPrimPath);
        if (it == _childIndexPairs.end()) {
            return result;
        }

        size_t entityIndex = it->second.first;
        size_t meshIndex = it->second.second;
        const std::shared_ptr<FileMeshAdapter>& adapter =
            _meshEntities[entityIndex].meshes[meshIndex];

        result.primType = HdPrimTypeTokens->mesh;
        result.dataSource = HdRetainedContainerDataSource::New(
            HdXformSchemaTokens->xform,
            identityXform,
            HdMeshSchemaTokens->mesh,
            adapter->GetMeshDataSource(),
            HdPrimvarsSchemaTokens->primvars,
            adapter->GetPrimvarsDataSource(),
            HdMaterialBindingsSchemaTokens->materialBindings,
            adapter->GetMaterialDataSource());
    }

    return result;
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

TF_REGISTRY_FUNCTION(TfDebug)
{
    TF_DEBUG_ENVIRONMENT_SYMBOL(
        GLMHYDRA_DIRTY_PRIMS, "which prims are dirty on Update()");
    TF_DEBUG_ENVIRONMENT_SYMBOL(
        GLMHYDRA_MOTION_BLUR, "motion blur debugging");
}
