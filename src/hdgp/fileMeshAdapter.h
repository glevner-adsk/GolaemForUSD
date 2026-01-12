#pragma once

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/retainedDataSource.h>

#include <pxr/usd/sdf/path.h>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/denseHashMap.h>
#include <pxr/base/tf/hash.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>

#include <glmArray.h>
#include <glmGeometryFile.h>
#include <glmVector3.h>

#include <memory>

namespace glmHydra {

PXR_NAMESPACE_USING_DIRECTIVE

/*
 * Class which provides Hydra data sources wrapping the topology found
 * in a GlmFileMesh, as well as the deformed vertices and normals at
 * any given frame, plus UVs and any other custom attributes (shader
 * and PP).
 */
class FileMeshAdapter
{
public:
using PrimvarDSMap =
    TfDenseHashMap<TfToken, HdSampledDataSourceHandle, TfHash>;
using PrimvarDSMapRef = std::shared_ptr<PrimvarDSMap>;

    FileMeshAdapter(
        const glm::crowdio::GlmFileMesh& fileMesh,
        const glm::Array<glm::Vector3>& deformedVertices,
        const glm::Array<glm::Vector3>& deformedNormals,
        const SdfPath& material,
        const PrimvarDSMapRef& customPrimvars);

    HdContainerDataSourceHandle GetMeshDataSource() const;
    HdContainerDataSourceHandle GetPrimvarsDataSource() const;
    HdContainerDataSourceHandle GetMaterialDataSource() const;

private:
    using IntArrayDS = HdRetainedTypedSampledDataSource<VtIntArray>;
    using Vec3fArrayDS = HdRetainedTypedSampledDataSource<VtArray<GfVec3f>>;
    using Vec2fArrayDS = HdRetainedTypedSampledDataSource<VtArray<GfVec2f>>;

    VtIntArray _vertexCounts;
    VtIntArray _vertexIndices;
    VtVec3fArray _vertices;
    VtIntArray _normalIndices;
    glm::crowdio::GlmNormalMode _normalMode;
    VtVec3fArray _normals;
    VtIntArray _uvIndices;
    glm::crowdio::GlmUVMode _uvMode;
    VtVec2fArray _uvs;
    SdfPath _material;
    const PrimvarDSMapRef _customPrimvars;
};

}  // namespace glmHydra
