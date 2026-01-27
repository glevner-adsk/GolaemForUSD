#pragma once

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/retainedDataSource.h>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/hash.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>

#include <glmArray.h>
#include <glmGeometryFile.h>
#include <glmVector3.h>

#include <vector>

namespace glmhydra {

PXR_NAMESPACE_USING_DIRECTIVE

/*
 * Class which provides Hydra data sources wrapping the topology and UVs found
 * in a GlmFileMesh, as well as the deformed vertices and normals at any given
 * frame. Use FileMeshInstance to add an xform, a material and any custom
 * primvars.
 */
class FileMeshAdapter
{
public:
    using DeformedVectors = glm::Array<glm::Array<glm::Array<glm::Vector3>>>;

    FileMeshAdapter(const glm::crowdio::GlmFileMesh& fileMesh);

    void SetGeometry(
        const glm::Array<glm::Vector3>& deformedVertices,
        const glm::Array<glm::Vector3>& deformedNormals);

    void SetGeometry(
        const glm::Array<HdSampledDataSource::Time>& shutterOffsets,
        const DeformedVectors& deformedVertices,
        const DeformedVectors& deformedNormals,
        size_t meshIndex);

    HdContainerDataSourceHandle GetMeshDataSource() const;
    HdContainerDataSourceHandle GetPrimvarsDataSource() const;

    bool IsRigid() const {
        return _isRigid;
    }

private:
    using IntArrayDS = HdRetainedTypedSampledDataSource<VtIntArray>;
    using Vec3fArrayDS = HdRetainedTypedMultisampledDataSource<VtVec3fArray>;
    using Vec2fArrayDS = HdRetainedTypedSampledDataSource<VtVec2fArray>;

    VtIntArray _vertexCounts;
    VtIntArray _vertexIndices;
    size_t _totalVertexCount;
    std::vector<VtVec3fArray> _vertices;
    VtIntArray _normalIndices;
    glm::crowdio::GlmNormalMode _normalMode;
    size_t _totalNormalCount;
    std::vector<VtVec3fArray> _normals;
    VtIntArray _uvIndices;
    glm::crowdio::GlmUVMode _uvMode;
    VtVec2fArray _uvs;
    std::vector<HdSampledDataSource::Time> _shutterOffsets;
    bool _isRigid;
};

}  // namespace glmhydra
