#pragma once

#include "hydraTools.h"

#include <glmArray.h>
#include <glmFurCache.h>
#include <glmVector3.h>

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <vector>

namespace glmhydra {

PXR_NAMESPACE_USING_DIRECTIVE

/*
 * Class which provides data sources for the curves defined by a Golaem
 * FurCache.
 */
class FurAdapter
{
public:
    FurAdapter(
        glm::crowdio::FurCache::SP furCachePtr, size_t meshInFurIndex,
        float scale, const SdfPath& material,
        const tools::PrimvarDSMapRef& customPrimvars,
        float renderPercent = 100.0f, int refineLevel = 0);

    void SetGeometry(const glm::Array<glm::Vector3>& deformedVertices);

    void SetGeometry(
        const glm::Array<HdSampledDataSource::Time>& shutterOffsets,
        const tools::DeformedVectors& deformedVertices, size_t furIndex);

    HdContainerDataSourceHandle GetDataSource() const;

private:
    void CopyVertices(int shutterIndex, const glm::Array<glm::Vector3>& src);

    HdContainerDataSourceHandle GetCurveDataSource() const;
    HdContainerDataSourceHandle GetPrimvarsDataSource() const;
    HdContainerDataSourceHandle GetMaterialDataSource() const;
    HdContainerDataSourceHandle GetDisplayStyleDataSource() const;

    glm::crowdio::FurCache::SP _furCachePtr;
    size_t _meshInFurIndex;
    int _curveIncr;
    SdfPath _material;
    const tools::PrimvarDSMapRef _customPrimvars;
    int _refineLevel;
    VtIntArray _vertexCounts;
    VtIntArray _vertexIndices;
    std::vector<VtVec3fArray> _vertices;
    VtFloatArray _widths;
    VtVec2fArray _uvs;
    TfToken _curveDegree;
    std::vector<HdSampledDataSource::Time> _shutterOffsets;
};

}  // namespace glmhydra
