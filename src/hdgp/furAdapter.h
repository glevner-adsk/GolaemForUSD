#pragma once

#include <glmArray.h>
#include <glmFurCache.h>
#include <glmVector3.h>

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

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
        float scale, float renderPercent = 100.0f, int refineLevel = 0);

    void SetGeometry(const glm::Array<glm::Vector3>& deformedVertices);

    // TODO: variant of SetGeometry() with shutter offsets

    HdContainerDataSourceHandle GetDataSource() const;

private:
    HdContainerDataSourceHandle GetXformDataSource() const;
    HdContainerDataSourceHandle GetCurveDataSource() const;
    HdContainerDataSourceHandle GetPrimvarsDataSource() const;
    HdContainerDataSourceHandle GetDisplayStyleDataSource() const;

    glm::crowdio::FurCache::SP _furCachePtr;
    size_t _meshInFurIndex;
    int _curveIncr;
    int _refineLevel;
    VtIntArray _vertexCounts;
    VtIntArray _vertexIndices;
    VtVec3fArray _vertices;
    VtFloatArray _widths;
    VtVec2fArray _uvs;
    TfToken _curveDegree;
};

}  // namespace glmhydra
