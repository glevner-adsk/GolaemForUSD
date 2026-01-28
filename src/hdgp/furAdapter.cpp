#include "furAdapter.h"

#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>
#include <iostream>

using glm::crowdio::FurCache;
using glm::crowdio::FurCurveGroup;

namespace glmhydra {

/*
 * Note that FurAdapter keeps a (smart) pointer to the fur cache, preventing it
 * from being deleted before the FurAdapter.
 */
FurAdapter::FurAdapter(
    FurCache::SP furCachePtr, size_t meshInFurIndex, float /*scale*/,
    float renderPercent)
    : _furCachePtr(furCachePtr),
      _meshInFurIndex(meshInFurIndex),
      _curveIncr(std::lround(100.0f / renderPercent))
{
    const FurCache& furCache = *_furCachePtr;

    size_t totalCurveCount = 0;
    size_t totalVertexCount = 0;
    size_t groupCount = furCache._curveGroups.size();

    for (size_t igroup = 0; igroup < groupCount; ++igroup) {
        const FurCurveGroup& group = furCache._curveGroups[igroup];
        if (group._supportMeshId == _meshInFurIndex) {
            size_t ncurve = group._numVertices.size();
            for (size_t icurve = 0; icurve < ncurve; icurve += _curveIncr) {
                totalCurveCount += 1;
                totalVertexCount += group._numVertices[icurve];
            }
        }
    }

    _vertexCounts.reserve(totalCurveCount);

    for (size_t igroup = 0; igroup < groupCount; ++igroup) {
        const FurCurveGroup& group = furCache._curveGroups[igroup];
        if (group._supportMeshId == _meshInFurIndex) {
            size_t ncurve = group._numVertices.size();
            for (size_t icurve = 0; icurve < ncurve; icurve += _curveIncr) {
                _vertexCounts.push_back(group._numVertices[icurve]);
            }
        }
    }

    _vertexIndices.reserve(totalVertexCount);

    for (size_t i = 0; i < totalVertexCount; ++i) {
        _vertexIndices.push_back(static_cast<int>(i));
    }

    _vertices.reserve(totalVertexCount);
}

void FurAdapter::SetGeometry(const glm::Array<glm::Vector3>& deformedVertices)
{
    const FurCache& furCache = *_furCachePtr;
    size_t groupCount = furCache._curveGroups.size();
    size_t deformedIndex = 0;

    _vertices.clear();

    for (size_t igroup = 0; igroup < groupCount; ++igroup) {
        const FurCurveGroup& group = furCache._curveGroups[igroup];
        size_t ncurve = group._numVertices.size();
        for (size_t icurve = 0; icurve < ncurve; icurve += _curveIncr) {
            size_t nvert = group._numVertices[icurve];
            if (group._supportMeshId == _meshInFurIndex) {
                for (size_t ivert = 0; ivert < nvert; ++ivert) {
                    if (deformedIndex + ivert >= deformedVertices.size()) {
                        std::cerr << "deformed fur vertex overflow\n";
                        return;
                    }
                    _vertices.emplace_back(
                        deformedVertices[deformedIndex + ivert].getFloatValues());
                }
            }
            deformedIndex += nvert;
        }
    }

    if (_vertices.size() != _vertexIndices.size()) {
        std::cerr << "expected " << _vertexIndices.size() << " hairs, got "
                  << _vertices.size() << '\n';
    }
}

HdContainerDataSourceHandle FurAdapter::GetXformDataSource() const
{
    static const HdContainerDataSourceHandle identityXform =
        HdXformSchema::Builder()
        .SetMatrix(
            HdRetainedTypedSampledDataSource<GfMatrix4d>::New(GfMatrix4d(1.0)))
        .Build();

    return identityXform;
}

HdContainerDataSourceHandle FurAdapter::GetCurveDataSource() const
{
    return HdBasisCurvesSchema::Builder()
        .SetTopology(
            HdBasisCurvesTopologySchema::Builder()
            .SetCurveVertexCounts(
                HdRetainedTypedSampledDataSource<VtIntArray>::New(
                    _vertexCounts))
            .SetCurveIndices(
                HdRetainedTypedSampledDataSource<VtIntArray>::New(
                    _vertexIndices))
            .SetBasis(
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    UsdGeomTokens->catmullRom))
            .SetType(
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    UsdGeomTokens->cubic))
            .SetWrap(
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    UsdGeomTokens->nonperiodic))
            .Build())
        .Build();
}

HdContainerDataSourceHandle FurAdapter::GetPrimvarsDataSource() const
{
    HdContainerDataSourceHandle vertexDataSource =
        HdPrimvarSchema::Builder()
        .SetPrimvarValue(
            HdRetainedTypedSampledDataSource<VtVec3fArray>::New(_vertices))
        .SetInterpolation(
            HdPrimvarSchema::BuildInterpolationDataSource(
                HdPrimvarSchemaTokens->vertex))
        .SetRole(
            HdPrimvarSchema::BuildRoleDataSource(
                HdPrimvarSchemaTokens->point))
        .Build();

    return HdRetainedContainerDataSource::New(
        HdPrimvarsSchemaTokens->points,
        vertexDataSource);
}

}  // namespace glmhydra
