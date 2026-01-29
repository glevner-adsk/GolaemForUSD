#include "furAdapter.h"

#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/legacyDisplayStyleSchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cmath>
#include <iostream>
#include <memory>

using glm::crowdio::FurCache;
using glm::crowdio::FurCurveGroup;

namespace glmhydra {

/*
 * Note that FurAdapter keeps a (smart) pointer to the fur cache, preventing it
 * from being deleted before the FurAdapter.
 *
 * Note also that the fur cache may contain more than one curve group, but it is
 * assumed here that all the information given for them in FurCurveGroup is in
 * fact shared by all of them: whether curves are cubic or linear, whether or
 * not they have widths and/or UVs, and additional float and vector properties.
 *
 * Call SetGeometry() afterwards with the fur vertex positions.
 */
FurAdapter::FurAdapter(
    FurCache::SP furCachePtr, size_t meshInFurIndex, float scale,
    float renderPercent, int refineLevel)
    : _furCachePtr(furCachePtr),
      _meshInFurIndex(meshInFurIndex),
      _curveIncr(std::lround(100.0f / renderPercent)),
      _refineLevel(refineLevel),
      _curveDegree(UsdGeomTokens->cubic)
{
    const FurCache& furCache = *_furCachePtr;

    // start by counting the number of visible curves and vertices

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

    if (totalCurveCount == 0) {
        return;
    }

    // some information is determined by the first curve group and is assumed to
    // be shared by all groups in the cache

    const FurCurveGroup& firstGroup = furCache._curveGroups[0];
    if (firstGroup._curveDegrees == 1) {
        _curveDegree = UsdGeomTokens->linear;
    }
    bool hasWidths = !firstGroup._widths.empty();
    if (hasWidths) {
        _widths.reserve(totalVertexCount);
    }

    // fill in vertex counts, indices, widths, etc.

    _vertexCounts.reserve(totalCurveCount);

    for (size_t igroup = 0; igroup < groupCount; ++igroup) {
        const FurCurveGroup& group = furCache._curveGroups[igroup];
        if (group._supportMeshId == _meshInFurIndex) {
            size_t ncurve = group._numVertices.size();
            for (size_t icurve = 0; icurve < ncurve; icurve += _curveIncr) {
                _vertexCounts.push_back(group._numVertices[icurve]);
                if (hasWidths) {
                    size_t nvert = group._numVertices[icurve];
                    if (group._widths.empty()) {
                        _widths.insert(_widths.end(), nvert, [](float *b, float *e) {
                            std::uninitialized_fill(b, e, 0.0f);
                        });
                    } else {
                        for (size_t ivert = 0; ivert < nvert; ++ivert) {
                            _widths.push_back(scale * group._widths[ivert]);
                        }
                    }
                }
            }
        }
    }

    _vertexIndices.reserve(totalVertexCount);

    for (size_t i = 0; i < totalVertexCount; ++i) {
        _vertexIndices.push_back(static_cast<int>(i));
    }

    _vertices.reserve(totalVertexCount);
}

/*
 * Sets the deformed fur vertices for the current frame.
 */
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
                        std::cerr << "FurAdapter: too many fur vertices\n";
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
        std::cerr << "FurAdapter: expected " << _vertexIndices.size()
                  << " hairs, got " << _vertices.size() << '\n';
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
                    _curveDegree))
            .SetWrap(
                HdRetainedTypedSampledDataSource<TfToken>::New(
                    UsdGeomTokens->nonperiodic))
            .Build())
        .Build();
}

HdContainerDataSourceHandle FurAdapter::GetPrimvarsDataSource() const
{
    VtTokenArray dataNames;
    VtArray<HdDataSourceBaseHandle> dataSources;

    dataNames.reserve(2);
    dataSources.reserve(2);

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

    dataNames.push_back(HdPrimvarsSchemaTokens->points);
    dataSources.push_back(vertexDataSource);

    if (!_widths.empty()) {
        HdContainerDataSourceHandle widthDataSource =
            HdPrimvarSchema::Builder()
            .SetPrimvarValue(
                HdRetainedTypedSampledDataSource<VtFloatArray>::New(_widths))
            .SetInterpolation(
                HdPrimvarSchema::BuildInterpolationDataSource(
                    HdPrimvarSchemaTokens->vertex))
            .Build();

        dataNames.push_back(HdPrimvarsSchemaTokens->widths);
        dataSources.push_back(widthDataSource);
    }

    return HdRetainedContainerDataSource::New(
        dataNames.size(), dataNames.cdata(), dataSources.cdata());
}

HdContainerDataSourceHandle FurAdapter::GetDisplayStyleDataSource() const
{
    return HdLegacyDisplayStyleSchema::Builder()
        .SetRefineLevel(HdRetainedTypedSampledDataSource<int>::New(_refineLevel))
        .Build();
}

HdContainerDataSourceHandle FurAdapter::GetDataSource() const
{
    VtTokenArray dataNames;
    VtArray<HdDataSourceBaseHandle> dataSources;

    dataNames.reserve(4);
    dataSources.reserve(4);

    dataNames.push_back(HdXformSchemaTokens->xform);
    dataSources.push_back(GetXformDataSource());

    dataNames.push_back(HdBasisCurvesSchemaTokens->basisCurves);
    dataSources.push_back(GetCurveDataSource());

    dataNames.push_back(HdPrimvarsSchemaTokens->primvars);
    dataSources.push_back(GetPrimvarsDataSource());

    if (!_widths.empty()) {
        dataNames.push_back(HdLegacyDisplayStyleSchemaTokens->displayStyle);
        dataSources.push_back(GetDisplayStyleDataSource());
    }

    return HdRetainedContainerDataSource::New(
        dataNames.size(), dataNames.cdata(), dataSources.cdata());
}

}  // namespace glmhydra
