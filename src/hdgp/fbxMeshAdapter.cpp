#include "fbxMeshAdapter.h"

#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopologySchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <fbxsdk.h>

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using Time = HdSampledDataSource::Time;

TF_DEFINE_PRIVATE_TOKENS(
    fbxMeshAdapterTokens,
    (st)
);

namespace glmhydra {

FbxMeshAdapter::FbxMeshAdapter(
    FbxMesh& mesh,
    const glm::Array<Time>& shutterOffsets,
    const tools::DeformedVectors& deformedVertices,
    const tools::DeformedVectors& deformedNormals,
    size_t meshIndex, int meshMaterialIndex)
    : _shutterOffsets(shutterOffsets.begin(), shutterOffsets.end())
{
    // polygons bound to a different material are ignored, meaning some number
    // of vertices may be superfluous, so we construct a map of original vertex
    // indices to used vertex indices

    const FbxLayer *fbxLayer0 = mesh.GetLayer(0);
    FbxLayerElementArrayTemplate<int> const *mtlArray = nullptr;

    if (fbxLayer0) {
        const FbxLayerElementMaterial *materials = fbxLayer0->GetMaterials();
        if (materials) {
            mtlArray = &materials->GetIndexArray();
        }
    }

    int allVertexCount = mesh.GetControlPointsCount();
    int allPolyCount = mesh.GetPolygonCount();
    std::vector<int> vertexMap(allVertexCount, -1);
    int usedVertexCount = 0;
    int usedPolyCount = 0;
    int usedPolyVertexCount = 0;

    for (int ipoly = 0; ipoly < allPolyCount; ++ipoly) {
        if (mtlArray && mtlArray->GetAt(ipoly) != meshMaterialIndex) {
            continue;
        }
        int nvert = mesh.GetPolygonSize(ipoly);
        for (int ivert = 0; ivert < nvert; ++ivert) {
            int vertIndex = mesh.GetPolygonVertex(ipoly, ivert);
            if (vertexMap[vertIndex] < 0) {
                vertexMap[vertIndex] = usedVertexCount++;
            }
        }
        ++usedPolyCount;
        usedPolyVertexCount += nvert;
    }

    // copy the vertex counts and indices of the visible polygons only

    _vertexCounts.reserve(usedPolyCount);
    _vertexIndices.reserve(usedPolyVertexCount);

    for (int ipoly = 0; ipoly < allPolyCount; ++ipoly) {
        if (mtlArray && mtlArray->GetAt(ipoly) != meshMaterialIndex) {
            continue;
        }
        int nvert = mesh.GetPolygonSize(ipoly);
        for (int ivert = 0; ivert < nvert; ++ivert) {
            int vertIndex = mesh.GetPolygonVertex(ipoly, ivert);
            _vertexIndices.push_back(vertexMap[vertIndex]);
        }
        _vertexCounts.push_back(nvert);
    }

    // for each time sample, copy the deformed vertices we need

    size_t sampleCount = shutterOffsets.size();
    _vertices.resize(sampleCount);

    assert(deformedVertices.size() == sampleCount);

    for (size_t i = 0; i < sampleCount; ++i) {
        assert(deformedVertices[i].size() > meshIndex);
        const glm::Array<glm::Vector3>& src = deformedVertices[i][meshIndex];
        assert(src.size() == allVertexCount);
        VtVec3fArray& dst = _vertices[i];
        dst.resize(usedVertexCount);
        for (int ivert = 0; ivert < allVertexCount; ++ivert) {
            int myIndex = vertexMap[ivert];
            if (myIndex >= 0) {
                dst[myIndex].Set(src[ivert].getFloatValues());
            }
        }
    }

    // TODO: handle transformation matrices
}

HdContainerDataSourceHandle FbxMeshAdapter::GetMeshDataSource() const
{
    return HdMeshSchema::Builder()
        .SetTopology(
            HdMeshTopologySchema::Builder()
            .SetFaceVertexCounts(IntArrayDS::New(_vertexCounts))
            .SetFaceVertexIndices(IntArrayDS::New(_vertexIndices))
            .Build())
        .SetSubdivisionScheme(
            HdRetainedTypedSampledDataSource<TfToken>::New(UsdGeomTokens->none))
        .Build();
}

HdContainerDataSourceHandle FbxMeshAdapter::GetPrimvarsDataSource() const
{
    VtTokenArray dataNames;
    VtArray<HdDataSourceBaseHandle> dataSources;
    size_t capacity = 3;  // points, normals and UVs

    dataNames.reserve(capacity);
    dataSources.reserve(capacity);

    // vertex data source

    HdContainerDataSourceHandle vertexDataSource =
        HdPrimvarSchema::Builder()
        .SetPrimvarValue(
            Vec3fArrayDS::New(
                _shutterOffsets.size(),
                const_cast<Time*>(_shutterOffsets.data()),
                const_cast<VtVec3fArray*>(_vertices.data())))
        .SetInterpolation(tools::GetVertexInterpDataSource())
        .SetRole(
            HdPrimvarSchema::BuildRoleDataSource(
                HdPrimvarSchemaTokens->point))
        .Build();

    dataNames.push_back(HdPrimvarsSchemaTokens->points);
    dataSources.push_back(vertexDataSource);

    // the final primvars data source contains the vertices, normals and UVs

    return HdRetainedContainerDataSource::New(
        dataNames.size(), dataNames.cdata(), dataSources.cdata());
}

HdContainerDataSourceHandle FbxMeshAdapter::GetDataSource() const
{
    VtTokenArray dataNames;
    VtArray<HdDataSourceBaseHandle> dataSources;

    dataNames.reserve(4);
    dataSources.reserve(4);

    dataNames.push_back(HdXformSchemaTokens->xform);
    dataSources.push_back(tools::GetIdentityXformDataSource());

    dataNames.push_back(HdMeshSchemaTokens->mesh);
    dataSources.push_back(GetMeshDataSource());

    dataNames.push_back(HdPrimvarsSchemaTokens->primvars);
    dataSources.push_back(GetPrimvarsDataSource());

    /*
    if (!_material.IsEmpty()) {
        dataNames.push_back(HdMaterialBindingsSchemaTokens->materialBindings);
        dataSources.push_back(tools::GetMaterialDataSource(_material));
    }
    */

    return HdRetainedContainerDataSource::New(
        dataNames.size(), dataNames.cdata(), dataSources.cdata());
}

}  // namespace glmhydra
