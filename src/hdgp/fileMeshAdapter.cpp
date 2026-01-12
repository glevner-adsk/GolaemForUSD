#include "fileMeshAdapter.h"

#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/meshSchema.h>
#include <pxr/imaging/hd/meshTopologySchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/usd/usdGeom/tokens.h>

#include <cassert>

PXR_NAMESPACE_USING_DIRECTIVE

namespace glmHydra {

TF_DEFINE_PRIVATE_TOKENS(
    fileMeshAdapterTokens,
    (st)
);

/*
 * The FileMeshAdapter constructor makes copies of all the data it
 * needs, so all the arguments can be deleted afterwards. It does
 * little else, leaving as much work as possible for the methods which
 * return the data sources, because they may be called from multiple
 * threads.
 */
FileMeshAdapter::FileMeshAdapter(
    const glm::crowdio::GlmFileMesh& fileMesh,
    const glm::Array<glm::Vector3>& deformedVertices,
    const glm::Array<glm::Vector3>& deformedNormals,
    const SdfPath& material, const PrimvarDSMapRef& customPrimvars)
    : _vertexCounts(fileMesh._polygonCount),
      _vertexIndices(fileMesh._polygonsTotalVertexCount),
      _vertices(fileMesh._vertexCount),
      _normals(fileMesh._normalCount),
      _normalMode(glm::crowdio::GlmNormalMode(fileMesh._normalMode)),
      _uvMode(glm::crowdio::GlmUVMode(fileMesh._uvMode)),
      _material(material),
      _customPrimvars(customPrimvars)
{
    assert(deformedVertices.size() == _vertices.size());
    assert(deformedNormals.size() == _normals.size());

    for (int i = 0; i < _vertexCounts.size(); ++i) {
        _vertexCounts[i] = fileMesh._polygonsVertexCount[i];
    }

    for (int i = 0; i < _vertexIndices.size(); ++i) {
        _vertexIndices[i] = fileMesh._polygonsVertexIndices[i];
    }

    for (int i = 0; i < _vertices.size(); ++i) {
        _vertices[i].Set(deformedVertices[i].getFloatValues());
    }

    if (_normals.size() > 0) {
        if (_normalMode ==
            glm::crowdio::GLM_NORMAL_PER_POLYGON_VERTEX_INDEXED) {
            _normalIndices.resize(fileMesh._polygonsTotalVertexCount);
            for (int i = 0; i < _normalIndices.size(); ++i) {
                _normalIndices[i] = fileMesh._polygonsNormalIndices[i];
            }
        }
        for (int i = 0; i < _normals.size(); ++i) {
            _normals[i].Set(deformedNormals[i].getFloatValues());
        }
    }

    // note that if there are multiple UV sets, we only take the
    // first; the others are ignored

    if (fileMesh._uvSetCount > 0 && fileMesh._uvCoordCount[0] > 0) {
        _uvs.resize(fileMesh._uvCoordCount[0]);
        if (_uvMode ==
            glm::crowdio::GLM_UV_PER_POLYGON_VERTEX_INDEXED) {
            _uvIndices.resize(fileMesh._polygonsTotalVertexCount);
            for (int i = 0; i < _uvIndices.size(); ++i) {
                _uvIndices[i] = fileMesh._polygonsUVIndices[i];
            }
        }
        for (int i = 0; i < _uvs.size(); ++i) {
            _uvs[i].Set(fileMesh._us[0][i], fileMesh._vs[0][i]);
        }
    }
}

HdContainerDataSourceHandle FileMeshAdapter::GetMeshDataSource() const
{
    return HdMeshSchema::Builder()
        .SetTopology(
            HdMeshTopologySchema::Builder()
            .SetFaceVertexCounts(IntArrayDS::New(_vertexCounts))
            .SetFaceVertexIndices(IntArrayDS::New(_vertexIndices))
            .Build())
        .SetSubdivisionScheme(
            HdRetainedTypedSampledDataSource<TfToken>::New(
                UsdGeomTokens->none))
        .Build();
}

HdContainerDataSourceHandle
FileMeshAdapter::GetPrimvarsDataSource() const
{
    VtTokenArray dataNames;
    VtArray<HdDataSourceBaseHandle> dataSources;
    size_t capacity = 3;  // points, normals and UVs

    if (_customPrimvars) {
        capacity += _customPrimvars->size();
    }

    dataNames.reserve(capacity);
    dataSources.reserve(capacity);

    // vertex data source

    HdContainerDataSourceHandle vertexDataSource =
        HdPrimvarSchema::Builder()
        .SetPrimvarValue(Vec3fArrayDS::New(_vertices))
        .SetInterpolation(
            HdPrimvarSchema::BuildInterpolationDataSource(
                HdPrimvarSchemaTokens->vertex))
        .SetRole(
            HdPrimvarSchema::BuildRoleDataSource(
                HdPrimvarSchemaTokens->point))
        .Build();

    dataNames.push_back(HdPrimvarsSchemaTokens->points);
    dataSources.push_back(vertexDataSource);

    // normal data source, if the mesh contains normals

    HdContainerDataSourceHandle normalDataSource;

    if (_normals.size() > 0) {
        HdPrimvarSchema::Builder normalBuilder;

        // normals may or may not be indexed

        if (_normalMode ==
            glm::crowdio::GLM_NORMAL_PER_POLYGON_VERTEX_INDEXED) {
            normalBuilder.SetIndexedPrimvarValue(
                Vec3fArrayDS::New(_normals));
            normalBuilder.SetIndices(IntArrayDS::New(_normalIndices));
        } else {
            normalBuilder.SetPrimvarValue(Vec3fArrayDS::New(_normals));
        }

        // normals may or may not be shared by polygons using the
        // same vertices

        if (_normalMode ==
            glm::crowdio::GLM_NORMAL_PER_CONTROL_POINT) {
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

        dataNames.push_back(HdPrimvarsSchemaTokens->normals);
        dataSources.push_back(normalBuilder.Build());
    }

    // UV data source, if the mesh contains UVs

    HdContainerDataSourceHandle uvDataSource;

    if (_uvs.size() > 0) {
        HdPrimvarSchema::Builder uvBuilder;

        // UVs may or may not be indexed

        if (_uvMode ==
            glm::crowdio::GLM_UV_PER_POLYGON_VERTEX_INDEXED) {
            uvBuilder.SetIndexedPrimvarValue(Vec2fArrayDS::New(_uvs));
            uvBuilder.SetIndices(IntArrayDS::New(_uvIndices));
        } else {
            uvBuilder.SetPrimvarValue(Vec2fArrayDS::New(_uvs));
        }

        // uvs may or may not be shared by polygons using the same
        // vertices

        if (_uvMode ==
            glm::crowdio::GLM_UV_PER_CONTROL_POINT) {
            uvBuilder.SetInterpolation(
                HdPrimvarSchema::BuildInterpolationDataSource(
                    HdPrimvarSchemaTokens->vertex));
        } else {
            uvBuilder.SetInterpolation(
                HdPrimvarSchema::BuildInterpolationDataSource(
                    HdPrimvarSchemaTokens->faceVarying));
        }

        uvBuilder.SetRole(
            HdPrimvarSchema::BuildRoleDataSource(
                HdPrimvarSchemaTokens->textureCoordinate));

        dataNames.push_back(fileMeshAdapterTokens->st);
        dataSources.push_back(uvBuilder.Build());
    }

    // custom primvars

    if (_customPrimvars) {
        for (auto it: *_customPrimvars) {
            dataNames.push_back(it.first);
            dataSources.push_back(
                HdPrimvarSchema::Builder()
                .SetPrimvarValue(it.second)
                .SetInterpolation(
                    HdPrimvarSchema::BuildInterpolationDataSource(
                        HdPrimvarSchemaTokens->constant))
                .Build());
        }
    }

    // the final primvars data source contains the vertices,
    // normals, UVs and custom primvars

    return HdRetainedContainerDataSource::New(
        dataNames.size(), dataNames.cdata(), dataSources.cdata());
}

HdContainerDataSourceHandle
FileMeshAdapter::GetMaterialDataSource() const
{
    if (_material.IsEmpty()) {
        return HdContainerDataSourceHandle();
    }

    return HdRetainedContainerDataSource::New(
        HdMaterialBindingsSchemaTokens->allPurpose,
        HdMaterialBindingSchema::Builder()
        .SetPath(
            HdRetainedTypedSampledDataSource<SdfPath>::New(_material))
        .Build());
}

}  // namespace golaemHydra
