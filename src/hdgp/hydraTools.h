#pragma once

#include <glmArray.h>
#include <glmVector3.h>

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/dataSourceTypeDefs.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/denseHashMap.h>
#include <pxr/base/vt/array.h>

#include <memory>

namespace glmhydra {
namespace tools {

PXR_NAMESPACE_USING_DIRECTIVE

/*
 * Type of the vector arrays found in glm::crowdio::OutputEntityGeoData. The
 * three dimensions correspond to the index of the frame being computed, the
 * index of the mesh or fur instance, and the index of the vector itself.
 */
using DeformedVectors = glm::Array<glm::Array<glm::Array<glm::Vector3>>>;

/*
 * A map of custom primvar names to the corresponding data sources, generates
 * from a crowd entity's shader and PP attributes.
 */
using PrimvarDSMap = TfDenseHashMap<TfToken, HdSampledDataSourceHandle, TfHash>;
using PrimvarDSMapRef = std::shared_ptr<PrimvarDSMap>;

HdContainerDataSourceHandle GetIdentityXformDataSource();
HdTokenDataSourceHandle GetConstantInterpDataSource();
HdTokenDataSourceHandle GetUniformInterpDataSource();
HdTokenDataSourceHandle GetFaceVaryingInterpDataSource();
HdTokenDataSourceHandle GetVertexInterpDataSource();
HdContainerDataSourceHandle GetMaterialDataSource(const SdfPath& material);

void CopyGlmVecArrayToVt(
    VtVec3fArray& dst, const glm::Array<glm::Vector3>& src);

}  // namespace tools
}  // namespace glmhydra
