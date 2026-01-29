#pragma once

#include <pxr/imaging/hd/dataSource.h>
#include <pxr/imaging/hd/dataSourceTypeDefs.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/denseHashMap.h>

#include <memory>

namespace glmhydra {
namespace tools {

PXR_NAMESPACE_USING_DIRECTIVE

using PrimvarDSMap = TfDenseHashMap<TfToken, HdSampledDataSourceHandle, TfHash>;
using PrimvarDSMapRef = std::shared_ptr<PrimvarDSMap>;

HdContainerDataSourceHandle GetIdentityXformDataSource();
HdTokenDataSourceHandle GetConstantInterpDataSource();
HdTokenDataSourceHandle GetUniformInterpDataSource();
HdTokenDataSourceHandle GetFaceVaryingInterpDataSource();
HdTokenDataSourceHandle GetVertexInterpDataSource();
HdContainerDataSourceHandle GetMaterialDataSource(const SdfPath& material);

}  // namespace tools
}  // namespace glmhydra
