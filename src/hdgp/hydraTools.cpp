#include "hydraTools.h"

#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/xformSchema.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace glmhydra {
namespace tools {

HdContainerDataSourceHandle GetIdentityXformDataSource()
{
    static const HdContainerDataSourceHandle identityXform =
        HdXformSchema::Builder()
        .SetMatrix(
            HdRetainedTypedSampledDataSource<GfMatrix4d>::New(GfMatrix4d(1.0)))
        .Build();

    return identityXform;
}

HdTokenDataSourceHandle GetConstantInterpDataSource()
{
    static const HdTokenDataSourceHandle constantInterp =
        HdPrimvarSchema::BuildInterpolationDataSource(
            HdPrimvarSchemaTokens->constant);

    return constantInterp;
}

HdTokenDataSourceHandle GetUniformInterpDataSource()
{
    static const HdTokenDataSourceHandle uniformInterp =
        HdPrimvarSchema::BuildInterpolationDataSource(
            HdPrimvarSchemaTokens->uniform);

    return uniformInterp;
}

HdTokenDataSourceHandle GetFaceVaryingInterpDataSource()
{
    static const HdTokenDataSourceHandle faceVaryingInterp =
        HdPrimvarSchema::BuildInterpolationDataSource(
            HdPrimvarSchemaTokens->faceVarying);

    return faceVaryingInterp;
}

HdTokenDataSourceHandle GetVertexInterpDataSource()
{
    static const HdTokenDataSourceHandle vertexInterp =
        HdPrimvarSchema::BuildInterpolationDataSource(
            HdPrimvarSchemaTokens->vertex);

    return vertexInterp;
}

HdContainerDataSourceHandle GetMaterialDataSource(const SdfPath& material)
{
    return HdRetainedContainerDataSource::New(
        HdMaterialBindingsSchemaTokens->allPurpose,
        HdMaterialBindingSchema::Builder()
        .SetPath(HdRetainedTypedSampledDataSource<SdfPath>::New(material))
        .Build());
}

}  // namespace tools
}  // namespace glmhydra
