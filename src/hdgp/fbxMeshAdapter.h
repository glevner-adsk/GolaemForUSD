#pragma once

#include "hydraTools.h"
#include "meshDataSourceBase.h"

#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <glmArray.h>

#include <fbxsdk/fbxsdk_def.h>

#include <vector>

namespace glm {
    namespace crowdio {
        class CrowdFBXCharacter;
    }
}

#include <fbxsdk/fbxsdk_nsbegin.h>
class FbxTime;
#include <fbxsdk/fbxsdk_nsend.h>

namespace glmhydra {

class FbxMeshAdapter: public MeshDataSourceBase
{
public:
    FbxMeshAdapter(
        glm::crowdio::CrowdFBXCharacter& fbxCharacter, size_t meshIndex,
        const FBXSDK_NAMESPACE::FbxTime& fbxTime,
        const glm::Array<PXR_NS::HdSampledDataSource::Time>& shutterOffsets,
        const tools::DeformedVectors& deformedVertices,
        const tools::DeformedVectors& deformedNormals,
        int meshMaterialIndex);

    PXR_NS::HdContainerDataSourceHandle GetDataSource() const override;

private:
    using IntArrayDS =
        PXR_NS::HdRetainedTypedSampledDataSource<PXR_NS::VtIntArray>;
    using Vec3fArrayDS =
        PXR_NS::HdRetainedTypedMultisampledDataSource<PXR_NS::VtVec3fArray>;

    PXR_NS::HdContainerDataSourceHandle GetMeshDataSource() const;
    PXR_NS::HdContainerDataSourceHandle GetPrimvarsDataSource() const;

    PXR_NS::VtIntArray _vertexCounts;
    PXR_NS::VtIntArray _vertexIndices;
    std::vector<PXR_NS::VtVec3fArray> _vertices;
    std::vector<PXR_NS::HdSampledDataSource::Time> _shutterOffsets;
};

}  // namespace glmhydra
