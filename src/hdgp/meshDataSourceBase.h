#pragma once

#include <pxr/imaging/hd/retainedDataSource.h>

namespace glmhydra {

/*
 * Interface implemented by the FileMeshInstance and FbxMeshAdapter classes, so
 * that the plugin need not know whether the mesh prims they generate come from
 * a GCG or FBX character.
 */
class MeshDataSourceBase
{
public:
    MeshDataSourceBase() {}
    virtual ~MeshDataSourceBase() {}

    virtual PXR_NS::HdContainerDataSourceHandle GetDataSource() const = 0;

    virtual bool IsRigid() const {
        return false;
    }
};

}  // namespace glmhydra
