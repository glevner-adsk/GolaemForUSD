#pragma once

#include "fileMeshAdapter.h"
#include "hydraTools.h"
#include "meshDataSourceBase.h"

#include <pxr/usd/sdf/path.h>
#include <pxr/base/tf/denseHashMap.h>

#include <memory>

namespace glmhydra {

PXR_NAMESPACE_USING_DIRECTIVE

/*
 * Adds xform, material and custom primvar data sources to the data sources for
 * a mesh's topology and geometry (provided by FileMeshAdapter). This class is
 * separated from FileMeshAdapter so that multiple instances can share the same
 * mesh but with different transformations and materials.
 */
class FileMeshInstance: public MeshDataSourceBase
{
public:
    FileMeshInstance(
        const std::shared_ptr<FileMeshAdapter>& adapter,
        const SdfPath& material, const tools::PrimvarDSMapRef& customPrimvars);

    void SetTransform(const float pos[3], const float rot[4], float scale);

    // TODO: variant of SetTransform() with shutter offsets

    HdContainerDataSourceHandle GetDataSource() const override;

    bool IsRigid() const override {
        return _adapter->IsRigid();
    }

private:
    HdContainerDataSourceHandle GetPrimvarsDataSource() const;

    std::shared_ptr<FileMeshAdapter> _adapter;
    SdfPath _material;
    const tools::PrimvarDSMapRef _customPrimvars;
    HdContainerDataSourceHandle _xform;
};

}  // namespace glmhydra
