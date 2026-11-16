#pragma once

#include <cstdint>
#include <memory>

namespace trdrop {

//! How much of a GPU's dedicated memory all programs together use. Windows counts it for every process
//! (performance counter "GPU Adapter Memory"); Vulkan's memory budget only knows the own process. Elsewhere
//! nothing is known and dedicated_bytes() is always < 0.
class GpuMemoryUse {
public:
    //! luid: the adapter's LUID as Vulkan gives it (VkPhysicalDeviceIDProperties::deviceLUID)
    explicit GpuMemoryUse(const uint8_t (&luid)[8]);
    ~GpuMemoryUse();
    GpuMemoryUse(const GpuMemoryUse &) = delete;
    GpuMemoryUse & operator=(const GpuMemoryUse &) = delete;

    //! bytes of dedicated memory in use now, < 0 if not known
    long long dedicated_bytes();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace trdrop
