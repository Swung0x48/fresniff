#include "gpu_memory.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>

#include <cstring>
#include <cwchar>
#include <string>
#include <vector>
#endif

namespace fresniff {

#ifdef _WIN32

struct GpuMemoryUse::Impl {
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    //! the start of the counter's instance names of this adapter, e.g. "luid_0x00000000_0x000117DA_" (any case)
    std::wstring prefix;
    std::vector<unsigned char> items;
};

GpuMemoryUse::GpuMemoryUse(const uint8_t (&luid)[8])
    : _impl(std::make_unique<Impl>())
{
    LUID value;
    static_assert(sizeof(value) == sizeof(luid), "LUID is 8 bytes");
    std::memcpy(&value, luid, sizeof(value));
    wchar_t prefix[64];
    std::swprintf(prefix, sizeof(prefix) / sizeof(prefix[0]), L"luid_0x%08lx_0x%08lx_",
                  static_cast<unsigned long>(value.HighPart), static_cast<unsigned long>(value.LowPart));
    _impl->prefix = prefix;
    if (PdhOpenQueryW(nullptr, 0, &_impl->query) != ERROR_SUCCESS)
    {
        _impl->query = nullptr;
        return;
    }
    // the English name works whatever the language of Windows is
    if (PdhAddEnglishCounterW(_impl->query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &_impl->counter) != ERROR_SUCCESS)
    {
        PdhCloseQuery(_impl->query);
        _impl->query = nullptr;
    }
}

GpuMemoryUse::~GpuMemoryUse()
{
    if (_impl->query) PdhCloseQuery(_impl->query);
}

long long GpuMemoryUse::dedicated_bytes()
{
    if (!_impl->query) return -1;
    if (PdhCollectQueryData(_impl->query) != ERROR_SUCCESS) return -1;
    DWORD size = 0;
    DWORD count = 0;
    PDH_STATUS status = PdhGetFormattedCounterArrayW(_impl->counter, PDH_FMT_LARGE | PDH_FMT_NOCAP100, &size, &count, nullptr);
    if (status != static_cast<PDH_STATUS>(PDH_MORE_DATA) || size == 0) return -1;
    _impl->items.resize(size);
    auto * items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W *>(_impl->items.data());
    status = PdhGetFormattedCounterArrayW(_impl->counter, PDH_FMT_LARGE | PDH_FMT_NOCAP100, &size, &count, items);
    if (status != ERROR_SUCCESS) return -1;
    // one instance per physical adapter of a linked adapter ("phys_0", "phys_1" ...)
    long long sum = 0;
    bool found = false;
    for (DWORD i = 0; i < count; ++i)
    {
        if (!items[i].szName || _wcsnicmp(items[i].szName, _impl->prefix.c_str(), _impl->prefix.size()) != 0) continue;
        if (items[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && items[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) continue;
        sum += items[i].FmtValue.largeValue;
        found = true;
    }
    return found ? sum : -1;
}

#else

struct GpuMemoryUse::Impl {};

GpuMemoryUse::GpuMemoryUse(const uint8_t (&)[8]) {}

GpuMemoryUse::~GpuMemoryUse() = default;

long long GpuMemoryUse::dedicated_bytes() { return -1; }

#endif

} // namespace fresniff
