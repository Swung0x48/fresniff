#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fresniff::vk_loader {

struct LoadedVulkan {
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    PFN_vkCreateInstance create_instance = nullptr;
    PFN_vkEnumerateInstanceLayerProperties enumerate_layers = nullptr;
    PFN_vkEnumerateInstanceExtensionProperties enumerate_extensions = nullptr;
};

inline bool env_flag(const char * name)
{
    const char * value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

inline LoadedVulkan load_vulkan_library(bool direct_moltenvk = false)
{
    LoadedVulkan loaded;
#ifdef _WIN32
    (void)direct_moltenvk;
    HMODULE library = LoadLibraryA("vulkan-1.dll");
    if (library)
    {
        loaded.get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            reinterpret_cast<void *>(GetProcAddress(library, "vkGetInstanceProcAddr")));
    }
#else
    std::vector<std::string> candidates;
#ifdef __APPLE__
    const char * sdk = std::getenv("VULKAN_SDK");
    const std::string sdk_dir = (sdk && *sdk) ? std::string(sdk) + "/lib/" : std::string();
#ifdef FRESNIFF_VULKAN_LIB_DIR
    const std::string build_dir = std::string(FRESNIFF_VULKAN_LIB_DIR).empty()
                                ? std::string()
                                : std::string(FRESNIFF_VULKAN_LIB_DIR) + "/";
#else
    const std::string build_dir;
#endif
    if (direct_moltenvk)
    {
        if (!sdk_dir.empty()) candidates.push_back(sdk_dir + "libMoltenVK.dylib");
        candidates.push_back("libMoltenVK.dylib");
        if (!build_dir.empty()) candidates.push_back(build_dir + "libMoltenVK.dylib");
        candidates.push_back("/opt/homebrew/lib/libMoltenVK.dylib");
        candidates.push_back("/usr/local/lib/libMoltenVK.dylib");
    }
    else
    {
        if (!sdk_dir.empty())
        {
            candidates.push_back(sdk_dir + "libvulkan.1.dylib");
            candidates.push_back(sdk_dir + "libvulkan.dylib");
            candidates.push_back(sdk_dir + "libMoltenVK.dylib");
        }
        candidates.push_back("libvulkan.1.dylib");
        candidates.push_back("libvulkan.dylib");
        candidates.push_back("libMoltenVK.dylib");
        if (!build_dir.empty())
        {
            candidates.push_back(build_dir + "libvulkan.1.dylib");
            candidates.push_back(build_dir + "libMoltenVK.dylib");
        }
        candidates.push_back("/opt/homebrew/lib/libvulkan.1.dylib");
        candidates.push_back("/opt/homebrew/lib/libMoltenVK.dylib");
        candidates.push_back("/usr/local/lib/libvulkan.1.dylib");
        candidates.push_back("/usr/local/lib/libMoltenVK.dylib");
    }
#else
    (void)direct_moltenvk;
    candidates = { "libvulkan.so.1", "libvulkan.so" };
#endif
    for (const std::string & name : candidates)
    {
        void * library = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library) continue;
        loaded.get_instance_proc_addr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
        if (loaded.get_instance_proc_addr) break;
        dlclose(library);
    }
#endif
    if (!loaded.get_instance_proc_addr)
    {
        throw std::runtime_error("the Vulkan loader library could not be loaded");
    }
    loaded.create_instance = reinterpret_cast<PFN_vkCreateInstance>(
        loaded.get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
    loaded.enumerate_layers = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
        loaded.get_instance_proc_addr(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties"));
    loaded.enumerate_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        loaded.get_instance_proc_addr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    if (!loaded.create_instance)
    {
        throw std::runtime_error("the Vulkan loader library is invalid");
    }
    return loaded;
}

inline VkResult try_create_instance(const LoadedVulkan & global, uint32_t api_version, VkInstance & out_instance)
{
    VkApplicationInfo application {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "fresniff";
    application.applicationVersion = 1;
    application.pEngineName = "fresniff";
    application.engineVersion = 1;
    application.apiVersion = api_version;

    // FRESNIFF_VK_VALIDATION=1 enables the Khronos validation layer if it is installed (debugging)
    std::vector<const char *> layers;
    if (env_flag("FRESNIFF_VK_VALIDATION") && global.enumerate_layers)
    {
        uint32_t count = 0;
        global.enumerate_layers(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        if (count > 0) global.enumerate_layers(&count, available.data());
        for (const VkLayerProperties & layer : available)
        {
            if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            {
                layers.push_back("VK_LAYER_KHRONOS_validation");
                break;
            }
        }
    }

    std::vector<const char *> extensions;
    VkInstanceCreateFlags flags = 0;
    if (global.enumerate_extensions)
    {
        uint32_t count = 0;
        if (global.enumerate_extensions(nullptr, &count, nullptr) == VK_SUCCESS && count > 0)
        {
            std::vector<VkExtensionProperties> available(count);
            if (global.enumerate_extensions(nullptr, &count, available.data()) == VK_SUCCESS)
            {
                available.resize(count);
                auto has_ext = [&](const char * name) {
                    for (const VkExtensionProperties & ext : available)
                    {
                        if (std::strcmp(ext.extensionName, name) == 0) return true;
                    }
                    return false;
                };
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
                if (has_ext("VK_KHR_portability_enumeration"))
                {
                    extensions.push_back("VK_KHR_portability_enumeration");
                    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
                }
#endif
                if (api_version == VK_API_VERSION_1_0 && has_ext("VK_KHR_get_physical_device_properties2"))
                {
                    extensions.push_back("VK_KHR_get_physical_device_properties2");
                }
            }
        }
    }

    VkInstanceCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.flags = flags;
    info.pApplicationInfo = &application;
    info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.data();
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    return global.create_instance(&info, nullptr, &out_instance);
}

inline void create_instance(uint32_t api_version, LoadedVulkan & out_global, VkInstance & out_instance)
{
    out_global = load_vulkan_library(false);
    VkResult result = try_create_instance(out_global, api_version, out_instance);
#ifdef __APPLE__
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        out_global = load_vulkan_library(true);
        result = try_create_instance(out_global, api_version, out_instance);
    }
#endif
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateInstance failed (VkResult " + std::to_string(static_cast<int>(result)) + ")");
    }
}

inline std::vector<const char *> device_portability_extensions(PFN_vkEnumerateDeviceExtensionProperties enum_ext, VkPhysicalDevice physical_device)
{
    std::vector<const char *> result;
    if (!enum_ext) return result;
    uint32_t count = 0;
    if (enum_ext(physical_device, nullptr, &count, nullptr) != VK_SUCCESS || count == 0) return result;
    std::vector<VkExtensionProperties> available(count);
    if (enum_ext(physical_device, nullptr, &count, available.data()) != VK_SUCCESS) return result;
    available.resize(count);
    for (const VkExtensionProperties & ext : available)
    {
        if (std::strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0)
        {
            result.push_back("VK_KHR_portability_subset");
            break;
        }
    }
    return result;
}

} // namespace fresniff::vk_loader
