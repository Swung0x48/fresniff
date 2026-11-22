// macOS GPU pipeline: decodes via FFmpeg VideoToolbox (falling back to FFmpeg CPU decoding for codecs
// VideoToolbox does not decode), runs all 11 Vulkan compute shaders on MoltenVK in Apple Silicon unified
// memory, and encodes motion / resolution videos directly from NV12 via FFmpeg (h264_videotoolbox).

#include "frame_pipeline.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "vulkan/vk_loader.h"

#include <CoreVideo/CoreVideo.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/version.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ffmpeg/ffmpeg_decoder.h"
#include "ffmpeg/ffmpeg_encoder.h"
#include "ffmpeg/yuv_picture.h"
#include "vulkan/vk_motion_shared.h"

#if LIBAVUTIL_VERSION_MAJOR < 60
#error "the macOS Vulkan pipeline needs FFmpeg 8.0 or newer"
#endif

namespace fresniff {

namespace {

const uint32_t kYuvDiffSpirv[] = {
#include "yuv_diff.spv.inc"
};

const uint32_t kMotionPyramidSpirv[] = {
#include "motion_pyramid.spv.inc"
};

const uint32_t kMotionLkSpirv[] = {
#include "motion_lk.spv.inc"
};

const uint32_t kMotionSelectSpirv[] = {
#include "motion_select.spv.inc"
};

const uint32_t kMotionTextureSpirv[] = {
#include "motion_texture.spv.inc"
};

const uint32_t kOverlayBaseSpirv[] = {
#include "overlay_base.spv.inc"
};

const uint32_t kOverlayBlurSpirv[] = {
#include "overlay_blur.spv.inc"
};

const uint32_t kOverlayDrawSpirv[] = {
#include "overlay_draw.spv.inc"
};

const uint32_t kResolutionLatticeSpirv[] = {
#include "resolution_lattice.spv.inc"
};

const uint32_t kResolutionEdgeSpirv[] = {
#include "resolution_edge.spv.inc"
};

const uint32_t kResolutionStairsSpirv[] = {
#include "resolution_stairs.spv.inc"
};

//! must match resolution_lattice.comp
struct LatticePush {
    uint32_t width;
    uint32_t luma_row_words;
    uint32_t rows;
    uint32_t mode;
    uint32_t offset;
    uint32_t row0;
};

//! must match resolution_edge.comp
struct EdgePush {
    uint32_t width;
    uint32_t luma_row_words;
    uint32_t row0;
    uint32_t segments;
    uint32_t edge;
};

//! must match resolution_stairs.comp
struct StairsPush {
    uint32_t width;
    uint32_t luma_row_words;
    uint32_t row0;
    uint32_t bands;
    uint32_t edge;
};

//! motion video frames being drawn at the same time per stream
constexpr size_t kDrawSlots = 3;

//! sets of frames in flight
constexpr size_t kSlots = 3;

//! ring buffer depth for host-written plane buffers (kSlots in flight + 1 idle slot for the CPU write)
constexpr size_t kPlaneBuffers = kSlots + 1;

constexpr uint32_t kBindings = 5;

uint32_t align4(uint32_t value) { return (value + 3u) & ~3u; }

template <typename T>
T vk_init(VkStructureType type)
{
    T info {};
    info.sType = type;
    return info;
}

struct NonLumaFormatError final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct NoFramesError final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PixelBufferLock {
    CVPixelBufferRef buffer = nullptr;
    explicit PixelBufferLock(CVPixelBufferRef buf)
        : buffer(buf)
    {
        if (CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess)
        {
            buffer = nullptr;
            throw std::runtime_error("CVPixelBufferLockBaseAddress failed");
        }
    }
    ~PixelBufferLock()
    {
        if (buffer) CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    }
    PixelBufferLock(const PixelBufferLock &) = delete;
    PixelBufferLock & operator=(const PixelBufferLock &) = delete;
};

#define FRESNIFF_VK_MAC_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)

#define FRESNIFF_VK_MAC_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdDispatch) \
    X(vkCmdDispatchIndirect) \
    X(vkQueueSubmit) \
    X(vkCreateFence) \
    X(vkDestroyFence) \
    X(vkWaitForFences) \
    X(vkResetFences) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkUnmapMemory) \
    X(vkInvalidateMappedMemoryRanges) \
    X(vkFlushMappedMemoryRanges) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkDeviceWaitIdle)

void check(VkResult result, const char * what)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(result)) + ")");
    }
}

bool is_vt_nv12(OSType pixel_format)
{
    return pixel_format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        || pixel_format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
}

class VulkanMacosPipeline final : public FramePipeline {
public:
    VulkanMacosPipeline(const std::vector<std::string> & paths, int device_index)
        : _videos(paths.size())
    {
        try
        {
            // cheap checks first before spinning up Vulkan / VideoToolbox
            for (const std::string & path : paths)
            {
                const ffmpeg::Probe probe = ffmpeg::probe(path);
                if (!probe.opened) throw std::runtime_error(probe.error);
                if (probe.format != AV_PIX_FMT_NONE && !ffmpeg::is_luma_format(probe.format))
                {
                    throw NonLumaFormatError("pixel format " + ffmpeg::format_name(probe.format) + " is no 8-bit YUV format");
                }
            }

            _create_instance_and_device(device_index);
            _create_pipeline();
            _pyramid = _create_compute(kMotionPyramidSpirv, sizeof(kMotionPyramidSpirv), 2, sizeof(PyramidPush));
            _lk = _create_compute(kMotionLkSpirv, sizeof(kMotionLkSpirv), 4, sizeof(LkPush));
            _select = _create_compute(kMotionSelectSpirv, sizeof(kMotionSelectSpirv), 3, sizeof(SelectPush));
            _texture = _create_compute(kMotionTextureSpirv, sizeof(kMotionTextureSpirv), 4, sizeof(TexturePush));
            _overlay_base = _create_compute(kOverlayBaseSpirv, sizeof(kOverlayBaseSpirv), 3, sizeof(OverlayBasePush));
            _overlay_blur = _create_compute(kOverlayBlurSpirv, sizeof(kOverlayBlurSpirv), 3, sizeof(OverlayBlurPush));
            _overlay_draw = _create_compute(kOverlayDrawSpirv, sizeof(kOverlayDrawSpirv), 7, sizeof(OverlayDrawPush));
            _lattice = _create_compute(kResolutionLatticeSpirv, sizeof(kResolutionLatticeSpirv), 2, sizeof(LatticePush));
            _edge = _create_compute(kResolutionEdgeSpirv, sizeof(kResolutionEdgeSpirv), 2, sizeof(EdgePush));
            _stairs = _create_compute(kResolutionStairsSpirv, sizeof(kResolutionStairsSpirv), 2, sizeof(StairsPush));
            _gpu_videos.resize(paths.size());
            _create_slots();

            if (av_hwdevice_ctx_create(&_vt_device, AV_HWDEVICE_TYPE_VIDEOTOOLBOX, nullptr, nullptr, 0) < 0)
            {
                _vt_device = nullptr;
            }

            for (size_t i = 0; i < paths.size(); ++i)
            {
                _open_video(_videos[i], paths[i]);
            }
        }
        catch (...)
        {
            _release();
            throw;
        }
    }

    ~VulkanMacosPipeline() override { _release(); }

    VulkanMacosPipeline(const VulkanMacosPipeline &) = delete;
    VulkanMacosPipeline & operator=(const VulkanMacosPipeline &) = delete;

    std::string description() const override
    {
        bool any_vt = false;
        bool all_vt = !_videos.empty();
        for (const VideoState & v : _videos)
        {
            any_vt |= v.videotoolbox;
            all_vt &= v.videotoolbox;
        }
        const std::string decoder = all_vt ? "videotoolbox decode"
                                  : any_vt ? "videotoolbox / ffmpeg cpu decode"
                                           : "ffmpeg cpu decode";
        return decoder + " + vulkan analysis (" + _device_name + ", device " + std::to_string(_device_index) + "), yuv"
             + (_motion ? ", motion on the GPU" : "") + (_resolution ? ", resolution on the GPU" : "");
    }

    void enable_motion() override
    {
        if (_max_group_size < kMotionGroupSize || _max_group_invocations < kMotionGroupSize)
        {
            throw std::runtime_error("the Vulkan device '" + _device_name + "' can't run the motion analysis (workgroups of "
                                   + std::to_string(kMotionGroupSize) + " needed), use --decode ffmpeg");
        }
        _motion = true;
    }

    void enable_motion_detail(bool native) override
    {
        _motion_detail = true;
        _native_pictures = native;
    }

    void enable_resolution() override
    {
        if (!_motion) throw std::logic_error("the resolution analysis needs the motion analysis");
        _resolution = true;
    }

    std::unique_ptr<MotionVideoTarget> start_motion_video(size_t index, const std::string & path, double fps, std::string & reason) override
    {
        std::vector<std::unique_ptr<MotionVideoTarget>> targets = start_videos(index, { path }, fps, reason);
        return targets.empty() ? nullptr : std::move(targets.front());
    }

    std::vector<std::unique_ptr<MotionVideoTarget>> start_videos(size_t index, const std::vector<std::string> & paths, double fps, std::string & reason) override
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        std::vector<std::unique_ptr<MotionVideoTarget>> targets;
        if (paths.empty()) return targets;
        try
        {
            if (gpu.active) throw std::logic_error("the videos of an input are started together");
            const AVFrame * frame = _videos.at(index).first_frame;
            if (!frame) throw std::logic_error("the motion video has to be started before the first frame");
            const cv::Size size = video_picture_size(frame->width, frame->height, _native_pictures);

            for (const std::string & path : paths)
            {
                gpu.streams.emplace_back();
                GpuStream & stream = gpu.streams.back();
                stream.encoder = std::make_unique<ffmpeg::EncodedVideo>(path, size.width, size.height, fps, true, _videos[index].color);

                VkDescriptorPoolSize pool_size { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (7 + 3 + 3) * static_cast<uint32_t>(kDrawSlots) };
                auto pool_info = vk_init<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
                pool_info.maxSets = 3 * static_cast<uint32_t>(kDrawSlots);
                pool_info.poolSizeCount = 1;
                pool_info.pPoolSizes = &pool_size;
                check(_fn.vkCreateDescriptorPool(_device, &pool_info, nullptr, &stream.descriptor_pool), "vkCreateDescriptorPool");

                for (DrawSlot & dslot : stream.draw_slots)
                {
                    auto cmd_info = vk_init<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
                    cmd_info.commandPool = _command_pool;
                    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                    cmd_info.commandBufferCount = 1;
                    check(_fn.vkAllocateCommandBuffers(_device, &cmd_info, &dslot.command_buffer), "vkAllocateCommandBuffers");

                    auto fence_info = vk_init<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
                    check(_fn.vkCreateFence(_device, &fence_info, nullptr, &dslot.fence), "vkCreateFence");

                    auto set_info = vk_init<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
                    set_info.descriptorPool = stream.descriptor_pool;
                    set_info.descriptorSetCount = 1;
                    set_info.pSetLayouts = &_overlay_draw.set_layout;
                    check(_fn.vkAllocateDescriptorSets(_device, &set_info, &dslot.set), "vkAllocateDescriptorSets");
                    set_info.pSetLayouts = &_overlay_blur.set_layout;
                    check(_fn.vkAllocateDescriptorSets(_device, &set_info, &dslot.blur_set_a), "vkAllocateDescriptorSets");
                    check(_fn.vkAllocateDescriptorSets(_device, &set_info, &dslot.blur_set_b), "vkAllocateDescriptorSets");

                    dslot.av_frame = av_frame_alloc();
                    if (!dslot.av_frame) throw std::runtime_error("out of memory");
                    dslot.av_frame->format = stream.encoder->format();
                    dslot.av_frame->width = size.width;
                    dslot.av_frame->height = size.height;
                    if (av_frame_get_buffer(dslot.av_frame, 0) < 0) throw std::runtime_error("out of memory");
                }
            }

            gpu.width = static_cast<uint32_t>(size.width);
            gpu.height = static_cast<uint32_t>(size.height);
            gpu.row_words = (gpu.width + 3) / 4;
            gpu.picture_bytes = static_cast<VkDeviceSize>(gpu.row_words) * 4 * (gpu.height + gpu.height / 2);
            gpu.active = true;
            for (size_t s = 0; s < paths.size(); ++s) targets.push_back(std::make_unique<GpuTarget>(*this, index, s));
            return targets;
        }
        catch (const std::exception & e)
        {
            _release_streams(gpu);
            reason = e.what();
            return {};
        }
    }

    bool submit_next(const std::vector<VideoOptions> & video_options) override
    {
        Slot & slot = _slots[_submitted % kSlots];
        if (slot.busy) throw std::logic_error("macos vulkan pipeline: submit_next() called while the slot is in flight");

        // decode the next frame of every video into the slot
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            VideoState & video = _videos[i];
            AVFrame * frame = slot.videos[i].frame;
            if (video.first_frame)
            {
                av_frame_move_ref(frame, video.first_frame);
                av_frame_free(&video.first_frame);
            }
            else if (!video.decoder->next(frame))
            {
                for (SlotVideo & v : slot.videos) av_frame_unref(v.frame);
                return false;
            }
            _prepare_source(frame, video.sw_frame, slot.videos[i]);
        }

        // check whether any persistent GPU buffer needs to grow
        bool reallocate = false;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & slot_video = slot.videos[i];
            const Plan & plan = slot_video.plan;
            reallocate |= _videos[i].luma[0].size < plan.luma_bytes || _videos[i].chroma[0].size < plan.chroma_bytes;
            if (_motion)
            {
                const MotionLayout layout = motion_layout(slot_video.width, slot_video.height);
                reallocate |= _videos[i].pyramid[0].size < layout.bytes || _videos[i].products.size < motion_product_bytes(layout);
            }
        }
        if (reallocate) check(_fn.vkDeviceWaitIdle(_device), "vkDeviceWaitIdle");

        const std::initializer_list<VkMemoryPropertyFlags> host_unified = {
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        const std::initializer_list<VkMemoryPropertyFlags> host_readback = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };

        std::array<PushConstants, kMaxVideoCount> parameters {};
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            VideoState & video = _videos[i];
            SlotVideo & slot_video = slot.videos[i];
            const Plan & plan = slot_video.plan;
            if (static_cast<uint32_t>(slot_video.height) > _max_workgroups)
            {
                throw std::runtime_error("frame height exceeds the device limit");
            }

            const bool compare = video.has_previous && video.width == slot_video.width && video.height == slot_video.height
                              && video.format == plan.format;

            if (video.luma[0].size < plan.luma_bytes)
            {
                for (Buffer & b : video.luma)
                {
                    _destroy_buffer(b);
                    b = _create_buffer(plan.luma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_unified);
                }
            }
            if (video.chroma[0].size < plan.chroma_bytes)
            {
                for (Buffer & b : video.chroma)
                {
                    _destroy_buffer(b);
                    b = _create_buffer(plan.chroma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_unified);
                }
            }

            const size_t cur_idx = video.frame_index % kPlaneBuffers;
            const size_t prev_idx = (video.frame_index + kPlaneBuffers - 1) % kPlaneBuffers;
            slot_video.plane_idx = cur_idx;
            _upload_planes(slot_video, video.luma[cur_idx], video.chroma[cur_idx]);
            // return decoded frame (and VideoToolbox CVPixelBuffer) to the decoder immediately
            av_frame_unref(slot_video.frame);
            av_frame_unref(video.sw_frame);

            const VkDeviceSize row_bytes = static_cast<VkDeviceSize>(slot_video.height) * sizeof(uint32_t);
            if (slot_video.row_counts.size < row_bytes)
            {
                _destroy_buffer(slot_video.row_counts);
                slot_video.row_counts = _create_buffer(row_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_readback);
            }

            GpuVideo & gpu = _gpu_videos[i];
            slot_video.picture = -1;
            if (gpu.active)
            {
                if (video_picture_size(slot_video.width, slot_video.height, _native_pictures) != cv::Size(static_cast<int>(gpu.width), static_cast<int>(gpu.height)))
                {
                    throw std::runtime_error("the size of the frames changed, the motion video can't be drawn on the GPU");
                }
                if (gpu.free_pictures.empty())
                {
                    gpu.pictures.push_back(_create_buffer(gpu.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 }));
                    gpu.picture_draws.push_back(0);
                    gpu.free_pictures.push_back(static_cast<int>(gpu.pictures.size()) - 1);
                }
                slot_video.picture = gpu.free_pictures.back();
                gpu.free_pictures.pop_back();
                const VkDescriptorBufferInfo base_buffers[3] = {
                    { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                    { video.chroma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                    { gpu.pictures[static_cast<size_t>(slot_video.picture)].buffer, 0, VK_WHOLE_SIZE },
                };
                std::array<VkWriteDescriptorSet, 3> base_writes {};
                for (uint32_t b = 0; b < 3; ++b)
                {
                    base_writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    base_writes[b].dstSet = slot_video.base_set;
                    base_writes[b].dstBinding = b;
                    base_writes[b].descriptorCount = 1;
                    base_writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    base_writes[b].pBufferInfo = &base_buffers[b];
                }
                _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(base_writes.size()), base_writes.data(), 0, nullptr);
            }

            video.width = slot_video.width;
            video.height = slot_video.height;
            video.format = plan.format;

            const std::array<VkDescriptorBufferInfo, kBindings> buffers = {{
                { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                { video.luma[prev_idx].buffer, 0, VK_WHOLE_SIZE },
                { video.chroma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                { video.chroma[prev_idx].buffer, 0, VK_WHOLE_SIZE },
                { slot_video.row_counts.buffer, 0, VK_WHOLE_SIZE },
            }};
            std::array<VkWriteDescriptorSet, kBindings> writes {};
            for (uint32_t b = 0; b < kBindings; ++b)
            {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = slot_video.descriptor_set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &buffers[b];
            }
            _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            parameters[i] = { static_cast<uint32_t>(slot_video.width), plan.luma_row_words,
                              static_cast<uint32_t>(std::max(0, video_options[i].pixel_difference)),
                              static_cast<uint32_t>(plan.layout.chroma), static_cast<uint32_t>(plan.layout.shift_x),
                              static_cast<uint32_t>(plan.layout.shift_y), plan.chroma_row_bytes, plan.v_offset };
            slot_video.compared = compare;

            if (_motion)
            {
                const MotionLayout layout = motion_layout(slot_video.width, slot_video.height);
                for (Buffer & pyramid : video.pyramid)
                {
                    if (pyramid.size >= layout.bytes) continue;
                    _destroy_buffer(pyramid);
                    pyramid = _create_buffer(layout.bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
                const VkDeviceSize product_bytes = motion_product_bytes(layout);
                if (video.products.size < product_bytes)
                {
                    _destroy_buffer(video.products);
                    video.products = _create_buffer(product_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
                }
                if (!video.sparse_latch.buffer)
                {
                    video.sparse_latch = _create_buffer(sizeof(SparseLatch), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    });
                    std::memset(video.sparse_latch.mapped, 0, sizeof(SparseLatch));
                    _flush(video.sparse_latch);
                }
                const std::pair<Buffer *, std::pair<VkDeviceSize, VkBufferUsageFlags>> read_back[3] = {
                    { &slot_video.motion_points, { kMotionPointBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT } },
                    { &slot_video.candidates, { kMotionCandidateBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT } },
                    { &slot_video.sparse_state, { sizeof(SparseState), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT } },
                };
                for (const auto & [buffer, wanted] : read_back)
                {
                    if (buffer->size >= wanted.first) continue;
                    _destroy_buffer(*buffer);
                    *buffer = _create_buffer(wanted.first, wanted.second, host_readback);
                }
                slot_video.motion = layout;
                slot_video.motion_estimated = compare;
                slot_video.sparse_possible = compare && sparse_motion_possible(layout);

                const VkDescriptorBufferInfo pyramid_buffers[2] = {
                    { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                    { video.pyramid[video.current].buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo lk_buffers[4] = {
                    { video.pyramid[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.pyramid[video.current].buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.motion_points.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.candidates.buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo select_buffers[3] = {
                    { slot_video.motion_points.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.sparse_state.buffer, 0, VK_WHOLE_SIZE },
                    { video.sparse_latch.buffer, 0, VK_WHOLE_SIZE },
                };
                const VkDescriptorBufferInfo texture_buffers[4] = {
                    { video.pyramid[1 - video.current].buffer, 0, VK_WHOLE_SIZE },
                    { video.products.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.sparse_state.buffer, 0, VK_WHOLE_SIZE },
                    { slot_video.candidates.buffer, 0, VK_WHOLE_SIZE },
                };
                std::vector<VkWriteDescriptorSet> motion_writes;
                auto write = [&](VkDescriptorSet set, const VkDescriptorBufferInfo * infos, uint32_t count) {
                    for (uint32_t b = 0; b < count; ++b)
                    {
                        auto entry = vk_init<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
                        entry.dstSet = set;
                        entry.dstBinding = b;
                        entry.descriptorCount = 1;
                        entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        entry.pBufferInfo = &infos[b];
                        motion_writes.push_back(entry);
                    }
                };
                write(slot_video.pyramid_set, pyramid_buffers, 2);
                write(slot_video.lk_set, lk_buffers, 4);
                write(slot_video.select_set, select_buffers, 3);
                write(slot_video.texture_set, texture_buffers, 4);
                _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(motion_writes.size()), motion_writes.data(), 0, nullptr);
            }

            slot_video.resolution = ResolutionLayout();
            if (_resolution)
            {
                if (static_cast<uint32_t>(slot_video.width) > _max_workgroups)
                {
                    throw std::runtime_error("frame width exceeds the device limit");
                }
                const ResolutionLayout layout = resolution_layout(slot_video.width, slot_video.height,
                                                                  video_options[i].resolution_grid_from, video_options[i].resolution_grid_to,
                                                                  video_options[i].resolution_pan_from, video_options[i].resolution_pan_to);
                slot_video.resolution = layout;
                if (layout.width > 0)
                {
                    const VkDeviceSize sums_bytes = static_cast<VkDeviceSize>(layout.width + layout.lattice_rows) * sizeof(uint32_t);
                    const VkDeviceSize power_bytes = static_cast<VkDeviceSize>(2 * layout.edge) * kResolutionBins * sizeof(float);
                    const VkDeviceSize stairs_bytes = static_cast<VkDeviceSize>(2 * layout.edge) * static_cast<VkDeviceSize>(std::max(1, layout.stair_bands)) * sizeof(uint32_t);
                    const std::pair<Buffer *, VkDeviceSize> read_back[3] = {
                        { &slot_video.lattice_sums, sums_bytes },
                        { &slot_video.edge_power, power_bytes },
                        { &slot_video.edge_stairs, stairs_bytes },
                    };
                    for (const auto & [buffer, bytes] : read_back)
                    {
                        if (buffer->size >= bytes) continue;
                        _destroy_buffer(*buffer);
                        *buffer = _create_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_readback);
                    }
                    const VkDescriptorBufferInfo resolution_buffers[6] = {
                        { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                        { slot_video.lattice_sums.buffer, 0, VK_WHOLE_SIZE },
                        { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                        { slot_video.edge_power.buffer, 0, VK_WHOLE_SIZE },
                        { video.luma[cur_idx].buffer, 0, VK_WHOLE_SIZE },
                        { slot_video.edge_stairs.buffer, 0, VK_WHOLE_SIZE },
                    };
                    std::array<VkWriteDescriptorSet, 6> resolution_writes {};
                    for (uint32_t b = 0; b < 6; ++b)
                    {
                        resolution_writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        resolution_writes[b].dstSet = b < 2 ? slot_video.lattice_set : b < 4 ? slot_video.edge_set : slot_video.stairs_set;
                        resolution_writes[b].dstBinding = b % 2;
                        resolution_writes[b].descriptorCount = 1;
                        resolution_writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        resolution_writes[b].pBufferInfo = &resolution_buffers[b];
                    }
                    _fn.vkUpdateDescriptorSets(_device, static_cast<uint32_t>(resolution_writes.size()), resolution_writes.data(), 0, nullptr);
                }
            }
        }

        // record compute dispatches
        check(_fn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        auto begin = vk_init<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_fn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        auto before = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &before, 0, nullptr, 0, nullptr);

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            if (!slot.videos[i].compared) continue;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline_layout,
                                        0, 1, &slot.videos[i].descriptor_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &parameters[i]);
            _fn.vkCmdDispatch(slot.command_buffer, static_cast<uint32_t>(slot.videos[i].height), 1, 1);
        }

        if (_resolution) _record_resolution(slot);

        bool base_bound = false;
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & slot_video = slot.videos[i];
            if (slot_video.picture < 0) continue;
            const GpuVideo & gpu = _gpu_videos[i];
            const Plan & plan = slot_video.plan;
            OverlayBasePush push {};
            push.width = gpu.width;
            push.height = gpu.height;
            push.row_words = gpu.row_words;
            push.scale = static_cast<uint32_t>(video_picture_scale(slot_video.width, slot_video.height, _native_pictures));
            push.luma_row_words = plan.luma_row_words;
            push.chroma_mode = static_cast<uint32_t>(plan.layout.chroma);
            push.chroma_row_bytes = plan.chroma_row_bytes;
            push.v_offset = plan.v_offset;
            push.chroma_step_x = std::max(1u, (2 * push.scale) >> plan.layout.shift_x);
            push.chroma_step_y = std::max(1u, (2 * push.scale) >> plan.layout.shift_y);
            if (!base_bound) _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.pipeline);
            base_bound = true;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_base.layout, 0, 1, &slot_video.base_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _overlay_base.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, (gpu.row_words + 7) / 8, (gpu.height / 2 + 7) / 8, 1);
        }

        if (_motion) _record_motion(slot);

        auto host = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &host, 0, nullptr, 0, nullptr);
        check(_fn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        check(_fn.vkResetFences(_device, 1, &slot.fence), "vkResetFences");
        auto submit_info = vk_init<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &slot.command_buffer;
        check(_fn.vkQueueSubmit(_queue, 1, &submit_info, slot.fence), "vkQueueSubmit");

        slot.busy = true;
        ++_submitted;
        ++_frames_read;
        for (VideoState & video : _videos)
        {
            video.has_previous = true;
            video.current = 1 - video.current;
            video.frame_index += 1;
        }
        return true;
    }

    size_t pending() const override { return _submitted - _collected; }
    size_t max_pending() const override { return kSlots; }

    void collect(std::vector<RowCounts> & out) override
    {
        Slot & slot = _slots[_collected % kSlots];
        if (!slot.busy) throw std::logic_error("macos vulkan pipeline: nothing to collect");
        check(_fn.vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        out.resize(_videos.size());
        for (size_t i = 0; i < _videos.size(); ++i)
        {
            const SlotVideo & slot_video = slot.videos[i];
            RowCounts & result = out[i];
            result.compared = slot_video.compared;
            result.cols = slot_video.width;
            result.coding = slot_video.coding;
            result.counts.clear();
            if (slot_video.compared)
            {
                _invalidate(slot_video.row_counts);
                const uint32_t * counts = static_cast<const uint32_t *>(slot_video.row_counts.mapped);
                result.counts.assign(counts, counts + slot_video.height);
            }
            result.detail.reset();
            if (_motion_detail)
            {
                result.detail = std::make_shared<MotionDetail>();
                if (_gpu_videos[i].active) result.detail->gpu_picture = slot_video.picture;
                else _read_picture(_videos[i], slot_video, result.detail->picture);
            }
            if (_motion) _read_motion(slot_video, result.motion, _motion_detail ? &result.detail->points : nullptr);
            result.resolution.reset();
            if (_resolution && slot_video.resolution.width > 0) result.resolution = _read_resolution(slot_video);
        }
        slot.busy = false;
        ++_collected;
    }

    double analysis_seconds() const override { return -1.0; }

    size_t count() const override { return _videos.size(); }
    double recorded_framerate(size_t index) const override { return _videos.at(index).decoder->framerate(); }
    long long reported_frame_count(size_t index) const override { return _videos.at(index).decoder->frame_count(); }
    int frame_width(size_t index) const override { return _videos.at(index).decoder->width(); }
    int frame_height(size_t index) const override { return _videos.at(index).decoder->height(); }
    long long frames_read() const override { return _frames_read; }
    std::string codec_name(size_t index) const override { return _videos.at(index).decoder->codec_name(); }
    bool lossy_codec(size_t index) const override { return _videos.at(index).decoder->lossy(); }

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void * mapped = nullptr;
        bool coherent = true;
    };

    struct Plan {
        AVPixelFormat format = AV_PIX_FMT_NONE;
        ffmpeg::YuvLayout layout;
        uint32_t luma_row_words = 0;
        VkDeviceSize luma_bytes = 0;
        uint32_t chroma_width = 0;
        uint32_t chroma_height = 0;
        uint32_t chroma_row_bytes = 0;
        uint32_t v_offset = 0;
        VkDeviceSize chroma_bytes = 0;
    };

    struct VideoState {
        std::unique_ptr<ffmpeg::VideoDecoder> decoder;
        bool videotoolbox = false;
        AVFrame * first_frame = nullptr;
        AVFrame * sw_frame = nullptr;
        ffmpeg::VideoColorInfo color;
        bool bt709 = false;
        std::array<Buffer, kPlaneBuffers> luma;
        std::array<Buffer, kPlaneBuffers> chroma;
        Buffer pyramid[2];
        Buffer products;
        Buffer sparse_latch;
        AVPixelFormat format = AV_PIX_FMT_NONE;
        int current = 0;
        size_t frame_index = 0;
        int width = 0;
        int height = 0;
        bool has_previous = false;
    };

    struct SlotVideo {
        AVFrame * frame = nullptr;
        CVPixelBufferRef pixbuf = nullptr;
        const AVFrame * cpu_frame = nullptr;
        Plan plan;
        size_t plane_idx = 0;
        FrameCoding coding = FrameCoding::Unknown;
        Buffer row_counts;
        int picture = -1;
        VkDescriptorSet base_set = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        Buffer motion_points;
        Buffer candidates;
        Buffer sparse_state;
        VkDescriptorSet pyramid_set = VK_NULL_HANDLE;
        VkDescriptorSet lk_set = VK_NULL_HANDLE;
        VkDescriptorSet select_set = VK_NULL_HANDLE;
        VkDescriptorSet texture_set = VK_NULL_HANDLE;
        MotionLayout motion;
        Buffer lattice_sums;
        Buffer edge_power;
        Buffer edge_stairs;
        VkDescriptorSet lattice_set = VK_NULL_HANDLE;
        VkDescriptorSet edge_set = VK_NULL_HANDLE;
        VkDescriptorSet stairs_set = VK_NULL_HANDLE;
        ResolutionLayout resolution;
        bool motion_estimated = false;
        bool sparse_possible = false;
        bool compared = false;
        int width = 0;
        int height = 0;
    };

    struct DrawSlot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSet blur_set_a = VK_NULL_HANDLE;
        VkDescriptorSet blur_set_b = VK_NULL_HANDLE;
        Buffer segments;
        Buffer tiles;
        Buffer text;
        Buffer card_data;
        Buffer blur_a;
        Buffer blur_b;
        uint64_t overlay_key = UINT64_MAX;
        uint64_t overlay_generation = 0;
        size_t overlay_bytes = 0;
        Buffer output;
        AVFrame * av_frame = nullptr;
    };

    struct GpuStream {
        std::unique_ptr<ffmpeg::EncodedVideo> encoder;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        std::array<DrawSlot, kDrawSlots> draw_slots;
        size_t draw_submitted = 0;
        size_t draw_collected = 0;
    };

    struct GpuVideo {
        bool active = false;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t row_words = 0;
        VkDeviceSize picture_bytes = 0;
        std::vector<GpuStream> streams;
        std::vector<Buffer> pictures;
        std::vector<int> free_pictures;
        std::vector<uint32_t> picture_draws;
    };

    struct Slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool busy = false;
        std::vector<SlotVideo> videos;
    };

    struct Compute {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    class GpuTarget final : public MotionVideoTarget {
    public:
        GpuTarget(VulkanMacosPipeline & pipeline, size_t index, size_t stream)
            : _pipeline(pipeline), _index(index), _stream(stream) { }
        std::string description() const override { return _pipeline._gpu_videos[_index].streams[_stream].encoder->encoder_name() + ", drawn on the GPU"; }
        void write(const MotionDetail & detail, const MotionVideoScene & scene) override { _pipeline._draw(_index, _stream, detail.gpu_picture, scene); }
        void finish() override { _pipeline._finish_video(_index, _stream); }

    private:
        VulkanMacosPipeline & _pipeline;
        size_t _index;
        size_t _stream;
    };

    void _create_instance_and_device(int device_index)
    {
        vk_loader::LoadedVulkan global;
        vk_loader::create_instance(VK_API_VERSION_1_1, global, _instance);
#define FRESNIFF_VK_LOAD_MAC_INSTANCE(name) \
        _ifn.name = reinterpret_cast<PFN_##name>(global.get_instance_proc_addr(_instance, #name)); \
        if (!_ifn.name) throw std::runtime_error("missing Vulkan function " #name);
        FRESNIFF_VK_MAC_INSTANCE_FUNCTIONS(FRESNIFF_VK_LOAD_MAC_INSTANCE)
#undef FRESNIFF_VK_LOAD_MAC_INSTANCE

        uint32_t device_count = 0;
        check(_ifn.vkEnumeratePhysicalDevices(_instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(device_count);
        if (device_count > 0) check(_ifn.vkEnumeratePhysicalDevices(_instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");
        devices.resize(device_count);
        if (devices.empty()) throw std::runtime_error("no Vulkan device found");

        if (device_index >= 0)
        {
            if (static_cast<size_t>(device_index) >= devices.size())
            {
                throw std::runtime_error("Vulkan device " + std::to_string(device_index) + " does not exist ("
                                       + std::to_string(devices.size()) + " available, see --list-vkdevices)");
            }
            _device_index = device_index;
        }
        else
        {
            _device_index = 0;
            for (size_t i = 0; i < devices.size(); ++i)
            {
                VkPhysicalDeviceProperties props;
                _ifn.vkGetPhysicalDeviceProperties(devices[i], &props);
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                {
                    _device_index = static_cast<int>(i);
                    break;
                }
            }
        }
        _physical_device = devices[static_cast<size_t>(_device_index)];
        VkPhysicalDeviceProperties properties;
        _ifn.vkGetPhysicalDeviceProperties(_physical_device, &properties);
        _device_name = properties.deviceName;
        _max_workgroups = properties.limits.maxComputeWorkGroupCount[0];
        _max_group_size = properties.limits.maxComputeWorkGroupSize[0];
        _max_group_invocations = properties.limits.maxComputeWorkGroupInvocations;
        _ifn.vkGetPhysicalDeviceMemoryProperties(_physical_device, &_memory_properties);

        uint32_t family_count = 0;
        _ifn.vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        _ifn.vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &family_count, families.data());
        bool found_queue = false;
        for (uint32_t i = 0; i < family_count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                _queue_family = i;
                found_queue = true;
                break;
            }
        }
        if (!found_queue) throw std::runtime_error("Vulkan device '" + _device_name + "' has no compute queue");

        const float priority = 1.0f;
        auto queue_info = vk_init<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
        queue_info.queueFamilyIndex = _queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        const std::vector<const char *> extensions = vk_loader::device_portability_extensions(
            _ifn.vkEnumerateDeviceExtensionProperties, _physical_device);

        auto device_info = vk_init<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames = extensions.data();
        check(_ifn.vkCreateDevice(_physical_device, &device_info, nullptr, &_device), "vkCreateDevice");

#define FRESNIFF_VK_LOAD_MAC_DEVICE(name) \
        _fn.name = reinterpret_cast<PFN_##name>(_ifn.vkGetDeviceProcAddr(_device, #name)); \
        if (!_fn.name) throw std::runtime_error("missing Vulkan function " #name);
        FRESNIFF_VK_MAC_DEVICE_FUNCTIONS(FRESNIFF_VK_LOAD_MAC_DEVICE)
#undef FRESNIFF_VK_LOAD_MAC_DEVICE

        _fn.vkGetDeviceQueue(_device, _queue_family, 0, &_queue);
    }

    void _open_video(VideoState & video, const std::string & path)
    {
        video.first_frame = av_frame_alloc();
        video.sw_frame = av_frame_alloc();
        if (!video.first_frame || !video.sw_frame) throw std::runtime_error("out of memory");

        const ffmpeg::Probe probe = ffmpeg::probe(path);
        if (!probe.opened) throw std::runtime_error(probe.error);

        if (_vt_device && probe.videotoolbox)
        {
            try
            {
                video.decoder = std::make_unique<ffmpeg::VideoDecoder>(path, _vt_device, static_cast<int>(kSlots) + 2);
                if (!video.decoder->next(video.first_frame)) throw NoFramesError("'" + path + "' contains no frames");
                SlotVideo check_slot;
                _prepare_source(video.first_frame, video.sw_frame, check_slot);
                av_frame_unref(video.sw_frame);
                video.videotoolbox = true;
            }
            catch (const NonLumaFormatError &)
            {
                throw;
            }
            catch (const NoFramesError &)
            {
                throw;
            }
            catch (const std::exception &)
            {
                av_frame_unref(video.first_frame);
                av_frame_unref(video.sw_frame);
                video.decoder.reset();
            }
        }

        if (!video.decoder)
        {
            video.decoder = std::make_unique<ffmpeg::VideoDecoder>(path, nullptr);
            if (!video.decoder->next(video.first_frame)) throw std::runtime_error("'" + path + "' contains no frames");
            SlotVideo check_slot;
            _prepare_source(video.first_frame, video.sw_frame, check_slot);
            av_frame_unref(video.sw_frame);
            video.videotoolbox = false;
        }

        video.color.colorspace = video.first_frame->colorspace;
        video.color.primaries = video.first_frame->color_primaries;
        video.color.trc = video.first_frame->color_trc;
        video.color.range = video.first_frame->color_range;
        video.bt709 = video.color.colorspace == AVCOL_SPC_BT709
                   || (video.color.colorspace == AVCOL_SPC_UNSPECIFIED && video.first_frame->height >= 720);
    }

    static void _check_cpu_frame(const AVFrame * frame)
    {
        const AVPixelFormat format = static_cast<AVPixelFormat>(frame->format);
        if (!ffmpeg::is_luma_format(format))
        {
            throw NonLumaFormatError("pixel format " + ffmpeg::format_name(format) + " is no 8-bit YUV format");
        }
        for (int plane = 0; plane < 3; ++plane)
        {
            if (frame->data[plane] && frame->linesize[plane] <= 0) throw std::runtime_error("unsupported frame layout");
        }
    }

    static void _prepare_source(const AVFrame * frame, AVFrame * sw_frame, SlotVideo & out)
    {
        out.pixbuf = nullptr;
        out.cpu_frame = nullptr;
        out.coding = ffmpeg::frame_coding(frame);
        out.width = frame->width;
        out.height = frame->height;
        if (out.width <= 0 || out.height <= 0) throw std::runtime_error("invalid frame dimensions");

        Plan plan;
        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX)
        {
            CVPixelBufferRef pb = reinterpret_cast<CVPixelBufferRef>(frame->data[3]);
            if (!pb) throw std::runtime_error("VideoToolbox frame has no pixel buffer");
            const OSType fmt = CVPixelBufferGetPixelFormatType(pb);
            if (is_vt_nv12(fmt) && CVPixelBufferGetPlaneCount(pb) == 2)
            {
                out.pixbuf = pb;
                plan.format = AV_PIX_FMT_NV12;
                plan.layout = { ffmpeg::YuvLayout::Interleaved, 1, 1 };
            }
            else
            {
                av_frame_unref(sw_frame);
                const int transferred = av_hwframe_transfer_data(sw_frame, frame, 0);
                if (transferred < 0)
                {
                    throw std::runtime_error("VideoToolbox frame transfer failed (" + ffmpeg::error_string(transferred) + ")");
                }
                _check_cpu_frame(sw_frame);
                out.cpu_frame = sw_frame;
                plan.format = static_cast<AVPixelFormat>(sw_frame->format);
                plan.layout = ffmpeg::yuv_layout(plan.format);
            }
        }
        else
        {
            _check_cpu_frame(frame);
            out.cpu_frame = frame;
            plan.format = static_cast<AVPixelFormat>(frame->format);
            plan.layout = ffmpeg::yuv_layout(plan.format);
        }

        plan.luma_row_words = static_cast<uint32_t>((out.width + 3) / 4);
        plan.luma_bytes = static_cast<VkDeviceSize>(plan.luma_row_words) * 4 * static_cast<VkDeviceSize>(out.height);
        plan.chroma_width = static_cast<uint32_t>(AV_CEIL_RSHIFT(out.width, plan.layout.shift_x));
        plan.chroma_height = static_cast<uint32_t>(AV_CEIL_RSHIFT(out.height, plan.layout.shift_y));
        uint32_t bytes = 4; // placeholder buffer without chroma
        if (plan.layout.chroma == ffmpeg::YuvLayout::Interleaved)
        {
            plan.chroma_row_bytes = 2 * plan.chroma_width;
            bytes = plan.chroma_row_bytes * plan.chroma_height;
        }
        else if (plan.layout.chroma == ffmpeg::YuvLayout::Planar)
        {
            plan.chroma_row_bytes = plan.chroma_width;
            plan.v_offset = align4(plan.chroma_width * plan.chroma_height);
            bytes = plan.v_offset + plan.chroma_width * plan.chroma_height;
        }
        plan.chroma_bytes = align4(bytes);
        out.plan = plan;
    }

    void _upload_planes(const SlotVideo & slot_video, const Buffer & luma_buf, const Buffer & chroma_buf)
    {
        const Plan & plan = slot_video.plan;
        const size_t w = static_cast<size_t>(slot_video.width);
        const size_t h = static_cast<size_t>(slot_video.height);
        const size_t luma_stride = static_cast<size_t>(plan.luma_row_words) * 4;
        uint8_t * dst_luma = static_cast<uint8_t *>(luma_buf.mapped);
        uint8_t * dst_chroma = static_cast<uint8_t *>(chroma_buf.mapped);

        if (slot_video.pixbuf)
        {
            PixelBufferLock lock(slot_video.pixbuf);
            const uint8_t * src_y = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(slot_video.pixbuf, 0));
            if (!src_y) throw std::runtime_error("VideoToolbox pixel buffer luma plane is null");
            const size_t src_y_bpr = CVPixelBufferGetBytesPerRowOfPlane(slot_video.pixbuf, 0);
            const size_t y_pw = CVPixelBufferGetWidthOfPlane(slot_video.pixbuf, 0);
            const size_t y_ph = CVPixelBufferGetHeightOfPlane(slot_video.pixbuf, 0);
            const size_t crop_x = (slot_video.frame->crop_left + w <= y_pw) ? slot_video.frame->crop_left : 0;
            const size_t crop_y = (slot_video.frame->crop_top + h <= y_ph) ? slot_video.frame->crop_top : 0;
            const size_t copy_w = std::min(w, y_pw - crop_x);
            const size_t copy_h = std::min(h, y_ph - crop_y);
            for (size_t row = 0; row < h; ++row)
            {
                uint8_t * dst_row = dst_luma + row * luma_stride;
                if (row < copy_h)
                {
                    std::memcpy(dst_row, src_y + (crop_y + row) * src_y_bpr + crop_x, copy_w);
                    if (luma_stride > copy_w) std::memset(dst_row + copy_w, 0, luma_stride - copy_w);
                }
                else
                {
                    std::memset(dst_row, 0, luma_stride);
                }
            }

            const uint8_t * src_uv = static_cast<const uint8_t *>(CVPixelBufferGetBaseAddressOfPlane(slot_video.pixbuf, 1));
            if (!src_uv) throw std::runtime_error("VideoToolbox pixel buffer chroma plane is null");
            const size_t src_uv_bpr = CVPixelBufferGetBytesPerRowOfPlane(slot_video.pixbuf, 1);
            const size_t uv_pw = CVPixelBufferGetWidthOfPlane(slot_video.pixbuf, 1);
            const size_t uv_ph = CVPixelBufferGetHeightOfPlane(slot_video.pixbuf, 1);
            const size_t uv_cx = ((crop_x >> 1) + plan.chroma_width <= uv_pw) ? (crop_x >> 1) : 0;
            const size_t uv_cy = ((crop_y >> 1) + plan.chroma_height <= uv_ph) ? (crop_y >> 1) : 0;
            const size_t copy_cw = std::min<size_t>(plan.chroma_width, uv_pw - uv_cx);
            const size_t copy_ch = std::min<size_t>(plan.chroma_height, uv_ph - uv_cy);
            for (size_t row = 0; row < plan.chroma_height; ++row)
            {
                uint8_t * dst_row = dst_chroma + row * plan.chroma_row_bytes;
                if (row < copy_ch)
                {
                    std::memcpy(dst_row, src_uv + (uv_cy + row) * src_uv_bpr + 2 * uv_cx, 2 * copy_cw);
                    if (plan.chroma_row_bytes > 2 * copy_cw) std::memset(dst_row + 2 * copy_cw, 128, plan.chroma_row_bytes - 2 * copy_cw);
                }
                else
                {
                    std::memset(dst_row, 128, plan.chroma_row_bytes);
                }
            }
        }
        else if (slot_video.cpu_frame)
        {
            const AVFrame * f = slot_video.cpu_frame;
            const size_t tail = luma_stride - w;
            for (size_t row = 0; row < h; ++row)
            {
                uint8_t * dst_row = dst_luma + row * luma_stride;
                std::memcpy(dst_row, f->data[0] + row * static_cast<size_t>(f->linesize[0]), w);
                if (tail > 0) std::memset(dst_row + w, 0, tail);
            }
            if (plan.layout.chroma == ffmpeg::YuvLayout::Interleaved)
            {
                for (size_t row = 0; row < plan.chroma_height; ++row)
                {
                    std::memcpy(dst_chroma + row * plan.chroma_row_bytes,
                                f->data[1] + row * static_cast<size_t>(f->linesize[1]),
                                plan.chroma_row_bytes);
                }
            }
            else if (plan.layout.chroma == ffmpeg::YuvLayout::Planar)
            {
                for (size_t row = 0; row < plan.chroma_height; ++row)
                {
                    std::memcpy(dst_chroma + row * plan.chroma_row_bytes,
                                f->data[1] + row * static_cast<size_t>(f->linesize[1]),
                                plan.chroma_row_bytes);
                    std::memcpy(dst_chroma + plan.v_offset + row * plan.chroma_row_bytes,
                                f->data[2] + row * static_cast<size_t>(f->linesize[2]),
                                plan.chroma_row_bytes);
                }
            }
        }
        _flush(luma_buf);
        _flush(chroma_buf);
    }

    void _record_resolution(Slot & slot)
    {
        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lattice.pipeline);
        for (const SlotVideo & slot_video : slot.videos)
        {
            const ResolutionLayout & layout = slot_video.resolution;
            if (layout.width == 0) continue;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lattice.layout, 0, 1, &slot_video.lattice_set, 0, nullptr);
            LatticePush push {};
            push.width = static_cast<uint32_t>(layout.width);
            push.luma_row_words = slot_video.plan.luma_row_words;
            push.rows = static_cast<uint32_t>(layout.lattice_rows);
            push.row0 = static_cast<uint32_t>(layout.lattice_row0);
            _fn.vkCmdPushConstants(slot.command_buffer, _lattice.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, push.width, 1, 1);
            push.mode = 1;
            push.offset = push.width;
            _fn.vkCmdPushConstants(slot.command_buffer, _lattice.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, push.rows, 1, 1);
        }
        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _edge.pipeline);
        for (const SlotVideo & slot_video : slot.videos)
        {
            const ResolutionLayout & layout = slot_video.resolution;
            if (layout.width == 0 || layout.segments < 1) continue;
            EdgePush push {};
            push.width = static_cast<uint32_t>(layout.width);
            push.luma_row_words = slot_video.plan.luma_row_words;
            push.row0 = static_cast<uint32_t>(layout.row0);
            push.segments = static_cast<uint32_t>(layout.segments);
            push.edge = static_cast<uint32_t>(layout.edge);
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _edge.layout, 0, 1, &slot_video.edge_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _edge.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, 2 * push.edge, 1, 1);
        }
        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _stairs.pipeline);
        for (const SlotVideo & slot_video : slot.videos)
        {
            const ResolutionLayout & layout = slot_video.resolution;
            if (layout.width == 0 || layout.segments < 1 || layout.stair_bands < 1) continue;
            StairsPush push {};
            push.width = static_cast<uint32_t>(layout.width);
            push.luma_row_words = slot_video.plan.luma_row_words;
            push.row0 = static_cast<uint32_t>(layout.row0);
            push.bands = static_cast<uint32_t>(layout.stair_bands);
            push.edge = static_cast<uint32_t>(layout.edge);
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _stairs.layout, 0, 1, &slot_video.stairs_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _stairs.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, 2 * push.edge, 1, 1);
        }
    }

    std::shared_ptr<ResolutionFrame> _read_resolution(const SlotVideo & slot_video)
    {
        const ResolutionLayout & layout = slot_video.resolution;
        std::shared_ptr<ResolutionFrame> frame = std::make_shared<ResolutionFrame>();
        frame->layout = layout;
        _invalidate(slot_video.lattice_sums);
        _invalidate(slot_video.edge_power);
        _invalidate(slot_video.edge_stairs);
        const uint32_t * sums = static_cast<const uint32_t *>(slot_video.lattice_sums.mapped);
        frame->column_sums.assign(sums, sums + layout.width);
        frame->row_sums.assign(sums + layout.width, sums + layout.width + layout.lattice_rows);
        if (layout.segments >= 1)
        {
            const float * power = static_cast<const float *>(slot_video.edge_power.mapped);
            frame->edge_power.assign(power, power + static_cast<size_t>(2 * layout.edge) * kResolutionBins);
            const uint32_t * stairs = static_cast<const uint32_t *>(slot_video.edge_stairs.mapped);
            const size_t stair_count = static_cast<size_t>(2 * layout.edge) * static_cast<size_t>(layout.stair_bands);
            frame->edge_stairs.resize(stair_count);
            for (size_t k = 0; k < stair_count; ++k) frame->edge_stairs[k] = static_cast<uint16_t>(stairs[k]);
            frame->edges = true;
        }
        return frame;
    }

    void _record_motion(Slot & slot)
    {
        auto written = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        written.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        written.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.pipeline);
        for (uint32_t level = 0; level < kMaxMotionLevels; ++level)
        {
            bool dispatched = false;
            for (const SlotVideo & slot_video : slot.videos)
            {
                const MotionLayout & layout = slot_video.motion;
                if (level >= layout.levels) continue;
                PyramidPush push {};
                push.dst_width = layout.width[level];
                push.dst_height = layout.height[level];
                push.dst_offset = layout.offset[level];
                if (level == 0)
                {
                    push.mode = 0;
                    push.factor = layout.scale;
                    push.src_width = static_cast<uint32_t>(slot_video.width);
                    push.src_height = static_cast<uint32_t>(slot_video.height);
                    push.src_base = slot_video.plan.luma_row_words;
                }
                else
                {
                    push.mode = 1;
                    push.factor = 2;
                    push.src_width = layout.width[level - 1];
                    push.src_height = layout.height[level - 1];
                    push.src_base = layout.offset[level - 1];
                }
                _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pyramid.layout, 0, 1, &slot_video.pyramid_set, 0, nullptr);
                _fn.vkCmdPushConstants(slot.command_buffer, _pyramid.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                _fn.vkCmdDispatch(slot.command_buffer, (push.dst_width + 15) / 16, (push.dst_height + 15) / 16, 1);
                dispatched = true;
            }
            if (dispatched)
            {
                _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &written, 0, nullptr, 0, nullptr);
            }
        }

        auto lk_push = [](const MotionLayout & layout) {
            LkPush push {};
            for (uint32_t level = 0; level < layout.levels; ++level)
            {
                push.level_width[level] = layout.width[level];
                push.level_height[level] = layout.height[level];
                push.level_offset[level] = layout.offset[level];
            }
            push.levels = layout.levels;
            push.grid_x = static_cast<uint32_t>(layout.grid.points_x());
            push.grid_y = static_cast<uint32_t>(layout.grid.points_y());
            push.iterations = kMotionIterations;
            push.max_residual = kMaxResidual;
            return push;
        };
        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.pipeline);
        bool sparse_possible = false;
        for (const SlotVideo & slot_video : slot.videos)
        {
            if (!slot_video.motion_estimated) continue;
            const LkPush push = lk_push(slot_video.motion);
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _lk.layout, 0, 1, &slot_video.lk_set, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _lk.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            _fn.vkCmdDispatch(slot.command_buffer, kMotionPoints, 1, 1);
            sparse_possible |= slot_video.sparse_possible;
        }
        if (!sparse_possible) return;

        auto decided = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        decided.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        decided.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

        for (int stage = 0; stage < 4; ++stage)
        {
            const Compute & compute = stage == 0 ? _select : stage == 3 ? _lk : _texture;
            if (stage == 1)
            {
                _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                         0, 1, &decided, 0, nullptr, 0, nullptr);
            }
            else
            {
                _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &written, 0, nullptr, 0, nullptr);
            }
            _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
            for (const SlotVideo & slot_video : slot.videos)
            {
                if (!slot_video.sparse_possible) continue;
                const SparseMotionPushes pushes = sparse_motion_pushes(slot_video.motion, lk_push(slot_video.motion));
                const VkDescriptorSet set = stage == 0 ? slot_video.select_set : stage == 3 ? slot_video.lk_set : slot_video.texture_set;
                _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.layout, 0, 1, &set, 0, nullptr);
                if (stage == 0)
                {
                    _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushes.select), &pushes.select);
                    _fn.vkCmdDispatch(slot.command_buffer, 1, 1, 1);
                    continue;
                }
                if (stage == 3) _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushes.points), &pushes.points);
                else _fn.vkCmdPushConstants(slot.command_buffer, compute.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TexturePush), stage == 1 ? &pushes.products : &pushes.parts);
                _fn.vkCmdDispatchIndirect(slot.command_buffer, slot_video.sparse_state.buffer, static_cast<VkDeviceSize>(stage - 1) * 3 * sizeof(uint32_t));
            }
        }
    }

    void _read_picture(const VideoState & video, const SlotVideo & slot_video, cv::Mat & picture)
    {
        const Plan & plan = slot_video.plan;
        ffmpeg::YuvPlanes planes;
        planes.width = slot_video.width;
        planes.height = slot_video.height;
        planes.layout = plan.layout;
        planes.v_first = plan.format == AV_PIX_FMT_NV21 || plan.format == AV_PIX_FMT_NV42;
        planes.data[0] = static_cast<const uint8_t *>(video.luma[slot_video.plane_idx].mapped);
        planes.stride[0] = static_cast<size_t>(plan.luma_row_words) * 4;
        planes.data[1] = static_cast<const uint8_t *>(video.chroma[slot_video.plane_idx].mapped);
        planes.stride[1] = plan.chroma_row_bytes;
        planes.data[2] = planes.data[1] + plan.v_offset;
        planes.stride[2] = plan.chroma_row_bytes;
        ffmpeg::motion_picture(planes, picture, _scratch, _native_pictures);
    }

    void _read_motion(const SlotVideo & slot_video, MotionField & field, std::vector<MotionPoint> * detail)
    {
        field = MotionField();
        field.pixel_scale = static_cast<float>(slot_video.motion.scale);
        if (detail) detail->clear();
        if (!slot_video.motion_estimated) return;
        field.estimated = true;
        _invalidate(slot_video.motion_points);

        const MotionGrid & grid = slot_video.motion.grid;
        const float points_x = static_cast<float>(grid.points_x());
        const float points_y = static_cast<float>(grid.points_y());
        _binner.clear(grid);
        const float * points = static_cast<const float *>(slot_video.motion_points.mapped);
        for (int y = 0; y < grid.points_y(); ++y)
        {
            for (int x = 0; x < grid.points_x(); ++x)
            {
                const float * point = points + 4 * (y * grid.points_x() + x);
                const bool texture = point[2] >= kMotionMinTexture && std::isfinite(point[0]) && std::isfinite(point[1]);
                const bool returns = texture && point[3] <= kMotionMaxBackError;
                if (detail)
                {
                    MotionPoint tracked;
                    tracked.x = (static_cast<float>(x) + 0.5f) * static_cast<float>(slot_video.motion.width[0]) / points_x - 0.5f;
                    tracked.y = (static_cast<float>(y) + 0.5f) * static_cast<float>(slot_video.motion.height[0]) / points_y - 0.5f;
                    tracked.state = returns ? MotionPoint::Used : texture ? MotionPoint::Rejected : MotionPoint::NoTexture;
                    if (texture)
                    {
                        tracked.dx = point[0];
                        tracked.dy = point[1];
                    }
                    detail->push_back(tracked);
                }
                if (!returns) continue;
                _binner.add((static_cast<float>(x) + 0.5f) / points_x, (static_cast<float>(y) + 0.5f) / points_y, point[0], point[1]);
            }
        }
        if (slot_video.sparse_possible)
        {
            _invalidate(slot_video.sparse_state);
            _invalidate(slot_video.candidates);
            const float width = static_cast<float>(slot_video.motion.width[0]);
            const float height = static_cast<float>(slot_video.motion.height[0]);
            read_sparse_motion(*static_cast<const SparseState *>(slot_video.sparse_state.mapped),
                               static_cast<const float *>(slot_video.candidates.mapped), points,
                               [&](float x, float y, float dx, float dy, bool texture, bool returns) {
                if (detail)
                {
                    MotionPoint tracked;
                    tracked.x = x;
                    tracked.y = y;
                    tracked.state = returns ? MotionPoint::Used : texture ? MotionPoint::Rejected : MotionPoint::NoTexture;
                    if (texture)
                    {
                        tracked.dx = dx;
                        tracked.dy = dy;
                    }
                    detail->push_back(tracked);
                }
                if (returns) _binner.add((x + 0.5f) / width, (y + 0.5f) / height, dx, dy);
            });
        }
        _binner.finish(field, kMotionMinCellPoints, field.pixel_scale,
                       slot_video.sparse_possible && static_cast<const SparseState *>(slot_video.sparse_state.mapped)->sparse != 0);
    }

    void _collect_draw(const GpuVideo & gpu, GpuStream & stream)
    {
        DrawSlot & slot = stream.draw_slots[stream.draw_collected % kDrawSlots];
        if (!slot.busy) throw std::logic_error("macos vulkan pipeline: draw slot is not busy");
        check(_fn.vkWaitForFences(_device, 1, &slot.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        _invalidate(slot.output);

        if (av_frame_make_writable(slot.av_frame) < 0) throw std::runtime_error("out of memory");
        const uint8_t * src = static_cast<const uint8_t *>(slot.output.mapped);
        const size_t src_stride = static_cast<size_t>(gpu.row_words) * 4;
        const size_t w = gpu.width;
        const size_t h = gpu.height;
        const uint8_t * src_uv = src + src_stride * h;

        for (size_t row = 0; row < h; ++row)
        {
            std::memcpy(slot.av_frame->data[0] + row * static_cast<size_t>(slot.av_frame->linesize[0]),
                        src + row * src_stride, w);
        }
        const size_t cw = w / 2;
        const size_t ch = h / 2;
        if (slot.av_frame->format == AV_PIX_FMT_NV12)
        {
            for (size_t row = 0; row < ch; ++row)
            {
                std::memcpy(slot.av_frame->data[1] + row * static_cast<size_t>(slot.av_frame->linesize[1]),
                            src_uv + row * src_stride, w);
            }
        }
        else
        {
            for (size_t row = 0; row < ch; ++row)
            {
                const uint8_t * pairs = src_uv + row * src_stride;
                uint8_t * dst_u = slot.av_frame->data[1] + row * static_cast<size_t>(slot.av_frame->linesize[1]);
                uint8_t * dst_v = slot.av_frame->data[2] + row * static_cast<size_t>(slot.av_frame->linesize[2]);
                for (size_t x = 0; x < cw; ++x)
                {
                    dst_u[x] = pairs[2 * x];
                    dst_v[x] = pairs[2 * x + 1];
                }
            }
        }
        stream.encoder->encode(slot.av_frame);
        slot.busy = false;
        ++stream.draw_collected;
    }

    void _draw(size_t index, size_t stream_index, int picture, const MotionVideoScene & scene)
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        GpuStream & stream = gpu.streams.at(stream_index);
        if (picture < 0 || picture >= static_cast<int>(gpu.pictures.size()))
        {
            throw std::logic_error("motion video: the frame has no picture on the GPU");
        }
        if (stream.draw_submitted - stream.draw_collected >= kDrawSlots)
        {
            _collect_draw(gpu, stream);
        }
        DrawSlot & slot = stream.draw_slots[stream.draw_submitted % kDrawSlots];

        const VideoState & video = _videos.at(index);
        build_overlay_upload(scene, gpu.width, gpu.height, gpu.row_words, video.bt709, _overlay_upload);
        const OverlayDrawPush & push = _overlay_upload.push;
        _upload(slot.segments, _overlay_upload.segments.data(), _overlay_upload.segments.size() * sizeof(float));
        _upload(slot.tiles, _overlay_upload.tiles.data(), _overlay_upload.tiles.size() * sizeof(uint32_t));
        const uint64_t overlay_key = (static_cast<uint64_t>(index) << 32) | static_cast<uint64_t>(stream_index);
        if (_overlay_upload.rgba_overlay)
        {
            const size_t bytes = _overlay_upload.overlay_bytes;
            const bool unchanged = slot.overlay_key == overlay_key
                                && slot.overlay_generation == _overlay_upload.overlay_generation
                                && slot.overlay_bytes == bytes
                                && _overlay_upload.overlay_generation != 0
                                && slot.text.size >= bytes;
            if (!unchanged)
            {
                _ensure_buffer(slot.text, std::max<VkDeviceSize>(bytes, 32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                });
                if (bytes > 0) copy_rgba_overlay(scene.overlay, slot.text.mapped);
                _flush(slot.text);
                slot.overlay_key = overlay_key;
                slot.overlay_generation = _overlay_upload.overlay_generation;
                slot.overlay_bytes = bytes;
            }
        }
        else
        {
            _upload(slot.text, _overlay_upload.text.data(), _overlay_upload.text.size());
            slot.overlay_key = UINT64_MAX;
            slot.overlay_generation = 0;
            slot.overlay_bytes = 0;
        }
        _upload(slot.card_data, _overlay_upload.card_data.data(), _overlay_upload.card_data.size());
        const VkDeviceSize blur_bytes = _overlay_upload.card_enabled
            ? std::max<VkDeviceSize>(static_cast<VkDeviceSize>(_overlay_upload.blur.blur_w) * static_cast<VkDeviceSize>(_overlay_upload.blur.blur_h) * 4 * sizeof(float), 64)
            : 64;
        _ensure_buffer(slot.blur_a, blur_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
        if (_overlay_upload.card_enabled)
        {
            _ensure_buffer(slot.blur_b, blur_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, { VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 });
        }
        _ensure_buffer(slot.output, gpu.picture_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        });

        const VkBuffer picture_buffer = gpu.pictures[static_cast<size_t>(picture)].buffer;
        const VkDescriptorBufferInfo buffers[7] = {
            { picture_buffer, 0, VK_WHOLE_SIZE },
            { slot.segments.buffer, 0, VK_WHOLE_SIZE },
            { slot.tiles.buffer, 0, VK_WHOLE_SIZE },
            { slot.text.buffer, 0, VK_WHOLE_SIZE },
            { slot.output.buffer, 0, VK_WHOLE_SIZE },
            { slot.blur_a.buffer, 0, VK_WHOLE_SIZE },
            { slot.card_data.buffer, 0, VK_WHOLE_SIZE },
        };
        const VkDescriptorBufferInfo blur_a_buffers[3] = {
            { picture_buffer, 0, VK_WHOLE_SIZE },
            { slot.blur_b.buffer, 0, VK_WHOLE_SIZE },
            { slot.blur_a.buffer, 0, VK_WHOLE_SIZE },
        };
        const VkDescriptorBufferInfo blur_b_buffers[3] = {
            { picture_buffer, 0, VK_WHOLE_SIZE },
            { slot.blur_a.buffer, 0, VK_WHOLE_SIZE },
            { slot.blur_b.buffer, 0, VK_WHOLE_SIZE },
        };
        std::array<VkWriteDescriptorSet, 13> writes {};
        uint32_t write_count = 0;
        auto push_writes = [&](VkDescriptorSet dst_set, const VkDescriptorBufferInfo * infos, uint32_t count) {
            for (uint32_t b = 0; b < count; ++b)
            {
                VkWriteDescriptorSet & w = writes[write_count++];
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = dst_set;
                w.dstBinding = b;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &infos[b];
            }
        };
        push_writes(slot.set, buffers, 7);
        if (_overlay_upload.card_enabled)
        {
            push_writes(slot.blur_set_a, blur_a_buffers, 3);
            push_writes(slot.blur_set_b, blur_b_buffers, 3);
        }
        _fn.vkUpdateDescriptorSets(_device, write_count, writes.data(), 0, nullptr);

        check(_fn.vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        auto begin = vk_init<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(_fn.vkBeginCommandBuffer(slot.command_buffer, &begin), "vkBeginCommandBuffer");

        auto before = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        before.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &before, 0, nullptr, 0, nullptr);

        if (_overlay_upload.card_enabled)
        {
            const uint32_t blur_gx = (_overlay_upload.blur.blur_w + 7u) / 8u;
            const uint32_t blur_gy = (_overlay_upload.blur.blur_h + 7u) / 8u;
            OverlayBlurPush blur_push = _overlay_upload.blur;
            auto blur_barrier = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
            blur_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            blur_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_blur.pipeline);
            blur_push.pass = 0u;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_blur.layout, 0, 1, &slot.blur_set_a, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _overlay_blur.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_push), &blur_push);
            _fn.vkCmdDispatch(slot.command_buffer, blur_gx, blur_gy, 1);
            _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &blur_barrier, 0, nullptr, 0, nullptr);

            blur_push.pass = 1u;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_blur.layout, 0, 1, &slot.blur_set_b, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _overlay_blur.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_push), &blur_push);
            _fn.vkCmdDispatch(slot.command_buffer, blur_gx, blur_gy, 1);
            _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &blur_barrier, 0, nullptr, 0, nullptr);

            blur_push.pass = 2u;
            _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_blur.layout, 0, 1, &slot.blur_set_a, 0, nullptr);
            _fn.vkCmdPushConstants(slot.command_buffer, _overlay_blur.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blur_push), &blur_push);
            _fn.vkCmdDispatch(slot.command_buffer, blur_gx, blur_gy, 1);
            _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &blur_barrier, 0, nullptr, 0, nullptr);
        }

        _fn.vkCmdBindPipeline(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.pipeline);
        _fn.vkCmdBindDescriptorSets(slot.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _overlay_draw.layout, 0, 1, &slot.set, 0, nullptr);
        _fn.vkCmdPushConstants(slot.command_buffer, _overlay_draw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        _fn.vkCmdDispatch(slot.command_buffer, (gpu.row_words + 7) / 8, (gpu.height / 2 + 7) / 8, 1);

        auto host = vk_init<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        _fn.vkCmdPipelineBarrier(slot.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &host, 0, nullptr, 0, nullptr);
        check(_fn.vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        check(_fn.vkResetFences(_device, 1, &slot.fence), "vkResetFences");
        auto submit_info = vk_init<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &slot.command_buffer;
        check(_fn.vkQueueSubmit(_queue, 1, &submit_info, slot.fence), "vkQueueSubmit");

        slot.busy = true;
        ++stream.draw_submitted;

        uint32_t & draws = gpu.picture_draws.at(static_cast<size_t>(picture));
        if (++draws >= gpu.streams.size())
        {
            draws = 0;
            gpu.free_pictures.push_back(picture);
        }
    }

    void _finish_video(size_t index, size_t stream_index)
    {
        GpuVideo & gpu = _gpu_videos.at(index);
        GpuStream & stream = gpu.streams.at(stream_index);
        while (stream.draw_collected < stream.draw_submitted)
        {
            _collect_draw(gpu, stream);
        }
        if (stream.encoder) stream.encoder->finish();
    }

    void _release_streams(GpuVideo & gpu)
    {
        if (_device && _fn.vkDeviceWaitIdle) _fn.vkDeviceWaitIdle(_device);
        for (GpuStream & stream : gpu.streams)
        {
            for (DrawSlot & slot : stream.draw_slots)
            {
                av_frame_free(&slot.av_frame);
                if (_device)
                {
                    for (Buffer * buffer : { &slot.segments, &slot.tiles, &slot.text, &slot.card_data, &slot.blur_a, &slot.blur_b, &slot.output })
                    {
                        _destroy_buffer(*buffer);
                    }
                    if (slot.fence) _fn.vkDestroyFence(_device, slot.fence, nullptr);
                }
                slot.fence = VK_NULL_HANDLE;
                slot.busy = false;
            }
            if (_device && stream.descriptor_pool) _fn.vkDestroyDescriptorPool(_device, stream.descriptor_pool, nullptr);
            stream.descriptor_pool = VK_NULL_HANDLE;
            stream.encoder.reset();
        }
        gpu.streams.clear();
        gpu.active = false;
    }

    void _create_pipeline()
    {
        std::array<VkDescriptorSetLayoutBinding, kBindings> bindings {};
        for (uint32_t b = 0; b < kBindings; ++b)
        {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        auto set_layout_info = vk_init<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
        set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        set_layout_info.pBindings = bindings.data();
        check(_fn.vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &_set_layout), "vkCreateDescriptorSetLayout");

        VkPushConstantRange push_range { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
        auto layout_info = vk_init<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &_set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        check(_fn.vkCreatePipelineLayout(_device, &layout_info, nullptr, &_pipeline_layout), "vkCreatePipelineLayout");

        auto module_info = vk_init<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        module_info.codeSize = sizeof(kYuvDiffSpirv);
        module_info.pCode = kYuvDiffSpirv;
        VkShaderModule module = VK_NULL_HANDLE;
        check(_fn.vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");
        auto pipeline_info = vk_init<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = _pipeline_layout;
        const VkResult result = _fn.vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_pipeline);
        _fn.vkDestroyShaderModule(_device, module, nullptr);
        check(result, "vkCreateComputePipelines");
    }

    Compute _create_compute(const uint32_t * code, size_t code_bytes, uint32_t bindings, uint32_t push_bytes)
    {
        Compute compute;
        try
        {
            std::vector<VkDescriptorSetLayoutBinding> layout_bindings(bindings);
            for (uint32_t b = 0; b < bindings; ++b)
            {
                layout_bindings[b].binding = b;
                layout_bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                layout_bindings[b].descriptorCount = 1;
                layout_bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            auto set_layout_info = vk_init<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
            set_layout_info.bindingCount = bindings;
            set_layout_info.pBindings = layout_bindings.data();
            check(_fn.vkCreateDescriptorSetLayout(_device, &set_layout_info, nullptr, &compute.set_layout), "vkCreateDescriptorSetLayout");

            VkPushConstantRange push_range { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
            auto layout_info = vk_init<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
            layout_info.setLayoutCount = 1;
            layout_info.pSetLayouts = &compute.set_layout;
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &push_range;
            check(_fn.vkCreatePipelineLayout(_device, &layout_info, nullptr, &compute.layout), "vkCreatePipelineLayout");

            auto module_info = vk_init<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
            module_info.codeSize = code_bytes;
            module_info.pCode = code;
            VkShaderModule module = VK_NULL_HANDLE;
            check(_fn.vkCreateShaderModule(_device, &module_info, nullptr, &module), "vkCreateShaderModule");
            auto pipeline_info = vk_init<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
            pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipeline_info.stage.module = module;
            pipeline_info.stage.pName = "main";
            pipeline_info.layout = compute.layout;
            const VkResult result = _fn.vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &compute.pipeline);
            _fn.vkDestroyShaderModule(_device, module, nullptr);
            check(result, "vkCreateComputePipelines");
        }
        catch (...)
        {
            _destroy_compute(compute);
            throw;
        }
        return compute;
    }

    void _destroy_compute(Compute & compute)
    {
        if (compute.pipeline) _fn.vkDestroyPipeline(_device, compute.pipeline, nullptr);
        if (compute.layout) _fn.vkDestroyPipelineLayout(_device, compute.layout, nullptr);
        if (compute.set_layout) _fn.vkDestroyDescriptorSetLayout(_device, compute.set_layout, nullptr);
        compute = Compute();
    }

    void _create_slots()
    {
        const uint32_t set_count = static_cast<uint32_t>(kSlots * _videos.size());
        VkDescriptorPoolSize pool_size { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (kBindings + 2 + 4 + 3 + 3 + 4 + 2 + 2 + 2) * set_count };
        auto pool_info = vk_init<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
        pool_info.maxSets = 9 * set_count;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(_fn.vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool), "vkCreateDescriptorPool");

        auto command_pool_info = vk_init<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = _queue_family;
        check(_fn.vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool), "vkCreateCommandPool");

        for (Slot & slot : _slots)
        {
            slot.videos.resize(_videos.size());
            auto command_info = vk_init<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
            command_info.commandPool = _command_pool;
            command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_info.commandBufferCount = 1;
            check(_fn.vkAllocateCommandBuffers(_device, &command_info, &slot.command_buffer), "vkAllocateCommandBuffers");
            auto fence_info = vk_init<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
            check(_fn.vkCreateFence(_device, &fence_info, nullptr, &slot.fence), "vkCreateFence");
            for (SlotVideo & video : slot.videos)
            {
                video.frame = av_frame_alloc();
                if (!video.frame) throw std::runtime_error("out of memory");
                auto set_info = vk_init<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
                set_info.descriptorPool = _descriptor_pool;
                set_info.descriptorSetCount = 1;
                set_info.pSetLayouts = &_set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.descriptor_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_pyramid.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.pyramid_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_lk.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.lk_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_select.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.select_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_texture.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.texture_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_overlay_base.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.base_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_lattice.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.lattice_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_edge.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.edge_set), "vkAllocateDescriptorSets");
                set_info.pSetLayouts = &_stairs.set_layout;
                check(_fn.vkAllocateDescriptorSets(_device, &set_info, &video.stairs_set), "vkAllocateDescriptorSets");
            }
        }
    }

    Buffer _create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, std::initializer_list<VkMemoryPropertyFlags> preferred)
    {
        Buffer result;
        result.size = size;
        auto buffer_info = vk_init<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(_fn.vkCreateBuffer(_device, &buffer_info, nullptr, &result.buffer), "vkCreateBuffer");
        try
        {
            VkMemoryRequirements requirements;
            _fn.vkGetBufferMemoryRequirements(_device, result.buffer, &requirements);
            uint32_t type_index = UINT32_MAX;
            for (VkMemoryPropertyFlags flags : preferred)
            {
                for (uint32_t i = 0; i < _memory_properties.memoryTypeCount && type_index == UINT32_MAX; ++i)
                {
                    if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
                    if ((_memory_properties.memoryTypes[i].propertyFlags & flags) == flags) type_index = i;
                }
                if (type_index != UINT32_MAX) break;
            }
            if (type_index == UINT32_MAX) throw std::runtime_error("macos vulkan pipeline: no suitable memory type");
            auto allocate_info = vk_init<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
            allocate_info.allocationSize = requirements.size;
            allocate_info.memoryTypeIndex = type_index;
            check(_fn.vkAllocateMemory(_device, &allocate_info, nullptr, &result.memory), "vkAllocateMemory");
            check(_fn.vkBindBufferMemory(_device, result.buffer, result.memory, 0), "vkBindBufferMemory");
            const VkMemoryPropertyFlags properties = _memory_properties.memoryTypes[type_index].propertyFlags;
            result.coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                check(_fn.vkMapMemory(_device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory");
            }
        }
        catch (...)
        {
            _destroy_buffer(result);
            throw;
        }
        return result;
    }

    void _destroy_buffer(Buffer & buffer)
    {
        if (buffer.mapped) _fn.vkUnmapMemory(_device, buffer.memory);
        if (buffer.buffer) _fn.vkDestroyBuffer(_device, buffer.buffer, nullptr);
        if (buffer.memory) _fn.vkFreeMemory(_device, buffer.memory, nullptr);
        buffer = Buffer();
    }

    void _ensure_buffer(Buffer & buffer, VkDeviceSize bytes, VkBufferUsageFlags usage, std::initializer_list<VkMemoryPropertyFlags> preferred)
    {
        if (buffer.size >= bytes) return;
        _destroy_buffer(buffer);
        buffer = _create_buffer(std::max(bytes, buffer.size * 2), usage, preferred);
    }

    void _flush(const Buffer & buffer) const
    {
        if (buffer.coherent || !buffer.memory) return;
        auto range = vk_init<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = buffer.memory;
        range.size = VK_WHOLE_SIZE;
        check(_fn.vkFlushMappedMemoryRanges(_device, 1, &range), "vkFlushMappedMemoryRanges");
    }

    void _invalidate(const Buffer & buffer) const
    {
        if (buffer.coherent || !buffer.memory) return;
        auto range = vk_init<VkMappedMemoryRange>(VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        range.memory = buffer.memory;
        range.size = VK_WHOLE_SIZE;
        check(_fn.vkInvalidateMappedMemoryRanges(_device, 1, &range), "vkInvalidateMappedMemoryRanges");
    }

    void _upload(Buffer & buffer, const void * data, size_t bytes)
    {
        _ensure_buffer(buffer, std::max<VkDeviceSize>(bytes, 32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        });
        if (bytes > 0) std::memcpy(buffer.mapped, data, bytes);
        _flush(buffer);
    }

    void _release()
    {
        if (_device && _fn.vkDeviceWaitIdle)
        {
            _fn.vkDeviceWaitIdle(_device);
        }
        for (GpuVideo & gpu : _gpu_videos)
        {
            _release_streams(gpu);
        }
        if (_device && _fn.vkDestroyBuffer)
        {
            for (GpuVideo & gpu : _gpu_videos)
            {
                for (Buffer & buffer : gpu.pictures) _destroy_buffer(buffer);
                gpu.pictures.clear();
            }
            _destroy_compute(_overlay_base);
            _destroy_compute(_overlay_blur);
            _destroy_compute(_overlay_draw);
            for (Slot & slot : _slots)
            {
                for (SlotVideo & video : slot.videos)
                {
                    _destroy_buffer(video.row_counts);
                    _destroy_buffer(video.motion_points);
                    _destroy_buffer(video.candidates);
                    _destroy_buffer(video.sparse_state);
                    _destroy_buffer(video.lattice_sums);
                    _destroy_buffer(video.edge_power);
                    _destroy_buffer(video.edge_stairs);
                }
                if (slot.fence) _fn.vkDestroyFence(_device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
            for (VideoState & video : _videos)
            {
                for (Buffer & buffer : video.luma) _destroy_buffer(buffer);
                for (Buffer & buffer : video.chroma) _destroy_buffer(buffer);
                for (Buffer & buffer : video.pyramid) _destroy_buffer(buffer);
                _destroy_buffer(video.products);
                _destroy_buffer(video.sparse_latch);
            }
            _destroy_compute(_pyramid);
            _destroy_compute(_lk);
            _destroy_compute(_select);
            _destroy_compute(_texture);
            _destroy_compute(_lattice);
            _destroy_compute(_edge);
            _destroy_compute(_stairs);
            if (_command_pool) _fn.vkDestroyCommandPool(_device, _command_pool, nullptr);
            if (_descriptor_pool) _fn.vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
            if (_pipeline) _fn.vkDestroyPipeline(_device, _pipeline, nullptr);
            if (_pipeline_layout) _fn.vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
            if (_set_layout) _fn.vkDestroyDescriptorSetLayout(_device, _set_layout, nullptr);
            _command_pool = VK_NULL_HANDLE;
            _descriptor_pool = VK_NULL_HANDLE;
            _pipeline = VK_NULL_HANDLE;
            _pipeline_layout = VK_NULL_HANDLE;
            _set_layout = VK_NULL_HANDLE;
            _fn.vkDestroyDevice(_device, nullptr);
            _device = VK_NULL_HANDLE;
        }
        for (Slot & slot : _slots)
        {
            for (SlotVideo & video : slot.videos) av_frame_free(&video.frame);
        }
        for (VideoState & video : _videos)
        {
            av_frame_free(&video.first_frame);
            av_frame_free(&video.sw_frame);
            video.decoder.reset();
        }
        av_buffer_unref(&_vt_device);
        if (_instance && _ifn.vkDestroyInstance)
        {
            _ifn.vkDestroyInstance(_instance, nullptr);
            _instance = VK_NULL_HANDLE;
        }
    }

    VkInstance _instance = VK_NULL_HANDLE;
    struct {
#define FRESNIFF_VK_DECLARE(name) PFN_##name name = nullptr;
        FRESNIFF_VK_MAC_INSTANCE_FUNCTIONS(FRESNIFF_VK_DECLARE)
#undef FRESNIFF_VK_DECLARE
    } _ifn;

    VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
    VkDevice _device = VK_NULL_HANDLE;
    VkQueue _queue = VK_NULL_HANDLE;
    uint32_t _queue_family = 0;
    int _device_index = 0;
    std::string _device_name;
    uint32_t _max_workgroups = 65535;
    uint32_t _max_group_size = 0;
    uint32_t _max_group_invocations = 0;
    VkPhysicalDeviceMemoryProperties _memory_properties {};
    struct {
#define FRESNIFF_VK_DECLARE(name) PFN_##name name = nullptr;
        FRESNIFF_VK_MAC_DEVICE_FUNCTIONS(FRESNIFF_VK_DECLARE)
#undef FRESNIFF_VK_DECLARE
    } _fn;

    AVBufferRef * _vt_device = nullptr;

    bool _motion = false;
    bool _motion_detail = false;
    bool _native_pictures = false;
    cv::Mat _scratch;
    std::vector<GpuVideo> _gpu_videos;
    OverlayUpload _overlay_upload;
    Compute _overlay_base;
    Compute _overlay_blur;
    Compute _overlay_draw;
    Compute _pyramid;
    Compute _lk;
    Compute _select;
    Compute _texture;
    MotionCellBinner _binner;
    bool _resolution = false;
    Compute _lattice;
    Compute _edge;
    Compute _stairs;

    VkDescriptorSetLayout _set_layout = VK_NULL_HANDLE;
    VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
    VkPipeline _pipeline = VK_NULL_HANDLE;
    VkDescriptorPool _descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool _command_pool = VK_NULL_HANDLE;

    std::vector<VideoState> _videos;
    std::array<Slot, kSlots> _slots;
    size_t _submitted = 0;
    size_t _collected = 0;
    long long _frames_read = 0;
};

} // namespace

std::unique_ptr<FramePipeline> make_macos_vulkan_pipeline(const std::vector<std::string> & paths, int device_index)
{
    return std::make_unique<VulkanMacosPipeline>(paths, device_index);
}

} // namespace fresniff
