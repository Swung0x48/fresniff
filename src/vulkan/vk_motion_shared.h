#pragma once

// What the pipelines that analyze on the GPU share: the constants the compute shaders are called with and
// how a motion video scene is handed to overlay_draw.comp. The Vulkan Video pipeline (FFmpeg, desktop) and
// the MediaCodec pipeline (Android) run the same shaders on the same buffer layouts.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "motion.h"
#include "motion_video_target.h"

namespace fresniff {

//! must match yuv_diff.comp
struct PushConstants {
    uint32_t width;
    uint32_t luma_row_words;
    uint32_t threshold;
    uint32_t chroma_mode;
    uint32_t shift_x;
    uint32_t shift_y;
    uint32_t chroma_row_bytes;
    uint32_t v_offset;
};

//! must match motion_pyramid.comp
struct PyramidPush {
    uint32_t mode;
    uint32_t factor;
    uint32_t src_width;
    uint32_t src_height;
    uint32_t src_base;
    uint32_t dst_width;
    uint32_t dst_height;
    uint32_t dst_offset;
};

constexpr uint32_t kMaxMotionLevels = 4;

//! must match motion_lk.comp
struct LkPush {
    uint32_t level_width[kMaxMotionLevels];
    uint32_t level_height[kMaxMotionLevels];
    uint32_t level_offset[kMaxMotionLevels];
    uint32_t levels;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t iterations;
    float max_residual;
    uint32_t list;
    uint32_t list_output;
};

//! must match motion_select.comp
struct SelectPush {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t cell_points;
    uint32_t min_cell_points;
    uint32_t sparse_cells;
    uint32_t products_groups;
    uint32_t parts_groups;
    uint32_t points_groups;
    float min_texture;
    float max_back_error;
    float moving_squared;
};

//! what motion_select.comp writes: the indirect dispatches of the points added for sparse motion, whether the
//! frame has it, and per cell whether it shows something static
struct SparseState {
    uint32_t commands[9];
    uint32_t sparse;
    uint32_t with_motion;
    uint32_t complete;
    uint32_t cell_static[kMotionCells];
};

//! what motion_select.comp keeps of the last frame of a video with motion (starts as zeros)
struct SparseLatch {
    uint32_t sparse;
    uint32_t cell_static[kMotionCells];
};

//! must match motion_texture.comp
struct TexturePush {
    uint32_t mode;
    uint32_t width;
    uint32_t height;
    uint32_t offset;
    uint32_t level0_width;
    uint32_t level0_height;
    uint32_t parts_x;
    uint32_t parts_y;
    uint32_t cell_parts;
    int32_t radius;
    float minimum;
};
constexpr uint32_t kTextureGroupSize = 64; // local_size_x of motion_texture.comp

//! the motion is measured at the grid of points of motion.h (kMotionPoints of them), at its resolution
constexpr uint32_t kMotionIterations = 20; // at most, per level
constexpr uint32_t kMotionGroupSize = 256; // local_size_x of motion_lk.comp
//! the frames have to match at a point after the tracking (mean absolute difference)
constexpr float kMaxResidual = 12.0f;

//! pyramid of the luma plane a frame's motion is measured on
struct MotionLayout {
    uint32_t scale = 1;
    MotionGrid grid;
    uint32_t levels = 0;
    uint32_t width[kMaxMotionLevels] = {};
    uint32_t height[kMaxMotionLevels] = {};
    uint32_t offset[kMaxMotionLevels] = {};
    uint64_t bytes = 0;
};

//! the dispatches of a frame with sparse motion (motion.h, kMotionSparseCells), recorded after the tracking of the
//! regular grid: what the grid found, then - only if it found sparse motion, the dispatches are indirect - the
//! texture of the previous frame, the point with the most texture per part of a cell and the tracking of those.
//! Used by both GPU pipelines, which differ in how they call Vulkan: `dispatch(stage)` binds the pipeline and the
//! set of the stage and dispatches it with the push constants this fills in.
struct SparseMotionPushes {
    SelectPush select {};
    TexturePush products {};
    TexturePush parts {};
    LkPush points {};
};

inline bool sparse_motion_possible(const MotionLayout & layout) { return layout.levels >= 2; }

//! lk: the push constants of the tracking of the regular grid
inline SparseMotionPushes sparse_motion_pushes(const MotionLayout & layout, const LkPush & lk)
{
    SparseMotionPushes pushes;
    const uint32_t parts_x = static_cast<uint32_t>(layout.grid.cells_x * kMotionTextureParts);
    const uint32_t parts_y = static_cast<uint32_t>(layout.grid.cells_y * kMotionTextureParts);
    pushes.select.grid_x = static_cast<uint32_t>(layout.grid.points_x());
    pushes.select.grid_y = static_cast<uint32_t>(layout.grid.points_y());
    pushes.select.cell_points = static_cast<uint32_t>(kMotionCellPoints);
    pushes.select.min_cell_points = static_cast<uint32_t>(kMotionMinCellPoints);
    pushes.select.sparse_cells = static_cast<uint32_t>(kMotionSparseCells);
    pushes.select.products_groups = (layout.width[1] * layout.height[1] + kTextureGroupSize - 1) / kTextureGroupSize;
    pushes.select.parts_groups = (parts_x * parts_y + kTextureGroupSize - 1) / kTextureGroupSize;
    pushes.select.points_groups = parts_x * parts_y;
    pushes.select.min_texture = kMotionMinTexture;
    pushes.select.max_back_error = kMotionMaxBackError;
    pushes.select.moving_squared = kMotionMovingPoint * kMotionMovingPoint;

    pushes.products.mode = 0;
    pushes.products.width = layout.width[1];
    pushes.products.height = layout.height[1];
    pushes.products.offset = layout.offset[1];
    pushes.products.level0_width = layout.width[0];
    pushes.products.level0_height = layout.height[0];
    pushes.products.parts_x = parts_x;
    pushes.products.parts_y = parts_y;
    pushes.products.cell_parts = static_cast<uint32_t>(kMotionTextureParts);
    pushes.products.radius = kMotionTextureBlock / 2;
    pushes.products.minimum = kMotionTextureMin;
    pushes.parts = pushes.products;
    pushes.parts.mode = 1;

    pushes.points = lk;
    pushes.points.list = 1;
    pushes.points.list_output = static_cast<uint32_t>(kMotionPoints);
    return pushes;
}

//! bytes of the buffers of a frame: the tracked points (regular grid, then the added ones), the added points as
//! motion_texture.comp chose them, the gradient products of level 1
constexpr uint64_t kMotionPointBytes = static_cast<uint64_t>(kMotionPoints + kMotionTexturePoints) * 4 * sizeof(float);
constexpr uint64_t kMotionCandidateBytes = static_cast<uint64_t>(kMotionTexturePoints) * 4 * sizeof(float);
inline uint64_t motion_product_bytes(const MotionLayout & layout) { return static_cast<uint64_t>(layout.width[1]) * layout.height[1] * 4 * sizeof(float); }

//! Adds the points of a frame with sparse motion to the cells. state, candidates, points: what the shaders wrote
//! (points: the whole buffer). add(x, y, dx, dy, texture, returns): a point at (x, y) of the measured resolution
template <typename Add>
inline void read_sparse_motion(const SparseState & state, const float * candidates, const float * points, Add && add)
{
    if (!state.sparse) return;
    for (int part = 0; part < kMotionTexturePoints; ++part)
    {
        const float * candidate = candidates + 4 * part;
        if (!(candidate[2] > 0.0f)) continue;
        const float * point = points + 4 * (kMotionPoints + part);
        const bool texture = point[2] >= kMotionMinTexture && std::isfinite(point[0]) && std::isfinite(point[1]);
        add(candidate[0], candidate[1], point[0], point[1], texture, texture && point[3] <= kMotionMaxBackError);
    }
}

inline MotionLayout motion_layout(int width, int height)
{
    MotionLayout layout;
    layout.scale = static_cast<uint32_t>(motion_scale(width, height));
    layout.grid = motion_grid(width, height);
    uint32_t w = static_cast<uint32_t>(width) / layout.scale;
    uint32_t h = static_cast<uint32_t>(height) / layout.scale;
    uint32_t offset = 0;
    while (layout.levels < kMaxMotionLevels && (layout.levels == 0 || std::min(w, h) >= 32))
    {
        layout.width[layout.levels] = w;
        layout.height[layout.levels] = h;
        layout.offset[layout.levels] = offset;
        offset += w * h;
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        ++layout.levels;
    }
    layout.bytes = static_cast<uint64_t>(offset) * sizeof(float);
    return layout;
}

//! must match overlay_base.comp
struct OverlayBasePush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t scale;
    uint32_t luma_row_words;
    uint32_t chroma_mode;
    uint32_t chroma_row_bytes;
    uint32_t v_offset;
    uint32_t chroma_step_x;
    uint32_t chroma_step_y;
};

//! must match overlay_blur.comp
struct OverlayBlurPush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t roi_x;
    uint32_t roi_y;
    uint32_t blur_w;
    uint32_t blur_h;
    uint32_t ds;
    uint32_t pass;
    int32_t radius;
    float sigma;
    float dim;
};
static_assert(sizeof(OverlayBlurPush) == 48 && sizeof(OverlayBlurPush) <= 128, "OverlayBlurPush size");

constexpr uint32_t kOverlayFlagCard = 1u << 0;
constexpr uint32_t kOverlayFlagRgba = 1u << 1;
constexpr uint32_t kOverlayFlagBt709 = 1u << 2;

//! must match overlay_draw.comp
struct OverlayDrawPush {
    uint32_t width;
    uint32_t height;
    uint32_t row_words;
    uint32_t tile_size;
    uint32_t tiles_x;
    uint32_t tiles_y;
    uint32_t box_x;
    uint32_t box_y;
    uint32_t box_width;
    uint32_t box_height;
    uint32_t text_row_words;
    float dim;
    float box_dim;
    uint32_t flags;
};
static_assert(sizeof(OverlayDrawPush) == 56 && sizeof(OverlayDrawPush) <= 128, "OverlayDrawPush size");

//! must match binding 6 (Card) in overlay_draw.comp (std430, followed by vec2 samples[])
struct GpuCardHeader {
    float card_rect[4];            // x, y, width, height
    uint32_t expand_rect[4];       // x0, y0, x1, y1
    uint32_t blur_info[4];         // roi_x, roi_y, blur_w, blur_h
    uint32_t blur_ds;
    uint32_t has_chart;
    uint32_t chart_cols;
    uint32_t grid_count;
    float corner_radius;
    float vibrancy;
    float sheen_alpha;
    float sheen_height;
    float tint_top[4];             // Y, U, V, alpha
    float tint_bottom[4];          // Y, U, V, alpha
    float border_width;
    float border_alpha_top;
    float border_alpha_bottom;
    float border_power;
    float shadow_offset_y;
    float shadow_alpha;
    float shadow_decay;
    float shadow_pad;
    float plot_rect[4];            // x, y, width, height
    float grid_y[4];
    float grid_alpha[4];
    float primary_color[4];        // Y, U, V, alpha
    float secondary_color[4];      // Y, U, V, alpha
    float primary_half_width;
    float secondary_half_width;
    float primary_fill_alpha;
    float primary_fill_power;
    float grid_half_width;
    float primary_glow_alpha;
    float secondary_glow_alpha;
    float pad0;
    float primary_dot[4];          // x, y, core_radius, glow_radius
    float secondary_dot[4];        // x, y, core_radius, glow_radius
};
static_assert(sizeof(GpuCardHeader) == 288, "GpuCardHeader size");
static_assert(offsetof(GpuCardHeader, blur_ds) == 48, "GpuCardHeader blur_ds offset");
static_assert(offsetof(GpuCardHeader, tint_top) == 80, "GpuCardHeader tint_top offset");
static_assert(offsetof(GpuCardHeader, plot_rect) == 144, "GpuCardHeader plot_rect offset");
static_assert(offsetof(GpuCardHeader, primary_color) == 192, "GpuCardHeader primary_color offset");
static_assert(offsetof(GpuCardHeader, primary_dot) == 256, "GpuCardHeader primary_dot offset");

constexpr size_t kMinCardBufferBytes = sizeof(GpuCardHeader) + 2 * sizeof(float);

//! limited-range RGB (0..255) -> YUV (Y: 16..235, U/V: 16..240)
inline void rgb_to_yuv_limited(float r, float g, float b, bool bt709, float out[3])
{
    if (bt709)
    {
        out[0] = 16.0f + (46.559f * r + 156.629f * g + 15.812f * b) / 255.0f;
        out[1] = 128.0f + (-25.664f * r - 86.336f * g + 112.0f * b) / 255.0f;
        out[2] = 128.0f + (112.0f * r - 101.730f * g - 10.270f * b) / 255.0f;
    }
    else
    {
        // as OpenCV converts BGR to YUV
        out[0] = 16.0f + (65.481f * r + 128.553f * g + 24.966f * b) / 255.0f;
        out[1] = 128.0f + (-37.797f * r - 74.203f * g + 112.0f * b) / 255.0f;
        out[2] = 128.0f + (112.0f * r - 93.786f * g - 18.214f * b) / 255.0f;
    }
}

//! copies premultiplied RGBA8 (CV_8UC4) into tightly packed destination memory
inline void copy_rgba_overlay(const cv::Mat & overlay, void * dst)
{
    const size_t row_bytes = static_cast<size_t>(overlay.cols) * 4;
    if (overlay.isContinuous())
    {
        std::memcpy(dst, overlay.data, row_bytes * static_cast<size_t>(overlay.rows));
        return;
    }
    uint8_t * out = static_cast<uint8_t *>(dst);
    for (int y = 0; y < overlay.rows; ++y)
    {
        std::memcpy(out + static_cast<size_t>(y) * row_bytes, overlay.ptr(y), row_bytes);
    }
}

//! the segments of a motion video frame are sorted into tiles of this size (a multiple of 4)
constexpr uint32_t kOverlayTile = 32;

//! a motion video scene as the buffers and the constants of overlay_blur.comp and overlay_draw.comp
struct OverlayUpload {
    //! 8 floats per segment: x0, y0, x1, y1, radius, Y, U, V
    std::vector<float> segments;
    //! per tile where its segments start in the index list (one more entry than tiles), then the indices
    std::vector<uint32_t> tiles;
    //! legacy mode: coverage of the text, rows padded to whole uints
    std::vector<uint8_t> text;
    //! frosted-glass card mode: GpuCardHeader followed by chart_cols * vec2 samples
    std::vector<uint8_t> card_data;
    bool card_enabled = false;
    bool rgba_overlay = false;
    size_t overlay_bytes = 0;
    uint64_t overlay_generation = 0;
    OverlayBlurPush blur {};
    OverlayDrawPush push {};
};

//! width, height, row_words: of the picture (see overlay_base.comp).
//! bt709: the colors become YUV with the BT.709 matrix instead of BT.601 (limited range)
inline void build_overlay_upload(const MotionVideoScene & scene, uint32_t width, uint32_t height, uint32_t row_words, bool bt709, OverlayUpload & upload)
{
    const int tiles_x = static_cast<int>((width + kOverlayTile - 1) / kOverlayTile);
    const int tiles_y = static_cast<int>((height + kOverlayTile - 1) / kOverlayTile);
    const size_t tile_count = static_cast<size_t>(tiles_x) * static_cast<size_t>(tiles_y);
    // the tiles a segment reaches into; overlay_draw.comp fades a segment out a pixel beyond its radius
    auto tiles_of = [&](const MotionVideoSegment & segment, int & x0, int & x1, int & y0, int & y1) {
        const float reach = segment.radius + 1.5f;
        const float tile = static_cast<float>(kOverlayTile);
        x0 = static_cast<int>(std::floor((std::min(segment.x0, segment.x1) - reach) / tile));
        x1 = static_cast<int>(std::floor((std::max(segment.x0, segment.x1) + reach) / tile));
        y0 = static_cast<int>(std::floor((std::min(segment.y0, segment.y1) - reach) / tile));
        y1 = static_cast<int>(std::floor((std::max(segment.y0, segment.y1) + reach) / tile));
        if (x1 < 0 || y1 < 0 || x0 >= tiles_x || y0 >= tiles_y) return false;
        x0 = std::max(x0, 0);
        y0 = std::max(y0, 0);
        x1 = std::min(x1, tiles_x - 1);
        y1 = std::min(y1, tiles_y - 1);
        return true;
    };
    std::vector<uint32_t> & tiles = upload.tiles;
    tiles.assign(tile_count + 1, 0);
    int x0, x1, y0, y1;
    for (const MotionVideoSegment & segment : scene.segments)
    {
        if (!tiles_of(segment, x0, x1, y0, y1)) continue;
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x) tiles[static_cast<size_t>(y * tiles_x + x) + 1] += 1;
        }
    }
    for (size_t t = 0; t < tile_count; ++t) tiles[t + 1] += tiles[t];
    // the offsets stay in front, a copy of them counts up while the indices are filled in
    std::vector<uint32_t> next(tiles.begin(), tiles.begin() + static_cast<std::ptrdiff_t>(tile_count));
    tiles.resize(tile_count + 1 + tiles[tile_count], 0);
    upload.segments.resize(scene.segments.size() * 8);
    for (size_t s = 0; s < scene.segments.size(); ++s)
    {
        const MotionVideoSegment & segment = scene.segments[s];
        float * data = upload.segments.data() + 8 * s;
        data[0] = segment.x0;
        data[1] = segment.y0;
        data[2] = segment.x1;
        data[3] = segment.y1;
        // the sign tells overlay_draw.comp that the segment is drawn after the text box was darkened
        data[4] = segment.above_box ? -std::max(segment.radius, 1e-3f) : segment.radius;
        rgb_to_yuv_limited(static_cast<float>(segment.red), static_cast<float>(segment.green), static_cast<float>(segment.blue), bt709, data + 5);
        if (!tiles_of(segment, x0, x1, y0, y1)) continue;
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                const size_t tile = static_cast<size_t>(y * tiles_x + x);
                tiles[tile_count + 1 + next[tile]++] = static_cast<uint32_t>(s);
            }
        }
    }

    OverlayDrawPush & push = upload.push;
    push = OverlayDrawPush();
    push.width = width;
    push.height = height;
    push.row_words = row_words;
    push.tile_size = kOverlayTile;
    push.tiles_x = static_cast<uint32_t>(tiles_x);
    push.tiles_y = static_cast<uint32_t>(tiles_y);
    push.dim = scene.dim;
    push.box_dim = scene.box_dim;
    push.flags = bt709 ? kOverlayFlagBt709 : 0u;

    upload.text.clear();
    upload.card_enabled = false;
    upload.rgba_overlay = false;
    upload.overlay_bytes = 0;
    upload.overlay_generation = 0;
    upload.blur = OverlayBlurPush();

    const cv::Rect picture(0, 0, static_cast<int>(width), static_cast<int>(height));
    const MotionVideoCard & card = scene.card;
    if (card.enabled && !card.rect.empty() && (card.rect & picture) == card.rect && width >= 2u && height >= 2u)
    {
        upload.card_enabled = true;
        push.flags |= kOverlayFlagCard;

        const float sigma = std::max(0.5f, card.blur_sigma);
        const uint32_t ds = sigma < 10.0f ? 2u : (sigma <= 28.0f ? 4u : 8u);
        const float box_var = static_cast<float>(ds * ds - 1u) / 12.0f;
        const float sigma_ds = std::sqrt(std::max(0.25f, sigma * sigma - box_var)) / static_cast<float>(ds);
        const int32_t blur_radius = std::max<int32_t>(1, static_cast<int32_t>(std::ceil(3.0f * sigma_ds)));
        const int pad = (static_cast<int>(std::ceil(3.0f * sigma)) + 1) & ~1;

        const int rx0 = std::max(0, (card.rect.x - pad) & ~1);
        const int ry0 = std::max(0, (card.rect.y - pad) & ~1);
        const int rx1 = std::min(static_cast<int>(width), (card.rect.x + card.rect.width + pad + 1) & ~1);
        const int ry1 = std::min(static_cast<int>(height), (card.rect.y + card.rect.height + pad + 1) & ~1);
        const uint32_t blur_w = std::max(1u, (static_cast<uint32_t>(std::max(1, rx1 - rx0)) + ds - 1u) / ds);
        const uint32_t blur_h = std::max(1u, (static_cast<uint32_t>(std::max(1, ry1 - ry0)) + ds - 1u) / ds);

        upload.blur.width = width;
        upload.blur.height = height;
        upload.blur.row_words = row_words;
        upload.blur.roi_x = static_cast<uint32_t>(rx0);
        upload.blur.roi_y = static_cast<uint32_t>(ry0);
        upload.blur.blur_w = blur_w;
        upload.blur.blur_h = blur_h;
        upload.blur.ds = ds;
        upload.blur.pass = 0u;
        upload.blur.radius = blur_radius;
        upload.blur.sigma = sigma_ds;
        upload.blur.dim = scene.dim;

        GpuCardHeader header {};
        header.card_rect[0] = static_cast<float>(card.rect.x);
        header.card_rect[1] = static_cast<float>(card.rect.y);
        header.card_rect[2] = static_cast<float>(card.rect.width);
        header.card_rect[3] = static_cast<float>(card.rect.height);

        const int sh_pad_x = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(card.shadow_pad)) + 2 : 2;
        const int sh_pad_top = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(std::max(0.0f, card.shadow_pad - card.shadow_offset_y))) + 2 : 2;
        const int sh_pad_bot = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(card.shadow_pad + std::max(0.0f, card.shadow_offset_y))) + 2 : 2;
        header.expand_rect[0] = static_cast<uint32_t>(std::max(0, card.rect.x - sh_pad_x));
        header.expand_rect[1] = static_cast<uint32_t>(std::max(0, card.rect.y - sh_pad_top));
        header.expand_rect[2] = static_cast<uint32_t>(std::min(static_cast<int>(width), card.rect.x + card.rect.width + sh_pad_x));
        header.expand_rect[3] = static_cast<uint32_t>(std::min(static_cast<int>(height), card.rect.y + card.rect.height + sh_pad_bot));

        header.blur_info[0] = upload.blur.roi_x;
        header.blur_info[1] = upload.blur.roi_y;
        header.blur_info[2] = blur_w;
        header.blur_info[3] = blur_h;
        header.blur_ds = ds;

        const size_t chart_cols = card.has_chart ? std::max(card.primary_y.size(), card.secondary_y.size()) : 0;
        header.has_chart = (card.has_chart && chart_cols >= 2) ? 1u : 0u;
        header.chart_cols = static_cast<uint32_t>(chart_cols);
        header.grid_count = static_cast<uint32_t>(std::clamp(card.grid_count, 0, 4));

        header.corner_radius = card.corner_radius;
        header.vibrancy = card.vibrancy;
        header.sheen_alpha = card.sheen_alpha;
        header.sheen_height = card.sheen_height;

        auto pack_color = [&](const MotionVideoColor & c, float out[4]) {
            rgb_to_yuv_limited(static_cast<float>(c.r), static_cast<float>(c.g), static_cast<float>(c.b), bt709, out);
            out[3] = c.alpha;
        };
        pack_color(card.tint_top, header.tint_top);
        pack_color(card.tint_bottom, header.tint_bottom);
        header.border_width = card.border_width;
        header.border_alpha_top = card.border_alpha_top;
        header.border_alpha_bottom = card.border_alpha_bottom;
        header.border_power = card.border_power;
        header.shadow_offset_y = card.shadow_offset_y;
        header.shadow_alpha = card.shadow_alpha;
        header.shadow_decay = card.shadow_decay;
        header.shadow_pad = card.shadow_pad;

        header.plot_rect[0] = card.plot.x;
        header.plot_rect[1] = card.plot.y;
        header.plot_rect[2] = card.plot.width;
        header.plot_rect[3] = card.plot.height;
        for (int g = 0; g < 4; ++g)
        {
            header.grid_y[g] = card.grid_y[g];
            header.grid_alpha[g] = card.grid_alpha[g];
        }
        pack_color(card.primary_color, header.primary_color);
        pack_color(card.secondary_color, header.secondary_color);
        header.primary_half_width = card.primary_half_width;
        header.secondary_half_width = card.secondary_half_width;
        header.primary_fill_alpha = card.primary_fill_alpha;
        header.primary_fill_power = card.primary_fill_power;
        header.grid_half_width = card.grid_half_width;
        header.primary_glow_alpha = card.primary_glow_alpha;
        header.secondary_glow_alpha = card.secondary_glow_alpha;
        header.pad0 = 0.0f;

        header.primary_dot[0] = card.primary_dot_x;
        header.primary_dot[1] = card.primary_dot_y;
        header.primary_dot[2] = card.primary_dot_radius;
        header.primary_dot[3] = card.primary_glow_radius;
        header.secondary_dot[0] = card.secondary_dot_x;
        header.secondary_dot[1] = card.secondary_dot_y;
        header.secondary_dot[2] = card.secondary_dot_radius;
        header.secondary_dot[3] = card.secondary_glow_radius;

        const size_t sample_slots = std::max<size_t>(1, chart_cols);
        upload.card_data.resize(sizeof(GpuCardHeader) + sample_slots * 2 * sizeof(float));
        std::memcpy(upload.card_data.data(), &header, sizeof(GpuCardHeader));
        float * samples = reinterpret_cast<float *>(upload.card_data.data() + sizeof(GpuCardHeader));
        if (chart_cols == 0)
        {
            samples[0] = -1.0f;
            samples[1] = -1.0f;
        }
        else
        {
            for (size_t i = 0; i < chart_cols; ++i)
            {
                const float py = i < card.primary_y.size() && std::isfinite(card.primary_y[i]) ? card.primary_y[i] : -1.0f;
                const float sy = i < card.secondary_y.size() && std::isfinite(card.secondary_y[i]) ? card.secondary_y[i] : -1.0f;
                samples[2 * i] = py;
                samples[2 * i + 1] = sy;
            }
        }

        if (!scene.overlay.empty() && scene.overlay.size() == card.rect.size() && scene.overlay.type() == CV_8UC4)
        {
            push.flags |= kOverlayFlagRgba;
            push.box_x = static_cast<uint32_t>(card.rect.x);
            push.box_y = static_cast<uint32_t>(card.rect.y);
            push.box_width = static_cast<uint32_t>(card.rect.width);
            push.box_height = static_cast<uint32_t>(card.rect.height);
            push.text_row_words = push.box_width;
            upload.rgba_overlay = true;
            upload.overlay_bytes = static_cast<size_t>(push.box_width) * static_cast<size_t>(push.box_height) * 4;
            upload.overlay_generation = scene.overlay_generation;
        }
        return;
    }

    // legacy mode (ResolutionVideo): keep valid dummy CardData buffer size for Vulkan validation
    upload.card_data.assign(kMinCardBufferBytes, 0);
    if (!scene.box.empty() && (scene.box & picture) == scene.box && scene.text.size() == scene.box.size() && scene.text.type() == CV_8UC1)
    {
        push.box_x = static_cast<uint32_t>(scene.box.x);
        push.box_y = static_cast<uint32_t>(scene.box.y);
        push.box_width = static_cast<uint32_t>(scene.box.width);
        push.box_height = static_cast<uint32_t>(scene.box.height);
        push.text_row_words = (push.box_width + 3) / 4;
        upload.text.assign(static_cast<size_t>(push.text_row_words) * 4 * push.box_height, 0);
        for (int y = 0; y < scene.box.height; ++y)
        {
            std::memcpy(upload.text.data() + static_cast<size_t>(y) * push.text_row_words * 4, scene.text.ptr(y), static_cast<size_t>(scene.box.width));
        }
    }
}

} // namespace fresniff
