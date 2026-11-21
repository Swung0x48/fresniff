#include "motion_video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "video_writer.h"

namespace trdrop {

namespace {

//! brightness of the picture when --motion-arrows is enabled
constexpr float kArrowsDim = 0.72f;
//! motion in pixels (of the measured resolution) that gets the last color
constexpr double kLongestMotion = 45.0;
//! the dark end of the color map is left out
constexpr int kFirstColor = 48;
//! arrows shorter than this many pixels are drawn as a dot
constexpr double kShortestArrow = 1.5;
//! the tips of an arrow: this long (at most half the arrow), at 45 degrees
constexpr double kTipLength = 6.0;
//! half the size of the x of a rejected point
constexpr float kCrossSize = 2.0f;
//! positions are passed to OpenCV with this many fractional bits: the arrows follow sub pixel motion
constexpr int kShift = 3;

//! history window shown by the chart in both glass and detailed modes
constexpr double kChartSeconds = 10.0;

//! legacy detailed OSD constants
constexpr float kLegacyDim = 0.6f;
constexpr float kLegacyBoxDim = 0.25f;
constexpr int kLegacyFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr int kLegacyMargin = 8;
constexpr int kLegacyPadding = 10;
constexpr int kLegacyLines = 4;
constexpr int kLegacyChartLines = 5;
constexpr int kLegacyChartMinWidth = 200;
const cv::Vec3b kLegacyFramerateColor(90, 220, 90);
const cv::Vec3b kLegacyPerceivedColor(255, 170, 70);
const cv::Vec3b kLegacyGridColor(85, 85, 85);

//! the top of the chart: the next common framerate at or above the recorded one
double chart_top(double fps)
{
    for (const double top : { 30.0, 60.0, 120.0, 144.0, 240.0, 360.0, 480.0 })
    {
        if (fps <= top + 0.5) return top;
    }
    return 60.0 * std::ceil(fps / 60.0);
}

MotionVideoSegment chart_segment(float x0, float y0, float x1, float y1, float radius, const cv::Vec3b & color)
{
    MotionVideoSegment segment;
    segment.x0 = x0;
    segment.y0 = y0;
    segment.x1 = x1;
    segment.y1 = y1;
    segment.radius = radius;
    segment.blue = color[0];
    segment.green = color[1];
    segment.red = color[2];
    segment.above_box = true;
    return segment;
}

cv::Point fixed_point(double x, double y)
{
    return cv::Point(static_cast<int>(std::lround(x * (1 << kShift))), static_cast<int>(std::lround(y * (1 << kShift))));
}

std::string format(const char * pattern, double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), pattern, value);
    return buffer;
}

//! table for cv::LUT: 8-bit values times `factor`
cv::Mat scale_table(float factor)
{
    cv::Mat table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) table.at<uchar>(i) = cv::saturate_cast<uchar>(i * factor);
    return table;
}

inline void blend_bgr(float & cb, float & cg, float & cr, float sb, float sg, float sr, float alpha)
{
    const float inv = 1.0f - alpha;
    cb = cb * inv + sb * alpha;
    cg = cg * inv + sg * alpha;
    cr = cr * inv + sr * alpha;
}

//! exact perpendicular distance from (px, py) to the local union of segments around column floor(px - plot_x0)
float polyline_distance(float px, float py, float plot_x0, const std::vector<float> & ys, int cols, int search_r)
{
    const int center = static_cast<int>(std::floor(px - plot_x0));
    const int c_lo = std::max(0, center - search_r);
    const int c_hi = std::min(cols - 1, center + search_r);
    float min_d2 = 1e30f;
    for (int c = c_lo; c <= c_hi; ++c)
    {
        const float y0 = ys[static_cast<size_t>(c)];
        if (y0 < 0.0f) continue;
        const float x0 = plot_x0 + static_cast<float>(c);
        const float rx = px - x0;
        const float ry = py - y0;
        if (c + 1 < cols && ys[static_cast<size_t>(c + 1)] >= 0.0f)
        {
            const float dy = ys[static_cast<size_t>(c + 1)] - y0;
            const float t = std::clamp((rx + ry * dy) / (1.0f + dy * dy), 0.0f, 1.0f);
            const float ex = rx - t;
            const float ey = ry - t * dy;
            min_d2 = std::min(min_d2, ex * ex + ey * ey);
        }
        else
        {
            min_d2 = std::min(min_d2, rx * rx + ry * ry);
        }
    }
    return min_d2 < 1e29f ? std::sqrt(min_d2) : 1e15f;
}

//! draws the scene with OpenCV and encodes the pictures with a VideoWriter
class CpuMotionVideoTarget final : public MotionVideoTarget {
public:
    CpuMotionVideoTarget(const std::string & path, cv::Size size, double fps)
        : _size(size), _writer(make_video_writer(path, size.width, size.height, fps))
    { }

    std::string description() const override { return _writer->description() + ", drawn on the CPU"; }

    void write(const MotionDetail & detail, const MotionVideoScene & scene) override
    {
        if (detail.picture.empty()) throw std::runtime_error("motion video: the pipeline didn't keep the picture of the frame");
        cv::cvtColor(detail.picture, _picture, cv::COLOR_YUV2BGR_I420);
        // the size of the frames changed within the video
        if (_picture.size() != _size) cv::resize(_picture, _picture, _size, 0, 0, cv::INTER_AREA);
        if (scene.dim != 1.0f)
        {
            if (_dim != scene.dim)
            {
                _dim = scene.dim;
                _dim_table = scale_table(_dim);
            }
            cv::LUT(_picture, _dim_table, _picture);
        }

        auto draw_segments = [&](bool above_box) {
            for (const MotionVideoSegment & segment : scene.segments)
            {
                if (segment.above_box != above_box) continue;
                const cv::Scalar color(segment.blue, segment.green, segment.red);
                if (segment.x0 == segment.x1 && segment.y0 == segment.y1)
                {
                    cv::circle(_picture, fixed_point(segment.x0, segment.y0), static_cast<int>(std::lround(segment.radius * (1 << kShift))),
                               color, cv::FILLED, cv::LINE_AA, kShift);
                }
                else
                {
                    cv::line(_picture, fixed_point(segment.x0, segment.y0), fixed_point(segment.x1, segment.y1), color,
                             std::max(1, static_cast<int>(std::lround(2.0f * segment.radius))), cv::LINE_AA, kShift);
                }
            }
        };
        if (!scene.segments.empty()) draw_segments(false);

        if (scene.card.enabled)
        {
            _draw_card(scene);
        }
        else
        {
            const cv::Rect box = scene.box & cv::Rect(cv::Point(0, 0), _size);
            const bool has_box = !box.empty() && scene.text.size() == scene.box.size();
            if (has_box)
            {
                if (_box_dim != scene.box_dim)
                {
                    _box_dim = scene.box_dim;
                    _box_table = scale_table(_box_dim);
                }
                cv::Mat region = _picture(box);
                cv::LUT(region, _box_table, region);
            }
            if (!scene.segments.empty()) draw_segments(true);
            if (has_box)
            {
                cv::Mat region = _picture(box);
                // white text, by its coverage
                for (int y = 0; y < box.height; ++y)
                {
                    const uchar * coverage = scene.text.ptr<uchar>(y + box.y - scene.box.y) + (box.x - scene.box.x);
                    cv::Vec3b * pixels = region.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < box.width; ++x)
                    {
                        const int a = coverage[x];
                        if (a == 0) continue;
                        for (int c = 0; c < 3; ++c) pixels[x][c] = static_cast<uchar>(pixels[x][c] + (255 - pixels[x][c]) * a / 255);
                    }
                }
            }
        }
        _writer->write(_picture);
    }

    void finish() override { _writer->finish(); }

private:
    void _draw_card(const MotionVideoScene & scene)
    {
        const MotionVideoCard & card = scene.card;
        const cv::Rect frame_rect(0, 0, _size.width, _size.height);
        const cv::Rect card_clipped = card.rect & frame_rect;
        if (card_clipped.empty()) return;

        // 1. ROI-only downsampled Gaussian blur matching GPU overlay_blur.comp
        const float sigma = std::max(0.5f, card.blur_sigma);
        const int ds = (sigma < 10.0f) ? 2 : (sigma <= 28.0f ? 4 : 8);
        const float sigma_ds = std::sqrt(std::max(0.25f, sigma * sigma - static_cast<float>(ds * ds - 1) / 12.0f)) / static_cast<float>(ds);
        const int pad = (static_cast<int>(std::ceil(3.0f * sigma)) + 1) & ~1;
        const int roi_x0 = std::max(0, (card.rect.x - pad) & ~1);
        const int roi_y0 = std::max(0, (card.rect.y - pad) & ~1);
        const int roi_x1 = std::min(_size.width, (card.rect.x + card.rect.width + pad + 1) & ~1);
        const int roi_y1 = std::min(_size.height, (card.rect.y + card.rect.height + pad + 1) & ~1);
        const cv::Rect blur_roi = cv::Rect(roi_x0, roi_y0, std::max(1, roi_x1 - roi_x0), std::max(1, roi_y1 - roi_y0)) & frame_rect;
        const int blur_w = std::max(1, (blur_roi.width + ds - 1) / ds);
        const int blur_h = std::max(1, (blur_roi.height + ds - 1) / ds);

        cv::resize(_picture(blur_roi), _blur_small, cv::Size(blur_w, blur_h), 0, 0, cv::INTER_AREA);
        _blur_small.convertTo(_blur_small_f, CV_32FC3);
        cv::GaussianBlur(_blur_small_f, _blur_tmp_f, cv::Size(0, 0), sigma_ds, sigma_ds, cv::BORDER_REPLICATE);
        cv::resize(_blur_tmp_f, _blur_full_f, blur_roi.size(), 0, 0, cv::INTER_LINEAR);

        // 2. Expanded card + drop shadow bounding box
        const float shadow_pad = std::max(0.0f, card.shadow_pad);
        const int sh_pad_x = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(shadow_pad)) + 2 : 2;
        const int sh_pad_top = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(std::max(0.0f, shadow_pad - card.shadow_offset_y))) + 2 : 2;
        const int sh_pad_bot = card.shadow_alpha > 0.0f ? static_cast<int>(std::ceil(shadow_pad + std::max(0.0f, card.shadow_offset_y))) + 2 : 2;
        const int bx0 = std::max(0, card.rect.x - sh_pad_x);
        const int by0 = std::max(0, card.rect.y - sh_pad_top);
        const int bx1 = std::min(_size.width, card.rect.x + card.rect.width + sh_pad_x);
        const int by1 = std::min(_size.height, card.rect.y + card.rect.height + sh_pad_bot);

        const float cx = static_cast<float>(card.rect.x) + 0.5f * static_cast<float>(card.rect.width);
        const float cy = static_cast<float>(card.rect.y) + 0.5f * static_cast<float>(card.rect.height);
        const float hx = 0.5f * static_cast<float>(card.rect.width);
        const float hy = 0.5f * static_cast<float>(card.rect.height);
        const float R = std::min(card.corner_radius, std::min(hx, hy));
        const float qhx = hx - R;
        const float qhy = hy - R;
        const float inv_card_h = 1.0f / std::max(1.0f, static_cast<float>(card.rect.height));
        const float inv_sheen_h = card.sheen_height > 1e-5f ? 1.0f / card.sheen_height : 0.0f;
        const float inv_shadow_decay = card.shadow_decay > 1e-5f ? 1.0f / card.shadow_decay : 0.0f;

        const float tint_top_b = static_cast<float>(card.tint_top.b);
        const float tint_top_g = static_cast<float>(card.tint_top.g);
        const float tint_top_r = static_cast<float>(card.tint_top.r);
        const float tint_top_a = card.tint_top.alpha;
        const float tint_bot_b = static_cast<float>(card.tint_bottom.b);
        const float tint_bot_g = static_cast<float>(card.tint_bottom.g);
        const float tint_bot_r = static_cast<float>(card.tint_bottom.r);
        const float tint_bot_a = card.tint_bottom.alpha;

        const float prim_b = static_cast<float>(card.primary_color.b);
        const float prim_g = static_cast<float>(card.primary_color.g);
        const float prim_r = static_cast<float>(card.primary_color.r);
        const float prim_a = card.primary_color.alpha;

        const float sec_b = static_cast<float>(card.secondary_color.b);
        const float sec_g = static_cast<float>(card.secondary_color.g);
        const float sec_r = static_cast<float>(card.secondary_color.r);
        const float sec_a = card.secondary_color.alpha;

        const bool has_overlay = !scene.overlay.empty() && scene.overlay.size() == card.rect.size() && scene.overlay.type() == CV_8UC4;

        const int plot_cols = static_cast<int>(card.primary_y.size());
        const bool has_chart = card.has_chart && plot_cols >= 2 && static_cast<int>(card.secondary_y.size()) == plot_cols;
        const float plot_x0 = card.plot.x;
        const float plot_y0 = card.plot.y;
        const float plot_x1 = card.plot.x + card.plot.width;
        const float plot_y1 = card.plot.y + card.plot.height;
        const int search_r_prim = static_cast<int>(std::ceil(card.primary_half_width + 1.5f));
        const int search_r_sec = static_cast<int>(std::ceil(card.secondary_half_width + 1.5f));
        const float prim_reach = card.primary_half_width + 0.5f;
        const float sec_reach = card.secondary_half_width + 0.5f;
        const float max_curve_pad = std::max(card.primary_half_width, card.secondary_half_width) + 2.0f;

        // Precompute per-column local Y bounds so polyline_distance is skipped on pixels far from the curves
        if (has_chart)
        {
            auto build_bounds = [&](const std::vector<float> & ys, int sr, std::vector<float> & lo, std::vector<float> & hi) {
                lo.resize(static_cast<size_t>(plot_cols));
                hi.resize(static_cast<size_t>(plot_cols));
                for (int c = 0; c < plot_cols; ++c)
                {
                    float y_lo = 1e30f;
                    float y_hi = -1e30f;
                    const int a = std::max(0, c - sr);
                    const int b = std::min(plot_cols - 1, c + sr + 1);
                    for (int k = a; k <= b; ++k)
                    {
                        const float v = ys[static_cast<size_t>(k)];
                        if (v >= 0.0f)
                        {
                            y_lo = std::min(y_lo, v);
                            y_hi = std::max(y_hi, v);
                        }
                    }
                    lo[static_cast<size_t>(c)] = y_lo;
                    hi[static_cast<size_t>(c)] = y_hi;
                }
            };
            build_bounds(card.primary_y, search_r_prim, _prim_min_y, _prim_max_y);
            build_bounds(card.secondary_y, search_r_sec, _sec_min_y, _sec_max_y);
        }

        const bool has_prim_dot = has_chart && card.primary_dot_x >= 0.0f && card.primary_dot_y >= 0.0f;
        const bool has_sec_dot = has_chart && card.secondary_dot_x >= 0.0f && card.secondary_dot_y >= 0.0f;
        const float prim_dot_max_r = std::max(card.primary_glow_radius, card.primary_dot_radius + 1.0f);
        const float sec_dot_max_r = std::max(card.secondary_glow_radius, card.secondary_dot_radius + 1.0f);
        const int grid_count = std::clamp(card.grid_count, 0, 4);

        for (int iy = by0; iy < by1; ++iy)
        {
            const float py = static_cast<float>(iy) + 0.5f;
            const float qy = std::abs(py - cy) - qhy;
            const float ay = std::max(qy, 0.0f);
            const float qy_sh = std::abs(py - (cy + card.shadow_offset_y)) - qhy;
            const float ay_sh = std::max(qy_sh, 0.0f);

            const float v = std::clamp((py - static_cast<float>(card.rect.y)) * inv_card_h, 0.0f, 1.0f);
            const float tint_a = tint_top_a + (tint_bot_a - tint_top_a) * v;
            const float tint_b = tint_top_b + (tint_bot_b - tint_top_b) * v;
            const float tint_g = tint_top_g + (tint_bot_g - tint_top_g) * v;
            const float tint_r = tint_top_r + (tint_bot_r - tint_top_r) * v;
            const float sheen_t = std::clamp(1.0f - v * inv_sheen_h, 0.0f, 1.0f);
            const float sheen = card.sheen_alpha * sheen_t * sheen_t;
            const float border_a = card.border_alpha_bottom + (card.border_alpha_top - card.border_alpha_bottom) * std::pow(1.0f - v, card.border_power);

            float row_grid_a = 0.0f;
            if (has_chart)
            {
                for (int g = 0; g < grid_count; ++g)
                {
                    const float d = std::abs(py - card.grid_y[g]);
                    const float cov = std::clamp(card.grid_half_width + 0.5f - d, 0.0f, 1.0f);
                    row_grid_a = std::max(row_grid_a, cov * card.grid_alpha[g]);
                }
            }

            const bool in_plot_y_grid = has_chart && row_grid_a > 0.0f;
            const bool in_plot_y_fill = has_chart && py >= plot_y0 - 1.0f && py <= plot_y1 + 1.0f;
            const bool in_plot_y_curve = has_chart && py >= plot_y0 - max_curve_pad && py <= plot_y1 + max_curve_pad;
            const bool in_prim_dot_y = has_prim_dot && std::abs(py - card.primary_dot_y) <= prim_dot_max_r;
            const bool in_sec_dot_y = has_sec_dot && std::abs(py - card.secondary_dot_y) <= sec_dot_max_r;

            const int by_blur = std::clamp(iy - blur_roi.y, 0, blur_roi.height - 1);
            const cv::Vec3f * blur_row = _blur_full_f.ptr<cv::Vec3f>(by_blur);
            const int oy = iy - card.rect.y;
            const cv::Vec4b * overlay_row = (has_overlay && oy >= 0 && oy < card.rect.height) ? scene.overlay.ptr<cv::Vec4b>(oy) : nullptr;
            cv::Vec3b * pic_row = _picture.ptr<cv::Vec3b>(iy);

            for (int ix = bx0; ix < bx1; ++ix)
            {
                const float px = static_cast<float>(ix) + 0.5f;
                const float qx = std::abs(px - cx) - qhx;
                const float ax = std::max(qx, 0.0f);
                const float outer = (ax > 0.0f || ay > 0.0f) ? std::sqrt(ax * ax + ay * ay) : 0.0f;
                const float sdf = outer + std::min(std::max(qx, qy), 0.0f) - R;
                const float inside = std::clamp(0.5f - sdf, 0.0f, 1.0f);

                float shadow_cov = 0.0f;
                if (inside < 1.0f && card.shadow_alpha > 0.0f && shadow_pad > 0.0f && inv_shadow_decay > 0.0f)
                {
                    const float outer_sh = (ax > 0.0f || ay_sh > 0.0f) ? std::sqrt(ax * ax + ay_sh * ay_sh) : 0.0f;
                    const float sdf_sh = outer_sh + std::min(std::max(qx, qy_sh), 0.0f) - R;
                    if (sdf_sh < shadow_pad)
                    {
                        const float d_sh = std::max(sdf_sh, 0.0f);
                        const float edge_fade = std::clamp((shadow_pad - d_sh) / std::max(0.25f * shadow_pad, 1.0f), 0.0f, 1.0f);
                        shadow_cov = card.shadow_alpha * std::exp(-d_sh * inv_shadow_decay) * edge_fade;
                    }
                }
                if (inside <= 0.0f)
                {
                    if (shadow_cov > 0.0f)
                    {
                        const float keep = 1.0f - shadow_cov;
                        cv::Vec3b & dst_px = pic_row[ix];
                        dst_px[0] = static_cast<uchar>(dst_px[0] * keep + 0.5f);
                        dst_px[1] = static_cast<uchar>(dst_px[1] * keep + 0.5f);
                        dst_px[2] = static_cast<uchar>(dst_px[2] * keep + 0.5f);
                    }
                    continue;
                }

                // Blurred backdrop + vibrancy lift
                const int bx_blur = std::clamp(ix - blur_roi.x, 0, blur_roi.width - 1);
                const cv::Vec3f & bg = blur_row[bx_blur];
                const float luma = 0.114f * bg[0] + 0.587f * bg[1] + 0.299f * bg[2];
                float cb = std::clamp(luma + (bg[0] - luma) * card.vibrancy, 0.0f, 255.0f);
                float cg = std::clamp(luma + (bg[1] - luma) * card.vibrancy, 0.0f, 255.0f);
                float cr = std::clamp(luma + (bg[2] - luma) * card.vibrancy, 0.0f, 255.0f);

                // Vertical gradient frosted tint
                blend_bgr(cb, cg, cr, tint_b, tint_g, tint_r, tint_a);

                // Top ambient sheen
                if (sheen > 0.0f) blend_bgr(cb, cg, cr, 255.0f, 255.0f, 255.0f, sheen);

                // Directional specular inner hairline rim
                const float border_cov = inside * std::clamp(sdf + card.border_width + 0.5f, 0.0f, 1.0f);
                if (border_cov > 0.0f) blend_bgr(cb, cg, cr, 255.0f, 255.0f, 255.0f, border_cov * border_a);

                // Chart: gridlines, primary vertical area fill, primary & secondary polyline SDFs
                if (has_chart)
                {
                    if (in_plot_y_grid && px >= plot_x0 - 0.5f && px <= plot_x1 + 0.5f)
                    {
                        const float x_cov = std::clamp(std::min(px - plot_x0, plot_x1 - px) + 0.5f, 0.0f, 1.0f);
                        if (x_cov > 0.0f) blend_bgr(cb, cg, cr, 255.0f, 255.0f, 255.0f, row_grid_a * x_cov);
                    }

                    const float fx = px - plot_x0;
                    if (in_plot_y_fill && fx >= 0.0f && fx <= static_cast<float>(plot_cols - 1))
                    {
                        const int c0 = std::min(static_cast<int>(std::floor(fx)), plot_cols - 2);
                        const float y0_s = card.primary_y[static_cast<size_t>(c0)];
                        const float y1_s = card.primary_y[static_cast<size_t>(c0 + 1)];
                        if (y0_s >= 0.0f && y1_s >= 0.0f)
                        {
                            const float frac = fx - static_cast<float>(c0);
                            const float y_fps = y0_s + (y1_s - y0_s) * frac;
                            const float top_cov = std::clamp(py - y_fps + 0.5f, 0.0f, 1.0f);
                            const float bot_cov = std::clamp(plot_y1 - py + 0.5f, 0.0f, 1.0f);
                            if (top_cov > 0.0f && bot_cov > 0.0f)
                            {
                                const float h_norm = std::clamp((plot_y1 - py) / std::max(plot_y1 - y_fps, 1.0f), 0.0f, 1.0f);
                                const float fill_a = card.primary_fill_alpha * std::pow(h_norm, card.primary_fill_power) * top_cov * bot_cov;
                                blend_bgr(cb, cg, cr, prim_b, prim_g, prim_r, fill_a);
                            }
                        }
                    }

                    if (in_plot_y_curve && px >= plot_x0 - max_curve_pad && px <= plot_x1 + max_curve_pad)
                    {
                        const int center = static_cast<int>(std::floor(fx));
                        const bool check_prim = (center < 0 || center >= plot_cols)
                            || (py >= _prim_min_y[static_cast<size_t>(center)] - prim_reach && py <= _prim_max_y[static_cast<size_t>(center)] + prim_reach);
                        if (check_prim)
                        {
                            const float d_prim = polyline_distance(px, py, plot_x0, card.primary_y, plot_cols, search_r_prim);
                            const float cov_prim = std::clamp(prim_reach - d_prim, 0.0f, 1.0f);
                            if (cov_prim > 0.0f) blend_bgr(cb, cg, cr, prim_b, prim_g, prim_r, cov_prim * prim_a);
                        }

                        const bool check_sec = (center < 0 || center >= plot_cols)
                            || (py >= _sec_min_y[static_cast<size_t>(center)] - sec_reach && py <= _sec_max_y[static_cast<size_t>(center)] + sec_reach);
                        if (check_sec)
                        {
                            const float d_sec = polyline_distance(px, py, plot_x0, card.secondary_y, plot_cols, search_r_sec);
                            const float cov_sec = std::clamp(sec_reach - d_sec, 0.0f, 1.0f);
                            if (cov_sec > 0.0f) blend_bgr(cb, cg, cr, sec_b, sec_g, sec_r, cov_sec * sec_a);
                        }
                    }
                }

                // Live endpoint dots (unclipped inside card)
                if (in_prim_dot_y)
                {
                    const float dx = px - card.primary_dot_x;
                    if (std::abs(dx) <= prim_dot_max_r)
                    {
                        const float dy = py - card.primary_dot_y;
                        const float dist = std::sqrt(dx * dx + dy * dy);
                        if (card.primary_glow_radius > 0.0f && dist < card.primary_glow_radius)
                        {
                            const float gt = 1.0f - dist / card.primary_glow_radius;
                            blend_bgr(cb, cg, cr, prim_b, prim_g, prim_r, card.primary_glow_alpha * gt * gt);
                        }
                        const float core_cov = std::clamp(card.primary_dot_radius + 0.5f - dist, 0.0f, 1.0f);
                        if (core_cov > 0.0f) blend_bgr(cb, cg, cr, prim_b, prim_g, prim_r, core_cov * prim_a);
                    }
                }
                if (in_sec_dot_y)
                {
                    const float dx = px - card.secondary_dot_x;
                    if (std::abs(dx) <= sec_dot_max_r)
                    {
                        const float dy = py - card.secondary_dot_y;
                        const float dist = std::sqrt(dx * dx + dy * dy);
                        if (card.secondary_glow_radius > 0.0f && dist < card.secondary_glow_radius)
                        {
                            const float gt = 1.0f - dist / card.secondary_glow_radius;
                            blend_bgr(cb, cg, cr, sec_b, sec_g, sec_r, card.secondary_glow_alpha * gt * gt);
                        }
                        const float core_cov = std::clamp(card.secondary_dot_radius + 0.5f - dist, 0.0f, 1.0f);
                        if (core_cov > 0.0f) blend_bgr(cb, cg, cr, sec_b, sec_g, sec_r, core_cov * sec_a);
                    }
                }

                // Premultiplied RGBA8 typography overlay (byte order R, G, B, A -> BGR)
                if (overlay_row)
                {
                    const int ox = ix - card.rect.x;
                    if (ox >= 0 && ox < card.rect.width)
                    {
                        const cv::Vec4b ov = overlay_row[ox];
                        if (ov[3] != 0)
                        {
                            const float inv_a = 1.0f - static_cast<float>(ov[3]) * (1.0f / 255.0f);
                            cb = cb * inv_a + static_cast<float>(ov[2]);
                            cg = cg * inv_a + static_cast<float>(ov[1]);
                            cr = cr * inv_a + static_cast<float>(ov[0]);
                        }
                    }
                }

                cv::Vec3b & dst_px = pic_row[ix];
                if (inside >= 1.0f)
                {
                    dst_px[0] = static_cast<uchar>(std::clamp(cb + 0.5f, 0.0f, 255.0f));
                    dst_px[1] = static_cast<uchar>(std::clamp(cg + 0.5f, 0.0f, 255.0f));
                    dst_px[2] = static_cast<uchar>(std::clamp(cr + 0.5f, 0.0f, 255.0f));
                }
                else
                {
                    const float keep = 1.0f - shadow_cov;
                    const float inv_in = 1.0f - inside;
                    const float out_b = static_cast<float>(dst_px[0]) * keep * inv_in + cb * inside;
                    const float out_g = static_cast<float>(dst_px[1]) * keep * inv_in + cg * inside;
                    const float out_r = static_cast<float>(dst_px[2]) * keep * inv_in + cr * inside;
                    dst_px[0] = static_cast<uchar>(std::clamp(out_b + 0.5f, 0.0f, 255.0f));
                    dst_px[1] = static_cast<uchar>(std::clamp(out_g + 0.5f, 0.0f, 255.0f));
                    dst_px[2] = static_cast<uchar>(std::clamp(out_r + 0.5f, 0.0f, 255.0f));
                }
            }
        }
    }

    cv::Size _size;
    std::unique_ptr<VideoWriter> _writer;
    cv::Mat _picture;
    float _dim = -1.0f;
    float _box_dim = -1.0f;
    cv::Mat _dim_table;
    cv::Mat _box_table;

    // Reusable buffers for frosted-glass card blur & chart bounds
    cv::Mat _blur_small;
    cv::Mat _blur_small_f;
    cv::Mat _blur_tmp_f;
    cv::Mat _blur_full_f;
    std::vector<float> _prim_min_y;
    std::vector<float> _prim_max_y;
    std::vector<float> _sec_min_y;
    std::vector<float> _sec_max_y;
};

} // namespace

std::unique_ptr<MotionVideoTarget> make_cpu_motion_video_target(const std::string & path, cv::Size size, double fps)
{
    return std::make_unique<CpuMotionVideoTarget>(path, size, fps);
}

MotionVideo::MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale,
                         std::unique_ptr<MotionVideoTarget> target, bool native,
                         MotionOsdStyle osd_style, bool show_arrows)
    : _path(path)
    , _size(video_picture_size(frame_width, frame_height, native))
    , _measured(motion_picture_size(frame_width, frame_height))
    , _k(static_cast<float>(motion_scale(frame_width, frame_height) / video_picture_scale(frame_width, frame_height, native)))
    , _arrow_scale(arrow_scale)
    , _osd_style(osd_style)
    , _show_arrows(osd_style == MotionOsdStyle::Detailed || show_arrows)
    , _target(std::move(target))
{
    if (_size.width < 2 || _size.height < 2) throw std::runtime_error("the video is too small for a motion video");
    if (!_target) _target = make_cpu_motion_video_target(path, _size, fps);

    if (_show_arrows)
    {
        cv::Mat ramp(256, 1, CV_8UC1), colors;
        for (int i = 0; i < 256; ++i) ramp.at<uchar>(i) = static_cast<uchar>(i);
        cv::applyColorMap(ramp, colors, cv::COLORMAP_TURBO);
        for (int i = 0; i < 256; ++i) _colors.push_back(colors.at<cv::Vec3b>(i));
    }

    if (_osd_style == MotionOsdStyle::Detailed) _init_detailed(fps);
    else _init_glass(fps);
}

void MotionVideo::_init_glass(double fps)
{
    _scene.dim = _show_arrows ? kArrowsDim : 1.0f;

    // Small-frame safeguard: omit OSD card below 360x240
    if (_size.width < 360 || _size.height < 240) return;

    const int W = _size.width;
    const int H = _size.height;
    const float s = (W >= H) ? (static_cast<float>(H) / 1080.0f)
                             : std::min(static_cast<float>(H) / 1080.0f, static_cast<float>(W) / 1200.0f);

    int H_card = static_cast<int>(std::lround(264.0f * s)) & ~1;
    int M_bottom = static_cast<int>(std::lround(36.0f * s)) & ~1;
    const int M_side = std::max(static_cast<int>(std::lround(36.0f * s)), static_cast<int>(std::lround(0.04f * static_cast<float>(W))));
    const int W_card = std::min(static_cast<int>(std::lround(1120.0f * s)), W - 2 * M_side) & ~1;
    if (W_card < 2 || H_card < 2) return;

    const int x0 = ((W - W_card) / 2) & ~1;
    int y0 = (H - M_bottom - H_card) & ~1;
    const int min_y0 = static_cast<int>(std::floor(0.70 * static_cast<double>(H)));
    if (y0 < min_y0)
    {
        const int target_y0 = (min_y0 + 1) & ~1;
        const int avail = H - target_y0;
        if (avail < 2) return;
        M_bottom = std::min(M_bottom, std::max(0, (avail - H_card) & ~1));
        H_card = std::min(H_card, (avail - M_bottom) & ~1);
        if (H_card < 2) return;
        y0 = (H - M_bottom - H_card) & ~1;
    }

    MotionVideoCard & card = _scene.card;
    card.enabled = true;
    card.rect = cv::Rect(x0, y0, W_card, H_card);
    card.corner_radius = 24.0f * s;
    card.blur_sigma = 22.0f * s;
    card.vibrancy = 1.12f;
    card.tint_top = MotionVideoColor{ 20, 23, 30, 0.58f };
    card.tint_bottom = MotionVideoColor{ 10, 12, 18, 0.70f };
    card.sheen_alpha = 0.045f;
    card.sheen_height = 0.38f;
    card.border_width = 1.25f * s;
    card.border_alpha_top = 0.28f;
    card.border_alpha_bottom = 0.08f;
    card.border_power = 1.4f;
    card.shadow_offset_y = 6.0f * s;
    card.shadow_alpha = 0.28f;
    card.shadow_decay = 14.0f * s;
    card.shadow_pad = 36.0f * s;

    _font_number = OsdFontFace(OsdFontWeight::Bold, 58.5f * s);
    _font_label = OsdFontFace(OsdFontWeight::NotoSansScMedium, 18.0f * s);
    const OsdFontFace font_note(OsdFontWeight::NotoSansScMedium, 13.0f * s);
    _font_tick = OsdFontFace(OsdFontWeight::Medium, 15.3f * s);

    // Card-local typography geometry
    const float pad_x = 32.0f * s;
    const float local_left = pad_x;
    const float local_right = static_cast<float>(W_card) - pad_x;
    _num_baseline_y = 66.0f * s;
    const float label_baseline_y = 92.0f * s;

    const float swatch_w = 14.0f * s;
    const float swatch_h = 4.0f * s;
    const float swatch_r = 2.0f * s;
    const float swatch_gap = 8.0f * s;
    const float label_mid = _font_label.valid() ? _font_label.visual_center_above_baseline() : 7.0f * s;
    const float swatch_y = label_baseline_y - label_mid - 0.5f * swatch_h;

    const std::string label_fps = "\xe5\xb8\xa7\xe7\x8e\x87"; // 帧率
    const std::string label_perc = "\xe4\xbd\x93\xe6\x84\x9f\xe5\xb8\xa7\xe7\x8e\x87"; // 体感帧率
    const std::string label_note = "\xe5\x9f\xba\xe4\xba\x8e Animation Error"; // 基于 Animation Error

    _block1_x = local_left;
    const float block1_label_x = _block1_x + swatch_w + swatch_gap;
    const float block1_label_w = _font_label.valid() ? _font_label.measure_text(label_fps) : 36.0f * s;
    if (static_cast<float>(W_card) >= 560.0f * s)
    {
        _block2_x = local_left + 228.0f * s;
    }
    else
    {
        const float min_block2_x = block1_label_x + block1_label_w + 18.0f * s;
        const float prop_block2_x = local_left + 228.0f * s * (static_cast<float>(W_card) / (560.0f * s));
        _block2_x = std::min(local_left + 228.0f * s, std::max(min_block2_x, prop_block2_x));
    }

    _static_overlay = cv::Mat::zeros(card.rect.size(), CV_8UC4);

    const OsdColor primary_swatch_col{ 245, 248, 252, 0.96f };
    const OsdColor secondary_swatch_col{ 82, 182, 255, 0.96f };
    const OsdColor label_col{ 235, 240, 248, 0.76f };
    const OsdColor note_col{ 235, 240, 248, 0.46f };
    const OsdColor range_col{ 235, 240, 248, 0.40f };
    const OsdColor tick_col{ 235, 240, 248, 0.45f };

    draw_rounded_rect_rgba(_static_overlay, _block1_x, swatch_y, swatch_w, swatch_h, swatch_r, primary_swatch_col);
    _font_label.draw_text(_static_overlay, block1_label_x, label_baseline_y, label_fps, label_col);

    draw_rounded_rect_rgba(_static_overlay, _block2_x, swatch_y, swatch_w, swatch_h, swatch_r, secondary_swatch_col);
    const float block2_label_end = _font_label.draw_text(_static_overlay, _block2_x + swatch_w + swatch_gap, label_baseline_y, label_perc, label_col);

    // Chart geometry (frame coordinates)
    const float plot_x0 = static_cast<float>(x0) + local_left;
    const float plot_x1 = static_cast<float>(x0) + local_right - 42.0f * s;
    const float plot_y0 = static_cast<float>(y0) + 118.0f * s;
    const float plot_y1 = static_cast<float>(y0) + 240.0f * s;
    const float plot_w = plot_x1 - plot_x0;
    const float plot_h = plot_y1 - plot_y0;
    const bool show_chart = (plot_w >= 140.0f * s && plot_h >= 20.0f * s && plot_w >= 16.0f);

    float range_limit = local_right;
    if (show_chart)
    {
        _chart = true;
        _chart_top = chart_top(fps);
        _chart_frames = static_cast<size_t>(std::max(2.0, std::round(kChartSeconds * fps)));
        _bucket_frames = std::max<long long>(1, std::llround(0.07 * fps));
        _plot_cols = std::max(2, static_cast<int>(std::lround(plot_w)));

        card.has_chart = true;
        card.plot = cv::Rect2f(plot_x0, plot_y0, plot_w, plot_h);
        card.grid_count = 3;
        card.grid_y[0] = plot_y0;
        card.grid_alpha[0] = 0.08f;
        card.grid_y[1] = plot_y0 + 0.5f * plot_h;
        card.grid_alpha[1] = 0.08f;
        card.grid_y[2] = plot_y1;
        card.grid_alpha[2] = 0.13f;
        card.grid_half_width = std::max(0.25f, 0.5f * s);

        card.primary_color = MotionVideoColor{ 245, 248, 252, 0.96f };
        card.primary_half_width = 1.55f * s;
        card.primary_fill_alpha = 0.14f;
        card.primary_fill_power = 1.45f;

        card.secondary_color = MotionVideoColor{ 82, 182, 255, 0.92f };
        card.secondary_half_width = 1.10f * s;

        card.primary_dot_x = -1.0f;
        card.primary_dot_y = -1.0f;
        card.primary_dot_radius = 3.0f * s;
        card.primary_glow_radius = 8.0f * s;
        card.primary_glow_alpha = 0.28f;

        card.secondary_dot_x = -1.0f;
        card.secondary_dot_y = -1.0f;
        card.secondary_dot_radius = 2.4f * s;
        card.secondary_glow_radius = 6.5f * s;
        card.secondary_glow_alpha = 0.28f;

        card.primary_y.assign(static_cast<size_t>(_plot_cols), -1.0f);
        card.secondary_y.assign(static_cast<size_t>(_plot_cols), -1.0f);

        const std::string range_tag = format("%gs", kChartSeconds);
        const float range_w = _font_tick.valid() ? _font_tick.measure_text(range_tag) : 22.0f * s;
        if (block2_label_end + 12.0f * s < local_right - range_w)
        {
            _font_tick.draw_text_right(_static_overlay, local_right, label_baseline_y, range_tag, range_col);
            range_limit = local_right - range_w - 16.0f * s;
        }

        const float tick_cap = _font_tick.valid() ? _font_tick.cap_height() : 11.0f * s;
        const double tick_vals[3] = { _chart_top, _chart_top * 0.5, 0.0 };
        for (int i = 0; i < 3; ++i)
        {
            const float gy_local = card.grid_y[i] - static_cast<float>(y0);
            const float baseline = gy_local + 0.5f * tick_cap;
            _font_tick.draw_text_right(_static_overlay, local_right, baseline, format("%g", tick_vals[i]), tick_col);
        }
    }

    if (font_note.valid())
    {
        const float note_x = block2_label_end + 10.0f * s;
        const float note_w = font_note.measure_text(label_note);
        if (note_x + note_w <= range_limit)
        {
            font_note.draw_text(_static_overlay, note_x, label_baseline_y, label_note, note_col);
        }
    }

    const float num_desc = _font_number.valid() ? std::abs(_font_number.descent()) : 14.0f * s;
    const int band_h = std::min(H_card, static_cast<int>(std::ceil(_num_baseline_y + num_desc)) + 4);
    _num_band = cv::Rect(0, 0, W_card, std::max(1, band_h));
    _scene.overlay = _static_overlay.clone();
    _scene.overlay_generation = 0;
}

void MotionVideo::_init_detailed(double fps)
{
    // everything is laid out as at the motion's resolution and drawn _k times larger
    _margin = static_cast<int>(std::lround(kLegacyMargin * _k));
    _padding = static_cast<int>(std::lround(kLegacyPadding * _k));
    _thickness = std::max(1, static_cast<int>(std::lround(_k)));

    // the text: smaller for small pictures, every digit gets the width of the widest one
    _font_scale = std::clamp(0.6 * _measured.width / 1280.0, 0.4, 0.6) * _k;
    int baseline = 0;
    for (char digit = '0'; digit <= '9'; ++digit)
    {
        _digit_width = std::max(_digit_width, cv::getTextSize(std::string(1, digit), kLegacyFont, _font_scale, _thickness, &baseline).width);
    }
    _space = _digit_width;
    _line_height = static_cast<int>(std::lround(42.0 * _font_scale));
    for (const char * state : { "new", "duplicate", "torn", "first frame" })
    {
        _state_width = std::max(_state_width, cv::getTextSize(state, kLegacyFont, _font_scale, _thickness, &baseline).width);
    }
    // the box holds the longest text there can be
    MotionVideoFrame widest;
    widest.compared = true;
    widest.sample.measurable = true;
    widest.sample.has_error = true;
    widest.sample.has_perceived_fps = true;
    widest.sample.lost = 1;
    const cv::Size box(_legacy_text(nullptr, widest, 0, 0, true) + _padding, kLegacyLines * _line_height + _padding);
    _scene.dim = kLegacyDim;
    _scene.box_dim = kLegacyBoxDim;
    _scene.card.enabled = false;
    auto fits = [&](const cv::Size & size) { return size.width <= _size.width - 2 * _margin && size.height <= _size.height - 2 * _margin; };
    // no text on pictures it doesn't fit on
    if (!fits(box)) return;
    _scene.box = cv::Rect(cv::Point(_margin, _margin), box);

    // the chart: a line for the legend, the plot with the values at its left, room for the label at its bottom
    const int label_width = cv::getTextSize("480", kLegacyFont, _font_scale, _thickness, &baseline).width + _space;
    const int digit_height = cv::getTextSize("0", kLegacyFont, _font_scale, _thickness, &baseline).height;
    const cv::Size with_chart(box.width, box.height + (kLegacyChartLines + 1) * _line_height + _padding);
    const int plot_width = box.width - 2 * _padding - label_width;
    if (fits(with_chart) && plot_width >= kLegacyChartMinWidth * _k)
    {
        _chart = true;
        _scene.box.height = with_chart.height;
        _chart_top = chart_top(fps);
        _chart_frames = static_cast<size_t>(std::max(2.0, std::round(kChartSeconds * fps)));
        const float x0 = static_cast<float>(_margin + _padding + label_width);
        const float y0 = static_cast<float>(_margin + box.height + _line_height + digit_height / 2);
        _plot = cv::Rect2f(x0, y0, static_cast<float>(plot_width), static_cast<float>(kLegacyChartLines * _line_height - digit_height));
        _chart_text = cv::Mat::zeros(_scene.box.size(), CV_8UC1);

        const float thin = (_measured.width >= 1280 ? 0.5f : 0.4f) * _k;
        const float thick = (_measured.width >= 1280 ? 1.0f : 0.75f) * _k;
        // the grid: 0, 25, 50, 75, 100% of the top, the values at 0, 50 and 100%
        for (int k = 0; k <= 4; ++k)
        {
            const float y = _plot.y + _plot.height * (1.0f - 0.25f * static_cast<float>(k));
            _chart_segments.push_back(chart_segment(_plot.x, y, _plot.x + _plot.width, y, thin, kLegacyGridColor));
            if (k % 2 != 0) continue;
            const std::string value = format("%g", _chart_top * 0.25 * k);
            const int width = cv::getTextSize(value, kLegacyFont, _font_scale, _thickness, &baseline).width;
            const cv::Point at(static_cast<int>(_plot.x) - _space / 2 - width - _margin, static_cast<int>(std::lround(y)) + digit_height / 2 - _margin);
            cv::putText(_chart_text, value, at, kLegacyFont, _font_scale, cv::Scalar(255), _thickness, cv::LINE_AA);
        }
        // the legend: framerate --- perceived smoothness ---   last 10 s
        const int line = kLegacyLines;
        const float sample_y = static_cast<float>(_margin + _padding / 2 + _line_height * (line + 1) - _line_height / 4 - digit_height / 2);
        const float sample_length = 3.0f * static_cast<float>(_digit_width);
        int x = _legacy_label(&_chart_text, _padding, line, "framerate") + _space;
        _chart_segments.push_back(chart_segment(static_cast<float>(_margin + x), sample_y, static_cast<float>(_margin + x) + sample_length, sample_y, thick, kLegacyFramerateColor));
        x += static_cast<int>(sample_length) + 2 * _space;
        x = _legacy_label(&_chart_text, x, line, "perceived smoothness") + _space;
        _chart_segments.push_back(chart_segment(static_cast<float>(_margin + x), sample_y, static_cast<float>(_margin + x) + sample_length, sample_y, thick, kLegacyPerceivedColor));
        x += static_cast<int>(sample_length) + 2 * _space;
        _legacy_label(&_chart_text, x, line, "fps, last " + format("%g", kChartSeconds) + " s");
    }
    _scene.text.create(_scene.box.size(), CV_8UC1);
}

int MotionVideo::_legacy_label(cv::Mat * target, int x, int line, const std::string & text) const
{
    int baseline = 0;
    const int width = cv::getTextSize(text, kLegacyFont, _font_scale, _thickness, &baseline).width;
    if (target)
    {
        const int y = _padding / 2 + _line_height * (line + 1) - _line_height / 4;
        cv::putText(*target, text, cv::Point(x, y), kLegacyFont, _font_scale, cv::Scalar(255), _thickness, cv::LINE_AA);
    }
    return x + width;
}

int MotionVideo::_legacy_number(cv::Mat * target, int x, int line, int cells, const std::string & text) const
{
    const int end = x + cells * _digit_width;
    if (target)
    {
        const int y = _padding / 2 + _line_height * (line + 1) - _line_height / 4;
        int baseline = 0;
        for (size_t k = 0; k < text.size(); ++k)
        {
            // from the right: the last character is in the last cell
            const std::string character(1, text[text.size() - 1 - k]);
            const int width = cv::getTextSize(character, kLegacyFont, _font_scale, _thickness, &baseline).width;
            const int cell = end - static_cast<int>(k + 1) * _digit_width;
            cv::putText(*target, character, cv::Point(cell + (_digit_width - width) / 2, y), kLegacyFont, _font_scale, cv::Scalar(255), _thickness, cv::LINE_AA);
        }
    }
    return end;
}

int MotionVideo::_legacy_text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const
{
    const int left = _padding;
    int widest = 0;

    // frame 1234   new   2104 points used   37 rejected   (arrows x8)
    int x = _legacy_label(target, left, 0, "frame");
    x = _legacy_number(target, x, 0, 8, std::to_string(values.frame)) + 2 * _space;
    _legacy_label(target, x, 0, !values.compared ? "first frame" : values.duplicate ? "duplicate" : values.torn ? "torn" : "new");
    x += _state_width + _space;
    x = _legacy_number(target, x, 0, 5, counted ? std::to_string(used) : "") + _space;
    x = _legacy_label(counted ? target : nullptr, x, 0, "points used") + _space;
    x = _legacy_number(target, x, 0, 5, counted ? std::to_string(rejected) : "") + _space;
    x = _legacy_label(counted ? target : nullptr, x, 0, "rejected");
    if (_arrow_scale != 1.0) x = _legacy_label(target, x + 2 * _space, 0, "arrows x" + format("%g", _arrow_scale));
    widest = std::max(widest, x);

    // animation step 1.05 intervals   votes spread 0.26
    x = _legacy_label(target, left, 1, "animation step");
    if (values.sample.measurable)
    {
        x = _legacy_number(target, x, 1, 7, format("%.2f", values.sample.step)) + _space;
        x = _legacy_label(target, x, 1, "intervals") + 2 * _space;
        x = _legacy_label(target, x, 1, "votes spread");
        x = _legacy_number(target, x, 1, 6, format("%.2f", values.sample.step_spread));
    }
    else
    {
        x = _legacy_label(target, x + 2 * _space, 1, "not measurable");
    }
    widest = std::max(widest, x);

    // animation error +0.4 ms   1 picture lost by the capture
    x = _legacy_label(target, left, 2, "animation error");
    x = _legacy_number(target, x, 2, 7, values.sample.has_error ? format("%+.1f", values.sample.error_ms) : "") + _space;
    x = _legacy_label(values.sample.has_error ? target : nullptr, x, 2, "ms");
    if (values.sample.lost > 0) x = _legacy_label(target, x + 2 * _space, 2, "1 picture lost by the capture");
    widest = std::max(widest, x);

    // framerate 119.0 fps   perceived smoothness 118.6 fps
    x = _legacy_label(target, left, 3, "framerate");
    x = _legacy_number(target, x, 3, 6, format("%.1f", values.framerate)) + _space;
    x = _legacy_label(target, x, 3, "fps") + 2 * _space;
    x = _legacy_label(target, x, 3, "perceived smoothness");
    x = _legacy_number(target, x, 3, 6, values.sample.has_perceived_fps ? format("%.1f", values.sample.perceived_fps) : "") + _space;
    x = _legacy_label(values.sample.has_perceived_fps ? target : nullptr, x, 3, "fps");
    widest = std::max(widest, x);
    return widest;
}

void MotionVideo::_add_legacy_chart()
{
    _scene.segments.insert(_scene.segments.end(), _chart_segments.begin(), _chart_segments.end());
    // the framerate wider, under the perceived smoothness: where they are the same it shows at both sides
    const float wide = (_measured.width >= 1280 ? 1.75f : 1.25f) * _k;
    const float narrow = (_measured.width >= 1280 ? 0.75f : 0.5f) * _k;
    const float right = _plot.x + _plot.width;
    const float step = _plot.width / static_cast<float>(_chart_frames - 1);
    const size_t count = _history.size();
    for (const bool perceived : { false, true })
    {
        const cv::Vec3b & color = perceived ? kLegacyPerceivedColor : kLegacyFramerateColor;
        const float thick = perceived ? narrow : wide;
        // one column of pixels at a time: a line from the last value of the previous column to the first of this
        // one, and one from the lowest to the highest within it (more frames than pixels)
        bool open = false, have_previous = false;
        int column = 0;
        float first = 0.0f, last = 0.0f, low = 0.0f, high = 0.0f, previous_x = 0.0f, previous_y = 0.0f;
        auto flush = [&]() {
            if (!open) return;
            const float x = static_cast<float>(column);
            if (have_previous) _scene.segments.push_back(chart_segment(previous_x, previous_y, x, first, thick, color));
            if (high > low || !have_previous) _scene.segments.push_back(chart_segment(x, low, x, high, thick, color));
            have_previous = true;
            previous_x = x;
            previous_y = last;
            open = false;
        };
        for (size_t k = 0; k < count; ++k)
        {
            const float value = perceived ? _history[k].perceived : _history[k].fps;
            if (std::isnan(value))
            {
                flush();
                have_previous = false;
                continue;
            }
            const float x = right - step * static_cast<float>(count - 1 - k);
            const float y = _plot.y + _plot.height * (1.0f - static_cast<float>(std::clamp(static_cast<double>(value) / _chart_top, 0.0, 1.0)));
            const int c = static_cast<int>(std::lround(x));
            if (!open || c != column)
            {
                flush();
                open = true;
                column = c;
                first = last = low = high = y;
            }
            else
            {
                last = y;
                low = std::min(low, y);
                high = std::max(high, y);
            }
        }
        flush();
    }
}

void MotionVideo::_update_overlay(const MotionVideoFrame & values)
{
    const double fps_val = std::isfinite(values.framerate) ? std::max(0.0, values.framerate) : 0.0;
    const bool has_perc = values.sample.has_perceived_fps && std::isfinite(values.sample.perceived_fps);
    const std::string fps_str = format("%.1f", fps_val);
    const std::string perc_str = has_perc ? format("%.1f", std::max(0.0, values.sample.perceived_fps)) : std::string("--");

    if (fps_str == _last_fps_str && perc_str == _last_perc_str) return;
    _last_fps_str = fps_str;
    _last_perc_str = perc_str;

    _static_overlay(_num_band).copyTo(_scene.overlay(_num_band));

    const OsdColor fps_col{ 246, 249, 253, 0.98f };
    const OsdColor perc_col = has_perc ? OsdColor{ 232, 238, 247, 0.90f } : OsdColor{ 232, 238, 247, 0.42f };
    _font_number.draw_tabular(_scene.overlay, _block1_x, _num_baseline_y, fps_str, fps_col);
    _font_number.draw_tabular(_scene.overlay, _block2_x, _num_baseline_y, perc_str, perc_col);
    ++_scene.overlay_generation;
}

void MotionVideo::_build_chart_samples()
{
    MotionVideoCard & card = _scene.card;
    std::fill(card.primary_y.begin(), card.primary_y.end(), -1.0f);
    std::fill(card.secondary_y.begin(), card.secondary_y.end(), -1.0f);
    card.primary_dot_x = -1.0f;
    card.primary_dot_y = -1.0f;
    card.secondary_dot_x = -1.0f;
    card.secondary_dot_y = -1.0f;
    if (_history.empty() || _plot_cols < 2) return;

    const long long latest_frame = _history.back().frame;
    const double col_per_frame = static_cast<double>(_plot_cols - 1) / static_cast<double>(_chart_frames - 1);
    const float y_min = card.plot.y;
    const float y_max = card.plot.y + card.plot.height;

    auto frame_to_col = [&](double f) -> double {
        return static_cast<double>(_plot_cols - 1) - (static_cast<double>(latest_frame) - f) * col_per_frame;
    };
    auto value_to_y = [&](double val) -> double {
        const double norm = std::clamp(val / _chart_top, 0.0, 1.0);
        return static_cast<double>(card.plot.y) + static_cast<double>(card.plot.height) * (1.0 - norm);
    };

    struct Knot {
        double frame;
        double value;
    };

    auto evaluate_run = [&](const std::vector<Knot> & run, std::vector<float> & out) {
        const size_t n = run.size();
        if (n == 0) return;

        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(n);
        ys.reserve(n);
        for (const Knot & k : run)
        {
            const double x = frame_to_col(k.frame);
            const double y = value_to_y(k.value);
            if (!xs.empty() && x <= xs.back() + 1e-6)
            {
                xs.back() = x;
                ys.back() = y;
            }
            else
            {
                xs.push_back(x);
                ys.push_back(y);
            }
        }
        const size_t m = xs.size();
        if (xs.back() < -0.5 || xs.front() > static_cast<double>(_plot_cols - 1) + 0.5) return;

        if (m == 1)
        {
            const int col = static_cast<int>(std::lround(xs[0]));
            if (col >= 0 && col < _plot_cols)
            {
                out[static_cast<size_t>(col)] = static_cast<float>(std::clamp(ys[0], static_cast<double>(y_min), static_cast<double>(y_max)));
            }
            return;
        }

        // Fritsch-Carlson monotone cubic Hermite spline (Brodlie non-uniform tangents + monotonicity limiter)
        const size_t segs = m - 1;
        std::vector<double> h(segs), d(segs), tangents(m, 0.0);
        for (size_t i = 0; i < segs; ++i)
        {
            h[i] = xs[i + 1] - xs[i];
            d[i] = (ys[i + 1] - ys[i]) / h[i];
        }
        tangents[0] = d[0];
        tangents[m - 1] = d[segs - 1];
        for (size_t i = 1; i + 1 < m; ++i)
        {
            if (d[i - 1] * d[i] <= 0.0)
            {
                tangents[i] = 0.0;
            }
            else
            {
                const double w1 = 2.0 * h[i] + h[i - 1];
                const double w2 = h[i] + 2.0 * h[i - 1];
                tangents[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
            }
        }
        for (size_t i = 0; i < segs; ++i)
        {
            if (std::abs(d[i]) < 1e-12)
            {
                tangents[i] = 0.0;
                tangents[i + 1] = 0.0;
                continue;
            }
            double alpha = tangents[i] / d[i];
            double beta = tangents[i + 1] / d[i];
            if (alpha < 0.0) { tangents[i] = 0.0; alpha = 0.0; }
            if (beta < 0.0) { tangents[i + 1] = 0.0; beta = 0.0; }
            const double r2 = alpha * alpha + beta * beta;
            if (r2 > 9.0)
            {
                const double tau = 3.0 / std::sqrt(r2);
                tangents[i] = tau * alpha * d[i];
                tangents[i + 1] = tau * beta * d[i];
            }
        }

        const int c_start = std::max(0, static_cast<int>(std::ceil(xs.front() - 1e-4)));
        const int c_end = std::min(_plot_cols - 1, static_cast<int>(std::floor(xs.back() + 1e-4)));
        if (c_start > c_end)
        {
            const int col = static_cast<int>(std::lround(0.5 * (xs.front() + xs.back())));
            if (col >= 0 && col < _plot_cols)
            {
                out[static_cast<size_t>(col)] = static_cast<float>(std::clamp(0.5 * (ys.front() + ys.back()), static_cast<double>(y_min), static_cast<double>(y_max)));
            }
            return;
        }

        size_t seg = 0;
        for (int col = c_start; col <= c_end; ++col)
        {
            const double cx = static_cast<double>(col);
            while (seg + 1 < segs && xs[seg + 1] < cx) ++seg;
            const double t = std::clamp((cx - xs[seg]) / h[seg], 0.0, 1.0);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            const double h10 = t3 - 2.0 * t2 + t;
            const double h01 = -2.0 * t3 + 3.0 * t2;
            const double h11 = t3 - t2;
            const double val = h00 * ys[seg] + h10 * h[seg] * tangents[seg]
                             + h01 * ys[seg + 1] + h11 * h[seg] * tangents[seg + 1];
            out[static_cast<size_t>(col)] = static_cast<float>(std::clamp(val, static_cast<double>(y_min), static_cast<double>(y_max)));
        }
    };

    auto fill_series = [&](bool secondary, std::vector<float> & out, float & dot_x, float & dot_y) {
        std::vector<Knot> run;
        bool bucket_open = false;
        long long bucket_id = 0;
        double sum_frame = 0.0;
        double sum_val = 0.0;
        int bucket_count = 0;

        auto flush_bucket = [&]() {
            if (!bucket_open) return;
            run.push_back(Knot{ sum_frame / static_cast<double>(bucket_count), sum_val / static_cast<double>(bucket_count) });
            bucket_open = false;
            sum_frame = 0.0;
            sum_val = 0.0;
            bucket_count = 0;
        };
        auto flush_run = [&]() {
            flush_bucket();
            if (!run.empty())
            {
                evaluate_run(run, out);
                run.clear();
            }
        };

        for (const HistorySample & entry : _history)
        {
            const float val = secondary ? entry.perceived : entry.fps;
            if (!std::isfinite(val))
            {
                flush_run();
                continue;
            }
            const long long bid = entry.frame / _bucket_frames;
            if (bucket_open && bid != bucket_id) flush_bucket();
            bucket_open = true;
            bucket_id = bid;
            sum_frame += static_cast<double>(entry.frame);
            sum_val += static_cast<double>(val);
            ++bucket_count;
        }
        flush_bucket();

        // Append the latest live frame at the rightmost edge if separated from the last bucket mean
        const HistorySample & last = _history.back();
        const float last_val = secondary ? last.perceived : last.fps;
        if (std::isfinite(last_val) && !run.empty() && static_cast<double>(last.frame) > run.back().frame + 1e-6)
        {
            run.push_back(Knot{ static_cast<double>(last.frame), static_cast<double>(last_val) });
        }
        flush_run();

        for (int col = _plot_cols - 1; col >= 0; --col)
        {
            const float y = out[static_cast<size_t>(col)];
            if (y >= 0.0f)
            {
                dot_x = card.plot.x + static_cast<float>(col);
                dot_y = y;
                break;
            }
        }
    };

    fill_series(false, card.primary_y, card.primary_dot_x, card.primary_dot_y);
    fill_series(true, card.secondary_y, card.secondary_dot_x, card.secondary_dot_y);
}

void MotionVideo::_add_points(const std::vector<MotionPoint> & points, bool dimmed)
{
    const float thickness = (_measured.width >= 1280 ? 2.0f : 1.0f) * _k;
    const float cross = kCrossSize * _k;
    for (const MotionPoint & point : points)
    {
        // the points are in pixels of the motion: the center of such a pixel in the picture
        const float px = _k == 1.0f ? point.x : (point.x + 0.5f) * _k - 0.5f;
        const float py = _k == 1.0f ? point.y : (point.y + 0.5f) * _k - 0.5f;
        if (point.state == MotionPoint::Rejected)
        {
            MotionVideoSegment arm;
            arm.radius = 0.5f * _k;
            arm.blue = 40;
            arm.green = 40;
            arm.red = 230;
            arm.x0 = px - cross;
            arm.y0 = py - cross;
            arm.x1 = px + cross;
            arm.y1 = py + cross;
            _scene.segments.push_back(arm);
            std::swap(arm.y0, arm.y1);
            _scene.segments.push_back(arm);
        }
        if (point.state != MotionPoint::Used) continue;
        const double length = std::hypot(point.dx, point.dy);
        cv::Vec3b color = _colors[static_cast<size_t>(std::min(255.0, kFirstColor + (255 - kFirstColor) * length / kLongestMotion))];
        if (dimmed)
        {
            for (int c = 0; c < 3; ++c) color[c] = cv::saturate_cast<uchar>(0.45 * color[c] + 60.0);
        }
        MotionVideoSegment segment;
        segment.blue = color[0];
        segment.green = color[1];
        segment.red = color[2];
        segment.x0 = segment.x1 = px;
        segment.y0 = segment.y1 = py;
        const double drawn = length * _arrow_scale;
        if (drawn < kShortestArrow)
        {
            segment.radius = thickness;
            _scene.segments.push_back(segment);
            continue;
        }
        segment.radius = 0.5f * thickness;
        segment.x1 = static_cast<float>(px + point.dx * _arrow_scale * _k);
        segment.y1 = static_cast<float>(py + point.dy * _arrow_scale * _k);
        _scene.segments.push_back(segment);
        // the two lines of the tip, back from the end of the arrow
        const double tip = std::min(0.5 * drawn, kTipLength) * _k;
        const double angle = std::atan2(segment.y0 - segment.y1, segment.x0 - segment.x1);
        for (const double side : { CV_PI / 4.0, -CV_PI / 4.0 })
        {
            MotionVideoSegment line = segment;
            line.x0 = static_cast<float>(segment.x1 + tip * std::cos(angle + side));
            line.y0 = static_cast<float>(segment.y1 + tip * std::sin(angle + side));
            _scene.segments.push_back(line);
        }
    }
}

void MotionVideo::write(const MotionVideoFrame & values, const MotionDetail & detail)
{
    _scene.segments.clear();
    int used = 0, rejected = 0;
    const bool counted = !detail.points.empty() && !(values.compared && values.duplicate);
    if (_show_arrows)
    {
        if (!detail.points.empty())
        {
            if (values.compared && values.duplicate && !_held.empty())
            {
                _add_points(_held, true);
            }
            else
            {
                _add_points(detail.points, false);
                _held = detail.points;
            }
            if (_osd_style == MotionOsdStyle::Detailed)
            {
                for (const MotionPoint & point : detail.points)
                {
                    used += point.state == MotionPoint::Used;
                    rejected += point.state == MotionPoint::Rejected;
                }
            }
        }
        else
        {
            _held.clear();
        }
    }

    if (_osd_style == MotionOsdStyle::Detailed)
    {
        if (_chart)
        {
            _history.push_back(HistorySample{ _frames, static_cast<float>(values.framerate),
                                              values.sample.has_perceived_fps ? static_cast<float>(values.sample.perceived_fps) : std::nanf("") });
            while (_history.size() > _chart_frames) _history.pop_front();
            _add_legacy_chart();
        }
        if (!_scene.box.empty())
        {
            if (_chart) _chart_text.copyTo(_scene.text);
            else _scene.text.setTo(0);
            _legacy_text(&_scene.text, values, used, rejected, counted);
        }
    }
    else if (_scene.card.enabled)
    {
        _update_overlay(values);
        if (_chart)
        {
            const float fps_val = std::isfinite(values.framerate) ? static_cast<float>(std::max(0.0, values.framerate)) : std::numeric_limits<float>::quiet_NaN();
            const float perc_val = (values.sample.has_perceived_fps && std::isfinite(values.sample.perceived_fps))
                                 ? static_cast<float>(std::max(0.0, values.sample.perceived_fps))
                                 : std::numeric_limits<float>::quiet_NaN();
            _history.push_back(HistorySample{ _frames, fps_val, perc_val });
            // Keep one extra bucket off the left edge so the spline enters col 0 cleanly without breathing
            const size_t max_history = _chart_frames + static_cast<size_t>(_bucket_frames);
            while (_history.size() > max_history) _history.pop_front();
            _build_chart_samples();
        }
    }

    _target->write(detail, _scene);
    ++_frames;
}

} // namespace trdrop
