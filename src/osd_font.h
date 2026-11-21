#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace trdrop {

enum class OsdFontWeight { Bold, Medium };

struct OsdColor {
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
    float alpha = 1.0f;
};

class OsdFontFace {
public:
    OsdFontFace() = default;
    OsdFontFace(OsdFontWeight weight, float pixel_height);

    bool valid() const;
    float pixel_height() const;
    float ascent() const;
    float descent() const;
    float cap_height() const;
    float digit_cell_width() const;
    float dot_cell_width() const;
    float dash_cell_width() const;

    float measure_text(const std::string & text) const;
    float measure_tabular(const std::string & text) const;

    // All draw_* blend onto premultiplied RGBA8 (`CV_8UC4`, byte order R,G,B,A) with Porter-Duff source-over.
    // draw_text / draw_tabular return the final pen x; draw_text_right returns the starting left x.
    float draw_text(cv::Mat & rgba_pm, float x, float baseline_y, const std::string & text, const OsdColor & color) const;
    float draw_text_right(cv::Mat & rgba_pm, float right_x, float baseline_y, const std::string & text, const OsdColor & color) const;
    float draw_tabular(cv::Mat & rgba_pm, float x, float baseline_y, const std::string & text, const OsdColor & color) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> _impl;
};

void draw_rounded_rect_rgba(cv::Mat & rgba_pm, float x, float y, float w, float h, float radius, const OsdColor & color);

} // namespace trdrop
