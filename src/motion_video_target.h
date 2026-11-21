#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"

namespace trdrop {

//! a line with round ends in the motion video (a dot if both ends are the same), in pixels of the picture
struct MotionVideoSegment {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    //! half of its width
    float radius = 0.5f;
    uint8_t blue = 0;
    uint8_t green = 0;
    uint8_t red = 0;
    //! drawn after the text box was darkened (a chart in the box), not before it (an arrow under the box)
    bool above_box = false;
};

struct MotionVideoColor {
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
    float alpha = 1.0f;
};

struct MotionVideoCard {
    bool enabled = false;
    cv::Rect rect; // even x, y, width, height in frame pixels
    float corner_radius = 24.0f;
    float blur_sigma = 22.0f;
    float vibrancy = 1.12f;
    MotionVideoColor tint_top{ 20, 23, 30, 0.58f };
    MotionVideoColor tint_bottom{ 10, 12, 18, 0.70f };
    float sheen_alpha = 0.045f;
    float sheen_height = 0.38f;
    float border_width = 1.25f;
    float border_alpha_top = 0.28f;
    float border_alpha_bottom = 0.08f;
    float border_power = 1.4f;
    float shadow_offset_y = 6.0f;
    float shadow_alpha = 0.28f;
    float shadow_decay = 14.0f;
    float shadow_pad = 36.0f;

    bool has_chart = false;
    cv::Rect2f plot;
    int grid_count = 0;
    float grid_y[4] = {};
    float grid_alpha[4] = {};
    float grid_half_width = 0.5f;

    MotionVideoColor primary_color{ 245, 248, 252, 0.96f };
    float primary_half_width = 1.55f;
    float primary_fill_alpha = 0.14f;
    float primary_fill_power = 1.45f;

    MotionVideoColor secondary_color{ 82, 182, 255, 0.92f };
    float secondary_half_width = 1.10f;

    float primary_dot_x = -1.0f;
    float primary_dot_y = -1.0f;
    float primary_dot_radius = 3.0f;
    float primary_glow_radius = 8.0f;
    float primary_glow_alpha = 0.28f;

    float secondary_dot_x = -1.0f;
    float secondary_dot_y = -1.0f;
    float secondary_dot_radius = 2.4f;
    float secondary_glow_radius = 6.5f;
    float secondary_glow_alpha = 0.28f;

    std::vector<float> primary_y;   // length = plot_cols (0 if !has_chart), < 0.0f means missing/NaN
    std::vector<float> secondary_y; // length = plot_cols (0 if !has_chart), < 0.0f means missing/NaN
};

//! what is drawn on the picture of a frame, in this order: the picture dimmed, the segments that are not above_box,
//! the text box darkened, the segments above_box, the text (or the frosted-glass OSD card when card.enabled is true).
//! Segments above_box come after all others in the list
struct MotionVideoScene {
    //! brightness of the picture under everything else
    float dim = 1.0f;
    std::vector<MotionVideoSegment> segments;
    //! the text box: the picture in it is darkened by box_dim and gets the text in white. Empty: no text
    cv::Rect box;
    float box_dim = 1.0f;
    //! coverage of the text (0 - 255), of the size of the box
    cv::Mat text;

    MotionVideoCard card;
    cv::Mat overlay;             // premultiplied RGBA8 (CV_8UC4, byte order R,G,B,A) of size card.rect.size()
    uint64_t overlay_generation = 0;
};

//! Where the frames of a motion video go: puts the scene on the picture of the frame and encodes it.
//! On the CPU (MotionDetail::picture) or, by a pipeline that has the frames there, on the GPU.
class MotionVideoTarget {
public:
    virtual ~MotionVideoTarget() = default;

    //! encoder, and where the picture is drawn
    virtual std::string description() const = 0;
    virtual void write(const MotionDetail & detail, const MotionVideoScene & scene) = 0;
    //! writes what is left, the file is complete afterwards
    virtual void finish() = 0;
};

} // namespace trdrop
