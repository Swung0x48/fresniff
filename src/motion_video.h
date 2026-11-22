#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"
#include "motion_video_target.h"
#include "osd_font.h"

namespace fresniff {

//! what the analysis says about a frame of the motion video
struct MotionVideoFrame {
    long long frame = 0;
    //! the frame was compared with the previous one ...
    bool compared = false;
    //! ... and is a duplicate of it / shows a part of the previous frame (torn)
    bool duplicate = false;
    bool torn = false;
    //! framerate over the last second
    double framerate = 0.0;
    MotionSample sample;
};

//! overlay style of --motion-video
enum class MotionOsdStyle {
    Glass,
    Detailed,
};

//! Writes a copy of a video with either a frosted-glass framerate and perceived smoothness overlay card
//! at the bottom (Glass, optionally with optical-flow arrows when show_arrows is enabled), or the legacy
//! optical-flow arrows, top-left telemetry text box and 10 s chart (Detailed).
//!
//! This class decides what is drawn (MotionVideoScene), a MotionVideoTarget puts it on the picture and
//! encodes it: the one of the pipeline if it draws on the GPU, otherwise the one for the CPU.
class MotionVideo {
public:
    //! frame_width, frame_height: size of the frames of the video, fps: its recorded framerate,
    //! arrow_scale: the arrows are drawn this many times longer than the motion,
    //! target: where the frames go, nullptr: drawn on the CPU and written to `path`.
    //! throws std::runtime_error if the file can't be written
    //! native: the picture at the resolution of the frames (--motion-video-native),
    //! osd_style: Glass (frosted-glass bottom card) or Detailed (legacy telemetry box + arrows),
    //! show_arrows: draw optical-flow arrows and rejected points in Glass mode (always on in Detailed)
    MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale,
                std::unique_ptr<MotionVideoTarget> target, bool native = false,
                MotionOsdStyle osd_style = MotionOsdStyle::Glass, bool show_arrows = false);

    const std::string & path() const { return _path; }
    cv::Size size() const { return _size; }
    std::string description() const { return _target->description(); }
    long long frames() const { return _frames; }

    void write(const MotionVideoFrame & values, const MotionDetail & detail);
    void finish() { _target->finish(); }

private:
    struct HistorySample {
        long long frame = 0;
        float fps = 0.0f;
        float perceived = 0.0f;
    };

    void _init_glass(double fps);
    void _init_detailed(double fps);
    void _add_points(const std::vector<MotionPoint> & points, bool dimmed);
    void _update_overlay(const MotionVideoFrame & values);
    void _build_chart_samples();

    //! legacy detailed OSD helpers:
    //! cursor of a line of text: draws into `target` if given and returns the x position after the text
    int _legacy_label(cv::Mat * target, int x, int line, const std::string & text) const;
    //! a number in a field of `cells` digits, right aligned, every character in a cell of its own
    int _legacy_number(cv::Mat * target, int x, int line, int cells, const std::string & text) const;
    //! the four lines of values in the text box, returns the x position where the longest line ends
    int _legacy_text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const;
    //! the curves of the legacy chart, from _history
    void _add_legacy_chart();

    std::string _path;
    //! of the picture; of the motion (the points are in its pixels); the one over the other
    cv::Size _size;
    cv::Size _measured;
    float _k = 1.0f;
    double _arrow_scale = 1.0;
    MotionOsdStyle _osd_style = MotionOsdStyle::Glass;
    bool _show_arrows = false;
    std::unique_ptr<MotionVideoTarget> _target;
    long long _frames = 0;

    //! colors of the arrows by length (BGR), initialized only when _show_arrows is true
    std::vector<cv::Vec3b> _colors;
    //! the points of the last new frame
    std::vector<MotionPoint> _held;
    MotionVideoScene _scene;

    // frosted-glass card typography and cached static overlay (premultiplied RGBA8 of card.rect.size())
    OsdFontFace _font_number;
    OsdFontFace _font_label;
    OsdFontFace _font_tick;
    float _block1_x = 0.0f;
    float _block2_x = 0.0f;
    float _num_baseline_y = 0.0f;
    cv::Rect _num_band;
    cv::Mat _static_overlay;
    std::string _last_fps_str;
    std::string _last_perc_str;

    // legacy detailed text & chart layout
    int _margin = 8;
    int _padding = 10;
    int _thickness = 1;
    double _font_scale = 0.6;
    int _digit_width = 0;
    int _space = 0;
    int _line_height = 0;
    int _state_width = 0;
    cv::Rect2f _plot;
    cv::Mat _chart_text;
    std::vector<MotionVideoSegment> _chart_segments;

    // chart of the last kChartSeconds
    bool _chart = false;
    double _chart_top = 60.0;
    size_t _chart_frames = 0;
    long long _bucket_frames = 1;
    int _plot_cols = 0;
    std::deque<HistorySample> _history;
};

//! the target that draws on MotionDetail::picture on the CPU and writes the file with make_video_writer
std::unique_ptr<MotionVideoTarget> make_cpu_motion_video_target(const std::string & path, cv::Size size, double fps);

} // namespace fresniff
