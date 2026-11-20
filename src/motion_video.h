#pragma once

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"
#include "motion_video_target.h"

namespace trdrop {

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

//! Writes a copy of a video with the motion the analysis works with drawn on it, at the resolution the motion
//! is measured at:
//!   arrow    a point of the grid that was tracked: from where it was in the previous frame to where it is now,
//!            colored by its length. On a duplicate frame the arrows of the last new frame stay, dimmed
//!   red x    a point whose match doesn't come back, it is not used
//!   nothing  no texture at the point
//! and the values of the frame in the upper left corner. The numbers have fixed places, so the text stands still.
//! Under them a chart of the framerate and the perceived smoothness of the last seconds, the frame at its right
//! edge, if the picture has room for it.
//!
//! This class decides what is drawn (MotionVideoScene), a MotionVideoTarget puts it on the picture and
//! encodes it: the one of the pipeline if it draws on the GPU, otherwise the one for the CPU.
class MotionVideo {
public:
    //! frame_width, frame_height: size of the frames of the video, fps: its recorded framerate,
    //! arrow_scale: the arrows are drawn this many times longer than the motion,
    //! target: where the frames go, nullptr: drawn on the CPU and written to `path`.
    //! throws std::runtime_error if the file can't be written
    //! native: the picture at the resolution of the frames (--motion-video-native), the arrows, the text and the
    //! chart drawn as much larger as the frames are than the motion
    MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale,
                std::unique_ptr<MotionVideoTarget> target, bool native = false);

    const std::string & path() const { return _path; }
    cv::Size size() const { return _size; }
    std::string description() const { return _target->description(); }
    long long frames() const { return _frames; }

    void write(const MotionVideoFrame & values, const MotionDetail & detail);
    void finish() { _target->finish(); }

private:
    //! cursor of a line of text: draws into `target` if given and returns the x position after the text
    int _label(cv::Mat * target, int x, int line, const std::string & text) const;
    //! a number in a field of `cells` digits, right aligned, every character in a cell of its own
    int _number(cv::Mat * target, int x, int line, int cells, const std::string & text) const;
    //! the four lines of values in the text box, returns the x position where the longest line ends
    int _text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const;
    void _add_points(const std::vector<MotionPoint> & points, bool dimmed);
    //! the curves of the chart, from _history
    void _add_chart();

    std::string _path;
    //! of the picture; of the motion (the points are in its pixels); the one over the other
    cv::Size _size;
    cv::Size _measured;
    float _k = 1.0f;
    double _arrow_scale;
    std::unique_ptr<MotionVideoTarget> _target;
    long long _frames = 0;

    // text layout
    int _margin = 8;
    int _padding = 10;
    int _thickness = 1;
    double _font_scale = 0.6;
    int _digit_width = 0;
    int _space = 0;
    int _line_height = 0;
    int _state_width = 0;

    //! colors of the arrows by length (BGR)
    std::vector<cv::Vec3b> _colors;
    //! the points of the last new frame
    std::vector<MotionPoint> _held;
    MotionVideoScene _scene;

    // the chart under the text: framerate and perceived smoothness of the last kChartSeconds (none if it doesn't fit)
    bool _chart = false;
    //! where the curves are drawn, in pixels of the picture
    cv::Rect2f _plot;
    //! the value at the top of the plot
    double _chart_top = 60.0;
    //! frames the plot shows
    size_t _chart_frames = 0;
    //! per frame of the plot, the oldest first: framerate, perceived smoothness (NaN: none)
    std::deque<std::pair<float, float>> _history;
    //! the labels of the chart, of the size of the box; the grid and the legend
    cv::Mat _chart_text;
    std::vector<MotionVideoSegment> _chart_segments;
};

//! the target that draws on MotionDetail::picture on the CPU and writes the file with make_video_writer
std::unique_ptr<MotionVideoTarget> make_cpu_motion_video_target(const std::string & path, cv::Size size, double fps);

} // namespace trdrop
