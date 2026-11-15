#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "motion.h"
#include "motion_video_target.h"
#include "resolution.h"

namespace trdrop {

//! Writes a copy of a video that shows how its render resolution is estimated (--resolution-video), frame by
//! frame, at the resolution the motion is measured at:
//!   on the picture   the part the pixel grid is measured in (above a line), the columns next to the left and
//!                    the right edge whose spectra are compared in a pan, and in a frame of a pan: which edge the
//!                    picture enters at, the groups of columns by the frame they entered with, the strip whose
//!                    stair steps are counted and how far the picture moved
//!   three charts     what was collected up to the frame, as the summary reads it: the pixel grid (how many
//!                    times the background the spectrum of the columns / rows is), the contrast of the entering
//!                    picture against the leaving one per group, the stair steps by their height
//!   a table          what the frame was used for, and what the traces say so far: a value, what it is, the
//!                    render scale it means
//!
//! It draws with what a motion video is drawn with (MotionVideoScene on a MotionVideoTarget): lines and text.
//! The text layer covers the whole picture, the dark panels are lines.
class ResolutionVideo {
public:
    //! frame_width, frame_height: size of the frames of the video, fps: its recorded framerate,
    //! target: where the frames go, nullptr: drawn on the CPU and written to `path`.
    //! throws std::runtime_error if the file can't be written
    ResolutionVideo(const std::string & path, int frame_width, int frame_height, double fps, std::unique_ptr<MotionVideoTarget> target);

    const std::string & path() const { return _path; }
    cv::Size size() const { return _size; }
    std::string description() const { return _target->description(); }
    long long frames() const { return _frames; }

    //! frame: its number, duplicate: it shows the picture of the frame before, analyzer: after its push() of the
    //! frame, detail: the picture of the frame. On a duplicate what was drawn on the picture before stays: it
    //! has no motion, and the boxes of a pan would flicker
    void write(long long frame, bool duplicate, const ResolutionAnalyzer & analyzer, const MotionDetail & detail);
    void finish() { _target->finish(); }

private:
    //! a font size: digits, and the spaces that pad a number (every space after a space), get the width of the
    //! widest digit, so numbers of a fixed format stand still
    struct Font {
        double scale = 0.5;
        int digit = 0;
        int height = 0;
        std::array<int, 128> width {};
    };
    struct Label {
        int x = 0;
        int y = 0;
        std::string text;
        bool small = false;
        //! -1: ends at x, 0: centered on x, 1: starts at x
        int align = 1;
    };
    struct Rect {
        float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
        float width() const { return x1 - x0; }
        float height() const { return y1 - y0; }
    };

    static void _make_font(Font & font, double scale);
    //! how far a character moves the cursor, `previous`: the one before it (0: none)
    static int _advance(const Font & font, char c, char previous);
    int _text_width(const std::string & text, const Font & font) const;
    void _put(cv::Mat & target, const Label & label) const;
    void _line(std::vector<MotionVideoSegment> & out, float x0, float y0, float x1, float y1, const cv::Vec3b & color, float radius) const;
    void _dashes(std::vector<MotionVideoSegment> & out, float x0, float y0, float x1, float y1, const cv::Vec3b & color, float on, float off) const;
    void _fill(std::vector<MotionVideoSegment> & out, const Rect & rect, const cv::Vec3b & color) const;
    void _outline(std::vector<MotionVideoSegment> & out, const Rect & rect, const cv::Vec3b & color, float radius) const;

    //! once the layout of the measured frames is known: the panels, their axes and the text that stays
    void _prepare(const ResolutionAnalyzer & analyzer);
    //! the charts and the lines of text that follow what was collected
    void _refresh(const ResolutionAnalyzer & analyzer);
    //! what belongs to this frame, on the picture
    void _frame_overlay(const ResolutionAnalyzer::FrameState & state);
    std::string _frame_text(const ResolutionAnalyzer::FrameState & state) const;

    std::string _path;
    cv::Size _size;
    std::unique_ptr<MotionVideoTarget> _target;
    long long _frames = 0;

    //! picture pixels per pixel of the frame, and the unit of the layout: 1 for a picture 1920 wide
    float _sx = 1.0f;
    float _sy = 1.0f;
    float _unit = 1.0f;
    float _thin = 0.5f;
    Font _font;
    Font _small;

    bool _prepared = false;
    //! the picture is too small for the charts: the frame's overlay and text only
    bool _charts = true;
    ResolutionLayout _layout;
    Rect _text_panel;
    Rect _grid_panel, _grid_plot;
    Rect _pan_panel, _pan_plot;
    Rect _stairs_panel, _stairs_plot;

    //! panels and axes; the charts as of the last refresh; this frame
    std::vector<MotionVideoSegment> _static_segments;
    std::vector<MotionVideoSegment> _chart_segments;
    std::vector<MotionVideoSegment> _frame_segments;
    cv::Mat _static_text;
    std::vector<Label> _chart_labels;
    std::vector<Label> _frame_labels;
    //! the frame has an overlay (of its own, or held from the last new frame)
    bool _has_overlay = false;
    //! the rows of the text panel that follow the summary (grid, pan, stairs, result): the value that was
    //! read, what it is, and the render scale it means
    std::array<std::array<std::string, 3>, 4> _cells;
    long long _refreshed = -1;
    std::vector<double> _columns;
    std::vector<double> _rows;
    MotionVideoScene _scene;
};

} // namespace trdrop
