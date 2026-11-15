#include "resolution_video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "motion_video.h"

namespace trdrop {

namespace {

//! brightness of the picture under the drawing
constexpr float kDim = 0.55f;
//! the charts and the text that follow what was collected are made anew every this many frames
constexpr long long kRefresh = 6;
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

//! the pixel grid chart: frequencies and strengths (times the background, a logarithmic axis) it shows
constexpr double kGridFrom = 0.06;
constexpr double kGridTo = 0.5;
constexpr double kGridLow = 0.5;
constexpr double kGridOctaves = 7.0;
//! the highest contrast the pan chart shows
constexpr double kContrastTop = 1.6;

// colors, BGR
const cv::Vec3b kPanel(22, 22, 22);
const cv::Vec3b kAxis(120, 120, 120);
const cv::Vec3b kFaint(64, 64, 64);
const cv::Vec3b kWhite(255, 255, 255);
const cv::Vec3b kUnused(120, 120, 120);
const cv::Vec3b kColumns(255, 190, 70);
const cv::Vec3b kRows(70, 150, 255);
const cv::Vec3b kEntering(70, 225, 255);
const cv::Vec3b kLeaving(200, 200, 200);
//! the groups of edge columns, the newest first
const cv::Vec3b kAges[kResolutionAges] = {
    cv::Vec3b(70, 70, 255), cv::Vec3b(60, 150, 255), cv::Vec3b(70, 225, 255),
    cv::Vec3b(110, 220, 110), cv::Vec3b(235, 200, 90), cv::Vec3b(255, 130, 130),
};
const char * const kAgeLabels[kResolutionAges] = {
    "this frame, outer half", "this frame, inner half", "1 frame ago", "2 - 3 frames ago", "4 - 7 frames ago", "8 - 15 frames ago",
};

std::string format(const char * pattern, double a = 0.0, double b = 0.0, double c = 0.0, double d = 0.0)
{
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), pattern, a, b, c, d);
    return buffer;
}

} // namespace

ResolutionVideo::ResolutionVideo(const std::string & path, int frame_width, int frame_height, double fps, std::unique_ptr<MotionVideoTarget> target)
    : _path(path)
    , _size(motion_picture_size(frame_width, frame_height))
    , _target(std::move(target))
{
    if (_size.width < 2 || _size.height < 2) throw std::runtime_error("the video is too small for a resolution video");
    if (!_target) _target = make_cpu_motion_video_target(path, _size, fps);

    _sx = static_cast<float>(_size.width) / static_cast<float>(frame_width);
    _sy = static_cast<float>(_size.height) / static_cast<float>(frame_height);
    _unit = static_cast<float>(_size.width) / 1920.0f;
    _thin = _size.width >= 1280 ? 0.75f : 0.5f;
    _make_font(_font, std::clamp(0.5 * _unit, 0.35, 0.6));
    _make_font(_small, std::clamp(0.42 * _unit, 0.3, 0.5));

    // the text may stand anywhere: its layer covers the picture and darkens nothing
    _scene.dim = kDim;
    _scene.box_dim = 1.0f;
    _scene.box = cv::Rect(cv::Point(0, 0), _size);
    _scene.text.create(_size, CV_8UC1);
    _static_text = cv::Mat::zeros(_size, CV_8UC1);
}

void ResolutionVideo::_make_font(Font & font, double scale)
{
    font.scale = scale;
    int baseline = 0;
    for (int c = 32; c < 127; ++c)
    {
        // without the thickness of the line, which getTextSize adds to every text
        font.width[static_cast<size_t>(c)] = std::max(1, cv::getTextSize(std::string(1, static_cast<char>(c)), kFont, scale, 1, &baseline).width - 1);
    }
    for (char digit = '0'; digit <= '9'; ++digit) font.digit = std::max(font.digit, font.width[static_cast<size_t>(digit)]);
    font.height = cv::getTextSize("0", kFont, scale, 1, &baseline).height;
}

int ResolutionVideo::_advance(const Font & font, char c, char previous)
{
    if (c < 32 || c > 126) return 0;
    // a digit, and a space after a space: the padding of a number
    if ((c >= '0' && c <= '9') || (c == ' ' && previous == ' ')) return font.digit;
    return font.width[static_cast<size_t>(c)];
}

int ResolutionVideo::_text_width(const std::string & text, const Font & font) const
{
    int width = 0;
    char previous = 0;
    for (const char c : text)
    {
        width += _advance(font, c, previous);
        previous = c;
    }
    return width;
}

void ResolutionVideo::_put(cv::Mat & target, const Label & label) const
{
    const Font & font = label.small ? _small : _font;
    int x = label.x;
    if (label.align <= 0)
    {
        const int width = _text_width(label.text, font);
        x -= label.align < 0 ? width : width / 2;
    }
    char previous = 0;
    for (const char c : label.text)
    {
        const int advance = _advance(font, c, previous);
        previous = c;
        if (advance == 0) continue;
        if (c != ' ')
        {
            cv::putText(target, std::string(1, c), cv::Point(x + (advance - font.width[static_cast<size_t>(c)]) / 2, label.y), kFont, font.scale,
                        cv::Scalar(255), 1, cv::LINE_AA);
        }
        x += advance;
    }
}

void ResolutionVideo::_line(std::vector<MotionVideoSegment> & out, float x0, float y0, float x1, float y1, const cv::Vec3b & color, float radius) const
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
    out.push_back(segment);
}

void ResolutionVideo::_dashes(std::vector<MotionVideoSegment> & out, float x0, float y0, float x1, float y1, const cv::Vec3b & color, float on, float off) const
{
    const float length = std::hypot(x1 - x0, y1 - y0);
    if (length <= 0.0f) return;
    const float ux = (x1 - x0) / length, uy = (y1 - y0) / length;
    for (float at = 0.0f; at < length; at += on + off)
    {
        const float end = std::min(length, at + on);
        _line(out, x0 + ux * at, y0 + uy * at, x0 + ux * end, y0 + uy * end, color, _thin);
    }
}

void ResolutionVideo::_fill(std::vector<MotionVideoSegment> & out, const Rect & rect, const cv::Vec3b & color) const
{
    // lines of round ends, one below the other
    const float radius = std::max(3.0f, std::min(6.0f * _unit, 0.5f * std::min(rect.width(), rect.height())));
    const float last = rect.y1 - radius;
    for (float y = rect.y0 + radius; ; y += 1.5f * radius)
    {
        const float at = std::min(y, last);
        _line(out, rect.x0 + radius, at, rect.x1 - radius, at, color, radius);
        if (y >= last) break;
    }
}

void ResolutionVideo::_outline(std::vector<MotionVideoSegment> & out, const Rect & rect, const cv::Vec3b & color, float radius) const
{
    _line(out, rect.x0, rect.y0, rect.x1, rect.y0, color, radius);
    _line(out, rect.x1, rect.y0, rect.x1, rect.y1, color, radius);
    _line(out, rect.x1, rect.y1, rect.x0, rect.y1, color, radius);
    _line(out, rect.x0, rect.y1, rect.x0, rect.y0, color, radius);
}

void ResolutionVideo::_prepare(const ResolutionAnalyzer & analyzer)
{
    _prepared = true;
    _layout = analyzer.layout();
    _sx = static_cast<float>(_size.width) / static_cast<float>(_layout.width);
    _sy = static_cast<float>(_size.height) / static_cast<float>(_layout.height);
    const float width = static_cast<float>(_size.width), height = static_cast<float>(_size.height);
    const float u = _unit;
    const float gap = 12.0f * u;
    const float strip = static_cast<float>(_layout.edge) * _sx;
    // between the edge columns: the charts at the top, the text at the bottom
    const float x0 = strip + gap, x1 = width - strip - gap;
    const float line = static_cast<float>(_font.height) + 8.0f * u;
    const float text_height = 5.0f * line + 10.0f * u;
    _text_panel = { x0, height - 4.0f * u - text_height, x1, height - 4.0f * u };
    // as wide as the table: three candidates in its last row
    _text_panel.x1 = std::min(x1, x0 + 20.0f * u + 90.0f * static_cast<float>(_font.digit));
    // a narrow picture: the text needs the whole width
    if (_size.width < 1000)
    {
        _text_panel.x0 = 4.0f;
        _text_panel.x1 = width - 4.0f;
    }
    const float inner = x1 - x0 - 2.0f * gap;
    const float chart_height = 250.0f * u;
    _charts = inner >= 780.0f && _text_panel.y0 - 8.0f * u - chart_height >= 60.0f * u;
    const float top = 8.0f * u;
    _grid_panel = { x0, top, x0 + 0.44f * inner, top + chart_height };
    _pan_panel = { _grid_panel.x1 + gap, top, _grid_panel.x1 + gap + 0.34f * inner, top + chart_height };
    _stairs_panel = { _pan_panel.x1 + gap, top, x1, top + chart_height };
    const float title = static_cast<float>(_small.height) + 14.0f * u;
    const float below = static_cast<float>(_small.height) + 12.0f * u;
    _grid_plot = { _grid_panel.x0 + 40.0f * u, _grid_panel.y0 + title + below, _grid_panel.x1 - 12.0f * u, _grid_panel.y1 - below };
    int legend_width = _text_width("columns that entered", _small) - static_cast<int>(24.0f * u);
    for (const char * const name : kAgeLabels) legend_width = std::max(legend_width, _text_width(name, _small));
    _pan_plot = { _pan_panel.x0 + 36.0f * u, _pan_panel.y0 + title, _pan_panel.x1 - (14.0f + 24.0f + 12.0f) * u - static_cast<float>(legend_width), _pan_panel.y1 - below };
    _stairs_plot = { _stairs_panel.x0 + 44.0f * u, _stairs_panel.y0 + title + below, _stairs_panel.x1 - 12.0f * u, _stairs_panel.y1 - below };

    std::vector<MotionVideoSegment> & out = _static_segments;
    out.clear();
    _static_text.setTo(0);
    auto text = [this](float x, float y, const std::string & words, int align = 1, bool small = true) {
        Label label;
        label.x = static_cast<int>(std::lround(x));
        label.y = static_cast<int>(std::lround(y));
        label.text = words;
        label.small = small;
        label.align = align;
        _put(_static_text, label);
    };

    // the pixel grid is measured above this line
    const float lattice = static_cast<float>(_layout.lattice_rows) * _sy;
    if (_layout.lattice_rows < _layout.height)
    {
        _dashes(out, 0.0f, lattice, width, lattice, kLeaving, 10.0f * u, 8.0f * u);
        text(x1, std::min(lattice, _text_panel.y0) - 7.0f * u, "pixel grid: measured above the dashed line", -1);
    }
    _fill(out, _text_panel, kPanel);
    if (!_charts) return;

    const float label_drop = static_cast<float>(_small.height) + 6.0f * u;
    const float small_half = 0.5f * static_cast<float>(_small.height);
    auto sample = [&](float x, float y, const cv::Vec3b & color, const std::string & words) {
        _line(out, x, y - small_half, x + 18.0f * u, y - small_half, color, 1.5f * u);
        text(x + 24.0f * u, y, words);
        return x + 24.0f * u + static_cast<float>(_text_width(words, _small)) + 18.0f * u;
    };

    // ---- pixel grid
    {
        const Rect & panel = _grid_panel, & plot = _grid_plot;
        _fill(out, panel, kPanel);
        text(panel.x0 + 10.0f * u, panel.y0 + title - 6.0f * u,
             "pixel grid: spectrum of the columns / rows, times its background");
        float x = sample(plot.x0, panel.y0 + title + below - 8.0f * u, kColumns, "columns");
        x = sample(x, panel.y0 + title + below - 8.0f * u, kRows, "rows");
        if (!analyzer.summary().lossless)
        {
            _dashes(out, x, panel.y0 + title + below - 8.0f * u - small_half, x + 18.0f * u, panel.y0 + title + below - 8.0f * u - small_half, kAxis, 3.0f * u, 3.0f * u);
            text(x + 24.0f * u, panel.y0 + title + below - 8.0f * u, "block grid of video encoders");
        }
        for (const double value : { 1.0, 3.0, 10.0, 30.0 })
        {
            const float y = plot.y1 - static_cast<float>((std::log2(value / kGridLow)) / kGridOctaves) * plot.height();
            _line(out, plot.x0, y, plot.x1, y, value == 3.0 ? kAxis : kFaint, _thin);
            text(plot.x0 - 5.0f * u, y + small_half, format("%.0fx", value), -1);
        }
        for (int k = 1; k <= 5; ++k)
        {
            const double frequency = 0.1 * k;
            const float fx = plot.x0 + static_cast<float>((frequency - kGridFrom) / (kGridTo - kGridFrom)) * plot.width();
            _line(out, fx, plot.y1, fx, plot.y1 + 4.0f * u, kAxis, _thin);
            text(fx, plot.y1 + label_drop, format("%.1f", frequency), k == 5 ? -1 : 0);
        }
        if (!analyzer.summary().lossless)
        {
            for (int k = 1; k <= 3; ++k)
            {
                const float fx = plot.x0 + static_cast<float>((k / 8.0 - kGridFrom) / (kGridTo - kGridFrom)) * plot.width();
                _dashes(out, fx, plot.y0, fx, plot.y1, kAxis, 3.0f * u, 5.0f * u);
            }
        }
        _line(out, plot.x0, plot.y1, plot.x1, plot.y1, kAxis, _thin);
        _line(out, plot.x0, plot.y0, plot.x0, plot.y1, kAxis, _thin);
    }

    // ---- pan
    {
        const Rect & panel = _pan_panel, & plot = _pan_plot;
        _fill(out, panel, kPanel);
        text(panel.x0 + 10.0f * u, panel.y0 + title - 6.0f * u, "pan: contrast, entering / leaving columns");
        for (const double value : { 0.5, 1.0, 1.5 })
        {
            const float y = plot.y1 - static_cast<float>(value / kContrastTop) * plot.height();
            _line(out, plot.x0, y, plot.x1, y, value == 1.5 ? kFaint : kAxis, _thin);
            text(plot.x0 - 5.0f * u, y + small_half, format("%.1f", value), -1);
        }
        const double from = ResolutionAnalyzer::bin_frequency(0), to = ResolutionAnalyzer::bin_frequency(kResolutionBins - 1);
        for (int k = 1; k <= 4; ++k)
        {
            const double frequency = 0.1 * k;
            const float fx = plot.x0 + static_cast<float>((frequency - from) / (to - from)) * plot.width();
            _line(out, fx, plot.y1, fx, plot.y1 + 4.0f * u, kAxis, _thin);
            text(fx, plot.y1 + label_drop, format("%.1f", frequency), 0);
        }
        _line(out, plot.x0, plot.y1, plot.x1, plot.y1, kAxis, _thin);
        _line(out, plot.x0, plot.y0, plot.x0, plot.y1, kAxis, _thin);
        // the groups
        const float legend = plot.x1 + 14.0f * u;
        const float row = static_cast<float>(_small.height) + 9.0f * u;
        text(legend, plot.y0 + static_cast<float>(_small.height), "columns that entered");
        for (int age = 0; age < kResolutionAges; ++age)
        {
            sample(legend, plot.y0 + static_cast<float>(_small.height) + row * static_cast<float>(age + 1), kAges[age], kAgeLabels[age]);
        }
    }

    // ---- stair steps
    {
        const Rect & panel = _stairs_panel, & plot = _stairs_plot;
        _fill(out, panel, kPanel);
        text(panel.x0 + 10.0f * u, panel.y0 + title - 6.0f * u, "stair steps by height (pixels)");
        const float x = sample(plot.x0, panel.y0 + title + below - 8.0f * u, kEntering, "entering");
        sample(x, panel.y0 + title + below - 8.0f * u, kLeaving, "leaving");
        for (const double value : { 0.5, 1.0 })
        {
            const float y = plot.y1 - static_cast<float>(value) * plot.height();
            _line(out, plot.x0, y, plot.x1, y, kFaint, _thin);
            text(plot.x0 - 5.0f * u, y + small_half, format("%.0f%%", 100.0 * value), -1);
        }
        for (int step = 1; step <= kResolutionStairHeights; ++step)
        {
            const float fx = plot.x0 + (static_cast<float>(step) - 0.5f) / static_cast<float>(kResolutionStairHeights) * plot.width();
            text(fx, plot.y1 + label_drop, std::to_string(step) + (step == kResolutionStairHeights ? "+" : ""), 0);
        }
        _line(out, plot.x0, plot.y1, plot.x1, plot.y1, kAxis, _thin);
    }
}

void ResolutionVideo::_refresh(const ResolutionAnalyzer & analyzer)
{
    typedef ResolutionAnalyzer::Summary Summary;
    const Summary summary = analyzer.summary();
    const float u = _unit;
    std::vector<MotionVideoSegment> & out = _chart_segments;
    out.clear();
    _chart_labels.clear();
    auto label = [this](float x, float y, const std::string & words, int align) {
        Label entry;
        entry.x = static_cast<int>(std::lround(x));
        entry.y = static_cast<int>(std::lround(y));
        entry.text = words;
        entry.small = true;
        entry.align = align;
        _chart_labels.push_back(entry);
    };

    // ---- the text
    const Summary::Peak * peak = summary.peaks.empty() ? nullptr : &summary.peaks.front();
    const char * const nothing = "-";
    // grid: frequency | strength in the columns / rows | the two scales
    if (peak && peak->axes == Summary::Peak::Both)
    {
        _cells[0] = { format("%.4f", peak->frequency), format("%5.1fx / %5.1fx", peak->strength_x, peak->strength_y),
                      format("%.1f%% or %.1f%%", 100.0 * peak->frequency, 100.0 * (1.0 - peak->frequency)) };
    }
    else if (peak)
    {
        const bool columns = peak->axes == Summary::Peak::Columns;
        _cells[0] = { format("%.4f", peak->frequency), std::string(columns ? "columns only" : "rows only") + format(" %5.1fx", columns ? peak->strength_x : peak->strength_y),
                      format("%.1f%% or %.1f%% (weak)", 100.0 * peak->frequency, 100.0 * (1.0 - peak->frequency)) };
    }
    else if (summary.half_peak)
    {
        _cells[0] = { "0.5", format("%5.1fx / %5.1fx", summary.half_strength_x, summary.half_strength_y), "50%" };
    }
    else
    {
        _cells[0] = { "none", format("%5.1fx / %5.1fx", summary.lattice_strength_x, summary.lattice_strength_y), nothing };
    }

    // pan: where the newest columns are down to half the contrast | what the curve does | the scale
    const bool pan = !summary.contrast[0].empty();
    const long long pan_frames = summary.pan_frames_right + summary.pan_frames_left;
    if (!pan)
    {
        _cells[1] = { nothing, format("%.0f frames", static_cast<double>(pan_frames)), nothing };
    }
    else if (summary.half_contrast_frequency > 0.0)
    {
        _cells[1] = { format("%.3f", summary.half_contrast_frequency), "half contrast",
                      format("%.0f%%", 100.0 * std::min(1.0, 2.0 * summary.half_contrast_frequency)) };
    }
    else if (summary.fan_up)
    {
        _cells[1] = { nothing, format("max %.2f (sharper)", summary.highest_contrast), nothing };
    }
    else
    {
        _cells[1] = { nothing, format("min %.2f", summary.lowest_contrast), nothing };
    }

    // stairs: height of a rendered pixel | steps of 2 pixels and more, entering / leaving | the scale
    if (summary.stair_steps_entering + summary.stair_steps_leaving == 0)
    {
        _cells[2] = { nothing, "none", nothing };
    }
    else
    {
        _cells[2] = { summary.stairs ? format("%.1f px", summary.stair_height) : std::string(nothing),
                      format("%3.0f%% / %3.0f%% are 2+ px", 100.0 * summary.stair_high_entering, 100.0 * summary.stair_high_leaving),
                      summary.stairs ? format("%.0f%%", 100.0 * summary.stair_scale) : std::string(nothing) };
    }

    // result: the candidates
    static const char * const levels[] = { "low", "medium", "high" };
    _cells[3] = { summary.candidates.empty() ? "none" : "", "", "" };
    for (size_t i = 0; i < summary.candidates.size() && i < 4; ++i)
    {
        const Summary::Candidate & candidate = summary.candidates[i];
        _cells[3][0] += (i == 0 ? "" : "     ") + std::to_string(2 * std::llround(candidate.scale * summary.width / 2.0)) + "x"
                      + std::to_string(2 * std::llround(candidate.scale * summary.height / 2.0))
                      + format(" %.1f%% ", 100.0 * candidate.scale) + levels[candidate.confidence];
    }
    if (!_charts) return;

    // ---- pixel grid: the strongest bin of every pixel of the chart, so that a peak of one bin stays
    analyzer.grid_strengths(_columns, _rows);
    {
        const Rect & plot = _grid_plot;
        const int pixels = std::max(2, static_cast<int>(plot.width()));
        auto level = [&plot](double strength) {
            const double octaves = std::log2(std::max(kGridLow, strength) / kGridLow);
            return plot.y1 - static_cast<float>(std::min(1.0, octaves / kGridOctaves)) * plot.height();
        };
        auto curve = [&](const std::vector<double> & strengths, int bins_per_cycle, const cv::Vec3b & color) {
            if (strengths.empty() || bins_per_cycle <= 0) return;
            const int last = static_cast<int>(strengths.size()) - 1;
            float previous_x = 0.0f, previous_y = 0.0f;
            for (int px = 0; px < pixels; ++px)
            {
                const double from = kGridFrom + (kGridTo - kGridFrom) * px / pixels;
                const double to = kGridFrom + (kGridTo - kGridFrom) * (px + 1) / pixels;
                int a = static_cast<int>(std::ceil(from * bins_per_cycle - 1e-9));
                int b = px + 1 == pixels ? last : static_cast<int>(std::ceil(to * bins_per_cycle - 1e-9)) - 1;
                // no bin of its own: the nearest one
                if (b < a) a = b = static_cast<int>(std::lround(0.5 * (from + to) * bins_per_cycle));
                a = std::clamp(a, 0, last);
                b = std::clamp(b, a, last);
                const double strongest = *std::max_element(strengths.begin() + a, strengths.begin() + b + 1);
                const float x = plot.x0 + (static_cast<float>(px) + 0.5f) * plot.width() / static_cast<float>(pixels);
                const float y = level(strongest);
                if (px > 0) _line(out, previous_x, previous_y, x, y, color, _thin);
                previous_x = x;
                previous_y = y;
            }
        };
        curve(_rows, _layout.lattice_rows, kRows);
        curve(_columns, _layout.width, kColumns);
        if (peak && peak->frequency >= kGridFrom)
        {
            const float x = plot.x0 + static_cast<float>((peak->frequency - kGridFrom) / (kGridTo - kGridFrom)) * plot.width();
            const float y = level(std::max(peak->strength_x, peak->strength_y));
            _line(out, x, y, x, y, kWhite, 2.5f * u);
            const bool left = x > plot.x0 + 0.62f * plot.width();
            label(x + (left ? -9.0f : 9.0f) * u, y + 0.5f * static_cast<float>(_small.height),
                  format("%.4f: %.1fx / %.1fx", peak->frequency, peak->strength_x, peak->strength_y), left ? -1 : 1);
        }
        label(plot.x1, _grid_panel.y0 + static_cast<float>(_small.height) + 20.0f * u + static_cast<float>(_small.height) + 4.0f * u,
              format("%.0f frames", static_cast<double>(analyzer.grid_frames())), -1);
    }

    // ---- pan
    {
        const Rect & plot = _pan_plot;
        const double from = ResolutionAnalyzer::bin_frequency(0), to = ResolutionAnalyzer::bin_frequency(kResolutionBins - 1);
        auto at = [&](double frequency) { return plot.x0 + static_cast<float>((frequency - from) / (to - from)) * plot.width(); };
        auto level = [&plot](double contrast) {
            return plot.y1 - static_cast<float>(std::clamp(contrast, 0.0, kContrastTop) / kContrastTop) * plot.height();
        };
        if (!pan)
        {
            label(0.5f * (plot.x0 + plot.x1), 0.5f * (plot.y0 + plot.y1), format("%.0f frames of a pan so far", static_cast<double>(pan_frames)), 0);
        }
        // the newest columns last: on top
        for (int age = kResolutionAges - 1; age >= 0; --age)
        {
            const std::vector<double> & contrast = summary.contrast[static_cast<size_t>(age)];
            for (size_t bin = 1; bin < contrast.size(); ++bin)
            {
                _line(out, at(ResolutionAnalyzer::bin_frequency(static_cast<int>(bin) - 1)), level(contrast[bin - 1]),
                      at(ResolutionAnalyzer::bin_frequency(static_cast<int>(bin))), level(contrast[bin]), kAges[age], age == 0 ? 1.25f * u : _thin);
            }
        }
        if (pan && summary.half_contrast_frequency > 0.0)
        {
            const float x = at(summary.half_contrast_frequency);
            _dashes(out, x, level(0.5), x, plot.y1, kWhite, 3.0f * u, 3.0f * u);
            _line(out, x, level(0.5), x, level(0.5), kWhite, 2.5f * u);
            label(x + 7.0f * u, plot.y1 - 6.0f * u, format("%.3f: %.0f%%", summary.half_contrast_frequency,
                                                         100.0 * std::min(1.0, 2.0 * summary.half_contrast_frequency)), 1);
        }
        label(plot.x1 + 14.0f * u, plot.y1 - 4.0f * u, format("%.0f + %.0f frames", static_cast<double>(summary.pan_frames_right),
                                                              static_cast<double>(summary.pan_frames_left)), 1);
    }

    // ---- stair steps: shares of the steps by their height
    {
        const Rect & plot = _stairs_plot;
        const float slot = plot.width() / static_cast<float>(kResolutionStairHeights);
        const float bar = std::max(1.5f, std::min(6.0f * u, 0.16f * slot));
        const long long totals[2] = { summary.stair_steps_entering, summary.stair_steps_leaving };
        for (int side = 0; side < 2; ++side)
        {
            if (totals[side] == 0) continue;
            const std::array<long long, kResolutionStairHeights + 1> & steps = side == 0 ? summary.stairs_entering : summary.stairs_leaving;
            for (int step = 1; step <= kResolutionStairHeights; ++step)
            {
                const double share = static_cast<double>(steps[static_cast<size_t>(step)]) / static_cast<double>(totals[side]);
                // the round ends of the line stay inside the bar
                const float height = static_cast<float>(share) * plot.height();
                if (height < 2.0f * bar + 1.0f) continue;
                const float x = plot.x0 + (static_cast<float>(step) - 0.5f) * slot + (side == 0 ? -1.25f : 1.25f) * bar;
                _line(out, x, plot.y1 - bar, x, plot.y1 - height + bar, side == 0 ? kEntering : kLeaving, bar);
            }
        }
        label(plot.x1, plot.y0 + static_cast<float>(_small.height), format("%.0f / %.0f steps", static_cast<double>(totals[0]), static_cast<double>(totals[1])), -1);
    }
}

std::string ResolutionVideo::_frame_text(const ResolutionAnalyzer::FrameState & state) const
{
    typedef ResolutionAnalyzer::FrameState FrameState;
    const double dx = state.pan.dx;
    const std::string moved = format("%5.1f px", std::abs(dx));
    switch (state.edges)
    {
    case FrameState::NotMeasured: return "not measured";
    case FrameState::NoMotion: return "no pan: motion unknown";
    case FrameState::TooSlow: return std::abs(dx) < 0.05 ? std::string("no pan: still") : "no pan: " + moved + " (slow)";
    case FrameState::TooFast: return "no pan: " + moved + " (fast)";
    case FrameState::Vertical: return "no pan: up / down";
    case FrameState::Unsteady: return "no pan: not steady";
    case FrameState::Used: break;
    }
    return std::string("pan ") + (dx > 0.0 ? "right" : "left") + " " + moved;
}

void ResolutionVideo::_frame_overlay(const ResolutionAnalyzer::FrameState & state)
{
    typedef ResolutionAnalyzer::FrameState FrameState;
    std::vector<MotionVideoSegment> & out = _frame_segments;
    out.clear();
    _frame_labels.clear();
    if (_layout.edge <= 0 || _layout.segments < 1) return;
    const float u = _unit;
    const float width = static_cast<float>(_size.width);
    const float strip = static_cast<float>(_layout.edge) * _sx;
    const float top = static_cast<float>(_layout.row0) * _sy;
    const float bottom = static_cast<float>(_layout.row0 + (_layout.segments - 1) * kResolutionHop + kResolutionSegment) * _sy;
    const bool used = state.edges == FrameState::Used;
    // the picture enters at the left edge if it moves to the right
    const bool enters_left = state.pan.dx > 0.0f;
    auto label = [this](float x, float y, const std::string & words, int align) {
        Label entry;
        entry.x = static_cast<int>(std::lround(x));
        entry.y = static_cast<int>(std::lround(y));
        entry.text = words;
        entry.small = true;
        entry.align = align;
        _frame_labels.push_back(entry);
    };
    for (int side = 0; side < 2; ++side)
    {
        const bool left = side == 0;
        const bool entering = used && left == enters_left;
        const Rect rect = { left ? 1.0f : width - strip, top, left ? strip : width - 2.0f, bottom };
        _outline(out, rect, !used ? kUnused : entering ? kEntering : kLeaving, used ? 1.0f : _thin);
        label(left ? 6.0f * u : width - 6.0f * u, top - 8.0f * u, !used ? "edge columns" : entering ? "entering" : "leaving", left ? 1 : -1);
        if (!used) continue;
        // the groups of columns, by their distance from the edge
        const float direction = left ? 1.0f : -1.0f;
        const float edge = left ? 0.0f : width - 1.0f;
        for (int age = 0; age < state.ages; ++age)
        {
            const float from = edge + direction * state.bounds[static_cast<size_t>(age)] * _sx;
            const float to = edge + direction * state.bounds[static_cast<size_t>(age) + 1] * _sx;
            const cv::Vec3b color = entering ? kAges[age] : (age % 2 == 0 ? kLeaving : kUnused);
            for (const float y : { top + 5.0f * u, bottom - 5.0f * u })
            {
                // without the round ends, which would reach into the next group
                const float cap = std::min(2.0f * u, 0.45f * std::abs(to - from));
                _line(out, from + direction * cap, y, to - direction * cap, y, color, 2.0f * u);
            }
            if (age + 1 < state.ages) _line(out, to, top + 10.0f * u, to, bottom - 10.0f * u, color, 0.5f);
        }
        if (entering)
        {
            if (state.stair_strip > 0)
            {
                const float end = edge + direction * static_cast<float>(state.stair_strip) * _sx;
                _line(out, edge, bottom + 7.0f * u, end, bottom + 7.0f * u, kWhite, 1.0f);
                label(left ? 6.0f * u : width - 6.0f * u, bottom + 12.0f * u + static_cast<float>(_small.height), "stair steps", left ? 1 : -1);
            }
            // how far the picture moved, four times as long
            const float y = 0.5f * (top + bottom);
            const float from = left ? strip + 14.0f * u : width - strip - 14.0f * u;
            const float length = std::max(8.0f * u, 4.0f * std::abs(state.pan.dx) * _sx);
            const float to = from + direction * length;
            const float tip = std::min(8.0f * u, 0.5f * length);
            _line(out, from, y, to, y, kEntering, 1.25f);
            _line(out, to, y, to - direction * tip, y - tip, kEntering, 1.25f);
            _line(out, to, y, to - direction * tip, y + tip, kEntering, 1.25f);
            label(from, y - 16.0f * u, format("%.1f px per frame", std::abs(state.pan.dx)), left ? 1 : -1);
        }
    }
}

void ResolutionVideo::write(long long frame, bool duplicate, const ResolutionAnalyzer & analyzer, const MotionDetail & detail)
{
    const bool measured = analyzer.layout().width > 0;
    if (measured && !_prepared) _prepare(analyzer);
    if (measured && (_refreshed < 0 || frame - _refreshed >= kRefresh))
    {
        _refresh(analyzer);
        _refreshed = frame;
    }
    // a duplicate keeps what the last new frame had on its picture
    const bool held = duplicate && measured && _has_overlay;
    if (measured && !held)
    {
        _frame_overlay(analyzer.last_frame());
        _has_overlay = true;
    }

    _scene.segments.clear();
    _scene.segments.insert(_scene.segments.end(), _frame_segments.begin(), _frame_segments.end());
    _scene.segments.insert(_scene.segments.end(), _static_segments.begin(), _static_segments.end());
    _scene.segments.insert(_scene.segments.end(), _chart_segments.begin(), _chart_segments.end());

    _static_text.copyTo(_scene.text);
    for (const Label & label : _chart_labels) _put(_scene.text, label);
    for (const Label & label : _frame_labels) _put(_scene.text, label);
    // the text panel, a table: the frame, then what the traces say so far
    const float u = _unit;
    const float line = static_cast<float>(_font.height) + 8.0f * u;
    const float left = (_prepared ? _text_panel.x0 : 0.0f) + 10.0f * u;
    const float first = (_prepared ? _text_panel.y0 : 0.0f) + 5.0f * u + static_cast<float>(_font.height) + 3.0f * u;
    // the columns, in characters of the width of a digit
    const float columns[4] = { 0.0f, 8.0f, 19.0f, 42.0f };
    auto put = [&](int row, int column, const std::string & text) {
        if (text.empty()) return;
        Label label;
        label.x = static_cast<int>(std::lround(left + columns[column] * static_cast<float>(_font.digit)));
        label.y = static_cast<int>(std::lround(first + line * static_cast<float>(row)));
        label.text = text;
        _put(_scene.text, label);
    };
    put(0, 0, "frame");
    put(0, 1, format("%.0f", static_cast<double>(frame)));
    if (!measured)
    {
        put(0, 2, "not measured (the frames are too small)");
    }
    else
    {
        put(0, 2, held ? std::string("duplicate") : _frame_text(analyzer.last_frame()));
        static const char * const names[4] = { "grid", "pan", "stairs", "result" };
        for (int row = 0; row < 4; ++row)
        {
            put(row + 1, 0, names[row]);
            for (int column = 0; column < 3; ++column) put(row + 1, column + 1, _cells[static_cast<size_t>(row)][static_cast<size_t>(column)]);
        }
    }
    _target->write(detail, _scene);
    ++_frames;
}

} // namespace trdrop
