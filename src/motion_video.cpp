#include "motion_video.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "video_writer.h"

namespace trdrop {

namespace {

//! brightness of the picture under the arrows, and of the text box relative to that
constexpr float kDim = 0.6f;
constexpr float kBoxDim = 0.25f;
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

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr int kMargin = 8;
constexpr int kPadding = 10;
constexpr int kLines = 4;

//! the chart under the text: this many seconds, the plot this many lines of text high
constexpr double kChartSeconds = 10.0;
constexpr int kChartLines = 5;
//! narrower plots are left out
constexpr int kChartMinWidth = 200;
// colors of the chart (BGR)
const cv::Vec3b kFramerateColor(90, 220, 90);
const cv::Vec3b kPerceivedColor(255, 170, 70);
const cv::Vec3b kGridColor(85, 85, 85);

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
        if (_dim != scene.dim)
        {
            _dim = scene.dim;
            _dim_table = scale_table(_dim);
        }
        cv::LUT(_picture, _dim_table, _picture);

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
        draw_segments(false);

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
        draw_segments(true);
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
        _writer->write(_picture);
    }

    void finish() override { _writer->finish(); }

private:
    cv::Size _size;
    std::unique_ptr<VideoWriter> _writer;
    cv::Mat _picture;
    float _dim = -1.0f;
    float _box_dim = -1.0f;
    cv::Mat _dim_table;
    cv::Mat _box_table;
};

} // namespace

std::unique_ptr<MotionVideoTarget> make_cpu_motion_video_target(const std::string & path, cv::Size size, double fps)
{
    return std::make_unique<CpuMotionVideoTarget>(path, size, fps);
}

MotionVideo::MotionVideo(const std::string & path, int frame_width, int frame_height, double fps, double arrow_scale,
                         std::unique_ptr<MotionVideoTarget> target)
    : _path(path)
    , _size(motion_picture_size(frame_width, frame_height))
    , _arrow_scale(arrow_scale)
    , _target(std::move(target))
{
    if (_size.width < 2 || _size.height < 2) throw std::runtime_error("the video is too small for a motion video");
    if (!_target) _target = make_cpu_motion_video_target(path, _size, fps);

    cv::Mat ramp(256, 1, CV_8UC1), colors;
    for (int i = 0; i < 256; ++i) ramp.at<uchar>(i) = static_cast<uchar>(i);
    cv::applyColorMap(ramp, colors, cv::COLORMAP_TURBO);
    for (int i = 0; i < 256; ++i) _colors.push_back(colors.at<cv::Vec3b>(i));

    // the text: smaller for small pictures, every digit gets the width of the widest one
    _font_scale = std::clamp(0.6 * _size.width / 1280.0, 0.4, 0.6);
    int baseline = 0;
    for (char digit = '0'; digit <= '9'; ++digit)
    {
        _digit_width = std::max(_digit_width, cv::getTextSize(std::string(1, digit), kFont, _font_scale, 1, &baseline).width);
    }
    _space = _digit_width;
    _line_height = static_cast<int>(std::lround(42.0 * _font_scale));
    for (const char * state : { "new", "duplicate", "torn", "first frame" })
    {
        _state_width = std::max(_state_width, cv::getTextSize(state, kFont, _font_scale, 1, &baseline).width);
    }
    // the box holds the longest text there can be
    MotionVideoFrame widest;
    widest.compared = true;
    widest.sample.measurable = true;
    widest.sample.has_error = true;
    widest.sample.has_perceived_fps = true;
    widest.sample.lost = 1;
    const cv::Size box(_text(nullptr, widest, 0, 0, true) + kPadding, kLines * _line_height + kPadding);
    _scene.dim = kDim;
    _scene.box_dim = kBoxDim;
    auto fits = [&](const cv::Size & size) { return size.width <= _size.width - 2 * kMargin && size.height <= _size.height - 2 * kMargin; };
    // no text on pictures it doesn't fit on
    if (!fits(box)) return;
    _scene.box = cv::Rect(cv::Point(kMargin, kMargin), box);

    // the chart: a line for the legend, the plot with the values at its left, room for the label at its bottom
    const int label_width = cv::getTextSize("480", kFont, _font_scale, 1, &baseline).width + _space;
    const int digit_height = cv::getTextSize("0", kFont, _font_scale, 1, &baseline).height;
    const cv::Size with_chart(box.width, box.height + (kChartLines + 1) * _line_height + kPadding);
    const int plot_width = box.width - 2 * kPadding - label_width;
    if (fits(with_chart) && plot_width >= kChartMinWidth)
    {
        _chart = true;
        _scene.box.height = with_chart.height;
        _chart_top = chart_top(fps);
        _chart_frames = static_cast<size_t>(std::max(2.0, std::round(kChartSeconds * fps)));
        const float x0 = static_cast<float>(kMargin + kPadding + label_width);
        const float y0 = static_cast<float>(kMargin + box.height + _line_height + digit_height / 2);
        _plot = cv::Rect2f(x0, y0, static_cast<float>(plot_width), static_cast<float>(kChartLines * _line_height - digit_height));
        _chart_text = cv::Mat::zeros(_scene.box.size(), CV_8UC1);

        const float thin = _size.width >= 1280 ? 0.5f : 0.4f;
        const float thick = _size.width >= 1280 ? 1.0f : 0.75f;
        // the grid: 0, 25, 50, 75, 100% of the top, the values at 0, 50 and 100%
        for (int k = 0; k <= 4; ++k)
        {
            const float y = _plot.y + _plot.height * (1.0f - 0.25f * static_cast<float>(k));
            _chart_segments.push_back(chart_segment(_plot.x, y, _plot.x + _plot.width, y, thin, kGridColor));
            if (k % 2 != 0) continue;
            const std::string value = format("%g", _chart_top * 0.25 * k);
            const int width = cv::getTextSize(value, kFont, _font_scale, 1, &baseline).width;
            const cv::Point at(static_cast<int>(_plot.x) - _space / 2 - width - kMargin, static_cast<int>(std::lround(y)) + digit_height / 2 - kMargin);
            cv::putText(_chart_text, value, at, kFont, _font_scale, cv::Scalar(255), 1, cv::LINE_AA);
        }
        // the legend: framerate --- perceived smoothness ---   last 10 s
        const int line = kLines;
        const float sample_y = static_cast<float>(kMargin + kPadding / 2 + _line_height * (line + 1) - _line_height / 4 - digit_height / 2);
        const float sample_length = 3.0f * static_cast<float>(_digit_width);
        int x = _label(&_chart_text, kPadding, line, "framerate") + _space;
        _chart_segments.push_back(chart_segment(static_cast<float>(kMargin + x), sample_y, static_cast<float>(kMargin + x) + sample_length, sample_y, thick, kFramerateColor));
        x += static_cast<int>(sample_length) + 2 * _space;
        x = _label(&_chart_text, x, line, "perceived smoothness") + _space;
        _chart_segments.push_back(chart_segment(static_cast<float>(kMargin + x), sample_y, static_cast<float>(kMargin + x) + sample_length, sample_y, thick, kPerceivedColor));
        x += static_cast<int>(sample_length) + 2 * _space;
        _label(&_chart_text, x, line, "fps, last " + format("%g", kChartSeconds) + " s");
    }
    _scene.text.create(_scene.box.size(), CV_8UC1);
}

void MotionVideo::_add_chart()
{
    _scene.segments.insert(_scene.segments.end(), _chart_segments.begin(), _chart_segments.end());
    // the framerate wider, under the perceived smoothness: where they are the same it shows at both sides
    const float wide = _size.width >= 1280 ? 1.75f : 1.25f;
    const float narrow = _size.width >= 1280 ? 0.75f : 0.5f;
    const float right = _plot.x + _plot.width;
    const float step = _plot.width / static_cast<float>(_chart_frames - 1);
    const size_t count = _history.size();
    for (const bool perceived : { false, true })
    {
        const cv::Vec3b & color = perceived ? kPerceivedColor : kFramerateColor;
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
            const float value = perceived ? _history[k].second : _history[k].first;
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

int MotionVideo::_label(cv::Mat * target, int x, int line, const std::string & text) const
{
    int baseline = 0;
    const int width = cv::getTextSize(text, kFont, _font_scale, 1, &baseline).width;
    if (target)
    {
        const int y = kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        cv::putText(*target, text, cv::Point(x, y), kFont, _font_scale, cv::Scalar(255), 1, cv::LINE_AA);
    }
    return x + width;
}

int MotionVideo::_number(cv::Mat * target, int x, int line, int cells, const std::string & text) const
{
    const int end = x + cells * _digit_width;
    if (target)
    {
        const int y = kPadding / 2 + _line_height * (line + 1) - _line_height / 4;
        int baseline = 0;
        for (size_t k = 0; k < text.size(); ++k)
        {
            // from the right: the last character is in the last cell
            const std::string character(1, text[text.size() - 1 - k]);
            const int width = cv::getTextSize(character, kFont, _font_scale, 1, &baseline).width;
            const int cell = end - static_cast<int>(k + 1) * _digit_width;
            cv::putText(*target, character, cv::Point(cell + (_digit_width - width) / 2, y), kFont, _font_scale, cv::Scalar(255), 1, cv::LINE_AA);
        }
    }
    return end;
}

int MotionVideo::_text(cv::Mat * target, const MotionVideoFrame & values, int used, int rejected, bool counted) const
{
    const int left = kPadding;
    int widest = 0;

    // frame 1234   new   2104 points used   37 rejected   (arrows x8)
    int x = _label(target, left, 0, "frame");
    x = _number(target, x, 0, 8, std::to_string(values.frame)) + 2 * _space;
    _label(target, x, 0, !values.compared ? "first frame" : values.duplicate ? "duplicate" : values.torn ? "torn" : "new");
    x += _state_width + _space;
    x = _number(target, x, 0, 5, counted ? std::to_string(used) : "") + _space;
    x = _label(counted ? target : nullptr, x, 0, "points used") + _space;
    x = _number(target, x, 0, 5, counted ? std::to_string(rejected) : "") + _space;
    x = _label(counted ? target : nullptr, x, 0, "rejected");
    if (_arrow_scale != 1.0) x = _label(target, x + 2 * _space, 0, "arrows x" + format("%g", _arrow_scale));
    widest = std::max(widest, x);

    // animation step 1.05 intervals   votes spread 0.26
    x = _label(target, left, 1, "animation step");
    if (values.sample.measurable)
    {
        x = _number(target, x, 1, 7, format("%.2f", values.sample.step)) + _space;
        x = _label(target, x, 1, "intervals") + 2 * _space;
        x = _label(target, x, 1, "votes spread");
        x = _number(target, x, 1, 6, format("%.2f", values.sample.step_spread));
    }
    else
    {
        x = _label(target, x + 2 * _space, 1, "not measurable");
    }
    widest = std::max(widest, x);

    // animation error +0.4 ms   1 picture lost by the capture
    x = _label(target, left, 2, "animation error");
    x = _number(target, x, 2, 7, values.sample.has_error ? format("%+.1f", values.sample.error_ms) : "") + _space;
    x = _label(values.sample.has_error ? target : nullptr, x, 2, "ms");
    if (values.sample.lost > 0) x = _label(target, x + 2 * _space, 2, "1 picture lost by the capture");
    widest = std::max(widest, x);

    // framerate 119.0 fps   perceived smoothness 118.6 fps
    x = _label(target, left, 3, "framerate");
    x = _number(target, x, 3, 6, format("%.1f", values.framerate)) + _space;
    x = _label(target, x, 3, "fps") + 2 * _space;
    x = _label(target, x, 3, "perceived smoothness");
    x = _number(target, x, 3, 6, values.sample.has_perceived_fps ? format("%.1f", values.sample.perceived_fps) : "") + _space;
    x = _label(values.sample.has_perceived_fps ? target : nullptr, x, 3, "fps");
    widest = std::max(widest, x);
    return widest;
}

void MotionVideo::_add_points(const std::vector<MotionPoint> & points, bool dimmed)
{
    const float thickness = _size.width >= 1280 ? 2.0f : 1.0f;
    for (const MotionPoint & point : points)
    {
        if (point.state == MotionPoint::Rejected)
        {
            MotionVideoSegment arm;
            arm.radius = 0.5f;
            arm.blue = 40;
            arm.green = 40;
            arm.red = 230;
            arm.x0 = point.x - kCrossSize;
            arm.y0 = point.y - kCrossSize;
            arm.x1 = point.x + kCrossSize;
            arm.y1 = point.y + kCrossSize;
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
        segment.x0 = segment.x1 = point.x;
        segment.y0 = segment.y1 = point.y;
        const double drawn = length * _arrow_scale;
        if (drawn < kShortestArrow)
        {
            segment.radius = thickness;
            _scene.segments.push_back(segment);
            continue;
        }
        segment.radius = 0.5f * thickness;
        segment.x1 = static_cast<float>(point.x + point.dx * _arrow_scale);
        segment.y1 = static_cast<float>(point.y + point.dy * _arrow_scale);
        _scene.segments.push_back(segment);
        // the two lines of the tip, back from the end of the arrow
        const double tip = std::min(0.5 * drawn, kTipLength);
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
        for (const MotionPoint & point : detail.points)
        {
            used += point.state == MotionPoint::Used;
            rejected += point.state == MotionPoint::Rejected;
        }
    }
    else
    {
        _held.clear();
    }
    if (_chart)
    {
        _history.emplace_back(static_cast<float>(values.framerate),
                              values.sample.has_perceived_fps ? static_cast<float>(values.sample.perceived_fps) : std::nanf(""));
        while (_history.size() > _chart_frames) _history.pop_front();
        _add_chart();
    }
    if (!_scene.box.empty())
    {
        if (_chart) _chart_text.copyTo(_scene.text);
        else _scene.text.setTo(0);
        _text(&_scene.text, values, used, rejected, counted);
    }
    _target->write(detail, _scene);
    ++_frames;
}

} // namespace trdrop
