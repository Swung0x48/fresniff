// trdrop++ (trdrop-pp): analyzes captured video and writes the per frame framerate / frametime / tear
// data and, with --motion, how evenly the animation advances.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "clock_skew.h"
#include "diff_engine.h"
#include "frame_analyzer.h"
#include "frame_pipeline.h"
#include "motion_video.h"
#include "resolution_video.h"
#include "trdrop_library.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "options.h"

namespace trdrop {
namespace {

constexpr const char * kVersion = "1.1.1";

void print_usage(std::ostream & out)
{
    out <<
R"(trdrop++ - raw video analysis (framerate, frametime and tear detection)

usage: trdrop-pp [options] <video> [<video> [<video>]]

Compares up to 3 videos frame by frame and writes one CSV row per frame.

output:
  -o, --output <file>          per frame CSV (default: stdout)
      --tears <file>           additional CSV with one row per detected tear
      --no-summary             don't print the summary to stderr
  -q, --quiet                  no progress output

analysis (single value for all videos, or comma separated per video, e.g. 0,5,0):
      --pixel-diff <0-255>     grayscale difference a pixel may have without counting as
                               different (default: 0, lossy codecs: 4)
      --tolerance-pct <%>      for lossy codecs (H.264, HEVC, ...), whose duplicate frames differ
                               by compression noise: a frame only counts as new if more than
                               this percentage of its pixels is different, a row with at most
                               this percentage counts as unchanged
                               (default: 0; lossy codecs: automatic, at least 0.5 - less, down
                               to 0.05, where the duplicates of the recording change less -
                               and 5% of what new frames changed in the last 2 seconds)
      --keyframe-tolerance <x> with the automatic tolerance: an encoder codes a duplicate anew at
                               a keyframe, which can change a few % of the picture. A keyframe
                               only counts as new if it changed more than x times the least of
                               the last 3 new frames (0 - 1, 0: off; default: 0.5)
      --dismiss-tear-pct <%>   tears smaller than this percentage of the frame height are
                               dismissed (default: 10)
      --source-fps <fps>       override the recorded framerate of the container
      --lossless               the videos were recorded lossless (e.g. HEVC lossless) and never
                               coded lossy before, whatever the codec could do: a duplicate
                               frame is identical (pixel-diff and tolerance-pct 0, as for
                               lossless codecs), and --resolution reads the peaks on the block
                               grid of encoders, 25% and 75% at 0.25, as a pixel grid
      --max-frames <n>         stop after n frames

motion (smoothness of the animation, needs coherent motion like a moving camera):
      --motion                 measure the motion between the frames and how evenly the
                               animation advances: animation step, animation error and the
                               perceived smoothness. On the GPU with Vulkan Video decoding,
                               otherwise on the CPU
      --motion-window <ms>     the motion of a frame is compared to the average motion in
                               this window around it (default: 400, growing up to 1000 until
                               it holds 10 new frames)
      --motion-video <file>    --motion, and write a copy of the video with a frosted-glass
                               framerate and perceived smoothness overlay and 10 s chart at
                               the bottom. At the resolution the motion is measured at (up
                               to 1920x1080, or 1080x1920 for a video higher than wide),
                               H.264 in the container the name asks for (.mp4, .mkv, .mov).
                               One file per input video, comma separated. With Vulkan Video
                               decoding it is drawn and encoded on the GPU, otherwise on
                               the CPU
      --motion-video-native    the motion video at the resolution of the input video (4K
                               stays 4K) instead of the motion's; the analysis is the same.
                               Larger files, more memory for the pictures that wait; not
                               together with --resolution-video
      --motion-arrows          draw optical-flow arrows (colored by length; dimmed on
                               duplicate frames) and red x rejected matches on the motion
                               video, slightly dimming the picture underneath (off by
                               default)
      --arrow-scale <x>        draw the arrows of the motion video x times longer (implies
                               --motion-arrows; default: 1, the real motion; e.g. 8 for
                               slow motion)
      --async-capture          --motion for a capture that didn't run in step with the game's
                               output (variable refresh rate, or a capture program with its
                               own clock): a picture may be recorded an interval late, which
                               looks like a repeated frame, and the one that makes up for it
                               looks like a jump. The display time of a picture then counts
                               as known to one interval only - the animation error is what
                               that can't explain - and a step twice as long as those around
                               it, after a picture that was recorded once, counts as two
                               pictures of which the capture lost one (while that happens,
                               also after a picture that was recorded twice).
                               A game that really shows a picture an interval late looks the
                               same: use it only if you know the capture is not in step
      --ignore-clock-skew      filter what a 59.94 Hz source recorded at 60 fps (or 60 Hz at
                               59.94, or at a multiple) does every ~1001 frames (16.7 s): a
                               few seconds of pictures recorded alternately a tick early and
                               late, and an interval too many. Only episodes with that
                               pattern which lie on one lattice of that period for the whole
                               video (and whose motion agrees, with --motion) are filtered:
                               vN_fps, vN_frametime_ms, the animation error and the perceived
                               smoothness count them as shown for their cadence. Anything
                               else stays. Needs a game at half the recording's framerate or
                               less (a 60 fps game at 60 fps has nothing to filter). The CSV
                               is written at the end; the motion video shows the values as
                               recorded

resolution (render resolution of an upscaled picture):
      --resolution             --motion, and estimate the resolution the picture was rendered at
                               before it was scaled up, from three traces:
                               the pixel grid - a scaled picture repeats with the source sampling
                               rate: a peak at r cycles per pixel for a render scale of r or
                               1 - r (25% and 75% fall on the block grid of video encoders and
                               can't be told from it, except with --lossless; 50% leaves a
                               peak at 0.5);
                               the picture that enters at the edge of a horizontal pan - a
                               temporal upscaler has no earlier frames for it. With DLSS at
                               33% it showed what a single rendered frame resolves and picked
                               the right scale of the two; with TSR, FSR, DLSS 4.5 and others
                               not. That needs a steady pan of 4 - 64 pixels per frame;
                               the stair steps of that entering picture - an upscaler that
                               shows new pixels as they were rendered (XeSS, FSR, TSR) shows
                               nearly level edges climbing one rendered pixel at a time: steps
                               of 3 output pixels are a render scale of 33%. Needs straight
                               edges in the picture, and says nothing above about 60%.
                               The summary lists the render resolutions that fit, each with a
                               confidence - a level of the rules, not a probability: high (a
                               grid in both axes and the pan or the stair steps pick the
                               scale), medium (a grid in both axes; with --lossless the scale
                               its second harmonic picks; with an upward fan - the entering
                               picture has more contrast than the settled one - its larger
                               scale), low (a grid in one axis, the peak at 0.5, the pan
                               alone, the scale the pan or the fan speaks against).
                               The recording has to have the resolution the game put out and a
                               constant quality (QP 16 or better) or none lost. Checked against
                               32 recordings with a known setting; a scale that changes is not
                               handled. On the GPU with Vulkan Video decoding, otherwise on
                               the CPU
      --resolution-data <file> --resolution, and write what the summary is made of as CSV: the
                               spectra of the pixel grid and of the picture entering / leaving
                               in a pan (comma separated list, one file per video; plot it
                               with scripts/resolution.py to judge a recording yourself)
      --resolution-video <file>
                               --resolution, and write a copy of the video that shows how the
                               estimate is made, frame by frame: what is measured in the
                               picture (the part for the pixel grid, the edge columns of a pan
                               and their groups), the three traces as charts of what was
                               collected up to the frame, and what they say so far. Size,
                               codec and files as for --motion-video
      --resolution-grid-rows <a-b>
                               --resolution, and the part of the height, in percent from the top,
                               the pixel grid is measured in (default: 0-88, without the bottom
                               of the picture, where overlays tend to be; one value for all
                               videos or comma separated per video). A smaller part read less
                               in the recordings it was tried on: leave out no more than a HUD
                               that fills the width
      --resolution-pan-rows <a-b>
                               --resolution, and the part of the height whose edge columns are
                               measured in a pan, for the picture that enters and for the
                               stair steps (default: 9-74; per video as above). It should
                               hold no HUD that touches the left or the right edge - that is
                               drawn at the output resolution and doesn't move with the picture

backend:
      --vkdevice <n>           Vulkan device used for the per pixel work (default: the first
                               discrete GPU, otherwise device 0)
      --no-vk                  don't use Vulkan, analyze with OpenCV on the CPU
      --list-vkdevices         list the Vulkan devices and exit
      --decode <mode>          auto (default): the fastest of the modes below for the videos
                               (vulkan for HEVC / AV1 / ..., ffmpeg for H.264 on CPUs with
                               16+ threads and for YUV formats without Vulkan decoding,
                               opencv for RGB sources), the choice is printed
                               vulkan: decode on the GPU (FFmpeg Vulkan Video), compare the
                                 Y, U and V planes on the GPU, frames never leave the GPU
                               ffmpeg: decode with FFmpeg on the CPU, compare the Y, U and V
                                 planes on the CPU (same results as vulkan)
                               opencv: decode with OpenCV, compare BGR -> grayscale (also
                                 used with --no-vk)
                               YUV and grayscale differ slightly, so the results of
                               vulkan / ffmpeg differ slightly from opencv

  -h, --help                   show this help
  -v, --version                show the version

per frame columns (vN = N-th input video):
  frame            frame index, starting at 0
  vN_diff          difference to the previous frame: 0 = duplicate, 1 = new frame,
                   (0,1) = torn frame (1 - torn part); empty for frame 0
  vN_fps           sum of vN_diff over the last second (ceil(recorded fps) frames)
  vN_frametime_ms  how long the last completed frame was shown
  vN_tears         amount of tears detected in this frame

additional per frame columns with --motion:
  vN_anim_step      how far the animation advanced since the previous frame, in display
                    intervals (1 = the average around the frame); empty without usable motion
  vN_anim_step_spread  how much the moving parts of the picture disagree about the step (in
                    intervals, 0 = rigid motion); smaller differences between steps say little
  vN_anim_error_ms  new frames: animation step since the previous new frame minus the time
                    that frame was displayed
  vN_perceived_fps  perceived smoothness over the last second: the framerate of evenly paced
                    motion that looks as smooth, new frames per second / (1 + cv(step)^2)
  vN_lost           only with --async-capture, new frames: pictures the capture lost between
                    the previous new frame and this one (they count as frames in
                    vN_perceived_fps, vN_fps doesn't know them)

tear columns:
  frame, video, row_a, row_b, video_height   (the tear is between row_a and row_b)
)";
#ifndef USE_VULKAN
    out << "\nthis build has no Vulkan support (configure with -DUSE_VULKAN=ON), --no-vk is accepted and ignored\n";
#endif
#ifndef USE_FFMPEG
    out << "\nthis build has no FFmpeg support (configure with -DUSE_FFMPEG=ON), only --decode opencv is available\n";
#endif
}

std::vector<std::string> split(const std::string & text, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, delimiter)) parts.push_back(part);
    if (!text.empty() && text.back() == delimiter) parts.push_back("");
    return parts;
}

double parse_double(const std::string & text, const std::string & option)
{
    try
    {
        size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed == text.size() && std::isfinite(value)) return value;
    }
    catch (const std::exception &) { }
    throw std::invalid_argument("invalid value '" + text + "' for " + option);
}

long long parse_integer(const std::string & text, const std::string & option)
{
    try
    {
        size_t consumed = 0;
        const long long value = std::stoll(text, &consumed);
        if (consumed == text.size()) return value;
    }
    catch (const std::exception &) { }
    throw std::invalid_argument("invalid value '" + text + "' for " + option);
}

//! a value list for the per video options, either a single value or one per video
struct PerVideoValue {
    std::string option;
    std::vector<std::string> values;
};

template <typename Setter>
void apply_per_video(const PerVideoValue & value, std::vector<VideoOptions> & videos, Setter setter)
{
    if (value.values.empty()) return;
    if (value.values.size() != 1 && value.values.size() != videos.size())
    {
        throw std::invalid_argument(value.option + " expects 1 or " + std::to_string(videos.size()) + " comma separated values");
    }
    for (size_t i = 0; i < videos.size(); ++i)
    {
        const std::string & text = value.values.size() == 1 ? value.values[0] : value.values[i];
        if (!text.empty()) setter(videos[i], text);
    }
}

//! returns false if the program should exit without analyzing (help / version)
bool parse_arguments(int argc, char ** argv, Options & options)
{
    PerVideoValue pixel_diff { "--pixel-diff", {} };
    PerVideoValue dismiss    { "--dismiss-tear-pct", {} };
    PerVideoValue tolerance  { "--tolerance-pct", {} };
    PerVideoValue keyframe_tolerance { "--keyframe-tolerance", {} };
    PerVideoValue source_fps { "--source-fps", {} };
    PerVideoValue resolution_grid_rows { "--resolution-grid-rows", {} };
    PerVideoValue resolution_pan_rows { "--resolution-pan-rows", {} };
    bool lossless = false;

    bool only_inputs = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (only_inputs || arg.empty() || arg[0] != '-' || arg == "-")
        {
            options.inputs.push_back(arg);
        }
        else if (arg == "--") { only_inputs = true; }
        else if (arg == "-h" || arg == "--help") { print_usage(std::cout); return false; }
        else if (arg == "-v" || arg == "--version") { std::cout << "trdrop++ " << kVersion << "\n"; return false; }
        else if (arg == "-o" || arg == "--output") { options.output_path = next(); }
        else if (arg == "--tears") { options.tears_path = next(); }
        else if (arg == "--no-summary") { options.print_summary = false; }
        else if (arg == "-q" || arg == "--quiet") { options.quiet = true; }
        else if (arg == "--pixel-diff") { pixel_diff.values = split(next(), ','); }
        else if (arg == "--dismiss-tear-pct") { dismiss.values = split(next(), ','); }
        else if (arg == "--tolerance-pct") { tolerance.values = split(next(), ','); }
        else if (arg == "--keyframe-tolerance") { keyframe_tolerance.values = split(next(), ','); }
        else if (arg == "--source-fps") { source_fps.values = split(next(), ','); }
        else if (arg == "--no-vk") { options.use_vulkan = false; }
        else if (arg == "--motion") { options.motion = true; }
        else if (arg == "--async-capture") { options.motion = true; options.async_capture = true; }
        else if (arg == "--ignore-clock-skew") { options.ignore_clock_skew = true; }
        else if (arg == "--resolution") { options.motion = true; options.resolution = true; }
        else if (arg == "--lossless") { lossless = true; }
        else if (arg == "--resolution-pan-rows")
        {
            options.motion = true;
            options.resolution = true;
            resolution_pan_rows.values = split(next(), ',');
        }
        else if (arg == "--resolution-rows")
        {
            throw std::invalid_argument("--resolution-rows is now --resolution-pan-rows (the pixel grid has --resolution-grid-rows)");
        }
        else if (arg == "--resolution-grid-rows")
        {
            options.motion = true;
            options.resolution = true;
            resolution_grid_rows.values = split(next(), ',');
        }
        else if (arg == "--motion-video") { options.motion = true; options.motion_videos = split(next(), ','); }
        else if (arg == "--motion-video-native") { options.motion_video_native = true; }
        else if (arg == "--motion-arrows") { options.motion_arrows = true; }
        else if (arg == "--resolution-data")
        {
            options.motion = true;
            options.resolution = true;
            options.resolution_data = split(next(), ',');
        }
        else if (arg == "--resolution-video")
        {
            options.motion = true;
            options.resolution = true;
            options.resolution_videos = split(next(), ',');
        }
        else if (arg == "--arrow-scale")
        {
            options.arrow_scale = parse_double(next(), arg);
            options.motion_arrows = true;
            if (options.arrow_scale < 0.1 || options.arrow_scale > 100.0) throw std::invalid_argument("--arrow-scale must be between 0.1 and 100");
        }
        else if (arg == "--motion-window")
        {
            options.motion_window_ms = parse_double(next(), arg);
            options.motion = true;
            if (options.motion_window_ms < 50.0 || options.motion_window_ms > 5000.0)
            {
                throw std::invalid_argument("--motion-window must be between 50 and 5000 ms");
            }
        }
        else if (arg == "--decode")
        {
            options.decode = next();
            if (options.decode != "auto" && options.decode != "vulkan" && options.decode != "ffmpeg" && options.decode != "opencv")
            {
                throw std::invalid_argument("--decode expects auto, vulkan, ffmpeg or opencv");
            }
#ifndef USE_FFMPEG
            if (options.decode == "vulkan" || options.decode == "ffmpeg")
            {
                throw std::invalid_argument("--decode " + options.decode + ": this build has no FFmpeg support (configure with -DUSE_FFMPEG=ON)");
            }
#endif
#if !defined(USE_VULKAN) && !defined(TRDROP_MEDIACODEC_VULKAN)
            if (options.decode == "vulkan") throw std::invalid_argument("--decode vulkan: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#endif
        }
        else if (arg == "--vkdevice")
        {
            const long long value = parse_integer(next(), arg);
#ifndef USE_VULKAN
            (void)value;
            throw std::invalid_argument("--vkdevice: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#else
            if (value < 0 || value > 1000) throw std::invalid_argument("--vkdevice must be >= 0");
            options.vk_device = static_cast<int>(value);
#endif
        }
        else if (arg == "--list-vkdevices")
        {
#ifndef USE_VULKAN
            throw std::invalid_argument("--list-vkdevices: this build has no Vulkan support (configure with -DUSE_VULKAN=ON)");
#else
            const std::vector<VulkanDeviceInfo> devices = list_vulkan_devices();
            if (devices.empty()) std::cout << "no Vulkan devices found\n";
            for (size_t d = 0; d < devices.size(); ++d)
            {
                std::cout << d << ": " << devices[d].name << " (" << devices[d].type << ")\n";
            }
            return false;
#endif
        }
        else if (arg == "--max-frames")
        {
            options.max_frames = parse_integer(next(), arg);
            if (options.max_frames < 0) throw std::invalid_argument("--max-frames must be >= 0");
        }
        else throw std::invalid_argument("unknown option '" + arg + "' (see --help)");
    }

    if (!options.use_vulkan && options.vk_device >= 0) throw std::invalid_argument("--vkdevice and --no-vk can't be combined");
    if (!options.use_vulkan && options.decode == "vulkan") throw std::invalid_argument("--decode vulkan and --no-vk can't be combined");
    if (options.inputs.empty()) throw std::invalid_argument("no input video given (see --help)");
    if (options.inputs.size() > static_cast<size_t>(kMaxVideoCount))
    {
        throw std::invalid_argument("at most " + std::to_string(kMaxVideoCount) + " videos can be compared");
    }
    options.videos.assign(options.inputs.size(), VideoOptions());
    for (VideoOptions & v : options.videos) v.lossless = lossless;
    if (options.motion_videos.size() > options.inputs.size())
    {
        throw std::invalid_argument("--motion-video: more files than videos");
    }
    if (options.resolution_data.size() > options.inputs.size())
    {
        throw std::invalid_argument("--resolution-data: more files than videos");
    }
    if (options.resolution_videos.size() > options.inputs.size())
    {
        throw std::invalid_argument("--resolution-video: more files than videos");
    }
    if (options.motion_video_native)
    {
        bool any = false;
        for (const std::string & path : options.motion_videos) any |= !path.empty();
        if (!any) throw std::invalid_argument("--motion-video-native needs --motion-video");
        // the pictures of every input are made at its resolution, the resolution video is drawn at the motion's
        for (const std::string & path : options.resolution_videos)
        {
            if (!path.empty()) throw std::invalid_argument("--motion-video-native: not together with --resolution-video");
        }
    }

    apply_per_video(pixel_diff, options.videos, [](VideoOptions & v, const std::string & text) {
        const long long value = parse_integer(text, "--pixel-diff");
        if (value < 0 || value > 255) throw std::invalid_argument("--pixel-diff must be between 0 and 255");
        v.pixel_difference = static_cast<int>(value);
        v.pixel_difference_set = true;
    });
    apply_per_video(tolerance, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--tolerance-pct");
        if (value < 0.0 || value >= 100.0) throw std::invalid_argument("--tolerance-pct must be >= 0 and < 100");
        v.tolerance_pct = value;
    });
    apply_per_video(keyframe_tolerance, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--keyframe-tolerance");
        if (value < 0.0 || value > 1.0) throw std::invalid_argument("--keyframe-tolerance must be between 0 and 1");
        v.keyframe_tolerance = value;
    });
    apply_per_video(dismiss, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--dismiss-tear-pct");
        if (value < 0.0 || value > 100.0) throw std::invalid_argument("--dismiss-tear-pct must be between 0 and 100");
        v.dismiss_tear_percentage = value;
    });
    apply_per_video(source_fps, options.videos, [](VideoOptions & v, const std::string & text) {
        const double value = parse_double(text, "--source-fps");
        if (value <= 0.0) throw std::invalid_argument("--source-fps must be > 0");
        v.source_fps_override = value;
    });
    // a part of the height: <from>-<to> in percent
    auto parse_rows = [](const std::string & text, const std::string & option, const char * example, double & from, double & to) {
        const size_t dash = text.find('-');
        if (dash == std::string::npos || dash == 0) throw std::invalid_argument(option + " expects <from>-<to> in percent of the height, e.g. " + example);
        const double a = parse_double(text.substr(0, dash), option.c_str());
        const double b = parse_double(text.substr(dash + 1), option.c_str());
        if (a < 0.0 || b > 100.0 || b - a < 5.0) throw std::invalid_argument(option + ": 0 <= from < to <= 100, at least 5 apart");
        from = a;
        to = b;
    };
    apply_per_video(resolution_grid_rows, options.videos, [&](VideoOptions & v, const std::string & text) {
        parse_rows(text, "--resolution-grid-rows", "0-88", v.resolution_grid_from, v.resolution_grid_to);
    });
    apply_per_video(resolution_pan_rows, options.videos, [&](VideoOptions & v, const std::string & text) {
        parse_rows(text, "--resolution-pan-rows", "9-74", v.resolution_pan_from, v.resolution_pan_to);
    });
    return true;
}

std::string format_number(double value)
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

//! per video statistics for the summary
struct VideoStats {
    long long compared_frames = 0;
    long long new_frames = 0;
    long long duplicate_frames = 0;
    long long torn_frames = 0;
    long long tear_count = 0;
    double diff_sum = 0.0;
    // rolling fps / frametime after the first full window
    long long window_samples = 0;
    double fps_sum = 0.0;
    double fps_min = std::numeric_limits<double>::infinity();
    double fps_max = 0.0;
    long long frametime_samples = 0;
    double frametime_sum = 0.0;
    double frametime_max = 0.0;
};

//! --ignore-clock-skew: what a row holds of a video, kept until the whole video is known
struct StoredValues {
    bool compared = false;
    double diff = 0.0;
    double fps = 0.0;
    double frametime = 0.0;
    size_t tears = 0;
    MotionSample sample;
};

//! --ignore-clock-skew for video `video` of the rows: finds the episodes (clock_skew.h) and, if there are, recomputes
//! the framerate, the frametime, the animation error and the perceived smoothness of the rows as if every frame of
//! an episode had been recorded for its cadence, and the summaries. window: frames of the rolling framerate.
//! The re-timed rows count the source's refreshes - the filtered intervals are the ones the capture recorded twice -,
//! so they are turned into seconds at the source's rate: every value of the video changes by the ratio of the rates
//! (0.1% for 59.94 / 60), the effective framerate (frames per recorded second) stays.
ClockSkewResult filter_clock_skew(std::vector<std::vector<StoredValues>> & rows, size_t video, double fps, size_t window
                                , VideoStats & stats, MotionAnalyzer::Summary * motion)
{
    const size_t count = rows.size();
    std::vector<double> diff(count, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> step;
    if (motion) step.assign(count, std::numeric_limits<double>::quiet_NaN());
    for (size_t t = 0; t < count; ++t)
    {
        const StoredValues & v = rows[t][video];
        if (v.compared) diff[t] = v.diff;
        if (motion && v.sample.measurable) step[t] = v.sample.step;
    }
    ClockSkewResult result = detect_clock_skew(diff, step, fps);
    if (!result.found) return result;
    const std::vector<long long> times = clock_skew_times(diff, result.correction);
    const long long length = static_cast<long long>(window);
    const double source_fps = result.source_fps;
    // per second at the source's rate / per second at the recorded rate
    const double to_source = source_fps / fps;

    // rolling framerate and frametime: the frames of the last `window` refreshes of the re-timed time. Where those
    // hold the same rows as before and no correction, only the rate changes
    std::vector<int> corrected(count + 1, 0); // prefix sums of the rows with a correction
    for (size_t t = 0; t < count; ++t) corrected[t + 1] = corrected[t] + (result.correction[t] != 0 ? 1 : 0);
    size_t first = 0;
    while (first < count && !rows[first][video].compared) ++first;
    size_t lo = first;
    for (size_t t = first; t < count; ++t)
    {
        StoredValues & v = rows[t][video];
        if (!v.compared) continue;
        while (times[lo] <= times[t] - length) ++lo;
        const size_t original_lo = t + 1 >= first + window ? t + 1 - window : first;
        if (lo == original_lo && corrected[t + 1] == corrected[lo])
        {
            v.fps *= to_source;
            v.frametime /= to_source;
            continue;
        }
        double sum = 0.0;
        for (size_t j = lo; j <= t; ++j) sum += rows[j][video].diff;
        v.fps = sum * to_source;
        v.frametime = 0.0;
        if (sum < 2.0) continue;
        long long last_new = -1, previous_new = -1;
        for (size_t j = t + 1; j-- > lo;)
        {
            if (rows[j][video].diff <= 0.0) continue;
            if (last_new < 0) last_new = static_cast<long long>(j);
            else { previous_new = static_cast<long long>(j); break; }
        }
        if (previous_new >= 0)
        {
            v.frametime = static_cast<double>(times[static_cast<size_t>(last_new)] - times[static_cast<size_t>(previous_new)])
                        / static_cast<double>(window) * 1000.0 / to_source;
        }
    }

    // the summary of the framerate and the frametime, as while the video was analyzed
    stats.window_samples = 0;
    stats.fps_sum = 0.0;
    stats.fps_min = std::numeric_limits<double>::infinity();
    stats.fps_max = 0.0;
    stats.frametime_samples = 0;
    stats.frametime_sum = 0.0;
    stats.frametime_max = 0.0;
    long long compared = 0;
    for (size_t t = 0; t < count; ++t)
    {
        const StoredValues & v = rows[t][video];
        if (!v.compared) continue;
        if (++compared < static_cast<long long>(window)) continue;
        stats.window_samples += 1;
        stats.fps_sum += v.fps;
        stats.fps_min = std::min(stats.fps_min, v.fps);
        stats.fps_max = std::max(stats.fps_max, v.fps);
        if (v.frametime > 0.0)
        {
            stats.frametime_samples += 1;
            stats.frametime_sum += v.frametime;
            stats.frametime_max = std::max(stats.frametime_max, v.frametime);
        }
    }
    if (!motion) return result;

    // motion: a new frame of an episode counts as shown for its cadence in the animation error and the perceived
    // smoothness - if the motion's new frames are the frames the episode was found in
    RollingSmoothness rolling(source_fps, static_cast<size_t>(std::max(1.0, std::ceil(fps - 1e-9))));
    std::vector<float> errors;
    long long shown_delta = 0;
    long long previous_new = -1;
    for (size_t t = 0; t < count; ++t)
    {
        MotionSample & sample = rows[t][video].sample;
        const bool is_new = !std::isnan(diff[t]) && diff[t] > 0.0;
        if (sample.has_error)
        {
            double displayed = sample.chain_displayed;
            const int correction = result.correction[t];
            if (correction != 0 && sample.lost == 0 && previous_new >= 0 && sample.chain_shown == static_cast<long long>(t) - previous_new)
            {
                const int shown = sample.chain_shown + correction;
                displayed = static_cast<double>(shown);
                sample.error_ms = (sample.chain_step - displayed) * 1000.0 / source_fps;
                shown_delta += correction;
            }
            else
            {
                sample.error_ms /= to_source;
            }
            errors.push_back(static_cast<float>(sample.error_ms));
            const int pictures = 1 + sample.lost;
            for (int k = 0; k < pictures; ++k) rolling.add_step(times[t], sample.chain_step / pictures, displayed / pictures);
        }
        sample.has_perceived_fps = rolling.push_frame(times[t], sample.measurable, sample.perceived_fps);
        if (is_new) previous_new = static_cast<long long>(t);
    }
    MotionAnalyzer::Summary & m = *motion;
    m.shown_total += shown_delta;
    if (m.shown_total > 0)
    {
        m.motion_fps = source_fps * static_cast<double>(m.animation_frames) / static_cast<double>(m.shown_total);
        if (m.step_square_sum > 0.0) m.perceived_fps = source_fps * m.step_sum * m.step_sum / (static_cast<double>(m.shown_total) * m.step_square_sum);
    }
    if (!rolling.values().empty())
    {
        std::vector<float> sorted = rolling.values();
        std::sort(sorted.begin(), sorted.end());
        m.perceived_fps_median = sorted[sorted.size() / 2];
        m.perceived_fps_low = sorted[sorted.size() / 10];
    }
    m.median_step_spread_ms /= to_source;
    MotionAnalyzer::summarize_errors(errors, source_fps, m);
    return result;
}

//! --resolution: what was measured, then the render resolutions it fits
void print_resolution(size_t index, const ResolutionAnalyzer::Summary & r)
{
    typedef ResolutionAnalyzer::Summary Summary;
    const char * const unit = "cycles/pixel";
    const std::string prefix = "v" + std::to_string(index + 1) + " resolution: ";
    if (r.width == 0)
    {
        std::fprintf(stderr, "%snot measured (the frames are too small)\n", prefix.c_str());
        return;
    }
    auto format = [](const char * pattern, double a, double b) {
        char text[96];
        std::snprintf(text, sizeof(text), pattern, a, b);
        return std::string(text);
    };

    // ---- pixel grid
    if (r.grid_from != VideoOptions().resolution_grid_from || r.grid_to != VideoOptions().resolution_grid_to)
    {
        std::fprintf(stderr, "%spixel grid measured in the rows %s - %s%% of the height\n", prefix.c_str()
                   , format_number(r.grid_from).c_str(), format_number(r.grid_to).c_str());
    }
    auto print_no_grid = [&]() {
        if (r.grid_rows < r.grid_rows_needed)
        {
            std::fprintf(stderr, "%spixel grid not measured: its part of the height has %d rows, %d are needed\n", prefix.c_str(), r.grid_rows, r.grid_rows_needed);
            return;
        }
        std::fprintf(stderr, "%sno pixel grid in both axes (strongest peak %.1fx the background at %.4f %s)"
                   , prefix.c_str(), std::min(r.lattice_strength_x, r.lattice_strength_y), r.lattice_frequency, unit);
        if (r.codec_peak)
        {
            std::fprintf(stderr, ". The peak at %.4f %s (%.1fx) lies on the block grid of video encoders: a render scale of %.1f%% or %.1f%% can't be told from it"
                       , r.codec_frequency, unit, r.codec_strength, 100.0 * r.codec_frequency, 100.0 * (1.0 - r.codec_frequency));
        }
        std::fprintf(stderr, "\n");
    };
    for (int pass = 0; pass < 2; ++pass)
    {
        // what both axes have (or that there is nothing) first, then what only one axis has
        if (pass == 1 && !r.lattice) print_no_grid();
        for (const Summary::Peak & peak : r.peaks)
        {
            if ((peak.axes == Summary::Peak::Both) != (pass == 0)) continue;
            const double low = 100.0 * peak.frequency, high = 100.0 * (1.0 - peak.frequency);
            if (peak.axes == Summary::Peak::Both)
            {
                const std::string pieces = peak.share_x < 0.0 ? "" : format(", in %.0f%% / %.0f%% of the video", 100.0 * peak.share_x, 100.0 * peak.share_y);
                std::string harmonic;
                if (r.lossless && std::abs(peak.harmonic_frequency - peak.frequency) >= 0.006)
                {
                    harmonic = format("; its second harmonic at %.4f is %.2f", peak.harmonic_frequency, peak.harmonic_share_x)
                             + format(" / %.2f of it (strong: the smaller scale, weak: the larger)", peak.harmonic_share_y, 0.0);
                }
                std::fprintf(stderr, "%spixel grid at %.4f %s (%.1fx / %.1fx the background in the columns / rows%s): rendered at %.1f%% or %.1f%% of the output%s\n"
                           , prefix.c_str(), peak.frequency, unit, peak.strength_x, peak.strength_y, pieces.c_str(), low, high, harmonic.c_str());
            }
            else
            {
                const bool columns = peak.axes == Summary::Peak::Columns;
                std::fprintf(stderr, "%spixel grid in the %s only, at %.4f %s (%.1fx the background in %.0f%% of the video, %.1fx in the %s): %.1f%% or %.1f%% of the output."
                                     " Weak: a regular pattern in the picture (a fence, a HUD) leaves such a peak too\n"
                           , prefix.c_str(), columns ? "columns" : "rows", peak.frequency, unit, columns ? peak.strength_x : peak.strength_y
                           , 100.0 * (columns ? peak.share_x : peak.share_y), columns ? peak.strength_y : peak.strength_x, columns ? "rows" : "columns", low, high);
            }
        }
    }
    if (r.half_present && r.lossless)
    {
        std::fprintf(stderr, "%speak at 0.5 %s (%.1fx / %.1fx the background in the columns / rows; at 0.25: %.1fx / %.1fx): %s\n"
                   , prefix.c_str(), unit, r.half_strength_x, r.half_strength_y, r.quarter_strength_x, r.quarter_strength_y
                   , r.half_peak ? "a render scale of 50% leaves one there (25% and 75% would leave a peak at 0.25 as well, there is none)"
                                 : "the second harmonic of the pixel grid at 0.25");
    }
    else if (r.half_present)
    {
        std::fprintf(stderr, "%speak at 0.5 %s (%.1fx / %.1fx the background in the columns / rows; the encoder's at 0.25: %.1fx / %.1fx): %s\n"
                   , prefix.c_str(), unit, r.half_strength_x, r.half_strength_y, r.quarter_strength_x, r.quarter_strength_y
                   , r.half_peak ? "a render scale of 50% leaves one there, 25% or 75% as well"
                                 : "small against the encoder's, it may be the encoder's");
    }

    // ---- pan
    const long long pan_frames = r.pan_frames_right + r.pan_frames_left;
    if (r.contrast[0].empty())
    {
        std::fprintf(stderr, "%sno pan to measure (%lld frames of a steady horizontal pan of 4 - 64 pixels per frame, 30 needed)\n", prefix.c_str(), pan_frames);
    }
    else
    {
        std::fprintf(stderr, "%spicture entering at the edge of a pan, %lld + %lld frames (moving right + left), rows %s - %s%% of the height:"
                             " contrast against the picture that leaves down to %.2f (at %.3f %s)"
                   , prefix.c_str(), r.pan_frames_right, r.pan_frames_left, format_number(r.pan_from).c_str(), format_number(r.pan_to).c_str()
                   , r.lowest_contrast, r.lowest_contrast_frequency, unit);
        if (r.half_contrast_frequency > 0.0) std::fprintf(stderr, ", one half at %.3f %s", r.half_contrast_frequency, unit);
        if (r.highest_contrast >= 1.25) std::fprintf(stderr, ", up to %.2f (at %.3f %s)", r.highest_contrast, r.highest_contrast_frequency, unit);
        if (r.recovered_age > 0) std::fprintf(stderr, "; %.2f for what entered %s", r.recovered_contrast, kResolutionAgeNames[r.recovered_age]);
        if (r.fan_peak_right > 0.0 || r.fan_peak_left > 0.0)
        {
            const std::string right = r.fan_peak_right > 0.0 ? format("%.3f", r.fan_peak_right, 0.0) : std::string("-");
            const std::string left = r.fan_peak_left > 0.0 ? format("%.3f", r.fan_peak_left, 0.0) : std::string("-");
            std::fprintf(stderr, "; the newest columns peak at %s / %s %s (right / left: about 0.4 - 0.56 of the render scale in the"
                                 " recordings tried)", right.c_str(), left.c_str(), unit);
        }
        if (r.half_contrast_frequency > 0.0)
        {
            std::fprintf(stderr, ": a single frame resolves about %.0f%% of the output, the rest builds up over the frames (temporal upscaling)\n"
                       , 100.0 * std::min(1.0, 2.0 * r.half_contrast_frequency));
        }
        else if (r.fan_up)
        {
            std::fprintf(stderr, ": the entering picture has more contrast than the settled one and loses it with its age (an upward fan) -"
                                 " a temporal upscaler that shows new pixels as they were rendered\n");
        }
        else
        {
            std::fprintf(stderr, ": the entering picture keeps more than half of its contrast, the pan tells nothing about the render scale\n");
        }
    }

    // ---- stair steps
    if (!r.contrast[0].empty() && (r.stair_steps_entering > 0 || r.stair_steps_leaving > 0))
    {
        std::fprintf(stderr, "%sstair steps of nearly level edges: %lld in the entering picture, %.0f%% of them 2 pixels or higher; %lld in the leaving picture, %.0f%%: "
                   , prefix.c_str(), r.stair_steps_entering, 100.0 * r.stair_high_entering, r.stair_steps_leaving, 100.0 * r.stair_high_leaving);
        if (r.stairs)
        {
            std::fprintf(stderr, "what is new is shown as it was rendered, a rendered pixel is about %.1f output pixels high - a render scale of about %.0f%%%s\n"
                       , r.stair_height, 100.0 * r.stair_scale
                       , r.stair_height < 2.4 ? " (steps are whole pixels: anything from 50% to about 60% reads like this)" : "");
        }
        else if (r.stair_steps_entering < 100)
        {
            std::fprintf(stderr, "too few to tell (100 are needed; the picture needs straight edges that are nearly level)\n");
        }
        else if (r.stair_high_entering < 0.6)
        {
            std::fprintf(stderr, "too few of them are high to tell (60%% are needed): steps of 1 pixel say nothing, a smoothed picture has them and so has"
                                 " a render scale above about 60%%\n");
        }
        else
        {
            std::fprintf(stderr, "the leaving picture has high steps as well (or too few to compare): the picture itself is stepped, not only what is new\n");
        }
    }

    // ---- the candidates. Render resolutions are even
    auto size = [&r](double scale) {
        return std::to_string(2 * std::llround(scale * r.width / 2.0)) + "x" + std::to_string(2 * std::llround(scale * r.height / 2.0));
    };
    if (r.candidates.empty() && r.lossless)
    {
        std::fprintf(stderr, "%sno candidate: no trace of upscaling found (native resolution or an upscaler that leaves no trace)\n", prefix.c_str());
        return;
    }
    if (r.candidates.empty())
    {
        std::fprintf(stderr, "%sno candidate: no trace of upscaling found (native resolution, a render scale on the block grid of the encoder such as 25%% or 75%%,"
                             " an upscaler that leaves no trace, or a recording too compressed to keep one)\n", prefix.c_str());
        return;
    }
    static const char * const levels[] = { "low", "medium", "high" };
    int number = 0;
    for (size_t i = 0; i < r.candidates.size(); ++i)
    {
        const Summary::Candidate & candidate = r.candidates[i];
        std::string what = size(candidate.scale) + format(" (%.1f%% of %.0f", 100.0 * candidate.scale, static_cast<double>(r.width)) + "x" + std::to_string(r.height) + ")";
        // the two scales of a grid that nothing tells apart share a line
        if (i + 1 < r.candidates.size())
        {
            const Summary::Candidate & next = r.candidates[i + 1];
            if (candidate.peak >= 0 && next.peak == candidate.peak && next.evidence == candidate.evidence && next.confidence == candidate.confidence)
            {
                what += " or " + size(next.scale) + format(" (%.1f%%)", 100.0 * next.scale, 0.0);
                ++i;
            }
        }
        std::string why;
        switch (candidate.evidence)
        {
        case Summary::Candidate::GridAndPan: why = "the scale of the pixel grid that fits the pan"; break;
        case Summary::Candidate::GridOtherScale: why = "the other scale of that pixel grid, the pan speaks against it"; break;
        case Summary::Candidate::GridAndStairs: why = "the scale of the pixel grid that fits the stair steps of the entering picture"; break;
        case Summary::Candidate::GridAgainstStairs: why = "the other scale of that pixel grid, the stair steps speak against it"; break;
        case Summary::Candidate::GridAndFanPeak:
            why = std::string("the scale of the pixel grid that the peak of the newest columns of the pan picks")
                + (r.peaks[static_cast<size_t>(candidate.peak)].axes == Summary::Peak::Both ? ""
                   : r.peaks[static_cast<size_t>(candidate.peak)].axes == Summary::Peak::Columns ? " (a grid in the columns only)" : " (a grid in the rows only)");
            break;
        case Summary::Candidate::GridAgainstFanPeak: why = "the other scale of that pixel grid, the peak of the newest columns speaks against it"; break;
        case Summary::Candidate::GridAndHarmonic:
            why = candidate.scale < 0.5
                ? "the smaller scale of the pixel grid: its second harmonic is strong, as a picture enlarged by a whole factor has it (recorded lossless)"
                : "the larger scale of the pixel grid: its second harmonic is weak (recorded lossless)";
            break;
        case Summary::Candidate::GridAgainstHarmonic: why = "the other scale of that pixel grid, its second harmonic speaks against it"; break;
        case Summary::Candidate::HalfAndStairs: why = "the peak at 0.5 cycles/pixel, and stair steps of 2 pixels in the entering picture"; break;
        case Summary::Candidate::Stairs: why = "the stair steps of the entering picture alone, without a pixel grid: rough"; break;
        case Summary::Candidate::GridAndFan: why = "the larger scale of the pixel grid, the true one in the recordings with an upward fan"; break;
        case Summary::Candidate::GridAgainstFan: why = "the smaller scale of that pixel grid"; break;
        case Summary::Candidate::Grid:
            why = candidate.confidence == Summary::Candidate::Medium
                ? std::string("pixel grid in both axes, nothing tells its two scales apart") + (r.contrast[0].empty() ? " (a pan might)" : "")
                : std::string("a second, weaker pixel grid");
            break;
        case Summary::Candidate::GridOneAxis:
            why = std::string("pixel grid in the ") + (r.peaks[static_cast<size_t>(candidate.peak)].axes == Summary::Peak::Columns ? "columns" : "rows") + " only";
            break;
        case Summary::Candidate::HalfPeak:
            why = r.lossless ? "the peak at 0.5 cycles/pixel, without a grid at 0.25 (recorded lossless)"
                             : "the peak at 0.5 cycles/pixel (25% or 75% leave one there too)";
            break;
        case Summary::Candidate::Pan: why = "the pan alone, without a pixel grid to confirm it: rough"; break;
        }
        std::fprintf(stderr, "%scandidate %d: %s - %s confidence: %s\n", prefix.c_str(), ++number, what.c_str(), levels[candidate.confidence], why.c_str());
    }
}

class ProgressPrinter {
public:
    explicit ProgressPrinter(bool enabled) : _enabled(enabled), _last(std::chrono::steady_clock::now()) { }
    void update(const FramePipeline & sources, bool force = false)
    {
        if (!_enabled) return;
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - _last < std::chrono::milliseconds(500)) return;
        _last = now;
        const double progress = sources.progress();
        if (progress >= 0.0)
        {
            std::fprintf(stderr, "\rframe %lld (%.1f%%)   ", sources.frames_read(), progress * 100.0);
        }
        else
        {
            std::fprintf(stderr, "\rframe %lld   ", sources.frames_read());
        }
        std::fflush(stderr);
        _printed = true;
    }
    void finish()
    {
        if (_printed) std::fprintf(stderr, "\n");
    }

private:
    bool _enabled;
    bool _printed = false;
    std::chrono::steady_clock::time_point _last;
};

//! OpenCV decoding with the Vulkan engine (if enabled and available) or the CPU engine
std::unique_ptr<FramePipeline> make_opencv_decode_pipeline(const Options & options)
{
    std::unique_ptr<DiffEngine> engine;
#ifdef USE_VULKAN
    if (options.use_vulkan)
    {
        try
        {
            engine = make_vulkan_engine(options.vk_device, options.inputs.size());
        }
        catch (const std::exception & e)
        {
            // an explicitly requested device has to work, otherwise fall back to the CPU
            if (options.vk_device >= 0) throw;
            std::fprintf(stderr, "trdrop-pp: Vulkan not available (%s), using the CPU\n", e.what());
        }
    }
#endif
    if (!engine) engine = make_cpu_engine(options.inputs.size());
    return make_opencv_pipeline(options.inputs, std::move(engine));
}

std::unique_ptr<FramePipeline> make_pipeline(const Options & options)
{
#ifdef TRDROP_MEDIACODEC_VULKAN
    // Android: hardware decoding and analysis on the GPU if the device and the videos allow it
    if (options.decode == "vulkan") return make_mediacodec_vulkan_pipeline(options.inputs);
    if (options.decode == "auto" && options.use_vulkan)
    {
        try
        {
            return make_mediacodec_vulkan_pipeline(options.inputs);
        }
        catch (const std::exception & e)
        {
            if (!options.quiet) std::fprintf(stderr, "trdrop-pp: analyzing on the CPU (%s)\n", e.what());
        }
    }
#endif
#if defined(USE_FFMPEG) && defined(USE_VULKAN)
    if (options.decode == "vulkan") return make_vulkan_video_pipeline(options.inputs, options.vk_device);
#endif
#ifdef USE_FFMPEG
    if (options.decode == "ffmpeg") return make_ffmpeg_cpu_pipeline(options.inputs);
    // auto: the fastest decoder for the inputs; --no-vk keeps the OpenCV path
    if (options.decode == "auto" && options.use_vulkan)
    {
        auto note = [&](const std::string & what, const std::string & why) {
#ifdef TRDROP_MEDIACODEC_VULKAN
            // said above
            (void)what;
            (void)why;
#else
            if (!options.quiet) std::fprintf(stderr, "trdrop-pp: decoding with %s (%s)\n", what.c_str(), why.c_str());
#endif
        };
#ifdef USE_VULKAN
        const bool vulkan_available = true;
#else
        const bool vulkan_available = false;
#endif
        // with --motion the GPU also measures the motion, which is much slower on the CPU
        const DecoderChoice choice = choose_decoder(options.inputs, vulkan_available, options.motion);
        std::string fallback_reason = choice.reason;
#ifdef USE_VULKAN
        if (choice.decoder == "vulkan")
        {
            try
            {
                return make_vulkan_video_pipeline(options.inputs, options.vk_device);
            }
            catch (const std::exception & e)
            {
                // e.g. no Vulkan driver or the device can't decode this profile
                fallback_reason = std::string("no Vulkan Video decoding: ") + e.what();
            }
        }
#endif
        if (choice.decoder == "vulkan" || choice.decoder == "ffmpeg")
        {
            try
            {
                std::unique_ptr<FramePipeline> pipeline = make_ffmpeg_cpu_pipeline(options.inputs);
                note("FFmpeg on the CPU", fallback_reason);
                return pipeline;
            }
            catch (const std::exception & e)
            {
                fallback_reason = e.what();
            }
        }
        note("OpenCV", fallback_reason);
    }
#endif
    return make_opencv_decode_pipeline(options);
}

//! export_rows: receives the values of every row once the video is analyzed (--ignore-clock-skew: filtered).
//! render_rows: the second pass of --ignore-clock-skew with --motion-video - nothing is written but the motion
//! videos, and they show these values instead of the ones measured again
int run(const Options & options, std::vector<std::vector<StoredValues>> * export_rows = nullptr
      , const std::vector<std::vector<StoredValues>> * render_rows = nullptr)
{
    const bool render_only = render_rows != nullptr;
    std::unique_ptr<FramePipeline> pipeline_owner = make_pipeline(options);
    FramePipeline & pipeline = *pipeline_owner;
    if (options.motion) pipeline.enable_motion();
    if (!options.motion_videos.empty() || !options.resolution_videos.empty())
    {
        pipeline.enable_motion_detail(options.motion_video_native && !options.motion_videos.empty());
    }
    if (options.resolution) pipeline.enable_resolution();
    const FramePipeline & sources = pipeline;
    const size_t video_count = sources.count();

    // the framerate window has the size of the recorded framerate, as we can't detect a higher framerate
    std::vector<int> window_sizes;
    std::vector<double> recorded_framerates;
    for (size_t i = 0; i < video_count; ++i)
    {
        const double recorded = options.videos[i].source_fps_override > 0.0
                              ? options.videos[i].source_fps_override
                              : sources.recorded_framerate(i);
        if (!(recorded > 0.0) || !std::isfinite(recorded))
        {
            throw std::runtime_error("could not read the framerate of '" + options.inputs[i] + "', use --source-fps");
        }
        recorded_framerates.push_back(recorded);
        window_sizes.push_back(static_cast<int>(std::ceil(recorded - 1e-9)));
    }

    // lossy codecs: duplicate frames differ by compression noise, tolerate it unless told otherwise
    std::vector<VideoOptions> video_options = options.videos;
    std::vector<std::string> automatic(video_count);
    for (size_t i = 0; i < video_count; ++i)
    {
        VideoOptions & v = video_options[i];
        // --lossless: a duplicate is identical, whatever the codec could do
        const bool lossy = sources.lossy_codec(i) && !v.lossless;
        if (lossy && !v.pixel_difference_set)
        {
            v.pixel_difference = kLossyPixelDifference;
            automatic[i] = "pixel-diff";
        }
        if (v.tolerance_pct < 0.0)
        {
            v.tolerance_pct = lossy ? kLossyTolerancePct : 0.0;
            v.tolerance_auto = lossy;
            if (lossy) automatic[i] += automatic[i].empty() ? "tolerance-pct" : " and tolerance-pct";
        }
        if (!automatic[i].empty()) automatic[i] = ", " + automatic[i] + " set for the lossy codec";
        if (v.lossless && sources.lossy_codec(i)) automatic[i] += ", taken as lossless (--lossless)";
    }

    if (!options.quiet && render_only)
    {
        std::fprintf(stderr, "drawing the motion video%s with the filtered values (--ignore-clock-skew: a second pass over the video)\n"
                   , options.motion_videos.size() > 1 ? "s" : "");
    }
    if (!options.quiet && !render_only)
    {
        for (size_t i = 0; i < video_count; ++i)
        {
            std::fprintf(stderr, "v%zu: %s (%dx%d, %s, %s fps, %lld frames reported, pixel-diff %d, tolerance-pct %s, dismiss-tear-pct %s%s)\n"
                       , i + 1, options.inputs[i].c_str()
                       , sources.frame_width(i), sources.frame_height(i), sources.codec_name(i).c_str()
                       , format_number(recorded_framerates[i]).c_str(), sources.reported_frame_count(i)
                       , video_options[i].pixel_difference
                       , (video_options[i].tolerance_auto ? "auto, keyframe-tolerance " + format_number(video_options[i].keyframe_tolerance)
                                                          : format_number(video_options[i].tolerance_pct)).c_str()
                       , format_number(video_options[i].dismiss_tear_percentage).c_str()
                       , automatic[i].c_str());
        }
    }

    if (!options.quiet && !render_only) std::fprintf(stderr, "backend: %s\n", pipeline.description().c_str());

    // --motion-video, --resolution-video: a writer per video that has a file
    std::vector<std::unique_ptr<MotionVideo>> motion_videos(video_count);
    std::vector<std::unique_ptr<ResolutionVideo>> resolution_videos(video_count);
    for (size_t i = 0; i < video_count; ++i)
    {
        const std::string motion_path = i < options.motion_videos.size() ? options.motion_videos[i] : std::string();
        const std::string resolution_path = i < options.resolution_videos.size() ? options.resolution_videos[i] : std::string();
        std::vector<std::string> paths;
        if (!motion_path.empty()) paths.push_back(motion_path);
        if (!resolution_path.empty()) paths.push_back(resolution_path);
        if (paths.empty()) continue;
        if (motion_path == resolution_path) throw std::invalid_argument("--motion-video and --resolution-video: the same file for both");
        // drawn and encoded where the frames are if the pipeline can do that for all of them, otherwise on the CPU
        std::string reason;
        std::vector<std::unique_ptr<MotionVideoTarget>> targets = pipeline.start_videos(i, paths, recorded_framerates[i], reason);
        if (targets.size() != paths.size()) targets.clear();
        if (targets.empty() && !reason.empty() && !options.quiet)
        {
            std::fprintf(stderr, "trdrop-pp: drawing the %s on the CPU (%s)\n", paths.size() > 1 ? "videos" : !motion_path.empty() ? "motion video" : "resolution video"
                       , reason.c_str());
        }
        size_t next_target = 0;
        auto take_target = [&]() { return targets.empty() ? nullptr : std::move(targets[next_target++]); };
        if (!motion_path.empty())
        {
            motion_videos[i] = std::make_unique<MotionVideo>(motion_path, sources.frame_width(i), sources.frame_height(i)
                                                           , recorded_framerates[i], options.arrow_scale, take_target()
                                                           , options.motion_video_native, options.motion_arrows);
            if (!options.quiet)
            {
                std::fprintf(stderr, "v%zu motion video: %s (%dx%d, %s)\n", i + 1, motion_videos[i]->path().c_str()
                           , motion_videos[i]->size().width, motion_videos[i]->size().height, motion_videos[i]->description().c_str());
            }
        }
        if (!resolution_path.empty())
        {
            resolution_videos[i] = std::make_unique<ResolutionVideo>(resolution_path, sources.frame_width(i), sources.frame_height(i)
                                                                   , recorded_framerates[i], take_target());
            if (!options.quiet)
            {
                std::fprintf(stderr, "v%zu resolution video: %s (%dx%d, %s)\n", i + 1, resolution_videos[i]->path().c_str()
                           , resolution_videos[i]->size().width, resolution_videos[i]->size().height, resolution_videos[i]->description().c_str());
            }
        }
    }

    // outputs
    std::ofstream output_file;
    std::ostream discard(nullptr); // the render pass writes no CSV
    std::ostream * output = render_only ? &discard : &std::cout;
    if (!render_only && options.output_path != "-")
    {
        output_file.open(options.output_path, std::ios::out | std::ios::trunc);
        if (!output_file) throw std::runtime_error("could not open '" + options.output_path + "' for writing");
        output = &output_file;
    }
    std::ofstream tears_file;
    if (!options.tears_path.empty())
    {
        tears_file.open(options.tears_path, std::ios::out | std::ios::trunc);
        if (!tears_file) throw std::runtime_error("could not open '" + options.tears_path + "' for writing");
        tears_file << "frame,video,row_a,row_b,video_height\n";
    }

    *output << "frame";
    for (size_t i = 0; i < video_count; ++i)
    {
        const std::string prefix = "v" + std::to_string(i + 1) + "_";
        *output << ',' << prefix << "diff" << ',' << prefix << "fps" << ',' << prefix << "frametime_ms" << ',' << prefix << "tears";
        if (options.motion)
        {
            *output << ',' << prefix << "anim_step" << ',' << prefix << "anim_step_spread" << ',' << prefix << "anim_error_ms"
                    << ',' << prefix << "perceived_fps";
            if (options.async_capture) *output << ',' << prefix << "lost";
        }
    }
    *output << '\n';

    FrameAnalyzer analyzer(window_sizes);
    std::vector<VideoStats> stats(video_count);
    ProgressPrinter progress(!options.quiet);
    std::vector<size_t> tears_per_video(video_count);
    std::vector<RowCounts> row_counts;
    std::string line;

    // --ignore-clock-skew: the values of every row, per video, written once the whole video is known
    std::vector<std::vector<StoredValues>> stored_rows;
    auto format_values = [&](const StoredValues & v, std::string & out) {
        out += ',';
        if (v.compared) out += format_number(v.diff);
        out += ',' + format_number(v.fps);
        out += ',' + format_number(v.frametime);
        out += ',' + std::to_string(v.tears);
        if (!options.motion) return;
        out += ',';
        if (v.sample.measurable) out += format_number(v.sample.step);
        out += ',';
        if (v.sample.measurable) out += format_number(v.sample.step_spread);
        out += ',';
        if (v.sample.has_error) out += format_number(v.sample.error_ms);
        out += ',';
        if (v.sample.has_perceived_fps) out += format_number(v.sample.perceived_fps);
        if (options.async_capture)
        {
            out += ',';
            if (v.sample.has_error) out += std::to_string(v.sample.lost);
        }
    };

    // --motion: the motion results of a frame need the frames after it, the rows wait for them
    struct PendingRow {
        long long frame = 0;
        std::vector<std::string> columns; // per video, each starts with a comma
        std::vector<StoredValues> stored; // --ignore-clock-skew
        // for the motion videos, per video
        std::vector<MotionVideoFrame> values;
        std::vector<std::shared_ptr<MotionDetail>> details;
    };
    std::deque<PendingRow> pending_rows;
    std::vector<MotionAnalyzer> motion_analyzers;
    if (options.motion)
    {
        for (size_t i = 0; i < video_count; ++i)
        {
            motion_analyzers.emplace_back(recorded_framerates[i], options.motion_window_ms, options.async_capture);
        }
    }
    std::vector<ResolutionAnalyzer> resolution_analyzers(options.resolution ? video_count : 0);
    for (size_t i = 0; i < resolution_analyzers.size(); ++i) resolution_analyzers[i].set_lossless(video_options[i].lossless);
    auto write_ready_rows = [&]() {
        while (!pending_rows.empty())
        {
            for (const MotionAnalyzer & motion : motion_analyzers)
            {
                if (!motion.ready()) return;
            }
            PendingRow & row = pending_rows.front();
            line = std::to_string(row.frame);
            for (size_t i = 0; i < video_count; ++i)
            {
                MotionSample sample;
                motion_analyzers[i].pop(sample);
                line += row.columns[i];
                line += ',';
                if (sample.measurable) line += format_number(sample.step);
                line += ',';
                if (sample.measurable) line += format_number(sample.step_spread);
                line += ',';
                if (sample.has_error) line += format_number(sample.error_ms);
                line += ',';
                if (sample.has_perceived_fps) line += format_number(sample.perceived_fps);
                if (options.async_capture)
                {
                    line += ',';
                    if (sample.has_error) line += std::to_string(sample.lost);
                }
                if (motion_videos[i] && row.details[i])
                {
                    MotionVideoFrame & values = row.values[i];
                    values.sample = sample;
                    if (render_rows && static_cast<size_t>(row.frame) < render_rows->size() && i < (*render_rows)[static_cast<size_t>(row.frame)].size())
                    {
                        // the values of the first pass, as the CSV has them; the arrows are the ones measured now
                        const StoredValues & v = (*render_rows)[static_cast<size_t>(row.frame)][i];
                        values.compared = v.compared;
                        values.duplicate = v.compared && v.diff == 0.0;
                        values.torn = v.compared && v.diff > 0.0 && v.diff < 1.0;
                        values.framerate = v.fps;
                        values.sample = v.sample;
                    }
                    motion_videos[i]->write(values, *row.details[i]);
                }
                if (options.ignore_clock_skew) row.stored[i].sample = sample;
            }
            line += '\n';
            if (options.ignore_clock_skew) stored_rows.push_back(std::move(row.stored));
            else *output << line;
            pending_rows.pop_front();
        }
    };

    using Clock = std::chrono::steady_clock;
    const Clock::time_point start_time = Clock::now();
    Clock::duration analysis_time {};

    // takes the oldest result of the pipeline and writes its row
    long long frame = 0;
    auto emit_frame = [&]() {
        pipeline.collect(row_counts);
        const Clock::time_point analyzer_start = Clock::now();
        analyzer.process(row_counts, video_options);
        analysis_time += Clock::now() - analyzer_start;

        const std::vector<double> framerates = analyzer.framerates();
        const std::vector<double> frametimes = analyzer.frametimes();
        const std::vector<double> & differences = analyzer.last_differences();

        std::fill(tears_per_video.begin(), tears_per_video.end(), 0);
        for (const Tear & tear : analyzer.tears())
        {
            tears_per_video[tear.video_index] += 1;
            if (tears_file.is_open())
            {
                tears_file << frame << ',' << (tear.video_index + 1) << ',' << tear.index_a << ',' << tear.index_b
                           << ',' << tear.original_video_height << '\n';
            }
        }

        PendingRow row;
        row.frame = frame;
        row.columns.resize(video_count);
        row.values.resize(video_count);
        row.details.resize(video_count);
        if (options.ignore_clock_skew)
        {
            row.stored.resize(video_count);
            for (size_t i = 0; i < video_count; ++i)
            {
                StoredValues & v = row.stored[i];
                v.compared = analyzer.has_difference();
                v.diff = differences[i];
                v.fps = framerates[i];
                v.frametime = frametimes[i];
                v.tears = tears_per_video[i];
            }
        }
        for (size_t i = 0; i < video_count; ++i)
        {
            // the picture of the frame, for the videos that are drawn on it
            const std::shared_ptr<MotionDetail> detail = std::move(row_counts[i].detail);
            if (motion_videos[i])
            {
                MotionVideoFrame & values = row.values[i];
                values.frame = frame;
                values.compared = analyzer.has_difference();
                values.duplicate = values.compared && differences[i] == 0.0;
                values.torn = values.compared && differences[i] > 0.0 && differences[i] < 1.0;
                values.framerate = framerates[i];
                row.details[i] = detail;
            }
            std::string & column = row.columns[i];
            column = ',';
            if (analyzer.has_difference()) column += format_number(differences[i]);
            column += ',' + format_number(framerates[i]);
            column += ',' + format_number(frametimes[i]);
            column += ',' + std::to_string(tears_per_video[i]);
            if (options.motion)
            {
                motion_analyzers[i].push(row_counts[i].motion, analyzer.has_difference() && differences[i] == 0.0
                                       , analyzer.has_difference() ? analyzer.last_changed_shares()[i] : 1.0);
            }
            if (options.resolution) resolution_analyzers[i].push(row_counts[i].resolution.get(), row_counts[i].motion);
            // the resolution video shows the frame with what was collected up to it: it doesn't wait for the motion
            if (resolution_videos[i] && detail)
            {
                resolution_videos[i]->write(frame, analyzer.has_difference() && differences[i] == 0.0, resolution_analyzers[i], *detail);
            }

            if (!analyzer.has_difference()) continue;
            VideoStats & s = stats[i];
            const double diff = differences[i];
            s.compared_frames += 1;
            s.diff_sum += diff;
            if (diff == 0.0) s.duplicate_frames += 1;
            else if (diff >= 1.0) s.new_frames += 1;
            else s.torn_frames += 1;
            s.tear_count += static_cast<long long>(tears_per_video[i]);
            // the rolling values are only meaningful once the window is filled
            if (s.compared_frames >= window_sizes[i])
            {
                s.window_samples += 1;
                s.fps_sum += framerates[i];
                s.fps_min = std::min(s.fps_min, framerates[i]);
                s.fps_max = std::max(s.fps_max, framerates[i]);
                if (frametimes[i] > 0.0)
                {
                    s.frametime_samples += 1;
                    s.frametime_sum += frametimes[i];
                    s.frametime_max = std::max(s.frametime_max, frametimes[i]);
                }
            }
        }
        if (options.motion)
        {
            pending_rows.push_back(std::move(row));
            write_ready_rows();
        }
        else if (options.ignore_clock_skew)
        {
            stored_rows.push_back(std::move(row.stored));
        }
        else
        {
            line = std::to_string(row.frame);
            for (const std::string & column : row.columns) line += column;
            line += '\n';
            *output << line;
        }
        ++frame;
    };

    // the pipeline may work on a set of frames while the next one is decoded
    long long submitted = 0;
    while ((options.max_frames == 0 || submitted < options.max_frames) && pipeline.submit_next(video_options))
    {
        ++submitted;
        if (pipeline.pending() >= pipeline.max_pending()) emit_frame();
        progress.update(sources);
        if (options.progress && !options.progress(sources.frames_read(), sources.progress())) throw std::runtime_error("cancelled");
    }
    while (pipeline.pending() > 0) emit_frame();
    for (MotionAnalyzer & motion : motion_analyzers) motion.finish();
    write_ready_rows();
    std::vector<MotionAnalyzer::Summary> motion_summaries;
    for (const MotionAnalyzer & motion : motion_analyzers) motion_summaries.push_back(motion.summary());
    std::vector<ClockSkewResult> clock_skew(options.ignore_clock_skew ? video_count : 0);
    for (size_t i = 0; i < clock_skew.size(); ++i)
    {
        clock_skew[i] = filter_clock_skew(stored_rows, i, recorded_framerates[i], static_cast<size_t>(window_sizes[i])
                                        , stats[i], options.motion ? &motion_summaries[i] : nullptr);
    }
    if (options.ignore_clock_skew)
    {
        for (size_t row = 0; row < stored_rows.size(); ++row)
        {
            line = std::to_string(row);
            for (const StoredValues & v : stored_rows[row]) format_values(v, line);
            line += '\n';
            *output << line;
        }
        if (export_rows) *export_rows = std::move(stored_rows);
    }
    for (const std::unique_ptr<MotionVideo> & video : motion_videos)
    {
        if (video) video->finish();
    }
    for (const std::unique_ptr<ResolutionVideo> & video : resolution_videos)
    {
        if (video) video->finish();
    }
    progress.update(sources, true);
    progress.finish();
    output->flush();
    if (!render_only && !*output) throw std::runtime_error("failed writing the output");
    for (size_t i = 0; i < options.resolution_data.size() && i < resolution_analyzers.size(); ++i)
    {
        const std::string & path = options.resolution_data[i];
        if (path.empty()) continue;
        std::ofstream data(path, std::ios::out | std::ios::trunc);
        if (!data) throw std::runtime_error("could not open '" + path + "' for writing");
        resolution_analyzers[i].write_data(data, options.inputs[i]);
        data.flush();
        if (!data) throw std::runtime_error("failed writing '" + path + "'");
        if (options.print_summary) std::fprintf(stderr, "v%zu resolution data: %s\n", i + 1, path.c_str());
    }

    const double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_time).count();
    // decoding and comparison may run fused on the GPU, then only the total is meaningful
    const double pipeline_analysis_s = pipeline.analysis_seconds();
    const double analysis_ms = std::chrono::duration<double, std::milli>(analysis_time).count() + pipeline_analysis_s * 1000.0;

    if (options.print_summary)
    {
        std::fprintf(stderr, "analyzed %lld frames in %.2f s (%.1f fps)"
                   , frame, total_ms / 1000.0, total_ms > 0.0 ? static_cast<double>(frame) / (total_ms / 1000.0) : 0.0);
        if (pipeline_analysis_s >= 0.0)
        {
            std::fprintf(stderr, " | analysis %.1f ms (%.3f ms/frame)", analysis_ms, frame > 0 ? analysis_ms / static_cast<double>(frame) : 0.0);
        }
        std::fprintf(stderr, " | %s\n", pipeline.description().c_str());
        for (size_t i = 0; i < video_count; ++i)
        {
            const VideoStats & s = stats[i];
            const double effective_fps = s.compared_frames > 0 ? s.diff_sum / static_cast<double>(s.compared_frames) * recorded_framerates[i] : 0.0;
            std::fprintf(stderr, "v%zu: effective %.2f fps | new %lld, duplicate %lld, torn %lld (%lld tears)"
                       , i + 1, effective_fps, s.new_frames, s.duplicate_frames, s.torn_frames, s.tear_count);
            if (s.window_samples > 0)
            {
                std::fprintf(stderr, " | fps avg %.2f min %.2f max %.2f", s.fps_sum / s.window_samples, s.fps_min, s.fps_max);
            }
            if (s.frametime_samples > 0)
            {
                std::fprintf(stderr, " | frametime avg %.2f ms max %.2f ms", s.frametime_sum / s.frametime_samples, s.frametime_max);
            }
            std::fprintf(stderr, "\n");
            if (i < clock_skew.size())
            {
                const ClockSkewResult & c = clock_skew[i];
                if (!c.found)
                {
                    std::fprintf(stderr, "v%zu clock skew: nothing filtered (%s)\n", i + 1, c.reason.c_str());
                    continue;
                }
                const double fps = recorded_framerates[i];
                std::fprintf(stderr, "v%zu clock skew: a %.3f Hz source in the %.3f fps recording - one interval %s every %d frames (%.2f s);"
                                     " %zu episodes filtered at", i + 1, c.source_fps, fps, c.sign > 0 ? "too many" : "too few"
                           , c.period, c.period / fps, c.episodes.size());
                for (size_t e = 0; e < c.episodes.size(); ++e)
                {
                    std::fprintf(stderr, "%s %.1f-%.1f", e == 0 ? "" : ",", c.episodes[e].first / fps, c.episodes[e].second / fps);
                }
                std::fprintf(stderr, " s\n");
            }
        }
        for (size_t i = 0; i < motion_analyzers.size(); ++i)
        {
            const MotionAnalyzer::Summary & m = motion_summaries[i];
            // share of the new frames whose motion could be measured (duplicates are easy and say little)
            const double measurable = m.new_frames > 0 ? 100.0 * static_cast<double>(m.measurable_new_frames) / static_cast<double>(m.new_frames) : 0.0;
            std::fprintf(stderr, "v%zu motion: measured %lld of %lld new frames (%.1f%%)", i + 1, m.measurable_new_frames, m.new_frames, measurable);
            if (m.animation_frames > 0)
            {
                // per second values are robust against single long steps, short videos only have the overall value
                if (m.perceived_fps_median > 0.0)
                {
                    std::fprintf(stderr, " | %.2f new frames/s | perceived smoothness %.2f fps (10%% low %.2f, whole video %.2f)"
                               , m.motion_fps, m.perceived_fps_median, m.perceived_fps_low, m.perceived_fps);
                }
                else
                {
                    std::fprintf(stderr, " | %.2f new frames/s | perceived smoothness %.2f fps", m.motion_fps, m.perceived_fps);
                }
                std::fprintf(stderr, " | animation error median %.2f ms avg %.2f ms p95 %.2f ms max %.2f ms (spread of the votes on a step: %.2f ms)"
                           , m.median_abs_error_ms, m.mean_abs_error_ms, m.p95_abs_error_ms, m.max_abs_error_ms, m.median_step_spread_ms);
                if (options.async_capture)
                {
                    std::fprintf(stderr, " | asynchronous capture: %lld pictures re-timed by an interval, %lld lost pictures counted in"
                                         " (%lld of them after a repeated picture)"
                               , m.retimed_frames, m.lost_frames, m.lost_after_repeat);
                }
                else if (m.interval_errors_short >= 0.05 && m.interval_errors_long >= 0.05)
                {
                    // what a capture out of step with the output looks like; a game with uneven pacing too
                    std::fprintf(stderr, "\nv%zu note: %.0f%% of the animation errors are about one whole interval (%.1f ms), early and late alike."
                                         " A capture that doesn't run in step with the game's output (variable refresh rate, a capture program"
                                         " with its own clock) does that: if this is one, see --async-capture"
                               , i + 1, 100.0 * (m.interval_errors_short + m.interval_errors_long), 1000.0 / recorded_framerates[i]);
                }
            }
            else
            {
                std::fprintf(stderr, " | not enough coherent motion to measure the animation");
            }
            std::fprintf(stderr, "\n");
        }
        for (size_t i = 0; i < resolution_analyzers.size(); ++i) print_resolution(i, resolution_analyzers[i].summary());
    }
    const std::string warning = pipeline.warning();
    if (!warning.empty()) std::fprintf(stderr, "%s\n", warning.c_str());
    return 0;
}

//! --ignore-clock-skew with --motion-video: the filtered values are only known once the whole video is analyzed,
//! so the video is analyzed first (CSV, summary, everything but the motion videos) and read a second time to
//! draw the motion videos with those values. Otherwise one pass
int run_passes(const Options & options)
{
    if (!options.ignore_clock_skew || options.motion_videos.empty()) return run(options);
    Options analysis = options;
    analysis.motion_videos.clear();
    std::vector<std::vector<StoredValues>> rows;
    const int result = run(analysis, &rows);
    if (result != 0) return result;

    Options render = options;
    render.ignore_clock_skew = false;
    render.print_summary = false;
    render.tears_path.clear();
    render.resolution = false;
    render.resolution_videos.clear();
    render.resolution_data.clear();
    return run(render, nullptr, &rows);
}

} // namespace
} // namespace trdrop

#ifdef TRDROP_LIBRARY
int trdrop_run(int argc, char ** argv, const std::function<bool(long long, double)> & progress)
#else
int main(int argc, char ** argv)
#endif
{
#ifdef _WIN32
    // the manifest makes UTF-8 the process code page, the console has to print it as UTF-8 too
    SetConsoleOutputCP(CP_UTF8);
#endif
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    try
    {
        trdrop::Options options;
        if (!trdrop::parse_arguments(argc, argv, options)) return 0;
#ifdef TRDROP_LIBRARY
        options.progress = progress;
#endif
        return trdrop::run_passes(options);
    }
    catch (const std::invalid_argument & e)
    {
        std::fprintf(stderr, "trdrop-pp: %s\n", e.what());
        return 2;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "\ntrdrop-pp: error: %s\n", e.what());
        return 1;
    }
}
