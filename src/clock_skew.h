#pragma once

#include <string>
#include <vector>

namespace trdrop {

//! --ignore-clock-skew: a source at 59.94 Hz (60000 / 1001) recorded by a capture with its own clock at 60 fps (or
//! the other way round, or at a multiple of the rate) drifts by one interval every ~1001 recorded frames
//! (16.7 s at 60 fps). For a few seconds around each such point small differences in timing decide whether a
//! picture is recorded a tick earlier or later: the recorded lengths of the pictures alternate around their cadence
//! (a 30 fps game for 3, 1, 3, 1 instead of 2, 2 intervals), and the whole episode holds one interval more (or
//! less) than the source showed.
//!
//! Such an episode is recognized by
//!  - a run of new frames whose recorded lengths differ from the local cadence by exactly one interval, the
//!    differences alternating in sign, beginning and ending with the same sign (net one interval),
//!  - with motion: the animation steps of those frames are closer to the cadence than to the recorded lengths,
//!  - and the episodes lying on one lattice: one period (900 - 1101 recorded frames, the 1000 / 1001 ratio and
//!    the tolerance of the clocks), one phase, one sign for the whole video. The lattice is fitted to episodes
//!    with at least kClockSkewSeriesMin differences only (a single +1 is what a hitch looks like); it needs
//!    kClockSkewMinEpisodes of them, at kClockSkewMinOccupancy of the lattice points where the cadence is steady,
//!    and kClockSkewMinPurity of all such series of the same sign on it.
//! Two lengths adding up to the cadence (2 recorded as 1, 1) can't come from the clock and are left alone.
//! A cadence of 1 (a 60 fps game at 60 fps) can't alternate - a picture can't be recorded for 0 intervals -, so
//! nothing is filtered there.
constexpr int kClockSkewPeriodMin = 900;
constexpr int kClockSkewPeriodMax = 1101;
constexpr int kClockSkewSeriesMin = 3;
constexpr int kClockSkewMinEpisodes = 3;
constexpr double kClockSkewMinOccupancy = 0.4;
constexpr double kClockSkewMinPurity = 0.6;

struct ClockSkewResult {
    bool found = false;
    //! recorded frames per beat, +1: the recording has an interval too many per beat (it runs faster than the
    //! source), -1: one too few
    int period = 0;
    int sign = 0;
    //! the rate of the source this implies
    double source_fps = 0.0;
    //! filtered episodes: first and last row
    std::vector<std::pair<long long, long long>> episodes;
    //! per row: at a new frame of a filtered episode, the length of the frame before it after the correction
    //! minus the recorded one (-1, +1), else 0
    std::vector<int> correction;
    //! what was found if not `found`
    std::string reason;
};

//! diff: per row the frame difference (NaN where the row wasn't compared), step: per row the animation step
//! (NaN where unknown; empty without motion), fps: recorded framerate
ClockSkewResult detect_clock_skew(const std::vector<double> & diff, const std::vector<double> & step, double fps);

//! per row its time in intervals after the correction: a frame that was shortened loses its last duplicate rows
//! (they share the time of the row before), one that was lengthened is moved
std::vector<long long> clock_skew_times(const std::vector<double> & diff, const std::vector<int> & correction);

} // namespace trdrop
