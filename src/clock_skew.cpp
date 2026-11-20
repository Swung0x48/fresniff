#include "clock_skew.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace trdrop {

namespace {

//! the cadence is the mean length of the frames within a second before and after, if that is this close to a
//! whole number of intervals
constexpr double kCadenceTolerance = 0.25;
//! differences more than this many seconds apart belong to different episodes
constexpr double kEpisodeGapSeconds = 2.0;
//! with motion: this share of the measured differences of an episode has a step closer to the cadence
constexpr double kMotionAgreement = 0.75;
//! an episode lies on a lattice point if it is within this share of the period of it
constexpr double kLatticeTolerance = 0.1;
//! a lattice point counts as steady if the cadence is known for this share of the frames near it
constexpr double kSteadyShare = 0.8;

struct Run {
    long long end = 0;
    int length = 0;
    double step = NAN;
    int cadence = 0; // 0: none
    bool brk = false;
};

struct Episode {
    std::vector<std::pair<size_t, int>> differences; // run, length - cadence
    double center = 0.0;
    int net = 0;
    bool skew = false; // alternating, net +-1, the motion agrees
};

//! lattice points (index n of x0 + n * period) near which exactly one of `centers` lies, with that center
std::map<long long, double> lattice_hits(const std::vector<double> & centers, int period, double x0)
{
    const double tolerance = kLatticeTolerance * period;
    std::map<long long, std::vector<double>> near;
    for (double x : centers)
    {
        const long long n = std::llround((x - x0) / period);
        if (std::fabs(x - x0 - static_cast<double>(n) * period) <= tolerance) near[n].push_back(x);
    }
    std::map<long long, double> hits;
    for (const auto & [n, xs] : near)
    {
        if (xs.size() == 1) hits[n] = xs[0];
    }
    return hits;
}

} // namespace

ClockSkewResult detect_clock_skew(const std::vector<double> & diff, const std::vector<double> & step, double fps)
{
    ClockSkewResult result;
    result.correction.assign(diff.size(), 0);

    // the lengths of the new frames
    std::vector<long long> news;
    for (size_t t = 0; t < diff.size(); ++t)
    {
        if (!std::isnan(diff[t]) && diff[t] > 0.0) news.push_back(static_cast<long long>(t));
    }
    std::vector<Run> runs;
    for (size_t i = 1; i < news.size(); ++i)
    {
        Run run;
        run.end = news[i];
        run.length = static_cast<int>(news[i] - news[i - 1]);
        if (!step.empty())
        {
            double sum = 0.0;
            for (long long t = news[i - 1] + 1; t <= news[i]; ++t) sum += step[static_cast<size_t>(t)];
            run.step = sum;
        }
        runs.push_back(run);
    }
    if (runs.empty())
    {
        result.reason = "no new frames";
        return result;
    }

    // cadence: mean length within a second before and after
    const long long second = std::llround(fps);
    size_t lo = 0, hi = 0;
    long long sum = 0;
    for (Run & run : runs)
    {
        while (hi < runs.size() && runs[hi].end <= run.end + second) sum += runs[hi++].length;
        while (runs[lo].end < run.end - second) sum -= runs[lo++].length;
        const double mean = static_cast<double>(sum) / static_cast<double>(hi - lo);
        const long long cadence = std::llround(mean);
        run.cadence = cadence >= 1 && std::fabs(mean - static_cast<double>(cadence)) < kCadenceTolerance ? static_cast<int>(cadence) : 0;
    }

    // frames off by one interval; two lengths that add up to the cadence (a split frame) are left alone
    std::vector<std::pair<size_t, int>> differences;
    for (size_t i = 0; i < runs.size();)
    {
        const int cadence = runs[i].cadence;
        if (cadence == 0) { runs[i].brk = true; ++i; continue; }
        const int length = runs[i].length;
        if (length < cadence && i + 1 < runs.size() && length + runs[i + 1].length == cadence && runs[i + 1].length < cadence)
        {
            i += 2;
            continue;
        }
        const int difference = length - cadence;
        if (difference == 1 || difference == -1) differences.push_back({ i, difference });
        else if (difference != 0) runs[i].brk = true;
        ++i;
    }
    std::vector<int> breaks(runs.size() + 1, 0); // prefix sums
    for (size_t i = 0; i < runs.size(); ++i) breaks[i + 1] = breaks[i] + (runs[i].brk ? 1 : 0);

    // episodes
    const double gap = kEpisodeGapSeconds * fps;
    std::vector<Episode> episodes;
    for (const auto & difference : differences)
    {
        if (!episodes.empty())
        {
            const size_t previous = episodes.back().differences.back().first;
            if (static_cast<double>(runs[difference.first].end - runs[previous].end) <= gap
                && breaks[difference.first] - breaks[previous + 1] == 0)
            {
                episodes.back().differences.push_back(difference);
                continue;
            }
        }
        episodes.emplace_back();
        episodes.back().differences.push_back(difference);
    }
    for (Episode & episode : episodes)
    {
        // a frame the game really showed an interval longer (or shorter) right next to the alternation - a hitch
        // in a game that drops frames - would spoil it: keep the longest alternating part, of an odd length (net
        // one interval), the frames left out stay as they were recorded
        std::vector<std::pair<size_t, int>> & run = episode.differences;
        size_t best_from = 0, best_length = 0, from = 0;
        for (size_t j = 1; j <= run.size(); ++j)
        {
            if (j < run.size() && run[j].second != run[j - 1].second) continue;
            if (j - from > best_length)
            {
                best_from = from;
                best_length = j - from;
            }
            from = j;
        }
        if (best_length < run.size())
        {
            if (best_length % 2 == 0) --best_length;
            // only frames of one sign before it and of one sign after it: hitches at its ends. More than one
            // alternating part (what a variable refresh rate looks like) stays as it is, and is no episode
            bool ends_only = true;
            for (size_t j = 1; j < run.size(); ++j)
            {
                const bool outside = j <= best_from || j > best_from + best_length;
                if (outside && run[j].second != run[j - 1].second && j != best_from + best_length) ends_only = false;
            }
            if (ends_only)
            {
                run = std::vector<std::pair<size_t, int>>(run.begin() + static_cast<std::ptrdiff_t>(best_from),
                                                         run.begin() + static_cast<std::ptrdiff_t>(best_from + best_length));
            }
        }
        bool alternating = true;
        int measured = 0, agreeing = 0;
        for (size_t j = 0; j < episode.differences.size(); ++j)
        {
            const auto & [i, d] = episode.differences[j];
            episode.net += d;
            if (j > 0 && episode.differences[j - 1].second == d) alternating = false;
            if (!std::isnan(runs[i].step))
            {
                ++measured;
                if (std::fabs(runs[i].step - runs[i].cadence) < std::fabs(runs[i].step - runs[i].length)) ++agreeing;
            }
        }
        const bool motion = measured == 0 || agreeing >= kMotionAgreement * measured;
        episode.skew = alternating && (episode.net == 1 || episode.net == -1) && motion;
        episode.center = 0.5 * static_cast<double>(runs[episode.differences.front().first].end + runs[episode.differences.back().first].end);
    }

    // one lattice for the whole video, fitted to the series
    auto centers = [&](int sign, bool series) {
        std::vector<double> xs;
        for (const Episode & episode : episodes)
        {
            if (!episode.skew || episode.net != sign) continue;
            if ((episode.differences.size() >= static_cast<size_t>(kClockSkewSeriesMin)) == series) xs.push_back(episode.center);
        }
        return xs;
    };
    bool fitted = false;
    size_t best_hits = 0, best_count = 0;
    double best_residual = 0.0, best_x0 = 0.0;
    int best_period = 0, best_sign = 0;
    for (int sign : { 1, -1 })
    {
        const std::vector<double> xs = centers(sign, true);
        if (xs.size() < static_cast<size_t>(kClockSkewMinEpisodes)) continue;
        for (int period = kClockSkewPeriodMin; period <= kClockSkewPeriodMax; ++period)
        {
            for (double x0 : xs)
            {
                const std::map<long long, double> hits = lattice_hits(xs, period, x0);
                double residual = 0.0;
                for (const auto & [n, x] : hits) residual += std::fabs(x - x0 - static_cast<double>(n) * period);
                if (!fitted || hits.size() > best_hits || (hits.size() == best_hits && -residual > -best_residual))
                {
                    fitted = true;
                    best_hits = hits.size();
                    best_residual = residual;
                    best_period = period;
                    best_x0 = x0;
                    best_sign = sign;
                    best_count = xs.size();
                }
            }
        }
    }
    char text[256];
    if (!fitted)
    {
        std::snprintf(text, sizeof(text), "no beat: fewer than %d episodes look like one", kClockSkewMinEpisodes);
        result.reason = text;
        return result;
    }

    // lattice points where an episode could be seen: the cadence known for most frames near them, none of them
    // off by more than an interval
    const double tolerance = kLatticeTolerance * best_period;
    const double first = static_cast<double>(runs.front().end), last = static_cast<double>(runs.back().end);
    size_t steady = 0;
    for (long long n = static_cast<long long>(std::floor((first - best_x0) / best_period));
         best_x0 + static_cast<double>(n) * best_period <= last; ++n)
    {
        const double x = best_x0 + static_cast<double>(n) * best_period;
        if (x < first) continue;
        size_t near = 0, known = 0;
        bool broken = false;
        for (const Run & run : runs)
        {
            if (std::fabs(static_cast<double>(run.end) - x) > tolerance) continue;
            ++near;
            if (run.cadence != 0) { ++known; if (run.brk) broken = true; }
        }
        if (near > 0 && known >= kSteadyShare * static_cast<double>(near) && !broken) ++steady;
    }
    const double occupancy = std::min(1.0, static_cast<double>(best_hits) / static_cast<double>(std::max<size_t>(1, steady)));
    const double purity = static_cast<double>(best_hits) / static_cast<double>(best_count);
    if (best_hits < static_cast<size_t>(kClockSkewMinEpisodes) || occupancy < kClockSkewMinOccupancy || purity < kClockSkewMinPurity)
    {
        std::snprintf(text, sizeof(text), "no beat: at best %zu episodes every %d frames (%.0f%% of the steady points, %.0f%% of the candidates)"
                    , best_hits, best_period, 100.0 * occupancy, 100.0 * purity);
        result.reason = text;
        return result;
    }

    // the series on the lattice, and single differences at lattice points without a series
    const std::map<long long, double> series = lattice_hits(centers(best_sign, true), best_period, best_x0);
    const std::map<long long, double> singles = lattice_hits(centers(best_sign, false), best_period, best_x0);
    std::vector<double> confirmed;
    for (const auto & [n, x] : series) confirmed.push_back(x);
    for (const auto & [n, x] : singles)
    {
        if (series.find(n) == series.end()) confirmed.push_back(x);
    }
    for (const Episode & episode : episodes)
    {
        if (!episode.skew || episode.net != best_sign) continue;
        if (std::find(confirmed.begin(), confirmed.end(), episode.center) == confirmed.end()) continue;
        // every frame of the episode is shown for the cadence: the alternation and the interval the clocks drifted
        for (const auto & [i, d] : episode.differences) result.correction[static_cast<size_t>(runs[i].end)] = -d;
        const Run & front = runs[episode.differences.front().first];
        result.episodes.push_back({ front.end - front.length, runs[episode.differences.back().first].end });
    }
    std::sort(result.episodes.begin(), result.episodes.end());
    result.found = true;
    result.period = best_period;
    result.sign = best_sign;
    result.source_fps = fps * static_cast<double>(best_period - best_sign) / static_cast<double>(best_period);
    return result;
}

std::vector<long long> clock_skew_times(const std::vector<double> & diff, const std::vector<int> & correction)
{
    std::vector<long long> times(diff.size());
    long long previous = -1; // row of the previous new frame
    size_t t = 0;
    for (size_t row = 0; row < diff.size(); ++row)
    {
        if (std::isnan(diff[row]) || diff[row] <= 0.0) continue;
        const long long current = static_cast<long long>(row);
        if (previous < 0)
        {
            for (; t <= row; ++t) times[t] = static_cast<long long>(t);
        }
        else
        {
            const long long length = std::max<long long>(1, current - previous + correction[row]);
            const long long start = times[static_cast<size_t>(previous)];
            for (; t < row; ++t) times[t] = start + std::min<long long>(static_cast<long long>(t) - previous, length - 1);
            times[row] = start + length;
            ++t;
        }
        previous = current;
    }
    for (; t < diff.size(); ++t)
    {
        times[t] = previous < 0 ? static_cast<long long>(t) : times[static_cast<size_t>(previous)] + (static_cast<long long>(t) - previous);
    }
    return times;
}

} // namespace trdrop
