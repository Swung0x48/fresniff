#include "osd_font.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "osd_font_data.h"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#define STB_TRUETYPE_IMPLEMENTATION
#if defined(__has_include)
#if __has_include("stb_truetype.h")
#include "stb_truetype.h"
#else
#include "../third_party/stb_truetype.h"
#endif
#else
#include "stb_truetype.h"
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace fresniff {

namespace {

constexpr int kFirstChar = 32;
constexpr int kLastChar = 126;
constexpr int kGlyphCount = kLastChar - kFirstChar + 1;
constexpr int kSubpixelPhases = 4;

inline uint8_t clamp_u8(float value)
{
    if (value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return static_cast<uint8_t>(value + 0.5f);
}

// Exact Porter-Duff source-over into premultiplied RGBA8 (byte order R, G, B, A).
inline void blend_premultiplied(cv::Vec4b & dst, const OsdColor & color, float coverage)
{
    const float sa = std::clamp(color.alpha, 0.0f, 1.0f) * coverage;
    if (sa <= 0.0f) return;

    const float inv = 1.0f - sa;
    const uint8_t out_a = clamp_u8(255.0f * sa + static_cast<float>(dst[3]) * inv);
    const uint8_t out_r = std::min(out_a, clamp_u8(static_cast<float>(color.r) * sa + static_cast<float>(dst[0]) * inv));
    const uint8_t out_g = std::min(out_a, clamp_u8(static_cast<float>(color.g) * sa + static_cast<float>(dst[1]) * inv));
    const uint8_t out_b = std::min(out_a, clamp_u8(static_cast<float>(color.b) * sa + static_cast<float>(dst[2]) * inv));
    dst[0] = out_r;
    dst[1] = out_g;
    dst[2] = out_b;
    dst[3] = out_a;
}

struct SubpixelBitmap {
    int x0 = 0;
    int y0 = 0;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

struct CachedGlyph {
    int glyph_index = 0;
    float advance = 0.0f;
    std::array<SubpixelBitmap, kSubpixelPhases> phases;
};

uint32_t next_utf8_codepoint(const std::string & text, size_t & pos)
{
    const size_t n = text.size();
    if (pos >= n) return 0;
    const uint8_t b0 = static_cast<uint8_t>(text[pos]);
    if (b0 < 0x80u)
    {
        ++pos;
        return b0;
    }
    auto cont = [&](size_t idx) -> uint8_t { return static_cast<uint8_t>(text[idx]); };
    if ((b0 & 0xE0u) == 0xC0u && pos + 1 < n)
    {
        const uint8_t b1 = cont(pos + 1);
        if ((b1 & 0xC0u) == 0x80u)
        {
            const uint32_t cp = (static_cast<uint32_t>(b0 & 0x1Fu) << 6) | static_cast<uint32_t>(b1 & 0x3Fu);
            pos += 2;
            return cp >= 0x80u ? cp : 0xFFFDu;
        }
    }
    else if ((b0 & 0xF0u) == 0xE0u && pos + 2 < n)
    {
        const uint8_t b1 = cont(pos + 1);
        const uint8_t b2 = cont(pos + 2);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u)
        {
            const uint32_t cp = (static_cast<uint32_t>(b0 & 0x0Fu) << 12)
                              | (static_cast<uint32_t>(b1 & 0x3Fu) << 6)
                              | static_cast<uint32_t>(b2 & 0x3Fu);
            pos += 3;
            return (cp >= 0x800u && (cp < 0xD800u || cp > 0xDFFFu)) ? cp : 0xFFFDu;
        }
    }
    else if ((b0 & 0xF8u) == 0xF0u && pos + 3 < n)
    {
        const uint8_t b1 = cont(pos + 1);
        const uint8_t b2 = cont(pos + 2);
        const uint8_t b3 = cont(pos + 3);
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u)
        {
            const uint32_t cp = (static_cast<uint32_t>(b0 & 0x07u) << 18)
                              | (static_cast<uint32_t>(b1 & 0x3Fu) << 12)
                              | (static_cast<uint32_t>(b2 & 0x3Fu) << 6)
                              | static_cast<uint32_t>(b3 & 0x3Fu);
            pos += 4;
            return (cp >= 0x10000u && cp <= 0x10FFFFu) ? cp : 0xFFFDu;
        }
    }
    ++pos;
    return 0xFFFDu;
}

} // namespace

struct OsdFontFace::Impl {
    float pixel_height = 0.0f;
    float ascent = 0.0f;
    float descent = 0.0f;
    float cap_height = 0.0f;
    float visual_center_above_baseline = 0.0f;
    float digit_cell_width = 0.0f;
    float dot_cell_width = 0.0f;
    float dash_cell_width = 0.0f;
    std::array<CachedGlyph, kGlyphCount> glyphs;
    std::unordered_map<uint32_t, CachedGlyph> extra_glyphs;
    std::vector<float> kern; // kGlyphCount * kGlyphCount, row = prev, col = curr

    const CachedGlyph * find_glyph(uint32_t cp) const
    {
        if (cp >= static_cast<uint32_t>(kFirstChar) && cp <= static_cast<uint32_t>(kLastChar))
        {
            return &glyphs[static_cast<size_t>(cp - static_cast<uint32_t>(kFirstChar))];
        }
        const auto it = extra_glyphs.find(cp);
        return it != extra_glyphs.end() ? &it->second : nullptr;
    }

    void blit_glyph(cv::Mat & rgba_pm, const CachedGlyph & glyph, float pen_x, int base_y, const OsdColor & color) const
    {
        float gx = std::floor(pen_x);
        int phase = static_cast<int>(std::floor((pen_x - gx) * static_cast<float>(kSubpixelPhases) + 0.5f));
        if (phase >= kSubpixelPhases)
        {
            gx += 1.0f;
            phase = 0;
        }
        else if (phase < 0)
        {
            phase = 0;
        }

        const SubpixelBitmap & bmp = glyph.phases[static_cast<size_t>(phase)];
        if (bmp.width <= 0 || bmp.height <= 0 || bmp.pixels.empty()) return;

        const int dst_x0 = static_cast<int>(gx) + bmp.x0;
        const int dst_y0 = base_y + bmp.y0;
        const int x_start = std::max(0, dst_x0);
        const int y_start = std::max(0, dst_y0);
        const int x_end = std::min(rgba_pm.cols, dst_x0 + bmp.width);
        const int y_end = std::min(rgba_pm.rows, dst_y0 + bmp.height);
        if (x_start >= x_end || y_start >= y_end) return;

        for (int y = y_start; y < y_end; ++y)
        {
            const uint8_t * src_row = bmp.pixels.data() + static_cast<size_t>(y - dst_y0) * static_cast<size_t>(bmp.width);
            cv::Vec4b * dst_row = rgba_pm.ptr<cv::Vec4b>(y);
            for (int x = x_start; x < x_end; ++x)
            {
                const uint8_t cov_u8 = src_row[x - dst_x0];
                if (cov_u8 == 0) continue;
                blend_premultiplied(dst_row[x], color, static_cast<float>(cov_u8) * (1.0f / 255.0f));
            }
        }
    }
};

OsdFontFace::OsdFontFace(OsdFontWeight weight, float pixel_height)
{
    if (!(pixel_height > 0.0f) || !std::isfinite(pixel_height)) return;

    const uint8_t * font_data = nullptr;
    size_t font_size = 0;
    switch (weight)
    {
        case OsdFontWeight::Bold:
            font_data = kLeagueSpartanBoldTtf;
            font_size = kLeagueSpartanBoldTtfSize;
            break;
        case OsdFontWeight::Medium:
            font_data = kLeagueSpartanMediumTtf;
            font_size = kLeagueSpartanMediumTtfSize;
            break;
        case OsdFontWeight::NotoSansScMedium:
            font_data = kNotoSansScMediumTtf;
            font_size = kNotoSansScMediumTtfSize;
            break;
    }
    if (!font_data || font_size == 0) return;

    stbtt_fontinfo info {};
    if (!stbtt_InitFont(&info, font_data, 0)) return;

    auto impl = std::make_shared<Impl>();
    impl->pixel_height = pixel_height;

    const float scale = (weight == OsdFontWeight::NotoSansScMedium)
                      ? stbtt_ScaleForMappingEmToPixels(&info, pixel_height)
                      : stbtt_ScaleForPixelHeight(&info, pixel_height);
    int ascent_i = 0;
    int descent_i = 0;
    int line_gap_i = 0;
    stbtt_GetFontVMetrics(&info, &ascent_i, &descent_i, &line_gap_i);
    impl->ascent = static_cast<float>(ascent_i) * scale;
    impl->descent = static_cast<float>(descent_i) * scale;

    int cap_x0 = 0;
    int cap_y0 = 0;
    int cap_x1 = 0;
    int cap_y1 = 0;
    if (stbtt_GetCodepointBox(&info, 'H', &cap_x0, &cap_y0, &cap_x1, &cap_y1) && cap_y1 > 0)
    {
        impl->cap_height = static_cast<float>(cap_y1) * scale;
    }
    else
    {
        impl->cap_height = 0.7174f * pixel_height;
    }

    impl->visual_center_above_baseline = 0.5f * impl->cap_height;
    if (weight == OsdFontWeight::NotoSansScMedium)
    {
        int cjk_x0 = 0, cjk_y0 = 0, cjk_x1 = 0, cjk_y1 = 0;
        if (stbtt_GetCodepointBox(&info, 0x5E27, &cjk_x0, &cjk_y0, &cjk_x1, &cjk_y1) && cjk_y1 > cjk_y0)
        {
            impl->visual_center_above_baseline = 0.5f * static_cast<float>(cjk_y0 + cjk_y1) * scale;
        }
    }

    auto cache_glyph = [&](uint32_t cp, CachedGlyph & glyph) {
        glyph.glyph_index = stbtt_FindGlyphIndex(&info, static_cast<int>(cp));

        int adv_i = 0;
        int lsb_i = 0;
        stbtt_GetGlyphHMetrics(&info, glyph.glyph_index, &adv_i, &lsb_i);
        glyph.advance = static_cast<float>(adv_i) * scale;

        if (glyph.glyph_index == 0) return;

        for (int p = 0; p < kSubpixelPhases; ++p)
        {
            const float shift_x = static_cast<float>(p) / static_cast<float>(kSubpixelPhases);
            int ix0 = 0;
            int iy0 = 0;
            int ix1 = 0;
            int iy1 = 0;
            stbtt_GetGlyphBitmapBoxSubpixel(&info, glyph.glyph_index, scale, scale, shift_x, 0.0f, &ix0, &iy0, &ix1, &iy1);
            const int w = ix1 - ix0;
            const int h = iy1 - iy0;
            SubpixelBitmap & bmp = glyph.phases[static_cast<size_t>(p)];
            bmp.x0 = ix0;
            bmp.y0 = iy0;
            bmp.width = std::max(0, w);
            bmp.height = std::max(0, h);
            if (bmp.width > 0 && bmp.height > 0)
            {
                bmp.pixels.assign(static_cast<size_t>(bmp.width) * static_cast<size_t>(bmp.height), 0);
                stbtt_MakeGlyphBitmapSubpixel(&info, bmp.pixels.data(), bmp.width, bmp.height, bmp.width, scale, scale, shift_x, 0.0f, glyph.glyph_index);
            }
        }
    };

    for (int i = 0; i < kGlyphCount; ++i)
    {
        cache_glyph(static_cast<uint32_t>(kFirstChar + i), impl->glyphs[static_cast<size_t>(i)]);
    }
    if (weight == OsdFontWeight::NotoSansScMedium)
    {
        for (size_t i = 0; i < kNotoSansScCodepointCount; ++i)
        {
            const uint32_t cp = kNotoSansScCodepoints[i];
            cache_glyph(cp, impl->extra_glyphs[cp]);
        }
    }

    // Pre-cache horizontal kerning (OpenType GPOS / kern) for printable ASCII pairs.
    impl->kern.assign(static_cast<size_t>(kGlyphCount) * static_cast<size_t>(kGlyphCount), 0.0f);
    for (int i = 0; i < kGlyphCount; ++i)
    {
        const int g1 = impl->glyphs[static_cast<size_t>(i)].glyph_index;
        if (g1 == 0) continue;
        for (int j = 0; j < kGlyphCount; ++j)
        {
            const int g2 = impl->glyphs[static_cast<size_t>(j)].glyph_index;
            if (g2 == 0) continue;
            const int k = stbtt_GetGlyphKernAdvance(&info, g1, g2);
            if (k != 0)
            {
                impl->kern[static_cast<size_t>(i) * static_cast<size_t>(kGlyphCount) + static_cast<size_t>(j)] = static_cast<float>(k) * scale;
            }
        }
    }

    // Synthetic tabular cell widths for flicker-free live numeric readouts.
    float max_digit = 0.0f;
    for (int d = '0'; d <= '9'; ++d)
    {
        max_digit = std::max(max_digit, impl->glyphs[static_cast<size_t>(d - kFirstChar)].advance);
    }
    impl->digit_cell_width = max_digit;
    impl->dot_cell_width = impl->glyphs[static_cast<size_t>('.' - kFirstChar)].advance;
    impl->dash_cell_width = impl->glyphs[static_cast<size_t>('-' - kFirstChar)].advance;

    _impl = std::move(impl);
}

bool OsdFontFace::valid() const
{
    return static_cast<bool>(_impl);
}

float OsdFontFace::pixel_height() const
{
    return _impl ? _impl->pixel_height : 0.0f;
}

float OsdFontFace::ascent() const
{
    return _impl ? _impl->ascent : 0.0f;
}

float OsdFontFace::descent() const
{
    return _impl ? _impl->descent : 0.0f;
}

float OsdFontFace::cap_height() const
{
    return _impl ? _impl->cap_height : 0.0f;
}

float OsdFontFace::visual_center_above_baseline() const
{
    return _impl ? _impl->visual_center_above_baseline : 0.0f;
}

float OsdFontFace::digit_cell_width() const
{
    return _impl ? _impl->digit_cell_width : 0.0f;
}

float OsdFontFace::dot_cell_width() const
{
    return _impl ? _impl->dot_cell_width : 0.0f;
}

float OsdFontFace::dash_cell_width() const
{
    return _impl ? _impl->dash_cell_width : 0.0f;
}

float OsdFontFace::measure_text(const std::string & text) const
{
    if (!_impl || text.empty()) return 0.0f;
    float width = 0.0f;
    int prev_ascii_idx = -1;
    size_t pos = 0;
    while (pos < text.size())
    {
        const uint32_t cp = next_utf8_codepoint(text, pos);
        const CachedGlyph * glyph = _impl->find_glyph(cp);
        if (!glyph)
        {
            prev_ascii_idx = -1;
            continue;
        }
        if (cp >= static_cast<uint32_t>(kFirstChar) && cp <= static_cast<uint32_t>(kLastChar))
        {
            const int idx = static_cast<int>(cp) - kFirstChar;
            if (prev_ascii_idx >= 0)
            {
                width += _impl->kern[static_cast<size_t>(prev_ascii_idx) * static_cast<size_t>(kGlyphCount) + static_cast<size_t>(idx)];
            }
            prev_ascii_idx = idx;
        }
        else
        {
            prev_ascii_idx = -1;
        }
        width += glyph->advance;
    }
    return width;
}

float OsdFontFace::measure_tabular(const std::string & text) const
{
    if (!_impl || text.empty()) return 0.0f;
    float width = 0.0f;
    for (const char ch : text)
    {
        const int uc = static_cast<unsigned char>(ch);
        if (uc >= '0' && uc <= '9')
        {
            width += _impl->digit_cell_width;
        }
        else if (uc == '.')
        {
            width += _impl->dot_cell_width;
        }
        else if (uc == '-')
        {
            width += _impl->dash_cell_width;
        }
        else if (uc >= kFirstChar && uc <= kLastChar)
        {
            width += _impl->glyphs[static_cast<size_t>(uc - kFirstChar)].advance;
        }
    }
    return width;
}

float OsdFontFace::draw_text(cv::Mat & rgba_pm, float x, float baseline_y, const std::string & text, const OsdColor & color) const
{
    if (!_impl || text.empty()) return x;
    if (rgba_pm.empty() || rgba_pm.type() != CV_8UC4 || color.alpha <= 0.0f) return x + measure_text(text);

    const int base_y = static_cast<int>(std::lround(baseline_y));
    float pen_x = x;
    int prev_ascii_idx = -1;
    size_t pos = 0;
    while (pos < text.size())
    {
        const uint32_t cp = next_utf8_codepoint(text, pos);
        const CachedGlyph * glyph = _impl->find_glyph(cp);
        if (!glyph)
        {
            prev_ascii_idx = -1;
            continue;
        }
        if (cp >= static_cast<uint32_t>(kFirstChar) && cp <= static_cast<uint32_t>(kLastChar))
        {
            const int idx = static_cast<int>(cp) - kFirstChar;
            if (prev_ascii_idx >= 0)
            {
                pen_x += _impl->kern[static_cast<size_t>(prev_ascii_idx) * static_cast<size_t>(kGlyphCount) + static_cast<size_t>(idx)];
            }
            prev_ascii_idx = idx;
        }
        else
        {
            prev_ascii_idx = -1;
        }
        _impl->blit_glyph(rgba_pm, *glyph, pen_x, base_y, color);
        pen_x += glyph->advance;
    }
    return pen_x;
}

float OsdFontFace::draw_text_right(cv::Mat & rgba_pm, float right_x, float baseline_y, const std::string & text, const OsdColor & color) const
{
    const float start_x = right_x - measure_text(text);
    draw_text(rgba_pm, start_x, baseline_y, text, color);
    return start_x;
}

float OsdFontFace::draw_tabular(cv::Mat & rgba_pm, float x, float baseline_y, const std::string & text, const OsdColor & color) const
{
    if (!_impl || text.empty()) return x;
    if (rgba_pm.empty() || rgba_pm.type() != CV_8UC4 || color.alpha <= 0.0f) return x + measure_tabular(text);

    const int base_y = static_cast<int>(std::lround(baseline_y));
    float pen_x = x;
    for (const char ch : text)
    {
        const int uc = static_cast<unsigned char>(ch);
        if (uc < kFirstChar || uc > kLastChar) continue;
        const int idx = uc - kFirstChar;
        const CachedGlyph & glyph = _impl->glyphs[static_cast<size_t>(idx)];
        const float adv = glyph.advance;
        if (uc >= '0' && uc <= '9')
        {
            const float cell = _impl->digit_cell_width;
            _impl->blit_glyph(rgba_pm, glyph, pen_x + 0.5f * (cell - adv), base_y, color);
            pen_x += cell;
        }
        else if (uc == '.')
        {
            _impl->blit_glyph(rgba_pm, glyph, pen_x, base_y, color);
            pen_x += _impl->dot_cell_width;
        }
        else if (uc == '-')
        {
            _impl->blit_glyph(rgba_pm, glyph, pen_x, base_y, color);
            pen_x += _impl->dash_cell_width;
        }
        else
        {
            _impl->blit_glyph(rgba_pm, glyph, pen_x, base_y, color);
            pen_x += adv;
        }
    }
    return pen_x;
}

void draw_rounded_rect_rgba(cv::Mat & rgba_pm, float x, float y, float w, float h, float radius, const OsdColor & color)
{
    if (rgba_pm.empty() || rgba_pm.type() != CV_8UC4) return;
    if (!(w > 0.0f) || !(h > 0.0f) || !(color.alpha > 0.0f)) return;

    const float hx = 0.5f * w;
    const float hy = 0.5f * h;
    const float cx = x + hx;
    const float cy = y + hy;
    const float r = std::clamp(radius, 0.0f, std::min(hx, hy));
    const float bx = hx - r;
    const float by = hy - r;

    const int x_start = std::max(0, static_cast<int>(std::floor(x - 1.0f)));
    const int y_start = std::max(0, static_cast<int>(std::floor(y - 1.0f)));
    const int x_end = std::min(rgba_pm.cols, static_cast<int>(std::ceil(x + w + 1.0f)));
    const int y_end = std::min(rgba_pm.rows, static_cast<int>(std::ceil(y + h + 1.0f)));
    if (x_start >= x_end || y_start >= y_end) return;

    for (int py = y_start; py < y_end; ++py)
    {
        cv::Vec4b * row = rgba_pm.ptr<cv::Vec4b>(py);
        const float qy = std::abs(static_cast<float>(py) + 0.5f - cy) - by;
        const float qy_pos = std::max(qy, 0.0f);
        for (int px = x_start; px < x_end; ++px)
        {
            const float qx = std::abs(static_cast<float>(px) + 0.5f - cx) - bx;
            const float qx_pos = std::max(qx, 0.0f);
            const float d = std::hypot(qx_pos, qy_pos) + std::min(std::max(qx, qy), 0.0f) - r;
            const float cov = std::clamp(0.5f - d, 0.0f, 1.0f);
            if (cov > 0.0f) blend_premultiplied(row[px], color, cov);
        }
    }
}

} // namespace fresniff
