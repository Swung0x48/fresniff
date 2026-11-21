#pragma once

#include <cstddef>
#include <cstdint>

namespace trdrop {

// Embedded unmodified League Spartan static TrueType fonts (SIL OFL 1.1).
// Source: third_party/fonts/LeagueSpartan-Bold.ttf & LeagueSpartan-Medium.ttf
extern const uint8_t kLeagueSpartanBoldTtf[];
extern const size_t kLeagueSpartanBoldTtfSize;

extern const uint8_t kLeagueSpartanMediumTtf[];
extern const size_t kLeagueSpartanMediumTtfSize;

// Embedded subsetted Noto Sans SC Medium static TrueType font (SIL OFL 1.1).
// Source: third_party/fonts/NotoSansSC-Medium-Subset.ttf
extern const uint8_t kNotoSansScMediumTtf[];
extern const size_t kNotoSansScMediumTtfSize;
extern const uint32_t kNotoSansScCodepoints[];
extern const size_t kNotoSansScCodepointCount;

} // namespace trdrop
