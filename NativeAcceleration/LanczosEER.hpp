#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>   // std::iota
#include <cassert>

#ifndef M_PI
static constexpr float M_PI = 3.14159265358979323846f;
#endif

// ---------------------------------------------------------------------------
//  Event structure in *output-pixel* units (0-based, physical grid).
//  x,y can be fractional; weight is always 1.0 for EER.
// ---------------------------------------------------------------------------
struct EER_Event
{
    float x;
    float y;
};

// --------------------------------- helpers ---------------------------------
inline float sinc(float x)
{
    return x == 0.f ? 1.f : std::sin(M_PI * x) / (M_PI * x);
}

inline float lanczos(float x, int a = 3)
{
    x = std::fabs(x);
    return (x < a) ? sinc(x) * sinc(x / a) : 0.f;
}

// ---------------------------------------------------------------------------
//  5–term flat–top window (Harris 1978) * sinc
//  Radius a = 6 by default; set a larger value for an even flatter pass-band.
// ---------------------------------------------------------------------------
inline float flattop_sinc(float r, int a = 6)
{
    if (r >= a) return 0.f;                // outside support

    // Flat-top window coefficients (Harris, Table III; ? 0.001 dB ripple)
    constexpr float A0 = 1.00000f;
    constexpr float A1 = 1.93000f;
    constexpr float A2 = 1.29000f;
    constexpr float A3 = 0.38800f;
    constexpr float A4 = 0.03220f;

    const float t = r / static_cast<float>(a);    // 0 … 1
    const float p2 = 2.f * M_PI * t;

    const float window = A0
        - A1 * std::cos(p2)
        + A2 * std::cos(2.f * p2)
        - A3 * std::cos(3.f * p2)
        + A4 * std::cos(4.f * p2);

    return sinc(r) * window;
}

// width,height are the final output dimensions (4 k, 8 k …).
// tile_sz        – must be 32 for perfect  warp/block mapping later on GPU.
// a              – Lanczos radius; 3  ?  6×6 footprint.
// events         – already unpacked & converted to *physical* grid units.
// image_out      – caller-allocated float[width*height], *zero-initialised*.
//
// This function is *O(events + tiles*footprint)* and good enough for
// validation.  When you are happy with the numerics you can port the inner
// loop verbatim to CUDA (__shared__ tileBuf[] etc.).
//
void render_lanczos_tiled(const std::vector<EER_Event>& events,
    int  width, int height,
    int  tile_sz,
    int  a,
    float* image_out)
{
    assert(tile_sz == 32 && "GPU path assumes 32×32 blocks – keep host identical.");

    const int tiles_x = (width + tile_sz - 1) / tile_sz;
    const int tiles_y = (height + tile_sz - 1) / tile_sz;
    const int tile_count = tiles_x * tiles_y;

    // -----------------------------------------------------------------------
    //  1.  Build a list of event indices for every tile it overlaps.
    //      (events may appear in up to 4 lists – exactly what the GPU will see)
    // -----------------------------------------------------------------------
    std::vector<std::vector<int>> tile_events(tile_count);

    for (int ei = 0; ei < static_cast<int>(events.size()); ++ei)
    {
        const auto& e = events[ei];

        const int min_px = static_cast<int>(std::floor(e.x)) - a;
        const int max_px = static_cast<int>(std::floor(e.x)) + a;
        const int min_py = static_cast<int>(std::floor(e.y)) - a;
        const int max_py = static_cast<int>(std::floor(e.y)) + a;

        const int min_tx = std::max(0, (min_px) / tile_sz);
        const int max_tx = std::min(tiles_x - 1, (max_px) / tile_sz);
        const int min_ty = std::max(0, (min_py) / tile_sz);
        const int max_ty = std::min(tiles_y - 1, (max_py) / tile_sz);

        for (int ty = min_ty; ty <= max_ty; ++ty)
            for (int tx = min_tx; tx <= max_tx; ++tx)
            {
                //if (ty < 0 || ty >= tiles_y || tx < 0 || tx >= tiles_x)
                //    std::cout << "Error: Out of bounds" << std::endl;
                //try {
                    tile_events[ty * tiles_x + tx].push_back(ei);
                //}
                //catch (const std::bad_alloc& e) {
                //    std::cerr << "OOM after pushing event #" << ei << '\n';
                //    throw;                     // re-throw so behaviour is unchanged
                //}
            }
    }

    // -----------------------------------------------------------------------
    //  2.  Process tiles one-by-one, exactly as a CUDA block will do.
    // -----------------------------------------------------------------------
    std::vector<float> tile_buf(tile_sz * tile_sz, 0.f);

    std::vector<float> x_sincs(2 * a + 1);  // dynamic size based on radius
    std::vector<float> y_sincs(2 * a + 1);  // dynamic size based on radius

    for (int ty = 0; ty < tiles_y; ++ty)
        for (int tx = 0; tx < tiles_x; ++tx)
        {
            std::fill(tile_buf.begin(), tile_buf.end(), 0.f);      // clear “shared” memory

            const int tile_id = ty * tiles_x + tx;
            const int x0_global = tx * tile_sz;                    // inclusive
            const int y0_global = ty * tile_sz;

            // ---- iterate over *all* events that affect this tile ---------------
            for (int ei : tile_events[tile_id])
            {
                const auto& e = events[ei];

                const int cx = static_cast<int>(std::floor(e.x));
                const int cy = static_cast<int>(std::floor(e.y));

                // Precalculate y-axis sinc values with correct size (2*a + 1)
                for (int dy = -a; dy <= a; ++dy) 
                {
                    const float dyf = e.y - static_cast<float>(cy + dy);
                    y_sincs[dy + a] = sinc(dyf);
                }

                // Precalculate x-axis sinc values with correct size (2*a + 1)
                for (int dx = -a; dx <= a; ++dx) 
                {
                    const float dxf = e.x - static_cast<float>(cx + dx);
                    x_sincs[dx + a] = sinc(dxf);
                }

                // Scan window
                for (int dy = -a; dy <= a; ++dy)
                {
                    const int py = cy + dy;
                    if (py < y0_global || py >= y0_global + tile_sz) continue;

                    const int ly = py - y0_global;
                    const float y_sinc = y_sincs[dy + a];

                    for (int dx = -a; dx <= a; ++dx)
                    {
                        const int px = cx + dx;
                        if (px < x0_global || px >= x0_global + tile_sz) continue;

                        const int lx = px - x0_global;
                        const float x_sinc = x_sincs[dx + a];

                        // Separable filter - multiply components
                        const float w = x_sinc * y_sinc;
                        tile_buf[ly * tile_sz + lx] += w;
                    }
                }
            }

            // ---- scatter tile buffer back into the big image -------------------
            for (int ly = 0; ly < tile_sz; ++ly)
            {
                const int py = y0_global + ly;
                if (py >= height) break;

                float* line_dst = image_out + py * width + x0_global;
                const float* line_src = tile_buf.data() + ly * tile_sz;

                const int n = std::min(tile_sz, width - x0_global);

                for (int lx = 0; lx < n; ++lx)
                    line_dst[lx] += line_src[lx];          // accumulate, don’t overwrite
            }
        }
}

// ---------------------------------------------------------------------------
//  Convenience wrapper that converts packed EER 14-bit coordinates to
//  Event(x,y) with 0.25-pixel granularity *and* runs the renderer above.
//  positions, symbols  – the vectors you already fill in your I/O code.
// ---------------------------------------------------------------------------
inline void render_eer_frame_lanczos(const std::vector<uint32_t>& positions,
                                    const std::vector<uint8_t>& symbols,
                                    unsigned int                 n_electrons,
                                    int                          width,
                                    int                          height,
                                    float* image_out)        // caller-zeroed
{
    static constexpr int SENSOR_PHYS = 4096;          // physical sensor side length
    const float sx = static_cast<float>(width) / SENSOR_PHYS;
    const float sy = static_cast<float>(height) / SENSOR_PHYS;

    const int N = static_cast<int>(n_electrons);
    std::vector<EER_Event> events;
    //events.reserve(N);

    for (int i = 0; i < N; ++i)
    {
        // unpack raw quarter-pixel coordinates on the *sensor* grid
        const int  x_i = positions[i] & 0x0FFF;          // 0 … 4095
        const int  y_i = (positions[i] >> 12) & 0x0FFF;
        const int  sx_q = symbols[i] & 0x03;                    // sub-pixel (0 … 3)
        const int  sy_q = (symbols[i] & 0x0C) >> 2;

        const float x_phys = (x_i + sx_q * 0.25f);               // sensor units
        const float y_phys = (y_i + sy_q * 0.25f);

        // scale to the *output* grid
        events.push_back({ x_phys * sx,
                           y_phys * sy });
    }

	//events.push_back({ 28.0f, 33.0f });

    render_lanczos_tiled(events, width, height,
                        /*tile_sz =*/ 32,
                        /*a       =*/ 8,
                        image_out);
}