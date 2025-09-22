#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__clang__) && __has_feature(bfloat16_type)
#define HAVE_BF16 1
#else
#define HAVE_BF16 0
#endif

static inline uint32_t f32_bits(float f) {
    uint32_t u; memcpy(&u, &f, sizeof u); return u;
}
static inline float f32_from_bits(uint32_t u) {
    float f; memcpy(&f, &u, sizeof f); return f;
}

static inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t x = f32_bits(f);
    uint32_t sign = x & 0x80000000u;
    uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t frac = x & 0x007FFFFFu;

    if (exp == 0xFFu) {
        // Inf or NaN
        if (frac == 0) {
            // infinity: just shift sign|exp
            return (uint16_t)((sign >> 16) | (0xFFu << 7));
        } else {
            // NaN: propagate payload and ensure quiet bit
            uint16_t mant = (uint16_t)((frac >> 16) & 0x7Fu);
            if ((mant & 0x7Fu) == 0) mant = 0x40u; // canonical qNaN if payload halves to 0
            return (uint16_t)((sign >> 16) | (0xFFu << 7) | mant);
        }
    }
    // Nearest-even: add bias and chop
    uint32_t roundBias = 0x7FFFu + ((x >> 16) & 1u);
    uint32_t y = x + roundBias;
    return (uint16_t)(y >> 16);
}

static inline float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    return f32_from_bits(u);
}

int main(void) {
    float vals[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.1415926f, -123.456f,
        INFINITY, -INFINITY, NAN
    };
    int N = (int)(sizeof(vals)/sizeof(vals[0]));
    printf("HAVE_BF16=%d\n", HAVE_BF16);
    for (int i = 0; i < N; ++i) {
        float f = vals[i];
        uint32_t fbits = f32_bits(f);
        uint16_t mbits = f32_to_bf16_bits(f);
        float mback = bf16_bits_to_f32(mbits);
        printf("manual f32=%08x mbits=%04x back=%08x", fbits, mbits, f32_bits(mback));
#if HAVE_BF16
        __bf16 hb;
        memcpy(&hb, &mbits, sizeof(uint16_t)); // use same bits to avoid codegen surprises
        // Alternatively, let clang convert: hb = (__bf16)f;
        __bf16 hb2 = (__bf16)f;
        uint16_t libbits; memcpy(&libbits, &hb2, sizeof(uint16_t));
        float libback = (float)hb2;
        printf(" libbits=%04x libback=%08x", libbits, f32_bits(libback));
#endif
        printf("\n");
    }
    return 0;
}

