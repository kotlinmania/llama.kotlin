#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include "ggml-quants.h"

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void)file; (void)line; (void)fmt;
    abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type; (void)ne;
    return 0;
}
const char * ggml_type_name(enum ggml_type type) {
    (void)type;
    return "stub";
}
size_t ggml_type_size(enum ggml_type type) {
    (void)type;
    return 0;
}

#include "ggml-quants.c"

static void debug_make_qkx2_quants_subblock(const float *x) {
    const int n = 16;
    const int nmax = 3;
    const float rmin = -0.5f;
    const float rdelta = 0.1f;
    const int nstep = 15;
    const bool use_mad = true;

    uint8_t L[16];
    uint8_t Laux[16];
    float weights[16];

    float min = x[0];
    float max = x[0];
    weights[0] = fabsf(x[0]);
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        weights[i] = fabsf(x[i]);
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        printf("C debug: trivial block\n");
        return;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    printf("C debug: init scale=%f min=%f best=%f\n", scale, -min, best_mad);
    if (nstep >= 1) {
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
            if (is == 0) {
                printf("C debug: step=%d xi=%f l=%d\n", is, x[i], l);
            }
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (is == 0) {
            printf("C debug: step=%d sum_w=%f sum_l=%f sum_l2=%f bits=%08x %08x %08x\n", is, sum_w, sum_l, sum_l2,
                   *(uint32_t*)&sum_w, *(uint32_t*)&sum_l, *(uint32_t*)&sum_l2);
        }
        printf("C debug: step=%d denominator=%f sum_w=%f sum_l=%f sum_l2=%f sum_xl=%f bits=%08x\n", is, D, sum_w, sum_l, sum_l2, sum_xl,
               *(uint32_t*)&D);
        if (D > 0) {
                float numerator_scale = sum_w * sum_xl - sum_x * sum_l;
                float numerator_min = sum_l2 * sum_x - sum_l * sum_xl;
                float this_scale = numerator_scale / D;
                float this_min   = numerator_min / D;
                if (this_min > 0) {
                    this_min = 0;
                    this_scale = sum_xl / sum_l2;
                }
                float mad = 0;
                for (int i = 0; i < n; ++i) {
                    float diff = this_scale * Laux[i] + this_min - x[i];
                    diff = use_mad ? fabsf(diff) : diff * diff;
                    float w = weights[i];
                    mad += w * diff;
                }
            printf("C debug: step=%d scaleCand=%f minCand=%f metric=%f best=%f numeratorScale=%f numeratorMin=%f\n", is, this_scale, this_min, mad, best_mad, numerator_scale, numerator_min);
            if (mad < best_mad) {
                    for (int i = 0; i < n; ++i) {
                        L[i] = Laux[i];
                    }
                    best_mad = mad;
                    scale = this_scale;
                    min = this_min;
                }
            }
        }
    }
    printf("C debug: final scale=%f min=%f\n", scale, -min);
}

int main(void) {
    float values[QK_K];
    for (int i = 0; i < QK_K; ++i) {
        values[i] = 0.1f + 2.0f * cosf((float)i * 0.03f);
    }

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    FILE *txt = fopen("build/q2k-diagnostics/q2k-c.txt", "w");
    FILE *vals = fopen("build/q2k-diagnostics/q2k-values.txt", "w");
    if (!txt) {
        perror("fopen txt");
        return 1;
    }

    for (int i = 0; i < QK_K; ++i) {
        fprintf(txt, "% .8f ", values[i]);
        if ((i % 8) == 7) fprintf(txt, "\n");
        fprintf(vals, "% .8f\n", values[i]);
    }
    fclose(vals);
    fprintf(txt, "\n");

    for (int j = 0; j < QK_K/16; ++j) {
        for (int l = 0; l < 16; ++l) {
            weights[l] = fabsf(values[16*j + l]);
        }
        scales[j] = make_qkx2_quants(16, 3, values + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
        fprintf(txt, "subBlock=%d scale=%f min=%f quants=", j, scales[j], mins[j]);
        for (int q = 0; q < 4; ++q) {
            fprintf(txt, "%02x ", L[16*j + q]);
        }
        fprintf(txt, "\n");
    }

    fprintf(txt, "d=%04hx dmin=%04hx\n", 0, 0);
    fclose(txt);

    block_q2_K block;
    quantize_row_q2_K_ref(values, &block, QK_K);

    printf("C block subBlock0 codes=");
    for (int i = 0; i < 16; ++i) {
        int byte = block.qs[i/4];
        int code = (byte >> (2*(i%4))) & 0x3;
        printf("%d ", code);
    }
    printf("\n");

    float dBase = GGML_FP16_TO_FP32(block.d);
    float dmBase = GGML_FP16_TO_FP32(block.dmin);
    for (int j = 0; j < QK_K/16; ++j) {
        int scaleByte = block.scales[j];
        int scaleLow = scaleByte & 0x0F;
        int scaleHigh = (scaleByte >> 4) & 0x0F;
        float dScale = dBase * scaleLow;
        float dm = dmBase * scaleHigh;
        if (j == 0) {
            for (int lane = 0; lane < 16; ++lane) {
                float value = values[16*j + lane];
                int code = (block.qs[(16*j + lane)/4] >> (2*((16*j + lane)%4))) & 0x3;
                float ratio = (value + dm)/dScale;
                printf("C ratio subBlock=%d lane=%d value=%f ratio=%f code=%d\n", j, lane, value, ratio, code);
            }
        }
    }

    FILE *bin = fopen("build/q2k-diagnostics/q2k-c.bin", "wb");
    if (!bin) {
        perror("fopen bin");
        return 1;
    }
    fwrite(&block, sizeof(block), 1, bin);
    fclose(bin);

    debug_make_qkx2_quants_subblock(values + 16*0);
    debug_make_qkx2_quants_subblock(values + 16*1);
    debug_make_qkx2_quants_subblock(values + 16*11);
    debug_make_qkx2_quants_subblock(values + 16*12);
    debug_make_qkx2_quants_subblock(values + 16*14);

    return 0;
}
