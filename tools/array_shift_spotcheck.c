#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void print_arr(const char *label, const uint16_t *a, int len) {
    printf("%s:", label);
    for (int i = 0; i < len; ++i) printf(" %04x", a[i] & 0xFFFF);
    printf("\n");
}

// Right shift (logical) for little-endian 16-bit limbs. Returns carryOut (low s bits dropped from limb 0) and sticky.
static uint32_t rsh16_le(uint16_t *a, int from, int len, int s, uint32_t *sticky) {
    *sticky = 0;
    if (s == 0 || len <= 0) return 0;
    uint32_t nextCarry = 0;
    uint32_t carryOut = 0;
    for (int i = from + len - 1; i >= from; --i) {
        uint32_t cur = a[i] & 0xFFFFu;
        uint32_t out = (cur >> s) | (nextCarry << (16 - s));
        uint32_t dropped = cur & ((1u << s) - 1u);
        if (i == from) carryOut = dropped;
        if (dropped) *sticky = 1;
        a[i] = (uint16_t)(out & 0xFFFFu);
        nextCarry = dropped;
    }
    return carryOut & 0xFFFFu;
}

// Left shift (logical) for little-endian 16-bit limbs. Returns carryOut (upper s bits from last limb).
static uint32_t shl16_le(uint16_t *a, int from, int len, int s, uint32_t carryIn) {
    if (s == 0 || len <= 0) return carryIn & 0xFFFFu;
    for (int i = from; i < from + len; ++i) {
        uint32_t cur = a[i] & 0xFFFFu;
        uint32_t out = ((cur << s) & 0xFFFFu) | (carryIn & ((1u << s) - 1u));
        uint32_t carry = (cur >> (16 - s)) & ((1u << s) - 1u);
        a[i] = (uint16_t)(out & 0xFFFFu);
        carryIn = carry;
    }
    return carryIn & 0xFFFFu;
}

int main(void) {
    uint16_t arr[4] = {0x0003, 0x8000, 0x0000, 0x0000};
    print_arr("before", arr, 2);
    uint32_t sticky = 0;
    uint32_t c = rsh16_le(arr, 0, 2, 1, &sticky);
    print_arr("after_r1", arr, 2);
    printf("carryOut=%u sticky=%u\n", c, sticky);

    uint16_t arr2[3] = {0x0001, 0x0000, 0x0000};
    print_arr("before", arr2, 3);
    uint32_t c2 = shl16_le(arr2, 0, 3, 1, 0);
    print_arr("after_l1", arr2, 3);
    printf("carryOut=%u\n", c2);
    return 0;
}

