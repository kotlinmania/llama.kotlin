// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stdint.h>

#if defined(__aarch64__)
enum {
    KC_REG_X19 = 0,
    KC_REG_X20,
    KC_REG_X21,
    KC_REG_X22,
    KC_REG_X23,
    KC_REG_X24,
    KC_REG_X25,
    KC_REG_X26,
    KC_REG_X27,
    KC_REG_X28,
    KC_REG_MAX
};
#elif defined(__x86_64__)
enum {
    KC_REG_R12 = 0,
    KC_REG_R13,
    KC_REG_R14,
    KC_REG_R15,
    KC_REG_RBX,
    KC_REG_RBP,
    KC_REG_MAX
};
#else
enum { KC_REG_MAX = 1 };
#endif

typedef enum {
    KC_OP_LOAD_IMM = 0,
    KC_OP_END      = 1,
} kc_opcode;

typedef struct {
    uint8_t  op;
    uint8_t  dst;
    uint16_t flags;
    uint64_t imm;
} kc_token;

typedef struct {
    uint64_t gpr[KC_REG_MAX];
} kc_vm_state;

int kc_token_vm_run_demo(void);
void kc_vm_execute(const kc_token *tokens, kc_vm_state *state);
