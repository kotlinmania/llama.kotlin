// SPDX-License-Identifier: BSD-3-Clause
// token_vm_resume.c -- prototype interpreter-driven register restore
// This lab tool demonstrates how a tiny token stream can stage coroutine
// register state without emitting executable code.

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Opcodes -----------------------------------------------------------------
typedef enum {
    KC_OP_LOAD_IMM = 0,
    KC_OP_END      = 1,
} kc_opcode;

typedef struct {
    uint8_t   op;
    uint8_t   dst;   // logical register index
    uint16_t  flags;
    uint64_t  imm;   // immediate / pointer
} kc_token;

// Logical register mapping (subset of callee saved registers)
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

typedef struct {
    uint64_t gpr[KC_REG_MAX];
} kc_vm_state;

// Helper that prints the "live" machine registers.
static void dump_machine_registers(void)
{
#if defined(__aarch64__)
    uint64_t vals[KC_REG_MAX];
    __asm__ volatile(
        "mov %0, x19\n\t"
        "mov %1, x20\n\t"
        "mov %2, x21\n\t"
        "mov %3, x22\n\t"
        "mov %4, x23\n\t"
        "mov %5, x24\n\t"
        "mov %6, x25\n\t"
        "mov %7, x26\n\t"
        "mov %8, x27\n\t"
        "mov %9, x28\n\t"
        : "=r"(vals[0]), "=r"(vals[1]), "=r"(vals[2]), "=r"(vals[3]),
          "=r"(vals[4]), "=r"(vals[5]), "=r"(vals[6]), "=r"(vals[7]),
          "=r"(vals[8]), "=r"(vals[9]));
    printf("[cpu] x19=%#llx x20=%#llx x21=%#llx x22=%#llx\n",
           (unsigned long long)vals[0], (unsigned long long)vals[1],
           (unsigned long long)vals[2], (unsigned long long)vals[3]);
    printf("      x23=%#llx x24=%#llx x25=%#llx x26=%#llx\n",
           (unsigned long long)vals[4], (unsigned long long)vals[5],
           (unsigned long long)vals[6], (unsigned long long)vals[7]);
    printf("      x27=%#llx x28=%#llx\n",
           (unsigned long long)vals[8], (unsigned long long)vals[9]);
#elif defined(__x86_64__)
    uint64_t vals[KC_REG_MAX];
    __asm__ volatile(
        "mov %%r12, %0\n\t"
        "mov %%r13, %1\n\t"
        "mov %%r14, %2\n\t"
        "mov %%r15, %3\n\t"
        "mov %%rbx, %4\n\t"
        "mov %%rbp, %5\n\t"
        : "=r"(vals[0]), "=r"(vals[1]), "=r"(vals[2]), "=r"(vals[3]),
          "=r"(vals[4]), "=r"(vals[5]));
    printf("[cpu] r12=%#llx r13=%#llx r14=%#llx r15=%#llx\n",
           (unsigned long long)vals[0], (unsigned long long)vals[1],
           (unsigned long long)vals[2], (unsigned long long)vals[3]);
    printf("      rbx=%#llx rbp=%#llx\n",
           (unsigned long long)vals[4], (unsigned long long)vals[5]);
#else
    puts("[cpu] register dump not available on this arch");
#endif
}

// Interpreter ----------------------------------------------------------------
static void kc_vm_execute(const kc_token *tokens, kc_vm_state *state)
{
    const kc_token *pc = tokens;
    for (;;) {
        switch (pc->op) {
        case KC_OP_LOAD_IMM:
            if (pc->dst < KC_REG_MAX) {
                state->gpr[pc->dst] = pc->imm;
            }
            break;
        case KC_OP_END:
#if defined(__aarch64__)
        {
            register uint64_t x19 __asm__("x19") = state->gpr[KC_REG_X19];
            register uint64_t x20 __asm__("x20") = state->gpr[KC_REG_X20];
            register uint64_t x21 __asm__("x21") = state->gpr[KC_REG_X21];
            register uint64_t x22 __asm__("x22") = state->gpr[KC_REG_X22];
            register uint64_t x23 __asm__("x23") = state->gpr[KC_REG_X23];
            register uint64_t x24 __asm__("x24") = state->gpr[KC_REG_X24];
            register uint64_t x25 __asm__("x25") = state->gpr[KC_REG_X25];
            register uint64_t x26 __asm__("x26") = state->gpr[KC_REG_X26];
            register uint64_t x27 __asm__("x27") = state->gpr[KC_REG_X27];
            register uint64_t x28 __asm__("x28") = state->gpr[KC_REG_X28];
            (void)x19; (void)x20; (void)x21; (void)x22; (void)x23;
            (void)x24; (void)x25; (void)x26; (void)x27; (void)x28;
            void (*fn)(void) = (void (*)(void))pc->imm;
            __asm__ volatile("" :: "r"(x19), "r"(x20), "r"(x21), "r"(x22),
                                   "r"(x23), "r"(x24), "r"(x25), "r"(x26),
                                   "r"(x27), "r"(x28));
            fn();
        }
#elif defined(__x86_64__)
        {
            register uint64_t r12 __asm__("r12") = state->gpr[KC_REG_R12];
            register uint64_t r13 __asm__("r13") = state->gpr[KC_REG_R13];
            register uint64_t r14 __asm__("r14") = state->gpr[KC_REG_R14];
            register uint64_t r15 __asm__("r15") = state->gpr[KC_REG_R15];
            register uint64_t rbx __asm__("rbx") = state->gpr[KC_REG_RBX];
            register uint64_t rbp __asm__("rbp") = state->gpr[KC_REG_RBP];
            (void)r12; (void)r13; (void)r14; (void)r15; (void)rbx; (void)rbp;
            void (*fn)(void) = (void (*)(void))pc->imm;
            __asm__ volatile("" :: "r"(r12), "r"(r13), "r"(r14), "r"(r15),
                                   "r"(rbx), "r"(rbp));
            fn();
        }
#else
            ((void (*)(void))pc->imm)();
#endif
            return;
        default:
            fprintf(stderr, "unknown opcode %u\n", pc->op);
            return;
        }
        pc++;
    }
}

// Demo target that prints the logical state and a live register dump.
static void demo_target(void)
{
    printf("[target] executing with live register state:\n");
    dump_machine_registers();
}

int main(void)
{
    kc_vm_state state;
    memset(&state, 0, sizeof(state));

    const kc_token program[] = {
#if defined(__aarch64__)
        { KC_OP_LOAD_IMM, KC_REG_X19, 0, 0x1111 },
        { KC_OP_LOAD_IMM, KC_REG_X20, 0, 0x2222 },
        { KC_OP_LOAD_IMM, KC_REG_X21, 0, 0x3333 },
        { KC_OP_LOAD_IMM, KC_REG_X22, 0, 0x4444 },
        { KC_OP_END,      0,            0, (uint64_t)(void*)demo_target },
#elif defined(__x86_64__)
        { KC_OP_LOAD_IMM, KC_REG_R12, 0, 0xaaaa },
        { KC_OP_LOAD_IMM, KC_REG_R13, 0, 0xbbbb },
        { KC_OP_LOAD_IMM, KC_REG_R14, 0, 0xcccc },
        { KC_OP_LOAD_IMM, KC_REG_R15, 0, 0xdddd },
        { KC_OP_END,      0,           0, (uint64_t)(void*)demo_target },
#else
        { KC_OP_END,      0,           0, (uint64_t)(void*)demo_target },
#endif
    };

    printf("[vm] staging coroutine context via token stream...\n");
    kc_vm_execute(program, &state);

    return 0;
}
