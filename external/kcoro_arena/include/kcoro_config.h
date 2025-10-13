// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/*
 * ============================================================================
 * kcoro_arena: Golden Path Configuration
 * ============================================================================
 * 
 * This file has been intentionally left minimal as part of the security-
 * focused "golden path" architecture. All tuning parameters have been
 * replaced with optimal hardcoded constants in the implementation.
 * 
 * Rationale:
 *   - Zero user-configurable macros = zero misconfiguration attack surface
 *   - One tested code path = predictable, auditable behavior
 *   - Fixed optimal defaults = no performance cliffs from bad tuning
 * 
 * All constants are now hardcoded in their respective implementation files:
 *   - Cancellation timing: kc_cancel.c (10ms optimal)
 *   - Token bucket sizing: kc_token_kernel.c (1024 buckets optimal)
 *   - Channel capacities: kc_chan.c (256 initial unlimited optimal)
 * 
 * If you need platform-specific adjustments, modify the implementation
 * files directly and document the rationale.
 */
