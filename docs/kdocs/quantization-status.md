# Quantization Status (Q2_K focus)

Updated: 2025‑09‑21

## Current
- Destination-based quantization path in place; Q2_K block writer mirrors ggml layout (scales/mins then codes).
- Diagnostics harnesses added:
  - Kotlin recorder prints header, per-subblock scales/mins, and packed bytes
  - C tool (`tools/q2k_dump.c`) emits reference block and step traces
- Bitwise determinism addressed via KLang bit ops and helpers; mismatches point to scale/min refinement math.

## Findings
- Drift arises when the refinement denominator (`sumW*sumL2 - sumL*sumL`) collapses to 0 in Kotlin while C retains a small residual.
- KLang introduces precise accumulation and will adopt fused subtract structure for denominators/numerators; Float32 multiplication/sqrt exactness will remove the remaining bias.

## Plan
1) Land Float32 `mulBits`/`sqrtBits` exact ports (compiler‑rt) and use them in the refinement loop
2) Re-run snapshot test; compare codes per lane and header halves
3) Remove temporary diagnostics once parity is green; keep regression fixture

## Related Docs
- docs/kdocs/klangnative-soft-float-and-hpc16.md
- docs/kdocs/kotlin-native-simd-plan.md (SIMD after parity)
- docs/kdocs/test-triage-2025-09-21.md
