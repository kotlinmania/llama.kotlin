import Foundation
import Metal

struct Args {
    var sizes: [Int] = [8, 64, 4096, 262_144]
    // Smaller iteration counts to avoid huge command-buffer overhead on GPU
    var itersSmall = 5_000
    var itersMid = 1_000
    var itersBig = 100
}

func itersFor(_ n: Int, _ a: Args) -> Int {
    if n <= 64 { return a.itersSmall }
    if n <= 4096 { return a.itersMid }
    return a.itersBig
}

let metalSource = """
#include <metal_stdlib>
using namespace metal;

// Helpers: arithmetic-only decomposition and packing
inline uint udiv(uint x, uint d) { return x / d; }
inline uint umod(uint x, uint d) { return x - d * (x / d); }

inline void decompose_u8(uint x, thread uint &b0, thread uint &b1, thread uint &b2, thread uint &b3) {
    uint q0 = udiv(x, 256u);
    b0 = x - q0 * 256u;
    uint q1 = udiv(q0, 256u);
    b1 = q0 - q1 * 256u;
    uint q2 = udiv(q1, 256u);
    b2 = q1 - q2 * 256u;
    b3 = q2;
}

inline uint pack_u8(uint r0, uint r1, uint r2, uint r3) {
    return r0 + 256u * r1 + 65536u * r2 + 16777216u * r3;
}

inline void decompose_u16(uint x, thread uint &w0, thread uint &w1) {
    uint q = udiv(x, 65536u);
    w0 = x - q * 65536u;
    w1 = q;
}

inline uint pack_u16(uint r0, uint r1) {
    return r0 + 65536u * r1;
}

kernel void avg_u8_trunc_packs(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint axb = a ^ b;
    uint hfx = (axb & 0xFEFEFEFEu) >> 1; // logical in Metal
    uint avgv = (a & b) + hfx;
    outC[gid] = avgv;
}

kernel void avg_u8_round_packs(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint axb = a ^ b;
    uint hfx = (axb & 0xFEFEFEFEu) >> 1; // logical in Metal
    uint avgv = (a & b) + hfx;
    uint rnd = axb & 0x01010101u; // add 1 when per-lane sum is odd
    outC[gid] = avgv + rnd;
}

// Arithmetic-only kernels (no &, ^, >>, <<)
kernel void avg_u8_trunc_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1, a2, a3; decompose_u8(a, a0, a1, a2, a3);
    uint b0, b1, b2, b3; decompose_u8(b, b0, b1, b2, b3);
    uint r0 = udiv(a0 + b0, 2u);
    uint r1 = udiv(a1 + b1, 2u);
    uint r2 = udiv(a2 + b2, 2u);
    uint r3 = udiv(a3 + b3, 2u);
    outC[gid] = pack_u8(r0, r1, r2, r3);
}

kernel void avg_u8_round_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1, a2, a3; decompose_u8(a, a0, a1, a2, a3);
    uint b0, b1, b2, b3; decompose_u8(b, b0, b1, b2, b3);
    uint r0 = udiv(a0 + b0 + 1u, 2u);
    uint r1 = udiv(a1 + b1 + 1u, 2u);
    uint r2 = udiv(a2 + b2 + 1u, 2u);
    uint r3 = udiv(a3 + b3 + 1u, 2u);
    outC[gid] = pack_u8(r0, r1, r2, r3);
}

kernel void avg_u16_trunc_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1; decompose_u16(a, a0, a1);
    uint b0, b1; decompose_u16(b, b0, b1);
    uint r0 = udiv(a0 + b0, 2u);
    uint r1 = udiv(a1 + b1, 2u);
    outC[gid] = pack_u16(r0, r1);
}

kernel void avg_u16_round_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1; decompose_u16(a, a0, a1);
    uint b0, b1; decompose_u16(b, b0, b1);
    uint r0 = udiv(a0 + b0 + 1u, 2u);
    uint r1 = udiv(a1 + b1 + 1u, 2u);
    outC[gid] = pack_u16(r0, r1);
}
"""

func makeDevice() -> MTLDevice {
    guard let dev = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal device available")
    }
    return dev
}

func makePipeline(_ device: MTLDevice, _ name: String) -> MTLComputePipelineState {
    let options = MTLCompileOptions()
    options.languageVersion = .version3_0
    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: metalSource, options: options)
    } catch {
        fatalError("Library compile failed: \(error)")
    }
    guard let fn = library.makeFunction(name: name) else {
        fatalError("Function not found: \(name)")
    }
    do {
        return try device.makeComputePipelineState(function: fn)
    } catch {
        fatalError("Pipeline creation failed: \(error)")
    }
}

@inline(__always)
func checksum(_ arr: UnsafeBufferPointer<UInt32>) -> UInt64 {
    var s: UInt64 = 0
    for v in arr { s = (s &* 131) &+ UInt64(v) }
    return s & 0x7FFF_FFFF
}

func runKernel(device: MTLDevice, queue: MTLCommandQueue, pso: MTLComputePipelineState,
               a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, n: Int, iters: Int) -> (ns: UInt64, cs: UInt64) {
    let tgSize = min(256, pso.maxTotalThreadsPerThreadgroup)
    let grid = MTLSize(width: n, height: 1, depth: 1)
    let tgs = MTLSize(width: tgSize, height: 1, depth: 1)

    let t0 = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iters {
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { fatalError("enc fail") }
        enc.setComputePipelineState(pso)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(c, offset: 0, index: 2)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgs)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let t1 = DispatchTime.now().uptimeNanoseconds

    let outPtr = c.contents().bindMemory(to: UInt32.self, capacity: n)
    let cs = checksum(UnsafeBufferPointer(start: outPtr, count: n))
    return (ns: t1 &- t0, cs: cs)
}

func main() {
    // Optional: dump disassembly of the kernels (AIR -> ASM)
    func run(_ args: [String]) throws -> (Int32, String, String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        p.arguments = args
        let outPipe = Pipe(), errPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = errPipe
        try p.run()
        p.waitUntilExit()
        let out = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let err = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return (p.terminationStatus, out, err)
    }

    func writeText(_ path: String, _ text: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.data(using: .utf8)?.write(to: url)
    }

    do {
        let disDir = FileManager.default.currentDirectoryPath + "/metal-disasm"
        let srcPath = disDir + "/kernels.metal"
        let airPath = disDir + "/kernels.air"
        let asmPath = disDir + "/kernels.asm"
        try writeText(srcPath, metalSource)
        // Compile to AIR
        var (st1, out1, err1) = try run(["--sdk", "macosx", "metal", "-std=metal3.0", srcPath, "-o", airPath])
        if st1 != 0 {
            // Fallback language version
            (st1, out1, err1) = try run(["--sdk", "macosx", "metal", "-std=metal2.4", srcPath, "-o", airPath])
        }
        if st1 == 0 {
            let (st2, out2, err2) = try run(["--sdk", "macosx", "metal-disassemble", airPath])
            if st2 == 0 {
                try writeText(asmPath, out2)
                print("[metal-disasm] Wrote: \(asmPath)")
            } else {
                fputs("[metal-disasm] metal-disassemble failed: \(err2)\n", stderr)
            }
        } else {
            fputs("[metal-disasm] metal compile failed: \(err1)\n", stderr)
        }
    } catch {
        fputs("[metal-disasm] error: \(error)\n", stderr)
    }

    let device = makeDevice()
    guard let queue = device.makeCommandQueue() else { fatalError("No queue") }
    let psoTrunc = makePipeline(device, "avg_u8_trunc_packs")
    let psoRound = makePipeline(device, "avg_u8_round_packs")

    let args = Args()
    print("variant,size,iters,ns,GBps,GBps_eff,checksum")
    for n in args.sizes {
        let iters = itersFor(n, args)
        let bytesPerPackEff = 12.0 // 2 inputs (4B each) + 1 output (4B)
        let bytesPerPackLog = 8.0  // logical (lanes)
        let bufBytes = n * 4
        guard let bufA = device.makeBuffer(length: bufBytes, options: [.storageModeShared]),
              let bufB = device.makeBuffer(length: bufBytes, options: [.storageModeShared]),
              let bufC = device.makeBuffer(length: bufBytes, options: [.storageModeShared]) else {
            fatalError("Buffers alloc failed")
        }
        // Seed inputs deterministically
        let ap = bufA.contents().bindMemory(to: UInt32.self, capacity: n)
        let bp = bufB.contents().bindMemory(to: UInt32.self, capacity: n)
        for i in 0..<n {
            let b0a = UInt32((i &* 17 &+ 5) & 0xFF)
            let b1a = UInt32((i &* 29 &+ 7) & 0xFF)
            let b2a = UInt32((i &* 43 &+ 11) & 0xFF)
            let b3a = UInt32((i &* 61 &+ 13) & 0xFF)
            ap[i] = b0a | (b1a << 8) | (b2a << 16) | (b3a << 24)

            let b0b = UInt32((i &* 31 &+ 3) & 0xFF)
            let b1b = UInt32((i &* 37 &+ 9) & 0xFF)
            let b2b = UInt32((i &* 19 &+ 15) & 0xFF)
            let b3b = UInt32((i &* 23 &+ 21) & 0xFF)
            bp[i] = b0b | (b1b << 8) | (b2b << 16) | (b3b << 24)
        }

        func emit(_ label: String, _ ns: UInt64, _ cs: UInt64) {
            let gbe = (Double(n) * bytesPerPackEff * Double(iters) / 1e9) / (Double(ns) / 1e9)
            let gbl = (Double(n) * bytesPerPackLog * Double(iters) / 1e9) / (Double(ns) / 1e9)
            print("\(label),\(n),\(iters),\(ns),\(String(format: "%.3f", gbl)),\(String(format: "%.3f", gbe)),\(cs)")
        }

        // Bitwise SWAR (reference)
        do {
            let (ns, cs) = runKernel(device: device, queue: queue, pso: psoTrunc, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u8-trunc", ns, cs)
        }
        // Round
        do {
            let (ns, cs) = runKernel(device: device, queue: queue, pso: psoRound, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u8-round", ns, cs)
        }

        // Strict arithmetic kernels
        do {
            let p = makePipeline(device, "avg_u8_trunc_arith")
            let (ns, cs) = runKernel(device: device, queue: queue, pso: p, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u8-trunc-arith", ns, cs)
        }
        do {
            let p = makePipeline(device, "avg_u8_round_arith")
            let (ns, cs) = runKernel(device: device, queue: queue, pso: p, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u8-round-arith", ns, cs)
        }
        do {
            let p = makePipeline(device, "avg_u16_trunc_arith")
            let (ns, cs) = runKernel(device: device, queue: queue, pso: p, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u16-trunc-arith", ns, cs)
        }
        do {
            let p = makePipeline(device, "avg_u16_round_arith")
            let (ns, cs) = runKernel(device: device, queue: queue, pso: p, a: bufA, b: bufB, c: bufC, n: n, iters: iters)
            emit("metal-u16-round-arith", ns, cs)
        }
    }
}

main()
