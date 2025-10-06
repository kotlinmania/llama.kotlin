import Foundation
import Metal

private struct Params {
    var blockCount: UInt32
    var iterations: UInt32
}

final class MetalDequantizer {
    enum Error: Swift.Error {
        case noDevice
        case pipelineCreationFailed
        case commandEncodingFailed
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw Error.noDevice
        }
        guard let q = dev.makeCommandQueue() else {
            throw Error.pipelineCreationFailed
        }
        let pipeline = try MetalDequantizer.makePipeline(device: dev)
        self.device = dev
        self.queue = q
        self.pipeline = pipeline
    }

    private static func makePipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let metalSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct Params {
            uint blockCount;
            uint iterations;
        };

        inline void decode_scale(uint16_t bits, thread int &mantissa, thread int &exponent) {
            int sign = (bits >> 15) & 0x1;
            int exp = (bits >> 10) & 0x1F;
            int mant = bits & 0x3FF;
            if (exp == 0) {
                if (mant == 0) {
                    mantissa = 0;
                    exponent = 0;
                    return;
                }
                exp = 1;
                int shiftGuard = 0;
                while (((mant & 0x400) == 0) && shiftGuard < 10) {
                    mant <<= 1;
                    exp -= 1;
                    shiftGuard += 1;
                }
                mant &= 0x3FF;
            }
            mant |= 0x400;
            if (sign != 0) {
                mant = -mant;
            }
            mantissa = mant;
            exponent = exp - 25;
        }

        kernel void swar_q8_dequant(const device ushort *scales [[buffer(0)]],
                                    const device char *weights [[buffer(1)]],
                                    device int *values [[buffer(2)]],
                                    device int *exponents [[buffer(3)]],
                                    constant Params &p [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
            if (gid >= p.blockCount) {
                return;
            }
            ushort bits = scales[gid];
            int mantissa = 0;
            int exponent = 0;
            decode_scale(bits, mantissa, exponent);
            int base = int(gid) * 32;
            const device char *blockWeights = weights + base;
            device int *outValues = values + base;
            for (uint iter = 0; iter < p.iterations; ++iter) {
                for (uint lane = 0; lane < 32; ++lane) {
                    int w = int(blockWeights[lane]);
                    outValues[lane] = w * mantissa;
                }
            }
            exponents[gid] = exponent;
        }
        """
        let options = MTLCompileOptions()
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(source: metalSource, options: options)
        guard let fn = library.makeFunction(name: "swar_q8_dequant") else {
            throw Error.pipelineCreationFailed
        }
        return try device.makeComputePipelineState(function: fn)
    }

    func run(blocks: [Q8Block32], iterations: Int) throws -> (values: [Int32], exponents: [Int32], elapsedNanoseconds: UInt64) {
        let blockCount = blocks.count
        guard blockCount > 0 else {
            return ([], [], 0)
        }
        var scaleBits = [UInt16](repeating: 0, count: blockCount)
        var weights = [Int8](repeating: 0, count: blockCount * 32)
        for (index, block) in blocks.enumerated() {
            scaleBits[index] = block.scale.bitPattern
            let base = index * 32
            for lane in 0..<16 {
                weights[base + lane] = block.lo[lane]
                weights[base + 16 + lane] = block.hi[lane]
            }
        }

        guard let scaleBuffer = device.makeBuffer(bytes: scaleBits,
                                                   length: scaleBits.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared),
              let weightBuffer = device.makeBuffer(bytes: weights,
                                                  length: weights.count * MemoryLayout<Int8>.size,
                                                  options: .storageModeShared),
              let valuesBuffer = device.makeBuffer(length: blockCount * 32 * MemoryLayout<Int32>.size,
                                                   options: .storageModeShared),
              let exponentsBuffer = device.makeBuffer(length: blockCount * MemoryLayout<Int32>.size,
                                                      options: .storageModeShared) else {
            throw Error.pipelineCreationFailed
        }

        var params = Params(blockCount: UInt32(blockCount), iterations: UInt32(max(1, iterations)))
        guard let paramsBuffer = device.makeBuffer(bytes: &params,
                                                   length: MemoryLayout<Params>.stride,
                                                   options: .storageModeShared) else {
            throw Error.pipelineCreationFailed
        }

        let threadsPerGroup = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let tgSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let threadgroups = (blockCount + threadsPerGroup - 1) / threadsPerGroup
        let gridSize = MTLSize(width: threadgroups * threadsPerGroup, height: 1, depth: 1)

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Error.commandEncodingFailed
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(valuesBuffer, offset: 0, index: 2)
        encoder.setBuffer(exponentsBuffer, offset: 0, index: 3)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 4)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()

        let start = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let elapsed = DispatchTime.now().uptimeNanoseconds - start

        let valuesPointer = valuesBuffer.contents().bindMemory(to: Int32.self, capacity: blockCount * 32)
        let exponentsPointer = exponentsBuffer.contents().bindMemory(to: Int32.self, capacity: blockCount)

        let values = Array(UnsafeBufferPointer(start: valuesPointer, count: blockCount * 32))
        let exponents = Array(UnsafeBufferPointer(start: exponentsPointer, count: blockCount))

        return (values, exponents, elapsed)
    }
}
