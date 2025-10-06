import Foundation
import Combine
import simd

struct Q8Block32: Sendable {
    var scale: Float16
    var lo: SIMD16<Int8>
    var hi: SIMD16<Int8>

    init(scale: Float16, lo: SIMD16<Int8>, hi: SIMD16<Int8>) {
        self.scale = scale
        self.lo = lo
        self.hi = hi
    }

    init(randomGenerator rng: inout some RandomNumberGenerator) {
        self.scale = Float16.random(in: 0.01...1.0, using: &rng)
        var loVals = SIMD16<Int8>()
        var hiVals = SIMD16<Int8>()
        for idx in 0..<16 {
            loVals[idx] = Int8.random(in: -127...127, using: &rng)
            hiVals[idx] = Int8.random(in: -127...127, using: &rng)
        }
        self.lo = loVals
        self.hi = hiVals
    }
}

let q8LogicalBytesPerBlock: Double = 34.0

@inline(__always)
func decodeScaleBits(_ scale: Float16) -> (mantissa: Int32, exponent: Int32) {
    let bits = scale.bitPattern
    let sign = Int32((bits & 0x8000) >> 15)
    var exp = Int32((bits & 0x7C00) >> 10)
    var mant = Int32(bits & 0x03FF)
    if exp == 0 {
        if mant == 0 { return (0, 0) }
        exp = 1
        while mant & 0x0400 == 0 {
            mant <<= 1
            exp -= 1
        }
        mant &= 0x03FF
    }
    mant |= 0x0400
    if sign != 0 { mant = -mant }
    let exponent = exp - 25
    return (mantissa: mant, exponent: exponent)
}

@inline(__always)
func dequantize(block: Q8Block32, scaleMantissa: Int32, into destination: UnsafeMutablePointer<Int32>) {
    var dst = destination
    for lane in 0..<16 {
        dst.pointee = Int32(block.lo[lane]) * scaleMantissa
        dst += 1
    }
    for lane in 0..<16 {
        dst.pointee = Int32(block.hi[lane]) * scaleMantissa
        dst += 1
    }
}

struct FixedPointTile: Sendable {
    let values: [Int32]
    let exponents: [Int32]
}

enum BlockSource: Sendable {
    struct RandomBlocks: Sendable {
        let blocks: [Q8Block32]
    }

    struct GGUFBlocks: Sendable {
        let data: Data
        let baseOffset: Int
        let bytesPerBlock: Int
        let blockCount: Int

        @inline(__always)
        func withAccess<R>(_ body: (UnsafeRawPointer, Int) -> R) -> R {
            data.withUnsafeBytes { buffer in
                guard let rawBase = buffer.baseAddress else {
                    fatalError("GGUF data has no base address")
                }
                return body(UnsafeRawPointer(rawBase).advanced(by: baseOffset), bytesPerBlock)
            }
        }

        func packBlocks() -> RandomBlocks {
            var blocks: [Q8Block32] = []
            blocks.reserveCapacity(blockCount)
            data.withUnsafeBytes { buffer in
                guard let base = buffer.baseAddress else { return }
                let start = base.advanced(by: baseOffset)
                for idx in 0..<blockCount {
                    let blockPtr = start.advanced(by: idx * bytesPerBlock)
                    let scaleBits = UInt16(littleEndian: blockPtr.load(as: UInt16.self))
                    let scale = Float16(bitPattern: scaleBits)
                    let weightsPtr = blockPtr.advanced(by: 2).assumingMemoryBound(to: Int8.self)

                    var lo = SIMD16<Int8>()
                    var hi = SIMD16<Int8>()
                    for lane in 0..<16 {
                        lo[lane] = weightsPtr[lane]
                        hi[lane] = weightsPtr[16 + lane]
                    }
                    blocks.append(Q8Block32(scale: scale, lo: lo, hi: hi))
                }
            }
            return RandomBlocks(blocks: blocks)
        }
    }

    case random(RandomBlocks)
    case gguf(GGUFBlocks)

    var count: Int {
        switch self {
        case .random(let holder):
            return holder.blocks.count
        case .gguf(let holder):
            return holder.blockCount
        }
    }

    func materialized() -> RandomBlocks {
        switch self {
        case .random(let holder):
            return holder
        case .gguf(let holder):
            return holder.packBlocks()
        }
    }
}

struct CombineWorkItem {
    let blockIndex: Int
    let block: Q8Block32
}

actor TileBroadcaster<Element> {
    private var continuations: [UUID: AsyncStream<Element>.Continuation] = [:]
    private var finished = false

    func subscribe(bufferLimit: Int = 0) -> (id: UUID, stream: AsyncStream<Element>) {
        precondition(!finished, "Cannot subscribe after finish")
        var continuation: AsyncStream<Element>.Continuation!
        let buffering: AsyncStream<Element>.Continuation.BufferingPolicy = bufferLimit > 0 ? .bufferingNewest(bufferLimit) : .unbounded
        let stream = AsyncStream<Element>(bufferingPolicy: buffering) { continuation = $0 }
        let id = UUID()
        continuations[id] = continuation
        return (id, stream)
    }

    func unsubscribe(_ id: UUID) {
        continuations[id]?.finish()
        continuations[id] = nil
    }

    func send(_ element: Element) {
        guard !finished else { return }
        for continuation in continuations.values {
            continuation.yield(element)
        }
    }

    func finish() {
        guard !finished else { return }
        finished = true
        for continuation in continuations.values {
            continuation.finish()
        }
        continuations.removeAll()
    }
}

struct BenchmarkOutcome: Sendable {
    let tile: FixedPointTile
    let checksum: Int64
    let elapsedSeconds: Double
    let logicalGBps: Double
    let blockCount: Int
    let iterations: Int
    let chunkCount: Int
    let engine: String
}

struct CLIOptions {
    var blocks: Int = 4_096
    var iterations: Int?
    var ggufPath: String?
    var tensorName: String?
    var workerOverride: Int?
    var chunkOverride: Int?
    var useMetal: Bool = false
    var useCombine: Bool = false
    var useActor: Bool = false

    static func parse() -> CLIOptions {
        var opts = CLIOptions()
        var args = CommandLine.arguments.dropFirst()
        if let dashIdx = args.firstIndex(of: "--") {
            args.remove(at: dashIdx)
        }
        while let arg = args.first {
            args.removeFirst()
            switch arg {
            case "--blocks":
                guard let value = args.first, let intValue = Int(value), intValue > 0 else {
                    usage("--blocks requires positive integer")
                }
                args.removeFirst()
                opts.blocks = intValue
            case "--iterations":
                guard let value = args.first, let intValue = Int(value), intValue > 0 else {
                    usage("--iterations requires positive integer")
                }
                args.removeFirst()
                opts.iterations = intValue
            case "--gguf":
                guard let value = args.first else {
                    usage("--gguf requires a file path")
                }
                args.removeFirst()
                opts.ggufPath = value
            case "--tensor":
                guard let value = args.first else {
                    usage("--tensor requires a name")
                }
                args.removeFirst()
                opts.tensorName = value
            case "--workers":
                guard let value = args.first, let intValue = Int(value), intValue > 0 else {
                    usage("--workers requires positive integer")
                }
                args.removeFirst()
                opts.workerOverride = intValue
            case "--chunks":
                guard let value = args.first, let intValue = Int(value), intValue > 0 else {
                    usage("--chunks requires positive integer")
                }
                args.removeFirst()
                opts.chunkOverride = intValue
            case "--metal":
                opts.useMetal = true
            case "--combine":
                opts.useCombine = true
            case "--actor":
                opts.useActor = true
            case "--help", "-h":
                usage(nil)
            default:
                usage("Unknown flag: \(arg)")
            }
        }
        if opts.tensorName != nil && opts.ggufPath == nil {
            usage("--tensor requires --gguf")
        }
        let selected = [opts.useMetal, opts.useCombine, opts.useActor].filter { $0 }
        if selected.count > 1 {
            usage("Use at most one of --metal, --combine, or --actor")
        }
        return opts
    }

    static func usage(_ message: String?) -> Never {
        if let message {
            fputs("\(message)\n\n", stderr)
        }
        let tool = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "swar-prototype"
        fputs("Usage: \(tool) [options]\n", stderr)
        fputs("\n", stderr)
        fputs("  --blocks N       Number of Q8_0 blocks to process (default 4096)\n", stderr)
        fputs("  --iterations N   Override iteration count (defaults scale with blocks)\n", stderr)
        fputs("  --workers N      Logical worker count (default: active processors)\n", stderr)
        fputs("  --chunks N       Override chunk fan-out (default: min(workers, 4))\n", stderr)
        fputs("  --gguf PATH      Load blocks from GGUF file instead of synthetic data\n", stderr)
        fputs("  --tensor NAME    Pick specific Q8_0 tensor (requires --gguf)\n", stderr)
        fputs("  --metal          Use Metal GPU kernel for dequant benchmark\n", stderr)
        fputs("  --combine        Use Combine broadcast pipeline demo\n", stderr)
        fputs("  --actor          Use actor-based broadcast pipeline demo\n", stderr)
        fputs("  --help           Show this help\n", stderr)
        exit(EXIT_FAILURE)
    }
}

enum PrototypeError: Error, CustomStringConvertible {
    case noBlocksLoaded(String)

    var description: String {
        switch self {
        case .noBlocksLoaded(let tensor):
            return "Tensor \(tensor) did not yield any blocks"
        }
    }
}

@inline(__always)
func defaultIterations(for blockCount: Int) -> Int {
    if blockCount <= 64 { return 200_000 }
    if blockCount <= 4_096 { return 50_000 }
    return 2_000
}

func generateRandomBlocks(count: Int) -> [Q8Block32] {
    var rng = SystemRandomNumberGenerator()
    var blocks: [Q8Block32] = []
    blocks.reserveCapacity(count)
    for _ in 0..<count {
        blocks.append(Q8Block32(randomGenerator: &rng))
    }
    return blocks
}

func loadBlocks(using options: CLIOptions) throws -> (BlockSource, String, String?) {
    if let ggufPath = options.ggufPath {
        let loader = try GGUFLoader(path: ggufPath)
        guard let tensorName = options.tensorName ?? loader.defaultQ8TensorName() else {
            throw GGUFLoader.Error.noQ8Tensor
        }
        let source = try loader.makeQ8BlockSource(tensorName: tensorName, maxBlocks: options.blocks)
        guard source.blockCount > 0 else {
            throw PrototypeError.noBlocksLoaded(tensorName)
        }
        let descriptor = "gguf:\(URL(fileURLWithPath: ggufPath).lastPathComponent)"
        return (.gguf(source), descriptor, tensorName)
    }
    let randomSource = BlockSource.RandomBlocks(blocks: generateRandomBlocks(count: options.blocks))
    return (.random(randomSource), "random", nil)
}

struct DequantPointers: @unchecked Sendable {
    let values: UnsafeMutablePointer<Int32>
    let exponents: UnsafeMutablePointer<Int32>
}

func checksum(values: UnsafeBufferPointer<Int32>, exponents: UnsafeBufferPointer<Int32>) -> Int64 {
    var sum: Int64 = 0
    for v in values {
        sum = (sum &* 131 &+ Int64(v)) & 0x7FFF_FFFF
    }
    for e in exponents {
        sum = (sum &* 131 &+ Int64(e)) & 0x7FFF_FFFF
    }
    return sum
}

func runBenchmarkCPU(resolvedBlocks: BlockSource.RandomBlocks, iterations: Int, chunkCount: Int) async -> BenchmarkOutcome {
    let blocks = resolvedBlocks.blocks
    let blockCount = blocks.count
    let valuesCapacity = blockCount * 32
    let valuesPtr = UnsafeMutablePointer<Int32>.allocate(capacity: valuesCapacity)
    let exponentsPtr = UnsafeMutablePointer<Int32>.allocate(capacity: blockCount)
    let pointers = DequantPointers(values: valuesPtr, exponents: exponentsPtr)
    defer {
        valuesPtr.deallocate()
        exponentsPtr.deallocate()
    }

    let actualChunks = max(1, min(chunkCount, blockCount))
    let chunkSize = (blockCount + actualChunks - 1) / actualChunks

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iterations {
        await withTaskGroup(of: Void.self) { group in
            var chunk = 0
            while chunk < actualChunks {
                let startBlock = chunk * chunkSize
                let endBlock = min(blockCount, startBlock + chunkSize)
                if startBlock < endBlock {
                    group.addTask {
                        var valuesCursor = pointers.values.advanced(by: startBlock * 32)
                        var expCursor = pointers.exponents.advanced(by: startBlock)
                        var idx = startBlock
                        while idx < endBlock {
                            let block = blocks[idx]
                            let decoded = decodeScaleBits(block.scale)
                            expCursor.pointee = decoded.exponent
                            dequantize(block: block, scaleMantissa: decoded.mantissa, into: valuesCursor)
                            valuesCursor = valuesCursor.advanced(by: 32)
                            expCursor = expCursor.advanced(by: 1)
                            idx += 1
                        }
                    }
                }
                chunk += 1
            }
            await group.waitForAll()
        }
    }
    let elapsedNs = DispatchTime.now().uptimeNanoseconds - start
    let elapsedSeconds = Double(elapsedNs) / 1_000_000_000.0

    let valuesBuffer = UnsafeBufferPointer(start: valuesPtr, count: valuesCapacity)
    let exponentsBuffer = UnsafeBufferPointer(start: exponentsPtr, count: blockCount)
    let checksumValue = checksum(values: valuesBuffer, exponents: exponentsBuffer)

    let valuesArray = Array(valuesBuffer)
    let exponentsArray = Array(exponentsBuffer)
    let tile = FixedPointTile(values: valuesArray, exponents: exponentsArray)

    let logicalGBps = (Double(blockCount * iterations) * q8LogicalBytesPerBlock) / (elapsedSeconds * 1e9)

    return BenchmarkOutcome(tile: tile,
                            checksum: checksumValue,
                            elapsedSeconds: elapsedSeconds,
                            logicalGBps: logicalGBps,
                            blockCount: blockCount,
                            iterations: iterations,
                            chunkCount: actualChunks,
                            engine: "cpu")
}

func runBenchmarkCombine(resolvedBlocks: BlockSource.RandomBlocks, iterations: Int, workerCount: Int) -> BenchmarkOutcome {
    let blocks = resolvedBlocks.blocks
    let blockCount = blocks.count
    guard blockCount > 0 else {
        return BenchmarkOutcome(tile: FixedPointTile(values: [], exponents: []),
                                checksum: 0,
                                elapsedSeconds: 0,
                                logicalGBps: 0,
                                blockCount: 0,
                                iterations: iterations,
                                chunkCount: 0,
                                engine: "combine")
    }

    let valuesCapacity = blockCount * 32
    let valuesPtr = UnsafeMutablePointer<Int32>.allocate(capacity: valuesCapacity)
    let exponentsPtr = UnsafeMutablePointer<Int32>.allocate(capacity: blockCount)
    defer {
        valuesPtr.deallocate()
        exponentsPtr.deallocate()
    }

    let subject = PassthroughSubject<CombineWorkItem, Never>()
    var cancellables = Set<AnyCancellable>()

    subject
        .sink(receiveCompletion: { _ in }) { work in
            let decoded = decodeScaleBits(work.block.scale)
            let base = work.blockIndex * 32
            var dst = valuesPtr.advanced(by: base)
            for lane in 0..<16 {
                dst.pointee = Int32(work.block.lo[lane]) * decoded.mantissa
                dst = dst.advanced(by: 1)
            }
            for lane in 0..<16 {
                dst.pointee = Int32(work.block.hi[lane]) * decoded.mantissa
                dst = dst.advanced(by: 1)
            }
            exponentsPtr[work.blockIndex] = decoded.exponent
        }
        .store(in: &cancellables)

    var exponentSum = Int64(0)
    subject
        .sink(receiveCompletion: { _ in }) { work in
            let decoded = decodeScaleBits(work.block.scale)
            exponentSum += Int64(decoded.exponent)
        }
        .store(in: &cancellables)

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iterations {
        for index in 0..<blockCount {
            subject.send(CombineWorkItem(blockIndex: index, block: blocks[index]))
        }
    }
    subject.send(completion: .finished)
    let elapsedNs = DispatchTime.now().uptimeNanoseconds - start

    _ = exponentSum

    let valuesBuffer = UnsafeBufferPointer(start: valuesPtr, count: valuesCapacity)
    let exponentsBuffer = UnsafeBufferPointer(start: exponentsPtr, count: blockCount)
    let checksumValue = checksum(values: valuesBuffer, exponents: exponentsBuffer)

    let tile = FixedPointTile(values: Array(valuesBuffer), exponents: Array(exponentsBuffer))
    let elapsedSeconds = Double(elapsedNs) / 1_000_000_000.0
    let logicalGBps = (Double(blockCount * iterations) * q8LogicalBytesPerBlock) / (elapsedSeconds * 1e9)

    return BenchmarkOutcome(tile: tile,
                            checksum: checksumValue,
                            elapsedSeconds: elapsedSeconds,
                            logicalGBps: logicalGBps,
                            blockCount: blockCount,
                            iterations: iterations,
                            chunkCount: max(2, workerCount),
                            engine: "combine")
}

func runBenchmarkActor(resolvedBlocks: BlockSource.RandomBlocks, iterations: Int, workerCount: Int) async -> BenchmarkOutcome {
    let blocks = resolvedBlocks.blocks
    let blockCount = blocks.count
    guard blockCount > 0 else {
        return BenchmarkOutcome(tile: FixedPointTile(values: [], exponents: []),
                                checksum: 0,
                                elapsedSeconds: 0,
                                logicalGBps: 0,
                                blockCount: 0,
                                iterations: iterations,
                                chunkCount: workerCount,
                                engine: "actor")
    }

    let valuesCapacity = blockCount * 32
    let valuesPtr = UnsafeMutablePointer<Int32>.allocate(capacity: valuesCapacity)
    let exponentsPtr = UnsafeMutablePointer<Int32>.allocate(capacity: blockCount)
    defer {
        valuesPtr.deallocate()
        exponentsPtr.deallocate()
    }

    let broadcaster = TileBroadcaster<CombineWorkItem>()
    var workers: [Task<Void, Never>] = []
    for workerID in 0..<workerCount {
        let (token, stream) = await broadcaster.subscribe(bufferLimit: 2)
        let task = Task {
            var iterator = stream.makeAsyncIterator()
            while let work = await iterator.next() {
                guard work.blockIndex % workerCount == workerID else { continue }
                let decoded = decodeScaleBits(work.block.scale)
                let base = (work.blockIndex % blockCount) * 32
                var dst = valuesPtr.advanced(by: base)
                for lane in 0..<16 {
                    dst.pointee = Int32(work.block.lo[lane]) * decoded.mantissa
                    dst = dst.advanced(by: 1)
                }
                for lane in 0..<16 {
                    dst.pointee = Int32(work.block.hi[lane]) * decoded.mantissa
                    dst = dst.advanced(by: 1)
                }
                exponentsPtr[work.blockIndex % blockCount] = decoded.exponent
            }
            await broadcaster.unsubscribe(token)
        }
        workers.append(task)
    }

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iterations {
        for index in 0..<blockCount {
            await broadcaster.send(CombineWorkItem(blockIndex: index, block: blocks[index]))
        }
    }
    await broadcaster.finish()
    for worker in workers {
        await worker.value
    }
    let elapsedNs = DispatchTime.now().uptimeNanoseconds - start

    let valuesBuffer = UnsafeBufferPointer(start: valuesPtr, count: valuesCapacity)
    let exponentsBuffer = UnsafeBufferPointer(start: exponentsPtr, count: blockCount)
    let checksumValue = checksum(values: valuesBuffer, exponents: exponentsBuffer)

    let tile = FixedPointTile(values: Array(valuesBuffer), exponents: Array(exponentsBuffer))
    let elapsedSeconds = Double(elapsedNs) / 1_000_000_000.0
    let logicalGBps = (Double(blockCount * iterations) * q8LogicalBytesPerBlock) / (elapsedSeconds * 1e9)

    return BenchmarkOutcome(tile: tile,
                            checksum: checksumValue,
                            elapsedSeconds: elapsedSeconds,
                            logicalGBps: logicalGBps,
                            blockCount: blockCount,
                            iterations: iterations,
                            chunkCount: workerCount,
                            engine: "actor")
}

func runBenchmarkMetal(resolvedBlocks: BlockSource.RandomBlocks, iterations: Int, runner: MetalDequantizer) throws -> BenchmarkOutcome {
    let blocks = resolvedBlocks.blocks
    let blockCount = blocks.count
    guard blockCount > 0 else {
        return BenchmarkOutcome(tile: FixedPointTile(values: [], exponents: []),
                                checksum: 0,
                                elapsedSeconds: 0,
                                logicalGBps: 0,
                                blockCount: 0,
                                iterations: iterations,
                                chunkCount: 0,
                                engine: "metal")
    }

    let (values, exponents, elapsedNs) = try runner.run(blocks: blocks, iterations: iterations)
    let elapsedSeconds = Double(elapsedNs) / 1_000_000_000.0

    let checksumValue = values.withUnsafeBufferPointer { valuePtr in
        exponents.withUnsafeBufferPointer { expPtr in
            checksum(values: valuePtr, exponents: expPtr)
        }
    }

    let tile = FixedPointTile(values: values, exponents: exponents)
    let logicalGBps = (Double(blockCount * iterations) * q8LogicalBytesPerBlock) / (elapsedSeconds * 1e9)

    return BenchmarkOutcome(tile: tile,
                            checksum: checksumValue,
                            elapsedSeconds: elapsedSeconds,
                            logicalGBps: logicalGBps,
                            blockCount: blockCount,
                            iterations: iterations,
                            chunkCount: blockCount,
                            engine: "metal")
}

func runPrototype(options: CLIOptions) async -> Never {
    do {
        let (blockSource, sourceDescriptor, tensorName) = try loadBlocks(using: options)
        let resolvedBlocks = blockSource.materialized()
        let blockCount = resolvedBlocks.blocks.count
        guard blockCount > 0 else {
            throw PrototypeError.noBlocksLoaded(tensorName ?? sourceDescriptor)
        }
        let iterations = options.iterations ?? defaultIterations(for: blockCount)
        let workerCount = options.workerOverride ?? ProcessInfo.processInfo.activeProcessorCount
        let defaultChunks = min(max(1, workerCount), 4)
        let chunkCount = options.chunkOverride.map { max(1, $0) } ?? defaultChunks

        let outcome: BenchmarkOutcome
        if options.useMetal {
            let runner = try MetalDequantizer()
            outcome = try runBenchmarkMetal(resolvedBlocks: resolvedBlocks, iterations: iterations, runner: runner)
        } else if options.useCombine {
            outcome = runBenchmarkCombine(resolvedBlocks: resolvedBlocks, iterations: iterations, workerCount: workerCount)
        } else if options.useActor {
            outcome = await runBenchmarkActor(resolvedBlocks: resolvedBlocks, iterations: iterations, workerCount: workerCount)
        } else {
            outcome = await runBenchmarkCPU(resolvedBlocks: resolvedBlocks, iterations: iterations, chunkCount: chunkCount)
        }

        var parts: [String] = []
        parts.append("source=\(sourceDescriptor)")
        if let tensorName {
            parts.append("tensor=\(tensorName)")
        }
        parts.append("blocks=\(outcome.blockCount)")
        parts.append("iterations=\(outcome.iterations)")
        parts.append(String(format: "elapsed=%.3fs", outcome.elapsedSeconds))
        parts.append(String(format: "logical=%.2f GB/s", outcome.logicalGBps))
        parts.append("checksum=\(outcome.checksum)")
        parts.append("chunks=\(outcome.chunkCount)")
        parts.append("workers=\(workerCount)")
        parts.append("engine=\(outcome.engine)")
        print(parts.joined(separator: " "))
        exit(EXIT_SUCCESS)
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(EXIT_FAILURE)
    }
}

let options = CLIOptions.parse()
Task {
    await runPrototype(options: options)
}
RunLoop.main.run()
