import Foundation
import simd

// MARK: - GGUF metadata types
enum GGUFType: UInt32 {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12
}

enum GGUFValue {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case uint64(UInt64)
    case int64(Int64)
    case float32(Float)
    case float64(Double)
    case bool(Bool)
    case string(String)
    case array([GGUFValue])

    var intValue: Int? {
        switch self {
        case .int8(let v): return Int(v)
        case .int16(let v): return Int(v)
        case .int32(let v): return Int(v)
        case .int64(let v): return v > Int.max ? nil : Int(v)
        case .uint8(let v): return Int(v)
        case .uint16(let v): return Int(v)
        case .uint32(let v): return v > UInt32(Int.max) ? nil : Int(v)
        case .uint64(let v): return v > UInt64(Int.max) ? nil : Int(v)
        default: return nil
        }
    }
}

enum GGMLType: UInt32 {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_k = 10
    case q3_k = 11
    case q4_k = 12
    case q5_k = 13
    case q6_k = 14
    case q8_k = 15
    case iq2_xxs = 16
    case iq2_xs = 17
    case iq3_xxs = 18
    case iq1_s = 19
    case iq4_nl = 20
    case iq3_s = 21
    case iq2_s = 22
    case iq4_xs = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1_m = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35

    var blockSize: Int? {
        switch self {
        case .q8_0: return 32
        default: return nil
        }
    }

    var bytesPerBlock: Int? {
        switch self {
        case .q8_0: return 34 // 32 weights + fp16 scale
        default: return nil
        }
    }
}

struct GGUFTensorInfo {
    let name: String
    let dimensions: [Int]
    let type: GGMLType
    let offset: UInt64
}

struct GGUFContext {
    let version: UInt32
    let metadata: [String: GGUFValue]
    let tensors: [GGUFTensorInfo]
    let dataOffset: Int
    let alignment: Int
    let data: Data
}

struct GGUFParser {
    let data: Data
    private(set) var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    mutating func parse() throws -> GGUFContext {
        guard try readString(length: 4) == "GGUF" else {
            throw Error.invalidMagic
        }

        let version = try readUInt32()
        let tensorCount = try readUInt64()
        let metadataCount = try readUInt64()

        var metadata: [String: GGUFValue] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let keyLength = try readUInt64()
            let key = try readString(length: Int(keyLength))
            guard let type = GGUFType(rawValue: try readUInt32()) else {
                throw Error.unknownType
            }
            let value = try readValue(of: type)
            metadata[key] = value
        }

        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            tensors.append(try readTensorInfo())
        }

        let alignment = metadata["general.alignment"]?.intValue ?? 32
        let alignedOffset = align(offset, to: alignment)

        return GGUFContext(version: version,
                           metadata: metadata,
                           tensors: tensors,
                           dataOffset: alignedOffset,
                           alignment: alignment,
                           data: data)
    }

    private mutating func readValue(of type: GGUFType) throws -> GGUFValue {
        switch type {
        case .uint8:
            return .uint8(try readUInt8())
        case .int8:
            return .int8(try readInt8())
        case .uint16:
            return .uint16(try readUInt16())
        case .int16:
            return .int16(try readInt16())
        case .uint32:
            return .uint32(try readUInt32())
        case .int32:
            return .int32(try readInt32())
        case .uint64:
            return .uint64(try readUInt64())
        case .int64:
            return .int64(try readInt64())
        case .float32:
            return .float32(try readFloat32())
        case .float64:
            return .float64(try readFloat64())
        case .bool:
            return .bool(try readUInt8() != 0)
        case .string:
            let length = try readUInt64()
            return .string(try readString(length: Int(length)))
        case .array:
            guard let elementType = GGUFType(rawValue: try readUInt32()) else {
                throw Error.unknownType
            }
            let count = try readUInt64()
            var values: [GGUFValue] = []
            values.reserveCapacity(Int(count))
            for _ in 0..<count {
                values.append(try readValue(of: elementType))
            }
            return .array(values)
        }
    }

    private mutating func readTensorInfo() throws -> GGUFTensorInfo {
        let nameLength = try readUInt64()
        let name = try readString(length: Int(nameLength))
        let nDims = Int(try readUInt32())
        var dims: [Int] = []
        dims.reserveCapacity(nDims)
        for _ in 0..<nDims {
            dims.append(Int(try readUInt64()))
        }
        guard let type = GGMLType(rawValue: try readUInt32()) else {
            throw Error.unknownTensorType
        }
        let offset = try readUInt64()
        return GGUFTensorInfo(name: name, dimensions: dims, type: type, offset: offset)
    }

    private mutating func readUInt8() throws -> UInt8 {
        guard offset + 1 <= data.count else { throw Error.unexpectedEOF }
        defer { offset += 1 }
        return data[offset]
    }

    private mutating func readInt8() throws -> Int8 {
        return Int8(bitPattern: try readUInt8())
    }

    private mutating func readUInt16() throws -> UInt16 {
        return UInt16(littleEndian: try readInteger())
    }

    private mutating func readInt16() throws -> Int16 {
        return Int16(bitPattern: try readUInt16())
    }

    private mutating func readUInt32() throws -> UInt32 {
        return UInt32(littleEndian: try readInteger())
    }

    private mutating func readInt32() throws -> Int32 {
        return Int32(bitPattern: try readUInt32())
    }

    private mutating func readUInt64() throws -> UInt64 {
        return UInt64(littleEndian: try readInteger())
    }

    private mutating func readInt64() throws -> Int64 {
        return Int64(bitPattern: try readUInt64())
    }

    private mutating func readFloat32() throws -> Float {
        return Float(bitPattern: try readUInt32())
    }

    private mutating func readFloat64() throws -> Double {
        return Double(bitPattern: try readUInt64())
    }

    private mutating func readString(length: Int) throws -> String {
        guard length >= 0, offset + length <= data.count else { throw Error.unexpectedEOF }
        let range = offset..<(offset + length)
        defer { offset += length }
        return String(decoding: data[range], as: UTF8.self)
    }

    private mutating func readInteger<T: FixedWidthInteger>() throws -> T {
        let size = MemoryLayout<T>.size
        guard offset + size <= data.count else { throw Error.unexpectedEOF }
        let value: T = data.withUnsafeBytes { buffer in
            let base = buffer.baseAddress!.advanced(by: offset)
            return base.load(as: T.self)
        }
        offset += size
        return value
    }

    private func align(_ value: Int, to alignment: Int) -> Int {
        guard alignment > 0 else { return value }
        let mask = alignment - 1
        return (value + mask) & ~mask
    }

    enum Error: Swift.Error {
        case invalidMagic
        case unexpectedEOF
        case unknownType
        case unknownTensorType
    }
}

// MARK: - GGUF Loader -> Q8 blocks
struct GGUFLoader {
    enum Error: Swift.Error, CustomStringConvertible {
        case fileNotFound(String)
        case noQ8Tensor
        case tensorNotFound(String)
        case tensorNotQ8(String)
        case insufficientBlocks(tensor: String)

        var description: String {
            switch self {
            case .fileNotFound(let path):
                return "File not found: \(path)"
            case .noQ8Tensor:
                return "No Q8_0 tensor present in GGUF file"
            case .tensorNotFound(let tensor):
                return "Tensor \(tensor) not found"
            case .tensorNotQ8(let tensor):
                return "Tensor \(tensor) is not of type Q8_0"
            case .insufficientBlocks(let tensor):
                return "Tensor \(tensor) did not yield any Q8_0 blocks"
            }
        }
    }

    private let context: GGUFContext

    init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Error.fileNotFound(path)
        }
        let mappedData = try Data(contentsOf: url, options: .mappedIfSafe)
        var parser = GGUFParser(data: mappedData)
        context = try parser.parse()
    }

    func defaultQ8TensorName() -> String? {
        context.tensors.first { $0.type == .q8_0 }?.name
    }

    func makeQ8BlockSource(tensorName: String, maxBlocks: Int) throws -> BlockSource.GGUFBlocks {
        guard let info = context.tensors.first(where: { $0.name == tensorName }) else {
            throw Error.tensorNotFound(tensorName)
        }
        guard info.type == .q8_0, let bytesPerBlock = info.type.bytesPerBlock, let valuesPerBlock = info.type.blockSize else {
            throw Error.tensorNotQ8(tensorName)
        }

        let totalElements = info.dimensions.reduce(1) { partial, dim in
            let (result, overflow) = partial.multipliedReportingOverflow(by: dim)
            return overflow ? Int.max : result
        }
        guard totalElements > 0 else { throw Error.insufficientBlocks(tensor: tensorName) }
        let totalBlocks = (totalElements + valuesPerBlock - 1) / valuesPerBlock
        let blocksToRead = min(totalBlocks, maxBlocks)
        let requiredBytes = blocksToRead * bytesPerBlock

        let baseOffset = context.dataOffset + Int(info.offset)
        guard baseOffset + requiredBytes <= context.data.count else {
            throw Error.insufficientBlocks(tensor: tensorName)
        }

        return BlockSource.GGUFBlocks(data: context.data,
                                      baseOffset: baseOffset,
                                      bytesPerBlock: bytesPerBlock,
                                      blockCount: blocksToRead)
    }
}
