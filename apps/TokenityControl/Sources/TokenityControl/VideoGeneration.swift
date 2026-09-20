import AVFoundation
import CoreMedia
import CoreVideo
import Foundation

enum H3OptimizationProfile: String, CaseIterable, Codable, Identifiable {
    case stockQMM = "stock-qmm"
    case baseline
    case blockFusions = "block-fusions"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .stockQMM: return "Stock fused qmm"
        case .baseline: return "DQ + Steel baseline"
        case .blockFusions: return "DQ + block fusions"
        }
    }

    var detail: String {
        switch self {
        case .stockQMM:
            return "Validated default for the 512×256 TP2 workload."
        case .baseline:
            return "Compatibility profile for reproducing the full-weight DQ baseline."
        case .blockFusions:
            return "Experimental AdaLN, gate/residual and SwiGLU fusions."
        }
    }
}

enum VideoRuntimeState: Equatable {
    case stopped
    case starting
    case ready
    case stopping
    case failed(String)

    var title: String {
        switch self {
        case .stopped: return "Stopped"
        case .starting: return "Starting"
        case .ready: return "Ready"
        case .stopping: return "Stopping"
        case .failed: return "Failed"
        }
    }

    var detail: String? {
        guard case .failed(let message) = self else { return nil }
        return message
    }
}

enum VideoRecoveryAction: String, Hashable {
    case installOrRepair = "Install or Repair"
    case chooseFolder = "Choose Folder"
    case scanAgain = "Scan Again"
    case retry = "Retry"
    case useOneMac = "Use One Mac"
    case openNodeDetails = "Open Mac Details"
}

struct VideoReadinessIssue: Identifiable, Hashable {
    enum State: String, Hashable {
        case componentsNeedUpdate
        case modelNotFound
        case runtimeMissing
        case secondMacUnavailable
        case highSpeedConnectionUnavailable
        case runtimeInterrupted
    }

    var state: State
    var message: String
    var action: VideoRecoveryAction

    var id: String { "\(state.rawValue):\(message):\(action.rawValue)" }
}

struct AgentStartH3VideoRequest: Encodable {
    var model: String
    var binary: String
    var nodes: [AgentClusterNodeRequest]
    var connectionMode: String
    var startingPort: Int
    var host: String
    var port: Int
    var dryRun: Bool
    var apiIdentifier: String
    var leaseSeconds: Double
    var instanceID: String
    var operationID: String
    var optimizationProfile: H3OptimizationProfile
    var minimumFreeDiskBytes: Int64 = 2 * 1_024 * 1_024 * 1_024
    var memoryHeadroomRatio: Double? = nil

    enum CodingKeys: String, CodingKey {
        case model, binary, nodes, host, port
        case connectionMode = "connection_mode"
        case startingPort = "starting_port"
        case dryRun = "dry_run"
        case apiIdentifier = "api_identifier"
        case leaseSeconds = "lease_seconds"
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case optimizationProfile = "optimization_profile"
        case minimumFreeDiskBytes = "minimum_free_disk_bytes"
        case memoryHeadroomRatio = "memory_headroom_ratio"
    }
}

struct H3TurboReadiness: Decodable {
    var ready: Bool
    var issues: [String]
}

struct H3VideoRuntimeResponse: Decodable {
    var instanceID: String?
    var turbo: H3TurboReadiness?

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case turbo
    }
}

enum H3TurboPreset: Int, CaseIterable, Identifiable {
    case preview = 4
    case balanced = 6
    case detail = 8

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .preview: return "4 steps · Preview"
        case .balanced: return "6 steps · Balanced"
        case .detail: return "8 steps · Detail"
        }
    }
}

struct H3VideoGenerationRequest: Encodable, Equatable {
    var model: String? = nil
    var prompt: String
    var width: Int
    var height: Int
    var numFrames: Int
    var steps: Int
    var seed: Int
    var fast: Bool
    var turbo: Bool = false
    var stream: Bool

    // Mode preferences belong to this request editor, not the wire contract.
    private var ordinarySteps = 28
    private var ordinaryFast = false
    private var turboSteps = 6

    init(model: String? = nil, prompt: String, width: Int, height: Int,
         numFrames: Int, steps: Int, seed: Int, fast: Bool,
         turbo: Bool = false, stream: Bool) {
        self.model = model
        self.prompt = prompt
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = steps
        self.seed = seed
        self.fast = turbo ? false : fast
        self.turbo = turbo
        self.stream = stream
        if turbo {
            turboSteps = steps
        } else {
            ordinarySteps = steps
            ordinaryFast = fast
        }
    }

    mutating func setTurbo(_ enabled: Bool) {
        guard enabled != turbo else { return }
        if enabled {
            ordinarySteps = steps
            ordinaryFast = fast
            steps = turboSteps
            fast = false
        } else {
            turboSteps = steps
            steps = ordinarySteps
            fast = ordinaryFast
        }
        turbo = enabled
    }

    static let `default` = H3VideoGenerationRequest(
        prompt: "A cinematic tracking shot follows a weathered paper boat through a rain-soaked neon night market; the camera begins at water level, glides past steaming food stalls and pedestrians beneath translucent umbrellas, then rises into a wide overhead reveal as reflections ripple across the street, with realistic lighting, shallow depth of field, and natural motion.",
        width: 512,
        height: 256,
        numFrames: 124,
        steps: 28,
        seed: 42,
        fast: false,
        stream: true
    )

    enum CodingKeys: String, CodingKey {
        case model, prompt, width, height, steps, seed, fast, turbo, stream
        case numFrames = "num_frames"
    }

    func validated() -> H3VideoGenerationRequest {
        var copy = self
        copy.prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        copy.width = Self.validatedDimension(width)
        copy.height = Self.validatedDimension(height)
        copy.numFrames = Self.validatedFrameCount(numFrames)
        copy.steps = min(max(steps, 1), 50)
        if copy.turbo { copy.fast = false }
        copy.stream = true
        return copy
    }

    private static func validatedDimension(_ value: Int) -> Int {
        let bounded = min(max(value, 128), 1_024)
        return min(1_024, max(128, Int((Double(bounded) / 32).rounded()) * 32))
    }

    private static func validatedFrameCount(_ value: Int) -> Int {
        let bounded = min(max(value, 5), 345)
        let ladderIndex = Int(ceil(Double(bounded - 5) / 17))
        return min(345, ladderIndex * 17 + 5)
    }
}

struct H3VideoCompletePayload: Equatable {
    var frames: Int
    var height: Int
    var width: Int
    var fps: Int
    var format: String
    var videoData: Data
    var audioChannels: Int
    var audioFormat: String
    var audioSampleRate: Int
    var audioData: Data
}

enum H3VideoSSEEvent: Equatable {
    case progress(stage: String, step: Int, total: Int)
    case complete(H3VideoCompletePayload)
}

enum H3VideoContractError: LocalizedError {
    case invalidEvent
    case server(String)
    case unsupportedFormat(String)
    case invalidVideoLength(expected: Int, actual: Int)
    case invalidBase64(String)
    case mediaEncoding(String)

    var errorDescription: String? {
        switch self {
        case .invalidEvent:
            return "The MiniMax H3 server returned an unreadable video event."
        case .server(let message):
            return message
        case .unsupportedFormat(let format):
            return "The generated video uses unsupported format \(format)."
        case .invalidVideoLength(let expected, let actual):
            return "The generated video payload is incomplete (expected \(expected) bytes, received \(actual))."
        case .invalidBase64(let field):
            return "The generated \(field) payload is not valid base64."
        case .mediaEncoding(let detail):
            return "The generated media could not be saved: \(detail)"
        }
    }
}

enum H3VideoSSEParser {
    private struct Kind: Decodable {
        var type: String
    }

    private struct ServerError: Decodable {
        var message: String
    }

    private struct Progress: Decodable {
        var stage: String
        var step: Int
        var total: Int
    }

    private struct Complete: Decodable {
        var frames: Int
        var height: Int
        var width: Int
        var fps: Int
        var format: String
        var data: String
        var audioChannels: Int?
        var audioFormat: String?
        var audioSampleRate: Int?
        var audioData: String?

        enum CodingKeys: String, CodingKey {
            case frames, height, width, fps, format, data
            case audioChannels = "audio_channels"
            case audioFormat = "audio_format"
            case audioSampleRate = "audio_sample_rate"
            case audioData = "audio_data"
        }
    }

    static func parse(line: String) throws -> H3VideoSSEEvent? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard payload != "[DONE]", !payload.isEmpty else { return nil }
        guard let encoded = payload.data(using: .utf8) else {
            throw H3VideoContractError.invalidEvent
        }
        let decoder = JSONDecoder()
        switch try decoder.decode(Kind.self, from: encoded).type {
        case "error":
            throw H3VideoContractError.server(try decoder.decode(ServerError.self, from: encoded).message)
        case "progress":
            let event = try decoder.decode(Progress.self, from: encoded)
            return .progress(stage: event.stage, step: event.step, total: event.total)
        case "complete":
            let event = try decoder.decode(Complete.self, from: encoded)
            guard event.format == "rgb8" else {
                throw H3VideoContractError.unsupportedFormat(event.format)
            }
            guard let video = Data(base64Encoded: event.data) else {
                throw H3VideoContractError.invalidBase64("video")
            }
            let expected = event.frames
                .multipliedReportingOverflow(by: event.height)
            guard !expected.overflow else { throw H3VideoContractError.invalidEvent }
            let expectedPixels = expected.partialValue.multipliedReportingOverflow(by: event.width)
            guard !expectedPixels.overflow else { throw H3VideoContractError.invalidEvent }
            let expectedBytes = expectedPixels.partialValue.multipliedReportingOverflow(by: 3)
            guard !expectedBytes.overflow else { throw H3VideoContractError.invalidEvent }
            guard video.count == expectedBytes.partialValue else {
                throw H3VideoContractError.invalidVideoLength(
                    expected: expectedBytes.partialValue,
                    actual: video.count
                )
            }
            let audio: Data
            if let rawAudio = event.audioData, !rawAudio.isEmpty {
                guard let decodedAudio = Data(base64Encoded: rawAudio) else {
                    throw H3VideoContractError.invalidBase64("audio")
                }
                audio = decodedAudio
            } else {
                audio = Data()
            }
            return .complete(
                H3VideoCompletePayload(
                    frames: event.frames,
                    height: event.height,
                    width: event.width,
                    fps: event.fps,
                    format: event.format,
                    videoData: video,
                    audioChannels: event.audioChannels ?? 0,
                    audioFormat: event.audioFormat ?? "none",
                    audioSampleRate: event.audioSampleRate ?? 0,
                    audioData: audio
                )
            )
        default:
            return nil
        }
    }
}

struct GeneratedVideoArtifact: Identifiable, Equatable {
    var id: UUID
    var directoryURL: URL
    var movieURL: URL
    var waveURL: URL?
    var rawVideoURL: URL
    var rawAudioURL: URL?
    var frames: Int
    var width: Int
    var height: Int
    var fps: Int
    var durationSeconds: Double
    var hasMuxedAudio: Bool
}

struct VideoArtifactHistory {
    var completed: [GeneratedVideoArtifact]
    var interruptedCount: Int
}

enum H3VideoArtifactWriter {
    static func write(
        payload: H3VideoCompletePayload,
        request: H3VideoGenerationRequest,
        rootDirectory: URL? = nil
    ) async throws -> GeneratedVideoArtifact {
        let saveStarted = Date()
        let id = UUID()
        let root = rootDirectory ?? defaultRootDirectory()
        let directory = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let rawVideoURL = directory.appendingPathComponent("video.rgb")
        try payload.videoData.write(to: rawVideoURL, options: .atomic)
        let rawAudioURL = payload.audioData.isEmpty ? nil : directory.appendingPathComponent("audio.pcm")
        if let rawAudioURL {
            try payload.audioData.write(to: rawAudioURL, options: .atomic)
        }

        let waveURL: URL?
        if !payload.audioData.isEmpty,
           payload.audioFormat == "pcm_s16le",
           payload.audioChannels > 0,
           payload.audioSampleRate > 0 {
            let url = directory.appendingPathComponent("audio.wav")
            try writeWave(
                pcm: payload.audioData,
                channels: payload.audioChannels,
                sampleRate: payload.audioSampleRate,
                to: url
            )
            waveURL = url
        } else {
            waveURL = nil
        }

        let videoOnlyURL = directory.appendingPathComponent("video-only.mov")
        try await encodeRGBVideo(payload: payload, to: videoOnlyURL)
        let movieURL = directory.appendingPathComponent("generation.mov")
        var hasMuxedAudio = false
        if let waveURL {
            do {
                try await mux(videoURL: videoOnlyURL, audioURL: waveURL, outputURL: movieURL)
                hasMuxedAudio = true
                try? FileManager.default.removeItem(at: videoOnlyURL)
            } catch {
                try FileManager.default.copyItem(at: videoOnlyURL, to: movieURL)
            }
        } else {
            try FileManager.default.moveItem(at: videoOnlyURL, to: movieURL)
        }

        let metadata = ArtifactMetadata(
            prompt: request.prompt,
            seed: request.seed,
            steps: request.steps,
            fast: request.fast,
            turbo: request.turbo,
            turboStrength: request.turbo ? 1.0 : nil,
            saveSeconds: Date().timeIntervalSince(saveStarted),
            frames: payload.frames,
            width: payload.width,
            height: payload.height,
            fps: payload.fps,
            hasMuxedAudio: hasMuxedAudio
        )
        let metadataURL = directory.appendingPathComponent("metadata.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: metadataURL, options: .atomic)

        return GeneratedVideoArtifact(
            id: id,
            directoryURL: directory,
            movieURL: movieURL,
            waveURL: waveURL,
            rawVideoURL: rawVideoURL,
            rawAudioURL: rawAudioURL,
            frames: payload.frames,
            width: payload.width,
            height: payload.height,
            fps: payload.fps,
            durationSeconds: Double(payload.frames) / Double(max(payload.fps, 1)),
            hasMuxedAudio: hasMuxedAudio
        )
    }

    private struct ArtifactMetadata: Codable {
        var prompt: String
        var seed: Int
        var steps: Int
        var fast: Bool
        var turbo: Bool?
        var turboStrength: Double?
        var saveSeconds: Double?
        var frames: Int
        var width: Int
        var height: Int
        var fps: Int
        var hasMuxedAudio: Bool

        enum CodingKeys: String, CodingKey {
            case prompt, seed, steps, fast, turbo, frames, width, height, fps
            case turboStrength = "turbo_strength"
            case saveSeconds = "save_seconds"
            case hasMuxedAudio = "has_muxed_audio"
        }
    }

    static func loadHistory(
        rootDirectory: URL? = nil,
        limit: Int = 20
    ) -> VideoArtifactHistory {
        let root = rootDirectory ?? defaultRootDirectory()
        guard let directories = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return VideoArtifactHistory(completed: [], interruptedCount: 0)
        }

        var completed: [(Date, GeneratedVideoArtifact)] = []
        var interruptedCount = 0
        for directory in directories {
            guard let id = UUID(uuidString: directory.lastPathComponent) else { continue }
            let metadataURL = directory.appendingPathComponent("metadata.json")
            let movieURL = directory.appendingPathComponent("generation.mov")
            guard FileManager.default.fileExists(atPath: movieURL.path),
                  let data = try? Data(contentsOf: metadataURL),
                  let metadata = try? JSONDecoder().decode(ArtifactMetadata.self, from: data)
            else {
                if FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent("video.rgb").path
                ) {
                    interruptedCount += 1
                }
                continue
            }
            let waveURL = directory.appendingPathComponent("audio.wav")
            let rawVideoURL = directory.appendingPathComponent("video.rgb")
            let rawAudioURL = directory.appendingPathComponent("audio.pcm")
            let modified = (try? directory.resourceValues(
                forKeys: [.contentModificationDateKey]
            ).contentModificationDate) ?? .distantPast
            completed.append((
                modified,
                GeneratedVideoArtifact(
                    id: id,
                    directoryURL: directory,
                    movieURL: movieURL,
                    waveURL: FileManager.default.fileExists(atPath: waveURL.path) ? waveURL : nil,
                    rawVideoURL: rawVideoURL,
                    rawAudioURL: FileManager.default.fileExists(atPath: rawAudioURL.path) ? rawAudioURL : nil,
                    frames: metadata.frames,
                    width: metadata.width,
                    height: metadata.height,
                    fps: metadata.fps,
                    durationSeconds: Double(metadata.frames) / Double(max(metadata.fps, 1)),
                    hasMuxedAudio: metadata.hasMuxedAudio
                )
            ))
        }
        return VideoArtifactHistory(
            completed: completed.sorted { $0.0 > $1.0 }.prefix(max(0, limit)).map { $0.1 },
            interruptedCount: interruptedCount
        )
    }

    static func defaultRootDirectory() -> URL {
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        return applicationSupport
            .appendingPathComponent("Tokenity", isDirectory: true)
            .appendingPathComponent("GeneratedVideos", isDirectory: true)
    }

    private static func encodeRGBVideo(
        payload: H3VideoCompletePayload,
        to outputURL: URL
    ) async throws {
        guard payload.width > 0, payload.height > 0, payload.frames > 0, payload.fps > 0 else {
            throw H3VideoContractError.invalidEvent
        }
        try? FileManager.default.removeItem(at: outputURL)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: payload.width,
                AVVideoHeightKey: payload.height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: max(2_000_000, payload.width * payload.height * 12),
                    AVVideoExpectedSourceFrameRateKey: payload.fps,
                ],
            ]
        )
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: payload.width,
                kCVPixelBufferHeightKey as String: payload.height,
            ]
        )
        guard writer.canAdd(input) else {
            throw H3VideoContractError.mediaEncoding("The system video encoder rejected the RGB stream.")
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw H3VideoContractError.mediaEncoding(writer.error?.localizedDescription ?? "Video writer did not start.")
        }
        writer.startSession(atSourceTime: .zero)

        let bytesPerFrame = payload.width * payload.height * 3
        for frame in 0..<payload.frames {
            try Task.checkCancellation()
            while !input.isReadyForMoreMediaData {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(2))
            }
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                kCFAllocatorDefault,
                payload.width,
                payload.height,
                kCVPixelFormatType_32BGRA,
                [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
                &pixelBuffer
            )
            guard status == kCVReturnSuccess, let pixelBuffer else {
                writer.cancelWriting()
                throw H3VideoContractError.mediaEncoding("Could not allocate a video frame buffer.")
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            guard let destinationBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                writer.cancelWriting()
                throw H3VideoContractError.mediaEncoding("Video frame memory is unavailable.")
            }
            let destinationStride = CVPixelBufferGetBytesPerRow(pixelBuffer)
            payload.videoData.withUnsafeBytes { rawSource in
                guard let sourceBase = rawSource.bindMemory(to: UInt8.self).baseAddress else { return }
                let source = sourceBase.advanced(by: frame * bytesPerFrame)
                let destination = destinationBase.assumingMemoryBound(to: UInt8.self)
                for row in 0..<payload.height {
                    let sourceRow = source.advanced(by: row * payload.width * 3)
                    let destinationRow = destination.advanced(by: row * destinationStride)
                    for column in 0..<payload.width {
                        let sourcePixel = sourceRow.advanced(by: column * 3)
                        let destinationPixel = destinationRow.advanced(by: column * 4)
                        destinationPixel[0] = sourcePixel[2]
                        destinationPixel[1] = sourcePixel[1]
                        destinationPixel[2] = sourcePixel[0]
                        destinationPixel[3] = 255
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            let presentationTime = CMTime(value: CMTimeValue(frame), timescale: CMTimeScale(payload.fps))
            guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                writer.cancelWriting()
                throw H3VideoContractError.mediaEncoding(writer.error?.localizedDescription ?? "Could not append a video frame.")
            }
        }

        input.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        guard writer.status == .completed else {
            throw H3VideoContractError.mediaEncoding(writer.error?.localizedDescription ?? "Video encoding failed.")
        }
    }

    private static func writeWave(
        pcm: Data,
        channels: Int,
        sampleRate: Int,
        to outputURL: URL
    ) throws {
        guard channels > 0, sampleRate > 0, pcm.count <= Int(UInt32.max) else {
            throw H3VideoContractError.invalidEvent
        }
        let bitsPerSample = 16
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        var wave = Data()

        func appendASCII(_ value: String) {
            wave.append(contentsOf: value.utf8)
        }
        func appendLE<T: FixedWidthInteger>(_ value: T) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { wave.append(contentsOf: $0) }
        }

        appendASCII("RIFF")
        appendLE(UInt32(36 + pcm.count))
        appendASCII("WAVEfmt ")
        appendLE(UInt32(16))
        appendLE(UInt16(1))
        appendLE(UInt16(channels))
        appendLE(UInt32(sampleRate))
        appendLE(UInt32(byteRate))
        appendLE(UInt16(blockAlign))
        appendLE(UInt16(bitsPerSample))
        appendASCII("data")
        appendLE(UInt32(pcm.count))
        wave.append(pcm)
        try wave.write(to: outputURL, options: .atomic)
    }

    private static func mux(videoURL: URL, audioURL: URL, outputURL: URL) async throws {
        try? FileManager.default.removeItem(at: outputURL)
        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)
        let videoTracks = try await videoAsset.loadTracks(withMediaType: .video)
        let audioTracks = try await audioAsset.loadTracks(withMediaType: .audio)
        guard let sourceVideo = videoTracks.first, let sourceAudio = audioTracks.first else {
            throw H3VideoContractError.mediaEncoding("A generated video or audio track is missing.")
        }
        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ), let audioTrack = composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            throw H3VideoContractError.mediaEncoding("Could not create the output media tracks.")
        }
        let videoDuration = try await videoAsset.load(.duration)
        let audioDuration = try await audioAsset.load(.duration)
        try videoTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: videoDuration),
            of: sourceVideo,
            at: .zero
        )
        let muxedAudioDuration = CMTimeCompare(audioDuration, videoDuration) < 0
            ? audioDuration
            : videoDuration
        try audioTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: muxedAudioDuration),
            of: sourceAudio,
            at: .zero
        )
        guard let exporter = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetPassthrough
        ) else {
            throw H3VideoContractError.mediaEncoding("The system media muxer is unavailable.")
        }
        exporter.outputURL = outputURL
        exporter.outputFileType = .mov
        await withCheckedContinuation { continuation in
            exporter.exportAsynchronously { continuation.resume() }
        }
        guard exporter.status == .completed else {
            throw H3VideoContractError.mediaEncoding(exporter.error?.localizedDescription ?? "Audio/video mux failed.")
        }
    }
}
