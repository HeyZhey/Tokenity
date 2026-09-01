import Foundation

enum BenchmarkKind: String, CaseIterable, Codable, Identifiable {
    case languageModel = "language_model"
    case videoGeneration = "video_generation"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .languageModel: return "Language Model"
        case .videoGeneration: return "Video Generation"
        }
    }

    var modality: ModelModality {
        switch self {
        case .languageModel: return .language
        case .videoGeneration: return .video
        }
    }
}

enum BenchmarkTarget: String, CaseIterable, Codable, Identifiable {
    case currentMac = "current_mac"
    case selectedNode = "selected_node"
    case tensorParallel2 = "tp2"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .currentMac: return "Current Mac"
        case .selectedNode: return "Selected Mac"
        case .tensorParallel2: return "Multi-Mac TP2"
        }
    }

    var topology: String {
        switch self {
        case .currentMac: return "single-current"
        case .selectedNode: return "single-selected"
        case .tensorParallel2: return "tp2"
        }
    }

    var requiredNodeCount: Int { self == .tensorParallel2 ? 2 : 1 }
}

enum BenchmarkProfile: String, CaseIterable, Codable, Identifiable {
    case quick
    case standard
    case custom

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
}

enum BenchmarkPhase: String, CaseIterable, Codable {
    case idle
    case preflight
    case loading
    case warmup
    case running
    case cancelling
    case completed
    case failed

    var title: String {
        switch self {
        case .idle: return "Idle"
        case .preflight: return "Preflight"
        case .loading: return "Loading"
        case .warmup: return "Warmup"
        case .running: return "Running"
        case .cancelling: return "Cancelling"
        case .completed: return "Completed"
        case .failed: return "Failed"
        }
    }

    var isActive: Bool {
        switch self {
        case .preflight, .loading, .warmup, .running, .cancelling: return true
        case .idle, .completed, .failed: return false
        }
    }
}

enum BenchmarkThermalState: String, Codable, CaseIterable {
    case cold = "Cold"
    case warm = "Warm"
}

enum BenchmarkSampleStatus: String, Codable {
    case succeeded
    case failed
    case cancelled
}

struct BenchmarkResolutionPreset: Codable, Hashable, Identifiable {
    var width: Int
    var height: Int

    var id: String { "\(width)x\(height)" }
    var title: String { "\(width) × \(height)" }
    var isH3Valid: Bool { width > 0 && height > 0 && width.isMultiple(of: 32) && height.isMultiple(of: 32) }

    static let h3Presets = [
        BenchmarkResolutionPreset(width: 256, height: 256),
        BenchmarkResolutionPreset(width: 512, height: 256),
        BenchmarkResolutionPreset(width: 512, height: 288),
        BenchmarkResolutionPreset(width: 512, height: 512),
    ]
}

struct BenchmarkDurationPreset: Codable, Hashable, Identifiable {
    var name: String
    var frames: Int
    var nominalFPS: Int = 24

    var id: String { name.lowercased() }
    var seconds: Double { Double(frames) / Double(nominalFPS) }
    var title: String { "\(name): \(frames) frames · ≈ \(String(format: "%.1f", seconds)) s" }
    var isH3Valid: Bool { frames >= 5 && frames <= 345 && (frames - 5).isMultiple(of: 17) }

    static let h3Presets = [
        BenchmarkDurationPreset(name: "Short", frames: 22),
        BenchmarkDurationPreset(name: "Standard", frames: 124),
        BenchmarkDurationPreset(name: "Long", frames: 243),
        BenchmarkDurationPreset(name: "Extended", frames: 345),
    ]
}

struct BenchmarkLLMInputSize: Codable, Hashable, Identifiable {
    var tokenCount: Int

    var id: Int { tokenCount }
    var title: String {
        tokenCount == 512 ? "512" : "\(tokenCount / 1_024)K"
    }

    static let allCases = [
        BenchmarkLLMInputSize(tokenCount: 512),
        BenchmarkLLMInputSize(tokenCount: 1_024),
        BenchmarkLLMInputSize(tokenCount: 2_048),
        BenchmarkLLMInputSize(tokenCount: 4_096),
        BenchmarkLLMInputSize(tokenCount: 8_192),
        BenchmarkLLMInputSize(tokenCount: 16_384),
        BenchmarkLLMInputSize(tokenCount: 32_768),
        BenchmarkLLMInputSize(tokenCount: 65_536),
        BenchmarkLLMInputSize(tokenCount: 131_072),
    ]

    static let allowedTokenCounts = Set(allCases.map(\.tokenCount))

    static func title(for tokenCount: Int) -> String {
        allCases.first { $0.tokenCount == tokenCount }?.title
            ?? tokenCount.formatted()
    }
}

struct BenchmarkConfiguration: Codable, Hashable {
    static let languagePromptProfile = "tokenity-llm-input-sweep-v2"
    static let videoPromptProfile = "tokenity-h3-complex-v1"
    static let languagePrompt = """
    Explain how a distributed inference system can preserve correctness while reducing latency. Include one concrete failure mode, one mitigation, and a short verification checklist.
    """

    var kind: BenchmarkKind
    var target: BenchmarkTarget
    var targetNodeID: String? = nil
    var modelID: String
    var modelRevision: String? = nil
    var quantization: String? = nil
    var profile: BenchmarkProfile
    var customRuns: Int
    var promptProfile: String
    var prompt: String
    var concurrency: Int
    var temperature: Double
    var maximumOutputTokens: Int
    /// Optional for schema-v1 history compatibility. A missing value means the original 512-token workload.
    var inputTokenSizes: [Int]? = nil
    var width: Int
    var height: Int
    var frames: Int
    var steps: Int
    var seed: Int
    var fast: Bool
    var saveVideoPreview: Bool

    static func language(modelID: String = "") -> BenchmarkConfiguration {
        BenchmarkConfiguration(
            kind: .languageModel,
            target: .currentMac,
            modelID: modelID,
            profile: .standard,
            customRuns: 5,
            promptProfile: languagePromptProfile,
            prompt: languagePrompt,
            concurrency: 1,
            temperature: 0,
            maximumOutputTokens: 256,
            inputTokenSizes: [512],
            width: 512,
            height: 256,
            frames: 124,
            steps: 28,
            seed: 42,
            fast: false,
            saveVideoPreview: false
        )
    }

    static func video(modelID: String = "MiniMax-H3") -> BenchmarkConfiguration {
        let request = H3VideoGenerationRequest.default
        return BenchmarkConfiguration(
            kind: .videoGeneration,
            target: .currentMac,
            modelID: modelID,
            profile: .standard,
            customRuns: 4,
            promptProfile: videoPromptProfile,
            prompt: request.prompt,
            concurrency: 1,
            temperature: 0,
            maximumOutputTokens: 256,
            inputTokenSizes: nil,
            width: request.width,
            height: request.height,
            frames: request.numFrames,
            steps: request.steps,
            seed: request.seed,
            fast: request.fast,
            saveVideoPreview: false
        )
    }

    var warmupRuns: Int {
        guard kind == .languageModel else { return 0 }
        switch profile {
        case .quick, .standard: return 1
        case .custom: return 1
        }
    }

    var measuredRuns: Int {
        switch (kind, profile) {
        case (.languageModel, .quick): return 3
        case (.languageModel, .standard): return 5
        case (.videoGeneration, .quick): return 1
        case (.videoGeneration, .standard): return 4
        case (_, .custom): return min(max(customRuns, 1), 20)
        }
    }

    var runsPerInputSize: Int { warmupRuns + measuredRuns }

    var selectedInputTokenSizes: [Int] {
        let configured = inputTokenSizes ?? [512]
        return BenchmarkLLMInputSize.allCases
            .map(\.tokenCount)
            .filter { configured.contains($0) }
    }

    var totalRuns: Int {
        let workloadCount = kind == .languageModel ? max(1, selectedInputTokenSizes.count) : 1
        return runsPerInputSize * workloadCount
    }

    func languagePrompt(requestedInputTokens: Int) -> String {
        // Leading-space " token" is one token in the supported Qwen-family tokenizers. The
        // server-reported prompt_tokens remains authoritative and is stored beside this target.
        let suffix = "\n\n" + Self.languagePrompt
        let estimatedSuffixTokens = 40
        let fillerCount = max(1, requestedInputTokens - estimatedSuffixTokens)
        return String(repeating: " token", count: fillerCount) + suffix
    }

    var validationIssue: String? {
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Choose a model."
        }
        guard concurrency == 1 else { return "Benchmark concurrency must be 1." }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Benchmark prompt cannot be empty."
        }
        if kind == .languageModel {
            guard !selectedInputTokenSizes.isEmpty else {
                return "Choose at least one LLM input size."
            }
            guard Set(inputTokenSizes ?? [512]).isSubset(of: BenchmarkLLMInputSize.allowedTokenCounts) else {
                return "LLM input size contains an unsupported value."
            }
        }
        if kind == .videoGeneration {
            guard width.isMultiple(of: 32), height.isMultiple(of: 32) else {
                return "MiniMax H3 width and height must be multiples of 32."
            }
            guard frames >= 5, frames <= 345, (frames - 5).isMultiple(of: 17) else {
                return "MiniMax H3 frames must use the 17k+5 ladder."
            }
        }
        return nil
    }

    /// All fields that define a comparable workload, intentionally excluding topology.
    var comparisonKey: String {
        switch kind {
        case .languageModel:
            return [
                kind.rawValue, modelID, modelRevision ?? "", quantization ?? "",
                promptProfile, prompt, String(maximumOutputTokens), String(concurrency),
                selectedInputTokenSizes.map(String.init).joined(separator: ","),
                String(format: "%.8f", temperature),
            ].joined(separator: "|")
        case .videoGeneration:
            return [
                kind.rawValue, modelID, modelRevision ?? "", quantization ?? "",
                promptProfile, prompt, "\(width)x\(height)", String(frames), String(steps),
                String(seed), String(fast),
            ].joined(separator: "|")
        }
    }
}

enum BenchmarkModelFilter {
    static func eligibleModels(
        in rows: [ModelLibraryRow],
        for configuration: BenchmarkConfiguration
    ) -> [ModelLibraryRow] {
        rows.filter { row in
            guard row.modality == configuration.kind.modality, row.standaloneLoadable else {
                return false
            }
            if configuration.target == .tensorParallel2 {
                return row.distributedLoadable && row.nodes.count >= 2
            }
            return row.nodes.count >= 1 || row.modality == .video
        }
    }
}

struct BenchmarkRunSample: Codable, Hashable, Identifiable {
    var id = UUID()
    var runIndex: Int
    var isWarmup: Bool
    var thermalState: BenchmarkThermalState
    var kind: BenchmarkKind
    var modelID: String
    var modelRevision: String? = nil
    var quantization: String? = nil
    var topology: String
    var nodes: [String]
    var rankOrder: [String]
    var requestedInputTokens: Int? = nil
    var promptTokens: Int? = nil
    var completionTokens: Int? = nil
    var ttftMilliseconds: Double? = nil
    var prefillTokensPerSecond: Double? = nil
    var decodeTokensPerSecond: Double? = nil
    var totalMilliseconds: Double? = nil
    var requestID: String? = nil
    var didReceiveDone: Bool? = nil
    var width: Int? = nil
    var height: Int? = nil
    var frames: Int? = nil
    var actualFPS: Int? = nil
    var steps: Int? = nil
    var seed: Int? = nil
    var fast: Bool? = nil
    var samplingMilliseconds: Double? = nil
    var framesPerSecond: Double? = nil
    var secondsPerFrame: Double? = nil
    var outputPath: String? = nil
    var status: BenchmarkSampleStatus
    var error: String? = nil
    var timestamp: Date

    var isMeasuredSuccess: Bool { !isWarmup && status == .succeeded }
}

struct BenchmarkMetricSummary: Codable, Hashable {
    var count: Int
    var mean: Double?
    var p50: Double?
    var p95: Double?

    static func calculate(_ values: [Double]) -> BenchmarkMetricSummary {
        guard !values.isEmpty else {
            return BenchmarkMetricSummary(count: 0, mean: nil, p50: nil, p95: nil)
        }
        let ordered = values.sorted()
        return BenchmarkMetricSummary(
            count: ordered.count,
            mean: ordered.reduce(0, +) / Double(ordered.count),
            p50: percentile(ordered, quantile: 0.5),
            p95: ordered.count >= 5 ? percentile(ordered, quantile: 0.95) : nil
        )
    }

    private static func percentile(_ ordered: [Double], quantile: Double) -> Double {
        let position = Double(ordered.count - 1) * quantile
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        guard lower != upper else { return ordered[lower] }
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - Double(lower))
    }
}

struct BenchmarkThermalSummary: Codable, Hashable {
    var thermalState: BenchmarkThermalState
    var requestedInputTokens: Int? = nil
    var sampleCount: Int
    var ttftMilliseconds: BenchmarkMetricSummary
    var prefillTokensPerSecond: BenchmarkMetricSummary
    var decodeTokensPerSecond: BenchmarkMetricSummary
    var totalMilliseconds: BenchmarkMetricSummary
    var samplingMilliseconds: BenchmarkMetricSummary
    var framesPerSecond: BenchmarkMetricSummary

    static func calculate(
        thermalState: BenchmarkThermalState,
        requestedInputTokens: Int? = nil,
        samples: [BenchmarkRunSample]
    ) -> BenchmarkThermalSummary {
        let eligible = samples.filter {
            $0.isMeasuredSuccess
                && $0.thermalState == thermalState
                && (requestedInputTokens == nil || $0.requestedInputTokens == requestedInputTokens)
        }
        return BenchmarkThermalSummary(
            thermalState: thermalState,
            requestedInputTokens: requestedInputTokens,
            sampleCount: eligible.count,
            ttftMilliseconds: .calculate(eligible.compactMap(\.ttftMilliseconds)),
            prefillTokensPerSecond: .calculate(eligible.compactMap(\.prefillTokensPerSecond)),
            decodeTokensPerSecond: .calculate(eligible.compactMap(\.decodeTokensPerSecond)),
            totalMilliseconds: .calculate(eligible.compactMap(\.totalMilliseconds)),
            samplingMilliseconds: .calculate(eligible.compactMap(\.samplingMilliseconds)),
            framesPerSecond: .calculate(eligible.compactMap(\.framesPerSecond))
        )
    }
}

struct BenchmarkResult: Codable, Hashable, Identifiable {
    var schemaVersion: Int = 1
    var id = UUID()
    var configuration: BenchmarkConfiguration
    var topology: String
    var nodes: [String]
    var rankOrder: [String]
    var runtimeInstanceID: String?
    var borrowedResidentModel: Bool
    var modelLoadMilliseconds: Double?
    var startedAt: Date
    var completedAt: Date
    var samples: [BenchmarkRunSample]
    var summaries: [BenchmarkThermalSummary]

    enum CodingKeys: String, CodingKey {
        case id, configuration, topology, nodes, samples, summaries
        case schemaVersion = "schema_version"
        case rankOrder = "rank_order"
        case runtimeInstanceID = "runtime_instance_id"
        case borrowedResidentModel = "borrowed_resident_model"
        case modelLoadMilliseconds = "model_load_milliseconds"
        case startedAt = "started_at"
        case completedAt = "completed_at"
    }

    mutating func refreshSummaries() {
        let requestedSizes = Set(samples.filter(\.isMeasuredSuccess).compactMap(\.requestedInputTokens)).sorted()
        if configuration.kind == .languageModel, !requestedSizes.isEmpty {
            summaries = requestedSizes.flatMap { inputSize in
                BenchmarkThermalState.allCases.map {
                    BenchmarkThermalSummary.calculate(
                        thermalState: $0,
                        requestedInputTokens: inputSize,
                        samples: samples
                    )
                }
            }.filter { $0.sampleCount > 0 }
        } else {
            summaries = BenchmarkThermalState.allCases.map {
                BenchmarkThermalSummary.calculate(thermalState: $0, samples: samples)
            }.filter { $0.sampleCount > 0 }
        }
    }

    func isComparable(to other: BenchmarkResult) -> Bool {
        guard configuration.comparisonKey == other.configuration.comparisonKey else { return false }
        guard topology != other.topology else { return false }
        if configuration.kind == .languageModel {
            let left = Set(samples.filter(\.isMeasuredSuccess).compactMap(\.promptTokens))
            let right = Set(other.samples.filter(\.isMeasuredSuccess).compactMap(\.promptTokens))
            return !left.isEmpty && left == right
        }
        return true
    }

    /// Returns TP2 speedup as single-Mac mean duration divided by TP2 mean duration.
    func speedup(comparedWith other: BenchmarkResult) -> Double? {
        guard isComparable(to: other) else { return nil }
        let single = topology == "tp2" ? other : self
        let tp2 = topology == "tp2" ? self : other
        guard single.topology != "tp2", tp2.topology == "tp2" else { return nil }
        let preferredThermal: BenchmarkThermalState = .warm
        let singleSamples = single.samples.filter {
            $0.isMeasuredSuccess && $0.thermalState == preferredThermal
        }
        let tp2Samples = tp2.samples.filter {
            $0.isMeasuredSuccess && $0.thermalState == preferredThermal
        }
        let singleFallback = singleSamples.isEmpty
            ? single.samples.filter { $0.isMeasuredSuccess && $0.thermalState == .cold }
            : singleSamples
        let tp2Fallback = tp2Samples.isEmpty
            ? tp2.samples.filter { $0.isMeasuredSuccess && $0.thermalState == .cold }
            : tp2Samples
        let singleMean = BenchmarkMetricSummary.calculate(singleFallback.compactMap(\.totalMilliseconds)).mean
        let tp2Mean = BenchmarkMetricSummary.calculate(tp2Fallback.compactMap(\.totalMilliseconds)).mean
        guard let singleMean, let tp2Mean, tp2Mean > 0 else { return nil }
        return singleMean / tp2Mean
    }
}

struct BenchmarkResultStore {
    let rootDirectory: URL
    let retentionLimit: Int

    init(rootDirectory: URL? = nil, retentionLimit: Int = 20) {
        self.rootDirectory = rootDirectory ?? Self.defaultRootDirectory()
        self.retentionLimit = max(1, retentionLimit)
    }

    static func defaultRootDirectory() -> URL {
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        return applicationSupport
            .appendingPathComponent("Tokenity", isDirectory: true)
            .appendingPathComponent("Benchmarks", isDirectory: true)
    }

    func load() -> [BenchmarkResult] {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: rootDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }
        return urls
            .filter { $0.pathExtension.lowercased() == "json" }
            .compactMap { url -> BenchmarkResult? in
                guard let data = try? Data(contentsOf: url),
                      let result = try? Self.decoder.decode(BenchmarkResult.self, from: data),
                      result.schemaVersion == 1 else { return nil }
                return result
            }
            .sorted { $0.completedAt > $1.completedAt }
            .prefix(retentionLimit)
            .map { $0 }
    }

    @discardableResult
    func save(_ result: BenchmarkResult) throws -> URL {
        try FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        let url = rootDirectory.appendingPathComponent("\(result.id.uuidString).json")
        try Self.encoder.encode(result).write(to: url, options: .atomic)
        try prune()
        return url
    }

    func exportJSON(_ result: BenchmarkResult, to url: URL) throws {
        try Self.encoder.encode(result).write(to: url, options: .atomic)
    }

    func exportCSV(_ result: BenchmarkResult, to url: URL) throws {
        let header = [
            "schema_version", "result_id", "run_index", "warmup", "thermal", "kind",
            "model_id", "model_revision", "quantization", "topology", "nodes", "rank_order",
            "requested_input_tokens", "prompt_tokens", "completion_tokens", "ttft_ms", "prefill_tok_s", "decode_tok_s",
            "total_ms", "request_id", "done", "width", "height", "frames", "actual_fps",
            "steps", "seed", "fast", "sampling_ms", "frames_s", "seconds_frame",
            "output_path", "status", "error", "timestamp",
        ]
        var rows = [header.map(Self.csvEscape).joined(separator: ",")]
        let formatter = ISO8601DateFormatter()
        for sample in result.samples {
            var fields: [String] = [
                String(result.schemaVersion), result.id.uuidString, String(sample.runIndex), String(sample.isWarmup),
                sample.thermalState.rawValue, sample.kind.rawValue, sample.modelID,
                sample.modelRevision ?? "", sample.quantization ?? "", sample.topology,
                sample.nodes.joined(separator: ";"), sample.rankOrder.joined(separator: ";"),
            ]
            fields.append(sample.requestedInputTokens.map { String($0) } ?? "")
            fields.append(sample.promptTokens.map { String($0) } ?? "")
            fields.append(sample.completionTokens.map { String($0) } ?? "")
            fields.append(sample.ttftMilliseconds.map { String($0) } ?? "")
            fields.append(sample.prefillTokensPerSecond.map { String($0) } ?? "")
            fields.append(sample.decodeTokensPerSecond.map { String($0) } ?? "")
            fields.append(sample.totalMilliseconds.map { String($0) } ?? "")
            fields.append(sample.requestID ?? "")
            fields.append(sample.didReceiveDone.map { String($0) } ?? "")
            fields.append(sample.width.map { String($0) } ?? "")
            fields.append(sample.height.map { String($0) } ?? "")
            fields.append(sample.frames.map { String($0) } ?? "")
            fields.append(sample.actualFPS.map { String($0) } ?? "")
            fields.append(sample.steps.map { String($0) } ?? "")
            fields.append(sample.seed.map { String($0) } ?? "")
            fields.append(sample.fast.map { String($0) } ?? "")
            fields.append(sample.samplingMilliseconds.map { String($0) } ?? "")
            fields.append(sample.framesPerSecond.map { String($0) } ?? "")
            fields.append(sample.secondsPerFrame.map { String($0) } ?? "")
            fields.append(sample.outputPath ?? "")
            fields.append(sample.status.rawValue)
            fields.append(sample.error ?? "")
            fields.append(formatter.string(from: sample.timestamp))
            rows.append(fields.map(Self.csvEscape).joined(separator: ","))
        }
        try (rows.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private func prune() throws {
        let kept = load()
        let keptIDs = Set(kept.prefix(retentionLimit).map(\.id))
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: rootDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return }
        for url in urls where url.pathExtension.lowercased() == "json" {
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent),
                  !keptIDs.contains(id) else { continue }
            try FileManager.default.removeItem(at: url)
        }
    }

    private static func csvEscape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else { return value }
        return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
