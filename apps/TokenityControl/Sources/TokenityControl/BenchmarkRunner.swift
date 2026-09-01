import Foundation

struct BenchmarkPreparedTarget: Hashable {
    var kind: BenchmarkKind
    var modelID: String
    var topology: String
    var nodeIDs: [String]
    var nodes: [String]
    var rankOrder: [String]
    var modelRevision: String?
    var quantization: String?
    var instanceID: String?
    var borrowedResidentModel: Bool
    var needsLoading: Bool
    var serviceBaseURL: String?
    var loadMilliseconds: Double?
    var previousBackendMode: BackendMode
    var previousCoordinatorID: String
    var previousSelectedNodeIDs: Set<String>
    var previousH3WorkerAgentURL: String
}

struct BenchmarkStreamResponse: Hashable {
    var requestID: String?
    var modelRevision: String?
    var instanceID: String?
}

struct BenchmarkTimedStreamEvent: Hashable {
    enum Payload: Hashable {
        case response(BenchmarkStreamResponse)
        case line(String)
    }

    var timestamp: Double
    var payload: Payload
}

struct BenchmarkRequestTimeline: Hashable {
    var requestID: String
    var prefillStart: Double?
    var prefillEnd: Double?

    func prefillTokensPerSecond(promptTokens: Int, expectedRequestID: String) -> Double? {
        guard requestID == expectedRequestID,
              let prefillStart, let prefillEnd,
              prefillEnd > prefillStart else { return nil }
        return Double(promptTokens) / (prefillEnd - prefillStart)
    }
}

enum BenchmarkRunnerError: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case conflictingWorkload(String)
    case topologyUnavailable(String)
    case runtimeUnavailable(String)
    case malformedSSE(String)
    case missingDone
    case emptyOutput
    case missingUsage

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail), .conflictingWorkload(let detail),
             .topologyUnavailable(let detail), .runtimeUnavailable(let detail):
            return detail
        case .malformedSSE(let detail): return "Malformed SSE: \(detail)"
        case .missingDone: return "The language-model stream ended without [DONE]."
        case .emptyOutput: return "The stream completed without a non-empty content or reasoning token."
        case .missingUsage: return "The stream completed without prompt and completion token usage."
        }
    }
}

@MainActor
protocol BenchmarkTransport: AnyObject {
    func preflight(_ configuration: BenchmarkConfiguration) async throws -> BenchmarkPreparedTarget
    func prepare(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkPreparedTarget
    func languageStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error>
    func requestTimeline(
        requestID: String,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkRequestTimeline?
    func videoStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error>
    func saveVideoPreview(
        completionLine: String,
        configuration: BenchmarkConfiguration
    ) async throws -> URL?
    func cleanup(_ target: BenchmarkPreparedTarget) async throws
}

struct BenchmarkLLMSSEMeasurement {
    let startedAt: Double
    private(set) var requestID: String?
    private(set) var modelRevision: String?
    private(set) var instanceID: String?
    private(set) var promptTokens: Int?
    private(set) var completionTokens: Int?
    private(set) var firstTokenAt: Double?
    private(set) var completedAt: Double?
    private(set) var didReceiveDone = false
    private(set) var outputCharacterCount = 0

    mutating func consume(_ event: BenchmarkTimedStreamEvent) throws {
        switch event.payload {
        case .response(let response):
            requestID = response.requestID
            modelRevision = response.modelRevision
            instanceID = response.instanceID
        case .line(let line):
            guard line.hasPrefix("data:") else { return }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            guard !payload.isEmpty else { return }
            if payload == "[DONE]" {
                guard !didReceiveDone else {
                    throw BenchmarkRunnerError.malformedSSE("duplicate [DONE]")
                }
                didReceiveDone = true
                completedAt = event.timestamp
                return
            }
            guard !didReceiveDone else {
                throw BenchmarkRunnerError.malformedSSE("data received after [DONE]")
            }
            guard let data = payload.data(using: .utf8) else {
                throw BenchmarkRunnerError.malformedSSE("invalid UTF-8 payload")
            }
            let chunk: OpenAIChatChunk
            do {
                chunk = try JSONDecoder().decode(OpenAIChatChunk.self, from: data)
            } catch {
                throw BenchmarkRunnerError.malformedSSE(error.localizedDescription)
            }
            if let value = chunk.usage?.promptTokens { promptTokens = value }
            if let value = chunk.usage?.completionTokens { completionTokens = value }
            guard let delta = chunk.choices.first?.delta else { return }
            let reasoning = delta.reasoningContent ?? delta.reasoning ?? ""
            let content = delta.content ?? ""
            if !reasoning.isEmpty || !content.isEmpty {
                firstTokenAt = firstTokenAt ?? event.timestamp
                outputCharacterCount += reasoning.count + content.count
            }
        }
    }

    func validate() throws {
        guard didReceiveDone else { throw BenchmarkRunnerError.missingDone }
        guard firstTokenAt != nil, outputCharacterCount > 0 else {
            throw BenchmarkRunnerError.emptyOutput
        }
        guard promptTokens != nil, completionTokens.map({ $0 > 0 }) == true else {
            throw BenchmarkRunnerError.missingUsage
        }
    }

    var ttftMilliseconds: Double? {
        firstTokenAt.map { max(0, ($0 - startedAt) * 1_000) }
    }

    var totalMilliseconds: Double? {
        completedAt.map { max(0, ($0 - startedAt) * 1_000) }
    }

    var decodeTokensPerSecond: Double? {
        guard let completionTokens, let firstTokenAt, let completedAt,
              completedAt > firstTokenAt else { return nil }
        return Double(completionTokens) / (completedAt - firstTokenAt)
    }
}

private struct BenchmarkVideoEnvelope: Decodable {
    var type: String
    var stage: String?
    var step: Int?
    var total: Int?
    var elapsedMilliseconds: Double?
    var frames: Int?
    var height: Int?
    var width: Int?
    var fps: Int?

    enum CodingKeys: String, CodingKey {
        case type, stage, step, total, frames, height, width, fps
        case elapsedMilliseconds = "elapsed_ms"
    }
}

private struct BenchmarkVideoMeasurement {
    let startedAt: Double
    private(set) var firstSamplingAt: Double?
    private(set) var lastSamplingElapsedMilliseconds: Double?
    private(set) var completedAt: Double?
    private(set) var frames: Int?
    private(set) var width: Int?
    private(set) var height: Int?
    private(set) var fps: Int?
    private(set) var completionLine: String?
    private(set) var modelRevision: String?
    private(set) var instanceID: String?

    mutating func consume(_ event: BenchmarkTimedStreamEvent) throws {
        if case .response(let response) = event.payload {
            modelRevision = response.modelRevision
            instanceID = response.instanceID
            return
        }
        guard case .line(let line) = event.payload, line.hasPrefix("data:") else { return }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard !payload.isEmpty, payload != "[DONE]", let data = payload.data(using: .utf8) else { return }
        let envelope: BenchmarkVideoEnvelope
        do {
            envelope = try JSONDecoder().decode(BenchmarkVideoEnvelope.self, from: data)
        } catch {
            throw BenchmarkRunnerError.malformedSSE(error.localizedDescription)
        }
        switch envelope.type {
        case "progress":
            let stage = envelope.stage?.lowercased() ?? ""
            if stage.contains("generating") || stage.contains("sampling") || stage.contains("dit") {
                firstSamplingAt = firstSamplingAt ?? event.timestamp
                if let elapsed = envelope.elapsedMilliseconds {
                    lastSamplingElapsedMilliseconds = elapsed
                }
            }
        case "complete":
            guard let frames = envelope.frames, let width = envelope.width,
                  let height = envelope.height, let fps = envelope.fps,
                  frames > 0, width > 0, height > 0, fps > 0 else {
                throw BenchmarkRunnerError.malformedSSE("incomplete video completion metadata")
            }
            self.frames = frames
            self.width = width
            self.height = height
            self.fps = fps
            completedAt = event.timestamp
            completionLine = line
        case "error":
            throw BenchmarkRunnerError.runtimeUnavailable("MiniMax H3 reported a structured error event.")
        default:
            break
        }
    }

    func validate() throws {
        guard completedAt != nil, frames != nil else {
            throw BenchmarkRunnerError.malformedSSE("video stream ended without a complete event")
        }
    }

    var totalMilliseconds: Double? {
        completedAt.map { max(0, ($0 - startedAt) * 1_000) }
    }

    var samplingMilliseconds: Double? {
        if let elapsed = lastSamplingElapsedMilliseconds, elapsed >= 0 { return elapsed }
        guard let firstSamplingAt, let completedAt, completedAt >= firstSamplingAt else { return nil }
        return (completedAt - firstSamplingAt) * 1_000
    }
}

@MainActor
final class BenchmarkRunner: ObservableObject {
    @Published var state: BenchmarkPhase = .idle
    @Published var progress: Double = 0
    @Published var currentStage = "Ready"
    @Published var completedRuns = 0
    @Published var totalRuns = 0
    @Published var elapsedSeconds: Double = 0
    @Published var rankStatuses: [String: String] = [:]
    @Published var samples: [BenchmarkRunSample] = []
    @Published var currentResult: BenchmarkResult?
    @Published var history: [BenchmarkResult]
    @Published var errorMessage: String?

    var isActive: Bool { state.isActive || task != nil }

    private let transport: BenchmarkTransport
    private let resultStore: BenchmarkResultStore
    private let clock: () -> Double
    private var task: Task<Void, Never>?
    private var ticker: Task<Void, Never>?
    private var runID: UUID?

    init(
        transport: BenchmarkTransport,
        resultStore: BenchmarkResultStore = BenchmarkResultStore(),
        clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }
    ) {
        self.transport = transport
        self.resultStore = resultStore
        self.clock = clock
        history = resultStore.load()
    }

    func start(_ configuration: BenchmarkConfiguration) {
        guard task == nil, !state.isActive else { return }
        if let issue = configuration.validationIssue {
            state = .failed
            currentStage = "Configuration blocked"
            errorMessage = issue
            return
        }
        let id = UUID()
        runID = id
        state = .preflight
        progress = 0
        currentStage = "Checking topology, disk space, and workload conflicts"
        completedRuns = 0
        totalRuns = configuration.totalRuns
        elapsedSeconds = 0
        rankStatuses = [:]
        samples = []
        currentResult = nil
        errorMessage = nil
        let start = clock()
        ticker?.cancel()
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, self.runID == id else { return }
                self.elapsedSeconds = max(0, self.clock() - start)
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        task = Task { [weak self] in
            await self?.execute(configuration, runID: id, startedAt: Date(), startedClock: start)
        }
    }

    func cancel() {
        guard state.isActive, state != .cancelling else { return }
        state = .cancelling
        currentStage = "Cancelling and draining the active request"
        task?.cancel()
    }

    func cancelAndWait() async {
        cancel()
        if let task { await task.value }
    }

    func reset() {
        guard !state.isActive else { return }
        state = .idle
        progress = 0
        currentStage = "Ready"
        completedRuns = 0
        totalRuns = 0
        elapsedSeconds = 0
        rankStatuses = [:]
        samples = []
        currentResult = nil
        errorMessage = nil
    }

    func matchingSpeedup(for result: BenchmarkResult) -> Double? {
        history.lazy.compactMap { candidate -> Double? in
            guard candidate.id != result.id else { return nil }
            return result.speedup(comparedWith: candidate)
        }.first
    }

    func exportJSON(_ result: BenchmarkResult, to url: URL) throws {
        try resultStore.exportJSON(result, to: url)
    }

    func exportCSV(_ result: BenchmarkResult, to url: URL) throws {
        try resultStore.exportCSV(result, to: url)
    }

    private func execute(
        _ originalConfiguration: BenchmarkConfiguration,
        runID: UUID,
        startedAt: Date,
        startedClock: Double
    ) async {
        var prepared: BenchmarkPreparedTarget?
        var result: BenchmarkResult?
        do {
            var target = try await transport.preflight(originalConfiguration)
            try Task.checkCancellation()
            if target.needsLoading {
                state = .loading
                currentStage = "Loading the selected model outside measured time"
            }
            target = try await transport.prepare(originalConfiguration, target: target)
            prepared = target
            try Task.checkCancellation()

            var configuration = originalConfiguration
            configuration.modelRevision = target.modelRevision
            configuration.quantization = target.quantization
            rankStatuses = Dictionary(uniqueKeysWithValues: target.rankOrder.enumerated().map {
                ("Rank \($0.offset)", "Ready · \($0.element)")
            })
            result = BenchmarkResult(
                configuration: configuration,
                topology: target.topology,
                nodes: target.nodes,
                rankOrder: target.rankOrder,
                runtimeInstanceID: target.instanceID,
                borrowedResidentModel: target.borrowedResidentModel,
                modelLoadMilliseconds: target.loadMilliseconds,
                startedAt: startedAt,
                completedAt: startedAt,
                samples: [],
                summaries: []
            )

            let inputWorkloads: [Int?] = configuration.kind == .languageModel
                ? configuration.selectedInputTokenSizes.map(Optional.some)
                : [nil]
            var globalRunIndex = 0
            for requestedInputTokens in inputWorkloads {
                var workloadConfiguration = configuration
                if let requestedInputTokens {
                    workloadConfiguration.prompt = configuration.languagePrompt(
                        requestedInputTokens: requestedInputTokens
                    )
                }
                for runOffset in 0..<configuration.runsPerInputSize {
                    try Task.checkCancellation()
                    globalRunIndex += 1
                    let isWarmup = runOffset < configuration.warmupRuns
                    let measuredIndex = runOffset - configuration.warmupRuns + 1
                    let thermal: BenchmarkThermalState
                    if configuration.kind == .languageModel {
                        thermal = isWarmup ? .cold : .warm
                    } else {
                        thermal = runOffset == 0 ? .cold : .warm
                    }
                    let inputLabel = requestedInputTokens.map {
                        "\(BenchmarkLLMInputSize.title(for: $0)) input · "
                    } ?? ""
                    state = isWarmup ? .warmup : .running
                    currentStage = isWarmup
                        ? "\(inputLabel)warmup \(runOffset + 1) of \(configuration.warmupRuns) · excluded"
                        : "\(inputLabel)run \(measuredIndex) of \(configuration.measuredRuns) · \(thermal.rawValue)"
                    rankStatuses = Dictionary(uniqueKeysWithValues: target.rankOrder.enumerated().map {
                        ("Rank \($0.offset)", "Running · \($0.element)")
                    })
                    let sample: BenchmarkRunSample
                    do {
                        switch configuration.kind {
                        case .languageModel:
                            sample = try await languageSample(
                                workloadConfiguration,
                                target: target,
                                runIndex: globalRunIndex,
                                isWarmup: isWarmup,
                                thermal: thermal,
                                requestedInputTokens: requestedInputTokens
                            )
                        case .videoGeneration:
                            sample = try await videoSample(
                                workloadConfiguration,
                                target: target,
                                runIndex: globalRunIndex,
                                thermal: thermal,
                                savesPreview: configuration.saveVideoPreview
                                    && globalRunIndex == configuration.totalRuns
                            )
                        }
                    } catch is CancellationError {
                        let cancelled = failureSample(
                            workloadConfiguration,
                            target: target,
                            runIndex: globalRunIndex,
                            isWarmup: isWarmup,
                            thermal: thermal,
                            requestedInputTokens: requestedInputTokens,
                            status: .cancelled,
                            error: "Cancelled"
                        )
                        samples.append(cancelled)
                        result?.samples.append(cancelled)
                        throw CancellationError()
                    } catch {
                        let failed = failureSample(
                            workloadConfiguration,
                            target: target,
                            runIndex: globalRunIndex,
                            isWarmup: isWarmup,
                            thermal: thermal,
                            requestedInputTokens: requestedInputTokens,
                            status: .failed,
                            error: error.localizedDescription
                        )
                        samples.append(failed)
                        result?.samples.append(failed)
                        throw error
                    }
                    samples.append(sample)
                    result?.samples.append(sample)
                    if result?.configuration.modelRevision == nil, let revision = sample.modelRevision {
                        result?.configuration.modelRevision = revision
                    }
                    completedRuns += 1
                    progress = Double(completedRuns) / Double(max(configuration.totalRuns, 1))
                    elapsedSeconds = max(0, clock() - startedClock)
                }
            }

            guard var completed = result else { throw BenchmarkRunnerError.runtimeUnavailable("No result was produced.") }
            completed.completedAt = Date()
            completed.refreshSummaries()
            try resultStore.save(completed)
            currentResult = completed
            history.removeAll { $0.id == completed.id }
            history.insert(completed, at: 0)
            history = Array(history.prefix(20))
            progress = 1
            state = .completed
            currentStage = "Benchmark complete"
            rankStatuses = Dictionary(uniqueKeysWithValues: target.rankOrder.enumerated().map {
                ("Rank \($0.offset)", "Ready · \($0.element)")
            })
        } catch is CancellationError {
            if var partial = result {
                partial.completedAt = Date()
                partial.samples = samples
                partial.refreshSummaries()
                currentResult = partial
            }
            state = .completed
            currentStage = "Cancelled safely"
            errorMessage = nil
        } catch {
            if var partial = result {
                partial.completedAt = Date()
                partial.samples = samples
                partial.refreshSummaries()
                currentResult = partial
                _ = try? resultStore.save(partial)
                history.insert(partial, at: 0)
                history = Array(history.prefix(20))
            }
            state = .failed
            currentStage = "Benchmark failed"
            errorMessage = error.localizedDescription
        }

        if let prepared {
            do {
                try await transport.cleanup(prepared)
            } catch {
                state = .failed
                currentStage = "Cleanup failed"
                errorMessage = "Benchmark finished, but exact-instance cleanup failed: \(error.localizedDescription)"
            }
        }
        guard self.runID == runID else { return }
        ticker?.cancel()
        ticker = nil
        elapsedSeconds = max(0, clock() - startedClock)
        task = nil
        self.runID = nil
    }

    private func languageSample(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget,
        runIndex: Int,
        isWarmup: Bool,
        thermal: BenchmarkThermalState,
        requestedInputTokens: Int?
    ) async throws -> BenchmarkRunSample {
        let started = clock()
        var measurement = BenchmarkLLMSSEMeasurement(startedAt: started)
        for try await event in transport.languageStream(configuration, target: target) {
            try Task.checkCancellation()
            try measurement.consume(event)
        }
        try Task.checkCancellation()
        try measurement.validate()
        let timeline: BenchmarkRequestTimeline?
        if let requestID = measurement.requestID {
            timeline = try? await transport.requestTimeline(requestID: requestID, target: target)
        } else {
            timeline = nil
        }
        let prefill = measurement.requestID.flatMap { requestID in
            measurement.promptTokens.flatMap { promptTokens in
                timeline?.prefillTokensPerSecond(
                    promptTokens: promptTokens,
                    expectedRequestID: requestID
                )
            }
        }
        return BenchmarkRunSample(
            runIndex: runIndex,
            isWarmup: isWarmup,
            thermalState: thermal,
            kind: .languageModel,
            modelID: configuration.modelID,
            modelRevision: measurement.modelRevision ?? target.modelRevision,
            quantization: target.quantization,
            topology: target.topology,
            nodes: target.nodes,
            rankOrder: target.rankOrder,
            requestedInputTokens: requestedInputTokens,
            promptTokens: measurement.promptTokens,
            completionTokens: measurement.completionTokens,
            ttftMilliseconds: measurement.ttftMilliseconds,
            prefillTokensPerSecond: prefill,
            decodeTokensPerSecond: measurement.decodeTokensPerSecond,
            totalMilliseconds: measurement.totalMilliseconds,
            requestID: measurement.requestID,
            didReceiveDone: measurement.didReceiveDone,
            status: .succeeded,
            timestamp: Date()
        )
    }

    private func videoSample(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget,
        runIndex: Int,
        thermal: BenchmarkThermalState,
        savesPreview: Bool
    ) async throws -> BenchmarkRunSample {
        let started = clock()
        var measurement = BenchmarkVideoMeasurement(startedAt: started)
        for try await event in transport.videoStream(configuration, target: target) {
            try Task.checkCancellation()
            try measurement.consume(event)
        }
        try Task.checkCancellation()
        try measurement.validate()
        var outputURL: URL?
        if savesPreview, let line = measurement.completionLine {
            currentStage = "Saving one optional preview"
            outputURL = try await transport.saveVideoPreview(
                completionLine: line,
                configuration: configuration
            )
        }
        let totalSeconds = measurement.totalMilliseconds.map { $0 / 1_000 }
        let actualFrames = measurement.frames
        return BenchmarkRunSample(
            runIndex: runIndex,
            isWarmup: false,
            thermalState: thermal,
            kind: .videoGeneration,
            modelID: configuration.modelID,
            modelRevision: measurement.modelRevision ?? target.modelRevision,
            quantization: target.quantization,
            topology: target.topology,
            nodes: target.nodes,
            rankOrder: target.rankOrder,
            totalMilliseconds: measurement.totalMilliseconds,
            width: measurement.width,
            height: measurement.height,
            frames: actualFrames,
            actualFPS: measurement.fps,
            steps: configuration.steps,
            seed: configuration.seed,
            fast: configuration.fast,
            samplingMilliseconds: measurement.samplingMilliseconds,
            framesPerSecond: totalSeconds.flatMap { duration in
                guard duration > 0, let actualFrames else { return nil }
                return Double(actualFrames) / duration
            },
            secondsPerFrame: totalSeconds.flatMap { duration in
                guard let actualFrames, actualFrames > 0 else { return nil }
                return duration / Double(actualFrames)
            },
            outputPath: outputURL?.path,
            status: .succeeded,
            timestamp: Date()
        )
    }

    private func failureSample(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget,
        runIndex: Int,
        isWarmup: Bool,
        thermal: BenchmarkThermalState,
        requestedInputTokens: Int? = nil,
        status: BenchmarkSampleStatus,
        error: String
    ) -> BenchmarkRunSample {
        BenchmarkRunSample(
            runIndex: runIndex,
            isWarmup: isWarmup,
            thermalState: thermal,
            kind: configuration.kind,
            modelID: configuration.modelID,
            modelRevision: target.modelRevision,
            quantization: target.quantization,
            topology: target.topology,
            nodes: target.nodes,
            rankOrder: target.rankOrder,
            requestedInputTokens: requestedInputTokens,
            status: status,
            error: error,
            timestamp: Date()
        )
    }
}

@MainActor
final class TokenityStoreBenchmarkTransport: BenchmarkTransport {
    private unowned let store: TokenityStore

    init(store: TokenityStore) {
        self.store = store
    }

    func preflight(_ configuration: BenchmarkConfiguration) async throws -> BenchmarkPreparedTarget {
        try await store.benchmarkPreflight(configuration)
    }

    func prepare(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkPreparedTarget {
        try await store.benchmarkPrepare(configuration, target: target)
    }

    func languageStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error> {
        store.benchmarkLanguageStream(configuration, target: target)
    }

    func requestTimeline(
        requestID: String,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkRequestTimeline? {
        try await store.benchmarkRequestTimeline(requestID: requestID, target: target)
    }

    func videoStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error> {
        store.benchmarkVideoStream(configuration, target: target)
    }

    func saveVideoPreview(
        completionLine: String,
        configuration: BenchmarkConfiguration
    ) async throws -> URL? {
        try await store.benchmarkSaveVideoPreview(
            completionLine: completionLine,
            configuration: configuration
        )
    }

    func cleanup(_ target: BenchmarkPreparedTarget) async throws {
        try await store.benchmarkCleanup(target)
    }
}
