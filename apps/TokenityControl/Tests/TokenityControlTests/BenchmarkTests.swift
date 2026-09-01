import AppKit
import SwiftUI
import XCTest
@testable import TokenityControl

@MainActor
final class BenchmarkTests: XCTestCase {
    func testBenchmarkNavigationMetadataAndOperationsOrder() {
        let operations = AppSection.allCases.filter { $0.group == "Operations" }
        XCTAssertEqual(operations, [.api, .benchmark, .logs, .settings])
        XCTAssertEqual(AppSection.benchmark.title, "Benchmark")
        XCTAssertEqual(AppSection.benchmark.group, "Operations")
        XCTAssertEqual(AppSection.benchmark.symbol, "gauge.with.needle")
    }

    func testModelFilterSeparatesLanguageAndVideoAndHonorsTP2() {
        let store = TokenityStore(userDefaults: isolatedDefaults())
        var language = BenchmarkConfiguration.language()
        let languageRows = BenchmarkModelFilter.eligibleModels(
            in: store.modelLibraryRows,
            for: language
        )
        XCTAssertFalse(languageRows.isEmpty)
        XCTAssertTrue(languageRows.allSatisfy { $0.modality == .language })

        let videoRows = BenchmarkModelFilter.eligibleModels(
            in: store.modelLibraryRows,
            for: .video()
        )
        XCTAssertEqual(videoRows.map(\.id), ["MiniMax-H3"])
        XCTAssertTrue(videoRows.allSatisfy { $0.modality == .video })

        language.target = .tensorParallel2
        XCTAssertTrue(
            BenchmarkModelFilter.eligibleModels(in: store.modelLibraryRows, for: language)
                .allSatisfy { $0.distributedLoadable && $0.nodes.count >= 2 }
        )
    }

    func testH3ResolutionAndDurationPresetsAreLegal() {
        XCTAssertEqual(
            BenchmarkResolutionPreset.h3Presets.map(\.id),
            ["256x256", "512x256", "512x288", "512x512"]
        )
        XCTAssertTrue(BenchmarkResolutionPreset.h3Presets.allSatisfy(\.isH3Valid))
        XCTAssertEqual(BenchmarkDurationPreset.h3Presets.map(\.frames), [22, 124, 243, 345])
        XCTAssertTrue(BenchmarkDurationPreset.h3Presets.allSatisfy(\.isH3Valid))
        XCTAssertEqual(BenchmarkDurationPreset.h3Presets[1].seconds, 124.0 / 24.0, accuracy: 0.0001)
    }

    func testLLMInputSizePresetsSelectionValidationAndPromptGeneration() {
        XCTAssertEqual(
            BenchmarkLLMInputSize.allCases.map(\.tokenCount),
            [512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072]
        )
        XCTAssertEqual(
            BenchmarkLLMInputSize.allCases.map(\.title),
            ["512", "1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"]
        )

        var configuration = BenchmarkConfiguration.language(modelID: "Fixture-LLM")
        XCTAssertEqual(configuration.selectedInputTokenSizes, [512])
        XCTAssertEqual(configuration.totalRuns, 6)
        configuration.inputTokenSizes = [131_072, 512, 2_048]
        XCTAssertEqual(configuration.selectedInputTokenSizes, [512, 2_048, 131_072])
        XCTAssertEqual(configuration.totalRuns, 18)
        XCTAssertNil(configuration.validationIssue)

        let shortPrompt = configuration.languagePrompt(requestedInputTokens: 512)
        let longPrompt = configuration.languagePrompt(requestedInputTokens: 2_048)
        XCTAssertGreaterThan(longPrompt.utf8.count, shortPrompt.utf8.count)
        XCTAssertTrue(shortPrompt.hasSuffix(BenchmarkConfiguration.languagePrompt))

        configuration.inputTokenSizes = []
        XCTAssertEqual(configuration.validationIssue, "Choose at least one LLM input size.")
        configuration.inputTokenSizes = [777]
        XCTAssertEqual(configuration.validationIssue, "Choose at least one LLM input size.")
    }

    func testRunnerExecutesAndSummarizesEverySelectedLLMInputSize() async throws {
        let root = temporaryDirectory("input-sweep")
        defer { try? FileManager.default.removeItem(at: root) }
        let transport = BenchmarkMockTransport(target: preparedTarget(borrowed: true))
        let runner = BenchmarkRunner(
            transport: transport,
            resultStore: BenchmarkResultStore(rootDirectory: root),
            clock: { 100 }
        )
        var configuration = BenchmarkConfiguration.language(modelID: "Fixture-LLM")
        configuration.profile = .custom
        configuration.customRuns = 1
        configuration.inputTokenSizes = [512, 2_048]

        runner.start(configuration)
        try await waitUntil { !runner.isActive }

        XCTAssertEqual(runner.state, .completed)
        XCTAssertEqual(runner.totalRuns, 4)
        XCTAssertEqual(transport.languageConfigurations.count, 4)
        XCTAssertEqual(
            runner.currentResult?.samples.map(\.requestedInputTokens),
            [512, 512, 2_048, 2_048]
        )
        XCTAssertLessThan(
            transport.languageConfigurations[0].prompt.utf8.count,
            transport.languageConfigurations[2].prompt.utf8.count
        )
        XCTAssertEqual(
            runner.currentResult?.summaries.compactMap(\.requestedInputTokens),
            [512, 2_048]
        )
        XCTAssertEqual(runner.currentResult?.summaries.map(\.sampleCount), [1, 1])
    }

    func testTTFTUsesFirstReasoningOrContentTokenAndDecodeEndsAtDone() throws {
        var measurement = BenchmarkLLMSSEMeasurement(startedAt: 10)
        try measurement.consume(event(10.01, .response(.init(
            requestID: "request-1",
            modelRevision: "rev-1",
            instanceID: "instance-1"
        ))))
        try measurement.consume(event(
            10.20,
            .line(#"data: {"choices":[{"delta":{"reasoning_content":"Thinking"},"finish_reason":null}]}"#)
        ))
        try measurement.consume(event(
            10.40,
            .line(#"data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":null}]}"#)
        ))
        try measurement.consume(event(
            10.70,
            .line(#"data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}"#)
        ))
        try measurement.consume(event(10.80, .line("data: [DONE]")))
        try measurement.validate()

        XCTAssertEqual(try XCTUnwrap(measurement.ttftMilliseconds), 200, accuracy: 0.001)
        XCTAssertEqual(measurement.promptTokens, 20)
        XCTAssertEqual(measurement.completionTokens, 5)
        XCTAssertEqual(try XCTUnwrap(measurement.decodeTokensPerSecond), 5.0 / 0.6, accuracy: 0.001)
        XCTAssertTrue(measurement.didReceiveDone)
    }

    func testPrefillTimelineRequiresMatchingRequestAndNeverFallsBackToTTFT() throws {
        let timeline = BenchmarkRequestTimeline(
            requestID: "request-1",
            prefillStart: 100,
            prefillEnd: 100.25
        )
        XCTAssertEqual(
            try XCTUnwrap(timeline.prefillTokensPerSecond(promptTokens: 50, expectedRequestID: "request-1")),
            200,
            accuracy: 0.001
        )
        XCTAssertNil(timeline.prefillTokensPerSecond(promptTokens: 50, expectedRequestID: "other"))
        XCTAssertNil(
            BenchmarkRequestTimeline(requestID: "request-1", prefillStart: nil, prefillEnd: nil)
                .prefillTokensPerSecond(promptTokens: 50, expectedRequestID: "request-1")
        )
    }

    func testSSESuccessMalformedPayloadAndMissingDone() throws {
        var success = BenchmarkLLMSSEMeasurement(startedAt: 0)
        try success.consume(event(
            0.1,
            .line(#"data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#)
        ))
        try success.consume(event(0.2, .line("data: [DONE]")))
        XCTAssertNoThrow(try success.validate())

        var malformed = BenchmarkLLMSSEMeasurement(startedAt: 0)
        XCTAssertThrowsError(try malformed.consume(event(0.1, .line("data: {not-json}")))) {
            XCTAssertTrue($0 is BenchmarkRunnerError)
        }

        var missingDone = BenchmarkLLMSSEMeasurement(startedAt: 0)
        try missingDone.consume(event(
            0.1,
            .line(#"data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#)
        ))
        XCTAssertThrowsError(try missingDone.validate()) {
            XCTAssertEqual($0 as? BenchmarkRunnerError, .missingDone)
        }
    }

    func testWarmupExclusionAndMeanP50P95() throws {
        var samples = [sample(value: 1_000, run: 0, warmup: true)]
        samples.append(contentsOf: (1...5).map { sample(value: Double($0), run: $0, warmup: false) })
        let summary = BenchmarkThermalSummary.calculate(thermalState: .warm, samples: samples)
        XCTAssertEqual(summary.sampleCount, 5)
        XCTAssertEqual(try XCTUnwrap(summary.ttftMilliseconds.mean), 3, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(summary.ttftMilliseconds.p50), 3, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(summary.ttftMilliseconds.p95), 4.8, accuracy: 0.001)

        let tooFew = BenchmarkMetricSummary.calculate([1, 2, 3, 4])
        XCTAssertNil(tooFew.p95)
    }

    func testSpeedupRequiresIdenticalConfigurationAndPromptTokenCounts() throws {
        let single = result(topology: "single-current", total: 2_000, promptTokens: 100)
        let tp2 = result(topology: "tp2", total: 1_000, promptTokens: 100)
        XCTAssertTrue(single.isComparable(to: tp2))
        XCTAssertEqual(try XCTUnwrap(single.speedup(comparedWith: tp2)), 2, accuracy: 0.001)

        let differentPromptTokens = result(topology: "tp2", total: 1_000, promptTokens: 101)
        XCTAssertFalse(single.isComparable(to: differentPromptTokens))
        XCTAssertNil(single.speedup(comparedWith: differentPromptTokens))

        var differentOutput = tp2
        differentOutput.configuration.maximumOutputTokens = 128
        XCTAssertFalse(single.isComparable(to: differentOutput))

        var differentInputSizes = tp2
        differentInputSizes.configuration.inputTokenSizes = [1_024]
        XCTAssertFalse(single.isComparable(to: differentInputSizes))

        var videoSingle = result(
            topology: "single-current",
            total: 20_000,
            promptTokens: nil,
            configuration: .video()
        )
        var videoTP2 = result(
            topology: "tp2",
            total: 10_000,
            promptTokens: nil,
            configuration: .video()
        )
        XCTAssertEqual(try XCTUnwrap(videoSingle.speedup(comparedWith: videoTP2)), 2, accuracy: 0.001)
        videoTP2.configuration.seed = 7
        XCTAssertNil(videoSingle.speedup(comparedWith: videoTP2))
        videoSingle.configuration.fast = true
        XCTAssertNil(videoSingle.speedup(comparedWith: videoTP2))
    }

    func testCancelIsIdempotentAndRunnerCanRunAgain() async throws {
        let root = temporaryDirectory("cancel")
        defer { try? FileManager.default.removeItem(at: root) }
        let transport = BenchmarkMockTransport(target: preparedTarget(borrowed: true))
        transport.blocksLanguageStream = true
        let runner = BenchmarkRunner(
            transport: transport,
            resultStore: BenchmarkResultStore(rootDirectory: root),
            clock: { 100 }
        )
        var configuration = BenchmarkConfiguration.language(modelID: "Fixture-LLM")
        configuration.profile = .custom
        configuration.customRuns = 1
        runner.start(configuration)
        try await waitUntil { runner.state == .warmup }
        runner.cancel()
        runner.cancel()
        await runner.cancelAndWait()
        XCTAssertEqual(runner.state, .completed)
        XCTAssertEqual(runner.currentStage, "Cancelled safely")
        XCTAssertEqual(transport.streamTerminationCount, 1)

        transport.blocksLanguageStream = false
        runner.start(configuration)
        try await waitUntil { !runner.isActive }
        XCTAssertEqual(runner.state, .completed)
        XCTAssertEqual(runner.currentResult?.samples.filter(\.isMeasuredSuccess).count, 1)
    }

    func testBorrowedResidentCleanupDoesNotIssueAStopRequest() async {
        var requestCount = 0
        let store = TokenityStore(
            dataTransport: { request in
                requestCount += 1
                let response = HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: nil
                )!
                return (Data("{}".utf8), response)
            },
            userDefaults: isolatedDefaults()
        )
        try? await store.benchmarkCleanup(preparedTarget(borrowed: true))
        XCTAssertEqual(requestCount, 0)
    }

    func testResultJSONRoundTripSchemaVersionRetentionAndCSV() throws {
        let root = temporaryDirectory("persistence")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BenchmarkResultStore(rootDirectory: root, retentionLimit: 20)
        let original = result(topology: "single-current", total: 1_000, promptTokens: 42)
        let savedURL = try store.save(original)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: savedURL)) as? [String: Any]
        )
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(store.load().first, original)

        let exportURL = root.appendingPathComponent("export.csv")
        try store.exportCSV(original, to: exportURL)
        let csv = try String(contentsOf: exportURL, encoding: .utf8)
        XCTAssertTrue(csv.contains("schema_version"))
        XCTAssertTrue(csv.contains("requested_input_tokens"))
        XCTAssertTrue(csv.contains("request-1"))
        XCTAssertFalse(csv.localizedCaseInsensitiveContains("authorization"))
    }

    func testBenchmarkVisualStateMatrixLightDarkCompactStandardWide() throws {
        let output = URL(fileURLWithPath: "/tmp/tokenity-benchmark-matrix", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let store = TokenityStore(userDefaults: isolatedDefaults())
        let sizes: [(String, NSSize)] = [
            ("compact", NSSize(width: 820, height: 720)),
            ("standard", NSSize(width: 1_080, height: 820)),
            ("wide", NSSize(width: 1_360, height: 900)),
        ]
        let states: [(String, BenchmarkConfiguration, BenchmarkPhase, BenchmarkResult?, String?)] = [
            ("llm-idle", .language(), .idle, nil, nil),
            ("llm-running", .language(), .running, nil, nil),
            ("llm-completed", .language(), .completed, result(topology: "single-current", total: 1_000, promptTokens: 42), nil),
            ("video-configuration", .video(), .idle, nil, nil),
            ("video-running", .video(), .running, nil, nil),
            ("error", .language(), .failed, nil, "Fixture benchmark failure"),
        ]

        for (name, configuration, phase, result, error) in states {
            for scheme in [ColorScheme.light, .dark] {
                for size in sizes {
                    let transport = BenchmarkMockTransport(target: preparedTarget(borrowed: true))
                    let runner = BenchmarkRunner(
                        transport: transport,
                        resultStore: BenchmarkResultStore(rootDirectory: output.appendingPathComponent(UUID().uuidString))
                    )
                    runner.state = phase
                    runner.currentStage = phase == .running ? "Running fixture sample" : phase.title
                    runner.totalRuns = phase == .running ? 5 : 0
                    runner.completedRuns = phase == .running ? 2 : 0
                    runner.progress = phase == .running ? 0.4 : (phase == .completed ? 1 : 0)
                    runner.rankStatuses = phase == .running ? ["Rank 0": "Running · Mac A"] : [:]
                    runner.currentResult = result
                    runner.errorMessage = error

                    let rootView = BenchmarkView(runner: runner, configuration: configuration)
                        .environmentObject(store)
                        .tokenityThemed()
                        .environment(\.colorScheme, scheme)
                        .environment(\.displayScale, 1)
                    let hostingView = NSHostingView(rootView: rootView)
                    hostingView.frame = NSRect(origin: .zero, size: size.1)
                    hostingView.layoutSubtreeIfNeeded()
                    let representation = try XCTUnwrap(
                        hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
                    )
                    hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
                    let data = try XCTUnwrap(
                        representation.representation(using: .png, properties: [:])
                    )
                    let appearance = scheme == .light ? "light" : "dark"
                    try data.write(to: output.appendingPathComponent("\(name)-\(appearance)-\(size.0).png"))
                    XCTAssertGreaterThan(data.count, 4_000)
                }
            }
        }
    }

    private func event(
        _ timestamp: Double,
        _ payload: BenchmarkTimedStreamEvent.Payload
    ) -> BenchmarkTimedStreamEvent {
        BenchmarkTimedStreamEvent(timestamp: timestamp, payload: payload)
    }

    private func sample(value: Double, run: Int, warmup: Bool) -> BenchmarkRunSample {
        BenchmarkRunSample(
            runIndex: run,
            isWarmup: warmup,
            thermalState: .warm,
            kind: .languageModel,
            modelID: "Fixture-LLM",
            topology: "single-current",
            nodes: ["Mac A"],
            rankOrder: ["Mac A"],
            promptTokens: 100,
            completionTokens: 10,
            ttftMilliseconds: value,
            prefillTokensPerSecond: value,
            decodeTokensPerSecond: value,
            totalMilliseconds: value,
            requestID: "request-\(run)",
            didReceiveDone: true,
            status: .succeeded,
            timestamp: Date(timeIntervalSince1970: Double(run))
        )
    }

    private func result(
        topology: String,
        total: Double,
        promptTokens: Int?,
        configuration: BenchmarkConfiguration = .language(modelID: "Fixture-LLM")
    ) -> BenchmarkResult {
        var configuration = configuration
        configuration.modelRevision = "fixture-revision"
        configuration.quantization = "4-bit"
        let kind = configuration.kind
        let thermal: BenchmarkThermalState = kind == .videoGeneration ? .cold : .warm
        let sample = BenchmarkRunSample(
            runIndex: 1,
            isWarmup: false,
            thermalState: thermal,
            kind: kind,
            modelID: configuration.modelID,
            modelRevision: configuration.modelRevision,
            quantization: configuration.quantization,
            topology: topology,
            nodes: topology == "tp2" ? ["Mac A", "Mac B"] : ["Mac A"],
            rankOrder: topology == "tp2" ? ["Mac A", "Mac B"] : ["Mac A"],
            promptTokens: promptTokens,
            completionTokens: promptTokens == nil ? nil : 10,
            ttftMilliseconds: promptTokens == nil ? nil : 100,
            prefillTokensPerSecond: promptTokens == nil ? nil : 200,
            decodeTokensPerSecond: promptTokens == nil ? nil : 20,
            totalMilliseconds: total,
            requestID: "request-1",
            didReceiveDone: promptTokens == nil ? nil : true,
            width: kind == .videoGeneration ? configuration.width : nil,
            height: kind == .videoGeneration ? configuration.height : nil,
            frames: kind == .videoGeneration ? configuration.frames : nil,
            actualFPS: kind == .videoGeneration ? 24 : nil,
            steps: kind == .videoGeneration ? configuration.steps : nil,
            seed: kind == .videoGeneration ? configuration.seed : nil,
            fast: kind == .videoGeneration ? configuration.fast : nil,
            samplingMilliseconds: kind == .videoGeneration ? total * 0.9 : nil,
            framesPerSecond: kind == .videoGeneration ? Double(configuration.frames) / (total / 1_000) : nil,
            secondsPerFrame: kind == .videoGeneration ? (total / 1_000) / Double(configuration.frames) : nil,
            status: .succeeded,
            timestamp: Date(timeIntervalSince1970: 1)
        )
        var result = BenchmarkResult(
            configuration: configuration,
            topology: topology,
            nodes: sample.nodes,
            rankOrder: sample.rankOrder,
            runtimeInstanceID: "fixture-instance",
            borrowedResidentModel: true,
            modelLoadMilliseconds: nil,
            startedAt: Date(timeIntervalSince1970: 0),
            completedAt: Date(timeIntervalSince1970: 2),
            samples: [sample],
            summaries: []
        )
        result.refreshSummaries()
        return result
    }

    private func preparedTarget(borrowed: Bool) -> BenchmarkPreparedTarget {
        BenchmarkPreparedTarget(
            kind: .languageModel,
            modelID: "Fixture-LLM",
            topology: "single-current",
            nodeIDs: ["mac-a"],
            nodes: ["Mac A"],
            rankOrder: ["Mac A"],
            modelRevision: "fixture-revision",
            quantization: "4-bit",
            instanceID: "fixture-instance",
            borrowedResidentModel: borrowed,
            needsLoading: false,
            serviceBaseURL: "http://127.0.0.1:8000/v1",
            loadMilliseconds: nil,
            previousBackendMode: .singleNode,
            previousCoordinatorID: "mac-a",
            previousSelectedNodeIDs: ["mac-a"],
            previousH3WorkerAgentURL: ""
        )
    }

    private func isolatedDefaults() -> UserDefaults {
        let name = "BenchmarkTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    private func temporaryDirectory(_ label: String) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenityBenchmarkTests-\(label)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func waitUntil(
        _ predicate: @escaping @MainActor () -> Bool,
        timeout: Duration = .seconds(2)
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while !predicate() {
            if clock.now >= deadline { return XCTFail("Timed out waiting for Benchmark state") }
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

@MainActor
private final class BenchmarkMockTransport: BenchmarkTransport {
    var target: BenchmarkPreparedTarget
    var blocksLanguageStream = false
    var streamTerminationCount = 0
    var cleanupTargets: [BenchmarkPreparedTarget] = []
    var languageConfigurations: [BenchmarkConfiguration] = []

    init(target: BenchmarkPreparedTarget) {
        self.target = target
    }

    func preflight(_ configuration: BenchmarkConfiguration) async throws -> BenchmarkPreparedTarget {
        target
    }

    func prepare(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkPreparedTarget {
        target
    }

    func languageStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error> {
        languageConfigurations.append(configuration)
        if blocksLanguageStream {
            return AsyncThrowingStream { continuation in
                continuation.onTermination = { [weak self] _ in
                    Task { @MainActor in self?.streamTerminationCount += 1 }
                }
            }
        }
        return AsyncThrowingStream { continuation in
            continuation.yield(.init(
                timestamp: 100.01,
                payload: .response(.init(
                    requestID: "request-1",
                    modelRevision: target.modelRevision,
                    instanceID: target.instanceID
                ))
            ))
            continuation.yield(.init(
                timestamp: 100.10,
                payload: .line(#"data: {"choices":[{"delta":{"content":"Fixture output"},"finish_reason":null}]}"#)
            ))
            continuation.yield(.init(
                timestamp: 100.40,
                payload: .line(#"data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}"#)
            ))
            continuation.yield(.init(timestamp: 100.50, payload: .line("data: [DONE]")))
            continuation.finish()
        }
    }

    func requestTimeline(
        requestID: String,
        target: BenchmarkPreparedTarget
    ) async throws -> BenchmarkRequestTimeline? {
        BenchmarkRequestTimeline(requestID: requestID, prefillStart: 1, prefillEnd: 1.5)
    }

    func videoStream(
        _ configuration: BenchmarkConfiguration,
        target: BenchmarkPreparedTarget
    ) -> AsyncThrowingStream<BenchmarkTimedStreamEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func saveVideoPreview(
        completionLine: String,
        configuration: BenchmarkConfiguration
    ) async throws -> URL? {
        nil
    }

    func cleanup(_ target: BenchmarkPreparedTarget) async throws {
        cleanupTargets.append(target)
    }
}
