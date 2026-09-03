import AVFoundation
import AVKit
import AppKit
import Foundation
import SwiftUI
import XCTest
@testable import TokenityControl

final class VideoGenerationContractTests: XCTestCase {
    func testHistoryRestoresCompletedArtifactsAndMarksPartialOrCorruptRunsInterrupted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenityVideoHistory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let completedID = UUID()
        let completed = root.appendingPathComponent(completedID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: completed, withIntermediateDirectories: true)
        try Data([0]).write(to: completed.appendingPathComponent("generation.mov"))
        try Data([1, 2, 3]).write(to: completed.appendingPathComponent("video.rgb"))
        try Data(#"{"prompt":"test","seed":42,"steps":28,"fast":false,"frames":24,"width":512,"height":256,"fps":24,"has_muxed_audio":false}"#.utf8)
            .write(to: completed.appendingPathComponent("metadata.json"))

        for corruptMetadata in [false, true] {
            let interrupted = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(at: interrupted, withIntermediateDirectories: true)
            try Data([1, 2, 3]).write(to: interrupted.appendingPathComponent("video.rgb"))
            if corruptMetadata {
                try Data("not-json".utf8).write(to: interrupted.appendingPathComponent("metadata.json"))
                try Data([0]).write(to: interrupted.appendingPathComponent("generation.mov"))
            }
        }

        let history = H3VideoArtifactWriter.loadHistory(rootDirectory: root)

        XCTAssertEqual(history.completed.map(\.id), [completedID])
        XCTAssertEqual(history.completed.first?.durationSeconds, 1)
        XCTAssertEqual(history.interruptedCount, 2)
    }

    func testVideoWorkspaceIsAFirstClassClusterSection() {
        XCTAssertTrue(AppSection.allCases.contains(.video))
        XCTAssertEqual(AppSection.video.title, "Video")
        XCTAssertEqual(AppSection.video.group, "Cluster")
        XCTAssertEqual(AppSection.video.symbol, "film.stack")
    }

    func testH3StartRequestUsesTypedTwoNodeContract() throws {
        let request = AgentStartH3VideoRequest(
            model: "/fixtures/tokenity/h3-model",
            binary: "/fixtures/tokenity/mlx-serve",
            nodes: [
                AgentClusterNodeRequest(
                    id: "mac-a",
                    agentURL: "http://127.0.0.1:9100",
                    lanIP: "127.0.0.1",
                    rdmaIP: "tokenity-rdma-a.invalid",
                    rdmaDevices: ["rdma_en4"]
                ),
                AgentClusterNodeRequest(
                    id: "mac-b",
                    agentURL: "http://198.51.100.75:9200",
                    lanIP: "198.51.100.75",
                    rdmaIP: "tokenity-rdma-b.invalid",
                    rdmaDevices: ["rdma_en5"]
                ),
            ],
            connectionMode: "jaccl",
            startingPort: 30_096,
            host: "0.0.0.0",
            port: 11_242,
            dryRun: false,
            apiIdentifier: "MiniMax-H3",
            leaseSeconds: 30,
            instanceID: "h3-ui-instance",
            operationID: "h3-ui-operation",
            optimizationProfile: .stockQMM,
            memoryHeadroomRatio: 0.1
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let nodes = try XCTUnwrap(object["nodes"] as? [[String: Any]])

        XCTAssertEqual(nodes.count, 2)
        XCTAssertEqual(nodes[1]["agent_url"] as? String, "http://198.51.100.75:9200")
        XCTAssertEqual(object["optimization_profile"] as? String, "stock-qmm")
        XCTAssertEqual(object["api_identifier"] as? String, "MiniMax-H3")
        XCTAssertEqual(object["memory_headroom_ratio"] as? Double, 0.1)
        XCTAssertNil(object["environment"])
        XCTAssertNil(object["shell"])
    }

    func testVideoGenerationRequestKeepsFixedH3DefaultsAndClampsUnsafeValues() throws {
        var request = H3VideoGenerationRequest.default
        request.prompt = "  A paper boat crosses a neon street.  "
        request.width = 9_999
        request.height = 1
        request.numFrames = 1_000
        request.steps = 0

        let validated = request.validated()

        XCTAssertEqual(validated.prompt, "A paper boat crosses a neon street.")
        XCTAssertEqual(validated.width, 1_024)
        XCTAssertEqual(validated.height, 128)
        XCTAssertEqual(validated.numFrames, 345)
        XCTAssertEqual(validated.steps, 1)
        XCTAssertTrue(validated.stream)
        XCTAssertFalse(validated.fast)
    }

    func testVideoSSEParserSeparatesProgressAndCompletePayloads() throws {
        let progress = try XCTUnwrap(
            H3VideoSSEParser.parse(
                line: #"data: {"type":"progress","stage":"Generating","step":7,"total":28}"#
            )
        )
        XCTAssertEqual(progress, .progress(stage: "Generating", step: 7, total: 28))

        let video = Data([1, 2, 3, 4, 5, 6])
        let audio = Data([7, 8, 9, 10])
        let completeLine = """
        data: {"type":"complete","frames":1,"height":1,"width":2,"fps":24,"format":"rgb8","data":"\(video.base64EncodedString())","audio_channels":2,"audio_format":"pcm_s16le","audio_sample_rate":32000,"audio_data":"\(audio.base64EncodedString())"}
        """
        let complete = try XCTUnwrap(H3VideoSSEParser.parse(line: completeLine))
        guard case .complete(let payload) = complete else {
            return XCTFail("Expected complete video event")
        }
        XCTAssertEqual(payload.videoData, video)
        XCTAssertEqual(payload.audioData, audio)
        XCTAssertEqual(payload.width, 2)
        XCTAssertEqual(payload.audioSampleRate, 32_000)
    }

    func testArtifactWriterCreatesPlayableMovieAndPreservesRawEvidence() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenityVideoWriterTest-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let width = 128
        let height = 128
        let frames = 5
        let video = Data(repeating: 96, count: width * height * frames * 3)
        let audioSampleFrames = Int(ceil((Double(frames) / 24) * 32_000))
        let audio = Data(repeating: 0, count: audioSampleFrames * 2 * 2)
        let payload = H3VideoCompletePayload(
            frames: frames,
            height: height,
            width: width,
            fps: 24,
            format: "rgb8",
            videoData: video,
            audioChannels: 2,
            audioFormat: "pcm_s16le",
            audioSampleRate: 32_000,
            audioData: audio
        )
        var request = H3VideoGenerationRequest.default
        request.width = width
        request.height = height
        request.numFrames = frames

        let artifact = try await H3VideoArtifactWriter.write(
            payload: payload,
            request: request,
            rootDirectory: root
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: artifact.movieURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: artifact.rawVideoURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: try XCTUnwrap(artifact.waveURL).path))
        XCTAssertTrue(artifact.hasMuxedAudio)
        let asset = AVURLAsset(url: artifact.movieURL)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(videoTracks.count, 1)
        XCTAssertEqual(audioTracks.count, 1)

        let mountedNativePlayer = await MainActor.run {
            let player = AVPlayer(url: artifact.movieURL)
            let root = TokenityVideoPlayer(player: player)
                .frame(width: 640, height: 360)
            let hostingView = NSHostingView(rootView: root)
            hostingView.frame = NSRect(x: 0, y: 0, width: 640, height: 360)
            let window = NSWindow(
                contentRect: hostingView.frame,
                styleMask: [.titled],
                backing: .buffered,
                defer: false
            )
            window.contentView = hostingView
            hostingView.layoutSubtreeIfNeeded()

            var pending: [NSView] = [hostingView]
            var foundPlayerView = false
            while let view = pending.popLast() {
                if view is AVPlayerView {
                    foundPlayerView = true
                    break
                }
                pending.append(contentsOf: view.subviews)
            }

            player.pause()
            window.contentView = nil
            return foundPlayerView
        }
        XCTAssertTrue(mountedNativePlayer)
    }

    @MainActor
    func testH3WorkspaceStartsUnboundInsteadOfTargetingADeployment() throws {
        let store = TokenityStore()

        XCTAssertEqual(store.h3CoordinatorAgentURL, "http://127.0.0.1:9100")
        XCTAssertTrue(store.h3WorkerAgentURL.isEmpty)
        XCTAssertEqual(store.nodes.first { $0.id == "mac-b" }?.agentURL, "")
    }

    @MainActor
    func testMiniMaxH3IsAFirstClassVideoModelInUnifiedLibrary() throws {
        let store = TokenityStore()
        let model = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "MiniMax-H3" })

        XCTAssertEqual(model.displayName, "MiniMax H3")
        XCTAssertEqual(model.modality, .video)
        XCTAssertEqual(model.representativePath, store.h3ModelPath)
        XCTAssertEqual(model.quantization, "8-bit · group 64")
        XCTAssertEqual(model.loadState, .notLoaded)
        XCTAssertFalse(store.availableChatModelIDs.contains(model.id))
    }

    @MainActor
    func testVideoUsesTheSelectedClusterMacsWhenNoLegacyWorkerOverrideExists() async {
        let store = TokenityStore(dataTransport: { request in
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://localhost")!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (Data(TokenityTestFixtures.modernNodeInfoPayload(for: request).utf8), response)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].isOnline = true
        }

        await store.refreshVideoNodes()

        XCTAssertEqual(
            store.videoNodes.map(\.agentURL),
            ["http://127.0.0.1:9100", "http://198.51.100.75:9200"]
        )
    }

    @MainActor
    func testH3WorkerLoopbackIsNeverAcceptedAsMacB() async throws {
        let store = TokenityStore(dataTransport: { request in
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(Self.h3NodeInfoPayload(for: request).utf8), response)
        })
        store.h3WorkerAgentURL = "http://127.0.0.2:9200"

        await store.refreshVideoNodes()

        let worker = try XCTUnwrap(store.videoNodes.first { $0.id == "h3-mac-b" })
        XCTAssertFalse(worker.isOnline)
        XCTAssertTrue(store.videoReadiness.contains {
            $0.state == .secondMacUnavailable && $0.action == .useOneMac
        })
    }

    @MainActor
    func testH3TP2RequiresTwoStableMachineIDs() async {
        let store = TokenityStore(dataTransport: { request in
            let payload = Self.h3NodeInfoPayload(for: request)
                .replacingOccurrences(of: "mac-b-machine", with: "mac-a-machine")
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"

        await store.refreshVideoNodes()

        XCTAssertTrue(store.videoReadiness.contains {
            $0.state == .componentsNeedUpdate && $0.action == .installOrRepair
        })
    }

    @MainActor
    func testH3DryRunBlocksAnIncompleteModelBeforeStart() async throws {
        var sawDryRun = false
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let status: Int
            let payload: String
            switch path {
            case "/v1/node/info":
                status = 200
                payload = Self.h3NodeInfoPayload(for: request)
            case "/v1/node/start-minimax-h3-video":
                let body = try XCTUnwrap(request.httpBody)
                let object = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: body) as? [String: Any]
                )
                sawDryRun = object["dry_run"] as? Bool == true
                status = 412
                payload = #"{"detail":{"stage":"minimax_h3_preflight","issues":["Model directory is incomplete: config.json is missing."]}}"#
            default:
                status = 200
                payload = #"{}"#
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: status,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"

        await store.refreshVideoNodes(validateRuntime: true)

        XCTAssertTrue(sawDryRun)
        XCTAssertTrue(store.videoReadiness.contains {
            $0.state == .modelNotFound && $0.action == .chooseFolder
        })
        let videoModel = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "MiniMax-H3" })
        XCTAssertFalse(store.canLoadModel(videoModel))
    }

    @MainActor
    func testStoppingAnInFlightVideoLoadReleasesTheLoadFenceForRetry() async throws {
        let firstPreflight = expectation(description: "first H3 preflight began")
        let retryStart = expectation(description: "retry H3 start began")
        var startCount = 0
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            switch path {
            case "/v1/node/info":
                payload = Self.h3NodeInfoPayload(for: request)
            case "/v1/node/start-minimax-h3-video":
                startCount += 1
                let object: [String: Any]?
                if let body = request.httpBody {
                    object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
                } else {
                    object = nil
                }
                let isDryRun = object?["dry_run"] as? Bool == true
                if startCount == 1, isDryRun {
                    firstPreflight.fulfill()
                    try await Task.sleep(for: .seconds(30))
                } else if !isDryRun {
                    retryStart.fulfill()
                }
                payload = #"{"instance_id":"h3-retry-runtime","operation_id":"h3-retry-operation","api_base_url":"http://127.0.0.1:11242/v1"}"#
            default:
                payload = #"{}"#
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"
        let videoModel = try XCTUnwrap(
            store.modelLibraryRows.first { $0.id == "MiniMax-H3" }
        )

        store.beginLoadingModel(videoModel)
        await fulfillment(of: [firstPreflight], timeout: 2)
        XCTAssertEqual(store.videoRuntimeLoadProgress, 0.15)
        await store.stopModel(videoModel)
        XCTAssertNil(store.videoRuntimeLoadProgress)
        store.beginLoadingModel(videoModel)
        await fulfillment(of: [retryStart], timeout: 2)
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(startCount, 3)
        XCTAssertEqual(store.videoRuntimeState, .ready)
        XCTAssertEqual(store.videoRuntimeLoadProgress, 1)
        await store.stopModel(videoModel)
        XCTAssertNil(store.videoRuntimeLoadProgress)
    }

    @MainActor
    func testStatusRefreshCannotRegressStoppingH3ToStarting() async throws {
        let stopRequestStarted = expectation(description: "H3 stop request started")
        var stopIsInFlight = false
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            switch path {
            case "/v1/node/info":
                let base = Self.h3NodeInfoPayload(for: request)
                payload = stopIsInFlight
                    ? base.replacingOccurrences(
                        of: #""instances":[]"#,
                        with: #""instances":[{"instance_id":"h3-stop-runtime","requested_model_id":"MiniMax-H3","state":"starting"}]"#
                    )
                    : base
            case "/v1/node/start-minimax-h3-video":
                payload = #"{"instance_id":"h3-stop-runtime","operation_id":"h3-stop-operation","api_base_url":"http://127.0.0.1:11242/v1"}"#
            case "/v1/node/instances/h3-stop-runtime/stop":
                stopIsInFlight = true
                stopRequestStarted.fulfill()
                try await Task.sleep(for: .seconds(1))
                payload = #"{}"#
            default:
                payload = #"{}"#
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"

        await store.startVideoRuntime()
        XCTAssertEqual(store.videoRuntimeState, .ready)

        let stopTask = Task { await store.stopVideoRuntime() }
        await fulfillment(of: [stopRequestStarted], timeout: 2)
        await store.refreshVideoNodes()

        XCTAssertEqual(store.videoRuntimeState, .stopping)
        await stopTask.value
        XCTAssertEqual(store.videoRuntimeState, .stopped)
    }

    @MainActor
    func testVideoRefreshRejectsAReadyCoordinatorWhenRankQuorumFailed() async throws {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            switch path {
            case "/v1/node/info":
                payload = Self.h3NodeInfoPayload(for: request)
            case "/v1/node/start-minimax-h3-video":
                payload = #"{"instance_id":"h3-quorum-runtime","operation_id":"h3-quorum-operation","api_base_url":"http://127.0.0.1:11242/v1"}"#
            case "/v1/node/instances/h3-quorum-runtime/quorum":
                payload = #"{"instance_id":"h3-quorum-runtime","ready":false,"issues":["rank 1 native H3 process is not running"],"rank_quorum":"2/2","ranks":[]}"#
            default:
                payload = #"{}"#
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })

        store.h3WorkerAgentURL = "http://198.51.100.75:9200"
        await store.startVideoRuntime()
        XCTAssertEqual(store.videoRuntimeState, .ready)
        await store.refreshVideoNodes()

        guard case .failed(let message) = store.videoRuntimeState else {
            return XCTFail("Expected failed video quorum")
        }
        XCTAssertTrue(message.contains("rank 1"))
        XCTAssertEqual(store.videoProgressStage, "Runtime rank failure")
        await store.stopVideoRuntime()
    }

    @MainActor
    func testStoreStartsTP2RuntimeAndRoutesStreamingGenerationThroughCoordinator() async throws {
        var startBody: [String: Any]?
        var generationBody: [String: Any]?
        var generationPath: String?
        var controlRequests: [URL] = []
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenityVideoUITest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let movieURL = temporary.appendingPathComponent("generation.mov")
        try Data().write(to: movieURL)

        let store = TokenityStore(
            dataTransport: { request in
                if let url = request.url {
                    controlRequests.append(url)
                }
                let path = request.url?.path ?? ""
                let payload: String
                switch path {
                case "/v1/node/info":
                    payload = Self.h3NodeInfoPayload(for: request)
                case "/v1/gateway/routes":
                    payload = #"{"data":[]}"#
                case "/v1/node/start-minimax-h3-video":
                    if let body = request.httpBody {
                        startBody = try JSONSerialization.jsonObject(with: body) as? [String: Any]
                    }
                    payload = #"{"instance_id":"h3-ui-runtime","operation_id":"h3-ui-operation","api_base_url":"http://127.0.0.1:11242/v1"}"#
                default:
                    payload = #"{}"#
                }
                let response = try XCTUnwrap(
                    HTTPURLResponse(
                        url: request.url ?? URL(string: "http://localhost")!,
                        statusCode: 200,
                        httpVersion: nil,
                        headerFields: ["Content-Type": "application/json"]
                    )
                )
                return (Data(payload.utf8), response)
            },
            lineStreamTransport: { request in
                generationPath = request.url?.path
                if let body = request.httpBody {
                    generationBody = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
                }
                return AsyncThrowingStream { continuation in
                    continuation.yield(.line(#"data: {"type":"progress","stage":"Generating","step":14,"total":28}"#))
                    continuation.yield(.line(#"data: {"type":"complete","frames":1,"height":1,"width":2,"fps":24,"format":"rgb8","data":"AQIDBAUG","audio_channels":0,"audio_format":"none","audio_sample_rate":0,"audio_data":""}"#))
                    continuation.finish()
                }
            },
            videoArtifactTransport: { payload, _ in
                GeneratedVideoArtifact(
                    id: UUID(),
                    directoryURL: temporary,
                    movieURL: movieURL,
                    waveURL: nil,
                    rawVideoURL: temporary.appendingPathComponent("video.rgb"),
                    rawAudioURL: nil,
                    frames: payload.frames,
                    width: payload.width,
                    height: payload.height,
                    fps: payload.fps,
                    durationSeconds: Double(payload.frames) / Double(payload.fps),
                    hasMuxedAudio: false
                )
            }
        )
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"
        let videoModel = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "MiniMax-H3" })
        await store.loadModel(videoModel)
        XCTAssertEqual(store.videoRuntimeState, .ready)
        XCTAssertEqual(store.videoRuntimeInstanceID, "h3-ui-runtime")
        XCTAssertEqual(startBody?["optimization_profile"] as? String, "stock-qmm")
        XCTAssertEqual(startBody?["starting_port"] as? Int, 30_096)
        XCTAssertEqual(startBody?["lease_seconds"] as? Double, 120)
        XCTAssertEqual((startBody?["nodes"] as? [[String: Any]])?.count, 2)
        XCTAssertEqual(
            (startBody?["nodes"] as? [[String: Any]])?[1]["agent_url"] as? String,
            "http://198.51.100.75:9200"
        )

        await store.generateVideo()

        XCTAssertEqual(generationPath, "/v1/video/generations")
        XCTAssertEqual(generationBody?["model"] as? String, "h3-ui-runtime")
        XCTAssertEqual(generationBody?["num_frames"] as? Int, 124)
        XCTAssertEqual(store.videoProgressStage, "Complete")
        XCTAssertEqual(store.videoProgress, 1)
        XCTAssertEqual(store.generatedVideoArtifact?.movieURL, movieURL)

        let loadedVideoModel = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "MiniMax-H3" })
        XCTAssertEqual(loadedVideoModel.loadState, .loaded)
        await store.stopModel(loadedVideoModel)
        XCTAssertEqual(store.videoRuntimeState, .stopped)
        XCTAssertTrue(controlRequests.contains { url in
            url.host == "127.0.0.1"
                && url.port == 9_100
                && url.path == "/v1/node/instances/h3-ui-runtime/stop"
        })
        XCTAssertFalse(controlRequests.contains { url in
            url.host == "198.51.100.75"
                && url.port == 9_200
                && url.path.contains("/v1/node/instances/")
        })
    }

    @MainActor
    func testUserCancelledVideoStreamIsNotReportedAsAnUnreadableEvent() async throws {
        var generationContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                let payload: String
                switch path {
                case "/v1/node/info":
                    payload = Self.h3NodeInfoPayload(for: request)
                case "/v1/node/start-minimax-h3-video":
                    payload = #"{"instance_id":"h3-cancel-runtime","operation_id":"h3-cancel-operation","api_base_url":"http://127.0.0.1:11242/v1"}"#
                default:
                    payload = #"{}"#
                }
                let response = try XCTUnwrap(
                    HTTPURLResponse(
                        url: request.url ?? URL(string: "http://localhost")!,
                        statusCode: 200,
                        httpVersion: nil,
                        headerFields: ["Content-Type": "application/json"]
                    )
                )
                return (Data(payload.utf8), response)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { generationContinuation = $0 }
            }
        )
        store.h3WorkerAgentURL = "http://198.51.100.75:9200"
        await store.startVideoRuntime()

        store.beginVideoGeneration()
        for _ in 0..<100 where generationContinuation == nil { await Task.yield() }
        XCTAssertTrue(store.isVideoGenerating)
        store.cancelVideoGeneration()
        generationContinuation?.finish()
        for _ in 0..<100 where store.isVideoGenerating { await Task.yield() }

        XCTAssertFalse(store.isVideoGenerating)
        XCTAssertEqual(store.videoProgressStage, "Cancelled")
        XCTAssertNil(store.videoGenerationError)
        await store.stopVideoRuntime()
    }

    @MainActor
    func testStructuredH3StartupFailureIsShownInsteadOfGenericServerError() async throws {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let status: Int
            let payload: String
            switch path {
            case "/v1/node/info":
                status = 200
                payload = Self.h3NodeInfoPayload(for: request)
            case "/v1/node/start-minimax-h3-video":
                status = 412
                payload = #"{"detail":{"stage":"minimax_h3_preflight","issues":["rank shard mismatch"],"suggestion":"Redeploy matching shards."}}"#
            case _ where path.contains("/v1/node/instances/") && !path.hasSuffix("/stop"):
                status = 404
                payload = #"{"detail":"Unknown model instance"}"#
            default:
                status = 200
                payload = #"{}"#
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url ?? URL(string: "http://localhost")!,
                    statusCode: status,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (Data(payload.utf8), response)
        })

        store.h3WorkerAgentURL = "http://198.51.100.75:9200"
        await store.startVideoRuntime()

        guard case .failed(let message) = store.videoRuntimeState else {
            return XCTFail("Expected a failed H3 runtime")
        }
        XCTAssertEqual(
            message,
            "The video runtime was interrupted. Retry, or install or repair Tokenity components."
        )
        XCTAssertFalse(message.contains("Internal Server Error"))
        XCTAssertNil(store.videoRuntimeInstanceID)
        XCTAssertFalse(store.canStopVideoRuntime)
    }

    private static func h3NodeInfoPayload(for request: URLRequest) -> String {
        let isWorker = request.url?.port == 9_200
        let nodeID = isWorker ? "mac-b" : "mac-a"
        let lanIP = isWorker ? "198.51.100.75" : "127.0.0.1"
        let machineID = isWorker ? "mac-b-machine" : "mac-a-machine"
        let rdmaIP = isWorker ? "tokenity-rdma-b.invalid" : "tokenity-rdma-a.invalid"
        let rdmaDevice = isWorker ? "rdma_en5" : "rdma_en4"
        return """
        {"node_id":"\(nodeID)","hostname":"\(lanIP)","user":"test","ips":["\(lanIP)"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/runtime/python","mlx_version":"0.32.0","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","machine_id":"\(machineID)","tokenity_code_revision":"test-h3-ui","agent_contract":{"version":1,"capabilities":["managed_instances","instance_runtimes","instance_quorum","cluster_runtime","minimax_h3_video"]},"process_roles":[],"memory":{"total_bytes":549755813888,"used_bytes":107374182400,"free_bytes":442381631488,"used_ratio":0.1953125},"rdma":{"rdma_enabled":true,"rdma_devices":["\(rdmaDevice)"],"rdma_port_state":{"\(rdmaDevice)":"active"},"thunderbolt_ip":"\(rdmaIP)","rdma_errors":[]},"instances":[]}
        """
    }
}
