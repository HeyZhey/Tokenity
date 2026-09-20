import AVFoundation
import AppKit
import Combine
import Foundation
import SwiftUI
import XCTest
@testable import TokenityControl

/// Opt-in hardware validation through the same Store, Node Agent, SSE parser,
/// AVFoundation writer and native player as the Video workspace.
final class VideoGenerationLiveTests: XCTestCase {
    @MainActor
    func testLiveGatewayJSONErrorKeepsNativeDetail() async throws {
        guard let endpoint = ProcessInfo.processInfo.environment["TOKENITY_H3_ERROR_AGENT"] else {
            throw XCTSkip("Set TOKENITY_H3_ERROR_AGENT to a local test Node Agent.")
        }
        let defaultsName = "TokenityH3HTTPError-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: defaultsName)!
        defer { defaults.removePersistentDomain(forName: defaultsName) }
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/start-minimax-h3-video" {
                let data = Data(#"{"instance_id":"h3-deliberately-absent-instance"}"#.utf8)
                return (data, HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: [:])!)
            }
            let (data, response) = try await URLSession.shared.data(for: request)
            return (data, try XCTUnwrap(response as? HTTPURLResponse))
        }, userDefaults: defaults)
        store.h3CoordinatorAgentURL = endpoint
        store.h3WorkerAgentURL = ""
        await store.startVideoRuntime()
        XCTAssertTrue(store.isVideoRuntimeReady)
        await store.generateVideo()
        XCTAssertFalse(store.isVideoGenerating)
        XCTAssertTrue(store.videoGenerationError?.contains("No ready MiniMax H3 instance matches") == true,
                      store.videoGenerationError ?? "No error received")
    }

    @MainActor
    func testLiveGatewayMatrix() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let output = env["TOKENITY_H3_LIVE_OUT"],
              let binary = env["TOKENITY_H3_LIVE_BINARY"],
              let model = env["TOKENITY_H3_LIVE_MODEL"] else {
            throw XCTSkip("Set TOKENITY_H3_LIVE_OUT/BINARY/MODEL for the hardware gateway matrix.")
        }
        let root = URL(fileURLWithPath: output, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let defaultsName = "TokenityH3Live-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: defaultsName)!
        defer { defaults.removePersistentDomain(forName: defaultsName) }
        var startResponse: [String: Any] = [:]
        var runDirectory = root
        var runStarted = Date()
        var streamSeconds = 0.0
        var saveSeconds = 0.0
        var progress: [[String: Any]] = []
        let store = TokenityStore(
            dataTransport: { request in
                let (data, response) = try await URLSession.shared.data(for: request)
                if request.url?.path == "/v1/node/start-minimax-h3-video" {
                    startResponse = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
                }
                return (data, try XCTUnwrap(response as? HTTPURLResponse))
            },
            videoArtifactTransport: { payload, request in
                streamSeconds = Date().timeIntervalSince(runStarted)
                let saveStarted = Date()
                let artifact = try await H3VideoArtifactWriter.write(payload: payload, request: request, rootDirectory: runDirectory)
                saveSeconds = Date().timeIntervalSince(saveStarted)
                return artifact
            },
            userDefaults: defaults
        )
        store.h3CoordinatorAgentURL = env["TOKENITY_H3_LIVE_AGENT"] ?? "http://127.0.0.1:19100"
        store.h3WorkerAgentURL = ""
        store.h3ModelPath = model
        store.h3BinaryPath = binary
        store.h3OptimizationProfile = .stockQMM
        store.selectedNodeIDs = []
        store.startStatusMonitoring()
        defer { store.stopStatusMonitoring() }
        store.videoRequest.prompt = "A cinematic medium shot of a young woman in a yellow raincoat at a rainy neon street food stall. She smiles, picks up a steaming ceramic cup with both hands and slowly turns toward the camera. The camera gently tracks sideways. Rain patters and the cup softly clinks. Natural facial features, clearly visible fingers, detailed wet fabric, coherent smooth motion, synchronized sound."
        let observation = store.$videoProgress.sink { value in
            progress.append(["elapsed_s": Date().timeIntervalSince(runStarted), "progress": value])
        }
        defer { observation.cancel() }
        let configs = (env["TOKENITY_H3_LIVE_CONFIGS"] ?? "turbo6,baseline,turbo4,turbo8").split(separator: ",").map(String.init)
        let warmRuns = Int(env["TOKENITY_H3_LIVE_WARM_RUNS"] ?? "3") ?? 3
        func write(_ value: Any, to url: URL) throws {
            try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        }
        // Mount the actual Video workspace and its native AVPlayer view.
        let view = VideoGenerationPage().environmentObject(store).frame(width: 1200, height: 1000)
        let hosting = NSHostingView(rootView: view)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 1000), styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = hosting
        defer { window.contentView = nil }
        for config in configs {
            let folder = root.appendingPathComponent(config, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            store.setVideoTurbo(config.hasPrefix("turbo"))
            if config.hasPrefix("turbo") { store.videoRequest.steps = Int(config.dropFirst(5))! }
            XCTAssertEqual(store.videoRequest.steps, config == "baseline" ? 28 : Int(config.dropFirst(5))!)
            await store.startVideoRuntime()
            guard store.isVideoRuntimeReady else {
                throw H3VideoContractError.server(store.videoGenerationError ?? "Live runtime did not start")
            }
            try write(startResponse, to: folder.appendingPathComponent("start-response.json"))
            let status = try XCTUnwrap(startResponse["status"] as? [String: Any])
            let pid = try XCTUnwrap(status["pid"] as? Int)
            let logPath = try XCTUnwrap(status["log_path"] as? String)
            do {
                for run in 0...warmRuns {
                    let label = run == 0 ? "cold" : "warm\(run)"
                    runDirectory = folder.appendingPathComponent(label, isDirectory: true)
                    guard !FileManager.default.fileExists(atPath: runDirectory.path) else {
                        throw H3VideoContractError.server("Use a fresh output directory; existing runs are never overwritten.")
                    }
                    try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
                    var submittedRequest = store.videoRequest.validated()
                    submittedRequest.model = store.videoRuntimeInstanceID
                    let requestData = try JSONEncoder().encode(submittedRequest)
                    try requestData.write(to: runDirectory.appendingPathComponent("request.json"))
                    let offset = (try FileManager.default.attributesOfItem(atPath: logPath)[.size] as? NSNumber)?.intValue ?? 0
                    runStarted = Date()
                    progress = []
                    try write(["pid": pid, "directory": runDirectory.path, "started_epoch": runStarted.timeIntervalSince1970], to: root.appendingPathComponent("active-run.json"))
                    await store.generateVideo()
                    let total = Date().timeIntervalSince(runStarted)
                    guard let artifact = store.generatedVideoArtifact, store.videoGenerationError == nil else {
                        throw H3VideoContractError.server(store.videoGenerationError ?? "Live generation did not return an artifact")
                    }
                    XCTAssertEqual(store.videoProgress, 1)
                    XCTAssertEqual(store.videoProgressStage, "Complete")
                    XCTAssertEqual(artifact.frames, 124)
                    XCTAssertTrue(artifact.hasMuxedAudio)
                    let asset = AVURLAsset(url: artifact.movieURL)
                    let playable = try await asset.load(.isPlayable)
                    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
                    XCTAssertTrue(playable)
                    XCTAssertEqual(audioTracks.count, 1)
                    hosting.layoutSubtreeIfNeeded()
                    let rawLog = try Data(contentsOf: URL(fileURLWithPath: logPath))
                    try rawLog.dropFirst(offset).write(to: runDirectory.appendingPathComponent("server.log"))
                    try write(progress, to: runDirectory.appendingPathComponent("ui-progress.json"))
                    try write([
                        "config": config, "run": run, "cold": run == 0,
                        "pid": pid, "instance_id": store.videoRuntimeInstanceID ?? "",
                        "gateway_stream_s": streamSeconds, "ui_save_s": saveSeconds, "ui_total_s": total,
                        "backend_http_s": NSNull(), "artifact_directory": artifact.directoryURL.path,
                        "avfoundation_playable": playable, "has_muxed_audio": artifact.hasMuxedAudio,
                        "finished_epoch": Date().timeIntervalSince1970,
                    ], to: runDirectory.appendingPathComponent("ui-result.json"))
                    print("TOKENITY_H3_LIVE \(config) \(label) total=\(total) save=\(saveSeconds) artifact=\(artifact.movieURL.path)")
                    fflush(stdout)
                }
                await store.stopVideoRuntime()
                XCTAssertEqual(store.videoRuntimeState, .stopped)
            } catch {
                await store.stopVideoRuntime()
                throw error
            }
        }
        try write(["complete": true], to: root.appendingPathComponent("active-run.json"))
        try write(store.logs, to: root.appendingPathComponent("store-log.json"))
    }
}
