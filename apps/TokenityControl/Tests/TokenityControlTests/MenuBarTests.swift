import AppKit
import Combine
import XCTest
@testable import TokenityControl

@MainActor
final class MenuBarTests: XCTestCase {
    func testServerAndModelStatesMapToMenuBarLevels() async throws {
        let store = TokenityStore(dataTransport: Self.successfulTransport)

        XCTAssertEqual(store.menuBarSnapshot.level, .stopped)
        XCTAssertEqual(store.menuBarSnapshot.serverStatus, "Stopped")

        store.phase = .loadingModel
        XCTAssertEqual(store.menuBarSnapshot.level, .busy)
        XCTAssertEqual(store.menuBarSnapshot.serverStatus, "Starting")

        store.phase = .failed
        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertEqual(store.menuBarSnapshot.serverStatus, "Error")

        let readyStore = try await makeReadyStore(backendMode: .singleNode)
        XCTAssertEqual(readyStore.menuBarSnapshot.level, .ready)
        XCTAssertEqual(readyStore.menuBarSnapshot.serverStatus, "Ready")
        XCTAssertEqual(readyStore.menuBarSnapshot.inferenceMode, "Single Mac")

        let modelID = try XCTUnwrap(readyStore.loadedModelName)
        readyStore.modelLoadStates[modelID] = .unloading
        XCTAssertEqual(readyStore.menuBarSnapshot.level, .busy)
    }

    func testNodeSnapshotsCoverAllOnlinePartiallyOfflineAndAllOffline() async throws {
        let store = try await makeReadyStore(backendMode: .distributed)

        XCTAssertTrue(store.menuBarSnapshot.nodes.allSatisfy { $0.onlineText == "Online" })
        XCTAssertEqual(store.menuBarSnapshot.level, .ready)

        let workerIndex = try XCTUnwrap(store.nodes.firstIndex { $0.id == "mac-b" })
        store.nodes[workerIndex].isOnline = false
        store.nodes[workerIndex].agentError = "Node Agent timed out."

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertEqual(store.menuBarSnapshot.nodes.filter { $0.onlineText == "Offline" }.count, 1)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("partially available") == true)

        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].isOnline = false
            store.nodes[index].agentError = "Node Agent timed out."
        }

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertEqual(store.menuBarSnapshot.nodes.filter { $0.onlineText == "Offline" }.count, 2)
    }

    func testTwoMacReadinessRequiresRolesWorldSizeAndConnectionMode() async throws {
        let store = try await makeReadyStore(backendMode: .distributed)

        XCTAssertEqual(store.menuBarSnapshot.inferenceMode, "Two-Mac distributed")
        XCTAssertEqual(store.menuBarSnapshot.level, .ready)

        let workerIndex = try XCTUnwrap(store.nodes.firstIndex { $0.id == "mac-b" })
        store.nodes[workerIndex].clusterRuntime?.worldSize = 1

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("world size") == true)

        store.nodes[workerIndex].clusterRuntime?.worldSize = 2
        store.nodes[workerIndex].clusterRuntime?.connectionMode = "jaccl"

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("connection mode") == true)
    }

    func testProbeVerifiedServiceWithoutClusterRuntimeMetadataRemainsWarning() async throws {
        let store = try await makeReadyStore(backendMode: .distributed)

        for node in store.selectedNodes {
            let index = try XCTUnwrap(store.nodes.firstIndex { $0.id == node.id })
            store.nodes[index].clusterRuntime = nil
        }

        XCTAssertNotNil(store.loadedModelName, "The fixture must complete the real streaming inference probe.")
        XCTAssertEqual(store.serverHealth, .ready)
        XCTAssertTrue(store.selectedNodes.allSatisfy(\.isOnline))
        XCTAssertTrue(
            store.nodes.first(where: { $0.id == "mac-a" })?.roles.contains {
                $0.role == "distributed-openai" && $0.state == "running" && $0.pid != nil
            } == true
        )
        XCTAssertTrue(
            store.nodes.first(where: { $0.id == "mac-b" })?.roles.contains {
                $0.role == "distributed-openai-rank" && $0.state == "running" && $0.pid != nil
            } == true
        )
        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("runtime metadata") == true)
    }

    func testModernAgentMissingDeclaredClusterRuntimeRemainsWarning() async throws {
        let store = try await makeReadyStore(backendMode: .distributed)
        let contract = AgentContractInfo(
            version: 1,
            capabilities: [
                "cluster_runtime",
                "instance_quorum",
                "instance_runtimes",
                "managed_instances",
            ]
        )
        for node in store.selectedNodes {
            let index = try XCTUnwrap(store.nodes.firstIndex { $0.id == node.id })
            store.nodes[index].agentContract = contract
            store.nodes[index].clusterRuntime = nil
            store.nodes[index].clusterRuntimes = []
        }

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("runtime metadata") == true)
    }

    func testPartialRuntimeMetadataOrMissingWorkerRoleRemainsWarning() async throws {
        let store = try await makeReadyStore(backendMode: .distributed)
        let workerIndex = try XCTUnwrap(store.nodes.firstIndex { $0.id == "mac-b" })

        store.nodes[workerIndex].clusterRuntime = nil

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("runtime metadata") == true)

        configureRuntime(on: store, backendMode: .distributed)
        store.nodes[workerIndex].roles.removeAll { $0.role == "distributed-openai-rank" }

        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertTrue(store.menuBarSnapshot.overallIssue?.contains("missing its Worker inference role") == true)
    }

    func testMenuBarIconSymbolsIncludeGeneratingOverrideWithoutHidingWarnings() async throws {
        XCTAssertEqual(TokenityMenuBarLevel.stopped.symbol, "circle")
        XCTAssertEqual(TokenityMenuBarLevel.busy.symbol, "clock.badge")
        XCTAssertEqual(TokenityMenuBarLevel.ready.symbol, "checkmark.circle.fill")
        XCTAssertEqual(TokenityMenuBarLevel.warning.symbol, "exclamationmark.triangle.fill")

        let store = try await makeReadyStore(backendMode: .distributed)
        XCTAssertEqual(store.menuBarSnapshot.iconSymbol, "checkmark.circle.fill")

        store.isChatRunning = true
        XCTAssertEqual(store.menuBarSnapshot.level, .ready)
        XCTAssertEqual(store.menuBarSnapshot.iconSymbol, "ellipsis.message.fill")

        store.phase = .failed
        XCTAssertEqual(store.menuBarSnapshot.level, .warning)
        XCTAssertEqual(store.menuBarSnapshot.iconSymbol, "exclamationmark.triangle.fill")
    }

    func testStoppingServerCancelsStreamingGeneration() async throws {
        var streamContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let store = TokenityStore(
            dataTransport: Self.successfulTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    streamContinuation = continuation
                }
            }
        )
        store.backendMode = .singleNode
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        configureRuntime(on: store, backendMode: .singleNode)
        store.chatInput = "Keep generating."

        store.beginSendingChatMessage()
        for _ in 0..<100 where !store.isChatRunning || streamContinuation == nil {
            await Task.yield()
        }
        XCTAssertTrue(store.isChatRunning)
        XCTAssertTrue(store.menuBarSnapshot.isGenerating)

        await store.stopCluster()

        XCTAssertFalse(store.isChatRunning)
        XCTAssertEqual(store.phase, .stopped)
        XCTAssertEqual(store.menuBarSnapshot.level, .stopped)
        XCTAssertTrue(store.chatMessages.last?.content.contains("cluster was stopped") == true)
    }

    func testMenuBarUsesAppDelegateStoreAndNavigatesSharedWindowState() {
        let delegate = TokenityAppDelegate()
        let store = delegate.store
        let content = TokenityMenuBarContent(snapshot: store.menuBarSnapshot, store: store)

        XCTAssertTrue(content.store === store)
        store.navigateFromMenuBar(to: .models)
        XCTAssertEqual(delegate.store.selectedSection, .models)
        store.navigateFromMenuBar(to: .chat)
        XCTAssertEqual(delegate.store.selectedSection, .chat)
    }

    func testRefreshMonitoringStartsOnlyOneLoopAndManualRefreshDoesNotAddOne() async {
        let store = TokenityStore(dataTransport: Self.successfulTransport)

        store.startStatusMonitoring(interval: .seconds(60))
        store.startStatusMonitoring(interval: .seconds(60))
        XCTAssertTrue(store.isStatusMonitoring)
        XCTAssertEqual(store.statusMonitoringStartCount, 1)

        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(store.statusMonitoringStartCount, 1)

        store.stopStatusMonitoring()
        XCTAssertFalse(store.isStatusMonitoring)
    }

    func testOneMissedAgentPollDoesNotFlashTheGlobalStatusOffline() async throws {
        var shouldFail = false
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/info" {
                if shouldFail { throw URLError(.timedOut) }
                return try Self.nodeInfoResponse(for: request, serviceRunning: false)
            }
            if request.url?.path == "/v1/node/status" {
                throw URLError(.timedOut)
            }
            return Self.response(for: request, payload: #"{}"#)
        })

        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertTrue(store.selectedNodes.allSatisfy(\.isOnline))

        shouldFail = true
        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertTrue(store.selectedNodes.allSatisfy(\.isOnline))
        XCTAssertTrue(store.selectedNodes.allSatisfy { $0.consecutiveAgentFailures == 1 })

        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertTrue(store.selectedNodes.allSatisfy { !$0.isOnline })
        XCTAssertTrue(store.selectedNodes.allSatisfy { $0.consecutiveAgentFailures == 2 })
    }

    func testRefreshDecodesAgentContractCapabilities() async throws {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/info" {
                let (data, response) = try Self.nodeInfoResponse(
                    for: request,
                    serviceRunning: false
                )
                var object = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: data) as? [String: Any]
                )
                object["tokenity_code_revision"] = "revision-test"
                object["agent_contract"] = [
                    "version": 1,
                    "capabilities": [
                        "cluster_runtime",
                        "instance_quorum",
                        "instance_runtimes",
                        "managed_instances",
                    ],
                ]
                return (try JSONSerialization.data(withJSONObject: object), response)
            }
            throw URLError(.cannotConnectToHost)
        })

        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertTrue(store.selectedNodes.allSatisfy {
            $0.tokenityCodeRevision == "revision-test"
                && $0.agentContract?.version == 1
                && $0.agentContract?.supports("instance_runtimes") == true
        })
    }

    func testUnchangedBackgroundRefreshDoesNotPublishGlobalStoreUpdates() async throws {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/info" {
                return try Self.nodeInfoResponse(for: request, serviceRunning: false)
            }
            return Self.response(for: request, payload: #"{}"#)
        })
        await store.refreshSelectedNodeStatus(showsActivity: false)

        var publicationCount = 0
        let cancellable = store.objectWillChange.sink { publicationCount += 1 }
        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertEqual(publicationCount, 0)
        withExtendedLifetime(cancellable) {}
    }

    func testBackgroundRefreshOnlyRenewsLeaseWhileChatIsStreaming() async throws {
        var requestCount = 0
        let store = TokenityStore(dataTransport: { request in
            requestCount += 1
            if request.url?.path == "/v1/node/info" {
                return Self.response(
                    for: request,
                    payload: TokenityTestFixtures.basicNodeInfoPayload(for: request)
                )
            }
            return try await Self.successfulTransport(request)
        })
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        requestCount = 0
        store.isChatRunning = true
        var publicationCount = 0
        let cancellable = store.objectWillChange.sink { publicationCount += 1 }

        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertEqual(requestCount, store.selectedNodes.count)
        XCTAssertEqual(publicationCount, 0)
        withExtendedLifetime(cancellable) {}
    }

    func testRefreshAdoptsAndThenDetectsExternallyStoppedService() async throws {
        var serviceRunning = true
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                return try Self.nodeInfoResponse(for: request, serviceRunning: serviceRunning)
            }
            if path == "/v1/readiness" {
                guard serviceRunning else { throw URLError(.cannotConnectToHost) }
                return Self.response(
                    for: request,
                    payload: #"{"phase":"ready","ready_evidence":{"one_token_probe":true,"warmup_cache_isolated":true}}"#
                )
            }
            return Self.response(for: request, payload: #"{}"#)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.connectionMode = .ring

        await store.refreshSelectedNodeStatus()

        XCTAssertEqual(store.loadedModelName, "External-Qwen")
        XCTAssertEqual(store.phase, .running)
        XCTAssertEqual(store.menuBarSnapshot.level, .ready)

        serviceRunning = false
        await store.refreshSelectedNodeStatus()

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.phase, .readyToLoad)
        XCTAssertEqual(store.menuBarSnapshot.level, .stopped)
        XCTAssertTrue(store.modelLoadMessage.contains("outside Tokenity"))
    }

    func testExternalServiceNeverLooksLoadedWithoutVerifiedWarmupEvidence() async {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/info" {
                return try Self.nodeInfoResponse(for: request, serviceRunning: true)
            }
            if request.url?.path == "/v1/readiness" {
                return Self.response(for: request, payload: #"{"phase":"ready","ready_evidence":{}}"#)
            }
            return Self.response(for: request, payload: #"{}"#)
        })
        store.connectionMode = .ring

        await store.refreshSelectedNodeStatus()

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.modelLoadStates["External-Qwen"], .loading)
        XCTAssertEqual(store.phase, .running)
        XCTAssertEqual(store.menuBarSnapshot.level, .busy)
        XCTAssertFalse(store.isChatReady)
    }

    func testClosingLastWindowDoesNotTerminateApplication() {
        let delegate = TokenityAppDelegate()

        XCTAssertFalse(delegate.applicationShouldTerminateAfterLastWindowClosed(NSApplication.shared))
    }

    private func makeReadyStore(backendMode: BackendMode) async throws -> TokenityStore {
        let store = TokenityStore(dataTransport: Self.successfulTransport)
        if backendMode == .distributed {
            TokenityTestFixtures.bindLoopbackWorkers(on: store)
        }
        store.backendMode = backendMode
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        configureRuntime(on: store, backendMode: backendMode)
        return store
    }

    private func configureRuntime(on store: TokenityStore, backendMode: BackendMode) {
        let selected = store.selectedNodes
        let requiredCount = backendMode == .singleNode ? 1 : selected.count
        for (rank, node) in selected.enumerated() {
            guard let index = store.nodes.firstIndex(where: { $0.id == node.id }) else { continue }
            store.nodes[index].isOnline = true
            store.nodes[index].agentError = nil
            store.nodes[index].agentLatencyMilliseconds = 12
            store.nodes[index].lastAgentResponseAt = Date()
            if backendMode == .singleNode && rank > 0 {
                continue
            }
            let isController = rank == 0
            let role = isController
                ? (backendMode == .singleNode ? "single-node-openai" : "distributed-openai")
                : "distributed-openai-rank"
            store.nodes[index].roles = [
                ProcessRole(role: role, state: "running", pid: 1_000 + rank)
            ]
            store.nodes[index].clusterRuntime = ClusterRuntimeStatus(
                clusterID: "cluster-test",
                rank: rank,
                worldSize: requiredCount,
                connectionMode: store.effectiveConnectionMode.cliValue,
                role: isController ? "controller" : "worker"
            )
        }
    }

    private static func successfulTransport(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let payload: String
        if path == "/v1/models" {
            payload = #"{"data":[{"id":"Qwen3.5-122B-A10B-4bit"}]}"#
        } else if path == "/v1/chat/completions" {
            payload = "data: {\"choices\":[{\"delta\":{\"content\":\"Ready\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
        } else if path == "/v1/readiness" {
            payload = #"{"phase":"ready"}"#
        } else if path == "/v1/node/info" {
            payload = TokenityTestFixtures.modernNodeInfoPayload(for: request)
        } else if path == "/v1/node/status" {
            throw URLError(.cannotConnectToHost)
        } else {
            payload = #"{}"#
        }
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "http://127.0.0.1")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        return (Data(payload.utf8), response)
    }

    private static func nodeInfoResponse(
        for request: URLRequest,
        serviceRunning: Bool
    ) throws -> (Data, HTTPURLResponse) {
        let isController = request.url?.port == 9_100
        let role = isController ? "distributed-openai" : "distributed-openai-rank"
        let rank = isController ? 0 : 1
        let command: [String] = [
            "/runtime/python", "-m", "tokenity", "distributed-openai", "serve",
            "--model", "/models/External-Qwen", "--api-identifier", "External-Qwen",
        ]
        let rolePayload: [String: Any] = [
            "role": role,
            "state": serviceRunning ? "running" : "stopped",
            "pid": serviceRunning ? 1_234 + rank : NSNull(),
            "command": command,
        ]
        let runtime: Any = serviceRunning ? [
            "cluster_id": "external-cluster",
            "rank": rank,
            "world_size": 2,
            "connection_mode": "ring",
            "role": isController ? "controller" : "worker",
        ] : NSNull()
        let payload: [String: Any] = [
            "node_id": isController ? "mac-a" : "mac-b",
            "hostname": isController ? "controller" : "worker",
            "user": "tokenity",
            "ips": [request.url?.host ?? "127.0.0.1"],
            "architecture": "arm64",
            "macos_version": "26.0",
            "python_path": "/runtime/python",
            "mlx_version": "0.31.2",
            "mlx_lm_version": "0.31.3",
            "tokenity_version": "0.1.0",
            "process_roles": [rolePayload],
            "cluster_runtime": runtime,
            "rdma": [
                "rdma_enabled": false,
                "rdma_devices": [],
                "rdma_port_state": [:],
                "rdma_errors": [],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        return (data, response)
    }

    private static func response(
        for request: URLRequest,
        payload: String
    ) -> (Data, HTTPURLResponse) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        return (Data(payload.utf8), response)
    }
}
