import XCTest
@testable import TokenityControl

@MainActor
final class TokenityStoreTests: XCTestCase {
    func testAutomaticTopologyUsesSingleMacWithoutCollectivesAndStableMultiMacFallback() {
        let store = TokenityStore()

        XCTAssertEqual(store.effectiveBackendMode, .singleNode)
        XCTAssertEqual(store.effectiveConnectionMode, .ring)

        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].isOnline = true
            store.nodes[index].rdma.rdmaEnabled = true
            store.nodes[index].rdma.thunderboltIP = "203.0.113.\(index + 1)"
        }
        XCTAssertEqual(store.effectiveBackendMode, .distributed)
        XCTAssertEqual(store.effectiveConnectionMode, .jacclRing)

        store.connectionMode = .ring
        XCTAssertEqual(store.effectiveConnectionMode, .ring)

        store.connectionMode = .jaccl
        XCTAssertEqual(store.effectiveConnectionMode, .jacclRing)

        let workerIndex = store.nodes.firstIndex { $0.id == "mac-b" }!
        store.nodes[workerIndex].rdma.rdmaEnabled = false
        XCTAssertEqual(store.effectiveConnectionMode, .ring)

        store.nodes[workerIndex].rdma.rdmaEnabled = true
        store.nodes[workerIndex].rdma.thunderboltIP = "169.254.10.2"
        XCTAssertEqual(store.effectiveConnectionMode, .ring)
        store.rebuildLaunchPreview()
        XCTAssertTrue(store.launchPreview.readinessIssues.contains {
            $0.contains("Thunderbolt RDMA is selected")
        })
    }

    func testCreatingClusterSelectsMacsButNeverClaimsTheServerIsRunning() {
        var requestCount = 0
        let store = TokenityStore(dataTransport: { request in
            requestCount += 1
            return try await Self.successfulModelTransport(request)
        })

        store.createCluster()

        XCTAssertEqual(store.phase, .readyToLoad)
        XCTAssertEqual(store.serverHealth, .stopped)
        XCTAssertNil(store.loadedModelName)
        XCTAssertFalse(store.isChatReady)
        XCTAssertEqual(requestCount, 0)
    }

    func testPreparedClusterCanBeClosedBeforeLoadingAModel() async {
        let store = TokenityStore(dataTransport: Self.successfulModelTransport)

        store.createCluster()
        XCTAssertTrue(store.canStopCluster)

        await store.stopCluster()

        XCTAssertEqual(store.phase, .stopped)
        XCTAssertFalse(store.canStopCluster)
    }

    func testModelScanFallsBackToAgentDefaultWhenConfiguredFolderIsEmpty() async throws {
        var requestedRoots: [String?] = []
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/models" {
                let root = URLComponents(url: try XCTUnwrap(request.url), resolvingAgainstBaseURL: false)?
                    .queryItems?.first(where: { $0.name == "root" })?.value
                requestedRoots.append(root)
                let payload = root == nil
                    ? #"{"root":"/agent/models","models":[{"id":"Qwen-test","path":"/agent/models/Qwen-test"}]}"#
                    : #"{"root":"/missing","models":[]}"#
                return Self.response(for: request, payload: payload)
            }
            return try await Self.successfulModelTransport(request)
        })
        store.modelRoot = "/missing"

        await store.scanModels()

        XCTAssertEqual(requestedRoots.count, 2)
        XCTAssertEqual(requestedRoots[0], "/missing")
        XCTAssertNil(requestedRoots[1])
        XCTAssertEqual(store.modelRoot, "/agent/models")
        XCTAssertTrue(store.modelLibraryRows.contains { $0.id == "Qwen-test" })
    }

    func testModelScanSummaryCountsUniqueLanguageModelsOnly() async {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/models" {
                return Self.response(
                    for: request,
                    payload: #"{"root":"/models","models":[{"id":"Qwen","path":"/models/Qwen","model_type":"qwen3_5"},{"id":"MiniMax-H3","path":"/models/MiniMax-H3","model_type":"minimax_h3"}]}"#
                )
            }
            return try await Self.successfulModelTransport(request)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)

        await store.scanModels()

        XCTAssertTrue(store.modelScanSummary.hasPrefix("1 text model(s) across 2 selected Mac(s)"))
        XCTAssertEqual(
            store.modelLibraryRows.filter { $0.modality == .language }.map(\.id),
            ["Qwen"]
        )
    }

    func testLegacyIdentityBlocksMultiMacLaunchUntilComponentsAreUpdated() {
        let store = TokenityStore()
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        let workerIndex = store.nodes.firstIndex { $0.id == "mac-b" }!
        store.nodes[workerIndex].machineID = nil

        XCTAssertTrue(store.launchPreview.readinessIssues.contains {
            $0.contains("components need an update")
        })
    }

    func testIPAddressHostnameUsesOneSharedNonDuplicatingDisplayFormat() {
        var node = TokenityNode.samples[0]
        node.hostname = "192.0.2.10"
        node.ips = ["192.0.2.10"]

        XCTAssertEqual(node.displayName, "Mac at 192.0.2.10")
        XCTAssertEqual(node.identityDetail, "")
    }

    func testSectionsIncludeChatAndModelsInClusterGroup() {
        XCTAssertEqual(AppSection.chat.group, "Cluster")
        XCTAssertEqual(AppSection.models.group, "Cluster")
        XCTAssertTrue(AppSection.allCases.contains(.chat))
        XCTAssertTrue(AppSection.allCases.contains(.models))
        XCTAssertEqual(AppSection.api.group, "Operations")
        XCTAssertTrue(AppSection.allCases.contains(.api))
        XCTAssertFalse(AppSection.allCases.contains { $0.title == "Nodes" })
    }

    func testClusterModelRequestUsesAgentHTTPURLsWithoutSSHFields() throws {
        let request = AgentStartModelRequest(
            model: "/models/qwen",
            nodes: [
                AgentClusterNodeRequest(
                    id: "mac-b",
                    agentURL: "http://198.51.100.75:9200",
                    lanIP: "198.51.100.75",
                    rdmaIP: "tokenity-rdma-b.invalid",
                    rdmaDevices: ["rdma_en5"]
                )
            ],
            connectionMode: "jaccl",
            startingPort: 30_020,
            host: "0.0.0.0",
            port: 8_000,
            dryRun: false,
            maxTokens: 131_072,
            promptCacheSize: 4,
            prefillStepSize: 2_048,
            decodeConcurrency: 1,
            promptConcurrency: 1,
            trustRemoteCode: false
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let nodes = try XCTUnwrap(object["nodes"] as? [[String: Any]])

        XCTAssertEqual(nodes.first?["agent_url"] as? String, "http://198.51.100.75:9200")
        XCTAssertNil(nodes.first?["ssh"])
        XCTAssertNil(object["native_mtp"])

        var autoRequest = request
        autoRequest.nativeMTP = NativeMTPConfiguration(mode: .auto)
        let autoObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(autoRequest)) as? [String: Any]
        )
        let nativeMTP = try XCTUnwrap(autoObject["native_mtp"] as? [String: Any])

        XCTAssertEqual(nativeMTP["mode"] as? String, "auto")
        XCTAssertEqual(nativeMTP["max_depth"] as? Int, 1)
        XCTAssertEqual(nativeMTP["head_placement"] as? String, "replicated")
    }

    func testNativeMTPRequiredBlocksKnownMissingWeightsAndAutoWarns() {
        let store = TokenityStore()
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        for index in store.nodes.indices {
            for modelIndex in store.nodes[index].models.indices {
                store.nodes[index].models[modelIndex].nativeMTP = NativeMTPCapability(
                    status: "missing_weights",
                    modelType: "qwen3_5_moe_text",
                    declaredLayers: 1,
                    weightsPresent: false,
                    reason: "checkpoint_missing_mtp_weights",
                    message: "The checkpoint declares MTP but contains no MTP tensors.",
                    tensorFormat: "safetensors-index",
                    tensorKeyDigest: "same",
                    missingGroups: []
                )
            }
        }

        store.nativeMTPMode = .required
        XCTAssertTrue(store.launchPreview.readinessIssues.contains { $0.contains("Native MTP is required") })

        store.nativeMTPMode = .auto
        XCTAssertTrue(store.launchPreview.readinessIssues.allSatisfy { !$0.contains("Native MTP is required") })
        XCTAssertTrue(store.launchPreview.warnings.contains { $0.contains("Auto falls back") })
    }

    func testNativeMTPNodeMismatchAndUnsupportedBackendForceOff() {
        let store = TokenityStore()
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        for index in store.nodes.indices {
            for modelIndex in store.nodes[index].models.indices {
                store.nodes[index].models[modelIndex].nativeMTP = NativeMTPCapability(
                    status: "supported",
                    modelType: "qwen3_5_moe_text",
                    declaredLayers: 1,
                    weightsPresent: true,
                    reason: nil,
                    message: nil,
                    tensorFormat: "safetensors-index",
                    tensorKeyDigest: "digest-\(index)",
                    missingGroups: []
                )
            }
        }
        XCTAssertEqual(store.aggregatedNativeMTPCapability?.status, "node_mismatch")

        store.nativeMTPMode = .auto
        store.backendMode = .singleNode
        XCTAssertEqual(store.effectiveNativeMTPConfiguration.mode, .off)
        XCTAssertFalse(store.canEditNativeMTP)
    }

    func testClusterSelectionCanChangeBeforeCreation() {
        let store = TokenityStore()
        TokenityTestFixtures.bindLoopbackWorkers(on: store, includesThirdNode: true)
        let macC = store.nodes.first { $0.id == "mac-c" }

        if let macC {
            store.toggleNodeSelection(macC)
        }

        XCTAssertEqual(store.selectedNodes.count, 2)

        if let macC {
            store.toggleNodeSelection(macC)
        }

        XCTAssertEqual(store.selectedNodes.count, 3)
        XCTAssertTrue(store.launchPreview.summary.contains { $0.title == "Selected Macs" && $0.value == "3" })
    }

    func testReaddedOriginalPrimaryKeepsCurrentCoordinatorAsRankZero() async throws {
        var startHost: String?
        var rankNodeIDs: [String] = []
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/start-distributed-openai" {
                startHost = request.url?.host
                let body = try XCTUnwrap(request.httpBody)
                let object = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: body) as? [String: Any]
                )
                let nodes = try XCTUnwrap(object["nodes"] as? [[String: Any]])
                rankNodeIDs = nodes.compactMap { $0["id"] as? String }
            }
            return try await Self.successfulModelTransport(request)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.connectionMode = .ring
        let mango = try XCTUnwrap(store.nodes.first { $0.id == "mac-a" })

        store.toggleNodeSelection(mango)
        XCTAssertEqual(store.coordinatorID, "mac-b")
        store.toggleNodeSelection(mango)
        XCTAssertEqual(store.coordinatorID, "mac-b")

        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)

        XCTAssertEqual(startHost, "198.51.100.75")
        XCTAssertEqual(rankNodeIDs, ["mac-b", "mac-a"])
        XCTAssertEqual(store.loadedModelName, model.id)
    }

    func testOfflineSelectedNodesBlockClusterReadinessAndDoNotLookRuntimeReady() throws {
        let store = TokenityStore()
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.connectionMode = .ring

        let mango = try XCTUnwrap(store.nodes.first { $0.id == "mac-a" })
        XCTAssertFalse(mango.isOnline)
        XCTAssertEqual(mango.displayRuntime, "Offline")
        XCTAssertEqual(mango.memoryPercentText, "Unavailable")
        XCTAssertEqual(mango.memoryUsageText, "Node offline")
        XCTAssertTrue(store.launchPreview.readinessIssues.contains { $0.contains("Mac A Node Agent is offline") })
        XCTAssertTrue(store.launchPreview.readinessIssues.contains { $0.contains("Mac B Node Agent is offline") })
        XCTAssertTrue(store.launchPreview.networkPlan.allSatisfy { $0.readiness == "Needs attention" })
    }

    func testBackgroundRefreshPopulatesUnknownMemoryForAnOnlineNode() async throws {
        let store = TokenityStore(dataTransport: { request in
            guard request.url?.path == "/v1/node/info" else {
                return try await Self.successfulModelTransport(request)
            }
            return Self.response(
                for: request,
                payload: TokenityTestFixtures.basicNodeInfoPayload(for: request)
            )
        })
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].isOnline = true
            store.nodes[index].lastAgentResponseAt = Date().addingTimeInterval(-3)
            store.nodes[index].memory = .unknown
        }

        await store.refreshSelectedNodeStatus(showsActivity: false)

        for node in store.selectedNodes {
            XCTAssertEqual(node.memory.totalBytes, 549_755_813_888)
            XCTAssertEqual(node.memory.inUseBytes, 8_589_934_592)
            XCTAssertNotEqual(node.memoryPercentText, "Memory unknown")
        }
    }

    func testModelLoadRequiresCreatedCluster() async {
        let store = TokenityStore(dataTransport: Self.successfulModelTransport)
        store.connectionMode = .ring
        let rows = store.modelLibraryRows

        XCTAssertFalse(rows.isEmpty)
        XCTAssertEqual(rows.first?.displayName, "Qwen3.5-122B-A10B-4bit")

        if let first = rows.first {
            await store.loadModel(first)
            XCTAssertNil(store.loadedModelName)

            store.createCluster()
            await store.loadModel(first)
        }

        XCTAssertEqual(store.loadedModelName, "Qwen3.5-122B-A10B-4bit")

        if let first = store.modelLibraryRows.first {
            await store.stopModel(first)
        }
        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.phase, .readyToLoad)
    }

    func testModelLoadOmitsOffForLegacyAgentAndAutoFallsBack() async throws {
        var startBodies: [[String: Any]] = []
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path.contains("/v1/node/start") {
                let body = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: Any]
                )
                startBodies.append(body)
                if body["native_mtp"] != nil {
                    let response = HTTPURLResponse(
                        url: request.url ?? URL(string: "http://127.0.0.1")!,
                        statusCode: 422,
                        httpVersion: nil,
                        headerFields: nil
                    )!
                    let payload = #"{"detail":[{"loc":["body","native_mtp"],"msg":"Extra inputs are not permitted"}]}"#
                    return (Data(payload.utf8), response)
                }
            }
            return try await Self.successfulModelTransport(request)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.backendMode = .distributed
        store.connectionMode = .ring
        store.nativeMTPMode = .off
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        await store.loadModel(model)
        XCTAssertNotNil(store.loadedModelName)
        XCTAssertEqual(startBodies.count, 1)
        XCTAssertNil(startBodies[0]["native_mtp"])

        await store.stopModel(model)
        store.nativeMTPMode = .auto
        await store.loadModel(model)

        XCTAssertNotNil(store.loadedModelName)
        XCTAssertEqual(startBodies.count, 3)
        XCTAssertNotNil(startBodies[1]["native_mtp"])
        XCTAssertNil(startBodies[2]["native_mtp"])
        XCTAssertEqual(store.nativeMTPRuntime?.fallbackReason, "unsupported_backend")
        XCTAssertTrue(store.logs.contains { $0.contains("Auto is falling back") })
    }

    func testLegacyInstanceContractFallsBackAndChatStillStreams() async throws {
        var startBodies: [[String: Any]] = []
        var streamedChatURL: URL?
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(
                        JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: Any]
                    )
                    startBodies.append(body)
                    if body["instance_id"] != nil || body["operation_id"] != nil {
                        return Self.response(
                            for: request,
                            payload: #"{"detail":[{"loc":["body","instance_id"],"msg":"Extra inputs are not permitted"},{"loc":["body","operation_id"],"msg":"Extra inputs are not permitted"}]}"#,
                            statusCode: 422
                        )
                    }
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { request in
                streamedChatURL = request.url
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Legacy chat works\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
        )
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        await store.loadModel(model)

        XCTAssertEqual(startBodies.count, 2)
        XCTAssertNotNil(startBodies[0]["instance_id"])
        XCTAssertNotNil(startBodies[0]["operation_id"])
        XCTAssertNil(startBodies[1]["instance_id"])
        XCTAssertNil(startBodies[1]["operation_id"])
        XCTAssertEqual(store.loadedModelName, model.id)
        XCTAssertNil(store.activeModelInstanceID)
        XCTAssertTrue(store.logs.contains { $0.contains("legacy model start contract") })

        store.chatInput = "Verify legacy chat routing."
        await store.sendChatMessage()

        XCTAssertEqual(streamedChatURL?.host, "127.0.0.1")
        XCTAssertEqual(streamedChatURL?.port, 8_000)
        XCTAssertEqual(store.chatMessages.last?.content, "Legacy chat works")
        XCTAssertEqual(store.chatMessages.last?.generationState, .completed)
    }

    func testLegacyModelSwitchPollingDoesNotDisableChatAfterLoad() async throws {
        var stopAllCount = 0
        var blocksSecondCleanup = false
        var secondCleanupContinuation: CheckedContinuation<Void, Never>?
        var servedModel = "Qwen3.5-122B-A10B-4bit"

        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    var payload = TokenityTestFixtures.basicNodeInfoPayload(for: request)
                    if blocksSecondCleanup {
                        payload = payload.replacingOccurrences(
                            of: #""process_roles":[]"#,
                            with: #""process_roles":[{"role":"distributed-openai","state":"stopped"}]"#
                        )
                    }
                    return Self.response(for: request, payload: payload)
                }
                if path == "/v1/node/stop-all" {
                    stopAllCount += 1
                    if stopAllCount == 3 {
                        blocksSecondCleanup = true
                        await withCheckedContinuation { continuation in
                            secondCleanupContinuation = continuation
                        }
                    }
                    return Self.response(for: request, payload: #"{}"#)
                }
                if path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(
                        JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: Any]
                    )
                    if body["instance_id"] != nil || body["operation_id"] != nil {
                        return Self.response(
                            for: request,
                            payload: #"{"detail":[{"loc":["body","instance_id"],"msg":"Extra inputs are not permitted"},{"loc":["body","operation_id"],"msg":"Extra inputs are not permitted"}]}"#,
                            statusCode: 422
                        )
                    }
                    if let modelPath = body["model"] as? String {
                        servedModel = URL(fileURLWithPath: modelPath).lastPathComponent
                    }
                    return Self.response(for: request, payload: #"{"status":{"state":"running"}}"#)
                }
                if path == "/v1/models" {
                    return Self.response(
                        for: request,
                        payload: "{\"data\":[{\"id\":\"\(servedModel)\"}]}"
                    )
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Switched model chat works\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
        )
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.connectionMode = .ring
        store.createCluster()
        let first = try XCTUnwrap(
            store.modelLibraryRows.first { $0.id == "Qwen3.5-122B-A10B-4bit" }
        )
        await store.loadModel(first)
        XCTAssertTrue(store.isChatReady)

        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].models.append(
                ModelEntry(id: "SecondModel", path: "/fixtures/tokenity/models/SecondModel")
            )
        }
        let second = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "SecondModel" })
        let switchTask = Task { await store.loadModel(second) }
        for _ in 0..<200 where !blocksSecondCleanup {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertTrue(blocksSecondCleanup)

        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertEqual(store.phase, .running)
        XCTAssertFalse(store.logs.contains { $0.contains("stopped outside Tokenity") })
        let cleanupContinuation = try XCTUnwrap(secondCleanupContinuation)
        secondCleanupContinuation = nil
        cleanupContinuation.resume()
        await switchTask.value

        XCTAssertEqual(store.loadedModelName, second.id)
        XCTAssertEqual(store.phase, .running)
        XCTAssertTrue(store.isChatReady)

        store.chatInput = "Verify chat after switching models."
        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.last?.content, "Switched model chat works")
        XCTAssertEqual(store.chatMessages.last?.generationState, .completed)
    }

    func testNativeMTPRequiredDoesNotFallBackForLegacyAgent() async throws {
        var startBodies: [[String: Any]] = []
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path.contains("/v1/node/start") {
                let body = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: Any]
                )
                startBodies.append(body)
                let response = HTTPURLResponse(
                    url: request.url ?? URL(string: "http://127.0.0.1")!,
                    statusCode: 422,
                    httpVersion: nil,
                    headerFields: nil
                )!
                let payload = #"{"detail":[{"loc":["body","native_mtp"],"msg":"Extra inputs are not permitted"}]}"#
                return (Data(payload.utf8), response)
            }
            return try await Self.successfulModelTransport(request)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        store.backendMode = .distributed
        store.connectionMode = .ring
        for nodeIndex in store.nodes.indices {
            for modelIndex in store.nodes[nodeIndex].models.indices {
                store.nodes[nodeIndex].models[modelIndex].nativeMTP = NativeMTPCapability(
                    status: "supported",
                    modelType: "qwen3_5_text",
                    declaredLayers: 1,
                    weightsPresent: true,
                    reason: nil,
                    message: nil,
                    tensorFormat: "safetensors-index",
                    tensorKeyDigest: "same",
                    missingGroups: []
                )
            }
        }
        store.nativeMTPMode = .required
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        await store.loadModel(model)

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(startBodies.count, 1)
        XCTAssertNotNil(startBodies[0]["native_mtp"])
        XCTAssertTrue(store.modelLoadMessage.contains("Restart the installed Tokenity Node Agent"))
    }

    func testMultiMacLoadFallsBackToStandardNetworkWhenRDMALinkIsInactive() async {
        var startConnectionMode: String?
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let host = request.url?.host ?? ""
            let statusCode: Int
            let payload: String

            if path == "/v1/node/info", host == "127.0.0.1", request.url?.port == 9_100 {
                statusCode = 200
                payload = """
                {"node_id":"mac-a","machine_id":"machine-mac-a","hostname":"127.0.0.1","user":"node-a-user","ips":["127.0.0.1"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/fixtures/tokenity/runtime/current/.venv/bin/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","agent_contract":{"version":1,"capabilities":["agent_health","cluster_runtime","instance_quorum","instance_runtimes","managed_instances"]},"process_roles":[],"memory":{"total_bytes":1,"used_bytes":0,"free_bytes":1,"used_ratio":0},"rdma":{"rdma_enabled":false,"rdma_devices":["rdma_en4"],"rdma_port_state":{"rdma_en4":"down"},"thunderbolt_ip":"tokenity-rdma-a.invalid","rdma_errors":["RDMA devices were found, but no active RDMA port was detected."]}}
                """
            } else if path == "/v1/node/info", host == "198.51.100.75", request.url?.port == 9_200 {
                statusCode = 200
                payload = """
                {"node_id":"mac-b","machine_id":"machine-mac-b","hostname":"198.51.100.75","user":"node-b-user","ips":["198.51.100.75"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/fixtures/tokenity/runtime/current/.venv/bin/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","agent_contract":{"version":1,"capabilities":["agent_health","cluster_runtime","instance_quorum","instance_runtimes","managed_instances"]},"process_roles":[],"memory":{"total_bytes":1,"used_bytes":0,"free_bytes":1,"used_ratio":0},"rdma":{"rdma_enabled":true,"rdma_devices":["rdma_en5"],"rdma_port_state":{"rdma_en5":"active"},"thunderbolt_ip":"tokenity-rdma-b.invalid","rdma_errors":[]}}
                """
            } else if path.contains("/v1/node/start") {
                if let body = request.httpBody,
                   let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
                    startConnectionMode = object["connection_mode"] as? String
                }
                return try await Self.successfulModelTransport(request)
            } else {
                return try await Self.successfulModelTransport(request)
            }

            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(payload.utf8), response)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }

        store.createCluster()
        await store.loadModel(first)

        XCTAssertNotNil(store.loadedModelName)
        XCTAssertEqual(startConnectionMode, "ring")
        XCTAssertEqual(store.effectiveConnectionMode, .ring)
        XCTAssertTrue(store.launchPreview.readinessIssues.allSatisfy { !$0.contains("RDMA") })
    }

    func testModelLoadShowsIncompatibleAgentMessage() async {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/info" {
                return Self.response(
                    for: request,
                    payload: TokenityTestFixtures.modernNodeInfoPayload(for: request)
                )
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: 404,
                httpVersion: nil,
                headerFields: nil
            )!
            let payload = #"{"error":{"message":"Unknown path: /v1/node/start-distributed-openai"}}"#
            return (Data(payload.utf8), response)
        })
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }

        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)

        XCTAssertNil(store.loadedModelName)
        XCTAssertTrue(store.modelLoadMessage.contains("older Node Agent"))
    }

    func testModelStaysLoadingUntilInferenceProbeCompletes() async throws {
        let probeStarted = expectation(description: "inference probe started")
        var probeRequest: URLRequest?
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/chat/completions" {
                probeRequest = request
                probeStarted.fulfill()
                try await Task.sleep(for: .milliseconds(200))
            }
            return try await Self.successfulModelTransport(request)
        })
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        let loadTask = Task { await store.loadModel(model) }
        await fulfillment(of: [probeStarted], timeout: 2)

        XCTAssertEqual(store.modelLoadStates[model.id], .loading)
        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.modelLoadProgress, 0.98)
        XCTAssertTrue(store.modelLoadMessage.contains("Verifying inference"))
        let probeBody = try XCTUnwrap(probeRequest?.httpBody)
        let probeObject = try XCTUnwrap(JSONSerialization.jsonObject(with: probeBody) as? [String: Any])
        XCTAssertEqual(probeObject["stream"] as? Bool, true)

        let controllerIndex = try XCTUnwrap(store.nodes.firstIndex { $0.id == store.coordinatorID })
        store.nodes[controllerIndex].roles = [
            ProcessRole(
                role: "distributed-openai",
                state: "running",
                pid: 4_242,
                command: [
                    "/runtime/python", "-m", "tokenity", "distributed-openai", "serve",
                    "--model", model.representativePath,
                    "--api-identifier", model.id,
                ]
            )
        ]
        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertEqual(store.modelLoadStates[model.id], .loading)
        XCTAssertNil(store.loadedModelName)

        await loadTask.value
        XCTAssertEqual(store.modelLoadStates[model.id], .loaded)
        XCTAssertEqual(store.loadedModelName, model.id)
        XCTAssertEqual(store.modelLoadProgress, 1)
    }

    func testModelProbeAcceptsCompletionUsageWithoutVisibleText() async throws {
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/chat/completions" {
                let payload = """
                data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":8,"completion_tokens":16,"total_tokens":24}}

                data: [DONE]

                """
                let response = HTTPURLResponse(
                    url: request.url ?? URL(string: "http://127.0.0.1")!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: nil
                )!
                return (Data(payload.utf8), response)
            }
            return try await Self.successfulModelTransport(request)
        })
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        await store.loadModel(model)

        XCTAssertEqual(store.modelLoadStates[model.id], .loaded)
        XCTAssertEqual(store.loadedModelName, model.id)
    }

    func testModelLoadStopsWhenBackendExits() async {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let statusCode: Int
            let payload: String
            if path.contains("/v1/node/start") {
                statusCode = 200
                payload = #"{"status":{"role":"distributed-openai","state":"running"}}"#
            } else if path.contains("/v1/models") {
                statusCode = 503
                payload = #"{"error":{"message":"Model service is still starting"}}"#
            } else if path.contains("/v1/node/status") {
                statusCode = 200
                payload = #"{"roles":[{"role":"distributed-openai","state":"stopped","return_code":1,"message":"distributed-openai exited with code 1.\nModel type qwen3_5_moe not supported."}]}"#
            } else {
                statusCode = 200
                payload = #"{}"#
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(payload.utf8), response)
        })
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }

        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.modelLoadStates[first.id], .failed)
        XCTAssertTrue(store.modelLoadMessage.contains("qwen3_5_moe"))
    }

    func testModelLoadStopsWhenReadinessFails() async {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let statusCode: Int
            let payload: String
            if path.contains("/v1/node/start") {
                statusCode = 200
                payload = #"{"status":{"role":"distributed-openai","state":"running"}}"#
            } else if path.contains("/v1/models") {
                statusCode = 503
                payload = #"{"detail":"Loading model across MLX ranks."}"#
            } else if path.contains("/v1/readiness") {
                statusCode = 503
                payload = #"{"phase":"failed","message":"Tokenity distributed runtime failed: [jaccl] Changing queue pair to RTR failed with errno 96"}"#
            } else {
                statusCode = 200
                payload = #"{"roles":[]}"#
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(payload.utf8), response)
        })
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }

        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.modelLoadStates[first.id], .failed)
        XCTAssertTrue(store.modelLoadMessage.contains("Changing queue pair"))
    }

    func testQwen3MoETwoMacTopologyIsBlockedAndOneMacRequestUsesOneRank() async throws {
        var startNodeCount: Int?
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/start-distributed-openai" {
                let body = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: Any]
                )
                startNodeCount = (body["nodes"] as? [[String: Any]])?.count
            }
            if request.url?.path == "/v1/models" {
                return Self.response(
                    for: request,
                    payload: #"{"data":[{"id":"Qwen3-30B-A3B-Instruct-2507-4bit"}]}"#
                )
            }
            return try await Self.successfulModelTransport(request)
        })
        TokenityTestFixtures.bindLoopbackWorkers(on: store)
        for nodeIndex in store.nodes.indices {
            store.nodes[nodeIndex].models = [
                ModelEntry(
                    id: "Qwen3-30B-A3B-Instruct-2507-4bit",
                    path: "/fixtures/tokenity/models/Qwen3-30B-A3B-Instruct-2507-4bit",
                    architecture: "Qwen3MoeForCausalLM",
                    modelType: "qwen3_moe",
                    distributedLoadable: false,
                    distributedLoadBlockReason: "Current runtime does not support this model in two-Mac mode. Choose Load on one Mac instead."
                )
            ]
        }
        store.connectionMode = .ring
        store.createCluster()
        let row = try XCTUnwrap(store.modelLibraryRows.first)

        XCTAssertNotNil(store.topologyIssue(for: row))
        XCTAssertTrue(store.topologyIssue(for: row)?.contains("Load on one Mac") == true)

        store.loadOnOneMac(row)
        for _ in 0..<100 where startNodeCount == nil {
            try? await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(store.backendMode, .singleNode)
        XCTAssertEqual(startNodeCount, 1)
        XCTAssertEqual(store.loadedModelName, row.id)
        XCTAssertEqual(store.modelLoadTargetSummary(for: row), "1/1 load target · Mac at 127.0.0.1")
        XCTAssertEqual(store.chatTopologyLabel, "1 Mac · Single")
        XCTAssertFalse(store.chatUsesMultipleNodes)
        XCTAssertEqual(store.activeBackendDisplayName, "Single Mac")
        XCTAssertEqual(store.activeConnectionDisplayName, "Single Mac")
    }

    func testDoubleLoadSubmissionCreatesOnlyOneStartTransaction() async throws {
        let startSeen = expectation(description: "one start transaction")
        startSeen.expectedFulfillmentCount = 1
        startSeen.assertForOverFulfill = true
        var startCount = 0
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/start-distributed-openai" {
                startCount += 1
                startSeen.fulfill()
                try await Task.sleep(for: .milliseconds(100))
            }
            return try await Self.successfulModelTransport(request)
        })
        store.connectionMode = .ring
        store.createCluster()
        let row = try XCTUnwrap(store.modelLibraryRows.first)

        store.beginLoadingModel(row)
        store.beginLoadingModel(row)
        await fulfillment(of: [startSeen], timeout: 2)
        for _ in 0..<100 where store.isModelLoading {
            try? await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertEqual(startCount, 1)
        XCTAssertTrue(store.logs.contains { $0.contains("Ignored duplicate Load") })
    }

    func testWatchdogRecoveryAndNetworkUnreachableRemainDistinct() async throws {
        let recoveringStore = TokenityStore(dataTransport: { request in
            if [9_101, 9_201].contains(request.url?.port) {
                return Self.response(
                    for: request,
                    payload: #"{"phase":"restarting","consecutive_failures":3,"restart_count":2,"last_failure_reason":"health_timeout","last_restart_time":1785942000}"#
                )
            }
            throw URLError(.timedOut)
        })
        await recoveringStore.refreshSelectedNodeStatus()

        XCTAssertTrue(recoveringStore.selectedNodes.allSatisfy { $0.agentHealthState == .restarting })
        XCTAssertTrue(recoveringStore.selectedNodes.allSatisfy { $0.watchdogRestartCount == 2 })

        let unreachableStore = TokenityStore(dataTransport: { _ in
            throw URLError(.cannotConnectToHost)
        })
        await unreachableStore.refreshSelectedNodeStatus()
        await unreachableStore.refreshSelectedNodeStatus()

        XCTAssertTrue(unreachableStore.selectedNodes.allSatisfy { $0.agentHealthState == .unreachable })
        XCTAssertTrue(unreachableStore.selectedNodes.allSatisfy {
            $0.agentHealthDetail?.contains("will not restart") == true
        })
    }

    func testStopDuringLoadingCancelsTheRequestAndCleansEverySelectedNode() async {
        let startSeen = expectation(description: "backend start began")
        var stopAllRequests = 0
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            if path == "/v1/node/info" {
                payload = TokenityTestFixtures.basicNodeInfoPayload(for: request)
            } else if path == "/v1/node/stop-all" {
                stopAllRequests += 1
                payload = #"{"statuses":[]}"#
            } else if path.contains("/v1/node/start") {
                startSeen.fulfill()
                try await Task.sleep(for: .seconds(30))
                payload = #"{"status":{"state":"running"}}"#
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
        })
        store.connectionMode = .ring
        store.createCluster()
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }

        store.beginLoadingModel(first)
        await fulfillment(of: [startSeen], timeout: 2)
        await store.stopModel(first)
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertFalse(store.isModelLoading)
        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.modelLoadMessage, "No model loaded")
        XCTAssertEqual(store.phase, .readyToLoad)
        XCTAssertGreaterThanOrEqual(stopAllRequests, store.selectedNodes.count * 2)
    }

    func testStopShowsUnloadingUntilEveryAgentConfirmsExit() async {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            if path == "/v1/node/stop-all" {
                try await Task.sleep(for: .milliseconds(200))
                payload = #"{"statuses":[]}"#
            } else {
                return try await Self.successfulModelTransport(request)
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(payload.utf8), response)
        })
        store.connectionMode = .ring
        store.createCluster()
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        await store.loadModel(first)

        let stopTask = Task { await store.stopModel(first) }
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertTrue(store.isModelUnloading)
        XCTAssertEqual(store.modelLoadStates[first.id], .unloading)
        XCTAssertTrue(store.modelLoadMessage.contains("releasing memory on all Macs"))
        XCTAssertNil(store.loadedModelName)

        await stopTask.value
        XCTAssertFalse(store.isModelUnloading)
        XCTAssertEqual(store.modelLoadStates[first.id], .notLoaded)
        XCTAssertEqual(store.modelLoadMessage, "No model loaded")
        XCTAssertEqual(store.phase, .readyToLoad)
    }

    func testMemoryStatsDecodePhysicalMemoryUsage() throws {
        let payload = #"{"total_bytes":549755813888,"used_bytes":415538003968,"free_bytes":134217809920,"used_ratio":0.756,"physical_used_bytes":415538003968,"physical_used_ratio":0.756,"in_use_bytes":7516192768,"in_use_ratio":0.0137,"reclaimable_bytes":405337620480,"wired_bytes":5368709120,"compressed_bytes":0,"anonymous_bytes":2147483648,"file_backed_bytes":405337620480,"pressure_available_ratio":0.99}"#

        let memory = try JSONDecoder().decode(MemoryStats.self, from: Data(payload.utf8))

        XCTAssertEqual(memory.usedBytes, 415_538_003_968)
        XCTAssertEqual(memory.freeBytes, 134_217_809_920)
        XCTAssertEqual(memory.usedRatio, 0.756)
        XCTAssertEqual(memory.physicalUsedBytes, 415_538_003_968)
        XCTAssertEqual(memory.physicalUsedRatio, 0.756)
        XCTAssertEqual(memory.inUseBytes, 7_516_192_768)
        XCTAssertEqual(memory.inUseRatio, 0.0137)
        XCTAssertEqual(memory.reclaimableBytes, 405_337_620_480)
        XCTAssertEqual(memory.pressureAvailableRatio, 0.99)
    }

    func testNodeMemoryLabelsStaleRuntimeMetricsInsteadOfPresentingThemAsCurrent() {
        var node = TokenityNode.samples[0]
        node.isOnline = true
        node.memory = MemoryStats(
            totalBytes: 16_000,
            usedBytes: 8_000,
            freeBytes: 8_000,
            usedRatio: 0.5,
            physicalUsedBytes: 8_000,
            physicalUsedRatio: 0.5,
            inUseBytes: 7_000,
            inUseRatio: 0.4375,
            reclaimableBytes: 1_000,
            wiredBytes: nil,
            compressedBytes: nil,
            anonymousBytes: nil,
            fileBackedBytes: nil,
            pressureAvailableRatio: nil
        )
        node.runtimeMemory = RuntimeMemoryStats(
            processResidentBytes: 5_000,
            processPhysFootprintBytes: nil,
            mlxActiveBytes: 4_000,
            mlxPeakBytes: 4_500,
            mlxCacheBytes: 500,
            modelWeightsEstimatedBytes: 3_000,
            modelResidentObservedBytes: 4_000,
            kvCacheBytes: nil,
            promptCacheBytes: nil,
            activeRequestCount: 0,
            sampledAt: 1,
            stale: true
        )

        XCTAssertTrue(node.memoryUsageText.contains("runtime metrics stale"))
        XCTAssertFalse(node.memoryUsageText.contains("MLX/model"))
    }

    func testApplicationTerminationLeavesAgentAcceptedInstancesResident() async {
        var stopAllHosts: Set<String> = []
        var heartbeatTTLs: [Double] = []
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/stop-all", let host = request.url?.host {
                stopAllHosts.insert(host)
            }
            if request.url?.path == "/v1/node/heartbeat",
               let body = request.httpBody,
               let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
               let ttl = object["ttl_seconds"] as? Double {
                heartbeatTTLs.append(ttl)
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: nil
            )!
            return (Data(#"{"statuses":[]}"#.utf8), response)
        })
        store.connectionMode = .ring
        store.createCluster()

        await store.shutdownForApplicationTermination()

        XCTAssertEqual(store.phase, .readyToLoad)
        XCTAssertTrue(stopAllHosts.isEmpty)
        // With no accepted managed instance there is no identity-safe lease to
        // renew. Recovery tests cover the resident grace heartbeat itself.
        XCTAssertTrue(heartbeatTTLs.isEmpty)
    }

    func testThinkingSplitterSeparatesThinkBlock() {
        let result = TokenityStore.splitThinking("<think>plan first</think>Final answer")

        XCTAssertEqual(result.thinking, "plan first")
        XCTAssertEqual(result.answer, "Final answer")
    }

    func testLiveChatStreamRecordsMetrics() async {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"checking latency\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Fast enough.\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "How fast are you?"

        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.filter { $0.role == .user }.count, 1)
        XCTAssertEqual(store.chatMessages.last?.role, .assistant)
        XCTAssertFalse(store.chatMessages.last?.thinking.isEmpty ?? true)
        XCTAssertNotNil(store.chatMetrics.firstTokenSeconds)
        XCTAssertNotNil(store.chatMetrics.totalSeconds)
        XCTAssertEqual(store.chatMetrics.outputTokens, 4)
    }

    func testLiveChatStreamUsesExactCompletionTokenCountFromFinalUsage() async {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"A short answer.\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":17,\"total_tokens\":26}}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "Count the output."

        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.last?.content, "A short answer.")
        XCTAssertEqual(store.chatMetrics.outputTokens, 17)
    }

    func testChatFallsBackToNonStreamingWhenStreamFailsBeforeFirstToken() async throws {
        var fallbackRequest: URLRequest?
        var streamingAttempts = 0
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/chat/completions" {
                    fallbackRequest = request
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                streamingAttempts += 1
                return AsyncThrowingStream { continuation in
                    continuation.finish(throwing: URLError(.networkConnectionLost))
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "Recover this answer."

        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.last?.role, .assistant)
        XCTAssertEqual(store.chatMessages.last?.content, "OK")
        XCTAssertFalse(store.chatMessages.last?.content.contains("not responding yet") ?? true)
        XCTAssertNotNil(store.chatMetrics.firstTokenSeconds)
        let body = try XCTUnwrap(fallbackRequest?.httpBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(object["stream"] as? Bool, false)
        XCTAssertEqual(streamingAttempts, 2)
    }

    func testFirstChatRetriesStreamingBeforeUsingNonStreamingFallback() async throws {
        var streamingAttempts = 0
        var nonStreamingRequests = 0
        var retryContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/chat/completions",
                   let body = request.httpBody,
                   let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                   object["stream"] as? Bool == false {
                    nonStreamingRequests += 1
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                streamingAttempts += 1
                if streamingAttempts == 1 {
                    return AsyncThrowingStream { continuation in
                        continuation.finish(throwing: URLError(.networkConnectionLost))
                    }
                }
                return AsyncThrowingStream { continuation in
                    retryContinuation = continuation
                }
            },
            userDefaults: defaults
        )
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        store.chatInput = "Keep this response streaming."

        let chatTask = Task { await store.sendChatMessage() }
        for _ in 0..<200 where retryContinuation == nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        let continuation = try XCTUnwrap(retryContinuation)
        continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"First part. \"},\"finish_reason\":null}]}")
        for _ in 0..<100 where store.chatMessages.last?.content != "First part. " {
            await Task.yield()
        }

        XCTAssertTrue(store.isChatRunning)
        XCTAssertEqual(store.chatMessages.last?.content, "First part. ")
        XCTAssertEqual(nonStreamingRequests, 0)

        continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Second part.\"},\"finish_reason\":null}]}")
        continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
        continuation.yield("data: [DONE]")
        continuation.finish()
        await chatTask.value

        XCTAssertEqual(streamingAttempts, 2)
        XCTAssertEqual(nonStreamingRequests, 0)
        XCTAssertEqual(store.chatMessages.last?.content, "First part. Second part.")
    }

    func testMalformedSSEDoesNotRetryOrFallBackToNonStreaming() async throws {
        var streamingAttempts = 0
        var nonStreamingRequests = 0
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/chat/completions",
                   let body = request.httpBody,
                   let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                   object["stream"] as? Bool == false {
                    nonStreamingRequests += 1
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                streamingAttempts += 1
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {not-valid-json")
                    continuation.finish()
                }
            }
        )
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        store.chatInput = "Do not hide this protocol error."

        await store.sendChatMessage()

        XCTAssertEqual(streamingAttempts, 1)
        XCTAssertEqual(nonStreamingRequests, 0)
        XCTAssertEqual(store.chatMessages.last?.generationState, .failed)
        XCTAssertFalse(store.chatMessages.last?.statusMessage?.isEmpty ?? true)
    }

    func testTwoManagedModelsCoexistAndStoppingOneLeavesSiblingRoutable() async throws {
        var stopInstancePaths: [String] = []
        var stopAllCount = 0
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    let base = TokenityTestFixtures.basicNodeInfoPayload(for: request)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let payload = String(base.dropLast())
                        + #","agent_contract":{"version":1,"capabilities":["cluster_runtime","instance_quorum","instance_runtimes","managed_instances"]},"instances":[]}"#
                    return Self.response(for: request, payload: payload)
                }
                if path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(request.httpBody)
                    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                    let modelPath = try XCTUnwrap(object["model"] as? String)
                    let isSecond = modelPath.contains("SecondModel")
                    let instanceID = isSecond ? "instance-second" : "instance-first"
                    let port = isSecond ? 18_001 : 18_000
                    return Self.response(
                        for: request,
                        payload: "{\"instance_id\":\"\(instanceID)\",\"operation_id\":\"operation-test\",\"api_base_url\":\"http://127.0.0.1:\(port)/v1\"}"
                    )
                }
                if path.hasPrefix("/v1/node/instances/") && path.hasSuffix("/quorum") {
                    let instanceID = path.split(separator: "/").dropLast().last.map(String.init) ?? ""
                    return Self.response(
                        for: request,
                        payload: "{\"instance_id\":\"\(instanceID)\",\"ready\":true,\"issues\":[],\"rank_quorum\":\"1/1\",\"ranks\":[]}"
                    )
                }
                if path.hasPrefix("/v1/node/instances/") && path.hasSuffix("/stop") {
                    stopInstancePaths.append(path)
                    return Self.response(for: request, payload: #"{"instance":{"state":"stopped"}}"#)
                }
                if path == "/v1/node/stop-all" {
                    stopAllCount += 1
                    return Self.response(for: request, payload: #"{}"#)
                }
                if path == "/v1/models" {
                    let model = request.url?.port == 18_001 ? "SecondModel" : "Qwen3.5-122B-A10B-4bit"
                    return Self.response(for: request, payload: "{\"data\":[{\"id\":\"\(model)\"}]}")
                }
                if path == "/v1/readiness" {
                    return Self.response(for: request, payload: #"{"phase":"ready"}"#)
                }
                if path == "/v1/chat/completions" {
                    return try await Self.successfulModelTransport(request)
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { request in
                XCTAssertEqual(request.url?.port, 9_100, "Managed chat must use the stable Node Agent gateway")
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Gateway OK\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
        )
        store.backendMode = .singleNode
        store.connectionMode = .ring
        store.createCluster()
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].models.append(
                ModelEntry(id: "SecondModel", path: "/fixtures/tokenity/models/SecondModel")
            )
        }
        let first = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "Qwen3.5-122B-A10B-4bit" })
        let second = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "SecondModel" })

        await store.loadModel(first)
        stopAllCount = 0
        await store.loadModel(second)

        XCTAssertEqual(store.modelLoadStates[first.id], .loaded)
        XCTAssertEqual(store.modelLoadStates[second.id], .loaded)
        XCTAssertEqual(store.loadedModelName, second.id)
        XCTAssertEqual(store.activeModelInstanceID, "instance-second")
        XCTAssertEqual(stopAllCount, 0, "A second managed load must not stop its sibling instance")

        let coordinatorIndex = try XCTUnwrap(
            store.nodes.firstIndex { $0.id == store.coordinator?.id }
        )
        store.nodes[coordinatorIndex].agentContract = AgentContractInfo(
            version: 1,
            capabilities: [
                "cluster_runtime",
                "instance_quorum",
                "instance_runtimes",
                "managed_instances",
            ]
        )
        store.nodes[coordinatorIndex].roles = [
            ProcessRole(
                role: "single-node-openai",
                state: "failed",
                instanceID: "instance-first",
                pid: nil,
                message: "Sibling stopped"
            ),
            ProcessRole(
                role: "single-node-openai",
                state: "running",
                instanceID: "instance-second",
                pid: 2_002
            ),
        ]
        let firstRuntime = ClusterRuntimeStatus(
            clusterID: "instance-first",
            instanceID: "instance-first",
            rank: 0,
            worldSize: 1,
            connectionMode: "ring",
            role: "controller"
        )
        let secondRuntime = ClusterRuntimeStatus(
            clusterID: "instance-second",
            instanceID: "instance-second",
            rank: 0,
            worldSize: 1,
            connectionMode: "ring",
            role: "controller"
        )
        store.nodes[coordinatorIndex].clusterRuntime = firstRuntime
        store.nodes[coordinatorIndex].clusterRuntimes = [firstRuntime, secondRuntime]

        XCTAssertEqual(store.menuBarSnapshot.level, .ready)
        XCTAssertEqual(
            store.menuBarSnapshot.nodes.first(where: { $0.id == store.coordinator?.id })?.inferenceRoleText,
            "single-node-openai: Running"
        )

        store.chatInput = "Route through the gateway."
        await store.sendChatMessage()
        XCTAssertEqual(store.chatMessages.last?.content, "Gateway OK")

        await store.stopModel(first)

        XCTAssertEqual(stopInstancePaths, ["/v1/node/instances/instance-first/stop"])
        XCTAssertEqual(store.modelLoadStates[first.id], .notLoaded)
        XCTAssertEqual(store.modelLoadStates[second.id], .loaded)
        XCTAssertEqual(store.loadedModelName, second.id)
        XCTAssertEqual(store.activeModelInstanceID, "instance-second")
        XCTAssertEqual(stopAllCount, 0)
    }

    func testReasoningOnlyChatDoesNotShowNotRespondingError() async {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"checking the plan\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "Continue."

        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.last?.role, .assistant)
        XCTAssertTrue(store.chatMessages.last?.thinking.contains("checking the plan") ?? false)
        XCTAssertFalse(store.chatMessages.last?.content.contains("not responding yet") ?? true)
        XCTAssertTrue(store.chatMessages.last?.content.contains("did not finish a final answer") ?? false)
    }

    func testPartialReasoningIsPreservedWhenChatStreamFails() async {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial thought\"},\"finish_reason\":null}]}")
                    continuation.finish(throwing: URLError(.timedOut))
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "Continue."

        await store.sendChatMessage()

        XCTAssertEqual(store.chatMessages.last?.role, .assistant)
        XCTAssertTrue(store.chatMessages.last?.thinking.contains("partial thought") ?? false)
        XCTAssertFalse(store.chatMessages.last?.content.contains("not responding yet") ?? true)
        XCTAssertTrue(store.chatMessages.last?.content.contains("did not finish a final answer") ?? false)
    }

    func testModelUnloadCancelsActiveChatAndIgnoresLateStreamChunks() async {
        var streamContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        var fallbackChatRequests = 0
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/chat/completions" {
                    fallbackChatRequests += 1
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    streamContinuation = continuation
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        fallbackChatRequests = 0
        store.chatInput = "Keep generating until I unload the model."

        store.beginSendingChatMessage()
        for _ in 0..<100 where !store.isChatRunning || streamContinuation == nil {
            await Task.yield()
        }
        XCTAssertTrue(store.isChatRunning)
        XCTAssertNotNil(streamContinuation)

        streamContinuation?.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial thought\"},\"finish_reason\":null}]}")
        for _ in 0..<100 where !(store.chatMessages.last?.thinking.contains("partial thought") ?? false) {
            await Task.yield()
        }
        XCTAssertTrue(store.chatMessages.last?.thinking.contains("partial thought") ?? false)

        await store.stopModel(first)
        let contentAfterUnload = store.chatMessages.last?.content
        let thinkingAfterUnload = store.chatMessages.last?.thinking

        streamContinuation?.yield("data: {\"choices\":[{\"delta\":{\"content\":\"late output that must be ignored\"},\"finish_reason\":null}]}")
        streamContinuation?.yield("data: [DONE]")
        streamContinuation?.finish()
        for _ in 0..<10 { await Task.yield() }

        XCTAssertFalse(store.isChatRunning)
        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(store.chatMessages.last?.content, contentAfterUnload)
        XCTAssertEqual(store.chatMessages.last?.thinking, thinkingAfterUnload)
        XCTAssertTrue(contentAfterUnload?.contains("model was unloaded") ?? false)
        XCTAssertEqual(store.chatMessages.suffix(2).map(\.includeInContext), [false, false])
        XCTAssertEqual(fallbackChatRequests, 0)
    }

    func testChatRequestExcludesWelcomeMessageAndUsesModelConfiguration() async throws {
        var streamedRequest: URLRequest?
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { request in
                streamedRequest = request
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Configured.\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        var configuration = store.modelConfiguration(for: first.id)
        configuration.maximumOutputTokens = 65_536
        configuration.temperature = 0.25
        configuration.topP = 0.9
        configuration.topK = 40
        configuration.minP = 0.05
        configuration.useRecommendedSampling = false
        store.updateModelConfiguration(configuration, for: first.id)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "Use the configured request."

        await store.sendChatMessage()

        let body = try XCTUnwrap(streamedRequest?.httpBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try XCTUnwrap(object["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages.first?["role"] as? String, "user")
        XCTAssertEqual(messages.first?["content"] as? String, "Use the configured request.")
        XCTAssertEqual(object["max_tokens"] as? Int, 65_536)
        XCTAssertEqual(object["temperature"] as? Double, 0.25)
        XCTAssertEqual(object["top_p"] as? Double, 0.9)
        XCTAssertEqual(object["top_k"] as? Int, 40)
        XCTAssertEqual(object["min_p"] as? Double, 0.05)
    }

    func testReasoningLengthLimitUsesAccurateMessageAndExcludesFailedTurn() async throws {
        var streamedRequests: [URLRequest] = []
        var requestCount = 0
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { request in
                streamedRequests.append(request)
                requestCount += 1
                return AsyncThrowingStream { continuation in
                    if requestCount == 1 {
                        continuation.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"long thought\"},\"finish_reason\":null}]}")
                        continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}")
                    } else {
                        continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Recovered.\"},\"finish_reason\":null}]}")
                    }
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "First turn"
        await store.sendChatMessage()

        XCTAssertTrue(store.chatMessages.last?.content.contains("maximum output length") ?? false)
        XCTAssertEqual(store.chatMessages.suffix(2).map(\.includeInContext), [false, false])

        store.chatInput = "Second turn"
        await store.sendChatMessage()

        let secondBody = try XCTUnwrap(streamedRequests.last?.httpBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: secondBody) as? [String: Any])
        let messages = try XCTUnwrap(object["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages.first?["content"] as? String, "Second turn")
    }

    func testModelConfigurationPersistsAndValidatesValues() {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let store = TokenityStore(userDefaults: defaults)
        let modelID = "Qwen"
        var configuration = ModelRuntimeConfiguration.default
        configuration.maximumOutputTokens = 999_999
        configuration.temperature = 3
        configuration.prefillStepSize = 64
        store.updateModelConfiguration(configuration, for: modelID)

        let restored = TokenityStore(userDefaults: defaults).modelConfiguration(for: modelID)
        XCTAssertEqual(restored.maximumOutputTokens, 262_144)
        XCTAssertEqual(restored.temperature, 2)
        XCTAssertEqual(restored.prefillStepSize, 128)
    }

    func testChatHistoryCreatesAndRestoresSeparateSessions() async {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"First answer\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        guard let first = store.modelLibraryRows.first else {
            XCTFail("Expected sample model")
            return
        }
        let firstSessionID = store.activeChatSessionID
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(first)
        store.chatInput = "First conversation"
        await store.sendChatMessage()

        store.newChatSession()

        XCTAssertEqual(store.chatSessions.count, 2)
        XCTAssertNotEqual(store.activeChatSessionID, firstSessionID)
        XCTAssertEqual(store.chatMessages.filter { $0.role == .user }.count, 0)

        store.selectChatSession(firstSessionID)
        XCTAssertTrue(store.chatMessages.contains { $0.content == "First conversation" })
        XCTAssertTrue(store.chatMessages.contains { $0.content == "First answer" })

        let restored = TokenityStore(userDefaults: defaults)
        XCTAssertEqual(restored.chatSessions.count, 2)
    }

    func testModelLibraryRowsExposeMetadataAndDefaultToSupportedContextLength() {
        let suiteName = "TokenityStoreTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(userDefaults: defaults)
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].models = [
                ModelEntry(
                    id: "Qwen",
                    path: "/models/Qwen",
                    format: "MLX",
                    quantization: "4-bit · group 64",
                    sizeBytes: 64 * 1_073_741_824,
                    architecture: "QwenMoeForCausalLM",
                    contextLength: 131_072,
                    shardCount: 10
                )
            ]
        }

        let row = store.modelLibraryRows.first { $0.id == "Qwen" }
        XCTAssertEqual(row?.format, "MLX")
        XCTAssertEqual(row?.quantization, "4-bit · group 64")
        XCTAssertEqual(row?.sizeText, "64 GB")
        XCTAssertEqual(row?.architecture, "QwenMoeForCausalLM")
        XCTAssertEqual(row?.contextLength, 131_072)
        XCTAssertEqual(row?.shardCount, 10)
        XCTAssertEqual(store.modelConfiguration(for: "Qwen").maximumOutputTokens, 131_072)

        var customized = store.modelConfiguration(for: "Qwen")
        customized.maximumOutputTokens = 4_096
        store.updateModelConfiguration(customized, for: "Qwen")

        XCTAssertEqual(store.modelConfiguration(for: "Qwen").maximumOutputTokens, 4_096)
    }

    func testLegacyModelConfigurationMigratesThinkingAndPenaltyDefaults() throws {
        let legacy = #"{"maximumOutputTokens":4096,"temperature":0,"topP":1,"topK":0,"minP":0,"promptCacheSize":4,"prefillStepSize":2048,"decodeConcurrency":1,"promptConcurrency":1,"trustRemoteCode":false}"#

        let configuration = try JSONDecoder().decode(
            ModelRuntimeConfiguration.self,
            from: Data(legacy.utf8)
        )

        XCTAssertEqual(configuration.thinkingMode, .automatic)
        XCTAssertTrue(configuration.useRecommendedSampling)
        XCTAssertEqual(configuration.presencePenalty, 0)
        XCTAssertEqual(configuration.repetitionPenalty, 1)
    }

    func testQwen35RecommendedNonThinkingPresetIsSentToChatAPI() async throws {
        var streamedRequest: URLRequest?
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { request in
                streamedRequest = request
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
        )
        for nodeIndex in store.nodes.indices {
            store.nodes[nodeIndex].models = [
                ModelEntry(
                    id: "Qwen3.5-122B-A10B-4bit",
                    path: "/models/Qwen3.5-122B-A10B-4bit",
                    architecture: "Qwen3_5MoeForCausalLM",
                    modelType: "qwen3_5_moe_text"
                )
            ]
        }
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        var configuration = store.modelConfiguration(for: model.id)
        configuration.thinkingMode = .disabled
        configuration.useRecommendedSampling = true
        store.updateModelConfiguration(configuration, for: model.id)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        store.selectChatModel(model.id)
        store.chatInput = "Answer directly."

        await store.sendChatMessage()

        let body = try XCTUnwrap(streamedRequest?.httpBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let template = try XCTUnwrap(object["chat_template_kwargs"] as? [String: Bool])
        XCTAssertEqual(template["enable_thinking"], false)
        XCTAssertEqual(object["temperature"] as? Double, 0.7)
        XCTAssertEqual(object["top_p"] as? Double, 0.8)
        XCTAssertEqual(object["top_k"] as? Int, 20)
        XCTAssertEqual(object["presence_penalty"] as? Double, 1.5)
        XCTAssertEqual(object["repetition_penalty"] as? Double, 1)
    }

    func testMTPDraftCheckpointCannotBeLoadedFromControlUI() async throws {
        var startRequests = 0
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path.contains("/v1/node/start") == true {
                startRequests += 1
            }
            return try await Self.successfulModelTransport(request)
        })
        for nodeIndex in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[nodeIndex].id) {
            store.nodes[nodeIndex].models = [
                ModelEntry(
                    id: "Qwen3.5-4B-MTP-4bit",
                    path: "/models/Qwen3.5-4B-MTP-4bit",
                    architecture: "Qwen3_5MTPForCausalLM"
                )
            ]
        }
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        await store.loadModel(model)

        XCTAssertNil(store.loadedModelName)
        XCTAssertEqual(startRequests, 0)
        XCTAssertTrue(store.modelLoadMessage.contains("standalone chat model"))
    }

    func testRepeatedStreamOutputStopsWithoutNonStreamingFallback() async throws {
        var fallbackRequests = 0
        let phrase = "wait while I reconsider the instruction. "
        let store = TokenityStore(
            dataTransport: { request in
                if request.url?.path == "/v1/chat/completions" {
                    fallbackRequests += 1
                }
                return try await Self.successfulModelTransport(request)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    for _ in 0..<3 {
                        let escaped = phrase.replacingOccurrences(of: "\"", with: "\\\"")
                        continuation.yield(.line("data: {\"choices\":[{\"delta\":{\"content\":\"\(escaped)\"},\"finish_reason\":null}]}"))
                    }
                    continuation.finish()
                }
            }
        )
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        fallbackRequests = 0
        store.chatInput = "Trigger the guard."

        await store.sendChatMessage()

        XCTAssertFalse(store.isChatRunning)
        XCTAssertEqual(fallbackRequests, 0)
        XCTAssertTrue(store.chatMessages.last?.content.contains("repeated output was detected") ?? false)
        XCTAssertEqual(store.chatMessages.suffix(2).map(\.includeInContext), [false, false])
    }

    func testStreamingTokensAreCoalescedIntoFewUIUpdates() async throws {
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    for index in 0..<100 {
                        continuation.yield(.line("data: {\"choices\":[{\"delta\":{\"content\":\"token\(index) \"},\"finish_reason\":null}]}"))
                    }
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
        )
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        store.chatInput = "Stream quickly."
        let revisionBefore = store.chatScrollRevision

        await store.sendChatMessage()

        XCTAssertTrue(store.chatMessages.last?.content.contains("token99") ?? false)
        XCTAssertLessThanOrEqual(store.chatScrollRevision - revisionBefore, 5)
    }

    func testExternalAPIUsesCoordinatorAddress() {
        let store = TokenityStore()

        XCTAssertEqual(store.openAIAPIBaseURL, "http://127.0.0.1:9100/v1")
        XCTAssertEqual(store.externalAPIModelName, "Load a model first")
    }

    private static func successfulModelTransport(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let payload: String
        if path == "/v1/node/info" {
            payload = TokenityTestFixtures.modernNodeInfoPayload(for: request)
        } else if path.contains("/v1/models") {
            payload = #"{"data":[{"id":"Qwen3.5-122B-A10B-4bit"}]}"#
        } else if path.contains("/v1/chat/completions") {
            let body = request.httpBody.flatMap {
                try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
            }
            if body?["stream"] as? Bool == true {
                payload = "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
            } else {
                payload = #"{"choices":[{"message":{"content":"OK"}}]}"#
            }
        } else if path.contains("/v1/node/start") {
            payload = #"{"status":{"state":"running"}}"#
        } else if path.contains("/v1/node/stop-role") {
            payload = #"{"status":{"state":"stopped"}}"#
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

    private static func response(
        for request: URLRequest,
        payload: String,
        statusCode: Int = 200
    ) -> (Data, HTTPURLResponse) {
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "http://127.0.0.1")!,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: nil
        )!
        return (Data(payload.utf8), response)
    }
}
