import XCTest
@testable import TokenityControl

@MainActor
final class TokenityStoreTests: XCTestCase {
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
                    agentURL: "http://192.168.5.75:9100",
                    lanIP: "192.168.5.75",
                    rdmaIP: "192.168.0.2",
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

        XCTAssertEqual(nodes.first?["agent_url"] as? String, "http://192.168.5.75:9100")
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
        let macC = store.nodes.first { $0.id == "mac-c" }

        XCTAssertEqual(store.selectedNodes.count, 2)

        if let macC {
            store.toggleNodeSelection(macC)
        }

        XCTAssertEqual(store.selectedNodes.count, 3)
        XCTAssertTrue(store.launchPreview.summary.contains { $0.title == "Selected Macs" && $0.value == "3" })
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
                    let payload = #"{"detail":[{"msg":"Extra inputs are not permitted"}]}"#
                    return (Data(payload.utf8), response)
                }
            }
            return try await Self.successfulModelTransport(request)
        })
        store.connectionMode = .ring
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
                let payload = #"{"detail":[{"msg":"Extra inputs are not permitted"}]}"#
                return (Data(payload.utf8), response)
            }
            return try await Self.successfulModelTransport(request)
        })
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

    func testRDMALoadRefreshesNodeInfoAndBlocksInactiveLink() async {
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let host = request.url?.host ?? ""
            let statusCode: Int
            let payload: String

            if path == "/v1/node/info", host == "192.168.5.23" {
                statusCode = 200
                payload = """
                {"node_id":"mac-a","hostname":"192.168.5.23","user":"apple","ips":["192.168.5.23"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/Users/Shared/TokenityRuntime/current/.venv/bin/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","process_roles":[],"memory":{"total_bytes":1,"used_bytes":0,"free_bytes":1,"used_ratio":0},"rdma":{"rdma_enabled":false,"rdma_devices":["rdma_en4"],"rdma_port_state":{"rdma_en4":"down"},"thunderbolt_ip":"192.168.0.1","rdma_errors":["RDMA devices were found, but no active RDMA port was detected."]}}
                """
            } else if path == "/v1/node/info", host == "192.168.5.75" {
                statusCode = 200
                payload = """
                {"node_id":"mac-b","hostname":"192.168.5.75","user":"probriefing","ips":["192.168.5.75"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/Users/Shared/TokenityRuntime/current/.venv/bin/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","process_roles":[],"memory":{"total_bytes":1,"used_bytes":0,"free_bytes":1,"used_ratio":0},"rdma":{"rdma_enabled":true,"rdma_devices":["rdma_en5"],"rdma_port_state":{"rdma_en5":"active"},"thunderbolt_ip":"192.168.0.2","rdma_errors":[]}}
                """
            } else if path.contains("/v1/node/start") {
                XCTFail("RDMA load should be blocked before starting the backend.")
                statusCode = 500
                payload = #"{"error":{"message":"unexpected start"}}"#
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

        store.createCluster()
        await store.loadModel(first)

        XCTAssertNil(store.loadedModelName)
        XCTAssertTrue(store.modelLoadMessage.contains("Thunderbolt RDMA is not ready"))
        XCTAssertTrue(store.modelLoadMessage.contains("no active RDMA port"))
        XCTAssertTrue(store.launchPreview.readinessIssues.contains { $0.contains("no active RDMA port") })
    }

    func testModelLoadShowsIncompatibleAgentMessage() async {
        let store = TokenityStore(dataTransport: { request in
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "http://127.0.0.1")!,
                statusCode: 404,
                httpVersion: nil,
                headerFields: nil
            )!
            let payload = #"{"error":{"message":"Unknown path: /v1/node/start-official-mlx-lm"}}"#
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
        XCTAssertEqual(store.modelLoadStates[first.id], .notLoaded)
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
                statusCode = 200
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
        XCTAssertEqual(store.modelLoadStates[first.id], .notLoaded)
        XCTAssertTrue(store.modelLoadMessage.contains("Changing queue pair"))
    }

    func testStopDuringLoadingCancelsTheRequestAndCleansEverySelectedNode() async {
        let startSeen = expectation(description: "backend start began")
        var stopAllRequests = 0
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            let payload: String
            if path == "/v1/node/stop-all" {
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

    func testApplicationTerminationCleanupStopsAllSelectedNodes() async {
        var stopAllHosts: Set<String> = []
        let store = TokenityStore(dataTransport: { request in
            if request.url?.path == "/v1/node/stop-all", let host = request.url?.host {
                stopAllHosts.insert(host)
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

        XCTAssertEqual(store.phase, .stopped)
        XCTAssertEqual(stopAllHosts, Set(["192.168.5.23", "192.168.5.75"]))
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
    }

    func testChatFallsBackToNonStreamingWhenStreamFailsBeforeFirstToken() async throws {
        var fallbackRequest: URLRequest?
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
                AsyncThrowingStream { continuation in
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
        var streamContinuation: AsyncThrowingStream<String, Error>.Continuation?
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

    func testModelLibraryRowsExposeFormatQuantizationAndSize() {
        let store = TokenityStore()
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].models = [
                ModelEntry(
                    id: "Qwen",
                    path: "/models/Qwen",
                    format: "MLX",
                    quantization: "4-bit · group 64",
                    sizeBytes: 64 * 1_073_741_824,
                    architecture: "QwenMoeForCausalLM",
                    shardCount: 10
                )
            ]
        }

        let row = store.modelLibraryRows.first { $0.id == "Qwen" }
        XCTAssertEqual(row?.format, "MLX")
        XCTAssertEqual(row?.quantization, "4-bit · group 64")
        XCTAssertEqual(row?.sizeText, "64 GB")
        XCTAssertEqual(row?.architecture, "QwenMoeForCausalLM")
        XCTAssertEqual(row?.shardCount, 10)
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
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        var configuration = store.modelConfiguration(for: model.id)
        configuration.thinkingMode = .disabled
        configuration.useRecommendedSampling = true
        store.updateModelConfiguration(configuration, for: model.id)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
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
                        continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"\(escaped)\"},\"finish_reason\":null}]}")
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
                        continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"token\(index) \"},\"finish_reason\":null}]}")
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

        XCTAssertEqual(store.openAIAPIBaseURL, "http://192.168.5.23:8000/v1")
        XCTAssertEqual(store.externalAPIModelName, "Load a model first")
    }

    private static func successfulModelTransport(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let payload: String
        if path.contains("/v1/models") {
            payload = #"{"data":[{"id":"Qwen3.5-122B-A10B-4bit"}]}"#
        } else if path.contains("/v1/chat/completions") {
            payload = #"{"choices":[{"message":{"content":"OK"}}]}"#
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
}
