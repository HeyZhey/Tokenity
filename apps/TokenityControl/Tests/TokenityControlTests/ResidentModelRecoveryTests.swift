import XCTest
@testable import TokenityControl

@MainActor
final class ResidentModelRecoveryTests: XCTestCase {
    func testRefreshRecoversReadyReplicasAndChatUsesStableGateway() async throws {
        let instances = [
            Self.instanceJSON(id: "replica-a", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000),
            Self.instanceJSON(id: "replica-b", model: "Qwen3.5-122B-A10B-4bit", state: "busy", port: 18_001),
        ]
        var heartbeatInstanceIDs: Set<String> = []
        var chatURL: URL?
        var chatRequestBody: Data?
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    return Self.response(
                        for: request,
                        payload: Self.nodeInfoPayload(for: request, instances: instances)
                    )
                }
                if path == "/v1/gateway/routes" {
                    return Self.response(
                        for: request,
                        payload: Self.routesPayload(instances: instances)
                    )
                }
                if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                    let instanceID = Self.instanceID(from: path)
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(instanceID: instanceID, ready: true)
                    )
                }
                if path == "/v1/node/heartbeat",
                   let body = request.httpBody,
                   let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                   let instanceID = object["instance_id"] as? String {
                    heartbeatInstanceIDs.insert(instanceID)
                    return Self.response(for: request, payload: #"{"status":"ok"}"#)
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            lineStreamTransport: { request in
                chatURL = request.url
                chatRequestBody = request.httpBody
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Recovered\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: Self.freshDefaults()
        )
        store.connectionMode = .ring

        await store.refreshSelectedNodeStatus()

        XCTAssertEqual(
            store.managedModelInstanceIDs(for: "Qwen3.5-122B-A10B-4bit"),
            Set(["replica-a", "replica-b"])
        )
        XCTAssertEqual(store.modelLoadStates["Qwen3.5-122B-A10B-4bit"], .loaded)
        XCTAssertEqual(heartbeatInstanceIDs, Set(["replica-a", "replica-b"]))
        XCTAssertEqual(store.phase, .running)
        XCTAssertTrue(store.isChatReady)
        store.setResidentInstanceAllowsAuto("replica-a", allowed: false)
        XCTAssertEqual(
            store.residentModelInstances.first { $0.id == "replica-a" }?.allowsAuto,
            false
        )
        XCTAssertEqual(
            store.residentModelInstances.first { $0.id == "replica-b" }?.allowsAuto,
            true
        )
        store.setResidentInstanceAllowsAuto("replica-b", allowed: false)
        XCTAssertTrue(store.residentModelInstances.allSatisfy { !$0.allowsAuto })
        store.setResidentInstanceAllowsAuto("replica-b", allowed: true)
        XCTAssertEqual(
            store.residentModelInstances.first { $0.id == "replica-a" }?.allowsAuto,
            false
        )
        XCTAssertEqual(
            store.residentModelInstances.first { $0.id == "replica-b" }?.allowsAuto,
            true
        )

        store.chatInput = "Use the recovered resident model."
        await store.sendChatMessage()

        XCTAssertEqual(chatURL?.host, "192.168.5.23")
        XCTAssertEqual(chatURL?.port, 9_100)
        XCTAssertEqual(store.chatMessages.last?.content, "Recovered")
        let body = try XCTUnwrap(chatRequestBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let constraints = try XCTUnwrap(object["tokenity_constraints"] as? [String: Any])
        XCTAssertEqual(constraints["allowed_instance_ids"] as? [String], ["replica-b"])
    }

    func testBackgroundRefreshPreservesLongLoadingInstanceIdentityAndLease() async throws {
        var instanceID = ""
        var started = false
        var serviceReady = false
        var heartbeatInstanceIDs: [String?] = []
        var requestedModelID = ""
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                let instances = started
                    ? [
                        Self.instanceJSON(
                            id: instanceID,
                            model: requestedModelID,
                            state: serviceReady ? "ready" : "loading_metadata",
                            port: 18_000
                        )
                    ]
                    : []
                if path == "/v1/node/info" {
                    return Self.response(
                        for: request,
                        payload: Self.nodeInfoPayload(for: request, instances: instances)
                    )
                }
                if path == "/v1/gateway/routes" {
                    return Self.response(
                        for: request,
                        payload: Self.routesPayload(instances: instances)
                    )
                }
                if path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(request.httpBody)
                    let object = try XCTUnwrap(
                        JSONSerialization.jsonObject(with: body) as? [String: Any]
                    )
                    requestedModelID = URL(
                        fileURLWithPath: try XCTUnwrap(object["model"] as? String)
                    ).lastPathComponent
                    instanceID = try XCTUnwrap(object["instance_id"] as? String)
                    started = true
                    return Self.response(
                        for: request,
                        payload: """
                        {"instance_id":"\(instanceID)","operation_id":"operation-slow","api_base_url":"http://192.168.5.23:18000/v1"}
                        """
                    )
                }
                if path == "/v1/node/heartbeat" {
                    let object = request.httpBody.flatMap {
                        try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
                    }
                    heartbeatInstanceIDs.append(object?["instance_id"] as? String)
                    return Self.response(for: request, payload: #"{"status":"ok"}"#)
                }
                if path == "/v1/models" {
                    return Self.response(
                        for: request,
                        payload: serviceReady
                            ? "{\"data\":[{\"id\":\"\(requestedModelID)\"}]}"
                            : #"{"detail":"Loading model across MLX ranks."}"#,
                        statusCode: serviceReady ? 200 : 503
                    )
                }
                if path == "/v1/readiness" {
                    return Self.response(
                        for: request,
                        payload: serviceReady
                            ? #"{"phase":"ready","progress":1.0}"#
                            : #"{"phase":"loading_model","progress":0.4}"#
                    )
                }
                if path == "/v1/node/status" {
                    return Self.response(
                        for: request,
                        payload: """
                        {"roles":[
                          {"role":"distributed-openai","state":"stopped","instance_id":"old-stopped-instance","return_code":0},
                          {"role":"distributed-openai","state":"running","instance_id":"\(instanceID)","pid":4242}
                        ]}
                        """
                    )
                }
                if path.hasSuffix("/quorum") {
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(instanceID: instanceID, ready: true)
                    )
                }
                if path == "/v1/chat/completions" {
                    return Self.response(
                        for: request,
                        payload: """
                        data: {"choices":[{"delta":{"content":"OK"},"finish_reason":null}]}

                        data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

                        data: [DONE]

                        """
                    )
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: Self.freshDefaults()
        )
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)

        let loadTask = Task { await store.loadModel(model) }
        for _ in 0..<100 where instanceID.isEmpty || store.activeModelInstanceID != instanceID {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(store.activeModelInstanceID, instanceID)

        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertEqual(store.activeModelInstanceID, instanceID)
        await store.refreshSelectedNodeStatus(showsActivity: false)
        XCTAssertTrue(heartbeatInstanceIDs.compactMap { $0 }.allSatisfy { $0 == instanceID })
        XCTAssertFalse(heartbeatInstanceIDs.contains { $0 == nil })

        serviceReady = true
        await loadTask.value

        XCTAssertEqual(
            store.modelLoadStates[model.id],
            .loaded,
            store.logs.joined(separator: "\n")
        )
        XCTAssertEqual(store.activeModelInstanceID, instanceID)
    }

    func testSecondResidentLoadPreservesNewServiceWhileRefreshSnapshotStillOmitsIt() async throws {
        let existingModelID = "Qwen3.5-122B-A10B-4bit"
        let existingInstance = Self.instanceJSON(
            id: "instance-existing",
            model: existingModelID,
            state: "ready",
            port: 18_000
        )
        var loadingInstanceID = ""
        var loadingModelID = ""
        var started = false
        var stoppedInstanceIDs: [String] = []
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    // A real background poll can race the coordinator start
                    // response and still contain only the already-ready GLM
                    // snapshot. It must not switch the active service back to
                    // the old model while the new Qwen load is being probed.
                    return Self.response(
                        for: request,
                        payload: Self.nodeInfoPayload(
                            for: request,
                            instances: [existingInstance]
                        )
                    )
                }
                if path == "/v1/gateway/routes" {
                    return Self.response(
                        for: request,
                        payload: Self.routesPayload(instances: [existingInstance])
                    )
                }
                if path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(request.httpBody)
                    let object = try XCTUnwrap(
                        JSONSerialization.jsonObject(with: body) as? [String: Any]
                    )
                    loadingModelID = URL(
                        fileURLWithPath: try XCTUnwrap(object["model"] as? String)
                    ).lastPathComponent
                    loadingInstanceID = try XCTUnwrap(object["instance_id"] as? String)
                    started = true
                    return Self.response(
                        for: request,
                        payload: """
                        {"instance_id":"\(loadingInstanceID)","operation_id":"operation-second","api_base_url":"http://192.168.5.23:18001/v1"}
                        """
                    )
                }
                if path == "/v1/models", request.url?.port == 18_001 {
                    return Self.response(
                        for: request,
                        payload: #"{"detail":"Loading model across MLX ranks."}"#,
                        statusCode: 503
                    )
                }
                if path == "/v1/readiness", request.url?.port == 18_001 {
                    return Self.response(
                        for: request,
                        payload: #"{"phase":"loading_model","progress":0.4}"#
                    )
                }
                if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/stop") {
                    stoppedInstanceIDs.append(Self.instanceID(from: path))
                    return Self.response(for: request, payload: #"{"status":"stopped"}"#)
                }
                if path.hasSuffix("/quorum") {
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(
                            instanceID: Self.instanceID(from: path),
                            ready: true
                        )
                    )
                }
                if path == "/v1/node/heartbeat" {
                    return Self.response(for: request, payload: #"{"status":"ok"}"#)
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            userDefaults: Self.freshDefaults()
        )
        store.connectionMode = .ring
        Self.addSecondModel(to: store)

        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(store.activeModelInstanceID, "instance-existing")
        let second = try XCTUnwrap(
            store.modelLibraryRows.first { $0.id == "SecondModel" }
        )

        let loadTask = Task { await store.loadModel(second) }
        for _ in 0..<100 where !started || store.activeModelInstanceID != loadingInstanceID {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(loadingModelID, second.id)
        XCTAssertEqual(store.activeModelInstanceID, loadingInstanceID)

        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertEqual(
            store.activeModelInstanceID,
            loadingInstanceID,
            "A stale resident snapshot must not redirect the new model probe to the existing model service."
        )
        loadTask.cancel()
        await loadTask.value
        XCTAssertEqual(stoppedInstanceIDs, [loadingInstanceID])
    }

    func testSecondResidentLoadIsBlockedWhenAnySelectedAgentLacksCapability() async throws {
        var startedInstances: [(id: String, model: String, port: Int)] = []
        var startCount = 0
        var stopAllCount = 0
        let incompleteCapabilities = [
            "managed_instances",
            "instance_runtimes",
            "cluster_runtime",
        ]
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                let instances = startedInstances.map {
                    Self.instanceJSON(id: $0.id, model: $0.model, state: "ready", port: $0.port)
                }
                return Self.response(
                    for: request,
                    payload: Self.nodeInfoPayload(
                        for: request,
                        instances: instances,
                        capabilities: incompleteCapabilities
                    )
                )
            }
            if path == "/v1/gateway/routes" {
                let instances = startedInstances.map {
                    Self.instanceJSON(id: $0.id, model: $0.model, state: "ready", port: $0.port)
                }
                return Self.response(for: request, payload: Self.routesPayload(instances: instances))
            }
            if path == "/v1/node/start-distributed-openai" {
                startCount += 1
                let body = try XCTUnwrap(request.httpBody)
                let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                let modelPath = try XCTUnwrap(object["model"] as? String)
                let model = URL(fileURLWithPath: modelPath).lastPathComponent
                let instanceID = try XCTUnwrap(object["instance_id"] as? String)
                let port = 18_000 + startedInstances.count
                startedInstances.append((instanceID, model, port))
                return Self.response(
                    for: request,
                    payload: "{\"instance_id\":\"\(instanceID)\",\"operation_id\":\"operation\",\"api_base_url\":\"http://192.168.5.23:\(port)/v1\"}"
                )
            }
            if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                return Self.response(
                    for: request,
                    payload: Self.quorumPayload(instanceID: Self.instanceID(from: path), ready: true)
                )
            }
            if path == "/v1/node/stop-all" {
                stopAllCount += 1
                return Self.response(for: request, payload: #"{}"#)
            }
            if path == "/v1/models" {
                let model = startedInstances.first(where: { $0.port == request.url?.port })?.model
                    ?? "Qwen3.5-122B-A10B-4bit"
                return Self.response(for: request, payload: "{\"data\":[{\"id\":\"\(model)\"}]}")
            }
            if path == "/v1/readiness" {
                return Self.response(for: request, payload: #"{"phase":"ready"}"#)
            }
            if path == "/v1/chat/completions" {
                return Self.response(
                    for: request,
                    payload: "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"
                )
            }
            return Self.response(for: request, payload: #"{}"#)
        }, userDefaults: Self.freshDefaults())
        store.connectionMode = .ring
        store.createCluster()
        Self.addSecondModel(to: store)
        let first = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "Qwen3.5-122B-A10B-4bit" })
        let second = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "SecondModel" })

        await store.loadModel(first)
        stopAllCount = 0
        await store.loadModel(second)

        XCTAssertEqual(startCount, 1)
        XCTAssertEqual(stopAllCount, 0)
        XCTAssertEqual(store.modelLoadStates[first.id], .loaded)
        XCTAssertNotEqual(store.modelLoadStates[second.id], .loaded)
        XCTAssertTrue(store.modelLoadMessage.contains("requires current Node Agents"))
        XCTAssertTrue(store.autoRouterHealthText.contains("legacy Node Agent"))
    }

    func testLoadingSiblingDoesNotCancelActiveChat() async throws {
        var startedInstances: [(id: String, model: String, port: Int)] = []
        var streamContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    let instances = startedInstances.map {
                        Self.instanceJSON(id: $0.id, model: $0.model, state: "ready", port: $0.port)
                    }
                    return Self.response(
                        for: request,
                        payload: Self.nodeInfoPayload(for: request, instances: instances)
                    )
                }
                if path == "/v1/gateway/routes" {
                    let instances = startedInstances.map {
                        Self.instanceJSON(id: $0.id, model: $0.model, state: "ready", port: $0.port)
                    }
                    return Self.response(for: request, payload: Self.routesPayload(instances: instances))
                }
                if path == "/v1/node/start-distributed-openai" {
                    let body = try XCTUnwrap(request.httpBody)
                    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
                    let modelPath = try XCTUnwrap(object["model"] as? String)
                    let model = URL(fileURLWithPath: modelPath).lastPathComponent
                    let instanceID = try XCTUnwrap(object["instance_id"] as? String)
                    let port = 18_000 + startedInstances.count
                    startedInstances.append((instanceID, model, port))
                    return Self.response(
                        for: request,
                        payload: "{\"instance_id\":\"\(instanceID)\",\"operation_id\":\"operation\",\"api_base_url\":\"http://192.168.5.23:\(port)/v1\"}"
                    )
                }
                if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(instanceID: Self.instanceID(from: path), ready: true)
                    )
                }
                if path == "/v1/models" {
                    let model = startedInstances.first(where: { $0.port == request.url?.port })?.model
                        ?? "Qwen3.5-122B-A10B-4bit"
                    return Self.response(for: request, payload: "{\"data\":[{\"id\":\"\(model)\"}]}")
                }
                if path == "/v1/readiness" {
                    return Self.response(for: request, payload: #"{"phase":"ready"}"#)
                }
                if path == "/v1/chat/completions" {
                    return Self.response(
                        for: request,
                        payload: "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"
                    )
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    streamContinuation = continuation
                }
            },
            userDefaults: Self.freshDefaults()
        )
        store.connectionMode = .ring
        store.createCluster()
        Self.addSecondModel(to: store)
        let first = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "Qwen3.5-122B-A10B-4bit" })
        let second = try XCTUnwrap(store.modelLibraryRows.first { $0.id == "SecondModel" })
        await store.loadModel(first)

        store.chatInput = "Keep this request alive."
        store.beginSendingChatMessage()
        for _ in 0..<100 where !store.isChatRunning || streamContinuation == nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertTrue(store.isChatRunning)

        await store.loadModel(second)

        XCTAssertTrue(store.isChatRunning)
        XCTAssertEqual(store.modelLoadStates[first.id], .loaded)
        XCTAssertEqual(store.modelLoadStates[second.id], .loaded)
        XCTAssertFalse(store.chatMessages.last?.content.contains("another model") ?? true)

        streamContinuation?.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Still running\"},\"finish_reason\":null}]}")
        streamContinuation?.yield("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}")
        streamContinuation?.yield("data: [DONE]")
        streamContinuation?.finish()
        for _ in 0..<100 where store.isChatRunning {
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertEqual(store.chatMessages.last?.content, "Still running")
    }

    func testNonActiveSiblingQuorumFailureDoesNotPoisonActiveInstance() async throws {
        let instances = [
            Self.instanceJSON(id: "instance-active", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000),
            Self.instanceJSON(id: "instance-sibling", model: "SecondModel", state: "ready", port: 18_001),
        ]
        var siblingIsHealthy = true
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                return Self.response(
                    for: request,
                    payload: Self.nodeInfoPayload(for: request, instances: instances)
                )
            }
            if path == "/v1/gateway/routes" {
                return Self.response(for: request, payload: Self.routesPayload(instances: instances))
            }
            if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                let instanceID = Self.instanceID(from: path)
                let ready = instanceID != "instance-sibling" || siblingIsHealthy
                return Self.response(
                    for: request,
                    payload: Self.quorumPayload(instanceID: instanceID, ready: ready)
                )
            }
            return Self.response(for: request, payload: #"{}"#)
        }, userDefaults: Self.freshDefaults())
        store.connectionMode = .ring
        Self.addSecondModel(to: store)

        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(store.activeModelInstanceID, "instance-active")
        XCTAssertEqual(store.modelLoadStates["SecondModel"], .loaded)

        siblingIsHealthy = false
        await store.refreshSelectedNodeStatus(showsActivity: false)

        XCTAssertEqual(store.activeModelInstanceID, "instance-active")
        XCTAssertEqual(store.modelLoadStates["Qwen3.5-122B-A10B-4bit"], .loaded)
        XCTAssertEqual(store.modelLoadStates["SecondModel"], .failed)
        XCTAssertTrue(store.isChatReady)
        XCTAssertEqual(store.serverHealth, .ready)
        let sibling = try XCTUnwrap(
            store.residentModelInstances.first { $0.id == "instance-sibling" }
        )
        XCTAssertFalse(sibling.isRoutable)
        XCTAssertFalse(sibling.isReady)
        XCTAssertEqual(sibling.displayState, "Unavailable")
    }

    func testAutoChatSendsOnlyRouterFieldsAndPersistsWhitelistedProvenance() async throws {
        let instances = [
            Self.instanceJSON(id: "route-a", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000),
            Self.instanceJSON(id: "route-b", model: "SecondModel", state: "ready", port: 18_001),
        ]
        var chatRequestBody: Data?
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    return Self.response(for: request, payload: Self.nodeInfoPayload(for: request, instances: instances))
                }
                if path == "/v1/gateway/routes" {
                    return Self.response(for: request, payload: Self.routesPayload(instances: instances))
                }
                if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(instanceID: Self.instanceID(from: path), ready: true)
                    )
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            lineStreamTransport: { request in
                chatRequestBody = request.httpBody
                return AsyncThrowingStream { continuation in
                    continuation.yield(
                        .response(
                            ChatStreamResponseMetadata(
                                statusCode: 200,
                                contentType: "text/event-stream",
                                route: ChatRouteMetadata(
                                    routedModelID: "SecondModel",
                                    modelRevision: "rev-42",
                                    instanceID: "route-b",
                                    routeReason: "quality policy favored the larger context window",
                                    confidence: 0.92,
                                    routingLatencyMilliseconds: 3.5,
                                    queueWaitMilliseconds: 1.25,
                                    requestID: "request-public"
                                )
                            )
                        )
                    )
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Routed\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: Self.freshDefaults()
        )
        store.connectionMode = .ring
        Self.addSecondModel(to: store)
        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(
            store.menuBarSnapshot.modelName,
            "2 models ready · 0 busy · Auto routing healthy"
        )
        XCTAssertEqual(store.menuBarSnapshot.routingHealth, "Auto routing healthy")
        let secondSummary = try XCTUnwrap(
            store.residentModelInstances.first { $0.modelID == "SecondModel" }
        )
        XCTAssertEqual(secondSummary.modelRevision, "revision-route-b")
        XCTAssertTrue(secondSummary.capabilities.contains("Tools"))
        XCTAssertTrue(secondSummary.capabilities.contains("JSON"))
        XCTAssertTrue(secondSummary.capabilities.contains("Text"))
        XCTAssertEqual(secondSummary.warmTTFTP50Milliseconds, 300)
        XCTAssertEqual(secondSummary.warmTTFTP95Milliseconds, 500)
        store.setChatRoutePolicy(.quality)
        store.setChatModelLocked(true)
        store.chatInput = "Choose the best resident model."

        await store.sendChatMessage()

        let body = try XCTUnwrap(chatRequestBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(object["model"] as? String, "tokenity-auto")
        XCTAssertEqual(object["tokenity_route_policy"] as? String, "quality")
        XCTAssertEqual(object["tokenity_lock_model"] as? Bool, true)
        XCTAssertNotNil(object["tokenity_session_id"] as? String)
        XCTAssertNil(object["max_tokens"])
        XCTAssertNil(object["temperature"])
        XCTAssertNil(object["chat_template_kwargs"])
        let constraints = try XCTUnwrap(object["tokenity_constraints"] as? [String: Any])
        XCTAssertEqual(
            Set(constraints["allowed_model_ids"] as? [String] ?? []),
            Set(["Qwen3.5-122B-A10B-4bit", "SecondModel"])
        )
        XCTAssertEqual(
            Set(constraints["allowed_instance_ids"] as? [String] ?? []),
            Set(["route-a", "route-b"])
        )

        let assistant = try XCTUnwrap(store.chatMessages.last)
        XCTAssertEqual(assistant.routedModelID, "SecondModel")
        XCTAssertEqual(assistant.modelRevision, "rev-42")
        XCTAssertEqual(assistant.instanceID, "route-b")
        XCTAssertEqual(assistant.routeReason, "quality policy favored the larger context window")
        XCTAssertEqual(assistant.routeConfidence, 0.92)
        XCTAssertEqual(assistant.routingLatencyMilliseconds, 3.5)
        XCTAssertEqual(assistant.queueWaitMilliseconds, 1.25)
        XCTAssertEqual(assistant.requestID, "request-public")
    }

    func testManualChatKeepsModelGenerationConfigurationAndSessionRoutingPreferences() async throws {
        let defaults = Self.freshDefaults()
        let instances = [
            Self.instanceJSON(id: "manual-a", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000)
        ]
        var chatRequestBody: Data?
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                if path == "/v1/node/info" {
                    return Self.response(for: request, payload: Self.nodeInfoPayload(for: request, instances: instances))
                }
                if path == "/v1/gateway/routes" {
                    return Self.response(for: request, payload: Self.routesPayload(instances: instances))
                }
                if path.hasPrefix("/v1/node/instances/"), path.hasSuffix("/quorum") {
                    return Self.response(
                        for: request,
                        payload: Self.quorumPayload(instanceID: "manual-a", ready: true)
                    )
                }
                return Self.response(for: request, payload: #"{}"#)
            },
            lineStreamTransport: { request in
                chatRequestBody = request.httpBody
                return AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Manual\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        store.connectionMode = .ring
        await store.refreshSelectedNodeStatus()
        var configuration = store.modelConfiguration(for: "Qwen3.5-122B-A10B-4bit")
        configuration.maximumOutputTokens = 4_321
        configuration.thinkingMode = .disabled
        store.updateModelConfiguration(configuration, for: "Qwen3.5-122B-A10B-4bit")
        store.selectChatModel("Qwen3.5-122B-A10B-4bit")
        store.setChatRoutePolicy(.fast)
        store.setChatModelLocked(true)
        store.chatInput = "Use this exact model."

        await store.sendChatMessage()

        let body = try XCTUnwrap(chatRequestBody)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual(object["model"] as? String, "Qwen3.5-122B-A10B-4bit")
        XCTAssertEqual(object["max_tokens"] as? Int, 4_321)
        XCTAssertNotNil(object["temperature"])
        XCTAssertNil(object["tokenity_route_policy"])
        XCTAssertNil(object["tokenity_session_id"])
        XCTAssertNil(object["tokenity_constraints"])

        let restored = TokenityStore(userDefaults: defaults)
        XCTAssertEqual(restored.chatSelectedModelID, "Qwen3.5-122B-A10B-4bit")
        XCTAssertEqual(restored.chatRoutePolicy, .fast)
        XCTAssertTrue(restored.locksChatModel)
    }

    func testPreciseResidentStopDoesNotUseGlobalCleanup() async throws {
        let instances = [
            Self.instanceJSON(id: "stop-a", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000),
            Self.instanceJSON(id: "stop-b", model: "SecondModel", state: "ready", port: 18_001),
        ]
        var stoppedPaths: [String] = []
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                return Self.response(for: request, payload: Self.nodeInfoPayload(for: request, instances: instances))
            }
            if path == "/v1/gateway/routes" {
                return Self.response(for: request, payload: Self.routesPayload(instances: instances))
            }
            if path.hasSuffix("/quorum") {
                return Self.response(
                    for: request,
                    payload: Self.quorumPayload(instanceID: Self.instanceID(from: path), ready: true)
                )
            }
            if request.httpMethod == "POST", path.contains("/instances/"), path.hasSuffix("/stop") {
                stoppedPaths.append(path)
                return Self.response(for: request, payload: #"{}"#)
            }
            if path == "/v1/node/stop-all" {
                XCTFail("Precise resident stop must never fall back to stop-all.")
            }
            return Self.response(for: request, payload: #"{}"#)
        }, userDefaults: Self.freshDefaults())
        store.connectionMode = .ring
        Self.addSecondModel(to: store)
        await store.refreshSelectedNodeStatus()

        await store.stopResidentModelInstance("stop-b")

        XCTAssertEqual(stoppedPaths, ["/v1/node/instances/stop-b/stop"])
        XCTAssertEqual(store.managedModelInstanceIDs(for: "Qwen3.5-122B-A10B-4bit"), Set(["stop-a"]))
        XCTAssertTrue(store.managedModelInstanceIDs(for: "SecondModel").isEmpty)
    }

    func testPreciseResidentStopTreatsMissingInstanceAsAlreadyStopped() async throws {
        let instances = [
            Self.instanceJSON(id: "missing-stop", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000)
        ]
        var stopAllCount = 0
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                return Self.response(for: request, payload: Self.nodeInfoPayload(for: request, instances: instances))
            }
            if path == "/v1/gateway/routes" {
                return Self.response(for: request, payload: Self.routesPayload(instances: instances))
            }
            if path.hasSuffix("/quorum") {
                return Self.response(
                    for: request,
                    payload: Self.quorumPayload(instanceID: "missing-stop", ready: true)
                )
            }
            if path == "/v1/node/instances/missing-stop/stop" {
                return Self.response(
                    for: request,
                    payload: #"{"detail":"not found"}"#,
                    statusCode: 404
                )
            }
            if path == "/v1/node/stop-all" {
                stopAllCount += 1
            }
            return Self.response(for: request, payload: #"{}"#)
        }, userDefaults: Self.freshDefaults())
        store.connectionMode = .ring
        await store.refreshSelectedNodeStatus()

        await store.stopResidentModelInstance("missing-stop")

        XCTAssertTrue(store.residentModelInstances.isEmpty)
        XCTAssertEqual(stopAllCount, 0)
    }

    func testRefreshPrunesMissingAndStoppedResidentRecords() async throws {
        var instances = [
            Self.instanceJSON(id: "gone-a", model: "Qwen3.5-122B-A10B-4bit", state: "ready", port: 18_000),
            Self.instanceJSON(id: "stopped-b", model: "SecondModel", state: "ready", port: 18_001),
        ]
        let store = TokenityStore(dataTransport: { request in
            let path = request.url?.path ?? ""
            if path == "/v1/node/info" {
                return Self.response(for: request, payload: Self.nodeInfoPayload(for: request, instances: instances))
            }
            if path == "/v1/gateway/routes" {
                return Self.response(for: request, payload: Self.routesPayload(instances: instances))
            }
            if path.hasSuffix("/quorum") {
                return Self.response(
                    for: request,
                    payload: Self.quorumPayload(instanceID: Self.instanceID(from: path), ready: true)
                )
            }
            return Self.response(for: request, payload: #"{}"#)
        }, userDefaults: Self.freshDefaults())
        store.connectionMode = .ring
        Self.addSecondModel(to: store)
        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(Set(store.residentModelInstances.map(\.id)), Set(["gone-a", "stopped-b"]))

        instances = [
            Self.instanceJSON(id: "stopped-b", model: "SecondModel", state: "stopped", port: 18_001)
        ]
        await store.refreshSelectedNodeStatus()

        XCTAssertTrue(store.residentModelInstances.isEmpty)
        XCTAssertEqual(store.modelLoadStates["Qwen3.5-122B-A10B-4bit"], .notLoaded)
        XCTAssertEqual(store.modelLoadStates["SecondModel"], .notLoaded)
    }

    func testGatewayRouteCapabilityContractDecodesAndLegacyFieldsRemainOptional() throws {
        let currentPayload = """
        {"data":[{"model":"ModelA","instance_id":"instance-a","model_revision":"rev-a","execution_mode":"single","state":"ready","active_request_count":0,"queue_depth":0,"api_base_url":"http://127.0.0.1:18000/v1","capabilities":{"tools":true,"json":true,"thinking":false,"modalities":["text"],"task_tags":["code"]},"warm_ttft_p50_ms":120.5,"warm_ttft_p95_ms":240.75}]}
        """
        let current = try JSONDecoder().decode(
            GatewayRoutesResponse.self,
            from: Data(currentPayload.utf8)
        ).data[0]
        XCTAssertEqual(current.modelRevision, "rev-a")
        XCTAssertEqual(current.capabilities?.tools, true)
        XCTAssertEqual(current.capabilities?.json, true)
        XCTAssertEqual(current.capabilities?.thinking, false)
        XCTAssertEqual(current.capabilities?.modalities, ["text"])
        XCTAssertEqual(current.capabilities?.taskTags, ["code"])
        XCTAssertEqual(current.warmTTFTP50Milliseconds, 120.5)
        XCTAssertEqual(current.warmTTFTP95Milliseconds, 240.75)

        let legacyPayload = """
        {"data":[{"model":"ModelA","instance_id":"instance-a","execution_mode":"single","state":"ready","active_request_count":0,"api_base_url":"http://127.0.0.1:18000/v1"}]}
        """
        let legacy = try JSONDecoder().decode(
            GatewayRoutesResponse.self,
            from: Data(legacyPayload.utf8)
        ).data[0]
        XCTAssertNil(legacy.modelRevision)
        XCTAssertNil(legacy.capabilities)
        XCTAssertNil(legacy.warmTTFTP50Milliseconds)
        XCTAssertNil(legacy.warmTTFTP95Milliseconds)
    }

    private static let requiredCapabilities = [
        "managed_instances",
        "instance_runtimes",
        "instance_quorum",
        "cluster_runtime",
    ]

    private static func addSecondModel(to store: TokenityStore) {
        for index in store.nodes.indices where store.selectedNodeIDs.contains(store.nodes[index].id) {
            store.nodes[index].models.append(
                ModelEntry(id: "SecondModel", path: "/Users/Shared/TokenityModels/SecondModel")
            )
        }
    }

    private static func nodeInfoPayload(
        for request: URLRequest,
        instances: [String],
        capabilities: [String]? = nil
    ) -> String {
        let base = TokenityTestFixtures.basicNodeInfoPayload(for: request)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let capabilityJSON = (capabilities ?? requiredCapabilities)
            .map { "\"\($0)\"" }
            .joined(separator: ",")
        return String(base.dropLast())
            + ",\"agent_contract\":{\"version\":1,\"capabilities\":[\(capabilityJSON)]}"
            + ",\"instances\":[\(instances.joined(separator: ","))]}"
    }

    private static func freshDefaults() -> UserDefaults {
        UserDefaults(suiteName: "ResidentModelRecoveryTests.\(UUID().uuidString)")!
    }

    private static func instanceJSON(id: String, model: String, state: String, port: Int) -> String {
        """
        {"instance_id":"\(id)","operation_id":"operation-\(id)","requested_model_id":"\(model)","resolved_path":"/Users/Shared/TokenityModels/\(model)","execution_mode":"single","selected_nodes":["mac-a"],"world_size":1,"connection_mode":"ring","coordinator":"mac-a","http_port":\(port),"state":"\(state)","active_request_count":0}
        """
    }

    private static func routesPayload(instances: [String]) -> String {
        let routes = instances.compactMap { json -> String? in
            guard let data = json.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let model = object["requested_model_id"] as? String,
                  let instanceID = object["instance_id"] as? String,
                  let state = object["state"] as? String,
                  let port = object["http_port"] as? Int
            else { return nil }
            return """
            {"model":"\(model)","instance_id":"\(instanceID)","model_revision":"revision-\(instanceID)","execution_mode":"single","state":"\(state)","active_request_count":0,"queue_depth":0,"api_base_url":"http://127.0.0.1:\(port)/v1","capabilities":{"tools":\(model == "SecondModel"),"json":true,"thinking":\(model != "SecondModel"),"modalities":["text"],"task_tags":["general"]},"warm_ttft_p50_ms":300.0,"warm_ttft_p95_ms":500.0}
            """
        }
        return "{\"data\":[\(routes.joined(separator: ","))]}"
    }

    private static func quorumPayload(instanceID: String, ready: Bool) -> String {
        let issues = ready ? "[]" : #"["Sibling quorum failed."]"#
        return """
        {"instance_id":"\(instanceID)","ready":\(ready),"issues":\(issues),"rank_quorum":"1/1","ranks":[]}
        """
    }

    private static func instanceID(from path: String) -> String {
        path.split(separator: "/").dropLast().last.map(String.init) ?? ""
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
