import XCTest
@testable import TokenityControl

@MainActor
final class TokenityStoreTests: XCTestCase {
    func testSectionsIncludeChatAndModelsInClusterGroup() {
        XCTAssertEqual(AppSection.chat.group, "Cluster")
        XCTAssertEqual(AppSection.models.group, "Cluster")
        XCTAssertTrue(AppSection.allCases.contains(.chat))
        XCTAssertTrue(AppSection.allCases.contains(.models))
        XCTAssertFalse(AppSection.allCases.contains { $0.title == "Nodes" })
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

    func testThinkingSplitterSeparatesThinkBlock() {
        let result = TokenityStore.splitThinking("<think>plan first</think>Final answer")

        XCTAssertEqual(result.thinking, "plan first")
        XCTAssertEqual(result.answer, "Final answer")
    }

    func testLiveChatStreamRecordsMetrics() async {
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"checking latency\"},\"finish_reason\":null}]}")
                    continuation.yield("data: {\"choices\":[{\"delta\":{\"content\":\"Fast enough.\"},\"finish_reason\":null}]}")
                    continuation.yield("data: [DONE]")
                    continuation.finish()
                }
            }
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
