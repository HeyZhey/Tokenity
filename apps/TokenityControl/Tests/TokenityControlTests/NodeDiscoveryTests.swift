import Foundation
import XCTest
@testable import TokenityControl

final class NodeDiscoveryTests: XCTestCase {
    @MainActor
    func testClusterBuilderExcludesSavedOfflineMacsAfterDiscovery() async throws {
        let suite = "TokenityNodeDiscoveryOfflineSavedTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(
            [
                "mac-b": "http://192.0.2.11:9100",
                "mac-c": "http://192.0.2.12:9100",
            ],
            forKey: "TokenityNodeAgentEndpoints.v1"
        )
        defaults.set(
            [
                "mac-b": "machine-b",
                "mac-c": "machine-c",
            ],
            forKey: "TokenityNodeMachineIdentities.v1"
        )
        let local = try Self.endpoint(
            agentURL: "http://127.0.0.1:9100",
            machineID: "machine-local",
            hostname: "local-mac",
            user: "local-user",
            lanIP: "127.0.0.1",
            rdmaIP: "tokenity-rdma-local.invalid",
            rdmaDevice: "rdma_en4"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [local] },
            userDefaults: defaults
        )

        await store.discoverNodes()

        XCTAssertEqual(store.nodeDiscoverySummary, "Found 1 Mac(s) automatically")
        XCTAssertEqual(store.clusterBuilderNodes.map(\.id), ["mac-a"])
        XCTAssertEqual(store.selectedNodeIDs, ["mac-a"])
        XCTAssertEqual(store.nodes.first { $0.id == "mac-a" }?.source, .automatic)
        XCTAssertEqual(store.nodes.first { $0.id == "mac-b" }?.source, .saved)
        XCTAssertFalse(store.nodes.first { $0.id == "mac-b" }?.isOnline ?? true)
        XCTAssertFalse(store.nodes.first { $0.id == "mac-c" }?.isOnline ?? true)
    }

    @MainActor
    func testDiscoverySummaryCountsMacsInsteadOfDuplicateLegacyEndpoints() async throws {
        let suite = "TokenityNodeDiscoverySummaryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let modern = try Self.endpoint(
            agentURL: "http://192.0.2.10:19100",
            machineID: "machine-a",
            hostname: "192.0.2.10",
            user: "service-user",
            lanIP: "192.0.2.10",
            rdmaIP: "tokenity-rdma-a.invalid",
            rdmaDevice: "rdma_en4"
        )
        let legacy = try Self.endpoint(
            agentURL: "http://192.0.2.10:9100",
            machineID: nil,
            hostname: "192.0.2.10",
            user: "service-user",
            lanIP: "192.0.2.10",
            rdmaIP: "tokenity-rdma-a.invalid",
            rdmaDevice: "rdma_en4"
        )
        let worker = try Self.endpoint(
            agentURL: "http://192.0.2.11:9100",
            machineID: "machine-b",
            hostname: "192.0.2.11",
            user: "service-user",
            lanIP: "192.0.2.11",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [modern, legacy, worker] },
            userDefaults: defaults
        )

        await store.discoverNodes()

        XCTAssertEqual(store.nodeDiscoverySummary, "Found 2 Mac(s) automatically")
    }

    @MainActor
    func testLegacyAgentAtKnownModernMacAddressCannotCreateADuplicate() async throws {
        let legacy = try Self.endpoint(
            agentURL: "http://192.0.2.10:9100",
            machineID: nil,
            hostname: "192.0.2.10",
            user: "service-user",
            lanIP: "192.0.2.10",
            rdmaIP: "tokenity-rdma-a.invalid",
            rdmaDevice: "rdma_en4"
        )
        let store = TokenityStore(nodeDiscoveryTransport: { _ in [legacy] })
        store.nodes[0].ips = ["192.0.2.10"]
        store.nodes[0].machineID = "modern-machine-a"
        store.nodes[0].agentContract = AgentContractInfo(version: 1, capabilities: ["managed_instances"])
        let originalCount = store.nodes.count

        await store.discoverNodes()

        XCTAssertEqual(store.nodes.count, originalCount)
        XCTAssertEqual(store.nodes.filter { $0.primaryIP == "192.0.2.10" }.count, 1)
        XCTAssertEqual(store.nodes[0].machineID, "modern-machine-a")
    }

    @MainActor
    func testSavedEndpointNeverChangesToADifferentMachineIdentity() async throws {
        let suite = "TokenityNodeDiscoveryConflictTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        var response = try Self.endpoint(
            agentURL: "http://node-b.local:9100",
            machineID: "machine-b-original",
            hostname: "node-b.local",
            user: "service-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let store = TokenityStore(
            dataTransport: { request in
                let data = try JSONEncoder().encode(response.info)
                return (
                    data,
                    HTTPURLResponse(
                        url: try XCTUnwrap(request.url),
                        statusCode: 200,
                        httpVersion: nil,
                        headerFields: nil
                    )!
                )
            },
            nodeDiscoveryTransport: { _ in [] },
            userDefaults: defaults
        )

        let firstConnection = await store.connectNode(agentURL: response.agentURL)
        XCTAssertTrue(firstConnection)
        response.info.machineID = "machine-b-replacement"
        let restored = TokenityStore(
            dataTransport: { request in
                let data = try JSONEncoder().encode(response.info)
                return (
                    data,
                    HTTPURLResponse(
                        url: try XCTUnwrap(request.url),
                        statusCode: 200,
                        httpVersion: nil,
                        headerFields: nil
                    )!
                )
            },
            nodeDiscoveryTransport: { _ in [] },
            userDefaults: defaults
        )

        XCTAssertEqual(restored.nodes.first { $0.id == "mac-b" }?.machineID, "machine-b-original")
        let replacementConnection = await restored.connectNode(agentURL: response.agentURL)
        XCTAssertFalse(replacementConnection)
        XCTAssertEqual(restored.nodes.first { $0.id == "mac-b" }?.machineID, "machine-b-original")
        XCTAssertTrue(restored.nodeDiscoverySummary.contains("another Mac"))
    }

    @MainActor
    func testDiscoveryRebindsNodeByMachineIdentityAndPersistsTheNewEndpoint() async throws {
        let suite = "TokenityNodeDiscoveryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let discovery = try Self.endpoint(
            agentURL: "http://node-a.local:9100",
            machineID: "machine-node-a",
            hostname: "node-a.local",
            user: "apple",
            lanIP: "node-a.local",
            rdmaIP: "tokenity-rdma-a.invalid",
            rdmaDevice: "rdma_en4"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [discovery] },
            userDefaults: defaults
        )
        store.nodes[0].rdma = RDMAStatus(
            rdmaEnabled: true,
            rdmaDevices: ["rdma_en4"],
            rdmaPortState: ["rdma_en4": "active"],
            thunderboltIP: "tokenity-rdma-a.invalid",
            rdmaErrors: []
        )
        store.nodes[0].machineID = "machine-node-a"
        XCTAssertEqual(store.nodes.first { $0.id == "mac-a" }?.agentURL, "http://127.0.0.1:9100")

        await store.discoverNodes()

        let node = try XCTUnwrap(store.nodes.first { $0.id == "mac-a" })
        XCTAssertEqual(node.agentURL, "http://node-a.local:9100")
        XCTAssertEqual(node.primaryIP, "node-a.local")
        XCTAssertEqual(node.machineID, "machine-node-a")
        XCTAssertEqual(node.displayName, "node-a.local")
        XCTAssertTrue(node.isOnline)
        XCTAssertTrue(store.selectedNodeIDs.contains("mac-a"))
        XCTAssertEqual(store.nodes.filter { $0.rdma.thunderboltIP == "tokenity-rdma-a.invalid" }.count, 1)

        let restored = TokenityStore(
            nodeDiscoveryTransport: { _ in [] },
            userDefaults: defaults
        )
        XCTAssertEqual(restored.nodes.first { $0.id == "mac-a" }?.agentURL, "http://node-a.local:9100")
        XCTAssertEqual(restored.nodes.first { $0.id == "mac-a" }?.primaryIP, "node-a.local")
    }

    @MainActor
    func testDiscoveryPrefersTheStandardAgentWhenOneMacExposesTwoPorts() async throws {
        let suite = "TokenityNodeDiscoveryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let stable = try Self.endpoint(
            agentURL: "http://node-b.local:9100",
            machineID: "machine-node-b",
            hostname: "node-b.local",
            user: "node-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let isolatedH3 = try Self.endpoint(
            agentURL: "http://node-b.local:9200",
            machineID: "machine-node-b",
            hostname: "node-b.local",
            user: "node-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [isolatedH3, stable] },
            userDefaults: defaults
        )
        store.nodes[1].rdma = RDMAStatus(
            rdmaEnabled: true,
            rdmaDevices: ["rdma_en5"],
            rdmaPortState: ["rdma_en5": "active"],
            thunderboltIP: "tokenity-rdma-b.invalid",
            rdmaErrors: []
        )

        await store.discoverNodes()

        let node = try XCTUnwrap(store.nodes.first { $0.id == "mac-b" })
        XCTAssertEqual(node.agentURL, "http://node-b.local:9100")
        XCTAssertEqual(node.displayName, "node-b.local")
        XCTAssertEqual(store.nodes.filter { $0.machineID == "machine-node-b" }.count, 1)
        XCTAssertFalse(store.nodes.contains { $0.agentURL == "http://node-b.local:9200" })
    }

    @MainActor
    func testDiscoveryClaimsUnboundWorkerAndMigratesLoopbackPlaceholder() async throws {
        let suite = "TokenityNodeDiscoveryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(
            [
                "mac-a": "http://127.0.0.1:9100",
                "mac-b": "http://127.0.0.1:9200",
                "mac-c": "http://127.0.0.1:9300",
            ],
            forKey: "TokenityNodeAgentEndpoints.v1"
        )
        let remote = try Self.endpoint(
            agentURL: "http://node-b.local:9100",
            machineID: nil,
            hostname: "node-b.local",
            user: "service-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [remote] },
            userDefaults: defaults
        )

        XCTAssertEqual(store.nodes.first { $0.id == "mac-b" }?.agentURL, "")
        let migrated = defaults.dictionary(forKey: "TokenityNodeAgentEndpoints.v1")
            as? [String: String]
        XCTAssertNil(migrated?["mac-b"])
        XCTAssertNil(migrated?["mac-c"])

        await store.discoverNodes()

        let worker = try XCTUnwrap(store.nodes.first { $0.id == "mac-b" })
        XCTAssertEqual(worker.agentURL, "http://node-b.local:9100")
        XCTAssertEqual(worker.hostname, "node-b.local")
        XCTAssertTrue(worker.isOnline)
        XCTAssertTrue(worker.models.isEmpty)
        XCTAssertTrue(store.selectedNodeIDs.contains("mac-b"))
        let persisted = defaults.dictionary(forKey: "TokenityNodeAgentEndpoints.v1")
            as? [String: String]
        XCTAssertEqual(persisted?["mac-b"], "http://node-b.local:9100")

        let restored = TokenityStore(
            nodeDiscoveryTransport: { _ in [] },
            userDefaults: defaults
        )
        XCTAssertEqual(
            restored.nodes.first { $0.id == "mac-b" }?.agentURL,
            "http://node-b.local:9100"
        )
    }

    @MainActor
    func testManualConnectionNormalizesAndPersistsVerifiedAgentURL() async throws {
        let suite = "TokenityNodeDiscoveryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let remote = try Self.endpoint(
            agentURL: "http://node-b.local:9100",
            machineID: nil,
            hostname: "node-b.local",
            user: "service-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let payload = try JSONEncoder().encode(remote.info)
        let store = TokenityStore(
            dataTransport: { request in
                XCTAssertEqual(request.url?.absoluteString, "http://node-b.local:9100/v1/node/info")
                return (
                    payload,
                    HTTPURLResponse(
                        url: try XCTUnwrap(request.url),
                        statusCode: 200,
                        httpVersion: nil,
                        headerFields: nil
                    )!
                )
            },
            nodeDiscoveryTransport: { _ in [] },
            userDefaults: defaults
        )

        let connected = await store.connectNode(agentURL: "node-b.local:9100/")

        XCTAssertTrue(connected)
        XCTAssertEqual(
            store.nodes.first { $0.id == "mac-b" }?.agentURL,
            "http://node-b.local:9100"
        )
        XCTAssertEqual(
            (defaults.dictionary(forKey: "TokenityNodeAgentEndpoints.v1") as? [String: String])?["mac-b"],
            "http://node-b.local:9100"
        )
    }

    @MainActor
    func testRemoteMacNeverPersistsOrClaimsALoopbackEndpoint() async throws {
        let suite = "TokenityNodeDiscoveryLoopbackTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        let loopback = try Self.endpoint(
            agentURL: "http://127.0.0.1:9200",
            machineID: "machine-local-isolated",
            hostname: "local-isolated",
            user: "service-user",
            lanIP: "127.0.0.1",
            rdmaIP: "tokenity-rdma-local.invalid",
            rdmaDevice: "rdma_en4"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [loopback] },
            userDefaults: defaults
        )

        await store.discoverNodes()

        XCTAssertEqual(store.nodes.first { $0.id == "mac-b" }?.agentURL, "")
        let saved = defaults.dictionary(forKey: "TokenityNodeAgentEndpoints.v1")
            as? [String: String]
        XCTAssertNil(saved?["mac-b"])
    }

    @MainActor
    func testActiveRDMAWorkerClaimsTheDefaultSelectedRemoteSlotFirst() async throws {
        let suite = "TokenityNodeDiscoveryTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        defer { defaults.removePersistentDomain(forName: suite) }
        var inactive = try Self.endpoint(
            agentURL: "http://node-c.local:9100",
            machineID: nil,
            hostname: "node-c.local",
            user: "service-user",
            lanIP: "node-c.local",
            rdmaIP: "tokenity-rdma-c.invalid",
            rdmaDevice: "rdma_en4"
        )
        inactive.info.rdma.rdmaEnabled = false
        inactive.info.rdma.rdmaPortState = ["rdma_en4": "down"]
        let active = try Self.endpoint(
            agentURL: "http://node-b.local:9100",
            machineID: nil,
            hostname: "node-b.local",
            user: "service-user",
            lanIP: "node-b.local",
            rdmaIP: "tokenity-rdma-b.invalid",
            rdmaDevice: "rdma_en5"
        )
        let store = TokenityStore(
            nodeDiscoveryTransport: { _ in [inactive, active] },
            userDefaults: defaults
        )

        await store.discoverNodes()

        XCTAssertEqual(
            store.nodes.first { $0.id == "mac-b" }?.agentURL,
            "http://node-b.local:9100"
        )
        XCTAssertEqual(
            store.nodes.first { $0.id == "mac-c" }?.agentURL,
            "http://node-c.local:9100"
        )
        XCTAssertTrue(store.selectedNodeIDs.contains("mac-b"))
        XCTAssertFalse(store.selectedNodeIDs.contains("mac-c"))
    }

    func testCandidateOriginsKeepSavedEndpointsWhileScanningLANPorts() {
        let origins = TokenityNodeDiscovery.candidateOrigins(
            seedOrigins: [URL(string: "http://node-a.local:9100")!]
        )

        XCTAssertTrue(origins.contains("http://node-a.local:9100"))
    }

    func testLiveAgentDiscoveryWhenConfigured() async throws {
        guard let rawURL = ProcessInfo.processInfo.environment["TOKENITY_LIVE_AGENT_URL"],
              let url = URL(string: rawURL)
        else {
            throw XCTSkip("Set TOKENITY_LIVE_AGENT_URL to run the live LAN discovery probe.")
        }

        let discoveries = await TokenityNodeDiscovery.discover(seedOrigins: [url])

        XCTAssertTrue(
            discoveries.contains { discovery in
                discovery.agentURL == rawURL.trimmingCharacters(
                    in: CharacterSet(charactersIn: "/")
                )
            },
            "The configured live Node Agent was not decoded by LAN discovery."
        )
    }

    private static func endpoint(
        agentURL: String,
        machineID: String?,
        hostname: String,
        user: String,
        lanIP: String,
        rdmaIP: String,
        rdmaDevice: String
    ) throws -> DiscoveredNodeEndpoint {
        let machineIDField = machineID.map { "\"machine_id\":\"\($0)\"," } ?? ""
        let payload = """
        {"node_id":"\(user)@\(hostname)",\(machineIDField)"hostname":"\(hostname)","user":"\(user)","ips":["\(lanIP)"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/runtime/python","mlx_version":"0.32.0","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","tokenity_code_revision":"discovery-test","agent_contract":{"version":1,"capabilities":["managed_instances","instance_runtimes","instance_quorum","cluster_runtime","minimax_h3_video"]},"process_roles":[],"memory":{"total_bytes":549755813888,"used_bytes":107374182400,"free_bytes":442381631488,"used_ratio":0.1953125},"rdma":{"rdma_enabled":true,"rdma_devices":["\(rdmaDevice)"],"rdma_port_state":{"\(rdmaDevice)":"active"},"thunderbolt_ip":"\(rdmaIP)","rdma_errors":[]},"instances":[]}
        """
        return DiscoveredNodeEndpoint(
            agentURL: agentURL,
            info: try JSONDecoder().decode(NodeInfoResponse.self, from: Data(payload.utf8))
        )
    }
}
