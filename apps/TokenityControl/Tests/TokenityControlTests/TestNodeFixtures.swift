import Foundation
@testable import TokenityControl

enum TokenityTestFixtures {
    @MainActor
    static func bindLoopbackWorkers(
        on store: TokenityStore,
        includesThirdNode: Bool = false
    ) {
        let endpoints = [
            "mac-b": "http://198.51.100.75:9200",
            "mac-c": "http://203.0.113.75:9300",
        ]
        for id in includesThirdNode ? ["mac-b", "mac-c"] : ["mac-b"] {
            guard let index = store.nodes.firstIndex(where: { $0.id == id }),
                  let endpoint = endpoints[id]
            else { continue }
            store.nodes[index].agentURL = endpoint
            store.nodes[index].ips = [id == "mac-b" ? "198.51.100.75" : "203.0.113.75"]
            store.nodes[index].machineID = "machine-\(id)"
            store.nodes[index].machineIdentityVerified = true
            store.nodes[index].agentContract = modernAgentContract
            store.selectedNodeIDs.insert(id)
        }
        if let coordinator = store.nodes.firstIndex(where: { $0.id == "mac-a" }) {
            store.nodes[coordinator].machineID = "machine-mac-a"
            store.nodes[coordinator].machineIdentityVerified = true
            store.nodes[coordinator].agentContract = modernAgentContract
        }
    }

    private static let modernAgentContract = AgentContractInfo(
        version: 1,
        capabilities: [
            "agent_health",
            "cluster_runtime",
            "instance_quorum",
            "instance_runtimes",
            "managed_instances",
            "minimax_h3_video",
        ]
    )

    static func basicNodeInfoPayload(for request: URLRequest) -> String {
        let isWorker = request.url?.port == 9_200
        let nodeID = isWorker ? "mac-b" : "mac-a"
        let lanIP = isWorker ? "198.51.100.75" : "127.0.0.1"
        let rdmaIP = isWorker ? "tokenity-rdma-b.invalid" : "tokenity-rdma-a.invalid"
        let rdmaDevice = isWorker ? "rdma_en5" : "rdma_en4"
        return """
        {"node_id":"\(nodeID)","machine_id":"machine-\(nodeID)","hostname":"\(lanIP)","user":"test","ips":["\(lanIP)"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/runtime/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","process_roles":[],"memory":{"total_bytes":549755813888,"used_bytes":107374182400,"free_bytes":442381631488,"used_ratio":0.1953125,"in_use_bytes":8589934592,"in_use_ratio":0.015625,"reclaimable_bytes":98784247808},"rdma":{"rdma_enabled":true,"rdma_devices":["\(rdmaDevice)"],"rdma_port_state":{"\(rdmaDevice)":"active"},"thunderbolt_ip":"\(rdmaIP)","rdma_errors":[]}}
        """
    }

    static func modernNodeInfoPayload(for request: URLRequest) -> String {
        let base = basicNodeInfoPayload(for: request)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let capabilities = [
            "agent_health",
            "cluster_runtime",
            "instance_quorum",
            "instance_runtimes",
            "managed_instances",
            "minimax_h3_video",
        ].map { "\"\($0)\"" }.joined(separator: ",")
        return String(base.dropLast())
            + ",\"agent_contract\":{\"version\":1,\"capabilities\":[\(capabilities)]}}"
    }
}
