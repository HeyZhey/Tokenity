import Foundation
@testable import TokenityControl

enum TokenityTestFixtures {
    @MainActor
    static func bindLoopbackWorkers(
        on store: TokenityStore,
        includesThirdNode: Bool = false
    ) {
        let endpoints = [
            "mac-b": "http://127.0.0.1:9200",
            "mac-c": "http://127.0.0.1:9300",
        ]
        for id in includesThirdNode ? ["mac-b", "mac-c"] : ["mac-b"] {
            guard let index = store.nodes.firstIndex(where: { $0.id == id }),
                  let endpoint = endpoints[id]
            else { continue }
            store.nodes[index].agentURL = endpoint
            store.nodes[index].ips = ["127.0.0.1"]
            store.selectedNodeIDs.insert(id)
        }
    }

    static func basicNodeInfoPayload(for request: URLRequest) -> String {
        let isWorker = request.url?.port == 9_200
        let nodeID = isWorker ? "mac-b" : "mac-a"
        let lanIP = isWorker ? "127.0.0.1" : "127.0.0.1"
        let rdmaIP = isWorker ? "tokenity-rdma-b.invalid" : "tokenity-rdma-a.invalid"
        let rdmaDevice = isWorker ? "rdma_en5" : "rdma_en4"
        return """
        {"node_id":"\(nodeID)","hostname":"\(lanIP)","user":"test","ips":["\(lanIP)"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/runtime/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","process_roles":[],"memory":{"total_bytes":549755813888,"used_bytes":107374182400,"free_bytes":442381631488,"used_ratio":0.1953125,"in_use_bytes":8589934592,"in_use_ratio":0.015625,"reclaimable_bytes":98784247808},"rdma":{"rdma_enabled":true,"rdma_devices":["\(rdmaDevice)"],"rdma_port_state":{"\(rdmaDevice)":"active"},"thunderbolt_ip":"\(rdmaIP)","rdma_errors":[]}}
        """
    }
}
