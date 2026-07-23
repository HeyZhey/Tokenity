import Foundation

enum TokenityTestFixtures {
    static func basicNodeInfoPayload(for request: URLRequest) -> String {
        let isWorker = request.url?.host == "192.168.5.75"
        let nodeID = isWorker ? "mac-b" : "mac-a"
        let lanIP = isWorker ? "192.168.5.75" : "192.168.5.23"
        let rdmaIP = isWorker ? "192.168.0.2" : "192.168.0.1"
        let rdmaDevice = isWorker ? "rdma_en5" : "rdma_en4"
        return """
        {"node_id":"\(nodeID)","hostname":"\(lanIP)","user":"test","ips":["\(lanIP)"],"architecture":"arm64","macos_version":"26.5.1","python_path":"/runtime/python","mlx_version":"0.31.2","mlx_lm_version":"0.31.3","tokenity_version":"0.1.0","process_roles":[],"memory":{"total_bytes":549755813888,"used_bytes":107374182400,"free_bytes":442381631488,"used_ratio":0.1953125,"in_use_bytes":8589934592,"in_use_ratio":0.015625,"reclaimable_bytes":98784247808},"rdma":{"rdma_enabled":true,"rdma_devices":["\(rdmaDevice)"],"rdma_port_state":{"\(rdmaDevice)":"active"},"thunderbolt_ip":"\(rdmaIP)","rdma_errors":[]}}
        """
    }
}
