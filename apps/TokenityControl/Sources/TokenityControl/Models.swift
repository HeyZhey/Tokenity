import Foundation

enum AppSection: String, CaseIterable, Identifiable {
    case overview
    case cluster
    case chat
    case video
    case models
    case network
    case api
    case logs
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: return "Overview"
        case .cluster: return "Cluster"
        case .chat: return "Chat"
        case .video: return "Video"
        case .models: return "Models"
        case .network: return "Network / RDMA"
        case .api: return "API Access"
        case .logs: return "Logs"
        case .settings: return "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .overview: return "gauge.with.dots.needle.50percent"
        case .cluster: return "point.3.connected.trianglepath.dotted"
        case .chat: return "bubble.left.and.bubble.right"
        case .video: return "film.stack"
        case .models: return "cube.transparent"
        case .network: return "network"
        case .api: return "point.3.connected.trianglepath.dotted"
        case .logs: return "scroll"
        case .settings: return "gearshape"
        }
    }

    var group: String {
        switch self {
        case .overview, .cluster, .chat, .video, .models, .network: return "Cluster"
        case .api, .logs, .settings: return "Operations"
        }
    }
}

enum BackendMode: String, CaseIterable, Identifiable {
    case distributed = "Tokenity Distributed Server"
    case singleNode = "Single-Mac Tokenity Server"

    var id: String { rawValue }

    var shortName: String {
        switch self {
        case .distributed: return "Tokenity Distributed"
        case .singleNode: return "Single Mac"
        }
    }

    var detail: String {
        switch self {
        case .distributed: return "Tokenity-managed inference across the selected Macs."
        case .singleNode: return "Run inference on one selected Mac only."
        }
    }
}

enum ConnectionMode: String, CaseIterable, Identifiable {
    case ring = "Standard Network"
    case jaccl = "Thunderbolt RDMA"
    case jacclRing = "Thunderbolt + Fallback"

    var id: String { rawValue }

    var shortName: String {
        switch self {
        case .ring: return "Standard Network"
        case .jaccl: return "Thunderbolt RDMA"
        case .jacclRing: return "RDMA + Fallback"
        }
    }

    var detail: String {
        switch self {
        case .ring: return "Use the regular LAN when Thunderbolt RDMA is unavailable."
        case .jaccl: return "Use the dedicated Thunderbolt RDMA link for maximum throughput."
        case .jacclRing: return "Prefer RDMA and keep a standard-network fallback path."
        }
    }

    var cliValue: String {
        switch self {
        case .ring: return "ring"
        case .jaccl: return "jaccl"
        case .jacclRing: return "jaccl-ring"
        }
    }
}

enum ClusterPhase: String, CaseIterable {
    case stopped = "Stopped"
    case launching = "Launching"
    case distributedInit = "Distributed init"
    case loadingModel = "Loading model"
    case compiling = "Compiling"
    case firstTokenPending = "First token pending"
    case running = "Running"
    case stopping = "Stopping"
    case failed = "Failed"
}

enum ModelLoadState: String, Hashable {
    case notLoaded = "Not loaded"
    case loading = "Loading"
    case loaded = "Loaded"
    case unloading = "Unloading"
    case failed = "Failed"
}

enum ModelModality: String, Hashable {
    case language = "Language"
    case video = "Video"
}

enum ServerHealthState: Equatable {
    case stopped
    case starting(String?)
    case ready
    case error(String)

    var title: String {
        switch self {
        case .stopped: return "Stopped"
        case .starting: return "Starting"
        case .ready: return "Ready"
        case .error: return "Error"
        }
    }

    var detail: String? {
        switch self {
        case .starting(let detail): return detail
        case .error(let detail): return detail
        case .stopped, .ready: return nil
        }
    }
}

enum AgentHealthDisplayState: String, Codable, Hashable {
    case online = "Online"
    case degraded = "Degraded"
    case unreachable = "Unreachable"
    case recovering = "Recovering"
    case restarting = "Restarting"
    case restarted = "Restarted"
    case circuitOpen = "Circuit Open"
}

struct AgentCoreHealthResponse: Codable, Hashable {
    struct FatalError: Codable, Hashable {
        var stage: String?
        var message: String?
    }

    var status: String
    var agentRevision: String?
    var processStartTime: Double?
    var uptimeSeconds: Double?
    var eventLoopHeartbeatAgeSeconds: Double?
    var activeInstanceCount: Int?
    var watchdogCompatibleMonotonic: Double?
    var stateStoreReadable: Bool?
    var lastFatalInternalError: FatalError?

    enum CodingKeys: String, CodingKey {
        case status
        case agentRevision = "agent_revision"
        case processStartTime = "process_start_time"
        case uptimeSeconds = "uptime_seconds"
        case eventLoopHeartbeatAgeSeconds = "event_loop_heartbeat_age_seconds"
        case activeInstanceCount = "active_instance_count"
        case watchdogCompatibleMonotonic = "watchdog_compatible_monotonic"
        case stateStoreReadable = "state_store_readable"
        case lastFatalInternalError = "last_fatal_internal_error"
    }
}

struct AgentWatchdogStatusResponse: Codable, Hashable {
    var phase: String
    var consecutiveFailures: Int
    var restartCount: Int
    var lastFailureReason: String?
    var lastRestartTime: Double?
    var lastRecoveryDurationSeconds: Double?
    var lastSuccessTime: Double?
    var updatedAt: Double?

    enum CodingKeys: String, CodingKey {
        case phase
        case consecutiveFailures = "consecutive_failures"
        case restartCount = "restart_count"
        case lastFailureReason = "last_failure_reason"
        case lastRestartTime = "last_restart_time"
        case lastRecoveryDurationSeconds = "last_recovery_duration_seconds"
        case lastSuccessTime = "last_success_time"
        case updatedAt = "updated_at"
    }
}

enum NativeMTPMode: String, CaseIterable, Codable, Identifiable {
    case off
    case auto
    case required

    var id: String { rawValue }

    var title: String {
        switch self {
        case .off: return "Off"
        case .auto: return "Auto"
        case .required: return "Required"
        }
    }
}

struct NativeMTPConfiguration: Codable, Hashable {
    var mode: NativeMTPMode = .off
    var maxDepth: Int = 1
    var headPlacement: String = "replicated"

    enum CodingKeys: String, CodingKey {
        case mode
        case maxDepth = "max_depth"
        case headPlacement = "head_placement"
    }
}

struct NativeMTPCapability: Codable, Hashable {
    var status: String
    var modelType: String?
    var declaredLayers: Int
    var weightsPresent: Bool
    var reason: String?
    var message: String?
    var tensorFormat: String?
    var tensorKeyDigest: String?
    var missingGroups: [String]?

    enum CodingKeys: String, CodingKey {
        case status, reason, message
        case modelType = "model_type"
        case declaredLayers = "declared_layers"
        case weightsPresent = "weights_present"
        case tensorFormat = "tensor_format"
        case tensorKeyDigest = "tensor_key_digest"
        case missingGroups = "missing_groups"
    }

    var displayStatus: String {
        switch status {
        case "supported": return "Supported"
        case "missing_weights": return "Weights missing"
        case "incomplete_weights": return "Weights incomplete"
        case "unsupported": return "Unsupported"
        case "node_mismatch": return "Node mismatch"
        default: return "Unknown"
        }
    }
}

struct NativeMTPReadiness: Codable, Hashable {
    var requestedMode: String
    var enabled: Bool
    var status: String
    var effectiveMode: String?
    var fallbackReason: String?
    var message: String?
    var proposedTokens: Int?
    var acceptedTokens: Int?
    var acceptanceRate: Double?

    enum CodingKeys: String, CodingKey {
        case enabled, status, message
        case requestedMode = "requested_mode"
        case effectiveMode = "effective_mode"
        case fallbackReason = "fallback_reason"
        case proposedTokens = "proposed_tokens"
        case acceptedTokens = "accepted_tokens"
        case acceptanceRate = "acceptance_rate"
    }
}

struct RDMAStatus: Codable, Hashable {
    var rdmaEnabled: Bool
    var rdmaDevices: [String]
    var rdmaPortState: [String: String]
    var thunderboltIP: String?
    var rdmaErrors: [String]

    enum CodingKeys: String, CodingKey {
        case rdmaEnabled = "rdma_enabled"
        case rdmaDevices = "rdma_devices"
        case rdmaPortState = "rdma_port_state"
        case thunderboltIP = "thunderbolt_ip"
        case rdmaErrors = "rdma_errors"
    }

    static let empty = RDMAStatus(
        rdmaEnabled: false,
        rdmaDevices: [],
        rdmaPortState: [:],
        thunderboltIP: nil,
        rdmaErrors: []
    )
}

struct ProcessRole: Codable, Hashable, Identifiable {
    var role: String
    var state: String
    var instanceID: String? = nil
    var operationID: String? = nil
    var pid: Int?
    var returnCode: Int?
    var command: [String]?
    var logPath: String?
    var message: String?
    var logTail: String?
    var processResidentBytes: Int64? = nil
    var sampledAt: Double? = nil

    var id: String { [role, instanceID].compactMap { $0 }.joined(separator: ":") }

    enum CodingKeys: String, CodingKey {
        case role, state, pid, command, message
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case returnCode = "return_code"
        case logPath = "log_path"
        case logTail = "log_tail"
        case processResidentBytes = "process_resident_bytes"
        case sampledAt = "sampled_at"
    }
}

struct AgentModelInstanceSnapshot: Codable, Hashable, Identifiable {
    var instanceID: String
    var operationID: String?
    var requestedModelID: String
    var resolvedPath: String?
    var modelRevision: String?
    var executionMode: String?
    var selectedNodes: [String]?
    var worldSize: Int?
    var connectionMode: String?
    var coordinator: String?
    var httpPort: Int?
    var memoryReservationBytes: Int64?
    var actualMemoryBytes: Int64?
    var state: String
    var activeRequestCount: Int?
    var queuedRequestCount: Int?
    var healthReady: Bool?
    var healthIssues: [String]?
    var healthSampledAt: Double?
    var updatedAt: Double?
    var deadline: Double?
    var lastError: AgentInstanceFailure?

    var id: String { instanceID }

    enum CodingKeys: String, CodingKey {
        case state
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case requestedModelID = "requested_model_id"
        case resolvedPath = "resolved_path"
        case modelRevision = "model_revision"
        case executionMode = "execution_mode"
        case selectedNodes = "selected_nodes"
        case worldSize = "world_size"
        case connectionMode = "connection_mode"
        case coordinator
        case httpPort = "http_port"
        case memoryReservationBytes = "memory_reservation_bytes"
        case actualMemoryBytes = "actual_memory_bytes"
        case activeRequestCount = "active_request_count"
        case queuedRequestCount = "queued_request_count"
        case healthReady = "health_ready"
        case healthIssues = "health_issues"
        case healthSampledAt = "health_sampled_at"
        case updatedAt = "updated_at"
        case deadline
        case lastError = "last_error"
    }
}

struct AgentInstanceFailure: Codable, Hashable {
    var stage: String?
    var message: String?
    var issues: [String]?
    var suggestion: String?

    var displayMessage: String? {
        var parts: [String] = []
        if let stage, !stage.isEmpty {
            parts.append(stage.replacingOccurrences(of: "_", with: " "))
        }
        if let message, !message.isEmpty {
            parts.append(message)
        }
        if let issues, !issues.isEmpty {
            parts.append(issues.joined(separator: " "))
        }
        if let suggestion, !suggestion.isEmpty {
            parts.append(suggestion)
        }
        return parts.isEmpty ? nil : parts.joined(separator: ": ")
    }
}

struct AgentInstanceDetailResponse: Decodable {
    var instance: AgentModelInstanceSnapshot
    var process: ProcessRole?
}

struct GatewayModelRoute: Codable, Hashable, Identifiable {
    var model: String
    var instanceID: String
    var modelRevision: String?
    var executionMode: String?
    var state: String
    var activeRequestCount: Int?
    var queueDepth: Int?
    var apiBaseURL: String?
    var capabilities: GatewayRouteCapabilities?
    var warmTTFTP50Milliseconds: Double?
    var warmTTFTP95Milliseconds: Double?

    var id: String { instanceID }

    enum CodingKeys: String, CodingKey {
        case model, state
        case instanceID = "instance_id"
        case modelRevision = "model_revision"
        case executionMode = "execution_mode"
        case activeRequestCount = "active_request_count"
        case queueDepth = "queue_depth"
        case apiBaseURL = "api_base_url"
        case capabilities
        case warmTTFTP50Milliseconds = "warm_ttft_p50_ms"
        case warmTTFTP95Milliseconds = "warm_ttft_p95_ms"
    }
}

struct GatewayRouteCapabilities: Codable, Hashable {
    var tools: Bool
    var json: Bool
    var thinking: Bool
    var modalities: [String]
    var taskTags: [String]

    enum CodingKeys: String, CodingKey {
        case tools, json, thinking, modalities
        case taskTags = "task_tags"
    }

    var displayLabels: [String] {
        var labels = modalities.map { $0.capitalized }
        if tools { labels.append("Tools") }
        if json { labels.append("JSON") }
        if thinking { labels.append("Thinking") }
        labels.append(contentsOf: taskTags.map { "Task: \($0)" })
        return Array(Set(labels)).sorted {
            $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
        }
    }
}

struct GatewayRoutesResponse: Codable {
    var data: [GatewayModelRoute]
}

struct NodeStatusResponse: Codable {
    var roles: [ProcessRole]
    var memory: MemoryStats?
    var clusterRuntime: ClusterRuntimeStatus?
    var clusterRuntimes: [ClusterRuntimeStatus]?
    var instances: [AgentModelInstanceSnapshot]?

    enum CodingKeys: String, CodingKey {
        case roles, memory, instances
        case clusterRuntime = "cluster_runtime"
        case clusterRuntimes = "cluster_runtimes"
    }
}

struct ClusterRuntimeStatus: Codable, Hashable {
    var clusterID: String
    var instanceID: String? = nil
    var operationID: String? = nil
    var rank: Int
    var worldSize: Int
    var connectionMode: String
    var role: String
    var modelRevision: String? = nil
    var epoch: Int? = nil

    enum CodingKeys: String, CodingKey {
        case rank, role, epoch
        case clusterID = "cluster_id"
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case worldSize = "world_size"
        case connectionMode = "connection_mode"
        case modelRevision = "model_revision"
    }
}

struct ModelEntry: Codable, Hashable, Identifiable {
    var id: String
    var path: String
    var format: String? = nil
    var quantization: String? = nil
    var sizeBytes: Int64? = nil
    var architecture: String? = nil
    var shardCount: Int? = nil
    var nativeMTP: NativeMTPCapability? = nil
    var modelType: String? = nil
    var standaloneLoadable: Bool? = nil
    var loadBlockReason: String? = nil
    var distributedLoadable: Bool? = nil
    var distributedLoadBlockReason: String? = nil
    var revision: String? = nil

    enum CodingKeys: String, CodingKey {
        case id, path, format, quantization, architecture, revision
        case sizeBytes = "size_bytes"
        case shardCount = "shard_count"
        case nativeMTP = "native_mtp"
        case modelType = "model_type"
        case standaloneLoadable = "standalone_loadable"
        case loadBlockReason = "load_block_reason"
        case distributedLoadable = "distributed_loadable"
        case distributedLoadBlockReason = "distributed_load_block_reason"
    }
}

struct NodeModelsResponse: Codable {
    var root: String
    var models: [ModelEntry]
}

struct AgentContractInfo: Codable, Hashable {
    var version: Int
    var capabilities: Set<String>

    func supports(_ capability: String) -> Bool {
        capabilities.contains(capability)
    }
}

struct NodeRuntimeIdentity: Codable, Hashable {
    var runtimeID: String?
    var manifestSHA256: String?
    var payloadSHA256: String?
    var architecture: String?
    var minimumMacOS: String?
    var pythonVersion: String?
    var packages: [String: String]?
    var installState: String?

    enum CodingKeys: String, CodingKey {
        case runtimeID = "runtime_id"
        case manifestSHA256 = "manifest_sha256"
        case payloadSHA256 = "payload_sha256"
        case architecture
        case minimumMacOS = "minimum_macos"
        case pythonVersion = "python_version"
        case packages
        case installState = "install_state"
    }
}

struct NodeInfoResponse: Codable {
    var nodeID: String
    var machineID: String?
    var hostname: String
    var user: String
    var ips: [String]
    var architecture: String
    var macOSVersion: String
    var pythonPath: String
    var venv: String?
    var mlxVersion: String?
    var mlxLMVersion: String?
    var tokenityVersion: String
    var tokenityCodeRevision: String?
    var runtime: NodeRuntimeIdentity?
    var agentContract: AgentContractInfo?
    var processRoles: [ProcessRole]
    var memory: MemoryStats?
    var rdma: RDMAStatus
    var clusterRuntime: ClusterRuntimeStatus?
    var clusterRuntimes: [ClusterRuntimeStatus]?
    var instances: [AgentModelInstanceSnapshot]?

    enum CodingKeys: String, CodingKey {
        case nodeID = "node_id"
        case machineID = "machine_id"
        case hostname, user, ips, architecture, venv, rdma
        case macOSVersion = "macos_version"
        case pythonPath = "python_path"
        case mlxVersion = "mlx_version"
        case mlxLMVersion = "mlx_lm_version"
        case tokenityVersion = "tokenity_version"
        case tokenityCodeRevision = "tokenity_code_revision"
        case runtime
        case agentContract = "agent_contract"
        case processRoles = "process_roles"
        case clusterRuntime = "cluster_runtime"
        case clusterRuntimes = "cluster_runtimes"
        case memory, instances
    }
}

struct MemoryStats: Codable, Hashable {
    var totalBytes: Int64?
    var usedBytes: Int64?
    var freeBytes: Int64?
    var usedRatio: Double?
    var physicalUsedBytes: Int64?
    var physicalUsedRatio: Double?
    var inUseBytes: Int64?
    var inUseRatio: Double?
    var reclaimableBytes: Int64?
    var wiredBytes: Int64?
    var compressedBytes: Int64?
    var anonymousBytes: Int64?
    var fileBackedBytes: Int64?
    var pressureAvailableRatio: Double?

    enum CodingKeys: String, CodingKey {
        case totalBytes = "total_bytes"
        case usedBytes = "used_bytes"
        case freeBytes = "free_bytes"
        case usedRatio = "used_ratio"
        case physicalUsedBytes = "physical_used_bytes"
        case physicalUsedRatio = "physical_used_ratio"
        case inUseBytes = "in_use_bytes"
        case inUseRatio = "in_use_ratio"
        case reclaimableBytes = "reclaimable_bytes"
        case wiredBytes = "wired_bytes"
        case compressedBytes = "compressed_bytes"
        case anonymousBytes = "anonymous_bytes"
        case fileBackedBytes = "file_backed_bytes"
        case pressureAvailableRatio = "pressure_available_ratio"
    }

    static let unknown = MemoryStats(
        totalBytes: nil,
        usedBytes: nil,
        freeBytes: nil,
        usedRatio: nil,
        physicalUsedBytes: nil,
        physicalUsedRatio: nil,
        inUseBytes: nil,
        inUseRatio: nil,
        reclaimableBytes: nil,
        wiredBytes: nil,
        compressedBytes: nil,
        anonymousBytes: nil,
        fileBackedBytes: nil,
        pressureAvailableRatio: nil
    )
}

struct RuntimeMemoryStats: Codable, Hashable {
    var processResidentBytes: Int64?
    var processPhysFootprintBytes: Int64?
    var mlxActiveBytes: Int64?
    var mlxPeakBytes: Int64?
    var mlxCacheBytes: Int64?
    var modelWeightsEstimatedBytes: Int64?
    var modelResidentObservedBytes: Int64?
    var kvCacheBytes: Int64?
    var promptCacheBytes: Int64?
    var activeRequestCount: Int?
    var sampledAt: Double?
    var stale: Bool?

    enum CodingKeys: String, CodingKey {
        case processResidentBytes = "process_resident_bytes"
        case processPhysFootprintBytes = "process_phys_footprint_bytes"
        case mlxActiveBytes = "mlx_active_bytes"
        case mlxPeakBytes = "mlx_peak_bytes"
        case mlxCacheBytes = "mlx_cache_bytes"
        case modelWeightsEstimatedBytes = "model_weights_estimated_bytes"
        case modelResidentObservedBytes = "model_resident_observed_bytes"
        case kvCacheBytes = "kv_cache_bytes"
        case promptCacheBytes = "prompt_cache_bytes"
        case activeRequestCount = "active_request_count"
        case sampledAt = "sampled_at"
        case stale
    }
}

struct TokenityNode: Identifiable, Hashable {
    var id: String
    var hostname: String
    var user: String
    var agentURL: String
    var ips: [String]
    var architecture: String
    var pythonPath: String
    var mlxVersion: String?
    var mlxLMVersion: String?
    var tokenityVersion: String
    var machineID: String? = nil
    var tokenityCodeRevision: String? = nil
    var agentContract: AgentContractInfo? = nil
    var rdma: RDMAStatus
    var roles: [ProcessRole]
    var memory: MemoryStats
    var models: [ModelEntry]
    var isOnline: Bool
    var clusterRuntime: ClusterRuntimeStatus? = nil
    var clusterRuntimes: [ClusterRuntimeStatus] = []
    var agentLatencyMilliseconds: Double? = nil
    var lastAgentResponseAt: Date? = nil
    var agentError: String? = nil
    var consecutiveAgentFailures: Int = 0
    var runtimeMemory: RuntimeMemoryStats? = nil
    var modelInstances: [AgentModelInstanceSnapshot] = []
    var agentHealthState: AgentHealthDisplayState = .unreachable
    var agentHealthDetail: String? = nil
    var watchdogRestartCount: Int = 0
    var lastAutomaticRecoveryAt: Date? = nil
    var displayName: String {
        hostname.isEmpty ? id : hostname
    }

    var primaryIP: String {
        ips.first ?? "unknown"
    }

    var identityDetail: String {
        "\(hostname) · \(primaryIP)"
    }

    var memoryPercentText: String {
        guard isOnline else { return "Unavailable" }
        guard let inUseRatio = memory.inUseRatio ?? memory.usedRatio else { return "Memory unknown" }
        return "\(Int((inUseRatio * 100).rounded()))% in use"
    }

    var memoryUsageText: String {
        guard isOnline else { return "Node offline" }
        guard let inUse = memory.inUseBytes ?? memory.usedBytes,
              let total = memory.totalBytes else { return "Memory unknown" }
        var parts = ["\(Self.formatBytes(inUse)) in use"]
        if let reclaimable = memory.reclaimableBytes, reclaimable > 0 {
            parts.append("\(Self.formatBytes(reclaimable)) reclaimable cache")
        }
        parts.append("\(Self.formatBytes(total)) total")
        if let runtime = runtimeMemory {
            if runtime.stale == true {
                parts.insert("runtime metrics stale", at: 0)
            } else if let observed = runtime.modelResidentObservedBytes ?? runtime.mlxActiveBytes {
                parts.insert("MLX/model \(Self.formatBytes(observed))", at: 0)
            } else if let resident = runtime.processResidentBytes {
                parts.insert("runtime resident \(Self.formatBytes(resident))", at: 0)
            }
        }
        return parts.joined(separator: " · ")
    }

    var displayRuntime: String {
        guard isOnline else { return "Offline" }
        if mlxVersion == nil && mlxLMVersion == nil {
            return "Not inspected"
        }
        return "Ready"
    }

    private static func formatBytes(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824
        if gib >= 10 {
            return String(format: "%.0f GB", gib)
        }
        return String(format: "%.1f GB", gib)
    }

    static var samples: [TokenityNode] {
        let endpoints = TokenityDeploymentConfiguration.nodeAgentURLs
        let fallbackEndpoints = [
            TokenityDeploymentConfiguration.agentBaseURL,
            "",
            "",
        ]
        let resolvedEndpoints = (0 ..< 3).map { index in
            let endpoint = endpoints.indices.contains(index) ? endpoints[index] : fallbackEndpoints[index]
            guard index > 0,
                  let host = URL(string: endpoint)?.host?.lowercased(),
                  host == "localhost"
                    || host.hasSuffix(".localhost")
                    || host.hasPrefix("127.")
                    || host == "::1"
            else { return endpoint }
            return ""
        }
        let modelPath = TokenityDeploymentConfiguration.modelRoot
            .appendingPathComponent("Qwen3.5-122B-A10B-4bit", isDirectory: true)
            .path
        return [
            TokenityNode(
                id: "mac-a",
                hostname: "Mac A",
                user: NSUserName(),
                agentURL: resolvedEndpoints[0],
                ips: URL(string: resolvedEndpoints[0]).flatMap(\.host).map { [$0] } ?? [],
                architecture: "arm64",
                pythonPath: TokenityDeploymentConfiguration.runtimePythonPath,
                mlxVersion: "-",
                mlxLMVersion: "-",
                tokenityVersion: "0.1.0",
                rdma: .empty,
                roles: [],
                memory: .unknown,
                models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: modelPath)],
                isOnline: false
            ),
            TokenityNode(
                id: "mac-b",
                hostname: "Mac B",
                user: NSUserName(),
                agentURL: resolvedEndpoints[1],
                ips: URL(string: resolvedEndpoints[1]).flatMap(\.host).map { [$0] } ?? [],
                architecture: "arm64",
                pythonPath: TokenityDeploymentConfiguration.runtimePythonPath,
                mlxVersion: "-",
                mlxLMVersion: "-",
                tokenityVersion: "0.1.0",
                rdma: .empty,
                roles: [],
                memory: .unknown,
                models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: modelPath)],
                isOnline: false
            ),
            TokenityNode(
                id: "mac-c",
                hostname: "Mac C",
                user: NSUserName(),
                agentURL: resolvedEndpoints[2],
                ips: URL(string: resolvedEndpoints[2]).flatMap(\.host).map { [$0] } ?? [],
                architecture: "arm64",
                pythonPath: TokenityDeploymentConfiguration.runtimePythonPath,
                mlxVersion: "-",
                mlxLMVersion: "-",
                tokenityVersion: "0.1.0",
                rdma: .empty,
                roles: [],
                memory: .unknown,
                models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: modelPath)],
                isOnline: false
            )
        ]
    }
}

struct LaunchSummaryItem: Identifiable, Hashable {
    var title: String
    var value: String

    var id: String { title }
}

struct NetworkPlanRow: Identifiable, Hashable {
    var nodeID: String
    var nodeName: String
    var role: String
    var link: String
    var readiness: String
    var detail: String

    var id: String { nodeID }
}

struct ModelLibraryRow: Identifiable, Hashable {
    var id: String
    var displayName: String
    var modality: ModelModality
    var nodes: [String]
    var availability: String
    var loadState: ModelLoadState
    var representativePath: String
    var format: String?
    var quantization: String?
    var sizeBytes: Int64?
    var architecture: String?
    var shardCount: Int?
    var nativeMTP: NativeMTPCapability?
    var modelType: String?
    var standaloneLoadable: Bool
    var loadBlockReason: String?
    var distributedLoadable: Bool
    var distributedLoadBlockReason: String?

    var usesQwen35Sampling: Bool {
        let identity = [architecture ?? "", modelType ?? ""]
            .joined(separator: " ")
            .lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: "-", with: "")
        return identity.contains("qwen35")
    }

    var sizeText: String {
        guard let sizeBytes else { return "Size unknown" }
        let gib = Double(sizeBytes) / 1_073_741_824
        return gib >= 10 ? String(format: "%.0f GB", gib) : String(format: "%.1f GB", gib)
    }
}

enum ModelThinkingMode: String, Codable, CaseIterable, Identifiable {
    case automatic = "auto"
    case enabled = "on"
    case disabled = "off"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic: return "Auto"
        case .enabled: return "On"
        case .disabled: return "Off"
        }
    }
}

struct ModelSamplingConfiguration: Hashable {
    var temperature: Double
    var topP: Double
    var topK: Int
    var minP: Double
    var presencePenalty: Double
    var repetitionPenalty: Double
}

struct ModelRuntimeConfiguration: Codable, Hashable {
    var maximumOutputTokens: Int = 32_768
    var temperature: Double = 0
    var topP: Double = 1
    var topK: Int = 0
    var minP: Double = 0
    var presencePenalty: Double = 0
    var repetitionPenalty: Double = 1
    var thinkingMode: ModelThinkingMode = .automatic
    var useRecommendedSampling: Bool = true
    var promptCacheSize: Int = 4
    var prefillStepSize: Int = 2_048
    var decodeConcurrency: Int = 1
    var promptConcurrency: Int = 1
    var trustRemoteCode: Bool = false

    init(
        maximumOutputTokens: Int = 32_768,
        temperature: Double = 0,
        topP: Double = 1,
        topK: Int = 0,
        minP: Double = 0,
        presencePenalty: Double = 0,
        repetitionPenalty: Double = 1,
        thinkingMode: ModelThinkingMode = .automatic,
        useRecommendedSampling: Bool = true,
        promptCacheSize: Int = 4,
        prefillStepSize: Int = 2_048,
        decodeConcurrency: Int = 1,
        promptConcurrency: Int = 1,
        trustRemoteCode: Bool = false
    ) {
        self.maximumOutputTokens = maximumOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.thinkingMode = thinkingMode
        self.useRecommendedSampling = useRecommendedSampling
        self.promptCacheSize = promptCacheSize
        self.prefillStepSize = prefillStepSize
        self.decodeConcurrency = decodeConcurrency
        self.promptConcurrency = promptConcurrency
        self.trustRemoteCode = trustRemoteCode
    }

    enum CodingKeys: String, CodingKey {
        case maximumOutputTokens, temperature, topP, topK, minP
        case presencePenalty, repetitionPenalty, thinkingMode, useRecommendedSampling
        case promptCacheSize, prefillStepSize, decodeConcurrency, promptConcurrency
        case trustRemoteCode
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        maximumOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maximumOutputTokens) ?? 32_768
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        topP = try container.decodeIfPresent(Double.self, forKey: .topP) ?? 1
        topK = try container.decodeIfPresent(Int.self, forKey: .topK) ?? 0
        minP = try container.decodeIfPresent(Double.self, forKey: .minP) ?? 0
        presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty) ?? 0
        repetitionPenalty = try container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1
        thinkingMode = try container.decodeIfPresent(ModelThinkingMode.self, forKey: .thinkingMode) ?? .automatic
        useRecommendedSampling = try container.decodeIfPresent(Bool.self, forKey: .useRecommendedSampling) ?? true
        promptCacheSize = try container.decodeIfPresent(Int.self, forKey: .promptCacheSize) ?? 4
        prefillStepSize = try container.decodeIfPresent(Int.self, forKey: .prefillStepSize) ?? 2_048
        decodeConcurrency = try container.decodeIfPresent(Int.self, forKey: .decodeConcurrency) ?? 1
        promptConcurrency = try container.decodeIfPresent(Int.self, forKey: .promptConcurrency) ?? 1
        trustRemoteCode = try container.decodeIfPresent(Bool.self, forKey: .trustRemoteCode) ?? false
    }

    static let `default` = ModelRuntimeConfiguration()

    func validated() -> ModelRuntimeConfiguration {
        var copy = self
        copy.maximumOutputTokens = min(max(maximumOutputTokens, 1), 262_144)
        copy.temperature = min(max(temperature, 0), 2)
        copy.topP = min(max(topP, 0), 1)
        copy.topK = min(max(topK, 0), 1_000)
        copy.minP = min(max(minP, 0), 1)
        copy.presencePenalty = min(max(presencePenalty, -2), 2)
        copy.repetitionPenalty = min(max(repetitionPenalty, 0), 2)
        copy.promptCacheSize = min(max(promptCacheSize, 1), 64)
        copy.prefillStepSize = min(max(prefillStepSize, 128), 8_192)
        copy.decodeConcurrency = min(max(decodeConcurrency, 1), 8)
        copy.promptConcurrency = min(max(promptConcurrency, 1), 8)
        return copy
    }

    func resolvedSampling(forQwen35: Bool) -> ModelSamplingConfiguration {
        guard forQwen35, useRecommendedSampling else {
            return ModelSamplingConfiguration(
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                presencePenalty: presencePenalty,
                repetitionPenalty: repetitionPenalty
            )
        }
        if thinkingMode == .disabled {
            return ModelSamplingConfiguration(
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                minP: 0,
                presencePenalty: 1.5,
                repetitionPenalty: 1
            )
        }
        return ModelSamplingConfiguration(
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0,
            presencePenalty: 1.5,
            repetitionPenalty: 1
        )
    }
}

enum ChatRole: String, Codable, Hashable {
    case user
    case assistant
}

enum ChatGenerationState: String, Codable, Hashable {
    case waiting
    case reasoning
    case answering
    case completed
    case stopped
    case repetitive
    case lengthLimited
    case failed

    var isGenerating: Bool {
        self == .waiting || self == .reasoning || self == .answering
    }
}

enum ChatRoutePolicy: String, CaseIterable, Codable, Hashable, Identifiable {
    case fast
    case balanced
    case quality

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fast: return "Speed"
        case .balanced: return "Balanced"
        case .quality: return "Quality"
        }
    }
}

struct ChatRouteMetadata: Codable, Hashable {
    var routedModelID: String?
    var modelRevision: String?
    var instanceID: String?
    var routeReason: String?
    var confidence: Double?
    var routingLatencyMilliseconds: Double?
    var queueWaitMilliseconds: Double?
    var requestID: String?

    static let empty = ChatRouteMetadata()
}

struct ChatStreamResponseMetadata: Hashable {
    var statusCode: Int
    var contentType: String?
    var route: ChatRouteMetadata
}

enum ChatStreamEvent: ExpressibleByStringLiteral {
    case response(ChatStreamResponseMetadata)
    case line(String)

    init(stringLiteral value: String) {
        self = .line(value)
    }
}

enum ChatRoutingState: Equatable {
    case idle
    case selecting
    case routed(ChatRouteMetadata)
}

struct ChatMessage: Identifiable, Codable, Hashable {
    let id: UUID
    var role: ChatRole
    var content: String
    var thinking: String
    var createdAt: Date
    var includeInContext: Bool
    var generationState: ChatGenerationState?
    var statusMessage: String?
    var metrics: ChatMetrics?
    var reasoningDurationSeconds: Double?
    var reasoningTokenCount: Int?
    var modelName: String?
    var routedModelID: String?
    var modelRevision: String?
    var instanceID: String?
    var routeReason: String?
    var routeConfidence: Double?
    var routingLatencyMilliseconds: Double?
    var queueWaitMilliseconds: Double?
    var requestID: String?

    init(
        id: UUID = UUID(),
        role: ChatRole,
        content: String,
        thinking: String = "",
        createdAt: Date = Date(),
        includeInContext: Bool = true,
        generationState: ChatGenerationState? = nil,
        statusMessage: String? = nil,
        metrics: ChatMetrics? = nil,
        reasoningDurationSeconds: Double? = nil,
        reasoningTokenCount: Int? = nil,
        modelName: String? = nil,
        routedModelID: String? = nil,
        modelRevision: String? = nil,
        instanceID: String? = nil,
        routeReason: String? = nil,
        routeConfidence: Double? = nil,
        routingLatencyMilliseconds: Double? = nil,
        queueWaitMilliseconds: Double? = nil,
        requestID: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinking = thinking
        self.createdAt = createdAt
        self.includeInContext = includeInContext
        self.generationState = generationState
        self.statusMessage = statusMessage
        self.metrics = metrics
        self.reasoningDurationSeconds = reasoningDurationSeconds
        self.reasoningTokenCount = reasoningTokenCount
        self.modelName = modelName
        self.routedModelID = routedModelID
        self.modelRevision = modelRevision
        self.instanceID = instanceID
        self.routeReason = routeReason
        self.routeConfidence = routeConfidence
        self.routingLatencyMilliseconds = routingLatencyMilliseconds
        self.queueWaitMilliseconds = queueWaitMilliseconds
        self.requestID = requestID
    }
}

struct ChatMetrics: Codable, Hashable {
    var firstTokenSeconds: Double?
    var totalSeconds: Double?
    var outputTokensPerSecond: Double?
    var outputTokens: Int? = nil

    static let empty = ChatMetrics()
}

struct ChatSession: Identifiable, Codable, Hashable {
    var id: UUID
    var title: String
    var createdAt: Date
    var updatedAt: Date
    var messages: [ChatMessage]
    var metrics: ChatMetrics
    var titleWasEdited: Bool?
    var selectedModelID: String?
    var routePolicy: ChatRoutePolicy?
    var locksModel: Bool?

    static func fresh(id: UUID = UUID(), now: Date = Date()) -> ChatSession {
        ChatSession(
            id: id,
            title: "New Chat",
            createdAt: now,
            updatedAt: now,
            messages: [],
            metrics: .empty,
            titleWasEdited: false,
            selectedModelID: "tokenity-auto",
            routePolicy: .balanced,
            locksModel: false
        )
    }

    var preview: String {
        guard let content = messages.reversed().first(where: {
            !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        })?.content else {
            return "No messages yet"
        }
        return content
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func matches(search query: String) -> Bool {
        let clean = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return true }
        return title.localizedCaseInsensitiveContains(clean) || preview.localizedCaseInsensitiveContains(clean)
    }
}

struct OpenAIChatRequest: Encodable {
    struct Message: Encodable {
        var role: String
        var content: String
    }

    var model: String
    var messages: [Message]
    var stream: Bool
    var maxTokens: Int?
    var temperature: Double? = nil
    var topP: Double? = nil
    var topK: Int? = nil
    var minP: Double? = nil
    var presencePenalty: Double? = nil
    var repetitionPenalty: Double? = nil
    var chatTemplateKwargs: [String: Bool]? = nil
    var tokenityRoutePolicy: String? = nil
    var tokenitySessionID: String? = nil
    var tokenityLockModel: Bool? = nil
    var tokenityConstraints: [String: JSONValue]? = nil

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature
        case maxTokens = "max_tokens"
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case presencePenalty = "presence_penalty"
        case repetitionPenalty = "repetition_penalty"
        case chatTemplateKwargs = "chat_template_kwargs"
        case tokenityRoutePolicy = "tokenity_route_policy"
        case tokenitySessionID = "tokenity_session_id"
        case tokenityLockModel = "tokenity_lock_model"
        case tokenityConstraints = "tokenity_constraints"
    }
}

enum JSONValue: Encodable, Hashable {
    case string(String)
    case bool(Bool)
    case integer(Int)
    case strings([String])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .strings(let value): try container.encode(value)
        }
    }
}

struct ResidentModelInstanceSummary: Identifiable, Hashable {
    var id: String
    var modelID: String
    var modelRevision: String?
    var state: String
    var activeRequestCount: Int
    var queueDepth: Int
    var selectedNodes: [String]
    var executionMode: String?
    var connectionMode: String?
    var reservedMemoryBytes: Int64?
    var actualMemoryBytes: Int64?
    var capabilities: [String]
    var warmTTFTP50Milliseconds: Double?
    var warmTTFTP95Milliseconds: Double?
    var allowsAuto: Bool
    var keepsResident: Bool
    var healthIssue: String?
    var isRoutable: Bool

    var isReady: Bool { isRoutable && state.lowercased() == "ready" }
    var isBusy: Bool { isRoutable && state.lowercased() == "busy" }
    var displayState: String {
        isRoutable ? state.capitalized : "Unavailable"
    }
}

struct OpenAIModelsResponse: Decodable {
    struct Model: Decodable {
        var id: String
    }

    var data: [Model]
}

struct AgentClusterNodeRequest: Encodable {
    var id: String
    var agentURL: String
    var lanIP: String?
    var rdmaIP: String?
    var rdmaDevices: [String]

    enum CodingKeys: String, CodingKey {
        case id
        case agentURL = "agent_url"
        case lanIP = "lan_ip"
        case rdmaIP = "rdma_ip"
        case rdmaDevices = "rdma_devices"
    }
}

struct AgentStartModelRequest: Encodable {
    var model: String
    var nodes: [AgentClusterNodeRequest]
    var connectionMode: String
    var startingPort: Int
    var host: String
    var port: Int
    var dryRun: Bool
    var maxTokens: Int
    var promptCacheSize: Int
    var prefillStepSize: Int
    var decodeConcurrency: Int
    var promptConcurrency: Int
    var trustRemoteCode: Bool
    var leaseSeconds: Double = 30
    // Omit the field for standard decoding so a new Control app can still
    // load models through pre-MTP Node Agents. New Agents already default a
    // missing field to off.
    var nativeMTP: NativeMTPConfiguration? = nil
    var instanceID: String? = nil
    var operationID: String? = nil
    var memoryReservationBytes: Int64? = nil

    enum CodingKeys: String, CodingKey {
        case model, nodes, host, port
        case connectionMode = "connection_mode"
        case startingPort = "starting_port"
        case dryRun = "dry_run"
        case maxTokens = "max_tokens"
        case promptCacheSize = "prompt_cache_size"
        case prefillStepSize = "prefill_step_size"
        case decodeConcurrency = "decode_concurrency"
        case promptConcurrency = "prompt_concurrency"
        case trustRemoteCode = "trust_remote_code"
        case leaseSeconds = "lease_seconds"
        case nativeMTP = "native_mtp"
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case memoryReservationBytes = "memory_reservation_bytes"
    }
}

struct AgentStartModelResponse: Decodable {
    var instanceID: String?
    var operationID: String?
    var apiBaseURL: String?

    enum CodingKeys: String, CodingKey {
        case instanceID = "instance_id"
        case operationID = "operation_id"
        case apiBaseURL = "api_base_url"
    }
}

struct AgentStopRoleRequest: Encodable {
    var role: String
    var timeout: Double
    var instanceID: String? = nil

    enum CodingKeys: String, CodingKey {
        case role, timeout
        case instanceID = "instance_id"
    }
}

struct AgentStopAllRequest: Encodable {
    var timeout: Double
}

struct AgentHeartbeatRequest: Encodable {
    var ttlSeconds: Double
    var instanceID: String? = nil

    enum CodingKeys: String, CodingKey {
        case ttlSeconds = "ttl_seconds"
        case instanceID = "instance_id"
    }
}

struct OpenAIChatChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            var content: String?
            var reasoningContent: String?
            var reasoning: String?

            enum CodingKeys: String, CodingKey {
                case content
                case reasoningContent = "reasoning_content"
                case reasoning
            }
        }

        var delta: Delta?
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    var choices: [Choice]
    var usage: OpenAIUsage?
}

struct OpenAIUsage: Decodable {
    var promptTokens: Int?
    var completionTokens: Int?
    var totalTokens: Int?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

struct OpenAIChatResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            var content: String?
            var reasoningContent: String?
            var reasoning: String?

            enum CodingKeys: String, CodingKey {
                case content
                case reasoningContent = "reasoning_content"
                case reasoning
            }
        }

        var message: Message?
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case message
            case finishReason = "finish_reason"
        }
    }

    var choices: [Choice]
    var usage: OpenAIUsage?
}

struct LaunchPreview: Equatable {
    var summary: [LaunchSummaryItem]
    var networkPlan: [NetworkPlanRow]
    var warnings: [String]
    var readinessIssues: [String]
}
