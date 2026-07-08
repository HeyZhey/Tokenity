import Foundation

enum AppSection: String, CaseIterable, Identifiable {
    case overview
    case cluster
    case chat
    case models
    case network
    case logs
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: return "Overview"
        case .cluster: return "Cluster"
        case .chat: return "Chat"
        case .models: return "Models"
        case .network: return "Network / RDMA"
        case .logs: return "Logs"
        case .settings: return "Settings"
        }
    }

    var symbol: String {
        switch self {
        case .overview: return "gauge.with.dots.needle.50percent"
        case .cluster: return "point.3.connected.trianglepath.dotted"
        case .chat: return "bubble.left.and.bubble.right"
        case .models: return "cube.transparent"
        case .network: return "network"
        case .logs: return "scroll"
        case .settings: return "gearshape"
        }
    }

    var group: String {
        switch self {
        case .overview, .cluster, .chat, .models, .network: return "Cluster"
        case .logs, .settings: return "Operations"
        }
    }
}

enum BackendMode: String, CaseIterable, Identifiable {
    case official = "Official MLX-LM Server (Experimental)"
    case distributed = "Tokenity Distributed Server"
    case singleNode = "Single-Mac Tokenity Server"

    var id: String { rawValue }
}

enum ConnectionMode: String, CaseIterable, Identifiable {
    case ring = "Standard Network"
    case jaccl = "Thunderbolt RDMA"
    case jacclRing = "Thunderbolt + Fallback"

    var id: String { rawValue }

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
    var pid: Int?
    var returnCode: Int?
    var command: [String]?
    var logPath: String?
    var message: String?
    var logTail: String?

    var id: String { role }

    enum CodingKeys: String, CodingKey {
        case role, state, pid, command, message
        case returnCode = "return_code"
        case logPath = "log_path"
        case logTail = "log_tail"
    }
}

struct NodeStatusResponse: Codable {
    var roles: [ProcessRole]
    var memory: MemoryStats?
}

struct ModelEntry: Codable, Hashable, Identifiable {
    var id: String
    var path: String
}

struct NodeModelsResponse: Codable {
    var root: String
    var models: [ModelEntry]
}

struct NodeInfoResponse: Codable {
    var nodeID: String
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
    var processRoles: [ProcessRole]
    var memory: MemoryStats?
    var rdma: RDMAStatus

    enum CodingKeys: String, CodingKey {
        case nodeID = "node_id"
        case hostname, user, ips, architecture, venv, rdma
        case macOSVersion = "macos_version"
        case pythonPath = "python_path"
        case mlxVersion = "mlx_version"
        case mlxLMVersion = "mlx_lm_version"
        case tokenityVersion = "tokenity_version"
        case processRoles = "process_roles"
        case memory
    }
}

struct MemoryStats: Codable, Hashable {
    var totalBytes: Int64?
    var usedBytes: Int64?
    var freeBytes: Int64?
    var usedRatio: Double?

    enum CodingKeys: String, CodingKey {
        case totalBytes = "total_bytes"
        case usedBytes = "used_bytes"
        case freeBytes = "free_bytes"
        case usedRatio = "used_ratio"
    }

    static let unknown = MemoryStats(totalBytes: nil, usedBytes: nil, freeBytes: nil, usedRatio: nil)
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
    var rdma: RDMAStatus
    var roles: [ProcessRole]
    var memory: MemoryStats
    var models: [ModelEntry]
    var isOnline: Bool
    var ssh: String

    var displayName: String {
        if agentURL.contains("192.168.5.23") || ssh.contains("192.168.5.23") || (user == "apple" && hostname.localizedCaseInsensitiveContains("Mac")) {
            return "Mango"
        }
        if agentURL.contains("192.168.5.75") || ssh.contains("192.168.5.75") {
            return "Kiwi"
        }
        if agentURL.contains("127.0.0.1") {
            return "Apple"
        }
        let names = ["Mango", "Kiwi", "Apple", "Lime", "Pear", "Plum", "Berry"]
        let key = [id, agentURL, ssh, hostname].joined(separator: "|")
        let index = key.unicodeScalars.reduce(0) { value, scalar in
            (value * 31 + Int(scalar.value)) % names.count
        }
        return names[index]
    }

    var primaryIP: String {
        ips.first ?? "unknown"
    }

    var identityDetail: String {
        "\(hostname) · \(primaryIP)"
    }

    var memoryPercentText: String {
        guard let usedRatio = memory.usedRatio else { return "Memory unknown" }
        return "\(Int((usedRatio * 100).rounded()))% memory"
    }

    var memoryUsageText: String {
        guard let used = memory.usedBytes, let total = memory.totalBytes else { return "Memory unknown" }
        return "\(Self.formatBytes(used)) / \(Self.formatBytes(total))"
    }

    var displayRuntime: String {
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

    static let samples: [TokenityNode] = [
        TokenityNode(
            id: "mac-a",
            hostname: "Mac A",
            user: "apple",
            agentURL: "http://192.168.5.23:9100",
            ips: ["192.168.5.23"],
            architecture: "arm64",
            pythonPath: "/usr/bin/python3",
            mlxVersion: "-",
            mlxLMVersion: "-",
            tokenityVersion: "0.1.0",
            rdma: RDMAStatus(
                rdmaEnabled: true,
                rdmaDevices: ["rdma_en4"],
                rdmaPortState: ["rdma_en4": "active"],
                thunderboltIP: "192.168.0.1",
                rdmaErrors: []
            ),
            roles: [],
            memory: .unknown,
            models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: "/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit")],
            isOnline: false,
            ssh: "127.0.0.1"
        ),
        TokenityNode(
            id: "mac-b",
            hostname: "Mac B",
            user: "probriefing",
            agentURL: "http://192.168.5.75:9100",
            ips: ["192.168.5.75"],
            architecture: "arm64",
            pythonPath: "/usr/bin/python3",
            mlxVersion: "-",
            mlxLMVersion: "-",
            tokenityVersion: "0.1.0",
            rdma: RDMAStatus(
                rdmaEnabled: true,
                rdmaDevices: ["rdma_en5"],
                rdmaPortState: ["rdma_en5": "active"],
                thunderboltIP: "192.168.0.2",
                rdmaErrors: []
            ),
            roles: [],
            memory: .unknown,
            models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: "/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit")],
            isOnline: false,
            ssh: "probriefing@192.168.5.75"
        ),
        TokenityNode(
            id: "mac-c",
            hostname: "Mac C",
            user: NSUserName(),
            agentURL: "http://127.0.0.1:9100",
            ips: ["192.168.5.219"],
            architecture: "arm64",
            pythonPath: "/usr/bin/python3",
            mlxVersion: "-",
            mlxLMVersion: "-",
            tokenityVersion: "0.1.0",
            rdma: .empty,
            roles: [],
            memory: .unknown,
            models: [ModelEntry(id: "Qwen3.5-122B-A10B-4bit", path: "/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit")],
            isOnline: false,
            ssh: "127.0.0.1"
        )
    ]
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
    var nodes: [String]
    var availability: String
    var loadState: ModelLoadState
    var representativePath: String
}

enum ChatRole: String, Codable, Hashable {
    case user
    case assistant
}

struct ChatMessage: Identifiable, Hashable {
    let id: UUID
    var role: ChatRole
    var content: String
    var thinking: String
    var createdAt: Date

    init(
        id: UUID = UUID(),
        role: ChatRole,
        content: String,
        thinking: String = "",
        createdAt: Date = Date()
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinking = thinking
        self.createdAt = createdAt
    }
}

struct ChatMetrics: Hashable {
    var firstTokenSeconds: Double?
    var totalSeconds: Double?
    var outputTokensPerSecond: Double?

    static let empty = ChatMetrics()
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

    enum CodingKeys: String, CodingKey {
        case model, messages, stream
        case maxTokens = "max_tokens"
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
    var ssh: String
    var lanIP: String?
    var rdmaIP: String?
    var rdmaDevices: [String]

    enum CodingKeys: String, CodingKey {
        case id, ssh
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

    enum CodingKeys: String, CodingKey {
        case model, nodes, host, port
        case connectionMode = "connection_mode"
        case startingPort = "starting_port"
        case dryRun = "dry_run"
    }
}

struct AgentStopRoleRequest: Encodable {
    var role: String
    var timeout: Double
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
    }

    var choices: [Choice]
}

struct LaunchPreview {
    var summary: [LaunchSummaryItem]
    var networkPlan: [NetworkPlanRow]
    var warnings: [String]
    var readinessIssues: [String]
}
