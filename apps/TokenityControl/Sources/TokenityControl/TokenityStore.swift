import Foundation

private struct UncheckedSendableBox<Value>: @unchecked Sendable {
    let value: Value
}

enum TokenityTransportError: LocalizedError {
    case invalidResponse
    case httpStatus(Int, String?)
    case missingClusterControl
    case missingModelService
    case modelServiceNotReady
    case modelServiceReturnedNoModels(String)
    case backendExited(String)
    case noChatContent
    case rdmaNotReady(String)
    case nativeMTPAgentUpgradeRequired
    case multiInstanceAgentUpgradeRequired
    case repetitiveOutput

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The cluster service returned an unreadable response."
        case .httpStatus(let status, let detail):
            if status == 404 {
                return detail ?? "The selected Mac is running an incompatible Node Agent. Restart the new Tokenity Node Agent, then try again."
            }
            if let detail, !detail.isEmpty {
                return detail
            }
            return "The cluster service rejected the request."
        case .missingClusterControl:
            return "The selected cluster Mac is not reachable."
        case .missingModelService:
            return "The model service address is not available."
        case .modelServiceNotReady:
            return "The model service did not become ready."
        case .modelServiceReturnedNoModels(let modelName):
            return "The model service responded, but did not report \(modelName) as loaded. Confirm the backend runtime and try Standard Network if the distributed transport is unavailable."
        case .backendExited(let message):
            return message
        case .noChatContent:
            return "The model service returned no answer text."
        case .rdmaNotReady(let message):
            return message
        case .nativeMTPAgentUpgradeRequired:
            return "Native MTP Required needs the current Node Agent. Restart the installed Tokenity Node Agent on every selected Mac, then try again."
        case .multiInstanceAgentUpgradeRequired:
            return "Loading an additional resident model requires current Node Agents on every selected Mac. Upgrade or restart all selected Node Agents, then try again."
        case .repetitiveOutput:
            return "Generation stopped because repeated output was detected."
        }
    }
}

struct ChatRepetitionDetector {
    private var rawText = ""
    private var uncheckedCharacterCount = 0
    private let windowCharacters = 8_192
    private let minimumUnitBytes = 24
    private let maximumUnitBytes = 512
    private let repetitions = 3

    mutating func observe(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        rawText = String((rawText + text).suffix(windowCharacters))
        uncheckedCharacterCount += text.count
        guard uncheckedCharacterCount >= minimumUnitBytes else { return false }
        uncheckedCharacterCount = 0

        let normalized = rawText
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        let words = normalized.split(separator: " ")
        let maximumWords = min(64, words.count / repetitions)
        if maximumWords >= 4 {
            for width in 4...maximumWords {
                let blockStart = words.count - width
                let block = words[blockStart..<words.count]
                var repeats = true
                for repetition in 2...repetitions {
                    let start = words.count - (width * repetition)
                    if !words[start..<(start + width)].elementsEqual(block) {
                        repeats = false
                        break
                    }
                }
                if repeats { return true }
            }
        }
        let bytes = Array(normalized.utf8)
        let maximum = min(maximumUnitBytes, bytes.count / repetitions)
        guard maximum >= minimumUnitBytes else { return false }

        for width in minimumUnitBytes...maximum {
            let blockStart = bytes.count - width
            let block = bytes[blockStart..<bytes.count]
            var repeats = true
            for repetition in 2...repetitions {
                let start = bytes.count - (width * repetition)
                if !bytes[start..<(start + width)].elementsEqual(block) {
                    repeats = false
                    break
                }
            }
            if repeats { return true }
        }
        return false
    }
}

private struct ModelReadinessResponse: Decodable {
    struct ReadyEvidence: Decodable {
        var oneTokenProbe: Bool?
        var warmupCacheIsolated: Bool?

        enum CodingKeys: String, CodingKey {
            case oneTokenProbe = "one_token_probe"
            case warmupCacheIsolated = "warmup_cache_isolated"
        }

        var verifiesInference: Bool {
            oneTokenProbe == true && warmupCacheIsolated == true
        }
    }

    var phase: String
    var message: String?
    var progress: Double?
    var progressCurrent: Int?
    var progressTotal: Int?
    var nativeMTP: NativeMTPReadiness?
    var instanceID: String?
    var rank: Int?
    var worldSize: Int?
    var connectionMode: String?
    var modelRevision: String?
    var memory: RuntimeMemoryStats?
    var readyEvidence: ReadyEvidence?

    enum CodingKeys: String, CodingKey {
        case phase, message, progress
        case progressCurrent = "progress_current"
        case progressTotal = "progress_total"
        case nativeMTP = "native_mtp"
        case instanceID = "instance_id"
        case rank
        case worldSize = "world_size"
        case connectionMode = "connection_mode"
        case modelRevision = "model_revision"
        case memory
        case readyEvidence = "ready_evidence"
    }
}

private struct InstanceQuorumResponse: Decodable {
    struct RankEvidence: Decodable {
        var node: String
        var runtime: ModelReadinessResponse?
    }

    var instanceID: String
    var ready: Bool
    var issues: [String]
    var rankQuorum: String
    var ranks: [RankEvidence]

    enum CodingKeys: String, CodingKey {
        case ready, issues, ranks
        case instanceID = "instance_id"
        case rankQuorum = "rank_quorum"
    }
}

private struct ManagedModelInstance {
    var modelID: String
    var serviceModelName: String
    var instanceID: String
    var apiBaseURL: String?
    var backendRole: String
    var lifecycleState: String = "ready"
    var quorumReady: Bool = true
    var routeAvailable: Bool = true
    var healthIssue: String? = nil
    var modelRevision: String? = nil
    var selectedNodes: [String] = []
    var executionMode: String? = nil
    var connectionMode: String? = nil
    var reservedMemoryBytes: Int64? = nil
    var actualMemoryBytes: Int64? = nil
    var activeRequestCount: Int = 0
    var queueDepth: Int = 0
    var routeCapabilities: GatewayRouteCapabilities? = nil
    var warmTTFTP50Milliseconds: Double? = nil
    var warmTTFTP95Milliseconds: Double? = nil

    var isRoutable: Bool {
        routeAvailable
            && quorumReady
            && ["ready", "busy"].contains(lifecycleState.lowercased())
    }
}

private struct H3VideoEndpoint {
    var id: String
    var hostname: String
    var user: String
    var agentURL: String
}

private enum DiscoveryMergeResult: Equatable {
    case unchanged
    case rebound
    case added
}

@MainActor
final class TokenityStore: ObservableObject {
    typealias DataTransport = (URLRequest) async throws -> (Data, HTTPURLResponse)
    typealias LineStreamTransport = (URLRequest) -> AsyncThrowingStream<ChatStreamEvent, Error>
    typealias VideoArtifactTransport = (
        H3VideoCompletePayload,
        H3VideoGenerationRequest
    ) async throws -> GeneratedVideoArtifact
    typealias NodeDiscoveryTransport = ([URL]) async -> [DiscoveredNodeEndpoint]

    @Published var selectedSection: AppSection? = .overview
    @Published var nodes: [TokenityNode] = TokenityNode.samples
    @Published var selectedNodeIDs: Set<String> = ["mac-a", "mac-b"] {
        didSet {
            keepClusterPrimaryInSelection()
            rebuildLaunchPreview()
        }
    }
    @Published var backendMode: BackendMode = .distributed {
        didSet { rebuildLaunchPreview() }
    }
    @Published var connectionMode: ConnectionMode = .jaccl {
        didSet { rebuildLaunchPreview() }
    }
    @Published var nativeMTPMode: NativeMTPMode = .off {
        didSet { rebuildLaunchPreview() }
    }
    @Published var phase: ClusterPhase = .stopped
    @Published private(set) var modelPath: String = ""
    @Published private(set) var coordinatorID: String = "mac-a" {
        didSet { rebuildLaunchPreview() }
    }
    @Published var agentBaseURL: String = TokenityDeploymentConfiguration.agentBaseURL
    @Published var modelRoot: String = TokenityDeploymentConfiguration.modelRoot.path
    @Published var modelLoadStates: [String: ModelLoadState] = [:]
    @Published var launchPreview = LaunchPreview(summary: [], networkPlan: [], warnings: [], readinessIssues: [])
    @Published var isScanningModels = false
    @Published var modelScanSummary = "Not scanned"
    @Published var modelLoadMessage = "No model loaded"
    @Published private(set) var modelLoadProgress: Double?
    @Published private(set) var nativeMTPRuntime: NativeMTPReadiness?
    @Published private(set) var serverHealth: ServerHealthState = .stopped
    @Published private(set) var isRefreshingStatus = false
    @Published private(set) var isDiscoveringNodes = false
    @Published private(set) var isConnectingNode = false
    @Published private(set) var nodeDiscoverySummary = "Automatic LAN discovery has not run yet"
    @Published private(set) var lastNodeDiscoveryAt: Date?
    private(set) var lastStatusRefreshAt: Date?
    private static let initialChatSession = ChatSession.fresh()

    @Published var chatInput = ""
    @Published var chatMessages: [ChatMessage] = TokenityStore.initialChatSession.messages
    @Published var chatMetrics = TokenityStore.initialChatSession.metrics
    @Published var isChatRunning = false
    @Published private(set) var chatRoutingState: ChatRoutingState = .idle
    @Published private(set) var chatSelectedModelID = "tokenity-auto"
    @Published private(set) var chatRoutePolicy: ChatRoutePolicy = .balanced
    @Published private(set) var locksChatModel = false
    @Published private(set) var chatSessions: [ChatSession] = [TokenityStore.initialChatSession]
    @Published private(set) var activeChatSessionID: UUID = TokenityStore.initialChatSession.id
    @Published private(set) var chatScrollRevision = 0
    @Published private(set) var chatComposerFocusRevision = 0
    @Published private(set) var isChatHistoryLoading = false
    @Published var h3CoordinatorAgentURL = TokenityDeploymentConfiguration.h3CoordinatorAgentURL
    @Published var h3WorkerAgentURL = TokenityDeploymentConfiguration.h3WorkerAgentURL
    @Published var h3ModelPath = TokenityDeploymentConfiguration.h3ModelPath
    @Published var h3BinaryPath = TokenityDeploymentConfiguration.h3BinaryPath
    @Published var h3OptimizationProfile: H3OptimizationProfile = .stockQMM
    @Published var videoRequest = H3VideoGenerationRequest.default
    @Published private(set) var videoNodes: [TokenityNode] = []
    @Published private(set) var videoRuntimeState: VideoRuntimeState = .stopped
    @Published private(set) var videoRuntimeInstanceID: String?
    @Published private(set) var isVideoGenerating = false
    @Published private(set) var videoProgress: Double = 0
    @Published private(set) var videoProgressStage = "Waiting for a MiniMax H3 runtime"
    @Published private(set) var videoGenerationError: String?
    @Published private(set) var generatedVideoArtifact: GeneratedVideoArtifact?
    @Published private(set) var apiAccessStatus = "Not checked"
    @Published private(set) var modelConfigurations: [String: ModelRuntimeConfiguration] = [:]
    @Published private(set) var isOnboardingPresented = false
    @Published var logs: [String] = [
        "Tokenity Control opened.",
        "No cluster is running."
    ]

    private let dataTransport: DataTransport
    private let lineStreamTransport: LineStreamTransport
    private let videoArtifactTransport: VideoArtifactTransport
    private let nodeDiscoveryTransport: NodeDiscoveryTransport
    private let userDefaults: UserDefaults
    private let modelConfigurationsKey = "TokenityModelRuntimeConfigurations.v1"
    private let residentAutoPreferencesKey = "TokenityResidentAutoPreferences.v1"
    private let chatSessionsKey = "TokenityChatSessions.v1"
    private let onboardingRevisionKey = "TokenityOnboarding.completedRevision"
    private let nodeEndpointOverridesKey = "TokenityNodeAgentEndpoints.v1"
    private static let currentOnboardingRevision = 2
    private let chatHistoryQueue = DispatchQueue(label: "ai.tokenity.chat-history", qos: .utility)
    private let writesChatHistorySynchronously: Bool
    private var chatHistoryRevision = 0
    private var mlxStartingPort: Int {
        Int(ProcessInfo.processInfo.environment["TOKENITY_CONTROL_STARTING_PORT"] ?? "")
            ?? 30_020
    }
    private var modelHTTPPort: Int {
        Int(ProcessInfo.processInfo.environment["TOKENITY_CONTROL_MODEL_PORT"] ?? "")
            ?? 8_000
    }
    // MiniMax H3 has a separately validated JACCL communicator. Keeping it
    // off the text-runtime port range also prevents stale communicators from
    // one workload blocking the other.
    private var h3StartingPort: Int {
        Int(ProcessInfo.processInfo.environment["TOKENITY_CONTROL_H3_STARTING_PORT"] ?? "")
            ?? 30_096
    }
    private static let h3ModelID = "MiniMax-H3"
    private let modelLeaseSeconds = 30.0
    private let appRestartLeaseGraceSeconds = 120.0
    private var loadedBackendRole: String?
    private var loadedServiceModelName: String?
    private var activeLoadedModelID: String?
    // Managed instances are keyed by their stable instance identity. A model ID
    // is only a routing alias and may legitimately have multiple ready replicas.
    private var managedModelInstances: [String: ManagedModelInstance] = [:]
    private(set) var activeModelInstanceID: String?
    private var activeModelServiceBaseURL: String?
    private var legacyNativeMTPFallback: NativeMTPReadiness?
    private var activeModelLoadID: UUID?
    private var pendingModelLoadID: UUID?
    private var modelLoadTask: Task<Void, Never>?
    private var activeChatRequestID: UUID?
    private var activeChatUserIndex: Int?
    private var activeChatAssistantIndex: Int?
    private var chatTask: Task<Void, Never>?
    private var chatTaskID: UUID?
    private var videoTask: Task<Void, Never>?
    private var activeVideoRuntimeOperationID: UUID?
    private var activeChatRoutedInstanceID: String?
    private var nextChatRouteConstraints: [String: JSONValue]?
    private var residentAutoPreferences: [String: Bool] = [:]
    private var leaseRenewalFailureKeys: Set<String> = []
    private var statusMonitoringTask: Task<Void, Never>?
    private var nodeDiscoveryTask: Task<Void, Never>?
    private var unboundPlaceholderNodeIDs: Set<String> = []
    private var autoSelectedPlaceholderNodeIDs: Set<String> = []
    private var statusRefreshInFlight = false
    // Every user-owned model transition advances this fence synchronously.
    // Background polling may still finish its network request, but a response
    // from an older epoch is never allowed to rewrite model/instance state.
    private var modelOperationEpoch: UInt64 = 0
    private var nodeTopologyRevision = 0
    private(set) var statusMonitoringStartCount = 0

    init(
        dataTransport: @escaping DataTransport = TokenityStore.liveData(for:),
        lineStreamTransport: @escaping LineStreamTransport = TokenityStore.liveLineStream(for:),
        videoArtifactTransport: @escaping VideoArtifactTransport = { payload, request in
            try await H3VideoArtifactWriter.write(payload: payload, request: request)
        },
        nodeDiscoveryTransport: @escaping NodeDiscoveryTransport = TokenityNodeDiscovery.discover(seedOrigins:),
        userDefaults: UserDefaults = .standard,
        loadsChatHistorySynchronously: Bool = true
    ) {
        self.dataTransport = dataTransport
        self.lineStreamTransport = lineStreamTransport
        self.videoArtifactTransport = videoArtifactTransport
        self.nodeDiscoveryTransport = nodeDiscoveryTransport
        self.userDefaults = userDefaults
        writesChatHistorySynchronously = loadsChatHistorySynchronously
        let environment = ProcessInfo.processInfo.environment
        h3CoordinatorAgentURL = environment["TOKENITY_H3_COORDINATOR_AGENT"]
            ?? h3CoordinatorAgentURL
        h3WorkerAgentURL = environment["TOKENITY_H3_WORKER_AGENT"]
            ?? h3WorkerAgentURL
        h3ModelPath = environment["TOKENITY_H3_MODEL_PATH"] ?? h3ModelPath
        h3BinaryPath = environment["TOKENITY_H3_BINARY_PATH"] ?? h3BinaryPath
        if TokenityDeploymentConfiguration.hasExplicitNodeAgentURLs {
            unboundPlaceholderNodeIDs = Set(
                nodes.lazy.filter { $0.agentURL.isEmpty }.map(\.id)
            )
        } else {
            unboundPlaceholderNodeIDs = Set(nodes.dropFirst().map(\.id))
        }
        autoSelectedPlaceholderNodeIDs = selectedNodeIDs.intersection(
            unboundPlaceholderNodeIDs
        )
        if let overrides = userDefaults.dictionary(forKey: nodeEndpointOverridesKey) as? [String: String] {
            var retainedOverrides = overrides
            let configuredEndpointCount = TokenityDeploymentConfiguration.nodeAgentURLs.count
            for index in nodes.indices {
                if TokenityDeploymentConfiguration.hasExplicitNodeAgentURLs,
                   index < configuredEndpointCount {
                    retainedOverrides.removeValue(forKey: nodes[index].id)
                    continue
                }
                guard let endpoint = overrides[nodes[index].id],
                      let url = Self.normalizedAgentBaseURL(endpoint)
                else { continue }
                if unboundPlaceholderNodeIDs.contains(nodes[index].id),
                   Self.isLoopbackAgentURL(url) {
                    // Migrate the loopback-only Mac B/Mac C placeholders
                    // written by the first portable build. They represented
                    // no verified machine and prevented LAN discovery from
                    // claiming the selected worker slot.
                    retainedOverrides.removeValue(forKey: nodes[index].id)
                    continue
                }
                nodes[index].agentURL = url.absoluteString
                nodes[index].ips = [url.host!]
                unboundPlaceholderNodeIDs.remove(nodes[index].id)
            }
            if retainedOverrides != overrides {
                userDefaults.set(retainedOverrides, forKey: nodeEndpointOverridesKey)
            }
        }
        // Explicit isolated-test settings override persisted LAN discovery.
        if let isolatedAgentPort = Int(
            ProcessInfo.processInfo.environment["TOKENITY_CONTROL_AGENT_PORT"] ?? ""
        ) {
            nodes = nodes.map { node in
                var configured = node
                if var components = URLComponents(string: node.agentURL) {
                    components.port = isolatedAgentPort
                    configured.agentURL = components.url?.absoluteString ?? node.agentURL
                }
                return configured
            }
            agentBaseURL = "http://127.0.0.1:\(isolatedAgentPort)"
            unboundPlaceholderNodeIDs.removeAll()
        }
        selectedNodeIDs.subtract(unboundPlaceholderNodeIDs)
        if let data = userDefaults.data(forKey: modelConfigurationsKey),
           let decoded = try? JSONDecoder().decode([String: ModelRuntimeConfiguration].self, from: data) {
            modelConfigurations = decoded
        }
        if let data = userDefaults.data(forKey: residentAutoPreferencesKey),
           let decoded = try? JSONDecoder().decode([String: Bool].self, from: data) {
            residentAutoPreferences = decoded
        }
        if loadsChatHistorySynchronously {
            if let sessions = Self.loadChatSessions(from: userDefaults, key: chatSessionsKey),
               let latest = sessions.first {
                chatSessions = sessions
                activeChatSessionID = latest.id
                chatMessages = latest.messages
                chatMetrics = latest.metrics
                chatSelectedModelID = latest.selectedModelID ?? "tokenity-auto"
                chatRoutePolicy = latest.routePolicy ?? .balanced
                locksChatModel = latest.locksModel ?? false
            }
        } else {
            isChatHistoryLoading = true
            let defaults = UncheckedSendableBox(value: userDefaults)
            let key = chatSessionsKey
            Task { [weak self] in
                let sessions = await Task.detached(priority: .utility) {
                    Self.loadChatSessions(from: defaults.value, key: key)
                }.value
                guard let self else { return }
                if self.chatHistoryRevision == 0,
                   let sessions,
                   let latest = sessions.first {
                    self.chatSessions = sessions
                    self.activeChatSessionID = latest.id
                    self.chatMessages = latest.messages
                    self.chatMetrics = latest.metrics
                    self.chatSelectedModelID = latest.selectedModelID ?? "tokenity-auto"
                    self.chatRoutePolicy = latest.routePolicy ?? .balanced
                    self.locksChatModel = latest.locksModel ?? false
                    self.chatScrollRevision += 1
                }
                self.isChatHistoryLoading = false
            }
        }
        rebuildLaunchPreview()
    }

    func prepareForAppLaunch() {
        isOnboardingPresented =
            userDefaults.integer(forKey: onboardingRevisionKey) < Self.currentOnboardingRevision
    }

    func presentOnboarding() {
        isOnboardingPresented = true
    }

    func completeOnboarding(opening section: AppSection? = nil) {
        userDefaults.set(Self.currentOnboardingRevision, forKey: onboardingRevisionKey)
        if let section {
            selectedSection = section
        }
        isOnboardingPresented = false
    }

    var selectedNodes: [TokenityNode] {
        nodes.filter { selectedNodeIDs.contains($0.id) }
    }

    private var plannedNodes: [TokenityNode] {
        let selected = selectedNodes
        guard let coordinatorIndex = selected.firstIndex(where: { $0.id == coordinatorID }) else {
            return selected
        }
        return [selected[coordinatorIndex]]
            + selected.enumerated().compactMap { index, node in
                index == coordinatorIndex ? nil : node
            }
    }

    var coordinator: TokenityNode? {
        plannedNodes.first
    }

    var openAIEndpoint: String {
        isChatReady ? "Ready for chat" : "Create a cluster and load a model"
    }

    var openAIAPIBaseURL: String {
        guard
            let agentURL = coordinator?.agentURL,
            var components = URLComponents(string: agentURL),
            components.host != nil
        else { return "Unavailable" }
        components.path = "/v1"
        components.query = nil
        return components.url?.absoluteString ?? "Unavailable"
    }

    var externalAPIModelName: String {
        loadedServiceModelName ?? loadedModelName ?? "Load a model first"
    }

    var selectedModelName: String {
        loadedModelName ?? "No model loaded"
    }

    var loadedModelName: String? {
        if let activeLoadedModelID,
           modelLoadStates[activeLoadedModelID] == .loaded {
            return activeLoadedModelID
        }
        return modelLoadStates
            .filter { $0.value == .loaded }
            .map(\.key)
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
            .first
    }

    var isChatReady: Bool {
        phase == .running && (
            residentModelInstances.contains { $0.allowsAuto && ($0.isReady || $0.isBusy) }
                || loadedModelName != nil
        )
    }

    func managedModelInstanceIDs(for modelID: String) -> Set<String> {
        Set(
            managedModelInstances.values
                .filter { $0.modelID == modelID && $0.isRoutable }
                .map(\.instanceID)
        )
    }

    var residentModelInstances: [ResidentModelInstanceSummary] {
        managedModelInstances.values
            .map { managed in
                return ResidentModelInstanceSummary(
                    id: managed.instanceID,
                    modelID: managed.modelID,
                    modelRevision: managed.modelRevision,
                    state: managed.lifecycleState,
                    activeRequestCount: managed.activeRequestCount,
                    queueDepth: managed.queueDepth,
                    selectedNodes: managed.selectedNodes,
                    executionMode: managed.executionMode,
                    connectionMode: managed.connectionMode,
                    reservedMemoryBytes: managed.reservedMemoryBytes,
                    actualMemoryBytes: managed.actualMemoryBytes,
                    capabilities: managed.routeCapabilities?.displayLabels ?? [],
                    warmTTFTP50Milliseconds: managed.warmTTFTP50Milliseconds,
                    warmTTFTP95Milliseconds: managed.warmTTFTP95Milliseconds,
                    allowsAuto: residentAutoPreferences[managed.instanceID] ?? true,
                    keepsResident: true,
                    healthIssue: managed.healthIssue,
                    isRoutable: managed.isRoutable
                )
            }
            .sorted {
                if $0.modelID != $1.modelID {
                    return $0.modelID.localizedCaseInsensitiveCompare($1.modelID) == .orderedAscending
                }
                return $0.id.localizedCaseInsensitiveCompare($1.id) == .orderedAscending
            }
    }

    var residentReadyModelCount: Int {
        Set(residentModelInstances.filter(\.isReady).map(\.modelID)).count
    }

    var residentBusyModelCount: Int {
        residentModelInstances.filter(\.isBusy).count
    }

    var residentPoolSummary: String {
        "\(residentReadyModelCount) models ready · \(residentBusyModelCount) busy"
    }

    var autoRouterHealthText: String {
        guard selectedAgentsSupportResidentModels else {
            return "Auto routing unavailable · update legacy Node Agent"
        }
        let allowed = residentModelInstances.filter { $0.allowsAuto && ($0.isReady || $0.isBusy) }
        guard !allowed.isEmpty else {
            return residentModelInstances.isEmpty
                ? "Auto routing waiting for resident models"
                : "Auto routing disabled for all resident models"
        }
        return "Auto routing healthy"
    }

    var residentRoutingSummary: String {
        "\(residentPoolSummary) · \(autoRouterHealthText)"
    }

    var activeBackendDisplayName: String {
        guard let active = activeResidentTopology else {
            return backendMode == .singleNode
                ? "Tokenity Single-Mac Server"
                : backendMode.rawValue
        }
        return active.executionMode?.lowercased() == "single"
            ? "Tokenity Single-Mac Server"
            : "Tokenity Distributed Server"
    }

    var activeConnectionDisplayName: String {
        guard let active = activeResidentTopology else {
            return backendMode == .singleNode ? "Single Mac" : connectionMode.rawValue
        }
        if active.executionMode?.lowercased() == "single"
            || active.selectedNodes.count <= 1 {
            return "Single Mac"
        }
        switch active.connectionMode?.lowercased() {
        case "jaccl", "jaccl-ring": return "Thunderbolt RDMA"
        case "ring": return "Standard Network"
        case let value?: return value.replacingOccurrences(of: "_", with: " ").capitalized
        case nil: return connectionMode.rawValue
        }
    }

    private var activeResidentTopology: ResidentModelInstanceSummary? {
        if let activeModelInstanceID,
           let exact = residentModelInstances.first(where: { $0.id == activeModelInstanceID }) {
            return exact
        }
        return residentModelInstances.first(where: { $0.isReady || $0.isBusy })
    }

    var availableChatModelIDs: [String] {
        var modelIDs = Set(residentModelInstances.filter { $0.isReady || $0.isBusy }.map(\.modelID))
        if let loadedModelName {
            modelIDs.insert(loadedModelName)
        }
        return Array(modelIDs)
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    var isAutoChatSelection: Bool {
        chatSelectedModelID == "tokenity-auto"
    }

    var isModelLoading: Bool {
        modelLoadStates.values.contains(.loading) || videoRuntimeState == .starting
    }

    var isModelUnloading: Bool {
        modelLoadStates.values.contains(.unloading) || videoRuntimeState == .stopping
    }

    var isModelTransitioning: Bool {
        isModelLoading || isModelUnloading
    }

    var canEditCluster: Bool {
        phase == .stopped || phase == .failed
    }

    var canEditNativeMTP: Bool {
        backendMode == .distributed && canEditCluster
    }

    var isVideoRuntimeReady: Bool {
        videoRuntimeState == .ready && videoRuntimeInstanceID != nil
    }

    var videoRuntimeReadinessIssues: [String] {
        var issues: [String] = []
        if videoNodes.isEmpty {
            let hasConfiguredEndpoint = !h3CoordinatorAgentURL
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !h3WorkerAgentURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            issues.append(
                hasConfiguredEndpoint
                    ? "Refresh the MiniMax H3 Node Agents to validate the topology."
                    : "Configure at least one MiniMax H3 Node Agent endpoint."
            )
        }
        if h3ModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append("MiniMax H3 model path is required.")
        }
        if h3BinaryPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            issues.append("MiniMax H3 native binary path is required.")
        }
        if videoNodes.count == 2,
           Set(videoNodes.compactMap(\.machineID)).count != videoNodes.count {
            issues.append("MiniMax H3 TP2 endpoints must resolve to two different Macs.")
        }
        for node in videoNodes {
            if !node.isOnline {
                issues.append("\(node.displayName) Node Agent is offline.")
            } else if node.agentContract?.supports("minimax_h3_video") != true {
                issues.append("\(node.displayName) does not advertise the minimax_h3_video capability.")
            }
        }
        if videoNodes.count == 2 {
            if connectionMode == .ring {
                issues.append("MiniMax H3 TP2 requires Thunderbolt RDMA; Standard Network is not supported.")
            }
            for node in videoNodes where !node.rdma.rdmaEnabled {
                issues.append("\(node.displayName) has no active Thunderbolt RDMA device.")
            }
        }
        return Array(Set(issues)).sorted()
    }

    var videoTopologySummary: String {
        switch videoNodes.count {
        case 2:
            return "TP2 · \(videoNodes[0].displayName) rank 0 → \(videoNodes[1].displayName) rank 1"
        case 1:
            return "Single Mac · \(videoNodes[0].displayName)"
        default:
            let configuredCount = [h3CoordinatorAgentURL, h3WorkerAgentURL]
                .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                .count
            return configuredCount == 2
                ? "TP2 endpoints awaiting refresh"
                : "No video nodes configured"
        }
    }

    func startVideoRuntime() async {
        guard videoRuntimeState != .starting, !isVideoRuntimeReady else { return }
        let runtimeOperationID = UUID()
        activeVideoRuntimeOperationID = runtimeOperationID
        videoRuntimeState = .starting
        videoGenerationError = nil
        videoProgressStage = "Validating MiniMax H3 topology"
        appendLog("Validating MiniMax H3 video topology.")
        await refreshVideoNodes()
        guard activeVideoRuntimeOperationID == runtimeOperationID else { return }

        let issues = videoRuntimeReadinessIssues
        guard issues.isEmpty else {
            let message = issues.joined(separator: " ")
            activeVideoRuntimeOperationID = nil
            videoRuntimeState = .failed(message)
            videoProgressStage = "Runtime blocked"
            videoGenerationError = message
            appendLog("MiniMax H3 start blocked: \(message)")
            return
        }
        guard let baseURL = videoControlBaseURL() else {
            let message = TokenityTransportError.missingClusterControl.errorDescription ?? "Cluster unavailable."
            activeVideoRuntimeOperationID = nil
            videoRuntimeState = .failed(message)
            videoGenerationError = message
            return
        }

        let instanceID = "h3-ui-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        let operationID = "h3-ui-op-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(21)
        let requestBody = AgentStartH3VideoRequest(
            model: h3ModelPath.trimmingCharacters(in: .whitespacesAndNewlines),
            binary: h3BinaryPath.trimmingCharacters(in: .whitespacesAndNewlines),
            nodes: videoNodes.map(agentNodePayload(for:)),
            connectionMode: videoNodes.count == 1 ? ConnectionMode.ring.cliValue : connectionMode.cliValue,
            startingPort: h3StartingPort,
            host: "0.0.0.0",
            port: 11_242,
            dryRun: false,
            apiIdentifier: "MiniMax-H3",
            leaseSeconds: appRestartLeaseGraceSeconds,
            instanceID: String(instanceID),
            operationID: String(operationID),
            optimizationProfile: h3OptimizationProfile
        )

        videoRuntimeInstanceID = String(instanceID)
        do {
            var request = try jsonRequest(
                url: baseURL.appendingPathComponent("/v1/node/start-minimax-h3-video"),
                body: requestBody
            )
            request.timeoutInterval = 300
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            let started = try JSONDecoder().decode(AgentStartModelResponse.self, from: data)
            guard let startedID = started.instanceID, !startedID.isEmpty else {
                throw TokenityTransportError.invalidResponse
            }
            guard activeVideoRuntimeOperationID == runtimeOperationID else {
                try? await stopVideoInstance(startedID)
                return
            }
            videoRuntimeInstanceID = startedID
            videoRuntimeState = .ready
            activeVideoRuntimeOperationID = nil
            videoProgressStage = "MiniMax H3 ready"
            appendLog("MiniMax H3 video runtime ready: \(videoTopologySummary) · \(h3OptimizationProfile.rawValue).")
        } catch {
            let diagnostic = await fetchVideoRuntimeFailure(instanceID: String(instanceID))
            try? await stopVideoInstance(String(instanceID))
            guard activeVideoRuntimeOperationID == runtimeOperationID else { return }
            activeVideoRuntimeOperationID = nil
            videoRuntimeInstanceID = nil
            let message = diagnostic ?? userFacingMessage(for: error)
            videoRuntimeState = .failed(message)
            videoGenerationError = message
            videoProgressStage = "Runtime failed"
            appendLog("MiniMax H3 video runtime failed: \(message)")
        }
    }

    func stopVideoRuntime() async {
        guard videoRuntimeInstanceID != nil || videoRuntimeState != .stopped else { return }
        activeVideoRuntimeOperationID = nil
        cancelVideoGeneration()
        if let videoTask {
            await videoTask.value
        }
        videoRuntimeState = .stopping
        let instanceID = videoRuntimeInstanceID
        do {
            if let instanceID {
                try await stopVideoInstance(instanceID)
            }
            appendLog("MiniMax H3 video runtime stopped.")
            videoRuntimeInstanceID = nil
            videoRuntimeState = .stopped
            videoProgressStage = "Waiting for a MiniMax H3 runtime"
        } catch {
            let message = userFacingMessage(for: error)
            videoRuntimeState = .failed(message)
            videoGenerationError = message
            appendLog("MiniMax H3 stop failed: \(message)")
        }
    }

    func beginVideoGeneration() {
        guard videoTask == nil, !isVideoGenerating else { return }
        videoTask = Task { [weak self] in
            await self?.generateVideo()
            self?.videoTask = nil
        }
    }

    func cancelVideoGeneration() {
        guard isVideoGenerating else { return }
        videoTask?.cancel()
        videoProgressStage = "Cancelling"
        appendLog("Cancelling MiniMax H3 video generation.")
    }

    func generateVideo() async {
        guard !isVideoGenerating else { return }
        guard isVideoRuntimeReady, let instanceID = videoRuntimeInstanceID else {
            videoGenerationError = "Start the MiniMax H3 runtime before generating video."
            return
        }
        var generation = videoRequest.validated()
        guard !generation.prompt.isEmpty else {
            videoGenerationError = "Enter a video prompt first."
            return
        }
        generation.model = instanceID
        videoRequest = generation
        guard let baseURL = videoControlBaseURL() else {
            videoGenerationError = TokenityTransportError.missingClusterControl.errorDescription
            return
        }

        let url = baseURL.appendingPathComponent("/v1/video/generations")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 14_400
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            request.httpBody = try JSONEncoder().encode(generation)
        } catch {
            videoGenerationError = userFacingMessage(for: error)
            return
        }

        isVideoGenerating = true
        videoProgress = 0
        videoProgressStage = "Submitting request"
        videoGenerationError = nil
        generatedVideoArtifact = nil
        appendLog("Generating MiniMax H3 video on \(videoTopologySummary).")
        var didComplete = false
        defer { isVideoGenerating = false }

        do {
            for try await streamEvent in lineStreamTransport(request) {
                try Task.checkCancellation()
                guard case .line(let line) = streamEvent else { continue }
                let lineBox = UncheckedSendableBox(value: line)
                let event = try await Task.detached(priority: .userInitiated) {
                    try H3VideoSSEParser.parse(line: lineBox.value)
                }.value
                guard let event else { continue }
                switch event {
                case .progress(let stage, let step, let total):
                    videoProgressStage = stage
                    let fraction = total > 0 ? Double(step) / Double(total) : 0
                    switch stage.lowercased() {
                    case let value where value.contains("encoding"):
                        videoProgress = 0.02
                    case let value where value.contains("generating"):
                        videoProgress = min(0.9, 0.05 + fraction * 0.85)
                    case let value where value.contains("video"):
                        videoProgress = 0.93
                    case let value where value.contains("audio"):
                        videoProgress = 0.97
                    default:
                        videoProgress = max(videoProgress, min(0.9, fraction * 0.9))
                    }
                case .complete(let payload):
                    videoProgressStage = "Saving video"
                    videoProgress = 0.99
                    generatedVideoArtifact = try await videoArtifactTransport(payload, generation)
                    videoProgress = 1
                    videoProgressStage = "Complete"
                    didComplete = true
                    appendLog(
                        "Video complete: \(payload.frames) frames · \(payload.width)×\(payload.height) · \(payload.fps) fps."
                    )
                }
            }
            guard didComplete else { throw H3VideoContractError.invalidEvent }
        } catch {
            if Task.isCancelled {
                videoProgressStage = "Cancelled"
                appendLog("MiniMax H3 video generation cancelled.")
                return
            }
            let message = userFacingMessage(for: error)
            videoGenerationError = message
            videoProgressStage = "Failed"
            appendLog("MiniMax H3 video generation failed: \(message)")
        }
    }

    var aggregatedNativeMTPCapability: NativeMTPCapability? {
        if let loadedModelName,
           let loaded = modelLibraryRows.first(where: { $0.id == loadedModelName }) {
            return loaded.nativeMTP
        }
        return modelLibraryRows.first?.nativeMTP
    }

    var effectiveNativeMTPConfiguration: NativeMTPConfiguration {
        NativeMTPConfiguration(
            mode: backendMode == .distributed ? nativeMTPMode : .off,
            maxDepth: 1,
            headPlacement: "replicated"
        )
    }

    func modelConfiguration(for modelID: String) -> ModelRuntimeConfiguration {
        modelConfigurations[modelID, default: .default]
    }

    func updateModelConfiguration(_ configuration: ModelRuntimeConfiguration, for modelID: String) {
        modelConfigurations[modelID] = configuration.validated()
        if let encoded = try? JSONEncoder().encode(modelConfigurations) {
            userDefaults.set(encoded, forKey: modelConfigurationsKey)
        }
        appendLog("Updated runtime configuration for \(modelID).")
    }

    func selectChatModel(_ modelID: String) {
        guard !isChatRunning else { return }
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized == "tokenity-auto" || availableChatModelIDs.contains(normalized) else { return }
        chatSelectedModelID = normalized
        syncActiveChatSession()
    }

    func setChatRoutePolicy(_ policy: ChatRoutePolicy) {
        guard !isChatRunning else { return }
        chatRoutePolicy = policy
        syncActiveChatSession()
    }

    func setChatModelLocked(_ locked: Bool) {
        guard !isChatRunning else { return }
        locksChatModel = locked
        syncActiveChatSession()
    }

    func setResidentInstanceAllowsAuto(_ instanceID: String, allowed: Bool) {
        guard let managed = managedModelInstances[instanceID] else { return }
        // Preferences used to be keyed by model ID, which coupled every
        // replica's switch. Seed each sibling from that legacy value once,
        // then keep all subsequent choices instance-local.
        if let legacyPreference = residentAutoPreferences.removeValue(forKey: managed.modelID) {
            for sibling in managedModelInstances.values where sibling.modelID == managed.modelID {
                if residentAutoPreferences[sibling.instanceID] == nil {
                    residentAutoPreferences[sibling.instanceID] = legacyPreference
                }
            }
        }
        residentAutoPreferences[instanceID] = allowed
        persistResidentAutoPreferences()
        objectWillChange.send()
    }

    func useResidentModelInChat(_ instanceID: String) {
        guard let managed = managedModelInstances[instanceID], managed.isRoutable else { return }
        selectChatModel(managed.modelID)
        selectedSection = .chat
    }

    func stopResidentModelInstance(_ instanceID: String) async {
        guard let managed = managedModelInstances[instanceID] else { return }
        if activeChatRoutedInstanceID == instanceID {
            let cancelled = cancelActiveChat(
                message: "Generation stopped because its model instance was stopped.",
                logReason: "resident instance stop"
            )
            if let cancelled { await cancelled.value }
        }
        do {
            try await cleanupInstanceOrLegacy(instanceID: instanceID, allowsGlobalFallback: false)
            managedModelInstances.removeValue(forKey: instanceID)
            residentAutoPreferences.removeValue(forKey: instanceID)
            persistResidentAutoPreferences()
            if !managedModelInstances.values.contains(where: { $0.modelID == managed.modelID }) {
                modelLoadStates[managed.modelID] = .notLoaded
            }
            recomputeManagedModelLoadStates()
            if activeModelInstanceID == instanceID {
                restoreActiveManagedInstance()
            }
            appendLog("Stopped resident model instance \(managed.modelID) (\(instanceID)).")
        } catch {
            appendLog("Could not stop resident model instance \(instanceID): \(userFacingMessage(for: error))")
        }
    }

    func setResidentInstanceKeepsResident(_ instanceID: String, keepsResident: Bool) {
        guard !keepsResident else { return }
        Task { await stopResidentModelInstance(instanceID) }
    }

    func newChatSession() {
        guard !isChatRunning else { return }
        syncActiveChatSession()
        let session = ChatSession.fresh()
        chatSessions.insert(session, at: 0)
        activeChatSessionID = session.id
        chatMessages = session.messages
        chatMetrics = session.metrics
        chatSelectedModelID = session.selectedModelID ?? "tokenity-auto"
        chatRoutePolicy = session.routePolicy ?? .balanced
        locksChatModel = session.locksModel ?? false
        chatInput = ""
        chatScrollRevision += 1
        chatComposerFocusRevision += 1
        persistChatSessions()
    }

    func selectChatSession(_ sessionID: UUID) {
        guard !isChatRunning, sessionID != activeChatSessionID else { return }
        syncActiveChatSession()
        guard let session = chatSessions.first(where: { $0.id == sessionID }) else { return }
        activeChatSessionID = session.id
        chatMessages = session.messages
        chatMetrics = session.metrics
        chatSelectedModelID = session.selectedModelID ?? "tokenity-auto"
        chatRoutePolicy = session.routePolicy ?? .balanced
        locksChatModel = session.locksModel ?? false
        chatInput = ""
        chatScrollRevision += 1
        chatComposerFocusRevision += 1
    }

    func deleteChatSession(_ sessionID: UUID) {
        guard !isChatRunning else { return }
        let deletedIndex = chatSessions.firstIndex(where: { $0.id == sessionID })
        chatSessions.removeAll { $0.id == sessionID }
        if chatSessions.isEmpty {
            let session = ChatSession.fresh()
            chatSessions = [session]
        }
        if sessionID == activeChatSessionID,
           !chatSessions.isEmpty {
            let nextIndex = min(deletedIndex ?? 0, chatSessions.count - 1)
            let next = chatSessions[nextIndex]
            activeChatSessionID = next.id
            chatMessages = next.messages
            chatMetrics = next.metrics
            chatSelectedModelID = next.selectedModelID ?? "tokenity-auto"
            chatRoutePolicy = next.routePolicy ?? .balanced
            locksChatModel = next.locksModel ?? false
            chatInput = ""
            chatScrollRevision += 1
            chatComposerFocusRevision += 1
        }
        persistChatSessions()
    }

    func renameChatSession(_ sessionID: UUID, title: String) {
        let clean = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty,
              let index = chatSessions.firstIndex(where: { $0.id == sessionID })
        else { return }
        chatSessions[index].title = String(clean.prefix(80))
        chatSessions[index].titleWasEdited = true
        chatSessions[index].updatedAt = Date()
        chatSessions.sort { $0.updatedAt > $1.updatedAt }
        persistChatSessions()
    }

    func editChatMessage(_ messageID: UUID) {
        guard !isChatRunning,
              let index = chatMessages.firstIndex(where: { $0.id == messageID }),
              chatMessages[index].role == .user
        else { return }
        let prompt = chatMessages[index].content
        chatMessages.removeSubrange(index...)
        chatInput = prompt
        chatScrollRevision += 1
        chatComposerFocusRevision += 1
        syncActiveChatSession()
    }

    func regenerateAssistantMessage(_ messageID: UUID) {
        guard !isChatRunning,
              let assistantIndex = chatMessages.firstIndex(where: { $0.id == messageID }),
              chatMessages[assistantIndex].role == .assistant,
              let userIndex = chatMessages[..<assistantIndex].lastIndex(where: { $0.role == .user })
        else { return }
        let prompt = chatMessages[userIndex].content
        chatMessages.removeSubrange(userIndex...)
        chatInput = prompt
        syncActiveChatSession()
        beginSendingChatMessage()
    }

    func regenerateAssistantMessageWithAnotherModel(_ messageID: UUID) {
        guard !isChatRunning,
              let assistantIndex = chatMessages.firstIndex(where: { $0.id == messageID }),
              chatMessages[assistantIndex].role == .assistant,
              let userIndex = chatMessages[..<assistantIndex].lastIndex(where: { $0.role == .user })
        else { return }
        let currentModelID = chatMessages[assistantIndex].routedModelID
            ?? chatMessages[assistantIndex].modelName
        let alternatives = Array(
            Set(
                residentModelInstances
                    .filter { $0.allowsAuto && ($0.isReady || $0.isBusy) }
                    .map(\.modelID)
                    .filter { $0 != currentModelID }
            )
        ).sorted()
        guard !alternatives.isEmpty else {
            appendLog("No alternative resident model is currently available.")
            return
        }
        nextChatRouteConstraints = [
            "topic_changed": .bool(true),
            "allowed_model_ids": .strings(alternatives),
        ]
        chatSelectedModelID = "tokenity-auto"
        locksChatModel = false
        let prompt = chatMessages[userIndex].content
        chatMessages.removeSubrange(userIndex...)
        chatInput = prompt
        syncActiveChatSession()
        beginSendingChatMessage()
    }

    func testExternalAPI() async {
        guard openAIAPIBaseURL.hasPrefix("http"),
              let url = URL(string: "\(openAIAPIBaseURL)/models") else {
            apiAccessStatus = "Unavailable"
            return
        }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 5
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            _ = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
            apiAccessStatus = "Reachable"
        } catch {
            apiAccessStatus = "Not ready"
        }
    }

    var modelLibraryRows: [ModelLibraryRow] {
        let grouped = Dictionary(grouping: selectedNodes.flatMap { node in
            node.models.map { (node, $0) }
        }, by: { $0.1.id })

        var rows = grouped.keys.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }.map { modelID in
            let entries = grouped[modelID] ?? []
            let nodeNames = entries.map { $0.0.displayName }.sorted()
            let modelEntries = entries.map(\.1)
            let representative = modelEntries.first
            let path = representative?.path ?? "\(modelRoot)/\(modelID)"
            let nativeMTP = aggregateNativeMTPCapability(
                modelEntries.compactMap(\.nativeMTP),
                expectedCount: entries.count
            )
            let draftOnlyEntry = modelEntries.first {
                $0.standaloneLoadable == false
                    || $0.modelType?.lowercased() == "qwen3_5_mtp"
                    || $0.architecture?.lowercased().contains("qwen3_5mtp") == true
            }
            let distributedBlockedEntry = modelEntries.first {
                $0.distributedLoadable == false || $0.modelType?.lowercased() == "qwen3_moe"
            }
            return ModelLibraryRow(
                id: modelID,
                displayName: modelID,
                modality: .language,
                nodes: nodeNames,
                availability: "\(nodeNames.count)/\(selectedNodes.count) selected Macs",
                loadState: modelLoadStates[modelID, default: .notLoaded],
                representativePath: path,
                format: modelEntries.compactMap(\.format).first,
                quantization: modelEntries.compactMap(\.quantization).first,
                sizeBytes: modelEntries.compactMap(\.sizeBytes).max(),
                architecture: modelEntries.compactMap(\.architecture).first,
                shardCount: modelEntries.compactMap(\.shardCount).max(),
                nativeMTP: nativeMTP,
                modelType: modelEntries.compactMap(\.modelType).first,
                standaloneLoadable: draftOnlyEntry == nil,
                loadBlockReason: draftOnlyEntry?.loadBlockReason ?? (draftOnlyEntry == nil ? nil : "Qwen3.5 MTP weights are a speculative-decoding draft model and cannot be loaded as a standalone chat model. Load the matching Qwen3.5 base model instead."),
                distributedLoadable: distributedBlockedEntry == nil,
                distributedLoadBlockReason: distributedBlockedEntry?.distributedLoadBlockReason
                    ?? (distributedBlockedEntry == nil ? nil : "The current Tokenity runtime does not support this model in two-Mac mode. Choose Load on one Mac instead.")
            )
        }
        let configuredVideoNodeCount = [h3CoordinatorAgentURL, h3WorkerAgentURL]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .count
        let onlineVideoNodes = videoNodes.filter(\.isOnline)
        rows.append(
            ModelLibraryRow(
                id: Self.h3ModelID,
                displayName: "MiniMax H3",
                modality: .video,
                nodes: videoNodes.map(\.displayName),
                availability: "\(onlineVideoNodes.count)/\(configuredVideoNodeCount) H3 Macs",
                loadState: h3ModelLoadState,
                representativePath: h3ModelPath,
                format: "MLX native",
                quantization: "8-bit · group 64",
                sizeBytes: nil,
                architecture: "MiniMax H3 TP2",
                shardCount: videoNodes.count == 2 ? 2 : nil,
                nativeMTP: nil,
                modelType: "minimax_h3",
                standaloneLoadable: true,
                loadBlockReason: nil,
                distributedLoadable: true,
                distributedLoadBlockReason: nil
            )
        )
        return rows
    }

    private var h3ModelLoadState: ModelLoadState {
        switch videoRuntimeState {
        case .stopped: return .notLoaded
        case .starting: return .loading
        case .ready: return .loaded
        case .stopping: return .unloading
        case .failed: return .failed
        }
    }

    func canLoadModel(_ row: ModelLibraryRow) -> Bool {
        guard row.standaloneLoadable, !isModelTransitioning else { return false }
        if row.modality == .video {
            return videoRuntimeReadinessIssues.isEmpty
        }
        return phase == .running && row.nodes.count >= selectedNodes.count
    }

    func modelLoadHelp(for row: ModelLibraryRow) -> String {
        if !row.standaloneLoadable {
            return row.loadBlockReason ?? "This checkpoint cannot be loaded independently."
        }
        if row.modality == .video {
            let issues = videoRuntimeReadinessIssues
            return issues.isEmpty
                ? "Start the MiniMax H3 video runtime"
                : issues.joined(separator: " ")
        }
        if phase != .running { return "Create a text cluster before loading this model" }
        if row.nodes.count < selectedNodes.count {
            return "The model must be available on every selected Mac"
        }
        return "Load model"
    }

    var modelLoadRequiredNodeCount: Int {
        backendMode == .singleNode ? 1 : selectedNodes.count
    }

    func modelLoadTargetSummary(for row: ModelLibraryRow) -> String {
        if row.modality == .video {
            return videoTopologySummary
        }
        if let resident = residentModelInstances.first(where: { $0.modelID == row.id }),
           !resident.selectedNodes.isEmpty {
            let names = resident.selectedNodes.map(nodeDisplayName(for:))
            return "\(names.count)/\(names.count) load target\(names.count == 1 ? "" : "s") · \(names.joined(separator: ", "))"
        }
        if backendMode == .singleNode, let target = plannedNodes.first {
            return "1/1 load target · \(target.displayName)"
        }
        return "\(row.availability) · \(row.nodes.joined(separator: ", "))"
    }

    func topologyIssue(for row: ModelLibraryRow) -> String? {
        guard row.modality == .language,
              backendMode == .distributed,
              selectedNodes.count > 1,
              !row.distributedLoadable
        else { return nil }
        return row.distributedLoadBlockReason
            ?? "The current Tokenity runtime does not support this model in two-Mac mode. Choose Load on one Mac instead."
    }

    func loadOnOneMac(_ row: ModelLibraryRow) {
        backendMode = .singleNode
        modelLoadMessage = "Loading \(row.displayName) on \(plannedNodes.first?.displayName ?? "one Mac")..."
        beginLoadingModel(row)
    }

    func refreshVideoNodes() async {
        let endpoints = [
            H3VideoEndpoint(
                id: "h3-mac-a",
                hostname: "Mac A",
                user: "unknown",
                agentURL: h3CoordinatorAgentURL.trimmingCharacters(in: .whitespacesAndNewlines)
            ),
            H3VideoEndpoint(
                id: "h3-mac-b",
                hostname: "Mac B",
                user: "unknown",
                agentURL: h3WorkerAgentURL.trimmingCharacters(in: .whitespacesAndNewlines)
            ),
        ].filter { !$0.agentURL.isEmpty }

        var sampledNodes: [TokenityNode] = []
        for endpoint in endpoints {
            let previousNode = videoNodes.first(where: { $0.id == endpoint.id })
            let startedAt = Date()
            guard let baseURL = URL(string: endpoint.agentURL),
                  let scheme = baseURL.scheme?.lowercased(),
                  ["http", "https"].contains(scheme),
                  baseURL.host != nil,
                  endpoint.id != "h3-mac-b" || !Self.isLoopbackAgentURL(baseURL)
            else {
                sampledNodes.append(offlineVideoNode(for: endpoint, previousNode: previousNode))
                continue
            }

            do {
                var request = URLRequest(url: baseURL.appendingPathComponent("/v1/node/info"))
                request.timeoutInterval = 4
                let (data, response) = try await dataTransport(request)
                try validate(response, data: data)
                let info = try JSONDecoder().decode(NodeInfoResponse.self, from: data)
                sampledNodes.append(
                    TokenityNode(
                        id: endpoint.id,
                        hostname: info.hostname,
                        user: info.user,
                        agentURL: endpoint.agentURL,
                        ips: info.ips,
                        architecture: info.architecture,
                        pythonPath: info.pythonPath,
                        mlxVersion: info.mlxVersion,
                        mlxLMVersion: info.mlxLMVersion,
                        tokenityVersion: info.tokenityVersion,
                        machineID: info.machineID,
                        tokenityCodeRevision: info.tokenityCodeRevision,
                        agentContract: info.agentContract,
                        rdma: info.rdma,
                        roles: info.processRoles,
                        memory: info.memory ?? .unknown,
                        models: previousNode?.models ?? [],
                        isOnline: true,
                        clusterRuntime: info.clusterRuntime,
                        clusterRuntimes: info.clusterRuntimes ?? [],
                        agentLatencyMilliseconds: Date().timeIntervalSince(startedAt) * 1_000,
                        lastAgentResponseAt: Date(),
                        agentError: nil,
                        consecutiveAgentFailures: 0,
                        modelInstances: info.instances ?? []
                    )
                )
            } catch {
                sampledNodes.append(offlineVideoNode(for: endpoint, previousNode: previousNode))
            }
        }

        videoNodes = sampledNodes
        synchronizeVideoRuntime(from: sampledNodes)
        await refreshVideoRuntimeQuorumIfNeeded()
    }

    func scanModels() async {
        isScanningModels = true
        modelScanSummary = "Scanning selected Macs..."
        defer { isScanningModels = false }

        var updatedNodes = nodes
        var scanned = 0
        var found = 0

        for node in selectedNodes {
            guard let index = updatedNodes.firstIndex(where: { $0.id == node.id }) else { continue }
            let fetched = await fetchModels(for: node)
            if let fetched {
                updatedNodes[index].models = fetched
                updatedNodes[index].isOnline = true
                scanned += 1
                found += fetched.count
            } else if !updatedNodes[index].models.isEmpty {
                scanned += 1
                found += updatedNodes[index].models.count
            }
        }

        nodes = updatedNodes
        await refreshVideoNodes()
        let videoOnline = videoNodes.filter(\.isOnline).count
        let textSummary = scanned == 0
            ? "Using saved text inventory"
            : "\(found) text model(s) across \(scanned) selected Mac(s)"
        modelScanSummary = "\(textSummary) · MiniMax H3 \(videoOnline)/\(videoNodes.count)"
        appendLog("Model library refreshed.")
        rebuildLaunchPreview()
    }

    var isStatusMonitoring: Bool {
        statusMonitoringTask != nil
    }

    var isNodeDiscoveryMonitoring: Bool {
        nodeDiscoveryTask != nil
    }

    func startNodeDiscoveryMonitoring(interval: Duration = .seconds(30)) {
        guard nodeDiscoveryTask == nil else { return }
        nodeDiscoveryTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.discoverNodes()
                guard !Task.isCancelled else { return }
                try? await Task.sleep(for: interval)
            }
        }
    }

    func stopNodeDiscoveryMonitoring() {
        nodeDiscoveryTask?.cancel()
        nodeDiscoveryTask = nil
    }

    func discoverNodes() async {
        guard !isDiscoveringNodes else { return }
        isDiscoveringNodes = true
        nodeDiscoverySummary = "Scanning local subnets for Tokenity Node Agents…"
        defer {
            isDiscoveringNodes = false
            lastNodeDiscoveryAt = Date()
        }

        let seedOrigins = (
            nodes.compactMap { URL(string: $0.agentURL) }
                + [h3CoordinatorAgentURL, h3WorkerAgentURL].compactMap(URL.init(string:))
        )
        let discoveries = await nodeDiscoveryTransport(seedOrigins)
        let preferred = preferredControlDiscoveries(from: discoveries)
        guard !preferred.isEmpty else {
            nodeDiscoverySummary = "No Node Agents found; saved endpoints were kept"
            return
        }

        var reboundCount = 0
        var addedCount = 0
        for discovery in preferred {
            switch mergeDiscoveredNode(discovery) {
            case .rebound: reboundCount += 1
            case .added: addedCount += 1
            case .unchanged: break
            }
        }
        nodeTopologyRevision += 1
        persistNodeEndpointOverrides()
        rebuildLaunchPreview()
        nodeDiscoverySummary = "Found \(preferred.count) Mac(s) automatically"
        if reboundCount > 0 || addedCount > 0 {
            appendLog(
                "LAN discovery updated \(reboundCount) saved endpoint(s) and added \(addedCount) Mac(s)."
            )
        }
    }

    @discardableResult
    func connectNode(agentURL rawValue: String) async -> Bool {
        guard phase == .stopped else {
            nodeDiscoverySummary = "Stop the cluster before changing a Node Agent endpoint"
            return false
        }
        guard !isConnectingNode else { return false }
        guard let baseURL = Self.normalizedAgentBaseURL(rawValue) else {
            nodeDiscoverySummary = "Enter a valid HTTP or HTTPS Node Agent URL"
            return false
        }

        isConnectingNode = true
        nodeDiscoverySummary = "Connecting to \(baseURL.host ?? "Node Agent")…"
        defer {
            isConnectingNode = false
            lastNodeDiscoveryAt = Date()
        }

        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/node/info"))
            request.timeoutInterval = 5
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            let info = try JSONDecoder().decode(NodeInfoResponse.self, from: data)
            let discovery = DiscoveredNodeEndpoint(
                agentURL: baseURL.absoluteString,
                info: info
            )
            let result = mergeDiscoveredNode(discovery)
            nodeTopologyRevision += 1
            persistNodeEndpointOverrides()
            rebuildLaunchPreview()
            let action = result == .added ? "added" : "connected"
            nodeDiscoverySummary = "\(info.hostname) \(action) at \(baseURL.absoluteString)"
            appendLog("Node Agent \(action): \(info.hostname) at \(baseURL.absoluteString).")
            return true
        } catch {
            nodeDiscoverySummary = "Could not connect to \(baseURL.absoluteString): \(error.localizedDescription)"
            return false
        }
    }

    func startStatusMonitoring(interval: Duration = .seconds(2)) {
        guard statusMonitoringTask == nil else { return }
        statusMonitoringStartCount += 1
        statusMonitoringTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshSelectedNodeStatus(showsActivity: false)
                guard !Task.isCancelled else { return }
                try? await Task.sleep(for: interval)
            }
        }
    }

    func stopStatusMonitoring() {
        statusMonitoringTask?.cancel()
        statusMonitoringTask = nil
    }

    func refreshSelectedNodeStatus(showsActivity: Bool = true) async {
        guard !statusRefreshInFlight else { return }
        statusRefreshInFlight = true
        let refreshModelEpoch = modelOperationEpoch
        let topologyRevisionAtStart = nodeTopologyRevision
        if showsActivity {
            isRefreshingStatus = true
        }
        defer {
            statusRefreshInFlight = false
            if showsActivity {
                isRefreshingStatus = false
            }
            lastStatusRefreshAt = Date()
        }

        await renewModelLeasesIfNeeded()
        await renewVideoLeaseIfNeeded()
        if selectedSection == .video || videoRuntimeInstanceID != nil {
            await refreshVideoNodes()
        }
        // A single active stream already proves its only model service is alive.
        // Once siblings exist, continue polling so a non-active instance cannot
        // remain falsely loaded while another model is generating.
        if !showsActivity, isChatRunning, managedModelInstances.count <= 1 {
            return
        }

        var updatedNodes = nodes
        for node in selectedNodes {
            guard let index = updatedNodes.firstIndex(where: { $0.id == node.id }) else { continue }
            let previousNode = updatedNodes[index]
            let coreHealth = await fetchAgentCoreHealth(for: node)
            let watchdogStatus = await fetchAgentWatchdogStatus(for: node)
            let telemetryAge = Date().timeIntervalSince(previousNode.lastAgentResponseAt ?? .distantPast)
            let telemetryInterval = previousNode.memory.totalBytes == nil ? 2.0 : 10.0
            let capturesVolatileTelemetry = showsActivity
                || !previousNode.isOnline
                || previousNode.lastAgentResponseAt == nil
                || telemetryAge >= telemetryInterval
            let startedAt = Date()
            if let info = await fetchNodeInfo(for: node) {
                updatedNodes[index].hostname = info.hostname
                updatedNodes[index].user = info.user
                updatedNodes[index].ips = info.ips
                updatedNodes[index].architecture = info.architecture
                updatedNodes[index].pythonPath = info.pythonPath
                updatedNodes[index].mlxVersion = info.mlxVersion
                updatedNodes[index].mlxLMVersion = info.mlxLMVersion
                updatedNodes[index].tokenityVersion = info.tokenityVersion
                updatedNodes[index].machineID = info.machineID
                updatedNodes[index].tokenityCodeRevision = info.tokenityCodeRevision
                updatedNodes[index].agentContract = info.agentContract
                updatedNodes[index].roles = stableRoles(
                    info.processRoles,
                    preservingVolatileFieldsFrom: previousNode.roles,
                    capturesVolatileTelemetry: capturesVolatileTelemetry
                )
                if capturesVolatileTelemetry {
                    updatedNodes[index].memory = info.memory ?? updatedNodes[index].memory
                }
                updatedNodes[index].rdma = info.rdma
                updatedNodes[index].clusterRuntime = info.clusterRuntime
                updatedNodes[index].clusterRuntimes = info.clusterRuntimes ?? []
                updatedNodes[index].modelInstances = info.instances ?? updatedNodes[index].modelInstances
                updatedNodes[index].isOnline = true
                updatedNodes[index].consecutiveAgentFailures = 0
                if capturesVolatileTelemetry {
                    updatedNodes[index].agentLatencyMilliseconds = Date().timeIntervalSince(startedAt) * 1_000
                    updatedNodes[index].lastAgentResponseAt = Date()
                }
                updatedNodes[index].agentError = nil
            } else if let status = await fetchStatus(for: node) {
                updatedNodes[index].roles = stableRoles(
                    status.roles,
                    preservingVolatileFieldsFrom: previousNode.roles,
                    capturesVolatileTelemetry: capturesVolatileTelemetry
                )
                if capturesVolatileTelemetry {
                    updatedNodes[index].memory = status.memory ?? updatedNodes[index].memory
                }
                updatedNodes[index].clusterRuntime = status.clusterRuntime
                updatedNodes[index].clusterRuntimes = status.clusterRuntimes ?? []
                updatedNodes[index].modelInstances = status.instances ?? updatedNodes[index].modelInstances
                updatedNodes[index].isOnline = true
                updatedNodes[index].consecutiveAgentFailures = 0
                if capturesVolatileTelemetry {
                    updatedNodes[index].agentLatencyMilliseconds = Date().timeIntervalSince(startedAt) * 1_000
                    updatedNodes[index].lastAgentResponseAt = Date()
                }
                updatedNodes[index].agentError = nil
            } else if let coreHealth {
                // The lightweight liveness endpoint is sufficient proof that
                // the Agent event loop is responsive. Keep the last model and
                // topology snapshot instead of flashing the whole Mac offline
                // because a heavier telemetry endpoint timed out.
                updatedNodes[index].isOnline = coreHealth.status != "unhealthy"
                updatedNodes[index].consecutiveAgentFailures = 0
                updatedNodes[index].lastAgentResponseAt = Date()
                updatedNodes[index].agentError = coreHealth.status == "unhealthy"
                    ? "The Node Agent responded but cannot safely provide control service."
                    : nil
            } else {
                let failures = previousNode.consecutiveAgentFailures + 1
                updatedNodes[index].consecutiveAgentFailures = failures
                if previousNode.isOnline && failures < 2 {
                    // Keep the last verified topology through one missed poll.
                    // Two consecutive failures are required before publishing
                    // Offline and changing the global status icon.
                    updatedNodes[index].isOnline = true
                    updatedNodes[index].agentError = nil
                } else {
                    updatedNodes[index].isOnline = false
                    updatedNodes[index].agentLatencyMilliseconds = nil
                    updatedNodes[index].agentError = "Node Agent did not respond within the health-check timeout."
                }
            }
            applyAgentHealth(
                coreHealth,
                watchdog: watchdogStatus,
                previous: previousNode,
                to: &updatedNodes[index]
            )
        }
        if nodeTopologyRevision == topologyRevisionAtStart, nodes != updatedNodes {
            nodes = updatedNodes
        }
        guard refreshModelEpoch == modelOperationEpoch else {
            rebuildLaunchPreview()
            return
        }
        await recoverManagedModelInstancesFromAgent(expectedEpoch: refreshModelEpoch)
        guard refreshModelEpoch == modelOperationEpoch else { return }
        if !managedModelInstances.isEmpty {
            await refreshManagedModelInstanceHealth(expectedEpoch: refreshModelEpoch)
        } else {
            adoptExternallyRunningServiceIfNeeded()
        }
        guard refreshModelEpoch == modelOperationEpoch else { return }
        if managedModelInstances.isEmpty,
           loadedServiceModelName != nil || loadedModelName != nil {
            do {
                if let readiness = try await fetchModelReadiness() {
                    guard refreshModelEpoch == modelOperationEpoch else { return }
                    let quorum = readiness.phase == "ready"
                        ? try await fetchInstanceQuorum()
                        : nil
                    guard refreshModelEpoch == modelOperationEpoch else { return }
                    if let quorum {
                        applyRuntimeMemory(from: quorum)
                    } else if let coordinatorIndex = nodes.firstIndex(where: { $0.id == coordinatorID }) {
                        nodes[coordinatorIndex].runtimeMemory = readiness.memory
                    }
                    var nextNativeMTPRuntime = readiness.nativeMTP ?? legacyNativeMTPFallback
                    if !showsActivity,
                       var next = nextNativeMTPRuntime,
                       let current = nativeMTPRuntime {
                        next.proposedTokens = current.proposedTokens
                        next.acceptedTokens = current.acceptedTokens
                        next.acceptanceRate = current.acceptanceRate
                        nextNativeMTPRuntime = next
                    }
                    if nativeMTPRuntime != nextNativeMTPRuntime {
                        nativeMTPRuntime = nextNativeMTPRuntime
                    }
                    let isUnverifiedExternalDiscovery = activeModelLoadID == nil
                        && loadedModelName == nil
                    if isUnverifiedExternalDiscovery,
                       readiness.phase == "ready",
                       readiness.readyEvidence?.verifiesInference == true,
                       let identifier = loadedServiceModelName {
                        var states = modelLoadStates
                        for key in states.keys where key != identifier {
                            states[key] = .notLoaded
                        }
                        states[identifier] = .loaded
                        modelLoadStates = states
                        activeLoadedModelID = identifier
                        if let instanceID = activeModelInstanceID {
                            managedModelInstances[instanceID] = ManagedModelInstance(
                                modelID: identifier,
                                serviceModelName: identifier,
                                instanceID: instanceID,
                                apiBaseURL: activeModelServiceBaseURL,
                                backendRole: loadedBackendRole ?? "distributed-openai"
                            )
                        }
                        modelLoadProgress = 1
                        modelLoadMessage = "\(identifier) is loaded."
                        appendLog("Verified externally running model service: \(identifier).")
                    }
                    let nextServerHealth: ServerHealthState
                    switch readiness.phase {
                    case "ready" where isUnverifiedExternalDiscovery
                        && readiness.readyEvidence?.verifiesInference != true:
                        nextServerHealth = .starting(
                            "Waiting for the model runtime to publish verified one-token warmup evidence"
                        )
                    case "ready" where activeModelInstanceID == nil:
                        // Legacy Agents have no managed instance/quorum
                        // endpoint. A Tokenity-managed load reaches this state
                        // only after the real streaming inference probe passes.
                        nextServerHealth = .ready
                    case "ready" where quorum?.ready == true:
                        nextServerHealth = .ready
                    case "ready":
                        let detail = quorum?.issues.joined(separator: " ")
                        nextServerHealth = .error(
                            detail?.isEmpty == false
                                ? detail!
                                : "The planned rank quorum is incomplete."
                        )
                    case "failed":
                        nextServerHealth = .error(readiness.message ?? "The model service reported a failed readiness state.")
                    default:
                        nextServerHealth = .starting(readiness.message ?? readiness.phase.replacingOccurrences(of: "_", with: " ").capitalized)
                    }
                    if serverHealth != nextServerHealth {
                        serverHealth = nextServerHealth
                    }
                }
            } catch {
                let nextServerHealth = ServerHealthState.error("Service health check failed: \(userFacingMessage(for: error))")
                if serverHealth != nextServerHealth {
                    serverHealth = nextServerHealth
                }
            }
        } else if phase == .stopped {
            if serverHealth != .stopped {
                serverHealth = .stopped
            }
        }
        guard refreshModelEpoch == modelOperationEpoch else { return }
        reconcileExternallyStoppedService()
        rebuildLaunchPreview()
    }

    private func adoptExternallyRunningServiceIfNeeded() {
        // A process becomes visible to the status monitor before its readiness
        // and streaming probes finish. Never let background discovery turn an
        // in-progress Tokenity load green prematurely.
        guard activeModelLoadID == nil,
              !isModelTransitioning,
              loadedModelName == nil,
              let controller = coordinator,
              let activeRole = controller.roles.first(where: Self.isActiveInferenceProcess),
              let command = activeRole.command,
              let modelFlag = command.firstIndex(of: "--model"),
              command.indices.contains(modelFlag + 1)
        else { return }

        let discoveredPath = command[modelFlag + 1]
        let identifier: String
        if let identifierFlag = command.firstIndex(of: "--api-identifier"),
           command.indices.contains(identifierFlag + 1) {
            identifier = command[identifierFlag + 1]
        } else {
            identifier = URL(fileURLWithPath: discoveredPath).lastPathComponent
        }
        guard !identifier.isEmpty else { return }

        var states = modelLoadStates
        for key in states.keys {
            states[key] = .notLoaded
        }
        states[identifier] = .loading
        modelLoadStates = states
        modelPath = discoveredPath
        loadedBackendRole = activeRole.role
        loadedServiceModelName = identifier
        activeModelInstanceID = activeRole.instanceID
        if let portFlag = command.firstIndex(of: "--port"),
           command.indices.contains(portFlag + 1),
           let port = Int(command[portFlag + 1]),
           var components = URLComponents(string: controller.agentURL) {
            components.port = port
            components.path = "/v1"
            activeModelServiceBaseURL = components.url?.absoluteString
        }
        phase = .running
        modelLoadProgress = nil
        modelLoadMessage = "Verifying externally running model service: \(identifier)..."
        serverHealth = .starting("Checking the externally running model service")
        appendLog("Discovered externally running model service: \(identifier).")
    }

    private func stableRoles(
        _ incoming: [ProcessRole],
        preservingVolatileFieldsFrom current: [ProcessRole],
        capturesVolatileTelemetry: Bool
    ) -> [ProcessRole] {
        guard !capturesVolatileTelemetry else { return incoming }
        return incoming.map { role in
            guard let existing = current.first(where: {
                $0.role == role.role && $0.instanceID == role.instanceID
            }) else { return role }
            var stable = role
            stable.logTail = existing.logTail
            return stable
        }
    }

    func inferenceRolesForActiveModel(on node: TokenityNode) -> [ProcessRole] {
        let inferenceRoles = node.roles.filter { Self.inferenceRoleNames.contains($0.role) }
        guard let activeModelInstanceID else { return inferenceRoles }

        let exact = inferenceRoles.filter { $0.instanceID == activeModelInstanceID }
        if !exact.isEmpty {
            return exact
        }

        // A legacy Agent cannot attach instance IDs to process roles. Only
        // accept those unscoped roles when that Agent does not advertise
        // managed-instance support.
        guard node.agentContract?.supports("managed_instances") != true else { return [] }
        return inferenceRoles.filter { $0.instanceID == nil }
    }

    func clusterRuntimeForActiveModel(on node: TokenityNode) -> ClusterRuntimeStatus? {
        guard let activeModelInstanceID else { return node.clusterRuntime }
        if let exact = node.clusterRuntimes.first(where: {
            $0.instanceID == activeModelInstanceID
        }) {
            return exact
        }
        if let singleton = node.clusterRuntime,
           singleton.instanceID == activeModelInstanceID {
            return singleton
        }
        guard node.agentContract?.supports("instance_runtimes") != true else { return nil }
        guard node.agentContract?.supports("managed_instances") != true else { return nil }
        return node.clusterRuntime?.instanceID == nil ? node.clusterRuntime : nil
    }

    func toggleNodeSelection(_ node: TokenityNode) {
        guard canEditCluster else {
            appendLog("Stop the cluster before changing selected Macs.")
            return
        }

        var selection = selectedNodeIDs
        if selection.contains(node.id) {
            guard selection.count > 1 else {
                appendLog("At least one Mac must stay selected.")
                return
            }
            selection.remove(node.id)
        } else {
            selection.insert(node.id)
        }
        selectedNodeIDs = selection
        appendLog("\(node.displayName) \(selection.contains(node.id) ? "added to" : "removed from") the cluster selection.")
    }

    func beginLoadingModel(_ row: ModelLibraryRow) {
        guard activeModelLoadID == nil, pendingModelLoadID == nil, modelLoadTask == nil else {
            appendLog("Ignored duplicate Load for \(row.displayName); the existing operation remains authoritative.")
            return
        }
        modelOperationEpoch &+= 1
        let reservation = UUID()
        pendingModelLoadID = reservation
        modelLoadTask = Task { [weak self] in
            await self?.loadModel(row, reservedOperationID: reservation)
        }
    }

    func loadModel(_ row: ModelLibraryRow) async {
        await loadModel(row, reservedOperationID: nil)
    }

    private func loadModel(_ row: ModelLibraryRow, reservedOperationID: UUID?) async {
        if reservedOperationID == nil {
            modelOperationEpoch &+= 1
        }
        if let reservedOperationID, pendingModelLoadID != reservedOperationID {
            return
        }
        if row.modality == .video {
            pendingModelLoadID = nil
            defer { modelLoadTask = nil }
            await startVideoRuntime()
            return
        }
        guard phase == .running else {
            pendingModelLoadID = nil
            modelLoadTask = nil
            appendLog("Create a cluster before loading a model.")
            return
        }
        guard row.standaloneLoadable else {
            let message = row.loadBlockReason ?? "This checkpoint is draft-only and cannot be loaded as a standalone chat model."
            modelLoadMessage = message
            appendLog("Model load blocked: \(message)")
            pendingModelLoadID = nil
            modelLoadTask = nil
            return
        }
        if let message = topologyIssue(for: row) {
            modelLoadMessage = message
            appendLog("Model load blocked before launch: \(message)")
            pendingModelLoadID = nil
            modelLoadTask = nil
            return
        }
        if nativeMTPMode == .required,
           backendMode == .distributed,
           let capability = row.nativeMTP,
           capability.status != "supported",
           capability.status != "unknown" {
            let detail = capability.message ?? capability.displayStatus
            modelLoadMessage = "Native MTP required mode is incompatible: \(detail)"
            appendLog(modelLoadMessage)
            pendingModelLoadID = nil
            modelLoadTask = nil
            return
        }
        guard clusterControlBaseURL() != nil else {
            modelLoadMessage = "Selected cluster Mac is not reachable."
            appendLog("Model load blocked because the selected cluster Mac is not reachable.")
            pendingModelLoadID = nil
            modelLoadTask = nil
            return
        }
        let operationID = reservedOperationID ?? UUID()
        activeModelLoadID = operationID
        pendingModelLoadID = nil
        await refreshSelectedNodeStatus()
        let targetNodes = backendMode == .singleNode
            ? Array(plannedNodes.prefix(1))
            : selectedNodes
        let issues = readinessIssues(for: targetNodes)
        guard issues.isEmpty else {
            let prefix = connectionMode == .ring
                ? "The selected cluster is not ready"
                : "Thunderbolt RDMA is not ready"
            let message = "\(prefix): \(issues.joined(separator: " "))"
            modelLoadMessage = message
            appendLog("Model load blocked. \(message)")
            activeModelLoadID = nil
            modelLoadTask = nil
            return
        }
        if !managedModelInstances.isEmpty,
           !selectedAgentsSupportResidentModels {
            let message = TokenityTransportError.multiInstanceAgentUpgradeRequired.errorDescription
                ?? "The selected Node Agents do not support resident model instances."
            modelLoadMessage = message
            appendLog("Additional resident model load blocked: \(message)")
            activeModelLoadID = nil
            modelLoadTask = nil
            return
        }

        var states = modelLoadStates
        states[row.id] = .loading
        modelLoadStates = states
        modelPath = row.representativePath
        modelLoadMessage = "Loading \(row.displayName)..."
        modelLoadProgress = 0
        serverHealth = .starting("Loading \(row.displayName)")
        nativeMTPRuntime = nil
        legacyNativeMTPFallback = nil
        appendLog("Loading model: \(row.displayName).")
        let configuration = modelConfiguration(for: row.id)
        let requestedInstanceID = UUID().uuidString

        do {
            // Clean unknown legacy roles only when Tokenity is not already
            // managing an instance. Existing instances must remain isolated so
            // a second model can be admitted on the same Mac(s).
            if managedModelInstances.isEmpty {
                let previousLoaded = loadedModelName
                try await cleanupActiveInstanceOrLegacy()
                if let previousLoaded, previousLoaded != row.id {
                    states = modelLoadStates
                    states[previousLoaded] = .notLoaded
                    states[row.id] = .loading
                    modelLoadStates = states
                }
            }
            try ensureActiveModelLoad(operationID)
            let role = "distributed-openai"
            loadedBackendRole = role
            let startResponse = try await startBackendModel(
                row,
                role: role,
                configuration: configuration,
                operationID: operationID,
                instanceID: requestedInstanceID
            )
            activeModelInstanceID = startResponse?.instanceID
            activeModelServiceBaseURL = startResponse?.apiBaseURL
            try ensureActiveModelLoad(operationID)
            let serviceModelName = try await waitForModelService(
                modelName: row.displayName,
                role: role,
                operationID: operationID
            )
            try ensureActiveModelLoad(operationID)
            modelLoadProgress = max(modelLoadProgress ?? 0, 0.98)
            modelLoadMessage = "Verifying inference for \(row.displayName)..."
            try await probeModelService(modelName: serviceModelName)
            try ensureActiveModelLoad(operationID)
            if let quorum = try await fetchInstanceQuorum() {
                applyRuntimeMemory(from: quorum)
                guard quorum.ready else {
                    throw TokenityTransportError.backendExited(
                        quorum.issues.joined(separator: " ")
                    )
                }
            }

            states = modelLoadStates
            states[row.id] = .loaded
            modelLoadStates = states
            loadedBackendRole = role
            loadedServiceModelName = serviceModelName
            activeLoadedModelID = row.id
            if let instanceID = startResponse?.instanceID {
                let launchNodes = backendMode == .singleNode
                    ? Array(plannedNodes.prefix(1))
                    : plannedNodes
                managedModelInstances[instanceID] = ManagedModelInstance(
                    modelID: row.id,
                    serviceModelName: serviceModelName,
                    instanceID: instanceID,
                    apiBaseURL: startResponse?.apiBaseURL,
                    backendRole: role,
                    selectedNodes: launchNodes.map(\.id),
                    executionMode: backendMode == .singleNode ? "single" : "distributed",
                    connectionMode: backendMode == .singleNode ? "single" : connectionMode.cliValue
                )
            }
            activeModelLoadID = nil
            modelLoadTask = nil
            modelLoadProgress = 1
            modelLoadMessage = "\(row.displayName) is loaded."
            // A verified model service is authoritative evidence that the
            // logical cluster is running. This also repairs any stale phase
            // published by a status sample taken while the previous model was
            // being stopped during a legacy-Agent model switch.
            phase = .running
            serverHealth = .ready
            appendLog("Model loaded: \(row.displayName).")
        } catch is CancellationError {
            try? await cleanupInstanceOrLegacy(
                instanceID: requestedInstanceID,
                allowsGlobalFallback: managedModelInstances.isEmpty
            )
            if activeModelLoadID == operationID {
                activeModelLoadID = nil
                states = modelLoadStates
                states[row.id] = .notLoaded
                modelLoadStates = states
                restoreActiveManagedInstance()
                modelLoadMessage = "Model loading cancelled."
                appendLog("Model loading cancelled: \(row.displayName).")
            }
            modelLoadTask = nil
        } catch {
            try? await cleanupInstanceOrLegacy(
                instanceID: requestedInstanceID,
                allowsGlobalFallback: managedModelInstances.isEmpty
            )
            guard activeModelLoadID == operationID else { return }
            activeModelLoadID = nil
            modelLoadTask = nil
            states = modelLoadStates
            states[row.id] = .failed
            modelLoadStates = states
            restoreActiveManagedInstance()
            modelLoadProgress = nil
            let message = userFacingMessage(for: error)
            serverHealth = loadedModelName == nil ? .error(message) : .ready
            modelLoadMessage = message
            appendLog("Model load failed: \(message)")
        }
    }

    func stopModel(_ row: ModelLibraryRow) async {
        if row.modality == .video {
            modelOperationEpoch &+= 1
            pendingModelLoadID = nil
            modelLoadTask?.cancel()
            modelLoadTask = nil
            await stopVideoRuntime()
            return
        }
        // Publish unloading before the first suspension point. Otherwise the
        // status monitor can observe a stopped process while the model still
        // looks loaded and incorrectly collapse the logical cluster state.
        modelOperationEpoch &+= 1
        activeModelLoadID = nil
        pendingModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        var states = modelLoadStates
        states[row.id] = .unloading
        modelLoadStates = states
        modelLoadProgress = nil
        nativeMTPRuntime = nil
        modelLoadMessage = "Unloading \(row.displayName) and releasing memory on all Macs..."
        appendLog("Unloading model: \(row.displayName).")

        let stopsActiveChatModel = activeLoadedModelID == row.id
        let cancelledChatTask = stopsActiveChatModel
            ? cancelActiveChat(
                message: "Generation stopped because the model was unloaded.",
                logReason: "model unload"
            )
            : nil
        if let cancelledChatTask { await cancelledChatTask.value }
        let targetInstanceIDs = managedModelInstances.values
            .filter { $0.modelID == row.id }
            .map(\.instanceID)
            .sorted()
        let fallbackInstanceIDs = targetInstanceIDs.isEmpty
            ? [activeLoadedModelID == row.id ? activeModelInstanceID : nil].compactMap { $0 }
            : targetInstanceIDs
        if fallbackInstanceIDs.isEmpty {
            do {
                try await cleanupInstanceOrLegacy(
                    instanceID: nil,
                    allowsGlobalFallback: managedModelInstances.isEmpty
                )
            } catch {
                appendLog("Some model stop requests could not reach a selected Mac: \(userFacingMessage(for: error))")
            }
        } else {
            for instanceID in fallbackInstanceIDs {
                do {
                    try await cleanupInstanceOrLegacy(
                        instanceID: instanceID,
                        allowsGlobalFallback: managedModelInstances.count <= fallbackInstanceIDs.count
                    )
                } catch {
                    appendLog("Some model stop requests could not reach a selected Mac: \(userFacingMessage(for: error))")
                }
            }
        }
        for instanceID in targetInstanceIDs {
            managedModelInstances.removeValue(forKey: instanceID)
        }
        states = modelLoadStates
        states[row.id] = .notLoaded
        modelLoadStates = states
        if activeLoadedModelID == row.id {
            restoreActiveManagedInstance()
        }
        await refreshSelectedNodeStatus()
        modelLoadProgress = nil
        modelLoadMessage = loadedModelName.map { "\($0) remains loaded." } ?? "No model loaded"
        if loadedModelName == nil {
            serverHealth = .starting("Waiting for a model to load")
        }
        appendLog("Model stopped: \(row.displayName).")
    }

    func beginSendingChatMessage() {
        let prompt = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isChatRunning, chatTask == nil else { return }
        guard loadedModelName != nil, phase == .running else {
            appendLog("Chat is waiting for a running cluster and loaded model.")
            return
        }
        let taskID = UUID()
        chatTaskID = taskID
        chatTask = Task { [weak self] in
            await self?.sendChatMessage()
            self?.finishChatTask(taskID)
        }
    }

    func cancelChatGeneration() {
        cancelActiveChat(
            message: "Generation stopped by the user.",
            logReason: "user request"
        )
    }

    func sendChatMessage() async {
        let prompt = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isChatRunning else { return }
        guard phase == .running else {
            appendLog("Chat is waiting for a running cluster and loaded model.")
            return
        }
        let requestedAuto = isAutoChatSelection
        let selectedInstance = managedModelInstances.values
            .filter { $0.modelID == chatSelectedModelID && $0.isRoutable }
            .sorted { $0.instanceID < $1.instanceID }
            .first
        let allowedAutoInstances = residentModelInstances
            .filter { $0.allowsAuto && ($0.isReady || $0.isBusy) }
        let allowedAutoModels = Array(Set(allowedAutoInstances.map(\.modelID))).sorted()
        let allowedAutoInstanceIDs = allowedAutoInstances.map(\.id).sorted()
        let usesAutoRouting = requestedAuto && !allowedAutoModels.isEmpty
        let fallbackModelID: String?
        if requestedAuto || (selectedInstance == nil && managedModelInstances.isEmpty) {
            // A legacy Agent has one active service and no instance routing
            // contract. After a model switch, that verified active model is the
            // only valid manual target.
            fallbackModelID = loadedModelName
        } else {
            fallbackModelID = chatSelectedModelID
        }
        let hasLegacyManualFallback = fallbackModelID != nil && fallbackModelID == loadedModelName
        guard usesAutoRouting || selectedInstance != nil || hasLegacyManualFallback else {
            appendLog(requestedAuto
                ? "Auto routing is waiting for an eligible resident model."
                : "The selected model is not currently resident and routable.")
            return
        }
        let resolvedManualModelID = selectedInstance?.modelID ?? fallbackModelID ?? chatSelectedModelID
        let serviceModelName = usesAutoRouting
            ? "tokenity-auto"
            : (selectedInstance?.serviceModelName ?? loadedServiceModelName ?? resolvedManualModelID)
        let configuration = usesAutoRouting ? nil : modelConfiguration(for: resolvedManualModelID)
        var routeConstraints = nextChatRouteConstraints ?? [:]
        nextChatRouteConstraints = nil
        if usesAutoRouting {
            if routeConstraints["allowed_model_ids"] == nil {
                routeConstraints["allowed_model_ids"] = .strings(allowedAutoModels)
            }
            routeConstraints["allowed_instance_ids"] = .strings(allowedAutoInstanceIDs)
        }

        chatInput = ""
        chatMessages.append(ChatMessage(role: .user, content: prompt))
        let userIndex = chatMessages.count - 1
        chatMessages.append(
            ChatMessage(
                role: .assistant,
                content: "",
                generationState: .waiting,
                modelName: usesAutoRouting ? "Auto" : resolvedManualModelID
            )
        )
        let assistantIndex = chatMessages.count - 1
        let requestID = UUID()
        activeChatRequestID = requestID
        activeChatUserIndex = userIndex
        activeChatAssistantIndex = assistantIndex
        activeChatRoutedInstanceID = nil
        isChatRunning = true
        chatRoutingState = .selecting
        chatMetrics = .empty
        chatScrollRevision += 1
        defer {
            if activeChatRequestID == requestID {
                activeChatRequestID = nil
                activeChatUserIndex = nil
                activeChatAssistantIndex = nil
                activeChatRoutedInstanceID = nil
                chatTask = nil
                chatTaskID = nil
                isChatRunning = false
                if chatRoutingState == .selecting {
                    chatRoutingState = .idle
                }
            }
            syncActiveChatSession()
        }

        let start = Date()
        var firstTokenAt: Date?
        var tokenEstimate = 0

        do {
            var streamingAttempt = 0
            while true {
                do {
                    try await streamClusterChat(
                        serviceModelName: serviceModelName,
                        prompt: prompt,
                        configuration: configuration,
                        routeConstraints: routeConstraints.isEmpty ? nil : routeConstraints,
                        userIndex: userIndex,
                        assistantIndex: assistantIndex,
                        requestID: requestID,
                        start: start,
                        firstTokenAt: &firstTokenAt,
                        tokenEstimate: &tokenEstimate
                    )
                    break
                } catch is CancellationError {
                    throw CancellationError()
                } catch TokenityTransportError.repetitiveOutput {
                    throw TokenityTransportError.repetitiveOutput
                } catch {
                    guard streamingAttempt == 0,
                          !assistantMessageHasVisibleOutput(at: assistantIndex),
                          isExplicitStreamTransportFailure(error)
                    else { throw error }
                    streamingAttempt += 1
                    firstTokenAt = nil
                    tokenEstimate = 0
                    chatMessages[assistantIndex].generationState = .waiting
                    chatMessages[assistantIndex].statusMessage = "The first streaming connection was not ready; reconnecting…"
                    appendLog("The first chat stream ended before producing output; retrying the streaming connection once: \(userFacingMessage(for: error))")
                    try ensureActiveChatRequest(requestID)
                }
            }
            try ensureActiveChatRequest(requestID)
            finishChatMetrics(
                start: start,
                firstTokenAt: firstTokenAt,
                tokenEstimate: tokenEstimate,
                assistantIndex: assistantIndex
            )
        } catch is CancellationError {
            finishCancelledChatIfActive(
                requestID,
                message: "Generation stopped before completion.",
                logReason: "request cancellation"
            )
        } catch TokenityTransportError.repetitiveOutput {
            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            let notice = "Generation stopped because repeated output was detected."
            let content = chatMessages[assistantIndex].content
                .trimmingCharacters(in: .whitespacesAndNewlines)
            chatMessages[assistantIndex].content = content.isEmpty
                ? notice
                : "\(chatMessages[assistantIndex].content)\n\n\(notice)"
            chatMessages[assistantIndex].generationState = .repetitive
            chatMessages[assistantIndex].statusMessage = notice
            chatScrollRevision += 1
            finishChatMetrics(
                start: start,
                firstTokenAt: firstTokenAt,
                tokenEstimate: tokenEstimate,
                assistantIndex: assistantIndex
            )
            appendLog("Chat generation stopped automatically after repeated output was detected.")
        } catch let streamingError {
            if !assistantMessageHasVisibleOutput(at: assistantIndex),
               isExplicitStreamTransportFailure(streamingError) {
                do {
                    try ensureActiveChatRequest(requestID)
                    chatMessages[assistantIndex].generationState = .waiting
                    chatMessages[assistantIndex].statusMessage = "Both streaming connections were interrupted; retrying once without streaming."
                    try await completeClusterChat(
                        serviceModelName: serviceModelName,
                        configuration: configuration,
                        routeConstraints: routeConstraints.isEmpty ? nil : routeConstraints,
                        assistantIndex: assistantIndex,
                        requestID: requestID,
                        firstTokenAt: &firstTokenAt,
                        tokenEstimate: &tokenEstimate
                    )
                    try ensureActiveChatRequest(requestID)
                    finishChatMetrics(
                        start: start,
                        firstTokenAt: firstTokenAt,
                        tokenEstimate: tokenEstimate,
                        assistantIndex: assistantIndex
                    )
                    appendLog("Streaming chat connection failed, then recovered with a non-streaming response: \(userFacingMessage(for: streamingError))")
                    return
                } catch is CancellationError {
                    finishCancelledChatIfActive(
                        requestID,
                        message: "Generation stopped before completion.",
                        logReason: "request cancellation"
                    )
                    return
                } catch let fallbackError {
                    chatMessages[userIndex].includeInContext = false
                    chatMessages[assistantIndex].includeInContext = false
                    chatMessages[assistantIndex].thinking = ""
                    let detail = userFacingMessage(for: fallbackError)
                    chatMessages[assistantIndex].content = "The model request could not complete. \(detail)"
                    chatMessages[assistantIndex].generationState = .failed
                    chatMessages[assistantIndex].statusMessage = detail
                    if chatInput.isEmpty { chatInput = prompt }
                    appendLog("Chat streaming failed: \(userFacingMessage(for: streamingError))")
                    appendLog("Chat fallback failed: \(detail)")
                    return
                }
            }

            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            if assistantMessageHasVisibleOutput(at: assistantIndex) {
                if chatMessages[assistantIndex].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    chatMessages[assistantIndex].content = "The model returned reasoning but did not finish a final answer. Try again if you need the final response."
                }
            }
            chatMessages[assistantIndex].generationState = .failed
            chatMessages[assistantIndex].statusMessage = userFacingMessage(for: streamingError)
            if chatInput.isEmpty { chatInput = prompt }
            appendLog("Chat request could not complete: \(userFacingMessage(for: streamingError))")
        }

    }

    func createCluster() {
        // Connectivity is refreshed asynchronously immediately before model
        // loading. The UI still disables this action while an Agent is offline,
        // but the synchronous state transition must not trust an unrefreshed
        // sample snapshot (and remains directly testable with mocked Agents).
        let configurationIssues = launchPreview.readinessIssues.filter {
            !$0.hasSuffix("Node Agent is offline.")
        }
        if configurationIssues.isEmpty {
            phase = .launching
            serverHealth = .starting("Waiting for a model to load")
            appendLog("Cluster created with \(selectedNodes.count) selected Mac(s).")
            phase = .running
        } else {
            phase = .failed
            serverHealth = .error("Cluster creation is blocked by readiness issues.")
            appendLog("Cluster creation blocked. Review readiness warnings.")
        }
    }

    func stopCluster() async {
        modelOperationEpoch &+= 1
        phase = .stopping
        serverHealth = .starting("Stopping model service")
        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because the cluster was stopped.",
            logReason: "cluster stop"
        )
        if let cancelledChatTask { await cancelledChatTask.value }
        cancelVideoGeneration()
        if let videoTask { await videoTask.value }
        if let videoRuntimeInstanceID {
            try? await stopVideoInstance(videoRuntimeInstanceID)
        }
        videoRuntimeInstanceID = nil
        videoRuntimeState = .stopped
        videoProgressStage = "Waiting for a MiniMax H3 runtime"
        activeModelLoadID = nil
        pendingModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        do {
            try await cleanupAllModelRoles()
            markInferenceRolesStoppedLocally()
        } catch {
            appendLog("Cluster cleanup could not reach every selected Mac: \(userFacingMessage(for: error))")
        }
        resetModelLoadState(message: "No model loaded")
        appendLog("Cluster stopped.")
        phase = .stopped
        serverHealth = .stopped
    }

    func shutdownForApplicationTermination() async {
        stopStatusMonitoring()
        stopNodeDiscoveryMonitoring()
        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because Tokenity is closing.",
            logReason: "application termination"
        )
        if let cancelledChatTask { await cancelledChatTask.value }
        let residentInstanceIDs = Set(
            managedModelInstances.values.map(\.instanceID)
                + [activeModelInstanceID].compactMap { $0 }
        )
        if !residentInstanceIDs.isEmpty {
            // Give the replacement UI process enough time to start, discover
            // the Agent-owned instance, verify quorum, and resume the regular
            // 30-second heartbeat. The Agent remains the final lease authority.
            await renewModelLeases(
                instanceIDs: residentInstanceIDs,
                ttlSeconds: appRestartLeaseGraceSeconds
            )
        }
        cancelVideoGeneration()
        if let videoTask { await videoTask.value }
        if let videoRuntimeInstanceID {
            try? await stopVideoInstance(videoRuntimeInstanceID)
        }
        videoRuntimeInstanceID = nil
        videoRuntimeState = .stopped
        activeModelLoadID = nil
        pendingModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        // Closing Control is not an instruction to stop Agent-owned resident
        // instances. The next app process recovers them from managed instance
        // identity, runtime evidence, and gateway routes.
        await waitForPendingChatHistoryWrites()
    }

    func stop() {
        Task { await stopCluster() }
    }

    func restartCluster() async {
        appendLog("Recreate cluster requested.")
        await stopCluster()
        createCluster()
    }

    func restart() {
        Task { await restartCluster() }
    }

    func rebuildLaunchPreview() {
        let nodes = selectedNodes
        var readiness = readinessIssues(for: nodes)
        if backendMode == .distributed,
           nativeMTPMode == .required,
           let capability = aggregatedNativeMTPCapability,
           capability.status != "supported",
           capability.status != "unknown" {
            readiness.append(
                "Native MTP is required, but the selected model capability is \(capability.displayStatus.lowercased())."
            )
        }
        let readinessText = readiness.isEmpty ? "Ready to create" : "Needs attention"
        let summary = [
            LaunchSummaryItem(title: "Backend", value: backendMode.rawValue),
            LaunchSummaryItem(title: "Connection", value: connectionMode.rawValue),
            LaunchSummaryItem(title: "Selected Macs", value: "\(nodes.count)"),
            LaunchSummaryItem(title: "Native MTP", value: effectiveNativeMTPConfiguration.mode.title),
            LaunchSummaryItem(title: "Readiness", value: readinessText),
        ]
        var warnings = ["Tokenity Distributed Server verifies readiness with a real chat probe before marking a model loaded."]
        if backendMode == .distributed && nativeMTPMode == .auto {
            warnings.append("Native MTP Auto falls back to standard decoding when the model, checkpoint, or runtime is incompatible.")
        }
        let nextPreview = LaunchPreview(
            summary: summary,
            networkPlan: networkPlan(for: nodes),
            warnings: warnings,
            readinessIssues: readiness
        )
        if launchPreview != nextPreview {
            launchPreview = nextPreview
        }
    }

    private func fetchModels(for node: TokenityNode) async -> [ModelEntry]? {
        guard let baseURL = Self.normalizedAgentBaseURL(node.agentURL) else { return nil }
        var components = URLComponents(
            url: baseURL.appendingPathComponent("v1/node/models"),
            resolvingAgainstBaseURL: false
        )
        components?.queryItems = [URLQueryItem(name: "root", value: modelRoot)]
        guard let url = components?.url else { return nil }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(NodeModelsResponse.self, from: data).models
        } catch {
            return nil
        }
    }

    private func fetchStatus(for node: TokenityNode) async -> NodeStatusResponse? {
        guard let baseURL = Self.normalizedAgentBaseURL(node.agentURL) else { return nil }
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/node/status"))
            request.timeoutInterval = 4
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try JSONDecoder().decode(NodeStatusResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func offlineVideoNode(
        for endpoint: H3VideoEndpoint,
        previousNode: TokenityNode?
    ) -> TokenityNode {
        let failureCount = (previousNode?.consecutiveAgentFailures ?? 0) + 1
        if var previousNode, previousNode.isOnline, failureCount < 2 {
            previousNode.agentURL = endpoint.agentURL
            previousNode.consecutiveAgentFailures = failureCount
            return previousNode
        }
        let host = URL(string: endpoint.agentURL)?.host
        return TokenityNode(
            id: endpoint.id,
            hostname: endpoint.hostname,
            user: endpoint.user,
            agentURL: endpoint.agentURL,
            ips: host.map { [$0] } ?? [],
            architecture: "arm64",
            pythonPath: "-",
            mlxVersion: nil,
            mlxLMVersion: nil,
            tokenityVersion: "-",
            rdma: .empty,
            roles: [],
            memory: .unknown,
            models: previousNode?.models ?? [],
            isOnline: false,
            agentError: "MiniMax H3 Node Agent did not respond within the health-check timeout.",
            consecutiveAgentFailures: failureCount
        )
    }

    private func preferredControlDiscoveries(
        from discoveries: [DiscoveredNodeEndpoint]
    ) -> [DiscoveredNodeEndpoint] {
        let grouped = Dictionary(grouping: discoveries) { discovery in
            if let machineID = discovery.info.machineID, !machineID.isEmpty {
                return "machine:\(machineID)"
            }
            if let rdmaIP = discovery.info.rdma.thunderboltIP, !rdmaIP.isEmpty {
                return "rdma:\(rdmaIP)"
            }
            return "node:\(discovery.info.nodeID)"
        }
        let savedEndpoints = Set(nodes.map(\.agentURL))
        return grouped.values.compactMap { candidates in
            candidates.max { lhs, rhs in
                discoveryControlScore(lhs, savedEndpoints: savedEndpoints)
                    < discoveryControlScore(rhs, savedEndpoints: savedEndpoints)
            }
        }.sorted { lhs, rhs in
            let lhsActiveRDMA = lhs.info.rdma.rdmaEnabled
                && lhs.info.rdma.rdmaPortState.values.contains("active")
            let rhsActiveRDMA = rhs.info.rdma.rdmaEnabled
                && rhs.info.rdma.rdmaPortState.values.contains("active")
            if lhsActiveRDMA != rhsActiveRDMA {
                return lhsActiveRDMA
            }
            return lhs.agentURL < rhs.agentURL
        }
    }

    private func discoveryControlScore(
        _ discovery: DiscoveredNodeEndpoint,
        savedEndpoints: Set<String>
    ) -> Int {
        guard let url = URL(string: discovery.agentURL) else { return 0 }
        var score = 0
        if url.port == 9_100 { score += 100 }
        if url.host != discovery.info.rdma.thunderboltIP { score += 10 }
        if savedEndpoints.contains(discovery.agentURL) { score += 1 }
        return score
    }

    private func mergeDiscoveredNode(
        _ discovery: DiscoveredNodeEndpoint
    ) -> DiscoveryMergeResult {
        let info = discovery.info
        guard let discoveredURL = Self.normalizedAgentBaseURL(discovery.agentURL) else {
            return .unchanged
        }
        var existingIndex = nodes.firstIndex { node in
            if node.agentURL == discovery.agentURL || node.id == info.nodeID {
                return true
            }
            if let machineID = info.machineID,
               !machineID.isEmpty,
               node.machineID == machineID {
                return true
            }
            if let rdmaIP = info.rdma.thunderboltIP,
               !rdmaIP.isEmpty,
               node.rdma.thunderboltIP == rdmaIP {
                return true
            }
            return node.hostname.caseInsensitiveCompare(info.hostname) == .orderedSame
                && node.user.caseInsensitiveCompare(info.user) == .orderedSame
        }
        if existingIndex == nil {
            existingIndex = nodes.indices.first {
                unboundPlaceholderNodeIDs.contains(nodes[$0].id)
                    && !Self.isLoopbackAgentURL(discoveredURL)
            }
        }

        var discoveredNode = TokenityNode(
            id: info.machineID ?? info.nodeID,
            hostname: info.hostname,
            user: info.user,
            agentURL: discovery.agentURL,
            ips: info.ips,
            architecture: info.architecture,
            pythonPath: info.pythonPath,
            mlxVersion: info.mlxVersion,
            mlxLMVersion: info.mlxLMVersion,
            tokenityVersion: info.tokenityVersion,
            machineID: info.machineID,
            tokenityCodeRevision: info.tokenityCodeRevision,
            agentContract: info.agentContract,
            rdma: info.rdma,
            roles: info.processRoles,
            memory: info.memory ?? .unknown,
            models: [],
            isOnline: true,
            clusterRuntime: info.clusterRuntime,
            clusterRuntimes: info.clusterRuntimes ?? [],
            lastAgentResponseAt: Date(),
            modelInstances: info.instances ?? []
        )

        if let existingIndex {
            let existing = nodes[existingIndex]
            let claimedPlaceholder = unboundPlaceholderNodeIDs.contains(existing.id)
            let endpointChanged = existing.agentURL != discovery.agentURL
            discoveredNode.id = existing.id
            discoveredNode.models = existing.models
            discoveredNode.runtimeMemory = existing.runtimeMemory
            nodes[existingIndex] = discoveredNode
            unboundPlaceholderNodeIDs.remove(existing.id)
            if claimedPlaceholder,
               autoSelectedPlaceholderNodeIDs.remove(existing.id) != nil {
                selectedNodeIDs.insert(existing.id)
            }
            return endpointChanged ? .rebound : .unchanged
        }

        if nodes.contains(where: { $0.id == discoveredNode.id }) {
            let port = URL(string: discovery.agentURL)?.port ?? 9_100
            discoveredNode.id += "-\(port)"
        }
        nodes.append(discoveredNode)
        return .added
    }

    private func persistNodeEndpointOverrides() {
        let overrides: [String: String] = Dictionary(
            uniqueKeysWithValues: nodes.compactMap { node in
                guard let url = Self.normalizedAgentBaseURL(node.agentURL),
                      !unboundPlaceholderNodeIDs.contains(node.id),
                      !Self.isLoopbackAgentURL(url)
                else { return nil }
                return (node.id, url.absoluteString)
            }
        )
        userDefaults.set(overrides, forKey: nodeEndpointOverridesKey)
    }

    private static func normalizedAgentBaseURL(_ rawValue: String) -> URL? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let candidate = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
        guard var components = URLComponents(string: candidate),
              let scheme = components.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              let host = components.host,
              !host.isEmpty,
              components.user == nil,
              components.password == nil
        else { return nil }
        components.scheme = scheme
        components.path = ""
        components.query = nil
        components.fragment = nil
        return components.url
    }

    private static func isLoopbackAgentURL(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else { return false }
        return host == "localhost"
            || host.hasSuffix(".localhost")
            || host.hasPrefix("127.")
            || host == "::1"
    }

    private func fetchNodeInfo(for node: TokenityNode) async -> NodeInfoResponse? {
        guard let baseURL = Self.normalizedAgentBaseURL(node.agentURL) else { return nil }
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/node/info"))
            request.timeoutInterval = 4
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try JSONDecoder().decode(NodeInfoResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func fetchAgentCoreHealth(for node: TokenityNode) async -> AgentCoreHealthResponse? {
        guard let baseURL = Self.normalizedAgentBaseURL(node.agentURL) else { return nil }
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/node/health"))
            request.timeoutInterval = 2
            let (data, response) = try await dataTransport(request)
            guard response.statusCode == 200 || response.statusCode == 503 else { return nil }
            return try JSONDecoder().decode(AgentCoreHealthResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func fetchAgentWatchdogStatus(for node: TokenityNode) async -> AgentWatchdogStatusResponse? {
        guard let baseURL = Self.normalizedAgentBaseURL(node.agentURL),
              var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
              let agentPort = components.port else { return nil }
        components.port = agentPort + 1
        components.path = "/v1/watchdog/status"
        components.query = nil
        guard let url = components.url else { return nil }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 2
            let (data, response) = try await dataTransport(request)
            guard response.statusCode == 200 || response.statusCode == 503 else { return nil }
            return try JSONDecoder().decode(AgentWatchdogStatusResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func applyAgentHealth(
        _ core: AgentCoreHealthResponse?,
        watchdog: AgentWatchdogStatusResponse?,
        previous: TokenityNode,
        to node: inout TokenityNode
    ) {
        if let watchdog {
            node.watchdogRestartCount = watchdog.restartCount
            if let timestamp = watchdog.lastRestartTime {
                node.lastAutomaticRecoveryAt = Date(timeIntervalSince1970: timestamp)
            }
            node.consecutiveAgentFailures = max(
                node.consecutiveAgentFailures,
                watchdog.consecutiveFailures
            )
            switch watchdog.phase {
            case "recovering", "maintenance":
                node.agentHealthState = .recovering
                node.agentHealthDetail = watchdog.phase == "maintenance"
                    ? "Agent update maintenance window is active."
                    : (watchdog.lastFailureReason ?? "The local watchdog is confirming Agent health.")
                return
            case "restarting":
                node.agentHealthState = .restarting
                node.agentHealthDetail = watchdog.lastFailureReason ?? "The local watchdog restarted the Node Agent."
                return
            case "circuit_open":
                node.agentHealthState = .circuitOpen
                node.agentHealthDetail = watchdog.lastFailureReason ?? "Automatic restart limit reached; this Mac needs attention."
                return
            default:
                if let lastRestart = watchdog.lastRestartTime,
                   Date().timeIntervalSince1970 - lastRestart < 30 {
                    node.agentHealthState = .restarted
                    node.agentHealthDetail = "Node Agent recovered automatically."
                    return
                }
            }
        }

        if let core {
            switch core.status {
            case "healthy":
                node.agentHealthState = .online
                node.agentHealthDetail = "Agent control service is healthy."
            case "degraded":
                node.agentHealthState = .degraded
                node.agentHealthDetail = "Agent is reachable; a non-core capability is degraded."
            default:
                node.agentHealthState = .degraded
                node.agentHealthDetail = core.lastFatalInternalError?.message
                    ?? "Agent is reachable but cannot safely provide control service."
            }
        } else if node.isOnline {
            // Rolling-upgrade compatibility for an older Agent without the
            // lightweight health contract.
            node.agentHealthState = .online
            node.agentHealthDetail = node.agentContract?.supports("agent_health") == true
                ? "Agent telemetry is reachable; health endpoint is temporarily unavailable."
                : "Online through the legacy Node Agent health contract."
        } else if node.consecutiveAgentFailures >= 2 {
            node.agentHealthState = .unreachable
            node.agentHealthDetail = "Network or host unreachable. The UI will not restart an Agent without loopback watchdog evidence."
        } else {
            node.agentHealthState = previous.agentHealthState
            node.agentHealthDetail = previous.agentHealthDetail
        }
    }

    private static let residentModelAgentCapabilities: Set<String> = [
        "managed_instances",
        "instance_runtimes",
        "instance_quorum",
        "cluster_runtime",
    ]

    private var selectedAgentsSupportResidentModels: Bool {
        !selectedNodes.isEmpty && selectedNodes.allSatisfy { node in
            guard let contract = node.agentContract else { return false }
            return Self.residentModelAgentCapabilities.isSubset(of: contract.capabilities)
        }
    }

    private func synchronizeVideoRuntime(from sampledNodes: [TokenityNode]) {
        guard let instanceID = videoRuntimeInstanceID else { return }
        guard let snapshot = sampledNodes
            .flatMap(\.modelInstances)
            .first(where: { $0.instanceID == instanceID })
        else { return }

        switch snapshot.state.lowercased() {
        case "ready", "busy":
            videoRuntimeState = .ready
        case "stopped":
            videoRuntimeInstanceID = nil
            videoRuntimeState = .stopped
            if !isVideoGenerating {
                videoProgressStage = "Waiting for a MiniMax H3 runtime"
            }
        case "failed", "orphaned":
            let message = snapshot.healthIssues?.joined(separator: " ")
                ?? "The MiniMax H3 runtime stopped unexpectedly."
            videoRuntimeState = .failed(message)
            videoGenerationError = message
        case "stopping":
            videoRuntimeState = .stopping
        default:
            videoRuntimeState = .starting
        }
    }

    private func refreshVideoRuntimeQuorumIfNeeded() async {
        guard case .ready = videoRuntimeState,
              let instanceID = videoRuntimeInstanceID,
              let baseURL = videoControlBaseURL()
        else { return }
        var request = URLRequest(
            url: baseURL.appendingPathComponent(
                "/v1/node/instances/\(instanceID)/quorum"
            )
        )
        request.timeoutInterval = 4
        do {
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            let quorum = try JSONDecoder().decode(InstanceQuorumResponse.self, from: data)
            guard quorum.instanceID == instanceID, !quorum.ready else { return }
            let detail = quorum.issues.joined(separator: " ")
            let message = detail.isEmpty
                ? "The MiniMax H3 rank quorum is not ready."
                : detail
            videoRuntimeState = .failed(message)
            videoGenerationError = message
            videoProgressStage = "Runtime rank failure"
            appendLog("MiniMax H3 rank quorum failed: \(message)")
        } catch {
            // Node reachability is already represented by videoNodes. A
            // transient quorum request must not overwrite a valid runtime.
        }
    }

    private func fetchGatewayRoutes() async -> [GatewayModelRoute]? {
        guard let baseURL = clusterControlBaseURL() else { return nil }
        var request = URLRequest(url: baseURL.appendingPathComponent("/v1/gateway/routes"))
        request.timeoutInterval = 5
        do {
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try JSONDecoder().decode(GatewayRoutesResponse.self, from: data).data
        } catch {
            return nil
        }
    }

    private func recoverManagedModelInstancesFromAgent(expectedEpoch: UInt64) async {
        guard selectedAgentsSupportResidentModels,
              let controller = coordinator,
              let routes = await fetchGatewayRoutes(),
              expectedEpoch == modelOperationEpoch
        else { return }

        let snapshots = controller.modelInstances
        let snapshotsByID = Dictionary(
            snapshots.map { ($0.instanceID, $0) },
            uniquingKeysWith: { _, latest in latest }
        )
        let routesByID = Dictionary(
            routes.map { ($0.instanceID, $0) },
            uniquingKeysWith: { _, latest in latest }
        )
        let previouslyKnownIDs = Set(managedModelInstances.keys)
        var removedInstanceIDs: Set<String> = []
        var removedModelIDs: Set<String> = []

        for instanceID in previouslyKnownIDs where snapshotsByID[instanceID] == nil {
            if let removed = managedModelInstances[instanceID] {
                removedModelIDs.insert(removed.modelID)
            }
            managedModelInstances.removeValue(forKey: instanceID)
            residentAutoPreferences.removeValue(forKey: instanceID)
            removedInstanceIDs.insert(instanceID)
        }

        for snapshot in snapshots {
            let route = routesByID[snapshot.instanceID]
            let state = snapshot.state.lowercased()
            if state == "stopped" {
                if let removed = managedModelInstances.removeValue(forKey: snapshot.instanceID) {
                    removedModelIDs.insert(removed.modelID)
                    removedInstanceIDs.insert(snapshot.instanceID)
                }
                residentAutoPreferences.removeValue(forKey: snapshot.instanceID)
                continue
            }
            let routeIsRoutable = route.map {
                ["ready", "busy"].contains($0.state.lowercased())
            } ?? false
            guard ["ready", "busy"].contains(state)
                    || managedModelInstances[snapshot.instanceID] != nil
            else { continue }

            let processRole = controller.roles.first {
                $0.instanceID == snapshot.instanceID
                    && Self.inferenceRoleNames.contains($0.role)
            }?.role
            var managed = managedModelInstances[snapshot.instanceID] ?? ManagedModelInstance(
                modelID: snapshot.requestedModelID,
                serviceModelName: route?.model ?? snapshot.requestedModelID,
                instanceID: snapshot.instanceID,
                // Gateway routes intentionally advertise a coordinator-local
                // 127.0.0.1 URL. Control keeps chat on the stable :9100 gateway.
                apiBaseURL: nil,
                backendRole: processRole ?? "distributed-openai",
                lifecycleState: state,
                quorumReady: false,
                routeAvailable: routeIsRoutable
            )
            managed.modelID = snapshot.requestedModelID
            managed.serviceModelName = route?.model ?? snapshot.requestedModelID
            managed.backendRole = processRole ?? managed.backendRole
            managed.lifecycleState = state
            managed.routeAvailable = routeIsRoutable
            managed.modelRevision = route?.modelRevision ?? snapshot.modelRevision
            managed.selectedNodes = snapshot.selectedNodes ?? []
            managed.executionMode = snapshot.executionMode ?? route?.executionMode
            managed.connectionMode = snapshot.connectionMode
            managed.reservedMemoryBytes = snapshot.memoryReservationBytes
            managed.actualMemoryBytes = snapshot.actualMemoryBytes
            managed.activeRequestCount = route?.activeRequestCount
                ?? snapshot.activeRequestCount
                ?? 0
            managed.queueDepth = route?.queueDepth
                ?? snapshot.queuedRequestCount
                ?? 0
            managed.routeCapabilities = route?.capabilities
            managed.warmTTFTP50Milliseconds = route?.warmTTFTP50Milliseconds
            managed.warmTTFTP95Milliseconds = route?.warmTTFTP95Milliseconds
            if snapshot.healthReady == false {
                managed.quorumReady = false
                managed.healthIssue = snapshot.healthIssues?.joined(separator: " ")
            }
            if !routeIsRoutable {
                managed.quorumReady = false
                managed.healthIssue = "The stable gateway does not advertise this instance as routable."
            }
            managedModelInstances[snapshot.instanceID] = managed
        }

        for modelID in removedModelIDs
        where !managedModelInstances.values.contains(where: { $0.modelID == modelID }) {
            modelLoadStates[modelID] = .notLoaded
        }
        migrateLegacyResidentAutoPreferences()
        persistResidentAutoPreferences()
        if !removedInstanceIDs.isEmpty {
            appendLog(
                "Removed \(removedInstanceIDs.count) stale resident model "
                    + "instance record\(removedInstanceIDs.count == 1 ? "" : "s")."
            )
        }
        let newlyDiscoveredIDs = Set(managedModelInstances.keys).subtracting(previouslyKnownIDs)
        recomputeManagedModelLoadStates()
        if let activeModelInstanceID {
            let activeSnapshotState = snapshotsByID[activeModelInstanceID]?.state.lowercased()
            let preservesLoadingIdentity = activeModelLoadID != nil
                && (activeSnapshotState.map {
                    !["stopped", "failed", "orphaned"].contains($0)
                } ?? true)
            if !preservesLoadingIdentity,
               managedModelInstances[activeModelInstanceID]?.isRoutable != true {
                restoreActiveManagedInstance()
            }
        } else {
            restoreActiveManagedInstance()
        }
        if !newlyDiscoveredIDs.isEmpty {
            await renewModelLeases(instanceIDs: newlyDiscoveredIDs)
        }
    }

    private func migrateLegacyResidentAutoPreferences() {
        let modelIDs = Set(managedModelInstances.values.map(\.modelID))
        for modelID in modelIDs {
            guard let legacyPreference = residentAutoPreferences.removeValue(forKey: modelID) else {
                continue
            }
            for managed in managedModelInstances.values where managed.modelID == modelID {
                if residentAutoPreferences[managed.instanceID] == nil {
                    residentAutoPreferences[managed.instanceID] = legacyPreference
                }
            }
        }
    }

    private func persistResidentAutoPreferences() {
        if let encoded = try? JSONEncoder().encode(residentAutoPreferences) {
            userDefaults.set(encoded, forKey: residentAutoPreferencesKey)
        }
    }

    private func refreshManagedModelInstanceHealth(expectedEpoch: UInt64) async {
        let instanceIDs = managedModelInstances.values
            .filter {
                $0.routeAvailable
                    && ["ready", "busy"].contains($0.lifecycleState.lowercased())
            }
            .map(\.instanceID)
            .sorted()

        for instanceID in instanceIDs {
            do {
                let quorum = try await fetchInstanceQuorum(instanceID: instanceID)
                guard expectedEpoch == modelOperationEpoch else { return }
                guard var managed = managedModelInstances[instanceID] else { continue }
                managed.quorumReady = quorum.ready
                let issue = quorum.issues.joined(separator: " ")
                managed.healthIssue = quorum.ready
                    ? nil
                    : (issue.isEmpty ? "The instance rank quorum is not ready." : issue)
                managedModelInstances[instanceID] = managed
                if quorum.ready, instanceID == activeModelInstanceID {
                    applyRuntimeMemory(from: quorum)
                }
            } catch {
                guard expectedEpoch == modelOperationEpoch else { return }
                guard var managed = managedModelInstances[instanceID] else { continue }
                managed.quorumReady = false
                managed.healthIssue = userFacingMessage(for: error)
                managedModelInstances[instanceID] = managed
            }
        }

        let previousActiveInstanceID = activeModelInstanceID
        recomputeManagedModelLoadStates()
        // The coordinator start response becomes authoritative before the
        // next background node-info snapshot necessarily contains that new
        // instance. Do not redirect its readiness/inference probe to an
        // already-ready sibling during this short propagation window.
        if activeModelLoadID == nil {
            if let previousActiveInstanceID,
               managedModelInstances[previousActiveInstanceID]?.isRoutable != true {
                cancelActiveChat(
                    message: "Generation stopped because the active model instance is no longer ready.",
                    logReason: "managed instance health"
                )
                restoreActiveManagedInstance()
            } else if activeModelInstanceID == nil {
                restoreActiveManagedInstance()
            }
        }

        if activeModelLoadID != nil {
            return
        }

        if let activeModelInstanceID,
           managedModelInstances[activeModelInstanceID]?.isRoutable == true {
            phase = .running
            serverHealth = .ready
        } else if !managedModelInstances.isEmpty {
            let issue = managedModelInstances.values
                .compactMap(\.healthIssue)
                .first
                ?? "No managed model instance has a ready rank quorum."
            serverHealth = .error(issue)
        }
    }

    private func recomputeManagedModelLoadStates() {
        var states = modelLoadStates
        let grouped = Dictionary(grouping: managedModelInstances.values, by: \.modelID)
        for (modelID, instances) in grouped {
            states[modelID] = instances.contains(where: \.isRoutable) ? .loaded : .failed
        }
        modelLoadStates = states
    }

    private func keepClusterPrimaryInSelection() {
        if !selectedNodeIDs.contains(coordinatorID) {
            coordinatorID = selectedNodeIDs.sorted().first ?? ""
        }
    }

    private func resetModelLoadState(message: String) {
        var states = modelLoadStates
        for key in states.keys {
            states[key] = .notLoaded
        }
        modelLoadStates = states
        modelPath = ""
        loadedBackendRole = nil
        loadedServiceModelName = nil
        activeLoadedModelID = nil
        managedModelInstances.removeAll()
        activeModelInstanceID = nil
        activeModelServiceBaseURL = nil
        for index in nodes.indices {
            nodes[index].runtimeMemory = nil
        }
        modelLoadProgress = nil
        nativeMTPRuntime = nil
        legacyNativeMTPFallback = nil
        modelLoadMessage = message
        serverHealth = phase == .running
            ? .starting("Waiting for a model to load")
            : .stopped
    }

    private func markInferenceRolesStoppedLocally() {
        for index in nodes.indices where selectedNodeIDs.contains(nodes[index].id) {
            nodes[index].roles = nodes[index].roles.map { role in
                guard Self.inferenceRoleNames.contains(role.role) else { return role }
                var stopped = role
                stopped.state = "stopped"
                stopped.pid = nil
                return stopped
            }
            nodes[index].clusterRuntime = nil
            nodes[index].clusterRuntimes = []
        }
    }

    private func reconcileExternallyStoppedService() {
        // Stopping the previous runtime is an expected part of loading or
        // unloading a model on legacy Agents. A polling sample can observe the
        // old role as stopped before the new start request completes; treating
        // that sample as an external exit races the transition and disables
        // Chat even after the new model becomes ready.
        guard activeModelLoadID == nil,
              !isModelTransitioning,
              loadedModelName != nil
        else { return }
        let inferenceRoles = selectedNodes.flatMap { inferenceRolesForActiveModel(on: $0) }
        guard !inferenceRoles.isEmpty else { return }

        if let failed = inferenceRoles.first(where: { $0.state.lowercased() == "failed" }) {
            let message = failed.message ?? "The \(failed.role) process failed."
            if failActiveManagedInstanceAndRestoreSibling(message: message) {
                return
            }
            serverHealth = .error(message)
            return
        }

        let hasActiveRole = inferenceRoles.contains { role in
            let state = role.state.lowercased()
            return role.pid != nil && state != "stopped" && state != "failed"
        }
        if selectedNodes.allSatisfy(\.isOnline), !hasActiveRole {
            let message = "The active model service exited outside Tokenity."
            if failActiveManagedInstanceAndRestoreSibling(message: message) {
                return
            }
            cancelActiveChat(
                message: "Generation stopped because the model service exited outside Tokenity.",
                logReason: "external service stop"
            )
            resetModelLoadState(message: "Model service stopped outside Tokenity.")
            phase = .stopped
            appendLog("Detected that the model service was stopped outside Tokenity.")
        }
    }

    private func failActiveManagedInstanceAndRestoreSibling(message: String) -> Bool {
        guard let failedModelID = activeLoadedModelID,
              let failedInstanceID = activeModelInstanceID,
              managedModelInstances[failedInstanceID]?.modelID == failedModelID,
              managedModelInstances.values.contains(where: {
                  $0.instanceID != failedInstanceID && $0.isRoutable
              })
        else { return false }

        cancelActiveChat(
            message: "Generation stopped because the active model instance exited.",
            logReason: "managed instance exit"
        )
        var states = modelLoadStates
        states[failedModelID] = .failed
        modelLoadStates = states
        managedModelInstances.removeValue(forKey: failedInstanceID)
        recomputeManagedModelLoadStates()
        restoreActiveManagedInstance()

        guard let replacement = loadedModelName else { return false }
        phase = .running
        serverHealth = .ready
        modelLoadMessage = "\(message) Switched to \(replacement), which remains loaded."
        appendLog("Managed model instance \(failedModelID) exited; switched to \(replacement).")
        return true
    }

    static let inferenceRoleNames: Set<String> = [
        "distributed-openai",
        "distributed-openai-rank",
        "single-node-openai",
    ]

    private static func isActiveInferenceProcess(_ role: ProcessRole) -> Bool {
        guard inferenceRoleNames.contains(role.role), role.pid != nil else { return false }
        let state = role.state.lowercased()
        return state != "stopped" && state != "failed"
    }

    private func aggregateNativeMTPCapability(
        _ capabilities: [NativeMTPCapability],
        expectedCount: Int
    ) -> NativeMTPCapability? {
        guard !capabilities.isEmpty else { return nil }
        guard capabilities.count == expectedCount, let first = capabilities.first else {
            return NativeMTPCapability(
                status: "unknown",
                modelType: nil,
                declaredLayers: 0,
                weightsPresent: false,
                reason: nil,
                message: "One or more selected Macs did not report Native MTP capability.",
                tensorFormat: nil,
                tensorKeyDigest: nil,
                missingGroups: nil
            )
        }
        let agrees = capabilities.dropFirst().allSatisfy {
            $0.status == first.status
                && $0.modelType == first.modelType
                && $0.declaredLayers == first.declaredLayers
                && $0.tensorKeyDigest == first.tensorKeyDigest
        }
        guard agrees else {
            return NativeMTPCapability(
                status: "node_mismatch",
                modelType: first.modelType,
                declaredLayers: first.declaredLayers,
                weightsPresent: false,
                reason: "node_mismatch",
                message: "Selected Macs report different Native MTP checkpoint metadata.",
                tensorFormat: first.tensorFormat,
                tensorKeyDigest: nil,
                missingGroups: nil
            )
        }
        return first
    }

    private func appendLog(_ line: String) {
        let stamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        logs.append("[\(stamp)] \(line)")
    }

    private func readinessIssues(for nodes: [TokenityNode]) -> [String] {
        var issues = nodes.compactMap { node in
            node.isOnline ? nil : "\(node.displayName) Node Agent is offline."
        }
        guard connectionMode != .ring else { return issues }
        issues.append(contentsOf: nodes.filter(\.isOnline).flatMap { node -> [String] in
            var issues: [String] = []
            if !node.rdma.rdmaEnabled {
                let detail = node.rdma.rdmaErrors.first.map { " \($0)" } ?? ""
                issues.append("\(node.displayName) needs an active Thunderbolt RDMA link.\(detail)")
            } else if node.rdma.rdmaDevices.isEmpty {
                issues.append("\(node.displayName) needs an active RDMA device.")
            }
            if node.rdma.thunderboltIP == nil {
                issues.append("\(node.displayName) needs a detected Thunderbolt network address.")
            }
            return issues
        })
        return issues
    }

    private func networkPlan(for nodes: [TokenityNode]) -> [NetworkPlanRow] {
        nodes.map { node in
            let ready: Bool
            let detail: String
            if !node.isOnline {
                ready = false
                detail = "Node Agent is offline or unreachable."
            } else {
                switch connectionMode {
                case .ring:
                    ready = true
                    detail = "Uses the standard network path."
                case .jaccl, .jacclRing:
                    ready = node.rdma.rdmaEnabled && node.rdma.thunderboltIP != nil
                    if ready {
                        detail = "Direct Thunderbolt link detected."
                    } else if let error = node.rdma.rdmaErrors.first {
                        detail = error
                    } else {
                        detail = "Thunderbolt link information is incomplete."
                    }
                }
            }
            return NetworkPlanRow(
                nodeID: node.id,
                nodeName: node.displayName,
                role: "Cluster member",
                link: connectionMode.rawValue,
                readiness: ready ? "Ready" : "Needs attention",
                detail: detail
            )
        }
    }

    private func startBackendModel(
        _ row: ModelLibraryRow,
        role: String,
        configuration: ModelRuntimeConfiguration,
        operationID: UUID,
        instanceID: String
    ) async throws -> AgentStartModelResponse? {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        let nativeMTP = effectiveNativeMTPConfiguration
        func requestBody(
            nativeMTP: NativeMTPConfiguration?,
            includesInstanceMetadata: Bool
        ) -> AgentStartModelRequest {
            // The Agent receiving this request always starts rank 0 locally.
            // Keep the selected coordinator first even after users temporarily
            // remove and re-add the original primary Mac.
            let rankNodes = backendMode == .singleNode
                ? Array(plannedNodes.prefix(1))
                : plannedNodes
            return AgentStartModelRequest(
                model: row.representativePath,
                nodes: rankNodes.map(agentNodePayload(for:)),
                connectionMode: backendMode == .singleNode ? ConnectionMode.ring.cliValue : connectionMode.cliValue,
                startingPort: mlxStartingPort,
                host: "0.0.0.0",
                port: modelHTTPPort,
                dryRun: false,
                maxTokens: configuration.maximumOutputTokens,
                promptCacheSize: configuration.promptCacheSize,
                prefillStepSize: configuration.prefillStepSize,
                decodeConcurrency: configuration.decodeConcurrency,
                promptConcurrency: configuration.promptConcurrency,
                trustRemoteCode: configuration.trustRemoteCode,
                leaseSeconds: modelLeaseSeconds,
                nativeMTP: nativeMTP,
                instanceID: includesInstanceMetadata ? instanceID : nil,
                operationID: includesInstanceMetadata ? operationID.uuidString : nil
            )
        }

        func send(_ body: AgentStartModelRequest) async throws -> AgentStartModelResponse? {
            var request = try jsonRequest(url: baseURL.appendingPathComponent("/v1/node/start-distributed-openai"), body: body)
            request.timeoutInterval = 40
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try? JSONDecoder().decode(AgentStartModelResponse.self, from: data)
        }

        var includesInstanceMetadata = true
        var requestedNativeMTP = nativeMTP.mode == .off ? nil : nativeMTP

        while true {
            do {
                // Optional feature fields are removed one capability at a time
                // when an older Agent rejects them. This preserves the stable
                // load/chat path during a rolling Control/Agent upgrade.
                return try await send(
                    requestBody(
                        nativeMTP: requestedNativeMTP,
                        includesInstanceMetadata: includesInstanceMetadata
                    )
                )
            } catch TokenityTransportError.httpStatus(let status, let detail)
                where status == 422 && isUnsupportedOptionalStartField(detail) {
                let rejectsInstanceMetadata = isUnsupportedInstanceMetadataField(detail)
                let rejectsNativeMTP = isUnsupportedNativeMTPField(detail)

                if includesInstanceMetadata,
                   rejectsInstanceMetadata || (!rejectsNativeMTP && !rejectsInstanceMetadata) {
                    guard managedModelInstances.isEmpty else {
                        throw TokenityTransportError.multiInstanceAgentUpgradeRequired
                    }
                    includesInstanceMetadata = false
                    appendLog("The running Node Agent predates managed model instances. Retrying with the legacy model start contract.")
                    continue
                }

                if requestedNativeMTP != nil,
                   rejectsNativeMTP || !rejectsInstanceMetadata {
                    switch nativeMTP.mode {
                    case .auto:
                        appendLog("The running Node Agent predates Native MTP. Auto is falling back to standard decoding for this load.")
                        legacyNativeMTPFallback = NativeMTPReadiness(
                            requestedMode: NativeMTPMode.auto.rawValue,
                            enabled: false,
                            status: "unsupported",
                            effectiveMode: "standard",
                            fallbackReason: "unsupported_backend",
                            message: "The running Node Agent predates Native MTP; standard decoding is active.",
                            proposedTokens: 0,
                            acceptedTokens: 0,
                            acceptanceRate: nil
                        )
                        nativeMTPRuntime = legacyNativeMTPFallback
                        requestedNativeMTP = nil
                        continue
                    case .required:
                        throw TokenityTransportError.nativeMTPAgentUpgradeRequired
                    case .off:
                        break
                    }
                }

                throw TokenityTransportError.httpStatus(status, detail)
            }
        }
    }

    private func isUnsupportedOptionalStartField(_ detail: String?) -> Bool {
        let normalized = detail?.lowercased() ?? ""
        return normalized.contains("extra inputs") && normalized.contains("not permitted")
    }

    private func isUnsupportedInstanceMetadataField(_ detail: String?) -> Bool {
        let normalized = detail?.lowercased() ?? ""
        return normalized.contains("instance_id")
            || normalized.contains("operation_id")
            || normalized.contains("memory_reservation_bytes")
    }

    private func isUnsupportedNativeMTPField(_ detail: String?) -> Bool {
        let normalized = detail?.lowercased() ?? ""
        return normalized.contains("native_mtp") && isUnsupportedOptionalStartField(detail)
    }

    private func stopBackendRole(_ role: String, on node: TokenityNode) async throws {
        guard let baseURL = URL(string: node.agentURL) else { throw TokenityTransportError.missingClusterControl }
        var request = try jsonRequest(
            url: baseURL.appendingPathComponent("/v1/node/stop-role"),
            body: AgentStopRoleRequest(role: role, timeout: 10, instanceID: activeModelInstanceID)
        )
        request.timeoutInterval = 15
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
    }

    private func cleanupActiveInstanceOrLegacy() async throws {
        try await cleanupInstanceOrLegacy(
            instanceID: activeModelInstanceID,
            allowsGlobalFallback: managedModelInstances.isEmpty
        )
    }

    private func cleanupInstanceOrLegacy(
        instanceID: String?,
        allowsGlobalFallback: Bool
    ) async throws {
        guard let instanceID,
              let baseURL = clusterControlBaseURL() else {
            if allowsGlobalFallback {
                try await cleanupAllModelRoles()
            }
            return
        }
        var request = try jsonRequest(
            url: baseURL.appendingPathComponent("/v1/node/instances/\(instanceID)/stop"),
            body: AgentStopAllRequest(timeout: 10)
        )
        request.timeoutInterval = 15
        do {
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
        } catch TokenityTransportError.httpStatus(let status, _) where status == 404 {
            if allowsGlobalFallback {
                try await cleanupAllModelRoles()
            }
            // A missing instance is already stopped from the caller's point of
            // view. Treat the precise stop as idempotent and never widen it to
            // a global cleanup that could terminate sibling instances.
        }
    }

    private func stopVideoInstance(_ instanceID: String) async throws {
        guard let baseURL = videoControlBaseURL() else {
            throw TokenityTransportError.missingClusterControl
        }
        var request = try jsonRequest(
            url: baseURL.appendingPathComponent("/v1/node/instances/\(instanceID)/stop"),
            body: AgentStopAllRequest(timeout: 10)
        )
        request.timeoutInterval = 15
        do {
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
        } catch TokenityTransportError.httpStatus(let status, _) where status == 404 {
            // The precise video instance is already stopped. Never widen this
            // to stop-all because the stable Chat Agent may share the machine.
        }
    }

    private func fetchVideoRuntimeFailure(instanceID: String) async -> String? {
        // Rank 1 owns the TP data-plane handshake, so prefer its diagnostic
        // when both ranks have recorded a failure for the same instance.
        for node in videoNodes.reversed() {
            guard let baseURL = URL(string: node.agentURL) else { continue }
            var request = URLRequest(
                url: baseURL.appendingPathComponent("/v1/node/instances/\(instanceID)")
            )
            request.timeoutInterval = 4
            do {
                let (data, response) = try await dataTransport(request)
                try validate(response, data: data)
                let detail = try JSONDecoder().decode(AgentInstanceDetailResponse.self, from: data)
                if let failure = detail.instance.lastError?.displayMessage {
                    return "\(node.displayName): \(failure)"
                }
                if let message = detail.process?.message, !message.isEmpty {
                    return "\(node.displayName): \(message)"
                }
            } catch {
                continue
            }
        }
        return nil
    }

    private func restoreActiveManagedInstance() {
        let next = managedModelInstances.values
            .filter(\.isRoutable)
            .sorted {
                if $0.modelID == $1.modelID {
                    return $0.instanceID.localizedCaseInsensitiveCompare($1.instanceID) == .orderedAscending
                }
                return $0.modelID.localizedCaseInsensitiveCompare($1.modelID) == .orderedAscending
            }
            .first
        guard let next else {
            activeLoadedModelID = nil
            loadedBackendRole = nil
            loadedServiceModelName = nil
            activeModelInstanceID = nil
            activeModelServiceBaseURL = nil
            modelPath = ""
            return
        }
        activeLoadedModelID = next.modelID
        loadedBackendRole = next.backendRole
        loadedServiceModelName = next.serviceModelName
        activeModelInstanceID = next.instanceID
        activeModelServiceBaseURL = next.apiBaseURL
        modelPath = modelLibraryRows.first(where: { $0.id == next.modelID })?.representativePath ?? ""
    }

    private func nodeDisplayName(for identity: String) -> String {
        nodes.first {
            $0.id == identity || $0.hostname == identity || $0.ips.contains(identity)
        }?.displayName ?? identity
    }

    private func cleanupAllModelRoles() async throws {
        let roles = ["distributed-openai", "distributed-openai-rank", "single-node-openai"]
        var firstError: Error?
        for node in selectedNodes {
            guard let baseURL = URL(string: node.agentURL) else {
                firstError = firstError ?? TokenityTransportError.missingClusterControl
                continue
            }
            do {
                var request = try jsonRequest(
                    url: baseURL.appendingPathComponent("/v1/node/stop-all"),
                    body: AgentStopAllRequest(timeout: 10)
                )
                request.timeoutInterval = 15
                let (data, response) = try await dataTransport(request)
                try validate(response, data: data)
            } catch {
                var fallbackSucceeded = false
                var fallbackError: Error = error
                for role in roles {
                    do {
                        try await stopBackendRole(role, on: node)
                        fallbackSucceeded = true
                    } catch {
                        fallbackError = error
                    }
                }
                if !fallbackSucceeded {
                    firstError = firstError ?? fallbackError
                }
            }
        }
        if let firstError {
            throw firstError
        }
    }

    private func renewModelLeasesIfNeeded() async {
        guard isModelTransitioning || loadedModelName != nil else { return }
        let instanceIDs = Set(
            managedModelInstances.values.map(\.instanceID)
                + [activeModelInstanceID].compactMap { $0 }
        )
        // A current managed Agent requires instance-scoped heartbeats. During
        // the short interval before a start request returns its identity there
        // is nothing valid to renew, so never fall back to the rejected legacy
        // global heartbeat.
        if instanceIDs.isEmpty, selectedAgentsSupportResidentModels {
            return
        }
        await renewModelLeases(instanceIDs: instanceIDs)
    }

    private func renewVideoLeaseIfNeeded() async {
        guard let instanceID = videoRuntimeInstanceID else { return }
        switch videoRuntimeState {
        case .starting, .ready, .stopping:
            break
        case .stopped, .failed:
            return
        }
        guard let baseURL = videoControlBaseURL(),
              let request = try? jsonRequest(
                url: baseURL.appendingPathComponent("/v1/node/heartbeat"),
                body: AgentHeartbeatRequest(
                    ttlSeconds: modelLeaseSeconds,
                    instanceID: instanceID
                )
              )
        else { return }
        var heartbeat = request
        heartbeat.timeoutInterval = 3
        let failureKey = "h3-video:\(instanceID)"
        do {
            let (data, response) = try await dataTransport(heartbeat)
            try validate(response, data: data)
            leaseRenewalFailureKeys.remove(failureKey)
        } catch {
            if leaseRenewalFailureKeys.insert(failureKey).inserted {
                appendLog("MiniMax H3 lease renewal failed: \(userFacingMessage(for: error))")
            }
        }
    }

    private func renewModelLeases(
        instanceIDs: Set<String>,
        ttlSeconds: Double? = nil
    ) async {
        let requestedTTL = ttlSeconds ?? modelLeaseSeconds
        let heartbeatInstanceIDs: [String?] = instanceIDs.isEmpty
            ? [nil]
            : instanceIDs.sorted().map(Optional.some)
        for node in selectedNodes {
            for instanceID in heartbeatInstanceIDs {
                guard let baseURL = URL(string: node.agentURL),
                      let request = try? jsonRequest(
                        url: baseURL.appendingPathComponent("/v1/node/heartbeat"),
                        body: AgentHeartbeatRequest(
                            ttlSeconds: requestedTTL,
                            instanceID: instanceID
                        )
                      ) else { continue }
                var heartbeat = request
                heartbeat.timeoutInterval = 3
                let failureKey = "\(node.id):\(instanceID ?? "legacy")"
                do {
                    let (data, response) = try await dataTransport(heartbeat)
                    try validate(response, data: data)
                    leaseRenewalFailureKeys.remove(failureKey)
                } catch {
                    if leaseRenewalFailureKeys.insert(failureKey).inserted {
                        appendLog(
                            "Model lease renewal failed for \(instanceID ?? "legacy service") "
                                + "on \(node.displayName): \(userFacingMessage(for: error))"
                        )
                    }
                }
            }
        }
    }

    private func ensureActiveModelLoad(_ operationID: UUID) throws {
        guard !Task.isCancelled, activeModelLoadID == operationID else {
            throw CancellationError()
        }
    }

    private func waitForModelService(modelName: String, role: String, operationID: UUID) async throws -> String {
        guard let url = modelServiceURL(path: "/v1/models") else { throw TokenityTransportError.missingModelService }
        let deadline = Date().addingTimeInterval(600)
        let emptyModelListDeadline = Date().addingTimeInterval(20)
        var lastError: Error?

        repeat {
            try ensureActiveModelLoad(operationID)
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 8
                let (data, response) = try await dataTransport(request)
                try validate(response, data: data)
                let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
                if let acceptedModel = decoded.data.first(where: { model in
                    model.id == modelName || model.id == modelPath || URL(fileURLWithPath: model.id).lastPathComponent == modelName
                }) {
                    if let readiness = try? await fetchModelReadiness() {
                        updateModelLoadProgress(readiness, modelName: modelName)
                        if readiness.phase == "failed" {
                            throw TokenityTransportError.backendExited(readiness.message ?? "The model backend reported a failed readiness state.")
                        }
                        if readiness.phase != "ready" {
                            lastError = TokenityTransportError.modelServiceNotReady
                            continue
                        }
                        if let quorum = try await fetchInstanceQuorum() {
                            applyRuntimeMemory(from: quorum)
                            guard quorum.ready else {
                                let detail = quorum.issues.joined(separator: " ")
                                lastError = TokenityTransportError.backendExited(
                                    detail.isEmpty ? "The planned rank quorum is not ready." : detail
                                )
                                continue
                            }
                        }
                    }
                    return acceptedModel.id
                }
                if Date() >= emptyModelListDeadline {
                    throw TokenityTransportError.modelServiceReturnedNoModels(modelName)
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if case TokenityTransportError.modelServiceReturnedNoModels = error {
                    throw error
                }
                lastError = error
            }
            try ensureActiveModelLoad(operationID)
            if let readiness = try? await fetchModelReadiness() {
                updateModelLoadProgress(readiness, modelName: modelName)
                if readiness.phase == "failed" {
                    throw TokenityTransportError.backendExited(readiness.message ?? "The model backend reported a failed readiness state.")
                }
            }
            if let status = try? await fetchBackendStatus(role: role), status.state == "stopped" || status.state == "failed" {
                throw TokenityTransportError.backendExited(backendExitMessage(for: status))
            }
            try await Task.sleep(for: .seconds(1))
        } while Date() < deadline

        throw lastError ?? TokenityTransportError.modelServiceNotReady
    }

    private func probeModelService(modelName: String) async throws {
        guard let url = modelServiceURL(path: "/v1/chat/completions") else { throw TokenityTransportError.missingModelService }
        let usesQwen35Sampling = modelLibraryRows
            .first(where: { $0.id == modelName })?
            .usesQwen35Sampling == true
        let probeBody: OpenAIChatRequest
        if usesQwen35Sampling {
            probeBody = OpenAIChatRequest(
                model: modelName,
                messages: [OpenAIChatRequest.Message(role: "user", content: "Reply with OK.")],
                stream: true,
                maxTokens: 16,
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                minP: 0,
                presencePenalty: 1.5,
                repetitionPenalty: 1,
                chatTemplateKwargs: ["enable_thinking": false]
            )
        } else {
            probeBody = OpenAIChatRequest(
                model: modelName,
                messages: [OpenAIChatRequest.Message(role: "user", content: "Reply with OK.")],
                stream: true,
                maxTokens: 16
            )
        }
        var request = try jsonRequest(
            url: url,
            body: probeBody
        )
        request.addValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 120
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        guard let eventStream = String(data: data, encoding: .utf8) else {
            throw TokenityTransportError.invalidResponse
        }
        var receivedToken = false
        var receivedCompletionUsage = false
        var receivedDone = false
        for line in eventStream.components(separatedBy: .newlines) {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" {
                receivedDone = true
                continue
            }
            guard let chunkData = payload.data(using: .utf8),
                  let chunk = try? JSONDecoder().decode(OpenAIChatChunk.self, from: chunkData)
            else { continue }
            let delta = chunk.choices.first?.delta
            let content = delta?.content?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let reasoning = delta?.reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? delta?.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? ""
            receivedToken = receivedToken || !content.isEmpty || !reasoning.isEmpty
            if let completionTokens = chunk.usage?.completionTokens, completionTokens > 0 {
                receivedCompletionUsage = true
            }
        }
        if (!receivedToken && !receivedCompletionUsage) || !receivedDone {
            throw TokenityTransportError.noChatContent
        }
    }

    private func fetchBackendStatus(role: String) async throws -> ProcessRole? {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        var request = URLRequest(url: baseURL.appendingPathComponent("/v1/node/status"))
        request.timeoutInterval = 5
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        let roles = try JSONDecoder().decode(NodeStatusResponse.self, from: data).roles
        let candidates = roles.filter { $0.role == role }
        guard let activeModelInstanceID else {
            return candidates.first
        }
        if let exact = candidates.first(where: { $0.instanceID == activeModelInstanceID }) {
            return exact
        }
        guard coordinator?.agentContract?.supports("managed_instances") != true else {
            return nil
        }
        return candidates.first(where: { $0.instanceID == nil })
    }

    private func fetchModelReadiness() async throws -> ModelReadinessResponse? {
        guard let url = modelServiceURL(path: "/v1/readiness") else { throw TokenityTransportError.missingModelService }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        let (data, response) = try await dataTransport(request)
        // A failed readiness document is the authoritative model lifecycle
        // result even though the endpoint correctly returns HTTP 503.
        if let readiness = try? JSONDecoder().decode(ModelReadinessResponse.self, from: data),
           readiness.phase == "failed" {
            return readiness
        }
        try validate(response, data: data)
        return try JSONDecoder().decode(ModelReadinessResponse.self, from: data)
    }

    private func fetchInstanceQuorum() async throws -> InstanceQuorumResponse? {
        guard let instanceID = activeModelInstanceID else { return nil }
        return try await fetchInstanceQuorum(instanceID: instanceID)
    }

    private func fetchInstanceQuorum(instanceID: String) async throws -> InstanceQuorumResponse {
        guard let baseURL = clusterControlBaseURL() else {
            throw TokenityTransportError.missingClusterControl
        }
        var request = URLRequest(
            url: baseURL.appendingPathComponent("/v1/node/instances/\(instanceID)/quorum")
        )
        request.timeoutInterval = 5
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        let quorum = try JSONDecoder().decode(InstanceQuorumResponse.self, from: data)
        guard quorum.instanceID == instanceID else {
            throw TokenityTransportError.invalidResponse
        }
        return quorum
    }

    private func applyRuntimeMemory(from quorum: InstanceQuorumResponse) {
        let rankNodes = plannedNodes
        for evidence in quorum.ranks {
            guard let runtime = evidence.runtime,
                  let rank = runtime.rank,
                  rankNodes.indices.contains(rank),
                  let index = nodes.firstIndex(where: { $0.id == rankNodes[rank].id })
            else { continue }
            nodes[index].runtimeMemory = runtime.memory
        }
    }

    private func updateModelLoadProgress(_ readiness: ModelReadinessResponse, modelName: String) {
        nativeMTPRuntime = readiness.nativeMTP ?? legacyNativeMTPFallback
        let phaseProgress: Double?
        switch readiness.phase {
        case "launching": phaseProgress = 0.01
        case "distributed_init": phaseProgress = 0.05
        case "loading_model": phaseProgress = readiness.progress ?? 0.1
        case "compiling": phaseProgress = readiness.progress ?? 0.96
        // Readiness means the weights are resident, but Tokenity still runs a
        // real inference probe before exposing the model as loaded.
        case "ready": phaseProgress = 0.98
        default: phaseProgress = readiness.progress
        }
        if let phaseProgress {
            let bounded = min(max(phaseProgress, 0), 1)
            modelLoadProgress = max(modelLoadProgress ?? 0, bounded)
        }
        if readiness.phase == "ready" {
            modelLoadMessage = "Verifying inference for \(modelName)..."
        } else if let progress = modelLoadProgress {
            let percent = Int((progress * 100).rounded())
            modelLoadMessage = "Loading \(modelName)... \(percent)%"
        } else if let message = readiness.message, !message.isEmpty {
            modelLoadMessage = message
        }
    }

    private func backendExitMessage(for status: ProcessRole) -> String {
        if let message = status.message, !message.isEmpty {
            return message
        }
        if let logTail = status.logTail, !logTail.isEmpty {
            return "The model backend stopped before it became ready.\n\(logTail)"
        }
        if let returnCode = status.returnCode {
            return "The model backend stopped before it became ready. Exit code: \(returnCode)."
        }
        if let logPath = status.logPath {
            return "The model backend stopped before it became ready. Check the log at \(logPath)."
        }
        return "The model backend stopped before it became ready."
    }

    private func agentNodePayload(for node: TokenityNode) -> AgentClusterNodeRequest {
        AgentClusterNodeRequest(
            id: node.id,
            agentURL: node.agentURL,
            lanIP: node.primaryIP == "unknown" ? nil : node.primaryIP,
            rdmaIP: node.rdma.thunderboltIP,
            rdmaDevices: node.rdma.rdmaDevices
        )
    }

    private func clusterControlBaseURL() -> URL? {
        guard let agentURL = coordinator?.agentURL else { return nil }
        return URL(string: agentURL)
    }

    private func videoControlBaseURL() -> URL? {
        let endpoint = h3CoordinatorAgentURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: endpoint),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              url.host != nil
        else { return nil }
        return url
    }

    private func modelServiceURL(path: String) -> URL? {
        if let activeModelServiceBaseURL,
           var components = URLComponents(string: activeModelServiceBaseURL),
           components.host != nil {
            components.path = path
            components.query = nil
            return components.url
        }
        guard
            let agentURL = coordinator?.agentURL,
            var components = URLComponents(string: agentURL),
            components.host != nil
        else { return nil }
        components.port = modelHTTPPort
        components.path = path
        components.query = nil
        return components.url
    }

    private func chatServiceURL() -> URL? {
        if activeModelInstanceID != nil,
           let baseURL = clusterControlBaseURL() {
            return baseURL.appendingPathComponent("/v1/chat/completions")
        }
        return modelServiceURL(path: "/v1/chat/completions")
    }

    private func jsonRequest<T: Encodable>(url: URL, body: T) throws -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        return request
    }

    private func validate(_ response: HTTPURLResponse, data: Data = Data()) throws {
        guard (200..<300).contains(response.statusCode) else {
            throw TokenityTransportError.httpStatus(response.statusCode, errorDetail(from: data, status: response.statusCode))
        }
    }

    private func errorDetail(from data: Data, status: Int) -> String? {
        guard !data.isEmpty else { return nil }
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let detail = object["detail"] as? String {
                return detail
            }
            if let detail = object["detail"] as? [String: Any],
               let formatted = structuredErrorDetail(detail) {
                return formatted
            }
            if let details = object["detail"] as? [[String: Any]], !details.isEmpty {
                return details.compactMap { item in
                    guard let message = item["msg"] as? String else { return nil }
                    let location = (item["loc"] as? [Any])?
                        .compactMap { component -> String? in
                            if let string = component as? String { return string }
                            if let number = component as? NSNumber { return number.stringValue }
                            return nil
                        }
                        .filter { $0 != "body" }
                        .joined(separator: ".")
                    guard let location, !location.isEmpty else { return message }
                    return "\(location): \(message)"
                }.joined(separator: "; ")
            }
            if let error = object["error"] as? [String: Any],
               let message = error["message"] as? String {
                if status == 404, message.contains("Unknown path") {
                    return "The selected Mac is running an older Node Agent that does not support model loading. Restart the new Tokenity Node Agent from this project."
                }
                return message
            }
        }
        return String(data: data, encoding: .utf8)
    }

    private func structuredErrorDetail(_ detail: [String: Any]) -> String? {
        var parts: [String] = []
        if let stage = detail["stage"] as? String, !stage.isEmpty {
            parts.append(stage.replacingOccurrences(of: "_", with: " "))
        }
        if let message = detail["message"] as? String, !message.isEmpty {
            parts.append(message)
        }
        if let issues = detail["issues"] as? [String], !issues.isEmpty {
            parts.append(issues.joined(separator: " "))
        }
        if let suggestion = detail["suggestion"] as? String, !suggestion.isEmpty {
            parts.append(suggestion)
        }
        return parts.isEmpty ? nil : parts.joined(separator: ": ")
    }

    private func userFacingMessage(for error: Error) -> String {
        if let localized = (error as? LocalizedError)?.errorDescription {
            return localized
        }
        let bridgedMessage = (error as NSError).localizedDescription
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !bridgedMessage.isEmpty {
            return bridgedMessage
        }
        return "The cluster service is not reachable."
    }

    private func isExplicitStreamTransportFailure(_ error: Error) -> Bool {
        if error is URLError { return true }
        let bridged = error as NSError
        return bridged.domain == NSURLErrorDomain
    }

    private func streamClusterChat(
        serviceModelName: String,
        prompt: String,
        configuration: ModelRuntimeConfiguration?,
        routeConstraints: [String: JSONValue]?,
        userIndex: Int,
        assistantIndex: Int,
        requestID: UUID,
        start: Date,
        firstTokenAt: inout Date?,
        tokenEstimate: inout Int
    ) async throws {
        try ensureActiveChatRequest(requestID)
        guard let url = chatServiceURL() else {
            throw TokenityTransportError.missingModelService
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 600
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(
            chatCompletionRequest(
                model: serviceModelName,
                stream: true,
                configuration: configuration,
                routeConstraints: routeConstraints
            )
        )

        var didReceiveContent = false
        var didReceiveThinking = false
        var finishReason: String?
        var repetitionDetector = ChatRepetitionDetector()
        var tagParser = ThinkingTagStreamParser()
        var pendingThinking = ""
        var pendingContent = ""
        var lastUIFlush = Date()
        var reasoningStartedAt: Date?
        var reasoningFinishedAt: Date?
        var reasoningTokenCount = 0

        func observeReasoning(_ text: String) {
            guard !text.isEmpty else { return }
            reasoningStartedAt = reasoningStartedAt ?? Date()
            reasoningTokenCount += estimateTokens(text)
            pendingThinking += text
            didReceiveThinking = true
        }

        func observeAnswer(_ text: String) {
            guard !text.isEmpty else { return }
            if reasoningStartedAt != nil, reasoningFinishedAt == nil {
                reasoningFinishedAt = Date()
            }
            pendingContent += text
        }

        func flushPendingTokens() {
            guard !pendingThinking.isEmpty || !pendingContent.isEmpty else { return }
            guard activeChatRequestID == requestID else {
                pendingThinking = ""
                pendingContent = ""
                return
            }
            let contentBefore = chatMessages[assistantIndex].content
            let thinkingBefore = chatMessages[assistantIndex].thinking
            if !pendingThinking.isEmpty {
                chatMessages[assistantIndex].thinking = appendToken(
                    pendingThinking,
                    to: chatMessages[assistantIndex].thinking
                )
                if pendingContent.isEmpty {
                    chatMessages[assistantIndex].generationState = .reasoning
                }
                if let reasoningStartedAt {
                    let end = reasoningFinishedAt ?? Date()
                    chatMessages[assistantIndex].reasoningDurationSeconds = end.timeIntervalSince(reasoningStartedAt)
                    chatMessages[assistantIndex].reasoningTokenCount = reasoningTokenCount
                }
            }
            if !pendingContent.isEmpty {
                chatMessages[assistantIndex].content += pendingContent
                chatMessages[assistantIndex].generationState = .answering
            }
            didReceiveContent = didReceiveContent || chatMessages[assistantIndex].content != contentBefore
            didReceiveThinking = didReceiveThinking || chatMessages[assistantIndex].thinking != thinkingBefore
            pendingThinking = ""
            pendingContent = ""
            lastUIFlush = Date()
            chatScrollRevision += 1
        }
        defer { flushPendingTokens() }

        for try await event in lineStreamTransport(request) {
            try ensureActiveChatRequest(requestID)
            if case .response(let responseMetadata) = event {
                applyRouteMetadata(
                    responseMetadata.route,
                    assistantIndex: assistantIndex,
                    localRequestID: requestID
                )
                continue
            }
            guard case .line(let line) = event else { continue }
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8) else { continue }
            let chunk = try JSONDecoder().decode(OpenAIChatChunk.self, from: data)
            let choice = chunk.choices.first
            let delta = choice?.delta
            if let reason = choice?.finishReason {
                finishReason = reason
            }
            var shouldFlushImmediately = false
            let thinking = delta?.reasoningContent ?? delta?.reasoning
            if let thinking, !thinking.isEmpty {
                if repetitionDetector.observe(thinking) {
                    flushPendingTokens()
                    throw TokenityTransportError.repetitiveOutput
                }
                observeReasoning(thinking)
                if firstTokenAt == nil {
                    firstTokenAt = Date()
                    shouldFlushImmediately = true
                }
                tokenEstimate += estimateTokens(thinking)
            }
            if let content = delta?.content, !content.isEmpty {
                if repetitionDetector.observe(content) {
                    flushPendingTokens()
                    throw TokenityTransportError.repetitiveOutput
                }
                if firstTokenAt == nil {
                    firstTokenAt = Date()
                    shouldFlushImmediately = true
                }
                let fragment = tagParser.consume(content)
                observeReasoning(fragment.reasoning)
                observeAnswer(fragment.answer)
                tokenEstimate += estimateTokens(content)
            }
            if let completionTokens = chunk.usage?.completionTokens {
                tokenEstimate = completionTokens
            }
            let pendingBytes = pendingThinking.utf8.count + pendingContent.utf8.count
            if shouldFlushImmediately || pendingBytes >= 4_096 || Date().timeIntervalSince(lastUIFlush) >= 0.033 {
                flushPendingTokens()
            }
        }
        let trailing = tagParser.finish()
        observeReasoning(trailing.reasoning)
        observeAnswer(trailing.answer)
        flushPendingTokens()
        try ensureActiveChatRequest(requestID)
        if let reasoningStartedAt {
            let end = reasoningFinishedAt ?? Date()
            chatMessages[assistantIndex].reasoningDurationSeconds = end.timeIntervalSince(reasoningStartedAt)
            chatMessages[assistantIndex].reasoningTokenCount = reasoningTokenCount
        }
        if finishReason == "tokenity_repetition" {
            throw TokenityTransportError.repetitiveOutput
        }
        if !didReceiveContent && didReceiveThinking {
            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            if finishReason == "length" {
                chatMessages[assistantIndex].content = "The model reached the configured maximum output length while reasoning. Increase Max Output Tokens in Model Configuration and try again."
                chatMessages[assistantIndex].generationState = .lengthLimited
                chatMessages[assistantIndex].statusMessage = "Stopped at the configured output token limit while reasoning."
            } else {
                chatMessages[assistantIndex].content = "The model returned reasoning but did not finish a final answer. Try again if you need the final response."
                chatMessages[assistantIndex].generationState = .failed
                chatMessages[assistantIndex].statusMessage = "Reasoning ended without a final answer."
            }
            return
        }
        if !didReceiveContent {
            throw TokenityTransportError.noChatContent
        }
        if finishReason == "length" {
            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            chatMessages[assistantIndex].generationState = .lengthLimited
            chatMessages[assistantIndex].statusMessage = "The answer reached the configured output token limit."
        } else {
            chatMessages[assistantIndex].generationState = .completed
            chatMessages[assistantIndex].statusMessage = nil
        }
    }

    private func completeClusterChat(
        serviceModelName: String,
        configuration: ModelRuntimeConfiguration?,
        routeConstraints: [String: JSONValue]?,
        assistantIndex: Int,
        requestID: UUID,
        firstTokenAt: inout Date?,
        tokenEstimate: inout Int
    ) async throws {
        try ensureActiveChatRequest(requestID)
        guard let url = chatServiceURL() else {
            throw TokenityTransportError.missingModelService
        }
        var request = try jsonRequest(
            url: url,
            body: chatCompletionRequest(
                model: serviceModelName,
                stream: false,
                configuration: configuration,
                routeConstraints: routeConstraints
            )
        )
        request.timeoutInterval = 600
        let (data, response) = try await dataTransport(request)
        try ensureActiveChatRequest(requestID)
        try validate(response, data: data)
        applyRouteMetadata(
            Self.routeMetadata(from: response),
            assistantIndex: assistantIndex,
            localRequestID: requestID
        )
        let decoded = try JSONDecoder().decode(OpenAIChatResponse.self, from: data)
        guard let message = decoded.choices.first?.message else {
            throw TokenityTransportError.noChatContent
        }

        let reasoning = message.reasoningContent ?? message.reasoning ?? ""
        let content = message.content ?? ""
        guard !reasoning.isEmpty || !content.isEmpty else {
            throw TokenityTransportError.noChatContent
        }

        firstTokenAt = firstTokenAt ?? Date()
        chatMessages[assistantIndex].thinking = reasoning
        chatMessages[assistantIndex].content = ""
        if !content.isEmpty {
            appendAssistantContent(content, assistantIndex: assistantIndex)
        }
        if !reasoning.isEmpty {
            tokenEstimate += estimateTokens(reasoning)
            chatMessages[assistantIndex].reasoningTokenCount = estimateTokens(reasoning)
        }
        if !content.isEmpty {
            tokenEstimate += estimateTokens(content)
        }
        if let completionTokens = decoded.usage?.completionTokens {
            tokenEstimate = completionTokens
        }

        if chatMessages[assistantIndex].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chatMessages[assistantIndex].content = "The model returned reasoning but did not finish a final answer. Try again if you need the final response."
            chatMessages[assistantIndex].generationState = .failed
            chatMessages[assistantIndex].statusMessage = "Reasoning ended without a final answer."
        } else if decoded.choices.first?.finishReason == "length" {
            chatMessages[assistantIndex].generationState = .lengthLimited
            chatMessages[assistantIndex].statusMessage = "The answer reached the configured output token limit."
        } else {
            chatMessages[assistantIndex].generationState = .completed
            chatMessages[assistantIndex].statusMessage = nil
        }
        chatScrollRevision += 1
        if decoded.choices.first?.finishReason == "tokenity_repetition" {
            throw TokenityTransportError.repetitiveOutput
        }
    }

    private func chatRequestMessages() -> [OpenAIChatRequest.Message] {
        chatMessages
            .dropLast()
            .filter { $0.includeInContext && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .suffix(10)
            .map { OpenAIChatRequest.Message(role: $0.role.rawValue, content: $0.content) }
    }

    private func chatCompletionRequest(
        model: String,
        stream: Bool,
        configuration: ModelRuntimeConfiguration?,
        routeConstraints: [String: JSONValue]? = nil
    ) -> OpenAIChatRequest {
        let isAuto = configuration == nil
        let modelID = isAuto ? model : chatSelectedModelID
        let row = modelLibraryRows.first { $0.id == modelID }
        let sampling = configuration?.resolvedSampling(
            forQwen35: row?.usesQwen35Sampling == true
        )
        let templateArguments: [String: Bool]?
        switch configuration?.thinkingMode {
        case .automatic:
            templateArguments = nil
        case .enabled:
            templateArguments = ["enable_thinking": true]
        case .disabled:
            templateArguments = ["enable_thinking": false]
        case nil:
            templateArguments = nil
        }
        return OpenAIChatRequest(
            model: model,
            messages: chatRequestMessages(),
            stream: stream,
            maxTokens: configuration?.maximumOutputTokens,
            temperature: sampling?.temperature,
            topP: sampling?.topP,
            topK: sampling?.topK,
            minP: sampling?.minP,
            presencePenalty: sampling?.presencePenalty,
            repetitionPenalty: sampling?.repetitionPenalty,
            chatTemplateKwargs: templateArguments,
            tokenityRoutePolicy: isAuto ? chatRoutePolicy.rawValue : nil,
            tokenitySessionID: isAuto ? activeChatSessionID.uuidString : nil,
            tokenityLockModel: isAuto ? locksChatModel : nil,
            tokenityConstraints: isAuto ? routeConstraints : nil
        )
    }

    private func applyRouteMetadata(
        _ metadata: ChatRouteMetadata,
        assistantIndex: Int,
        localRequestID: UUID
    ) {
        guard chatMessages.indices.contains(assistantIndex) else { return }
        chatMessages[assistantIndex].routedModelID = metadata.routedModelID
        chatMessages[assistantIndex].modelRevision = metadata.modelRevision
        chatMessages[assistantIndex].instanceID = metadata.instanceID
        chatMessages[assistantIndex].routeReason = metadata.routeReason
        chatMessages[assistantIndex].routeConfidence = metadata.confidence
        chatMessages[assistantIndex].routingLatencyMilliseconds = metadata.routingLatencyMilliseconds
        chatMessages[assistantIndex].queueWaitMilliseconds = metadata.queueWaitMilliseconds
        chatMessages[assistantIndex].requestID = metadata.requestID ?? localRequestID.uuidString
        if let routedModelID = metadata.routedModelID, !routedModelID.isEmpty {
            chatMessages[assistantIndex].modelName = routedModelID
        }
        activeChatRoutedInstanceID = metadata.instanceID
        chatRoutingState = .routed(metadata)
        chatScrollRevision += 1
    }

    private func assistantMessageHasVisibleOutput(at index: Int) -> Bool {
        guard chatMessages.indices.contains(index) else { return false }
        let message = chatMessages[index]
        let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let thinking = message.thinking.trimmingCharacters(in: .whitespacesAndNewlines)
        return !content.isEmpty || !thinking.isEmpty
    }

    private func ensureActiveChatRequest(_ requestID: UUID) throws {
        guard !Task.isCancelled, activeChatRequestID == requestID else {
            throw CancellationError()
        }
    }

    @discardableResult
    private func cancelActiveChat(message: String, logReason: String) -> Task<Void, Never>? {
        guard let requestID = activeChatRequestID else { return nil }
        let task = chatTask
        task?.cancel()
        finishCancelledChatIfActive(requestID, message: message, logReason: logReason)
        return task
    }

    private func finishCancelledChatIfActive(_ requestID: UUID, message: String, logReason: String) {
        guard activeChatRequestID == requestID else { return }
        if let userIndex = activeChatUserIndex, chatMessages.indices.contains(userIndex) {
            chatMessages[userIndex].includeInContext = false
        }
        if let assistantIndex = activeChatAssistantIndex, chatMessages.indices.contains(assistantIndex) {
            chatMessages[assistantIndex].includeInContext = false
            let content = chatMessages[assistantIndex].content.trimmingCharacters(in: .whitespacesAndNewlines)
            chatMessages[assistantIndex].content = content.isEmpty
                ? message
                : "\(chatMessages[assistantIndex].content)\n\n\(message)"
            chatMessages[assistantIndex].generationState = .stopped
            chatMessages[assistantIndex].statusMessage = message
        }
        activeChatRequestID = nil
        activeChatUserIndex = nil
        activeChatAssistantIndex = nil
        activeChatRoutedInstanceID = nil
        chatTask = nil
        chatTaskID = nil
        isChatRunning = false
        chatRoutingState = .idle
        chatScrollRevision += 1
        syncActiveChatSession()
        appendLog("Chat generation cancelled for \(logReason).")
    }

    private func finishChatTask(_ taskID: UUID) {
        guard chatTaskID == taskID else { return }
        chatTask = nil
        chatTaskID = nil
    }

    private func appendAssistantContent(_ rawToken: String, assistantIndex: Int) {
        let fragment = ThinkingTagStreamParser.parseComplete(rawToken)
        if !fragment.reasoning.isEmpty {
            chatMessages[assistantIndex].thinking = appendToken(fragment.reasoning, to: chatMessages[assistantIndex].thinking)
        }
        if !fragment.answer.isEmpty {
            chatMessages[assistantIndex].content += fragment.answer
        }
    }

    private func appendToken(_ token: String, to existing: String) -> String {
        existing + token
    }

    private func syncActiveChatSession() {
        guard let index = chatSessions.firstIndex(where: { $0.id == activeChatSessionID }) else { return }
        var session = chatSessions[index]
        if session.titleWasEdited != true,
           let firstPrompt = chatMessages.first(where: { $0.role == .user })?.content {
            let clean = firstPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !clean.isEmpty {
                session.title = String(clean.prefix(48))
            }
        }
        if session.selectedModelID != nil || chatSelectedModelID != "tokenity-auto" {
            session.selectedModelID = chatSelectedModelID
        }
        if session.routePolicy != nil || chatRoutePolicy != .balanced {
            session.routePolicy = chatRoutePolicy
        }
        if session.locksModel != nil || locksChatModel {
            session.locksModel = locksChatModel
        }
        guard session.messages != chatMessages
                || session.metrics != chatMetrics
                || session.title != chatSessions[index].title
                || session.selectedModelID != chatSessions[index].selectedModelID
                || session.routePolicy != chatSessions[index].routePolicy
                || session.locksModel != chatSessions[index].locksModel else { return }
        session.messages = chatMessages
        session.metrics = chatMetrics
        session.updatedAt = Date()
        chatSessions[index] = session
        chatSessions.sort { $0.updatedAt > $1.updatedAt }
        persistChatSessions()
    }

    private func persistChatSessions() {
        chatHistoryRevision += 1
        let snapshot = chatSessions
        if writesChatHistorySynchronously {
            if let encoded = try? JSONEncoder().encode(snapshot) {
                userDefaults.set(encoded, forKey: chatSessionsKey)
            }
            return
        }
        let defaults = UncheckedSendableBox(value: userDefaults)
        let key = chatSessionsKey
        chatHistoryQueue.async {
            guard let encoded = try? JSONEncoder().encode(snapshot) else { return }
            defaults.value.set(encoded, forKey: key)
        }
    }

    private nonisolated static func loadChatSessions(from defaults: UserDefaults, key: String) -> [ChatSession]? {
        guard let data = defaults.data(forKey: key),
              let decoded = try? JSONDecoder().decode([ChatSession].self, from: data),
              !decoded.isEmpty else { return nil }
        return decoded.sorted { $0.updatedAt > $1.updatedAt }
    }

    private func waitForPendingChatHistoryWrites() async {
        guard !writesChatHistorySynchronously else { return }
        await withCheckedContinuation { continuation in
            chatHistoryQueue.async {
                continuation.resume()
            }
        }
    }

    private func finishChatMetrics(
        start: Date,
        firstTokenAt: Date?,
        tokenEstimate: Int,
        assistantIndex: Int? = nil
    ) {
        let total = Date().timeIntervalSince(start)
        let first = firstTokenAt?.timeIntervalSince(start)
        let rate = total > 0 ? Double(tokenEstimate) / total : nil
        chatMetrics = ChatMetrics(
            firstTokenSeconds: first,
            totalSeconds: total,
            outputTokensPerSecond: rate,
            outputTokens: tokenEstimate
        )
        if let assistantIndex, chatMessages.indices.contains(assistantIndex) {
            chatMessages[assistantIndex].metrics = chatMetrics
        }
    }

    private func estimateTokens(_ text: String) -> Int {
        max(1, text.split(whereSeparator: { $0.isWhitespace }).count)
    }

    private nonisolated static func liveData(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw TokenityTransportError.invalidResponse
        }
        return (data, httpResponse)
    }

    private nonisolated static func routeMetadata(from response: HTTPURLResponse) -> ChatRouteMetadata {
        func header(_ name: String) -> String? {
            guard let value = response.value(forHTTPHeaderField: name)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
                !value.isEmpty
            else { return nil }
            return value
        }
        func number(_ name: String) -> Double? {
            header(name).flatMap(Double.init)
        }
        return ChatRouteMetadata(
            routedModelID: header("X-Tokenity-Routed-Model") ?? header("X-Tokenity-Model"),
            modelRevision: header("X-Tokenity-Model-Revision"),
            instanceID: header("X-Tokenity-Instance-ID"),
            routeReason: header("X-Tokenity-Route-Reason"),
            confidence: number("X-Tokenity-Route-Confidence"),
            routingLatencyMilliseconds: number("X-Tokenity-Routing-Latency-Ms"),
            queueWaitMilliseconds: number("X-Tokenity-Queue-Wait-Ms"),
            requestID: header("X-Tokenity-Request-ID")
        )
    }

    private nonisolated static func liveLineStream(for request: URLRequest) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingOldest(16)) { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw TokenityTransportError.invalidResponse
                    }
                    guard (200..<300).contains(httpResponse.statusCode) else {
                        throw TokenityTransportError.httpStatus(httpResponse.statusCode, nil)
                    }
                    guard httpResponse.value(forHTTPHeaderField: "Content-Type")?
                        .lowercased()
                        .hasPrefix("text/event-stream") == true else {
                        throw TokenityTransportError.invalidResponse
                    }
                    _ = continuation.yield(
                        .response(
                            ChatStreamResponseMetadata(
                                statusCode: httpResponse.statusCode,
                                contentType: httpResponse.value(forHTTPHeaderField: "Content-Type"),
                                route: routeMetadata(from: httpResponse)
                            )
                        )
                    )
                    for try await line in bytes.lines {
                        retry: while !Task.isCancelled {
                            switch continuation.yield(.line(line)) {
                            case .enqueued:
                                break retry
                            case .dropped:
                                try await Task.sleep(for: .milliseconds(1))
                            case .terminated:
                                return
                            @unknown default:
                                return
                            }
                        }
                        try Task.checkCancellation()
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func splitThinking(_ text: String) -> (thinking: String, answer: String) {
        let fragment = ThinkingTagStreamParser.parseComplete(text)
        return (fragment.reasoning, fragment.answer)
    }
}
