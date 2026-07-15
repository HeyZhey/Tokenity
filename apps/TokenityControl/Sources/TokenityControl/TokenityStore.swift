import Foundation

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
            return "The model service responded, but did not report \(modelName) as loaded. Official MLX-LM Server may be running without a usable chat model; try Tokenity Distributed Server or Standard Network after confirming the backend runtime."
        case .backendExited(let message):
            return message
        case .noChatContent:
            return "The model service returned no answer text."
        case .rdmaNotReady(let message):
            return message
        case .nativeMTPAgentUpgradeRequired:
            return "Native MTP Required needs the current Node Agent. Restart the installed Tokenity Node Agent on every selected Mac, then try again."
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
    var phase: String
    var message: String?
    var progress: Double?
    var progressCurrent: Int?
    var progressTotal: Int?
    var nativeMTP: NativeMTPReadiness?

    enum CodingKeys: String, CodingKey {
        case phase, message, progress
        case progressCurrent = "progress_current"
        case progressTotal = "progress_total"
        case nativeMTP = "native_mtp"
    }
}

@MainActor
final class TokenityStore: ObservableObject {
    typealias DataTransport = (URLRequest) async throws -> (Data, HTTPURLResponse)
    typealias LineStreamTransport = (URLRequest) -> AsyncThrowingStream<String, Error>

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
    @Published var agentBaseURL: String = "http://127.0.0.1:9100"
    @Published var modelRoot: String = "/Users/Shared/TokenityModels"
    @Published var modelLoadStates: [String: ModelLoadState] = [:]
    @Published var launchPreview = LaunchPreview(summary: [], networkPlan: [], warnings: [], readinessIssues: [])
    @Published var isScanningModels = false
    @Published var modelScanSummary = "Not scanned"
    @Published var modelLoadMessage = "No model loaded"
    @Published private(set) var modelLoadProgress: Double?
    @Published private(set) var nativeMTPRuntime: NativeMTPReadiness?
    private static let initialChatSession = ChatSession.fresh()

    @Published var chatInput = ""
    @Published var chatMessages: [ChatMessage] = TokenityStore.initialChatSession.messages
    @Published var chatMetrics = TokenityStore.initialChatSession.metrics
    @Published var isChatRunning = false
    @Published private(set) var chatSessions: [ChatSession] = [TokenityStore.initialChatSession]
    @Published private(set) var activeChatSessionID: UUID = TokenityStore.initialChatSession.id
    @Published private(set) var chatScrollRevision = 0
    @Published private(set) var apiAccessStatus = "Not checked"
    @Published private(set) var modelConfigurations: [String: ModelRuntimeConfiguration] = [:]
    @Published var logs: [String] = [
        "Tokenity Control opened.",
        "No cluster is running."
    ]

    private let dataTransport: DataTransport
    private let lineStreamTransport: LineStreamTransport
    private let userDefaults: UserDefaults
    private let modelConfigurationsKey = "TokenityModelRuntimeConfigurations.v1"
    private let chatSessionsKey = "TokenityChatSessions.v1"
    private let mlxStartingPort = 30020
    private let modelLeaseSeconds = 30.0
    private var loadedBackendRole: String?
    private var loadedServiceModelName: String?
    private var legacyNativeMTPFallback: NativeMTPReadiness?
    private var activeModelLoadID: UUID?
    private var modelLoadTask: Task<Void, Never>?
    private var activeChatRequestID: UUID?
    private var activeChatUserIndex: Int?
    private var activeChatAssistantIndex: Int?
    private var chatTask: Task<Void, Never>?
    private var chatTaskID: UUID?

    init(
        dataTransport: @escaping DataTransport = TokenityStore.liveData(for:),
        lineStreamTransport: @escaping LineStreamTransport = TokenityStore.liveLineStream(for:),
        userDefaults: UserDefaults = .standard
    ) {
        self.dataTransport = dataTransport
        self.lineStreamTransport = lineStreamTransport
        self.userDefaults = userDefaults
        if let data = userDefaults.data(forKey: modelConfigurationsKey),
           let decoded = try? JSONDecoder().decode([String: ModelRuntimeConfiguration].self, from: data) {
            modelConfigurations = decoded
        }
        if let data = userDefaults.data(forKey: chatSessionsKey),
           let decoded = try? JSONDecoder().decode([ChatSession].self, from: data),
           let latest = decoded.sorted(by: { $0.updatedAt > $1.updatedAt }).first {
            chatSessions = decoded.sorted(by: { $0.updatedAt > $1.updatedAt })
            activeChatSessionID = latest.id
            chatMessages = latest.messages
            chatMetrics = latest.metrics
        }
        rebuildLaunchPreview()
    }

    var selectedNodes: [TokenityNode] {
        nodes.filter { selectedNodeIDs.contains($0.id) }
    }

    var coordinator: TokenityNode? {
        selectedNodes.first { $0.id == coordinatorID } ?? selectedNodes.first
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
        components.port = 8000
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
        modelLoadStates
            .filter { $0.value == .loaded }
            .map(\.key)
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
            .first
    }

    var isChatReady: Bool {
        phase == .running && loadedModelName != nil
    }

    var isModelLoading: Bool {
        modelLoadStates.values.contains(.loading)
    }

    var isModelUnloading: Bool {
        modelLoadStates.values.contains(.unloading)
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

    func newChatSession() {
        guard !isChatRunning else { return }
        syncActiveChatSession()
        let session = ChatSession.fresh()
        chatSessions.insert(session, at: 0)
        activeChatSessionID = session.id
        chatMessages = session.messages
        chatMetrics = session.metrics
        chatInput = ""
        chatScrollRevision += 1
        persistChatSessions()
    }

    func selectChatSession(_ sessionID: UUID) {
        guard !isChatRunning, sessionID != activeChatSessionID else { return }
        syncActiveChatSession()
        guard let session = chatSessions.first(where: { $0.id == sessionID }) else { return }
        activeChatSessionID = session.id
        chatMessages = session.messages
        chatMetrics = session.metrics
        chatInput = ""
        chatScrollRevision += 1
    }

    func deleteChatSession(_ sessionID: UUID) {
        guard !isChatRunning else { return }
        chatSessions.removeAll { $0.id == sessionID }
        if chatSessions.isEmpty {
            let session = ChatSession.fresh()
            chatSessions = [session]
        }
        if sessionID == activeChatSessionID,
           let next = chatSessions.sorted(by: { $0.updatedAt > $1.updatedAt }).first {
            activeChatSessionID = next.id
            chatMessages = next.messages
            chatMetrics = next.metrics
            chatInput = ""
            chatScrollRevision += 1
        }
        persistChatSessions()
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

        return grouped.keys.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }.map { modelID in
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
            return ModelLibraryRow(
                id: modelID,
                displayName: modelID,
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
                loadBlockReason: draftOnlyEntry?.loadBlockReason ?? (draftOnlyEntry == nil ? nil : "Qwen3.5 MTP weights are a speculative-decoding draft model and cannot be loaded as a standalone chat model. Load the matching Qwen3.5 base model instead.")
            )
        }
    }

    func refreshLocalAgent() async {
        guard let url = URL(string: "\(agentBaseURL)/v1/node/info") else {
            appendLog("This Mac's control service address is not valid.")
            return
        }
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let decoded = try JSONDecoder().decode(NodeInfoResponse.self, from: data)
            let node = TokenityNode(
                id: decoded.nodeID,
                hostname: decoded.hostname,
                user: decoded.user,
                agentURL: agentBaseURL,
                ips: decoded.ips,
                architecture: decoded.architecture,
                pythonPath: decoded.pythonPath,
                mlxVersion: decoded.mlxVersion,
                mlxLMVersion: decoded.mlxLMVersion,
                tokenityVersion: decoded.tokenityVersion,
                rdma: decoded.rdma,
                roles: decoded.processRoles,
                memory: decoded.memory ?? .unknown,
                models: [],
                isOnline: true
            )
            upsert(node)
            appendLog("Refreshed this Mac: \(decoded.hostname)")
            rebuildLaunchPreview()
        } catch {
            appendLog("Could not reach this Mac's control service.")
        }
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
        modelScanSummary = scanned == 0 ? "Using saved model inventory" : "\(found) model(s) across \(scanned) selected Mac(s)"
        appendLog("Model library refreshed.")
        rebuildLaunchPreview()
    }

    func startStatusRefreshLoop() async {
        await refreshSelectedNodeStatus()
        while !Task.isCancelled {
            await renewModelLeasesIfNeeded()
            try? await Task.sleep(for: .seconds(5))
            await refreshSelectedNodeStatus()
        }
    }

    func refreshSelectedNodeStatus() async {
        var updatedNodes = nodes
        for node in selectedNodes {
            guard let index = updatedNodes.firstIndex(where: { $0.id == node.id }) else { continue }
            if let info = await fetchNodeInfo(for: node) {
                updatedNodes[index].hostname = info.hostname
                updatedNodes[index].user = info.user
                updatedNodes[index].ips = info.ips
                updatedNodes[index].architecture = info.architecture
                updatedNodes[index].pythonPath = info.pythonPath
                updatedNodes[index].mlxVersion = info.mlxVersion
                updatedNodes[index].mlxLMVersion = info.mlxLMVersion
                updatedNodes[index].tokenityVersion = info.tokenityVersion
                updatedNodes[index].roles = info.processRoles
                updatedNodes[index].memory = info.memory ?? updatedNodes[index].memory
                updatedNodes[index].rdma = info.rdma
                updatedNodes[index].isOnline = true
            } else if let status = await fetchStatus(for: node) {
                updatedNodes[index].roles = status.roles
                updatedNodes[index].memory = status.memory ?? updatedNodes[index].memory
                updatedNodes[index].isOnline = true
            } else {
                updatedNodes[index].isOnline = false
            }
        }
        nodes = updatedNodes
        if loadedModelName != nil,
           let readiness = try? await fetchModelReadiness() {
            nativeMTPRuntime = readiness.nativeMTP
        }
        rebuildLaunchPreview()
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
        modelLoadTask?.cancel()
        modelLoadTask = Task { [weak self] in
            await self?.loadModel(row)
        }
    }

    func loadModel(_ row: ModelLibraryRow) async {
        guard phase == .running else {
            appendLog("Create a cluster before loading a model.")
            return
        }
        guard row.standaloneLoadable else {
            let message = row.loadBlockReason ?? "This checkpoint is draft-only and cannot be loaded as a standalone chat model."
            modelLoadMessage = message
            appendLog("Model load blocked: \(message)")
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
            return
        }
        guard clusterControlBaseURL() != nil else {
            modelLoadMessage = "Selected cluster Mac is not reachable."
            appendLog("Model load blocked because the selected cluster Mac is not reachable.")
            return
        }
        if connectionMode != .ring {
            await refreshSelectedNodeStatus()
            let issues = readinessIssues(for: selectedNodes)
            guard issues.isEmpty else {
                let message = "Thunderbolt RDMA is not ready: \(issues.joined(separator: " "))"
                modelLoadMessage = message
                appendLog("Model load blocked. \(message)")
                return
            }
        }

        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because another model started loading.",
            logReason: "model load"
        )
        if let cancelledChatTask { await cancelledChatTask.value }

        var states = modelLoadStates
        for key in states.keys where key != row.id {
            states[key] = .notLoaded
        }
        states[row.id] = .loading
        modelLoadStates = states
        modelPath = row.representativePath
        modelLoadMessage = "Loading \(row.displayName)..."
        modelLoadProgress = 0
        nativeMTPRuntime = nil
        legacyNativeMTPFallback = nil
        appendLog("Loading model: \(row.displayName).")
        let configuration = modelConfiguration(for: row.id)
        let operationID = UUID()
        activeModelLoadID = operationID

        do {
            // Clear stale roles on every selected Mac. This also makes switching
            // from GLM to Qwen deterministic after the control App has restarted.
            try await cleanupAllModelRoles()
            try ensureActiveModelLoad(operationID)
            let role = backendRole
            loadedBackendRole = role
            try await startBackendModel(row, role: role, configuration: configuration)
            try ensureActiveModelLoad(operationID)
            let serviceModelName = try await waitForModelService(
                modelName: row.displayName,
                role: role,
                operationID: operationID
            )
            try ensureActiveModelLoad(operationID)

            states = modelLoadStates
            for key in states.keys where key != row.id {
                states[key] = .notLoaded
            }
            states[row.id] = .loaded
            modelLoadStates = states
            loadedBackendRole = role
            loadedServiceModelName = serviceModelName
            activeModelLoadID = nil
            modelLoadTask = nil
            modelLoadProgress = 1
            modelLoadMessage = "\(row.displayName) is loaded."
            appendLog("Model loaded: \(row.displayName).")
        } catch is CancellationError {
            try? await cleanupAllModelRoles()
            if activeModelLoadID == operationID {
                activeModelLoadID = nil
                resetModelLoadState(message: "Model loading cancelled.")
                appendLog("Model loading cancelled: \(row.displayName).")
            }
        } catch {
            try? await cleanupAllModelRoles()
            guard activeModelLoadID == operationID else { return }
            activeModelLoadID = nil
            modelLoadTask = nil
            states = modelLoadStates
            states[row.id] = .notLoaded
            modelLoadStates = states
            modelPath = ""
            loadedBackendRole = nil
            loadedServiceModelName = nil
            modelLoadProgress = nil
            let message = userFacingMessage(for: error)
            modelLoadMessage = message
            appendLog("Model load failed: \(message)")
        }
    }

    func stopModel(_ row: ModelLibraryRow) async {
        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because the model was unloaded.",
            logReason: "model unload"
        )
        if let cancelledChatTask { await cancelledChatTask.value }
        activeModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        var states = modelLoadStates
        for key in states.keys where key != row.id {
            states[key] = .notLoaded
        }
        states[row.id] = .unloading
        modelLoadStates = states
        modelLoadProgress = nil
        nativeMTPRuntime = nil
        modelLoadMessage = "Unloading \(row.displayName) and releasing memory on all Macs..."
        appendLog("Unloading model: \(row.displayName).")
        do {
            try await cleanupAllModelRoles()
        } catch {
            appendLog("Some model stop requests could not reach a selected Mac: \(userFacingMessage(for: error))")
        }
        await refreshSelectedNodeStatus()
        resetModelLoadState(message: "No model loaded")
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
        guard let modelName = loadedModelName, phase == .running else {
            appendLog("Chat is waiting for a running cluster and loaded model.")
            return
        }
        let serviceModelName = loadedServiceModelName ?? modelName
        let configuration = modelConfiguration(for: modelName)

        chatInput = ""
        chatMessages.append(ChatMessage(role: .user, content: prompt))
        let userIndex = chatMessages.count - 1
        chatMessages.append(ChatMessage(role: .assistant, content: "", thinking: "Preparing response..."))
        let assistantIndex = chatMessages.count - 1
        let requestID = UUID()
        activeChatRequestID = requestID
        activeChatUserIndex = userIndex
        activeChatAssistantIndex = assistantIndex
        isChatRunning = true
        chatMetrics = .empty
        chatScrollRevision += 1
        defer {
            if activeChatRequestID == requestID {
                activeChatRequestID = nil
                activeChatUserIndex = nil
                activeChatAssistantIndex = nil
                chatTask = nil
                chatTaskID = nil
                isChatRunning = false
            }
            syncActiveChatSession()
        }

        let start = Date()
        var firstTokenAt: Date?
        var tokenEstimate = 0

        do {
            try await streamClusterChat(
                serviceModelName: serviceModelName,
                prompt: prompt,
                configuration: configuration,
                userIndex: userIndex,
                assistantIndex: assistantIndex,
                requestID: requestID,
                start: start,
                firstTokenAt: &firstTokenAt,
                tokenEstimate: &tokenEstimate
            )
            try ensureActiveChatRequest(requestID)
            finishChatMetrics(start: start, firstTokenAt: firstTokenAt, tokenEstimate: tokenEstimate)
        } catch is CancellationError {
            finishCancelledChatIfActive(
                requestID,
                message: "Generation stopped before completion.",
                logReason: "request cancellation"
            )
        } catch TokenityTransportError.repetitiveOutput {
            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            let placeholder = chatMessages[assistantIndex].thinking
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if placeholder == "Preparing response..." || placeholder == "Waiting for first response..." {
                chatMessages[assistantIndex].thinking = ""
            }
            let notice = "Generation stopped because repeated output was detected."
            let content = chatMessages[assistantIndex].content
                .trimmingCharacters(in: .whitespacesAndNewlines)
            chatMessages[assistantIndex].content = content.isEmpty
                ? notice
                : "\(chatMessages[assistantIndex].content)\n\n\(notice)"
            chatScrollRevision += 1
            finishChatMetrics(start: start, firstTokenAt: firstTokenAt, tokenEstimate: tokenEstimate)
            appendLog("Chat generation stopped automatically after repeated output was detected.")
        } catch let streamingError {
            if !assistantMessageHasVisibleOutput(at: assistantIndex) {
                do {
                    try ensureActiveChatRequest(requestID)
                    chatMessages[assistantIndex].thinking = "Retrying without streaming..."
                    try await completeClusterChat(
                        serviceModelName: serviceModelName,
                        configuration: configuration,
                        assistantIndex: assistantIndex,
                        requestID: requestID,
                        firstTokenAt: &firstTokenAt,
                        tokenEstimate: &tokenEstimate
                    )
                    try ensureActiveChatRequest(requestID)
                    finishChatMetrics(start: start, firstTokenAt: firstTokenAt, tokenEstimate: tokenEstimate)
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
            appendLog("Chat request could not complete: \(userFacingMessage(for: streamingError))")
        }

    }

    func createCluster() {
        if launchPreview.readinessIssues.isEmpty {
            phase = .launching
            appendLog("Cluster created with \(selectedNodes.count) selected Mac(s).")
            phase = .running
        } else {
            phase = .failed
            appendLog("Cluster creation blocked. Review readiness warnings.")
        }
    }

    func startDryRun() {
        createCluster()
    }

    func stopCluster() async {
        phase = .stopping
        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because the cluster was stopped.",
            logReason: "cluster stop"
        )
        if let cancelledChatTask { await cancelledChatTask.value }
        activeModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        do {
            try await cleanupAllModelRoles()
        } catch {
            appendLog("Cluster cleanup could not reach every selected Mac: \(userFacingMessage(for: error))")
        }
        resetModelLoadState(message: "No model loaded")
        appendLog("Cluster stopped.")
        phase = .stopped
    }

    func shutdownForApplicationTermination() async {
        let cancelledChatTask = cancelActiveChat(
            message: "Generation stopped because Tokenity is closing.",
            logReason: "application termination"
        )
        if let cancelledChatTask { await cancelledChatTask.value }
        activeModelLoadID = nil
        modelLoadTask?.cancel()
        modelLoadTask = nil
        try? await cleanupAllModelRoles()
        resetModelLoadState(message: "No model loaded")
        phase = .stopped
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
        var warnings = backendMode == .official
            ? ["Experimental mode is best for compatibility checks. Use Tokenity Distributed Server as the stable target."]
            : ["Tokenity Distributed Server verifies readiness with a real chat probe before marking a model loaded."]
        if backendMode == .distributed && nativeMTPMode == .auto {
            warnings.append("Native MTP Auto falls back to standard decoding when the model, checkpoint, or runtime is incompatible.")
        }
        launchPreview = LaunchPreview(
            summary: summary,
            networkPlan: networkPlan(for: nodes),
            warnings: warnings,
            readinessIssues: readiness
        )
    }

    private func fetchModels(for node: TokenityNode) async -> [ModelEntry]? {
        var components = URLComponents(string: "\(node.agentURL)/v1/node/models")
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
        guard let url = URL(string: "\(node.agentURL)/v1/node/status") else { return nil }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 4
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try JSONDecoder().decode(NodeStatusResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func fetchNodeInfo(for node: TokenityNode) async -> NodeInfoResponse? {
        guard let url = URL(string: "\(node.agentURL)/v1/node/info") else { return nil }
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 4
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
            return try JSONDecoder().decode(NodeInfoResponse.self, from: data)
        } catch {
            return nil
        }
    }

    private func upsert(_ node: TokenityNode) {
        if let index = nodes.firstIndex(where: { $0.id == node.id || $0.agentURL == node.agentURL }) {
            var merged = node
            if merged.models.isEmpty {
                merged.models = nodes[index].models
            }
            if merged.memory.totalBytes == nil {
                merged.memory = nodes[index].memory
            }
            nodes[index] = merged
        } else {
            nodes.append(node)
        }
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
        modelLoadProgress = nil
        nativeMTPRuntime = nil
        legacyNativeMTPFallback = nil
        modelLoadMessage = message
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
        guard connectionMode != .ring else { return [] }
        return nodes.flatMap { node -> [String] in
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
        }
    }

    private func networkPlan(for nodes: [TokenityNode]) -> [NetworkPlanRow] {
        nodes.map { node in
            let ready: Bool
            let detail: String
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

    private var backendRole: String {
        switch backendMode {
        case .official:
            return "official-mlx-lm"
        case .distributed, .singleNode:
            return "distributed-openai"
        }
    }

    private var backendStartPath: String {
        switch backendMode {
        case .official:
            return "/v1/node/start-official-mlx-lm"
        case .distributed, .singleNode:
            return "/v1/node/start-distributed-openai"
        }
    }

    private func startBackendModel(
        _ row: ModelLibraryRow,
        role: String,
        configuration: ModelRuntimeConfiguration
    ) async throws {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        let nativeMTP = effectiveNativeMTPConfiguration

        func requestBody(nativeMTP: NativeMTPConfiguration?) -> AgentStartModelRequest {
            AgentStartModelRequest(
                model: row.representativePath,
                nodes: (backendMode == .singleNode ? Array(selectedNodes.prefix(1)) : selectedNodes)
                    .map(agentNodePayload(for:)),
                connectionMode: backendMode == .singleNode ? ConnectionMode.ring.cliValue : connectionMode.cliValue,
                startingPort: mlxStartingPort,
                host: "0.0.0.0",
                port: 8000,
                dryRun: false,
                maxTokens: configuration.maximumOutputTokens,
                promptCacheSize: configuration.promptCacheSize,
                prefillStepSize: configuration.prefillStepSize,
                decodeConcurrency: configuration.decodeConcurrency,
                promptConcurrency: configuration.promptConcurrency,
                trustRemoteCode: configuration.trustRemoteCode,
                leaseSeconds: modelLeaseSeconds,
                nativeMTP: nativeMTP
            )
        }

        func send(_ body: AgentStartModelRequest) async throws {
            var request = try jsonRequest(url: baseURL.appendingPathComponent(backendStartPath), body: body)
            request.timeoutInterval = 40
            let (data, response) = try await dataTransport(request)
            try validate(response, data: data)
        }

        do {
            // A missing field is equivalent to off for both old and new
            // Agents, so standard decoding stays backward compatible.
            try await send(requestBody(nativeMTP: nativeMTP.mode == .off ? nil : nativeMTP))
        } catch TokenityTransportError.httpStatus(let status, let detail)
            where status == 422 && isUnsupportedNativeMTPField(detail) {
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
                try await send(requestBody(nativeMTP: nil))
            case .required:
                throw TokenityTransportError.nativeMTPAgentUpgradeRequired
            case .off:
                throw TokenityTransportError.httpStatus(status, detail)
            }
        }
    }

    private func isUnsupportedNativeMTPField(_ detail: String?) -> Bool {
        let normalized = detail?.lowercased() ?? ""
        return normalized.contains("extra inputs") && normalized.contains("not permitted")
    }

    private func stopBackendRole(_ role: String, on node: TokenityNode) async throws {
        guard let baseURL = URL(string: node.agentURL) else { throw TokenityTransportError.missingClusterControl }
        var request = try jsonRequest(
            url: baseURL.appendingPathComponent("/v1/node/stop-role"),
            body: AgentStopRoleRequest(role: role, timeout: 10)
        )
        request.timeoutInterval = 15
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
    }

    private func cleanupAllModelRoles() async throws {
        let roles = ["distributed-openai", "distributed-openai-rank", "single-node-openai", "official-mlx-lm"]
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
        for node in selectedNodes {
            guard let baseURL = URL(string: node.agentURL),
                  let request = try? jsonRequest(
                    url: baseURL.appendingPathComponent("/v1/node/heartbeat"),
                    body: AgentHeartbeatRequest(ttlSeconds: modelLeaseSeconds)
                  ) else { continue }
            var heartbeat = request
            heartbeat.timeoutInterval = 3
            _ = try? await dataTransport(heartbeat)
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
        let normalizedModelName = modelName.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: "-", with: "")
        let isQwen35 = modelLibraryRows.first(where: { $0.id == modelName })?.isQwen35
            ?? normalizedModelName.contains("qwen35")
        let probeBody: OpenAIChatRequest
        if isQwen35 {
            probeBody = OpenAIChatRequest(
                model: modelName,
                messages: [OpenAIChatRequest.Message(role: "user", content: "Reply with OK.")],
                stream: false,
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
                stream: false,
                maxTokens: 16
            )
        }
        var request = try jsonRequest(
            url: url,
            body: probeBody
        )
        request.timeoutInterval = 120
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        let decoded = try JSONDecoder().decode(OpenAIChatResponse.self, from: data)
        let content = decoded.choices.first?.message?.content?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let reasoning = decoded.choices.first?.message?.reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? decoded.choices.first?.message?.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? ""
        if decoded.choices.isEmpty || (content.isEmpty && reasoning.isEmpty) {
            throw TokenityTransportError.noChatContent
        }
    }

    private func fetchBackendStatus(role: String) async throws -> ProcessRole? {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        var request = URLRequest(url: baseURL.appendingPathComponent("/v1/node/status"))
        request.timeoutInterval = 5
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        return try JSONDecoder().decode(NodeStatusResponse.self, from: data).roles.first { $0.role == role }
    }

    private func fetchModelReadiness() async throws -> ModelReadinessResponse? {
        guard let url = modelServiceURL(path: "/v1/readiness") else { throw TokenityTransportError.missingModelService }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
        return try JSONDecoder().decode(ModelReadinessResponse.self, from: data)
    }

    private func updateModelLoadProgress(_ readiness: ModelReadinessResponse, modelName: String) {
        nativeMTPRuntime = readiness.nativeMTP ?? legacyNativeMTPFallback
        let phaseProgress: Double?
        switch readiness.phase {
        case "launching": phaseProgress = 0.01
        case "distributed_init": phaseProgress = 0.05
        case "loading_model": phaseProgress = readiness.progress ?? 0.1
        case "compiling": phaseProgress = readiness.progress ?? 0.96
        case "ready": phaseProgress = 1
        default: phaseProgress = readiness.progress
        }
        if let phaseProgress {
            let bounded = min(max(phaseProgress, 0), 1)
            modelLoadProgress = max(modelLoadProgress ?? 0, bounded)
        }
        if readiness.phase == "ready" {
            modelLoadMessage = "Finalizing \(modelName)..."
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

    private func modelServiceURL(path: String) -> URL? {
        guard
            let agentURL = coordinator?.agentURL,
            var components = URLComponents(string: agentURL),
            components.host != nil
        else { return nil }
        components.port = 8000
        components.path = path
        components.query = nil
        return components.url
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
            if let details = object["detail"] as? [[String: Any]], !details.isEmpty {
                return details.compactMap { item in
                    item["msg"] as? String
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

    private func streamClusterChat(
        serviceModelName: String,
        prompt: String,
        configuration: ModelRuntimeConfiguration,
        userIndex: Int,
        assistantIndex: Int,
        requestID: UUID,
        start: Date,
        firstTokenAt: inout Date?,
        tokenEstimate: inout Int
    ) async throws {
        try ensureActiveChatRequest(requestID)
        guard let url = modelServiceURL(path: "/v1/chat/completions") else {
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
                configuration: configuration
            )
        )

        chatMessages[assistantIndex].thinking = "Waiting for first response..."
        var didReceiveContent = false
        var didReceiveThinking = false
        var finishReason: String?
        var repetitionDetector = ChatRepetitionDetector()
        var pendingThinking = ""
        var pendingContent = ""
        var lastUIFlush = Date()

        func flushPendingTokens() {
            guard !pendingThinking.isEmpty || !pendingContent.isEmpty else { return }
            let contentBefore = chatMessages[assistantIndex].content
            let thinkingBefore = chatMessages[assistantIndex].thinking
            if !pendingThinking.isEmpty {
                chatMessages[assistantIndex].thinking = appendToken(
                    pendingThinking,
                    to: chatMessages[assistantIndex].thinking
                )
            }
            if !pendingContent.isEmpty {
                appendAssistantContent(pendingContent, assistantIndex: assistantIndex)
            }
            didReceiveContent = didReceiveContent || chatMessages[assistantIndex].content != contentBefore
            didReceiveThinking = didReceiveThinking || chatMessages[assistantIndex].thinking != thinkingBefore
            pendingThinking = ""
            pendingContent = ""
            lastUIFlush = Date()
            chatScrollRevision += 1
        }
        defer { flushPendingTokens() }

        for try await line in lineStreamTransport(request) {
            try ensureActiveChatRequest(requestID)
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
                pendingThinking += thinking
                if firstTokenAt == nil {
                    firstTokenAt = Date()
                    shouldFlushImmediately = true
                }
                tokenEstimate += estimateTokens(thinking)
                didReceiveThinking = true
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
                pendingContent += content
                tokenEstimate += estimateTokens(content)
            }
            let pendingBytes = pendingThinking.utf8.count + pendingContent.utf8.count
            if shouldFlushImmediately || pendingBytes >= 4_096 || Date().timeIntervalSince(lastUIFlush) >= 0.05 {
                flushPendingTokens()
            }
        }
        flushPendingTokens()
        try ensureActiveChatRequest(requestID)
        if finishReason == "tokenity_repetition" {
            throw TokenityTransportError.repetitiveOutput
        }
        if !didReceiveContent && didReceiveThinking {
            chatMessages[userIndex].includeInContext = false
            chatMessages[assistantIndex].includeInContext = false
            if finishReason == "length" {
                chatMessages[assistantIndex].content = "The model reached the configured maximum output length while reasoning. Increase Max Output Tokens in Model Configuration and try again."
            } else {
                chatMessages[assistantIndex].content = "The model returned reasoning but did not finish a final answer. Try again if you need the final response."
            }
            return
        }
        if !didReceiveContent {
            throw TokenityTransportError.noChatContent
        }
    }

    private func completeClusterChat(
        serviceModelName: String,
        configuration: ModelRuntimeConfiguration,
        assistantIndex: Int,
        requestID: UUID,
        firstTokenAt: inout Date?,
        tokenEstimate: inout Int
    ) async throws {
        try ensureActiveChatRequest(requestID)
        guard let url = modelServiceURL(path: "/v1/chat/completions") else {
            throw TokenityTransportError.missingModelService
        }
        var request = try jsonRequest(
            url: url,
            body: chatCompletionRequest(
                model: serviceModelName,
                stream: false,
                configuration: configuration
            )
        )
        request.timeoutInterval = 600
        let (data, response) = try await dataTransport(request)
        try ensureActiveChatRequest(requestID)
        try validate(response, data: data)
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
        }
        if !content.isEmpty {
            tokenEstimate += estimateTokens(content)
        }

        if chatMessages[assistantIndex].content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chatMessages[assistantIndex].content = "The model returned reasoning but did not finish a final answer. Try again if you need the final response."
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
        configuration: ModelRuntimeConfiguration
    ) -> OpenAIChatRequest {
        let modelID = loadedModelName ?? model
        let row = modelLibraryRows.first { $0.id == modelID }
        let normalizedID = modelID.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: "-", with: "")
        let isQwen35 = row?.isQwen35 ?? normalizedID.contains("qwen35")
        let sampling = configuration.resolvedSampling(forQwen35: isQwen35)
        let templateArguments: [String: Bool]?
        switch configuration.thinkingMode {
        case .automatic:
            templateArguments = nil
        case .enabled:
            templateArguments = ["enable_thinking": true]
        case .disabled:
            templateArguments = ["enable_thinking": false]
        }
        return OpenAIChatRequest(
            model: model,
            messages: chatRequestMessages(),
            stream: stream,
            maxTokens: configuration.maximumOutputTokens,
            temperature: sampling.temperature,
            topP: sampling.topP,
            topK: sampling.topK,
            minP: sampling.minP,
            presencePenalty: sampling.presencePenalty,
            repetitionPenalty: sampling.repetitionPenalty,
            chatTemplateKwargs: templateArguments
        )
    }

    private func assistantMessageHasVisibleOutput(at index: Int) -> Bool {
        guard chatMessages.indices.contains(index) else { return false }
        let message = chatMessages[index]
        let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let thinking = message.thinking.trimmingCharacters(in: .whitespacesAndNewlines)
        let placeholders: Set<String> = ["Preparing response...", "Waiting for first response..."]
        return !content.isEmpty || (!thinking.isEmpty && !placeholders.contains(thinking))
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
            let thinking = chatMessages[assistantIndex].thinking.trimmingCharacters(in: .whitespacesAndNewlines)
            if thinking == "Preparing response..." || thinking == "Waiting for first response..." || thinking == "Retrying without streaming..." {
                chatMessages[assistantIndex].thinking = ""
            }
            let content = chatMessages[assistantIndex].content.trimmingCharacters(in: .whitespacesAndNewlines)
            chatMessages[assistantIndex].content = content.isEmpty
                ? message
                : "\(chatMessages[assistantIndex].content)\n\n\(message)"
        }
        activeChatRequestID = nil
        activeChatUserIndex = nil
        activeChatAssistantIndex = nil
        chatTask = nil
        chatTaskID = nil
        isChatRunning = false
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
        let parsed = Self.splitThinking(rawToken)
        if !parsed.thinking.isEmpty {
            chatMessages[assistantIndex].thinking = appendToken(parsed.thinking, to: chatMessages[assistantIndex].thinking)
        }
        if !parsed.answer.isEmpty {
            chatMessages[assistantIndex].content += parsed.answer
        }
    }

    private func appendToken(_ token: String, to existing: String) -> String {
        if existing.isEmpty || existing == "Preparing response..." || existing == "Waiting for first response..." {
            return token
        }
        return existing + token
    }

    private func syncActiveChatSession() {
        guard let index = chatSessions.firstIndex(where: { $0.id == activeChatSessionID }) else { return }
        var session = chatSessions[index]
        session.messages = chatMessages
        session.metrics = chatMetrics
        session.updatedAt = Date()
        if let firstPrompt = chatMessages.first(where: { $0.role == .user })?.content {
            let clean = firstPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            if !clean.isEmpty {
                session.title = String(clean.prefix(48))
            }
        }
        chatSessions[index] = session
        chatSessions.sort { $0.updatedAt > $1.updatedAt }
        persistChatSessions()
    }

    private func persistChatSessions() {
        if let encoded = try? JSONEncoder().encode(chatSessions) {
            userDefaults.set(encoded, forKey: chatSessionsKey)
        }
    }

    private func finishChatMetrics(start: Date, firstTokenAt: Date?, tokenEstimate: Int) {
        let total = Date().timeIntervalSince(start)
        let first = firstTokenAt?.timeIntervalSince(start)
        let rate = total > 0 ? Double(tokenEstimate) / total : nil
        chatMetrics = ChatMetrics(firstTokenSeconds: first, totalSeconds: total, outputTokensPerSecond: rate)
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

    private nonisolated static func liveLineStream(for request: URLRequest) -> AsyncThrowingStream<String, Error> {
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
                    for try await line in bytes.lines {
                        retry: while !Task.isCancelled {
                            switch continuation.yield(line) {
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
        var thinking = ""
        var answer = ""
        var remaining = text[...]

        while let start = remaining.range(of: "<think>") {
            answer += remaining[..<start.lowerBound]
            remaining = remaining[start.upperBound...]
            if let end = remaining.range(of: "</think>") {
                thinking += remaining[..<end.lowerBound]
                remaining = remaining[end.upperBound...]
            } else {
                thinking += remaining
                return (String(thinking), String(answer))
            }
        }
        answer += remaining
        return (String(thinking), String(answer))
    }
}
