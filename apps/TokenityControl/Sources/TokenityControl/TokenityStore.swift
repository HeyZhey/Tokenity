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
        }
    }
}

private struct ModelReadinessResponse: Decodable {
    var phase: String
    var message: String?
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
    @Published var chatInput = ""
    @Published var chatMessages: [ChatMessage] = [
        ChatMessage(role: .assistant, content: "Create a cluster, load a model, then send a prompt to measure first response and total generation time.")
    ]
    @Published var chatMetrics = ChatMetrics.empty
    @Published var isChatRunning = false
    @Published var logs: [String] = [
        "Tokenity Control opened.",
        "No cluster is running."
    ]

    private let dataTransport: DataTransport
    private let lineStreamTransport: LineStreamTransport
    private let mlxStartingPort = 30020
    private var loadedBackendRole: String?
    private var loadedServiceModelName: String?

    init(
        dataTransport: @escaping DataTransport = TokenityStore.liveData(for:),
        lineStreamTransport: @escaping LineStreamTransport = TokenityStore.liveLineStream(for:)
    ) {
        self.dataTransport = dataTransport
        self.lineStreamTransport = lineStreamTransport
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

    var canEditCluster: Bool {
        phase == .stopped || phase == .failed
    }

    var modelLibraryRows: [ModelLibraryRow] {
        let grouped = Dictionary(grouping: selectedNodes.flatMap { node in
            node.models.map { (node, $0) }
        }, by: { $0.1.id })

        return grouped.keys.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }.map { modelID in
            let entries = grouped[modelID] ?? []
            let nodeNames = entries.map { $0.0.displayName }.sorted()
            let path = entries.first?.1.path ?? "\(modelRoot)/\(modelID)"
            return ModelLibraryRow(
                id: modelID,
                displayName: modelID,
                nodes: nodeNames,
                availability: "\(nodeNames.count)/\(selectedNodes.count) selected Macs",
                loadState: modelLoadStates[modelID, default: .notLoaded],
                representativePath: path
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
                isOnline: true,
                ssh: "127.0.0.1"
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

    func loadModel(_ row: ModelLibraryRow) async {
        guard phase == .running else {
            appendLog("Create a cluster before loading a model.")
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

        var states = modelLoadStates
        for key in states.keys where key != row.id {
            states[key] = .notLoaded
        }
        states[row.id] = .loading
        modelLoadStates = states
        modelPath = row.representativePath
        modelLoadMessage = "Loading \(row.displayName)..."
        appendLog("Loading model: \(row.displayName).")

        do {
            if let role = loadedBackendRole {
                try await stopBackendRole(role)
            }
            let role = backendRole
            try await startBackendModel(row, role: role)
            let serviceModelName = try await waitForModelService(modelName: row.displayName, role: role)

            states = modelLoadStates
            for key in states.keys where key != row.id {
                states[key] = .notLoaded
            }
            states[row.id] = .loaded
            modelLoadStates = states
            loadedBackendRole = role
            loadedServiceModelName = serviceModelName
            modelLoadMessage = "\(row.displayName) is loaded."
            appendLog("Model loaded: \(row.displayName).")
        } catch {
            if let role = loadedBackendRole ?? (modelLoadStates[row.id] == .loading ? backendRole : nil) {
                try? await stopBackendRole(role)
            }
            states = modelLoadStates
            states[row.id] = .notLoaded
            modelLoadStates = states
            modelPath = ""
            loadedBackendRole = nil
            loadedServiceModelName = nil
            let message = userFacingMessage(for: error)
            modelLoadMessage = message
            appendLog("Model load failed: \(message)")
        }
    }

    func stopModel(_ row: ModelLibraryRow) async {
        if let role = loadedBackendRole {
            do {
                try await stopBackendRole(role)
            } catch {
                appendLog("Model stop request could not reach the cluster service.")
            }
        }
        var states = modelLoadStates
        states[row.id] = .notLoaded
        modelLoadStates = states
        if loadedModelName == nil {
            modelPath = ""
        }
        loadedBackendRole = nil
        loadedServiceModelName = nil
        modelLoadMessage = "No model loaded"
        appendLog("Model stopped: \(row.displayName).")
    }

    func sendChatMessage() async {
        let prompt = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isChatRunning else { return }
        guard let modelName = loadedModelName, phase == .running else {
            appendLog("Chat is waiting for a running cluster and loaded model.")
            return
        }
        let serviceModelName = loadedServiceModelName ?? modelName

        chatInput = ""
        chatMessages.append(ChatMessage(role: .user, content: prompt))
        chatMessages.append(ChatMessage(role: .assistant, content: "", thinking: "Preparing response..."))
        let assistantIndex = chatMessages.count - 1
        isChatRunning = true
        chatMetrics = .empty

        let start = Date()
        var firstTokenAt: Date?
        var tokenEstimate = 0

        do {
            try await streamClusterChat(
                serviceModelName: serviceModelName,
                prompt: prompt,
                assistantIndex: assistantIndex,
                start: start,
                firstTokenAt: &firstTokenAt,
                tokenEstimate: &tokenEstimate
            )
            finishChatMetrics(start: start, firstTokenAt: firstTokenAt, tokenEstimate: tokenEstimate)
        } catch {
            chatMessages[assistantIndex].thinking = ""
            chatMessages[assistantIndex].content = "The loaded model is not responding yet. Confirm the model is loaded on the cluster, then try again."
            appendLog("Chat request could not complete: \(userFacingMessage(for: error))")
        }

        isChatRunning = false
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
        if let role = loadedBackendRole {
            try? await stopBackendRole(role)
        }
        unloadAllModels()
        appendLog("Cluster stopped.")
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
        let readiness = readinessIssues(for: nodes)
        let readinessText = readiness.isEmpty ? "Ready to create" : "Needs attention"
        let summary = [
            LaunchSummaryItem(title: "Backend", value: backendMode.rawValue),
            LaunchSummaryItem(title: "Connection", value: connectionMode.rawValue),
            LaunchSummaryItem(title: "Selected Macs", value: "\(nodes.count)"),
            LaunchSummaryItem(title: "Readiness", value: readinessText),
        ]
        let warnings = backendMode == .official
            ? ["Experimental mode is best for compatibility checks. Use Tokenity Distributed Server as the stable target."]
            : ["Tokenity Distributed Server verifies readiness with a real chat probe before marking a model loaded."]
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

    private func unloadAllModels() {
        guard !modelLoadStates.isEmpty || !modelPath.isEmpty else { return }
        var states = modelLoadStates
        for key in states.keys {
            states[key] = .notLoaded
        }
        modelLoadStates = states
        modelPath = ""
        loadedServiceModelName = nil
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

    private func startBackendModel(_ row: ModelLibraryRow, role: String) async throws {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        let requestBody = AgentStartModelRequest(
            model: row.representativePath,
            nodes: selectedNodes.map(agentNodePayload(for:)),
            connectionMode: connectionMode.cliValue,
            startingPort: mlxStartingPort,
            host: "0.0.0.0",
            port: 8000,
            dryRun: false
        )
        var request = try jsonRequest(url: baseURL.appendingPathComponent(backendStartPath), body: requestBody)
        request.timeoutInterval = 30
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
    }

    private func stopBackendRole(_ role: String) async throws {
        guard let baseURL = clusterControlBaseURL() else { throw TokenityTransportError.missingClusterControl }
        var request = try jsonRequest(
            url: baseURL.appendingPathComponent("/v1/node/stop-role"),
            body: AgentStopRoleRequest(role: role, timeout: 10)
        )
        request.timeoutInterval = 15
        let (data, response) = try await dataTransport(request)
        try validate(response, data: data)
    }

    private func waitForModelService(modelName: String, role: String) async throws -> String {
        guard let url = modelServiceURL(path: "/v1/models") else { throw TokenityTransportError.missingModelService }
        let deadline = Date().addingTimeInterval(600)
        let emptyModelListDeadline = Date().addingTimeInterval(20)
        var lastError: Error?

        repeat {
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
            } catch {
                if case TokenityTransportError.modelServiceReturnedNoModels = error {
                    throw error
                }
                lastError = error
            }
            if let readiness = try? await fetchModelReadiness(), readiness.phase == "failed" {
                throw TokenityTransportError.backendExited(readiness.message ?? "The model backend reported a failed readiness state.")
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
        var request = try jsonRequest(
            url: url,
            body: OpenAIChatRequest(
                model: modelName,
                messages: [OpenAIChatRequest.Message(role: "user", content: "Reply with OK.")],
                stream: false,
                maxTokens: 16
            )
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
            ssh: launchSSH(for: node),
            lanIP: node.primaryIP == "unknown" ? nil : node.primaryIP,
            rdmaIP: node.rdma.thunderboltIP,
            rdmaDevices: node.rdma.rdmaDevices
        )
    }

    private func launchSSH(for node: TokenityNode) -> String {
        guard
            connectionMode != .ring,
            node.ssh != "127.0.0.1",
            let rdmaIP = node.rdma.thunderboltIP,
            !rdmaIP.isEmpty
        else { return node.ssh }

        if let atIndex = node.ssh.firstIndex(of: "@") {
            return "\(node.ssh[..<atIndex])@\(rdmaIP)"
        }
        return node.ssh
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
        return "The cluster service is not reachable."
    }

    private func streamClusterChat(
        serviceModelName: String,
        prompt: String,
        assistantIndex: Int,
        start: Date,
        firstTokenAt: inout Date?,
        tokenEstimate: inout Int
    ) async throws {
        guard let url = modelServiceURL(path: "/v1/chat/completions") else {
            throw TokenityTransportError.missingModelService
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        let history = chatMessages
            .dropLast()
            .filter { !$0.content.isEmpty }
            .suffix(10)
            .map { OpenAIChatRequest.Message(role: $0.role.rawValue, content: $0.content) }
        request.httpBody = try JSONEncoder().encode(
            OpenAIChatRequest(
                model: serviceModelName,
                messages: Array(history),
                stream: true,
                maxTokens: nil
            )
        )

        chatMessages[assistantIndex].thinking = "Waiting for first response..."
        var didReceiveContent = false

        for try await line in lineStreamTransport(request) {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8) else { continue }
            let chunk = try JSONDecoder().decode(OpenAIChatChunk.self, from: data)
            let delta = chunk.choices.first?.delta
            let thinking = delta?.reasoningContent ?? delta?.reasoning
            if let thinking, !thinking.isEmpty {
                chatMessages[assistantIndex].thinking = appendToken(thinking, to: chatMessages[assistantIndex].thinking)
            }
            if let content = delta?.content, !content.isEmpty {
                if firstTokenAt == nil { firstTokenAt = Date() }
                appendAssistantContent(content, assistantIndex: assistantIndex)
                tokenEstimate += estimateTokens(content)
                didReceiveContent = true
            }
        }
        if !didReceiveContent {
            throw TokenityTransportError.noChatContent
        }
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
        AsyncThrowingStream { continuation in
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
                        continuation.yield(line)
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
