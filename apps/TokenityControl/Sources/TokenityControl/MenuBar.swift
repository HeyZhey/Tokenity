import AppKit
import SwiftUI

enum TokenityMenuBarLevel: String, Equatable {
    case stopped
    case busy
    case ready
    case warning

    var title: String {
        switch self {
        case .stopped: return "Stopped"
        case .busy: return "Working"
        case .ready: return "Ready"
        case .warning: return "Attention required"
        }
    }

    var symbol: String {
        switch self {
        case .stopped: return "circle"
        case .busy: return "clock.badge"
        case .ready: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        }
    }

    var color: Color {
        switch self {
        case .stopped: return .secondary
        case .busy: return .yellow
        case .ready: return .green
        case .warning: return .red
        }
    }
}

struct TokenityMenuBarNodeSnapshot: Identifiable, Equatable {
    var id: String
    var name: String
    var hostname: String
    var ipAddress: String
    var role: String
    var onlineText: String
    var agentHealthText: String
    var inferenceRoleText: String
    var connectionText: String
    var responseText: String
    var issue: String?

    var symbol: String {
        issue == nil && onlineText == "Online" ? "checkmark.circle.fill" : "exclamationmark.circle.fill"
    }
}

struct TokenityMenuBarSnapshot: Equatable {
    var level: TokenityMenuBarLevel
    var serverStatus: String
    var serverDetail: String?
    var listenAddress: String
    var modelName: String
    var routingHealth: String
    var inferenceMode: String
    var nativeMTP: String
    var isGenerating: Bool
    var nodes: [TokenityMenuBarNodeSnapshot]
    var startBlockReason: String?
    var canStop: Bool
    var stopBlockReason: String?
    var isRefreshing: Bool
    var overallIssue: String?

    var iconSymbol: String {
        isGenerating && level == .ready ? "ellipsis.message.fill" : level.symbol
    }

    var tooltip: String {
        var text = "Tokenity — \(level.title). Server: \(serverStatus)."
        if isGenerating { text += " Generating content." }
        if let overallIssue {
            text += " \(overallIssue)"
        } else if let serverDetail {
            text += " \(serverDetail)"
        }
        return text
    }
}

extension TokenityStore {
    var menuBarSnapshot: TokenityMenuBarSnapshot {
        let requiredNodes: [TokenityNode]
        if backendMode == .singleNode {
            requiredNodes = coordinator.map { [$0] } ?? []
        } else {
            requiredNodes = selectedNodes
        }

        let hasActiveService = selectedNodes.contains { node in
            inferenceRolesForActiveModel(on: node).contains(where: Self.isActiveInferenceRole)
        }
        let hasServiceIntent = loadedModelName != nil || hasActiveService || isModelTransitioning
        let clusterIssues = hasServiceIntent ? menuBarClusterIssues(requiredNodes: requiredNodes) : []
        let transitionPhase = Self.isTransitionPhase(phase)

        let level: TokenityMenuBarLevel
        if phase == .failed || serverHealth.isError {
            level = .warning
        } else if transitionPhase || isModelTransitioning {
            level = .busy
        } else if phase == .stopped && !hasActiveService && loadedModelName == nil {
            level = .stopped
        } else if hasServiceIntent && !clusterIssues.isEmpty {
            level = .warning
        } else if loadedModelName != nil, serverHealth == .ready, phase == .running {
            level = .ready
        } else if hasActiveService && phase == .stopped {
            level = .warning
        } else {
            level = .busy
        }

        let serverStatus: String
        switch level {
        case .stopped: serverStatus = "Stopped"
        case .busy: serverStatus = "Starting"
        case .ready: serverStatus = "Ready"
        case .warning: serverStatus = "Error"
        }

        let firstIssue = serverHealth.detail ?? clusterIssues.first
        let compatibilityNotice = menuBarCompatibilityNotice(requiredNodes: requiredNodes)
        let startBlockReason = menuBarStartBlockReason(requiredNodes: requiredNodes)
        let canStop = phase != .stopped || hasActiveService || loadedModelName != nil || isChatRunning
        let stopBlockReason = phase == .stopping
            ? "A stop operation is already in progress."
            : (canStop ? nil : "No Server or generation task is running.")

        return TokenityMenuBarSnapshot(
            level: level,
            serverStatus: serverStatus,
            serverDetail: firstIssue ?? serverHealth.detailForMenu ?? compatibilityNotice,
            listenAddress: openAIAPIBaseURL.replacingOccurrences(of: "/v1", with: ""),
            modelName: residentRoutingSummary,
            routingHealth: autoRouterHealthText,
            inferenceMode: menuBarInferenceMode,
            nativeMTP: menuBarNativeMTPName,
            isGenerating: isChatRunning,
            nodes: selectedNodes.map(menuBarNodeSnapshot),
            startBlockReason: startBlockReason,
            canStop: canStop && phase != .stopping,
            stopBlockReason: stopBlockReason,
            isRefreshing: isRefreshingStatus,
            overallIssue: firstIssue
        )
    }

    func navigateFromMenuBar(to section: AppSection) {
        selectedSection = section
    }

    private var menuBarInferenceMode: String {
        switch backendMode {
        case .singleNode:
            return "Single Mac"
        case .distributed:
            switch selectedNodes.count {
            case 2: return "Two-Mac distributed"
            case 1: return "Single Mac"
            default: return "\(selectedNodes.count)-Mac distributed"
            }
        case .official:
            return "Official MLX-LM"
        }
    }

    private var menuBarNativeMTPName: String {
        switch nativeMTPMode {
        case .off:
            return "Off"
        case .auto:
            return nativeMTPRuntime?.enabled == true ? "Native MTP" : "Compatibility"
        case .required:
            return "Native MTP"
        }
    }

    private func menuBarStartBlockReason(requiredNodes: [TokenityNode]) -> String? {
        if Self.isTransitionPhase(phase) || isModelTransitioning {
            return "A Server transition is already in progress."
        }
        if phase == .running || loadedModelName != nil || selectedNodes.contains(where: {
            inferenceRolesForActiveModel(on: $0).contains(where: Self.isActiveInferenceRole)
        }) {
            return loadedModelName == nil
                ? "The cluster is ready. Load a model from Models to start inference."
                : "The Server is already running."
        }
        if requiredNodes.isEmpty {
            return "Select at least one Mac in Cluster."
        }
        if let offline = requiredNodes.first(where: { !$0.isOnline }) {
            return "\(offline.displayName) Node Agent is offline."
        }
        if let readinessIssue = launchPreview.readinessIssues.first {
            return readinessIssue
        }
        return nil
    }

    private func menuBarClusterIssues(requiredNodes: [TokenityNode]) -> [String] {
        guard !requiredNodes.isEmpty else { return ["No required nodes are selected."] }
        var issues: [String] = []

        for node in requiredNodes where !node.isOnline {
            issues.append("\(node.displayName) is offline; the distributed service is unavailable.")
        }
        guard issues.isEmpty else {
            return ["Network is partially available, but the distributed service is unavailable."] + issues
        }

        let expectedConnection = backendMode == .singleNode ? ConnectionMode.ring.cliValue : connectionMode.cliValue
        let runtimeNodes = requiredNodes.compactMap { clusterRuntimeForActiveModel(on: $0) }
        let isProbeVerifiedLegacyService = activeModelInstanceID == nil
            && loadedModelName != nil
            && serverHealth == .ready
            && phase == .running
        let allAgentsUseLegacyTelemetry = requiredNodes.allSatisfy {
            $0.agentContract?.supports("cluster_runtime") != true
        }
        let acceptsLegacyProbeEvidence = runtimeNodes.isEmpty
            && isProbeVerifiedLegacyService
            && allAgentsUseLegacyTelemetry

        if runtimeNodes.count != requiredNodes.count && !acceptsLegacyProbeEvidence {
            issues.append("One or more Node Agents do not report cluster runtime metadata; restart the current Node Agent.")
        } else if runtimeNodes.count == requiredNodes.count {
            if runtimeNodes.contains(where: { $0.worldSize != requiredNodes.count }) {
                issues.append("The reported world size does not match the selected Mac count.")
            }
            if runtimeNodes.contains(where: { $0.connectionMode != expectedConnection }) {
                issues.append("The running connection mode does not match \(connectionMode.shortName).")
            }
            if Set(runtimeNodes.map(\.clusterID)).count != 1 {
                issues.append("Selected Macs report different distributed cluster identities.")
            }
            let ranks = Set(runtimeNodes.map(\.rank))
            if ranks != Set(0..<requiredNodes.count) {
                issues.append("Distributed rank assignments do not cover the configured world size.")
            }
            if runtimeNodes.filter({ $0.role == "controller" }).count != 1 || runtimeNodes.first(where: { $0.role == "controller" })?.rank != 0 {
                issues.append("Controller and Worker role assignments are inconsistent.")
            }
        }

        for node in requiredNodes {
            let expectedRole: String
            if node.id == coordinator?.id {
                switch backendMode {
                case .official:
                    expectedRole = "official-mlx-lm"
                case .singleNode:
                    expectedRole = "single-node-openai"
                case .distributed:
                    expectedRole = "distributed-openai"
                }
            } else {
                expectedRole = "distributed-openai-rank"
            }
            let roles = inferenceRolesForActiveModel(on: node)
            guard let role = roles.first(where: { $0.role == expectedRole }) else {
                issues.append("\(node.displayName) is missing its \(node.id == coordinator?.id ? "Controller" : "Worker") inference role.")
                continue
            }
            if !Self.isActiveInferenceRole(role) {
                let reason = role.message.map { ": \($0)" } ?? ""
                issues.append("\(node.displayName) inference role is \(role.state)\(reason).")
            }
        }

        if connectionMode != .ring && backendMode != .singleNode {
            issues.append(contentsOf: launchPreview.readinessIssues)
        }
        return issues
    }

    private func menuBarCompatibilityNotice(requiredNodes: [TokenityNode]) -> String? {
        guard activeModelInstanceID == nil,
              loadedModelName != nil,
              serverHealth == .ready,
              phase == .running,
              requiredNodes.allSatisfy({
                  clusterRuntimeForActiveModel(on: $0) == nil
              }),
              requiredNodes.allSatisfy({
                  $0.agentContract?.supports("cluster_runtime") != true
              })
        else { return nil }
        return "Ready via a verified inference probe; update the Node Agents to enable instance-level topology telemetry."
    }

    private func menuBarNodeSnapshot(_ node: TokenityNode) -> TokenityMenuBarNodeSnapshot {
        let isController = node.id == coordinator?.id
        let roles = inferenceRolesForActiveModel(on: node)
        let process = roles.first(where: Self.isActiveInferenceRole)
            ?? roles.first
        let connectionText: String
        let effectiveConnection = backendMode == .singleNode ? ConnectionMode.ring : connectionMode
        switch effectiveConnection {
        case .ring:
            connectionText = "Standard network"
        case .jaccl:
            connectionText = node.rdma.rdmaEnabled && node.rdma.thunderboltIP != nil
                ? "JACCL · Thunderbolt/RDMA"
                : "JACCL · unavailable"
        case .jacclRing:
            connectionText = node.rdma.rdmaEnabled && node.rdma.thunderboltIP != nil
                ? "JACCL · network fallback"
                : "Standard network fallback"
        }

        let responseText: String
        if let latency = node.agentLatencyMilliseconds {
            responseText = "\(Int(latency.rounded())) ms · just now"
        } else if let date = node.lastAgentResponseAt {
            responseText = "Last response \(Self.menuBarRelativeTime(from: date))"
        } else {
            responseText = "No response recorded"
        }

        var issue = node.agentError
        if issue == nil, let process, process.state.lowercased() == "failed" {
            issue = process.message ?? "The inference process failed."
        }
        if issue == nil, connectionMode != .ring, !node.rdma.rdmaEnabled {
            issue = node.rdma.rdmaErrors.first ?? "Thunderbolt/RDMA is unavailable."
        }

        return TokenityMenuBarNodeSnapshot(
            id: node.id,
            name: node.displayName,
            hostname: node.hostname,
            ipAddress: node.primaryIP,
            role: isController ? "Controller" : "Worker",
            onlineText: node.isOnline ? "Online" : "Offline",
            agentHealthText: node.isOnline ? "Healthy" : "Unreachable",
            inferenceRoleText: process.map { "\($0.role): \($0.state.capitalized)" } ?? "Not running",
            connectionText: connectionText,
            responseText: responseText,
            issue: issue
        )
    }

    private static func isActiveInferenceRole(_ role: ProcessRole) -> Bool {
        guard inferenceRoleNames.contains(role.role), role.pid != nil else { return false }
        let state = role.state.lowercased()
        return state != "stopped" && state != "failed"
    }

    private static func isTransitionPhase(_ phase: ClusterPhase) -> Bool {
        switch phase {
        case .launching, .distributedInit, .loadingModel, .compiling, .firstTokenPending, .stopping:
            return true
        case .stopped, .running, .failed:
            return false
        }
    }

    private static func menuBarRelativeTime(from date: Date) -> String {
        let seconds = max(0, Int(Date().timeIntervalSince(date)))
        if seconds < 5 { return "just now" }
        if seconds < 60 { return "\(seconds)s ago" }
        return "\(seconds / 60)m ago"
    }
}

private extension ServerHealthState {
    var isError: Bool {
        if case .error = self { return true }
        return false
    }

    var detailForMenu: String? {
        switch self {
        case .starting(let detail): return detail
        case .error(let detail): return detail
        case .stopped, .ready: return nil
        }
    }
}

struct TokenityMenuBarLabel: View, Equatable {
    let snapshot: TokenityMenuBarSnapshot

    var body: some View {
        TokenityMenuBarMark(level: snapshot.level)
            .help(snapshot.tooltip)
            .accessibilityLabel(snapshot.tooltip)
    }
}

struct TokenityBrandMark: View {
    private static let spokeCount = 10

    var body: some View {
        ZStack {
            ForEach(0..<Self.spokeCount, id: \.self) { index in
                Capsule(style: .continuous)
                    .frame(
                        width: index.isMultiple(of: 2) ? 2.7 : 2.1,
                        height: index.isMultiple(of: 2) ? 5.8 : 4.8
                    )
                    .offset(y: -5.8)
                    .rotationEffect(
                        .degrees(Double(index) * (360 / Double(Self.spokeCount)))
                    )
            }

            RoundedRectangle(cornerRadius: 1.7, style: .continuous)
                .stroke(lineWidth: 1.25)
                .frame(width: 5.2, height: 5.2)

            Circle()
                .frame(width: 1.7, height: 1.7)
        }
        .foregroundStyle(.primary)
        .frame(width: 18, height: 18)
        .accessibilityHidden(true)
    }
}

struct TokenityMenuBarMark: View {
    let level: TokenityMenuBarLevel

    var body: some View {
        ZStack {
            TokenityBrandMark()
            Circle()
                .fill(level.color)
                .frame(width: 4.8, height: 4.8)
                .overlay {
                    Circle()
                        .stroke(.background.opacity(0.9), lineWidth: 1)
                }
                .offset(x: 6.2, y: 6.0)
                .accessibilityHidden(true)
        }
        .frame(width: 20, height: 19)
    }
}

struct TokenityMenuBarLabelHost: View {
    @ObservedObject var store: TokenityStore
    @Environment(\.openWindow) private var openWindow
    @State private var didRequestInitialWindow = false

    var body: some View {
        TokenityMenuBarLabel(snapshot: store.menuBarSnapshot)
            .equatable()
            .onAppear {
                guard !didRequestInitialWindow else { return }
                didRequestInitialWindow = true
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }
    }
}

struct TokenityMenuBarContentHost: View {
    @ObservedObject var store: TokenityStore

    var body: some View {
        TokenityMenuBarContent(snapshot: store.menuBarSnapshot, store: store)
            .equatable()
    }
}

struct TokenityMenuBarContent: View, Equatable {
    let snapshot: TokenityMenuBarSnapshot
    let store: TokenityStore

    @Environment(\.openWindow) private var openWindow

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.snapshot == rhs.snapshot && lhs.store === rhs.store
    }

    var body: some View {
        Section {
            Label("Tokenity · \(snapshot.level.title)", systemImage: snapshot.level.symbol)
            Text("Server: \(snapshot.serverStatus)")
            if let detail = snapshot.serverDetail {
                Text(detail)
            }
            Text("Listen: \(snapshot.listenAddress)")
            Text("Models: \(snapshot.modelName)")
            Text("Mode: \(snapshot.inferenceMode)")
            Text("Native MTP: \(snapshot.nativeMTP)")
            Text("Generation: \(snapshot.isGenerating ? "Generating" : "Idle")")
        }

        Divider()

        Menu("Machines (\(snapshot.nodes.count))") {
            ForEach(snapshot.nodes) { node in
                Menu {
                    Text("Host: \(node.hostname)")
                    Text("IP: \(node.ipAddress)")
                    Text("Role: \(node.role)")
                    Text("Node Agent: \(node.agentHealthText)")
                    Text("Inference: \(node.inferenceRoleText)")
                    Text("Connection: \(node.connectionText)")
                    Text("Response: \(node.responseText)")
                    if let issue = node.issue {
                        Divider()
                        Text("Issue: \(issue)")
                    }
                } label: {
                    Label("\(node.name) · \(node.onlineText)", systemImage: node.symbol)
                }
                .accessibilityLabel("\(node.name), \(node.role), \(node.onlineText), Node Agent \(node.agentHealthText)")
            }
        }

        Divider()

        Button("Open Tokenity") {
            showMainWindow()
        }
        .keyboardShortcut("o")

        Button("Start Server") {
            store.createCluster()
        }
        .disabled(snapshot.startBlockReason != nil)
        if let reason = snapshot.startBlockReason {
            Text("Start unavailable: \(reason)")
        }

        Button("Stop Server") {
            Task { await store.stopCluster() }
        }
        .disabled(!snapshot.canStop)
        if let reason = snapshot.stopBlockReason {
            Text("Stop unavailable: \(reason)")
        }

        Button(snapshot.isRefreshing ? "Refreshing Status…" : "Refresh Status") {
            Task { await store.refreshSelectedNodeStatus() }
        }
        .disabled(snapshot.isRefreshing)

        Divider()

        Button("Open Models") {
            store.navigateFromMenuBar(to: .models)
            showMainWindow()
        }
        Button("Open Chat") {
            store.navigateFromMenuBar(to: .chat)
            showMainWindow()
        }

        Divider()

        Button("Quit Tokenity") {
            NSApp.terminate(nil)
        }
        .keyboardShortcut("q")
    }

    private func showMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.title == "Tokenity" }) {
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
        } else {
            openWindow(id: "main")
        }
    }
}
