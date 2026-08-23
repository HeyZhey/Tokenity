import SwiftUI
import AppKit

struct TokenityRootView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        NavigationSplitView {
            SidebarView(selection: $store.selectedSection)
            .navigationSplitViewColumnWidth(min: 190, ideal: 210, max: 260)
        } detail: {
            switch store.selectedSection ?? .overview {
            case .overview: OverviewPage()
            case .cluster: ClusterPage()
            case .chat: ChatWorkspaceView()
            case .video: VideoGenerationPage()
            case .models: ModelsPage()
            case .network: NetworkPage()
            case .api: APIAccessPage()
            case .logs: LogsPage()
            case .settings: SettingsPage()
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 1_120, idealWidth: 1_280, minHeight: 720, idealHeight: 820)
        .background(theme.window)
    }
}

struct SidebarView: View {
    @Binding var selection: AppSection?

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            sidebarBrand

            Rectangle()
                .fill(theme.border)
                .frame(height: 0.5)
                .padding(.horizontal, 8)

            sidebarGroup("Cluster", sections: AppSection.allCases.filter { $0.group == "Cluster" })
            sidebarGroup("Operations", sections: AppSection.allCases.filter { $0.group == "Operations" })
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.top, 17)
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(theme.sidebar)
    }

    private var sidebarBrand: some View {
        VStack(spacing: 1) {
            TokenityBrandLockup()
                .frame(width: 112, height: 94)
            Text("Distributed AI")
                .font(.tokenityText(9.5, weight: .medium))
                .tracking(0.25)
                .foregroundStyle(theme.tertiaryText)
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Tokenity, Distributed AI")
        .accessibilityIdentifier("tokenity-sidebar-brand")
    }

    private func sidebarGroup(_ title: String, sections: [AppSection]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.tokenityText(11, weight: .semibold))
                .foregroundStyle(theme.tertiaryText)
                .padding(.horizontal, 8)
                .accessibilityAddTraits(.isHeader)
            ForEach(sections) { section in
                Button {
                    selection = section
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: section.symbol)
                            .font(.system(size: 13, weight: .regular))
                            .frame(width: 18)
                        Text(section.title)
                            .font(.tokenityText(13, weight: .medium))
                            .lineLimit(1)
                        Spacer(minLength: 0)
                    }
                    .foregroundStyle(selection == section ? theme.accent : theme.text)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 7)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(selection == section ? theme.selection : Color.clear)
                    )
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .help(section.title)
            }
        }
    }
}

struct OverviewPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        PageScaffold(title: "Overview") {
            InfoGroup(title: "Cluster State") {
                InfoRow(label: "Status") {
                    StatusPill(text: store.phase.rawValue, tone: tone(for: store.phase))
                }
                InfoRow(label: "Chat Access") {
                    Text(store.openAIEndpoint).lineLimit(1).truncationMode(.middle)
                }
                InfoRow(label: "Backend") {
                    Text(store.activeBackendDisplayName).lineLimit(1)
                }
                InfoRow(label: "Connection") {
                    Text(store.activeConnectionDisplayName)
                }
                InfoRow(label: "LAN Discovery") {
                    HStack(spacing: 8) {
                        StatusPill(
                            text: store.isDiscoveringNodes
                                ? "scanning"
                                : store.lastNodeDiscoveryAt == nil ? "waiting" : "automatic",
                            tone: store.isDiscoveringNodes
                                ? .accent
                                : store.lastNodeDiscoveryAt == nil ? .neutral : .good
                        )
                        Text(store.nodeDiscoverySummary)
                            .foregroundStyle(theme.secondaryText)
                            .lineLimit(1)
                    }
                }
            }

            InfoGroup(title: "Resident Models") {
                InfoRow(label: "Pool") {
                    Text(store.residentRoutingSummary)
                }
                InfoRow(label: "Router") {
                    StatusPill(
                        text: store.autoRouterHealthText,
                        tone: store.autoRouterHealthText == "Auto routing healthy" ? .good : .warning
                    )
                }
            }

            InfoGroup(title: "Selected Macs") {
                ForEach(store.selectedNodes) { node in
                    InfoRow(label: node.displayName) {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                StatusPill(text: node.agentHealthState.rawValue, tone: agentHealthTone(node.agentHealthState))
                                if !node.identityDetail.isEmpty {
                                    Text(node.identityDetail).lineLimit(1)
                                }
                                StatusPill(text: node.source.rawValue, tone: .neutral)
                                Spacer()
                                Text(node.rdma.rdmaEnabled ? "Thunderbolt ready" : "Standard network")
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            if let detail = node.agentHealthDetail {
                                HStack(spacing: 8) {
                                    Text(detail)
                                        .font(.tokenityText(11))
                                        .foregroundStyle(theme.secondaryText)
                                        .lineLimit(2)
                                    Spacer()
                                    Text("Failures \(node.consecutiveAgentFailures) · Restarts \(node.watchdogRestartCount)")
                                        .font(.tokenityMono(10))
                                        .foregroundStyle(theme.tertiaryText)
                                }
                            }
                            MemoryUsageBar(memory: node.memory)
                        }
                    }
                }
            }

            if !store.launchPreview.readinessIssues.isEmpty {
                InfoGroup(title: "Readiness") {
                    ForEach(store.launchPreview.readinessIssues, id: \.self) { issue in
                        InfoRow(label: "Blocked") {
                            Text(issue).foregroundStyle(theme.warning)
                        }
                    }
                }
            }
        }
    }

    private func tone(for phase: ClusterPhase) -> StatusPill.Tone {
        switch phase {
        case .running: return .good
        case .readyToLoad: return .accent
        case .failed: return .danger
        case .launching, .distributedInit, .loadingModel, .compiling, .firstTokenPending, .stopping: return .warning
        case .stopped: return .neutral
        }
    }

    private func agentHealthTone(_ state: AgentHealthDisplayState) -> StatusPill.Tone {
        switch state {
        case .online, .restarted: return .good
        case .degraded, .recovering, .restarting: return .warning
        case .unreachable: return .neutral
        case .circuitOpen: return .danger
        }
    }
}

struct ClusterPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var manualAgentURL = ""

    var body: some View {
        PageScaffold(title: "Cluster") {
            InfoGroup(title: "Cluster Builder") {
                ClusterNodeCanvas(
                    nodes: store.clusterBuilderNodes,
                    selectedNodeIDs: store.selectedNodeIDs,
                    coordinatorID: store.coordinatorID,
                    canEdit: store.canEditCluster,
                    onToggle: store.toggleNodeSelection
                )
            }

            InfoGroup(title: "Cluster Setup") {
                InfoRow(label: "Runtime") {
                    HStack(spacing: 10) {
                        Label(store.effectiveBackendMode == .singleNode ? "Single Mac" : "Multiple Macs", systemImage: "server.rack")
                            .lineLimit(1)
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(theme.tertiaryText)
                        if store.effectiveBackendMode == .singleNode {
                            Label("Single Mac", systemImage: connectionSymbol)
                                .lineLimit(1)
                        } else {
                            HStack(spacing: 4) {
                                Image(systemName: connectionSymbol)
                                Picker("Connection", selection: $store.connectionMode) {
                                    Text("Automatic").tag(ConnectionMode.jacclRing)
                                    Text("Standard Network").tag(ConnectionMode.ring)
                                    Text("Thunderbolt RDMA").tag(ConnectionMode.jaccl)
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                                .fixedSize()
                            }
                            .disabled(!store.canEditCluster)

                            Text(store.connectionPreferenceDetail)
                                .font(.tokenityText(11))
                                .foregroundStyle(theme.secondaryText)
                                .lineLimit(2)
                        }
                        Spacer(minLength: 0)
                    }
                }
                InfoRow(label: "LAN Discovery") {
                    HStack(spacing: 8) {
                        StatusPill(
                            text: store.isDiscoveringNodes
                                ? "scanning"
                                : store.lastNodeDiscoveryAt == nil ? "waiting" : "automatic",
                            tone: store.isDiscoveringNodes
                                ? .accent
                                : store.lastNodeDiscoveryAt == nil ? .neutral : .good
                        )
                        Text(store.nodeDiscoverySummary)
                            .foregroundStyle(theme.secondaryText)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                        if let lastDiscovery = store.lastNodeDiscoveryAt {
                            Text(lastDiscovery.formatted(date: .omitted, time: .standard))
                                .font(.tokenityMono(10))
                                .foregroundStyle(theme.tertiaryText)
                        }
                    }
                }
                InfoRow(label: "Connect by URL") {
                    HStack(spacing: 10) {
                        TextField("http://node-name.local:9100", text: $manualAgentURL)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit(connectManualAgent)
                        Button(store.isConnectingNode ? "Connecting…" : "Connect") {
                            connectManualAgent()
                        }
                        .buttonStyle(.bordered)
                        .disabled(
                            manualAgentURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                || store.isConnectingNode
                                || !store.canEditCluster
                        )
                    }
                    .help("Connect directly when automatic LAN discovery is unavailable. The verified endpoint is saved by node identity.")
                }
                InfoRow(label: "Actions") {
                    HStack(spacing: 10) {
                        Button {
                            if store.phase == .readyToLoad {
                                store.selectedSection = .models
                            } else {
                                store.createCluster()
                            }
                        } label: {
                            Label(
                                store.phase == .readyToLoad ? "Open Models" : "Create Cluster",
                                systemImage: store.phase == .readyToLoad ? "cube.transparent" : "play.fill"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .tint(theme.controlAccent)
                        .disabled(
                            !store.launchPreview.readinessIssues.isEmpty
                                || store.selectedNodes.isEmpty
                                || (!store.canEditCluster && store.phase != .readyToLoad)
                        )

                        Button {
                            store.stop()
                        } label: {
                            Label(store.phase == .readyToLoad ? "Close Cluster" : "Stop Cluster", systemImage: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .tint(theme.danger)
                        .disabled(!store.canStopCluster)

                        Button {
                            Task {
                                await store.discoverNodes()
                                await store.refreshSelectedNodeStatus()
                            }
                        } label: {
                            Label("Discover Macs", systemImage: "dot.radiowaves.left.and.right")
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(theme.secondaryText)
                        .disabled(store.isDiscoveringNodes)
                        .help("Scan local subnets for Tokenity Node Agents and update changed addresses")
                        Spacer(minLength: 0)
                    }
                }
            }

            InfoGroup(title: "Inference Acceleration") {
                InfoRow(label: "Native MTP") {
                    HStack(spacing: 10) {
                        StatusPill(text: "Automatic", tone: .accent)
                        Text("Tokenity enables compatible acceleration and safely falls back when needed.")
                            .foregroundStyle(theme.secondaryText)
                        Spacer(minLength: 0)
                    }
                }
                InfoRow(label: "Capability") {
                    if let capability = store.aggregatedNativeMTPCapability {
                        HStack(spacing: 8) {
                            StatusPill(
                                text: capability.displayStatus,
                                tone: capability.status == "supported" ? .good : capability.status == "unknown" ? .neutral : .warning
                            )
                            Text(capability.message ?? capability.modelType ?? "Static checkpoint metadata inspected")
                                .foregroundStyle(theme.secondaryText)
                                .lineLimit(2)
                        }
                    } else {
                        Text("Scan Models to inspect checkpoint capability on selected Macs.")
                            .foregroundStyle(theme.secondaryText)
                    }
                }
                if let runtime = store.nativeMTPRuntime {
                    InfoRow(label: "Runtime") {
                        HStack(spacing: 8) {
                            StatusPill(text: runtime.enabled ? "Enabled" : "Standard decode", tone: runtime.enabled ? .good : .neutral)
                            Text(runtime.enabled ? "Acceleration is active" : "Standard decoding is active")
                                .foregroundStyle(theme.secondaryText)
                        }
                    }
                    if runtime.fallbackReason != nil {
                        InfoRow(label: "Fallback reason") {
                            Text(runtime.message ?? "Tokenity selected the compatible decoding path.")
                                .foregroundStyle(theme.warning)
                        }
                    }
                    if let proposed = runtime.proposedTokens,
                       let accepted = runtime.acceptedTokens {
                        InfoRow(label: "Acceptance") {
                            Text("\(accepted)/\(proposed) drafts · \(Int(((runtime.acceptanceRate ?? 0) * 100).rounded()))%")
                                .foregroundStyle(theme.secondaryText)
                        }
                    }
                }
            }

            if !store.launchPreview.readinessIssues.isEmpty {
                InfoGroup(title: "Readiness") {
                    ForEach(store.launchPreview.readinessIssues, id: \.self) { issue in
                        InfoRow(label: "Needs attention") {
                            Text(issue).foregroundStyle(theme.warning)
                        }
                    }
                }
            }

            InfoGroup(title: "Cluster Plan") {
                ForEach(store.launchPreview.summary) { item in
                    InfoRow(label: item.title) {
                        Text(item.value)
                            .lineLimit(1)
                    }
                }
            }

            InfoGroup(title: "Selected Macs") {
                ForEach(store.launchPreview.networkPlan) { row in
                    InfoRow(label: row.nodeName) {
                        HStack {
                            StatusPill(text: row.role, tone: .accent)
                            Text(row.link)
                                .lineLimit(1)
                            Spacer()
                            StatusPill(text: row.readiness, tone: row.readiness == "Ready" ? .good : .warning)
                        }
                    }
                }
            }
        }
    }

    private func connectManualAgent() {
        let candidate = manualAgentURL
        Task {
            if await store.connectNode(agentURL: candidate) {
                manualAgentURL = ""
            }
        }
    }

    private var connectionSymbol: String {
        switch store.effectiveConnectionMode {
        case .ring: return "network"
        case .jaccl: return "bolt.horizontal.fill"
        case .jacclRing: return "arrow.triangle.branch"
        }
    }

}

private struct ClusterNodeCanvas: View {
    let nodes: [TokenityNode]
    let selectedNodeIDs: Set<String>
    let coordinatorID: String
    let canEdit: Bool
    let onToggle: (TokenityNode) -> Void

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Canvas { context, size in
                    drawLinks(in: &context, size: size)
                }

                ForEach(Array(nodes.enumerated()), id: \.element.id) { index, node in
                    Button {
                        onToggle(node)
                    } label: {
                        ClusterNodeCard(
                            node: node,
                            isSelected: selectedNodeIDs.contains(node.id),
                            role: role(for: node),
                            canEdit: canEdit
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canEdit)
                    .position(point(for: index, count: nodes.count, in: proxy.size))
                }

                if !canEdit {
                    Text("Stop the cluster to change selected Macs")
                        .font(.tokenityText(12, weight: .medium))
                        .foregroundStyle(theme.secondaryText)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 7)
                        .background(theme.window.opacity(0.9), in: Capsule())
                        .position(x: proxy.size.width / 2, y: proxy.size.height - 24)
                }
            }
        }
        .frame(minHeight: 360, idealHeight: 384)
        .padding(12)
        .background(theme.surface)
    }

    private func drawLinks(in context: inout GraphicsContext, size: CGSize) {
        let selected = nodes.enumerated()
            .filter { selectedNodeIDs.contains($0.element.id) }
            .map {
                (
                    id: $0.element.id,
                    point: point(for: $0.offset, count: nodes.count, in: size)
                )
            }

        guard selected.count > 1 else { return }

        let coordinator = selected.first(where: { $0.id == coordinatorID }) ?? selected[0]
        for worker in selected where worker.id != coordinator.id {
            var path = Path()
            path.move(to: coordinator.point)
            path.addLine(to: worker.point)
            context.stroke(
                path,
                with: .color(theme.accent.opacity(0.38)),
                style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [7, 7])
            )
        }
    }

    private func role(for node: TokenityNode) -> String {
        guard selectedNodeIDs.contains(node.id) else { return "Available" }
        return node.id == coordinatorID ? "Coordinator" : "Worker"
    }

    private func point(for index: Int, count: Int, in size: CGSize) -> CGPoint {
        let points: [UnitPoint]
        switch count {
        case 1:
            points = [UnitPoint(x: 0.5, y: 0.5)]
        case 2:
            points = [UnitPoint(x: 0.33, y: 0.5), UnitPoint(x: 0.67, y: 0.5)]
        case 3:
            points = [UnitPoint(x: 0.5, y: 0.27), UnitPoint(x: 0.25, y: 0.68), UnitPoint(x: 0.75, y: 0.68)]
        default:
            points = [
                UnitPoint(x: 0.5, y: 0.18),
                UnitPoint(x: 0.22, y: 0.52),
                UnitPoint(x: 0.78, y: 0.52),
                UnitPoint(x: 0.5, y: 0.84)
            ]
        }

        let unit = points[index % points.count]
        return CGPoint(x: size.width * unit.x, y: size.height * unit.y)
    }
}

private struct ClusterNodeCard: View {
    let node: TokenityNode
    let isSelected: Bool
    let role: String
    let canEdit: Bool

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 8) {
                Image(systemName: "laptopcomputer")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                Text(node.displayName)
                    .font(.tokenityText(13, weight: .semibold))
                    .lineLimit(1)
                    .help(node.displayName)
                Spacer(minLength: 0)
            }

            HStack(spacing: 7) {
                if !node.identityDetail.isEmpty {
                    Text(node.identityDetail)
                        .font(.tokenityText(10))
                        .foregroundStyle(theme.tertiaryText)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(node.identityDetail)
                }
                Spacer(minLength: 0)
                StatusPill(text: role, tone: isSelected ? .accent : .neutral)
                    .fixedSize(horizontal: true, vertical: false)
            }

            VStack(alignment: .leading, spacing: 5) {
                nodeSignal(label: "Runtime", value: node.displayRuntime)
                if !node.displayName.contains(node.primaryIP) {
                    nodeSignal(label: "IP", value: node.primaryIP)
                }
                nodeSignal(label: "Source", value: node.source.rawValue)
                nodeSignal(label: "Memory", value: node.memoryPercentText)
                MemoryUsageBar(memory: node.memory)
            }
        }
        .padding(14)
        .frame(width: 228, height: 184, alignment: .leading)
        .background(theme.raisedSurface, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(
                    isSelected ? theme.accent.opacity(0.82) : theme.border,
                    lineWidth: isSelected ? 1.2 : 0.7
                )
        )
        .opacity(canEdit || isSelected ? 1 : 0.76)
    }

    private func nodeSignal(label: String, value: String) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.tokenityText(11, weight: .medium))
                .foregroundStyle(theme.tertiaryText)
                .frame(width: 44, alignment: .leading)
            Text(value)
                .font(label == "IP" ? .tokenityMono(10.5) : .tokenityText(11))
                .foregroundStyle(theme.secondaryText)
                .lineLimit(1)
                .help(value)
            Spacer(minLength: 0)
        }
    }
}

struct NetworkPage: View {
    @EnvironmentObject private var store: TokenityStore

    var body: some View {
        PageScaffold(title: "Network / RDMA") {
            InfoGroup(title: "RDMA Readiness") {
                ForEach(store.selectedNodes) { node in
                    InfoRow(label: node.displayName) {
                        HStack {
                            StatusPill(text: node.rdma.rdmaEnabled ? "ready" : "blocked", tone: node.rdma.rdmaEnabled ? .good : .warning)
                            Text(node.rdma.rdmaEnabled ? "Direct Thunderbolt link available" : "Direct Thunderbolt link not detected")
                            Spacer()
                            Text("Selected")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            InfoGroup(title: "Node Properties") {
                ForEach(store.selectedNodes) { node in
                    InfoRow(label: node.displayName) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 12) {
                                if !node.identityDetail.isEmpty {
                                    Text(node.identityDetail)
                                        .font(.tokenityMono(11))
                                        .lineLimit(1)
                                }
                                Text(node.agentURL)
                                    .font(.tokenityMono(11))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                            MemoryUsageBar(memory: node.memory)
                        }
                    }
                }
            }

            InfoGroup(title: "Cluster Link Plan") {
                ForEach(store.launchPreview.networkPlan) { row in
                    InfoRow(label: row.nodeName) {
                        VStack(alignment: .leading, spacing: 3) {
                            HStack {
                                Text(row.link)
                                Spacer()
                                StatusPill(text: row.readiness, tone: row.readiness == "Ready" ? .good : .warning)
                            }
                            Text(row.detail)
                                .font(.tokenityText(12))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }
}

struct ModelsPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var configurationTarget: ModelLibraryRow?

    var body: some View {
        PageScaffold(title: "Models") {
            InfoGroup(title: "Model Library") {
                InfoRow(label: "Default folder") {
                    TextField("Model folder", text: $store.modelRoot)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            Task { await store.scanModels() }
                        }
                }
                InfoRow(label: "Cluster") {
                    HStack {
                        StatusPill(text: store.phase.rawValue, tone: clusterStatusTone)
                        Text(clusterStatusDetail)
                            .foregroundStyle(theme.secondaryText)
                            .lineLimit(1)
                    }
                }
                InfoRow(label: "Video runtime") {
                    HStack {
                        StatusPill(
                            text: store.videoRuntimeState.title,
                            tone: store.isVideoRuntimeReady ? .good : (store.videoRuntimeState == .stopped ? .neutral : .warning)
                        )
                        Text(store.videoTopologySummary)
                            .foregroundStyle(theme.secondaryText)
                            .lineLimit(1)
                        Spacer()
                        Button("Open Video") {
                            store.selectedSection = .video
                        }
                    }
                }
                InfoRow(label: "Model status") {
                    Text(store.loadedModelName ?? store.modelLoadMessage)
                        .foregroundStyle(store.loadedModelName == nil ? theme.secondaryText : theme.text)
                        .lineLimit(1)
                }
                InfoRow(label: "Inventory") {
                    HStack {
                        Text(store.modelScanSummary)
                            .lineLimit(1)
                        Spacer()
                        Button {
                            Task { await store.scanModels() }
                        } label: {
                            Label(store.isScanningModels ? "Scanning" : "Scan Models", systemImage: "arrow.clockwise")
                        }
                        .disabled(store.isScanningModels)
                    }
                }
            }

            InfoGroup(title: "Resident Language Model Pool") {
                if store.residentModelInstances.isEmpty {
                    InfoRow(label: "Status") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("No resident model instances")
                            Text(store.autoRouterHealthText)
                                .font(.tokenityText(11))
                                .foregroundStyle(theme.secondaryText)
                        }
                    }
                } else {
                    ForEach(store.residentModelInstances) { instance in
                        ResidentModelInstanceRow(instance: instance)
                    }
                }
            }

            InfoGroup(title: "Language Models") {
                let languageModels = store.modelLibraryRows.filter { $0.modality == .language }
                if languageModels.isEmpty {
                    InfoRow(label: "Status") {
                        Text("No language model found. Choose the model folder above, then scan again.")
                            .foregroundStyle(theme.secondaryText)
                    }
                } else {
                    ForEach(languageModels) { row in
                        modelLoadRow(for: row)
                    }
                }
            }

            InfoGroup(title: "Video Models") {
                let videoModels = store.modelLibraryRows.filter { $0.modality == .video }
                if videoModels.isEmpty {
                    InfoRow(label: "MiniMax H3") {
                        Text("Video model not found. Open Video to choose its folder and scan again.")
                            .foregroundStyle(theme.secondaryText)
                    }
                } else {
                    ForEach(videoModels) { row in
                        modelLoadRow(for: row)
                    }
                }
            }
        }
        .sheet(item: $configurationTarget) { row in
            ModelConfigurationSheet(
                row: row,
                initialConfiguration: store.modelConfiguration(for: row.id),
                saveAction: { configuration in
                    store.updateModelConfiguration(configuration, for: row.id)
                }
            )
        }
        .task {
            await store.scanModels()
        }
    }

    private func modelLoadRow(for row: ModelLibraryRow) -> some View {
        let loadingProgress = row.loadState == .loading
            ? (row.modality == .video ? store.videoRuntimeLoadProgress : store.modelLoadProgress)
            : nil
        return ModelLoadRow(
            row: row,
            selectedNodeCount: store.modelLoadRequiredNodeCount,
            loadTargetSummary: store.modelLoadTargetSummary(for: row),
            clusterIsReady: store.isClusterConfigured && !store.isModelTransitioning,
            loadEnabled: store.canLoadModel(row),
            loadButtonHelp: store.modelLoadHelp(for: row),
            loadingProgress: loadingProgress,
            topologyIssue: store.topologyIssue(for: row),
            configurationAction: { configure(row) },
            loadAction: { store.beginLoadingModel(row) },
            loadOnOneMacAction: { store.loadOnOneMac(row) },
            stopAction: { Task { await store.stopModel(row) } }
        )
    }

    private func configure(_ row: ModelLibraryRow) {
        if row.modality == .video {
            store.selectedSection = .video
        } else {
            configurationTarget = row
        }
    }

    private var clusterStatusDetail: String {
        switch store.phase {
        case .stopped:
            return "Create the cluster before loading a model"
        case .readyToLoad:
            return "Models can be loaded now"
        case .running:
            return "The cluster is serving a loaded model"
        case .failed:
            return "Review the cluster status before loading a model"
        case .launching, .distributedInit, .loadingModel, .compiling, .firstTokenPending, .stopping:
            return "A cluster operation is in progress"
        }
    }

    private var clusterStatusTone: StatusPill.Tone {
        switch store.phase {
        case .running: return .good
        case .readyToLoad: return .accent
        case .failed: return .danger
        case .launching, .distributedInit, .loadingModel, .compiling, .firstTokenPending, .stopping: return .warning
        case .stopped: return .neutral
        }
    }
}

private struct ResidentModelInstanceRow: View {
    let instance: ResidentModelInstanceSummary

    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 9) {
                Circle()
                    .fill(instance.isBusy ? theme.warning : (instance.isReady ? theme.success : theme.danger))
                    .frame(width: 9, height: 9)
                Text(instance.modelID)
                    .font(.tokenityMono(12, weight: .semibold))
                    .lineLimit(1)
                    .help(instance.modelID)
                StatusPill(text: instance.displayState, tone: stateTone)
                if instance.activeRequestCount > 0 {
                    StatusPill(text: "\(instance.activeRequestCount) active", tone: .accent)
                }
                if instance.queueDepth > 0 {
                    StatusPill(text: "\(instance.queueDepth) queued", tone: .warning)
                }
                Spacer(minLength: 0)
                Toggle("Allow Auto", isOn: allowsAutoBinding)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                Toggle("Keep Resident", isOn: keepsResidentBinding)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                Button("Use in Chat") {
                    store.useResidentModelInChat(instance.id)
                }
                .disabled(!instance.isReady && !instance.isBusy)
                Button(role: .destructive) {
                    Task { await store.stopResidentModelInstance(instance.id) }
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
            }

            HStack(spacing: 8) {
                Text("Instance \(instance.id)")
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let revision = instance.modelRevision {
                    Text("· \(revision)").lineLimit(1)
                }
                if !instance.selectedNodes.isEmpty {
                    Text("· \(instance.selectedNodes.joined(separator: " → "))")
                        .lineLimit(1)
                }
            }
            .font(.tokenityMono(10))
            .foregroundStyle(theme.tertiaryText)

            HStack(spacing: 8) {
                if let executionMode = instance.executionMode {
                    StatusPill(text: executionMode, tone: .neutral)
                }
                if let connectionMode = instance.connectionMode {
                    StatusPill(text: connectionMode, tone: .neutral)
                }
                ForEach(instance.capabilities, id: \.self) { capability in
                    StatusPill(text: capability, tone: .accent)
                }
                if instance.capabilities.isEmpty {
                    Text("Capabilities unavailable")
                        .foregroundStyle(theme.tertiaryText)
                }
            }
            .font(.tokenityText(10))
            .foregroundStyle(theme.secondaryText)

            HStack {
                Text(warmTTFTSummary)
                Spacer()
                Text("Reserved \(formatBytes(instance.reservedMemoryBytes)) · Actual \(formatBytes(instance.actualMemoryBytes))")
            }
            .font(.tokenityText(10))
            .foregroundStyle(theme.secondaryText)

            if let issue = instance.healthIssue, !issue.isEmpty {
                Label(issue, systemImage: "exclamationmark.triangle")
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.warning)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 16)
        }
    }

    private var allowsAutoBinding: Binding<Bool> {
        Binding(
            get: { instance.allowsAuto },
            set: { store.setResidentInstanceAllowsAuto(instance.id, allowed: $0) }
        )
    }

    private var keepsResidentBinding: Binding<Bool> {
        Binding(
            get: { instance.keepsResident },
            set: { store.setResidentInstanceKeepsResident(instance.id, keepsResident: $0) }
        )
    }

    private var stateTone: StatusPill.Tone {
        if instance.isReady { return .good }
        if instance.isBusy { return .warning }
        return .danger
    }

    private func formatBytes(_ bytes: Int64?) -> String {
        guard let bytes else { return "—" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }

    private func formatLatency(_ milliseconds: Double) -> String {
        if milliseconds >= 1_000 {
            return String(format: "%.2f s", milliseconds / 1_000)
        }
        return String(format: "%.0f ms", milliseconds)
    }

    private var warmTTFTSummary: String {
        switch (instance.warmTTFTP50Milliseconds, instance.warmTTFTP95Milliseconds) {
        case let (p50?, p95?):
            return "Profile warm TTFT p50 \(formatLatency(p50)) · p95 \(formatLatency(p95))"
        case let (p50?, nil):
            return "Profile warm TTFT p50 \(formatLatency(p50))"
        case let (nil, p95?):
            return "Profile warm TTFT p95 \(formatLatency(p95))"
        case (nil, nil):
            return "Profile warm TTFT unavailable"
        }
    }
}

private struct ModelLoadRow: View {
    let row: ModelLibraryRow
    let selectedNodeCount: Int
    let loadTargetSummary: String
    let clusterIsReady: Bool
    let loadEnabled: Bool
    let loadButtonHelp: String
    let loadingProgress: Double?
    let topologyIssue: String?
    let configurationAction: () -> Void
    let loadAction: () -> Void
    let loadOnOneMacAction: () -> Void
    let stopAction: () -> Void

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(stateColor)
                .frame(width: 9, height: 9)
                .accessibilityLabel(row.loadState.rawValue)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 8) {
                    Text(row.displayName)
                        .font(.tokenityMono(12, weight: .medium))
                        .lineLimit(1)
                        .help(row.displayName)
                    StatusPill(text: row.loadState.rawValue, tone: stateTone)
                    StatusPill(text: row.modality.rawValue, tone: row.modality == .video ? .accent : .neutral)
                }
                HStack(spacing: 6) {
                    if let format = row.format {
                        StatusPill(text: format, tone: .neutral)
                    }
                    if let quantization = row.quantization {
                        StatusPill(text: quantization, tone: .accent)
                    }
                    Text(row.sizeText)
                    if let architecture = row.architecture {
                        Text("· \(architecture)")
                            .lineLimit(1)
                    }
                    if let nativeMTP = row.nativeMTP {
                        StatusPill(
                            text: "MTP \(nativeMTP.displayStatus)",
                            tone: nativeMTP.status == "supported" ? .good : .neutral
                        )
                    }
                    if !row.standaloneLoadable {
                        StatusPill(text: "Draft only", tone: .warning)
                    }
                }
                .font(.tokenityText(11))
                .foregroundStyle(theme.secondaryText)
                Text(loadTargetSummary)
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.secondaryText)
                    .lineLimit(1)
                if row.loadState == .loading {
                    HStack(spacing: 8) {
                        if let loadingProgress {
                            ProgressView(value: loadingProgress, total: 1)
                            if loadingProgress >= 0.98 {
                                Text("Verifying inference...")
                            } else {
                                Text("\(Int((loadingProgress * 100).rounded()))%")
                                    .frame(width: 36, alignment: .trailing)
                            }
                        } else {
                            ProgressView()
                            Text("Preparing ranks...")
                        }
                    }
                    .font(.tokenityText(11, weight: .medium))
                    .foregroundStyle(theme.secondaryText)
                    .accessibilityLabel("Model loading progress")
                } else if row.loadState == .unloading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Releasing memory on all selected Macs...")
                    }
                    .font(.tokenityText(11, weight: .medium))
                    .foregroundStyle(theme.secondaryText)
                    .accessibilityLabel("Model unloading in progress")
                } else if let topologyIssue {
                    Label(topologyIssue, systemImage: "point.3.connected.trianglepath.dotted")
                        .font(.tokenityText(11, weight: .medium))
                        .foregroundStyle(theme.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer(minLength: 0)

            Button {
                configurationAction()
            } label: {
                Label(
                    row.modality == .video ? "Open Video" : "Configure",
                    systemImage: row.modality == .video ? "film" : "slider.horizontal.3"
                )
            }
            .help(
                row.modality == .video
                    ? "Open the MiniMax H3 generation workspace and runtime configuration"
                    : (row.loadState == .loaded ? "Changes take effect the next time this model is loaded" : "Configure model runtime and generation")
            )

            switch row.loadState {
            case .loaded:
                Button {
                    stopAction()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
            case .loading:
                Button {
                    stopAction()
                } label: {
                    Label("Cancel", systemImage: "xmark.circle.fill")
                }
                .help("Cancel loading and release model memory on every selected Mac")
            case .unloading:
                Button {
                } label: {
                    Label("Unloading", systemImage: "hourglass")
                }
                .disabled(true)
            case .notLoaded, .failed:
                if topologyIssue != nil {
                    Button {
                        loadOnOneMacAction()
                    } label: {
                        Label("Load on one Mac", systemImage: "laptopcomputer")
                    }
                    .disabled(!clusterIsReady || row.nodes.isEmpty || !row.standaloneLoadable)
                    .help("Switch this load to the supported single-Mac topology")
                } else {
                    Button {
                        loadAction()
                    } label: {
                        Label("Load", systemImage: "play.fill")
                    }
                    .disabled(!loadEnabled)
                    .help(loadButtonHelp)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 16)
        }
    }

    private var stateTone: StatusPill.Tone {
        switch row.loadState {
        case .loaded: return .good
        case .loading, .unloading: return .warning
        case .notLoaded, .failed: return .danger
        }
    }

    private var stateColor: Color {
        switch row.loadState {
        case .loaded: return theme.success
        case .loading, .unloading: return theme.warning
        case .notLoaded, .failed: return theme.danger
        }
    }
}

private struct ModelConfigurationSheet: View {
    let row: ModelLibraryRow
    let saveAction: (ModelRuntimeConfiguration) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var draft: ModelRuntimeConfiguration

    init(
        row: ModelLibraryRow,
        initialConfiguration: ModelRuntimeConfiguration,
        saveAction: @escaping (ModelRuntimeConfiguration) -> Void
    ) {
        self.row = row
        self.saveAction = saveAction
        _draft = State(initialValue: initialConfiguration)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    LabeledContent("Model") {
                        Text(row.displayName)
                            .textSelection(.enabled)
                    }
                    LabeledContent("Served as") {
                        Text(row.displayName)
                            .textSelection(.enabled)
                    }
                    LabeledContent("Path") {
                        Text(row.representativePath)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }

                Section("Generation") {
                    LabeledContent("Thinking Mode") {
                        Picker("Thinking Mode", selection: $draft.thinkingMode) {
                            ForEach(ModelThinkingMode.allCases) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(width: 260)
                    }
                    if row.usesQwen35Sampling {
                        Toggle("Use Qwen3.5 recommended sampling", isOn: $draft.useRecommendedSampling)
                        if draft.useRecommendedSampling {
                            Text(recommendedSamplingSummary)
                                .font(.tokenityText(11))
                                .foregroundStyle(.secondary)
                        }
                        Text("Adjusting any sampling field switches this model to custom sampling.")
                            .font(.tokenityText(11))
                            .foregroundStyle(.secondary)
                    }
                    integerField(
                        "Max Output Tokens",
                        value: $draft.maximumOutputTokens,
                        range: 1...262_144,
                        help: "Maximum reasoning and answer tokens for each chat completion."
                    )
                    Group {
                        decimalField("Temperature", value: customSamplingBinding(\.temperature), range: 0...2)
                        decimalField("Top P", value: customSamplingBinding(\.topP), range: 0...1)
                        integerField("Top K", value: customSamplingBinding(\.topK), range: 0...1_000)
                        decimalField("Min P", value: customSamplingBinding(\.minP), range: 0...1)
                        decimalField("Presence Penalty", value: customSamplingBinding(\.presencePenalty), range: -2...2)
                        decimalField("Repetition Penalty", value: customSamplingBinding(\.repetitionPenalty), range: 0...2)
                    }
                }

                Section("Distributed Runtime") {
                    integerField(
                        "Prefill Step Size",
                        value: $draft.prefillStepSize,
                        range: 128...8_192,
                        help: "Number of prompt tokens evaluated per prefill step."
                    )
                    integerField(
                        "Prompt Cache Entries",
                        value: $draft.promptCacheSize,
                        range: 1...64,
                        help: "Number of recent prompt caches retained by the model server."
                    )
                    integerField("Decode Concurrency", value: $draft.decodeConcurrency, range: 1...8)
                    integerField("Prompt Concurrency", value: $draft.promptConcurrency, range: 1...8)
                    Toggle("Trust tokenizer remote code", isOn: $draft.trustRemoteCode)
                }

                Section {
                    Text("Runtime changes take effect the next time the model is loaded. Generation settings are used for new chat requests.")
                        .font(.tokenityText(11))
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .navigationTitle("Configure \(row.displayName)")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        saveAction(draft.validated())
                        dismiss()
                    }
                    .keyboardShortcut(.defaultAction)
                }
            }
        }
        .frame(width: 620, height: 720)
    }

    private var recommendedSamplingSummary: String {
        let sampling = draft.resolvedSampling(forQwen35: true)
        return String(
            format: "Official %@ preset: temperature %.2g · top-p %.2g · top-k %d · presence %.2g · repetition %.2g",
            draft.thinkingMode == .disabled ? "non-thinking" : "thinking",
            sampling.temperature,
            sampling.topP,
            sampling.topK,
            sampling.presencePenalty,
            sampling.repetitionPenalty
        )
    }

    private func customSamplingBinding<Value>(
        _ keyPath: WritableKeyPath<ModelRuntimeConfiguration, Value>
    ) -> Binding<Value> {
        Binding(
            get: { draft[keyPath: keyPath] },
            set: { value in
                switchToCustomSamplingIfNeeded()
                draft[keyPath: keyPath] = value
            }
        )
    }

    private func switchToCustomSamplingIfNeeded() {
        guard row.usesQwen35Sampling, draft.useRecommendedSampling else { return }
        let recommended = draft.resolvedSampling(forQwen35: true)
        draft.temperature = recommended.temperature
        draft.topP = recommended.topP
        draft.topK = recommended.topK
        draft.minP = recommended.minP
        draft.presencePenalty = recommended.presencePenalty
        draft.repetitionPenalty = recommended.repetitionPenalty
        draft.useRecommendedSampling = false
    }

    @ViewBuilder
    private func integerField(
        _ label: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        help: String? = nil
    ) -> some View {
        LabeledContent(label) {
            HStack(spacing: 8) {
                if let help {
                    Image(systemName: "questionmark.circle")
                        .foregroundStyle(.secondary)
                        .help(help)
                }
                TextField("", value: value, format: .number)
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                Stepper("", value: value, in: range)
                    .labelsHidden()
            }
        }
    }

    @ViewBuilder
    private func decimalField(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>
    ) -> some View {
        LabeledContent(label) {
            HStack(spacing: 8) {
                Slider(value: value, in: range)
                    .frame(width: 220)
                TextField("", value: value, format: .number.precision(.fractionLength(0...3)))
                    .multilineTextAlignment(.trailing)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 90)
            }
        }
    }

}

struct APIAccessPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        PageScaffold(title: "API Access") {
            InfoGroup(title: "OpenAI-Compatible API") {
                InfoRow(label: "Status") {
                    HStack {
                        StatusPill(
                            text: store.isChatReady ? "Available" : "Load a model",
                            tone: store.isChatReady ? .good : .warning
                        )
                        Text(store.apiAccessStatus)
                            .foregroundStyle(theme.secondaryText)
                        Spacer()
                        Button("Test Connection") {
                            Task { await store.testExternalAPI() }
                        }
                    }
                }
                APIValueRow(label: "Base URL", value: store.openAIAPIBaseURL)
                APIValueRow(label: "Model", value: store.externalAPIModelName)
                InfoRow(label: "API key") {
                    Text("Not required. If a client requires one, enter tokenity-local.")
                        .foregroundStyle(theme.secondaryText)
                        .textSelection(.enabled)
                }
            }

            InfoGroup(title: "Supported Endpoints") {
                APIValueRow(label: "Models", value: "GET \(store.openAIAPIBaseURL)/models")
                APIValueRow(label: "Chat", value: "POST \(store.openAIAPIBaseURL)/chat/completions")
                APIValueRow(label: "Health", value: store.openAIAPIBaseURL.replacingOccurrences(of: "/v1", with: "/health"))
            }

            InfoGroup(title: "Client Setup") {
                InfoRow(label: "Cherry Studio / Msty") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Choose a custom OpenAI provider, paste the Base URL above, use the displayed model name, and enter tokenity-local only if the client requires an API key.")
                            .foregroundStyle(theme.secondaryText)
                        Text("Tokenity listens on the coordinator Mac over the local network. Keep that Mac and the model service running while external clients are connected.")
                            .font(.tokenityText(11))
                            .foregroundStyle(theme.tertiaryText)
                    }
                }
            }
        }
    }
}

private struct APIValueRow: View {
    let label: String
    let value: String
    @State private var didCopy = false
    @State private var copyResetTask: Task<Void, Never>?

    var body: some View {
        InfoRow(label: label) {
            HStack {
                Text(value)
                    .font(.tokenityMono(12))
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                    didCopy = true
                    copyResetTask?.cancel()
                    copyResetTask = Task {
                        try? await Task.sleep(for: .seconds(1.8))
                        guard !Task.isCancelled else { return }
                        didCopy = false
                    }
                } label: {
                    Label(didCopy ? "Copied" : "Copy", systemImage: didCopy ? "checkmark" : "doc.on.doc")
                }
                .help(didCopy ? "Copied to clipboard" : "Copy \(label)")
                .accessibilityLabel(didCopy ? "\(label) copied" : "Copy \(label)")
            }
        }
        .onDisappear {
            copyResetTask?.cancel()
        }
    }
}

struct LogsPage: View {
    @EnvironmentObject private var store: TokenityStore

    var body: some View {
        PageScaffold(title: "Logs") {
            InfoGroup(title: "Activity") {
                InfoRow(label: "Recent activity") {
                    OperationLog(lines: store.logs)
                        .frame(minHeight: 320)
                }
            }
        }
    }
}

struct SettingsPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        PageScaffold(title: "Settings") {
            InfoGroup(title: "Console") {
                InfoRow(label: "Discovery") {
                    Text("Local network and saved Macs")
                        .lineLimit(1)
                }
                InfoRow(label: "Model library") {
                    Text("Shared Tokenity model library")
                        .lineLimit(1)
                }
                InfoRow(label: "Access") {
                    Text(store.openAIAPIBaseURL)
                        .font(.tokenityMono(12))
                        .foregroundStyle(theme.text)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            InfoGroup(title: "Getting Started") {
                InfoRow(label: "Welcome guide") {
                    HStack {
                        Text("Review installation, cluster setup, model loading, and Chat.")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Show Guide") {
                            store.presentOnboarding()
                        }
                    }
                }
            }
        }
    }
}
