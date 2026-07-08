import SwiftUI

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
            case .chat: ChatPage()
            case .models: ModelsPage()
            case .network: NetworkPage()
            case .logs: LogsPage()
            case .settings: SettingsPage()
            }
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 920, idealWidth: 980, minHeight: 640, idealHeight: 700)
        .background(theme.window)
        .task {
            await store.startStatusRefreshLoop()
        }
    }
}

private struct SidebarView: View {
    @Binding var selection: AppSection?

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            sidebarGroup("Cluster", sections: AppSection.allCases.filter { $0.group == "Cluster" })
            sidebarGroup("Operations", sections: AppSection.allCases.filter { $0.group == "Operations" })
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(theme.sidebar)
    }

    private func sidebarGroup(_ title: String, sections: [AppSection]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.tokenityText(11, weight: .semibold))
                .foregroundStyle(theme.tertiaryText)
                .padding(.horizontal, 8)
            ForEach(sections) { section in
                Button {
                    selection = section
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: section.symbol)
                            .font(.system(size: 13, weight: .medium))
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
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .fill(selection == section ? theme.accent.opacity(0.13) : Color.clear)
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
                    Text(store.backendMode.rawValue).lineLimit(1)
                }
                InfoRow(label: "Connection") {
                    Text(store.connectionMode.rawValue)
                }
            }

            InfoGroup(title: "Selected Macs") {
                ForEach(store.selectedNodes) { node in
                    InfoRow(label: node.displayName) {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                StatusPill(text: node.isOnline ? "online" : "selected", tone: node.isOnline ? .good : .neutral)
                                Text(node.identityDetail)
                                    .lineLimit(1)
                                Spacer()
                                Text(node.rdma.rdmaEnabled ? "Thunderbolt ready" : "Standard network")
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
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
                            Text(issue).foregroundStyle(.orange)
                        }
                    }
                }
            }
        }
    }

    private func tone(for phase: ClusterPhase) -> StatusPill.Tone {
        switch phase {
        case .running: return .good
        case .failed: return .danger
        case .launching, .distributedInit, .loadingModel, .compiling, .firstTokenPending, .stopping: return .warning
        case .stopped: return .neutral
        }
    }
}

struct ClusterPage: View {
    @EnvironmentObject private var store: TokenityStore

    var body: some View {
        PageScaffold(title: "Cluster") {
            InfoGroup(title: "Cluster Builder") {
                ClusterNodeCanvas(
                    nodes: store.nodes,
                    selectedNodeIDs: store.selectedNodeIDs,
                    canEdit: store.canEditCluster,
                    onToggle: store.toggleNodeSelection
                )
            }

            InfoGroup(title: "Cluster Setup") {
                InfoRow(label: "Backend") {
                    Picker("", selection: $store.backendMode) {
                        ForEach(BackendMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.menu)
                }
                InfoRow(label: "Connection") {
                    Picker("", selection: $store.connectionMode) {
                        ForEach(ConnectionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 360)
                }
                InfoRow(label: "Actions") {
                    HStack {
                        Button {
                            store.phase == .running ? store.restart() : store.createCluster()
                        } label: {
                            Label(store.phase == .running ? "Recreate Cluster" : "Create Cluster", systemImage: store.phase == .running ? "arrow.clockwise" : "play.fill")
                        }
                        .disabled(!store.launchPreview.readinessIssues.isEmpty || store.selectedNodes.isEmpty)

                        Button {
                            store.stop()
                        } label: {
                            Label("Stop Cluster", systemImage: "stop.fill")
                        }
                        .disabled(store.phase == .stopped)

                        Button {
                            Task { await store.refreshLocalAgent() }
                        } label: {
                            Label("Refresh This Mac", systemImage: "arrow.clockwise")
                        }
                    }
                }
            }

            if !store.launchPreview.readinessIssues.isEmpty {
                InfoGroup(title: "Readiness") {
                    ForEach(store.launchPreview.readinessIssues, id: \.self) { issue in
                        InfoRow(label: "Needs attention") {
                            Text(issue).foregroundStyle(.orange)
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
}

private struct ClusterNodeCanvas: View {
    let nodes: [TokenityNode]
    let selectedNodeIDs: Set<String>
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
        .frame(minHeight: 340, idealHeight: 360)
        .padding(12)
        .background(theme.window.opacity(0.55))
    }

    private func drawLinks(in context: inout GraphicsContext, size: CGSize) {
        let selected = nodes.enumerated()
            .filter { selectedNodeIDs.contains($0.element.id) }
            .map { point(for: $0.offset, count: nodes.count, in: size) }

        guard selected.count > 1 else { return }

        for startIndex in selected.indices {
            for endIndex in selected.indices where endIndex > startIndex {
                var path = Path()
                path.move(to: selected[startIndex])
                path.addLine(to: selected[endIndex])
                context.stroke(
                    path,
                    with: .color(theme.accent.opacity(0.38)),
                    style: StrokeStyle(lineWidth: 1.5, lineCap: .round, dash: [7, 7])
                )
            }
        }
    }

    private func point(for index: Int, count: Int, in size: CGSize) -> CGPoint {
        let points: [UnitPoint]
        switch count {
        case 1:
            points = [UnitPoint(x: 0.5, y: 0.5)]
        case 2:
            points = [UnitPoint(x: 0.33, y: 0.5), UnitPoint(x: 0.67, y: 0.5)]
        case 3:
            points = [UnitPoint(x: 0.5, y: 0.22), UnitPoint(x: 0.25, y: 0.68), UnitPoint(x: 0.75, y: 0.68)]
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
    let canEdit: Bool

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 8) {
                Image(systemName: "laptopcomputer")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                VStack(alignment: .leading, spacing: 1) {
                    Text(node.displayName)
                        .font(.tokenityText(13, weight: .semibold))
                        .lineLimit(1)
                    Text(node.identityDetail)
                        .font(.tokenityText(10))
                        .foregroundStyle(theme.tertiaryText)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                StatusPill(text: isSelected ? "Selected" : "Available", tone: isSelected ? .good : .neutral)
            }

            VStack(alignment: .leading, spacing: 5) {
                nodeSignal(label: "Runtime", value: node.displayRuntime)
                nodeSignal(label: "IP", value: node.primaryIP)
                nodeSignal(label: "Memory", value: node.memoryPercentText)
                MemoryUsageBar(memory: node.memory)
            }
        }
        .padding(12)
        .frame(width: 228, height: 146, alignment: .leading)
        .background(theme.group, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(isSelected ? theme.accent.opacity(0.75) : theme.border.opacity(0.75), lineWidth: isSelected ? 1.4 : 0.7)
        )
        .opacity(canEdit || isSelected ? 1 : 0.76)
        .shadow(color: Color.black.opacity(isSelected ? 0.08 : 0.03), radius: isSelected ? 8 : 4, y: 3)
    }

    private func nodeSignal(label: String, value: String) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.tokenityText(11, weight: .medium))
                .foregroundStyle(theme.tertiaryText)
                .frame(width: 44, alignment: .leading)
            Text(value)
                .font(.tokenityText(11))
                .foregroundStyle(theme.secondaryText)
                .lineLimit(1)
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
                                Text(node.identityDetail)
                                    .lineLimit(1)
                                Text(node.ssh)
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

struct ChatPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            chatHeader
            Divider()
            ScrollView {
                LazyVStack(spacing: 14) {
                    ForEach(store.chatMessages) { message in
                        ChatBubble(message: message)
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .scrollIndicators(.visible)
            .contentShape(Rectangle())
            .background(theme.window)
            Divider()
            composer
                .padding(16)
                .background(theme.window)
        }
        .frame(minWidth: 700, minHeight: 620)
    }

    private var chatHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Chat")
                        .font(.tokenityText(20, weight: .semibold))
                    Text("Measure response latency after the cluster is created and a model is loaded.")
                        .font(.tokenityText(12))
                        .foregroundStyle(theme.secondaryText)
                }
                Spacer()
                StatusPill(text: chatStatusText, tone: chatStatusTone)
            }

            HStack(spacing: 10) {
                metricTile("First response", secondsText(store.chatMetrics.firstTokenSeconds))
                metricTile("Total time", secondsText(store.chatMetrics.totalSeconds))
                metricTile("Speed", speedText(store.chatMetrics.outputTokensPerSecond))
                Spacer()
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(theme.window)
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(store.isChatReady ? "Ask the loaded model..." : "Create a cluster and load a model first", text: $store.chatInput, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
                .onSubmit {
                    Task { await store.sendChatMessage() }
                }
            Button {
                Task { await store.sendChatMessage() }
            } label: {
                Label(store.isChatRunning ? "Sending" : "Send", systemImage: "paperplane.fill")
            }
            .disabled(store.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || store.isChatRunning || !store.isChatReady)
        }
    }

    private var chatStatusText: String {
        if store.isChatReady { return "Ready" }
        if store.phase == .running { return "Load a model" }
        return "Create cluster"
    }

    private var chatStatusTone: StatusPill.Tone {
        store.isChatReady ? .good : .warning
    }

    private func metricTile(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.tokenityText(11, weight: .semibold))
                .foregroundStyle(theme.tertiaryText)
            Text(value)
                .font(.tokenityText(14, weight: .semibold))
                .foregroundStyle(theme.text)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(minWidth: 120, alignment: .leading)
        .background(theme.group, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.border.opacity(0.8), lineWidth: 0.5)
        )
    }

    private func secondsText(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.2fs", value)
    }

    private func speedText(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.1f tok/s", value)
    }
}

private struct ChatBubble: View {
    let message: ChatMessage

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 80)
                bubble
            } else {
                bubble
                Spacer(minLength: 80)
            }
        }
    }

    private var bubble: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.role == .user ? "You" : "Tokenity")
                    .font(.tokenityText(11, weight: .semibold))
                    .foregroundStyle(message.role == .user ? Color.white.opacity(0.9) : theme.secondaryText)
                Spacer()
            }

            if message.role == .assistant, !message.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                DisclosureGroup {
                    Text(message.thinking)
                        .font(.tokenityText(12))
                        .foregroundStyle(theme.secondaryText)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                } label: {
                    Label("Thinking", systemImage: "brain.head.profile")
                        .font(.tokenityText(12, weight: .medium))
                        .foregroundStyle(theme.secondaryText)
                }
                .padding(10)
                .background(theme.group.opacity(0.75), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }

            Text(message.content.isEmpty ? "..." : message.content)
                .font(.tokenityText(13))
                .foregroundStyle(message.role == .user ? Color.white : theme.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(maxWidth: 580, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(message.role == .user ? theme.accent : theme.group)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(message.role == .user ? Color.clear : theme.border.opacity(0.8), lineWidth: 0.5)
        )
    }
}

struct ModelsPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

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
                        StatusPill(text: store.phase == .running ? "Created" : "Create first", tone: store.phase == .running ? .good : .warning)
                        Text(store.phase == .running ? "Models can be loaded now" : "Create a cluster before loading a model")
                            .foregroundStyle(theme.secondaryText)
                            .lineLimit(1)
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

            InfoGroup(title: "Available Models") {
                if store.modelLibraryRows.isEmpty {
                    InfoRow(label: "Status") {
                        Text("No model inventory yet. Scan selected Macs to load the list.")
                            .foregroundStyle(theme.secondaryText)
                    }
                } else {
                    ForEach(store.modelLibraryRows) { row in
                        ModelLoadRow(
                            row: row,
                            selectedNodeCount: store.selectedNodes.count,
                            clusterIsReady: store.phase == .running,
                            loadAction: { Task { await store.loadModel(row) } },
                            stopAction: { Task { await store.stopModel(row) } }
                        )
                    }
                }
            }
        }
    }
}

private struct ModelLoadRow: View {
    let row: ModelLibraryRow
    let selectedNodeCount: Int
    let clusterIsReady: Bool
    let loadAction: () -> Void
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
                        .font(.tokenityText(13, weight: .medium))
                        .lineLimit(1)
                    StatusPill(text: row.loadState.rawValue, tone: stateTone)
                }
                Text("\(row.availability) · \(row.nodes.joined(separator: ", "))")
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.secondaryText)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)

            switch row.loadState {
            case .loaded:
                Button {
                    stopAction()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
            case .loading:
                Button {
                } label: {
                    Label("Loading", systemImage: "hourglass")
                }
                .disabled(true)
            case .notLoaded:
                Button {
                    loadAction()
                } label: {
                    Label("Load", systemImage: "play.fill")
                }
                .disabled(!clusterIsReady || row.nodes.count < selectedNodeCount)
                .help(!clusterIsReady ? "Create a cluster before loading a model" : "Load model")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 12)
        }
    }

    private var stateTone: StatusPill.Tone {
        switch row.loadState {
        case .loaded: return .good
        case .loading: return .warning
        case .notLoaded: return .danger
        }
    }

    private var stateColor: Color {
        switch row.loadState {
        case .loaded: return theme.success
        case .loading: return theme.warning
        case .notLoaded: return theme.danger
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
                    Text("Available after the cluster starts")
                        .lineLimit(1)
                }
            }

            InfoGroup(title: "Backend Status") {
                InfoRow(label: "Official") {
                    StatusPill(text: "Experimental", tone: .warning)
                }
                InfoRow(label: "Tokenity") {
                    StatusPill(text: "Stable target", tone: .accent)
                }
            }
        }
    }
}
