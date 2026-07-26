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
        VStack(alignment: .leading, spacing: 18) {
            sidebarBrand

            Rectangle()
                .fill(theme.border.opacity(0.72))
                .frame(height: 0.5)
                .padding(.horizontal, 8)

            sidebarGroup("Cluster", sections: AppSection.allCases.filter { $0.group == "Cluster" })
            sidebarGroup("Operations", sections: AppSection.allCases.filter { $0.group == "Operations" })
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.top, 20)
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(theme.sidebar)
    }

    private var sidebarBrand: some View {
        HStack(spacing: 10) {
            TokenityBrandMark()
                .frame(width: 21, height: 21)
                .foregroundStyle(theme.accent)

            VStack(alignment: .leading, spacing: 1) {
                Text("Tokenity")
                    .font(.tokenityText(16, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text("Distributed AI")
                    .font(.tokenityText(9, weight: .medium))
                    .foregroundStyle(theme.tertiaryText)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
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
    @Environment(\.tokenityTheme) private var theme
    @State private var showsAdvancedSetup = false

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
                InfoRow(label: "Runtime") {
                    HStack(spacing: 10) {
                        Label(store.backendMode.shortName, systemImage: "server.rack")
                            .lineLimit(1)
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(theme.tertiaryText)
                        Label(store.connectionMode.shortName, systemImage: connectionSymbol)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                    }
                }
                InfoRow(label: "Actions") {
                    HStack(spacing: 10) {
                        Button {
                            store.phase == .running ? store.restart() : store.createCluster()
                        } label: {
                            Label(store.phase == .running ? "Restart Cluster" : "Create Cluster", systemImage: store.phase == .running ? "arrow.clockwise" : "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .tint(theme.accent)
                        .disabled(!store.launchPreview.readinessIssues.isEmpty || store.selectedNodes.isEmpty)

                        Button {
                            store.stop()
                        } label: {
                            Label("Stop Cluster", systemImage: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .tint(theme.danger)
                        .disabled(store.phase == .stopped)

                        Button {
                            Task { await store.refreshLocalAgent() }
                        } label: {
                            Label("Refresh Status", systemImage: "arrow.triangle.2.circlepath")
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(theme.secondaryText)
                        .help("Refresh node status")
                        Spacer(minLength: 0)
                    }
                }
                InfoRow(label: "Advanced") {
                    DisclosureGroup(isExpanded: $showsAdvancedSetup) {
                        VStack(alignment: .leading, spacing: 14) {
                            advancedPicker(
                                title: "Inference backend",
                                detail: store.backendMode.detail
                            ) {
                                Picker("Inference backend", selection: $store.backendMode) {
                                    ForEach(BackendMode.userSelectableCases) { mode in
                                        Text(mode.shortName).tag(mode)
                                    }
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                            }

                            advancedPicker(
                                title: "Mac-to-Mac connection",
                                detail: store.connectionMode.detail
                            ) {
                                Picker("Mac-to-Mac connection", selection: $store.connectionMode) {
                                    ForEach(ConnectionMode.allCases) { mode in
                                        Text(mode.shortName).tag(mode)
                                    }
                                }
                                .labelsHidden()
                                .pickerStyle(.menu)
                            }
                        }
                        .padding(.top, 12)
                    } label: {
                        Text("Backend and connection options")
                            .foregroundStyle(theme.secondaryText)
                    }
                }
            }

            InfoGroup(title: "Inference Acceleration") {
                InfoRow(label: "Native MTP") {
                    HStack(spacing: 10) {
                        Picker("Native MTP", selection: $store.nativeMTPMode) {
                            ForEach(NativeMTPMode.allCases) { mode in
                                Text(mode.title).tag(mode)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 300)
                        .disabled(!store.canEditNativeMTP)
                        Spacer(minLength: 0)
                        if !store.canEditNativeMTP {
                            Text(store.backendMode == .distributed ? "Stop cluster to edit" : "Distributed backend only")
                                .font(.tokenityText(11))
                                .foregroundStyle(theme.tertiaryText)
                        }
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
                InfoRow(label: "MVP boundary") {
                    Text("Depth 1 · Replicated head · Singleton decode")
                        .foregroundStyle(theme.secondaryText)
                }
                if store.nativeMTPMode == .auto {
                    InfoRow(label: "Auto fallback") {
                        Text("The server uses standard decoding if model weights, runtime shape, or topology are incompatible.")
                            .foregroundStyle(theme.warning)
                    }
                }
                if let runtime = store.nativeMTPRuntime {
                    InfoRow(label: "Runtime") {
                        HStack(spacing: 8) {
                            StatusPill(text: runtime.enabled ? "Enabled" : "Standard decode", tone: runtime.enabled ? .good : .neutral)
                            Text("Requested \(runtime.requestedMode) · Effective \(runtime.effectiveMode ?? (runtime.enabled ? "native_mtp" : "standard"))")
                                .foregroundStyle(theme.secondaryText)
                        }
                    }
                    if let reason = runtime.fallbackReason {
                        InfoRow(label: "Fallback reason") {
                            Text("\(reason) · \(runtime.message ?? "Server declined Native MTP")")
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

    private var connectionSymbol: String {
        switch store.connectionMode {
        case .ring: return "network"
        case .jaccl: return "bolt.horizontal.fill"
        case .jacclRing: return "arrow.triangle.branch"
        }
    }

    private func advancedPicker<PickerContent: View>(
        title: String,
        detail: String,
        @ViewBuilder picker: () -> PickerContent
    ) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(title)
                    .font(.tokenityText(12, weight: .medium))
                Spacer(minLength: 12)
                picker()
            }
            Text(detail)
                .font(.tokenityText(11))
                .foregroundStyle(theme.tertiaryText)
                .fixedSize(horizontal: false, vertical: true)
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
        .frame(width: 228, height: 162, alignment: .leading)
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
                                Text(node.agentURL)
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

            InfoGroup(title: "Resident Model Pool") {
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
                            clusterIsReady: store.phase == .running && !store.isModelTransitioning,
                            loadingProgress: row.loadState == .loading ? store.modelLoadProgress : nil,
                            configurationAction: { configurationTarget = row },
                            loadAction: { store.beginLoadingModel(row) },
                            stopAction: { Task { await store.stopModel(row) } }
                        )
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
            if store.modelScanSummary == "Not scanned" {
                await store.scanModels()
            }
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
                    .font(.tokenityText(13, weight: .semibold))
                    .lineLimit(1)
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
            .font(.tokenityText(10))
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
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 12)
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
    let clusterIsReady: Bool
    let loadingProgress: Double?
    let configurationAction: () -> Void
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
                Text("\(row.availability) · \(row.nodes.joined(separator: ", "))")
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
                }
            }

            Spacer(minLength: 0)

            Button {
                configurationAction()
            } label: {
                Label("Configure", systemImage: "slider.horizontal.3")
            }
            .help(row.loadState == .loaded ? "Changes take effect the next time this model is loaded" : "Configure model runtime and generation")

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
                    Label("Stop", systemImage: "stop.fill")
                }
                .help("Cancel loading and release model memory on every selected Mac")
            case .unloading:
                Button {
                } label: {
                    Label("Unloading", systemImage: "hourglass")
                }
                .disabled(true)
            case .notLoaded, .failed:
                Button {
                    loadAction()
                } label: {
                    Label("Load", systemImage: "play.fill")
                }
                .disabled(!clusterIsReady || row.nodes.count < selectedNodeCount || !row.standaloneLoadable)
                .help(loadButtonHelp)
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
        case .loading, .unloading: return .warning
        case .notLoaded, .failed: return .danger
        }
    }

    private var loadButtonHelp: String {
        if !row.standaloneLoadable {
            return row.loadBlockReason ?? "This draft checkpoint cannot be loaded as a standalone chat model."
        }
        if !clusterIsReady { return "Create a cluster before loading a model" }
        if row.nodes.count < selectedNodeCount { return "The model must be available on every selected Mac" }
        return "Load model"
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
                    if row.isQwen35 {
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
        guard row.isQwen35, draft.useRecommendedSampling else { return }
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

    var body: some View {
        InfoRow(label: label) {
            HStack {
                Text(value)
                    .font(.system(.body, design: .monospaced))
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
            }
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
                    Text(store.openAIAPIBaseURL)
                        .lineLimit(1)
                        .truncationMode(.middle)
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
