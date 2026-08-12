import AppKit
import SwiftUI

enum ChatWorkspaceMetrics {
    static let headerHeight: CGFloat = 100
}

extension TokenityStore {
    var autoChatEligibleModelIDs: [String] {
        var modelIDs = Set(
            residentModelInstances
                .filter { $0.allowsAuto && ($0.isReady || $0.isBusy) }
                .map(\.modelID)
        )
        if modelIDs.isEmpty,
           residentModelInstances.isEmpty,
           let loadedModelName {
            modelIDs.insert(loadedModelName)
        }
        return modelIDs.sorted {
            $0.localizedCaseInsensitiveCompare($1) == .orderedAscending
        }
    }

    var chatRoutingDisplayTitle: String {
        guard isAutoChatSelection else { return chatSelectedModelID }
        let count = autoChatEligibleModelIDs.count
        if count == 0 { return "Free routing · waiting" }
        return "Free routing · \(count) \(count == 1 ? "model" : "models")"
    }

    var chatRoutingDisplayHelp: String {
        guard isAutoChatSelection else {
            return "Messages are pinned to \(chatSelectedModelID)."
        }
        guard !autoChatEligibleModelIDs.isEmpty else {
            return "Free routing is waiting for an eligible resident model."
        }
        return "Free routing can select: \(autoChatEligibleModelIDs.joined(separator: ", "))."
    }

    var chatTopologyLabel: String {
        if let instance = chatTopologyInstance {
            let nodeCount = instance.selectedNodes.isEmpty
                ? (instance.executionMode?.lowercased() == "single" ? 1 : selectedNodes.count)
                : instance.selectedNodes.count
            if nodeCount <= 1 || instance.executionMode?.lowercased() == "single" {
                return "1 Mac · Single"
            }
            let mode = instance.connectionMode?
                .replacingOccurrences(of: "jaccl-ring", with: "Thunderbolt RDMA")
                .replacingOccurrences(of: "ring", with: "Standard Network")
                ?? connectionMode.shortName
            return "\(nodeCount) Macs · \(mode)"
        }
        if backendMode == .singleNode { return "1 Mac · Single" }
        return selectedNodes.count > 1
            ? "\(selectedNodes.count) Macs · \(connectionMode.shortName)"
            : "1 Mac · Single"
    }

    var chatUsesMultipleNodes: Bool {
        guard let instance = chatTopologyInstance else {
            return backendMode != .singleNode && selectedNodes.count > 1
        }
        return instance.executionMode?.lowercased() != "single"
            && instance.selectedNodes.count > 1
    }

    private var chatTopologyInstance: ResidentModelInstanceSummary? {
        let requestedModelID = isAutoChatSelection ? loadedModelName : chatSelectedModelID
        return residentModelInstances.first { instance in
            requestedModelID == nil || instance.modelID == requestedModelID
        }
    }
}

struct ChatWorkspaceView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @Environment(\.displayScale) private var displayScale
    @State private var showsHistory = true

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                ChatWorkspaceHeader(showsHistory: $showsHistory)
                    .frame(height: ChatWorkspaceMetrics.headerHeight)
                ChatTranscriptView()
                ChatComposerView()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(theme.window)

            if showsHistory {
                ChatSidebarView()
                    .frame(minWidth: 244, idealWidth: 278, maxWidth: 326)
                    .clipped()
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .background(theme.window)
        .frame(minWidth: 780, minHeight: 620)
        .animation(.easeInOut(duration: 0.18), value: showsHistory)
        .overlay(alignment: .top) {
            VStack(spacing: 0) {
                Color.clear
                    .frame(height: ChatWorkspaceMetrics.headerHeight)
                Rectangle()
                    .fill(theme.border.opacity(0.55))
                    .frame(height: hairlineWidth)
                Spacer(minLength: 0)
            }
            .allowsHitTesting(false)
        }
    }

    private var hairlineWidth: CGFloat {
        1 / max(displayScale, 1)
    }
}

private struct ChatWorkspaceHeader: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @Binding var showsHistory: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 12) {
                Text(currentTitle)
                    .font(.tokenityTitle(25))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 12)
                StatusPill(text: generationLabel, tone: generationTone)
                Button {
                    store.newChatSession()
                } label: {
                    Image(systemName: "square.and.pencil")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(TokenityIconButtonStyle())
                .disabled(store.isChatRunning)
                .keyboardShortcut("n", modifiers: [.command])
                .help(store.isChatRunning ? "Stop generation before starting a new conversation" : "New conversation")
                .accessibilityLabel("New conversation")

                Button {
                    showsHistory.toggle()
                } label: {
                    Image(systemName: "sidebar.trailing")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(TokenityIconButtonStyle())
                .help(showsHistory ? "Hide conversation history" : "Show conversation history")
                .accessibilityLabel(showsHistory ? "Hide conversation history" : "Show conversation history")
            }

            HStack(spacing: 9) {
                Label(
                    store.chatRoutingDisplayTitle,
                    systemImage: store.isAutoChatSelection ? "arrow.triangle.branch" : "cube"
                )
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(theme.group, in: Capsule())
                    .help(store.chatRoutingDisplayHelp)
                    .accessibilityLabel(store.chatRoutingDisplayTitle)

                Label(
                    store.chatTopologyLabel,
                    systemImage: store.chatUsesMultipleNodes
                        ? "point.3.connected.trianglepath.dotted"
                        : "desktopcomputer"
                )
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(theme.group, in: Capsule())
                .accessibilityLabel(store.chatTopologyLabel)

                if store.isAutoChatSelection {
                    Label(store.autoRouterHealthText, systemImage: "arrow.triangle.branch")
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
                Spacer(minLength: 0)
            }
            .font(.tokenityText(11))
            .foregroundStyle(theme.secondaryText)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(theme.window)
    }

    private var currentTitle: String {
        store.chatSessions.first(where: { $0.id == store.activeChatSessionID })?.title ?? "New Chat"
    }

    private var generationLabel: String {
        if case .selecting = store.chatRoutingState { return "Routing" }
        if case .routed(let route) = store.chatRoutingState,
           !store.isChatRunning,
           let model = route.routedModelID {
            return model
        }
        if store.isChatRunning { return "Generating" }
        if store.isChatReady { return "Ready" }
        if store.phase == .running { return "Load a model" }
        return "Cluster stopped"
    }

    private var generationTone: StatusPill.Tone {
        if store.isChatRunning { return .accent }
        return store.isChatReady ? .good : .warning
    }
}

struct ChatTranscriptView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var followState = ChatTranscriptFollowState()
    private let bottomID = "tokenity-modern-chat-bottom"

    var body: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 28) {
                            if store.chatMessages.isEmpty {
                                ChatEmptyState()
                                    .frame(maxWidth: .infinity)
                                    .padding(.top, max(70, viewport.size.height * 0.16))
                            } else {
                                ForEach(store.chatMessages) { message in
                                    ChatMessageView(message: message)
                                        .id(message.id)
                                }
                            }
                            Color.clear
                                .frame(height: 1)
                                .id(bottomID)
                        }
                        .id(store.activeChatSessionID)
                        .frame(maxWidth: 920, alignment: .leading)
                        .padding(.horizontal, 32)
                        .padding(.vertical, 30)
                        .frame(maxWidth: .infinity)
                    }
                    .scrollIndicators(.visible)
                    .background(Color.clear)
                    .overlay(alignment: .topLeading) {
                        TranscriptScrollPositionObserver(
                            currentFollowsLatest: followState.followsLatest
                        ) { followsLatest in
                            guard followState.followsLatest != followsLatest else {
                                return
                            }
                            if followsLatest {
                                followState.resume()
                            } else {
                                followState.userDidScroll()
                            }
                        }
                        .frame(width: 1, height: 1)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                    }
                    .onAppear {
                        proxy.scrollTo(bottomID, anchor: .bottom)
                    }
                    .onChange(of: store.activeChatSessionID) { _, _ in
                        followState.resume()
                        Task { @MainActor in
                            proxy.scrollTo(bottomID, anchor: .bottom)
                        }
                    }
                    .onChange(of: store.isChatRunning) { _, isRunning in
                        guard isRunning else { return }
                        followState.resume()
                        Task { @MainActor in
                            proxy.scrollTo(bottomID, anchor: .bottom)
                        }
                    }
                    .onChange(of: store.chatScrollRevision) { _, _ in
                        guard followState.followsLatest else { return }
                        proxy.scrollTo(bottomID, anchor: .bottom)
                    }

                    if !followState.followsLatest {
                        Button {
                            followState.resume()
                            withAnimation(.easeOut(duration: 0.18)) {
                                proxy.scrollTo(bottomID, anchor: .bottom)
                            }
                        } label: {
                            Label("Latest", systemImage: "arrow.down")
                                .font(.tokenityText(12, weight: .medium))
                        }
                        .buttonStyle(.bordered)
                        .background(theme.surface, in: Capsule())
                        .overlay(Capsule().stroke(theme.border, lineWidth: 0.6))
                        .padding(18)
                        .accessibilityLabel("Return to latest message")
                    }
                }
            }
        }
    }
}

private struct TranscriptScrollPositionObserver: NSViewRepresentable {
    let currentFollowsLatest: Bool
    let onFollowChange: (Bool) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(
            currentFollowsLatest: currentFollowsLatest,
            onFollowChange: onFollowChange
        )
    }

    func makeNSView(context: Context) -> ObserverView {
        let view = ObserverView()
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ nsView: ObserverView, context: Context) {
        context.coordinator.onFollowChange = onFollowChange
        nsView.coordinator = context.coordinator
        context.coordinator.attachIfPossible(from: nsView)
        context.coordinator.synchronizeFollowState(currentFollowsLatest)
    }

    static func dismantleNSView(_ nsView: ObserverView, coordinator: Coordinator) {
        nsView.coordinator = nil
        coordinator.detach()
    }

    final class ObserverView: NSView {
        weak var coordinator: Coordinator?

        override func viewDidMoveToSuperview() {
            super.viewDidMoveToSuperview()
            scheduleAttachment()
        }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            scheduleAttachment()
        }

        private func scheduleAttachment() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.coordinator?.attachIfPossible(from: self)
            }
        }
    }

    final class Coordinator {
        var onFollowChange: (Bool) -> Void
        private weak var scrollView: NSScrollView?
        private var observers: [NSObjectProtocol] = []
        private var isBracketedLiveScroll = false
        private var latestNearBottom = true
        private var transitionGate = TranscriptFollowTransitionGate()
        private var pendingFollowValue: Bool?
        private var emissionScheduled = false
        private var attachmentGeneration = 0
        private var synchronizedFollowsLatest: Bool

        init(
            currentFollowsLatest: Bool,
            onFollowChange: @escaping (Bool) -> Void
        ) {
            self.onFollowChange = onFollowChange
            synchronizedFollowsLatest = currentFollowsLatest
            transitionGate.synchronize(with: currentFollowsLatest)
        }

        deinit {
            detach()
        }

        func synchronizeFollowState(_ followsLatest: Bool) {
            synchronizedFollowsLatest = followsLatest
            transitionGate.synchronize(with: followsLatest)
        }

        func attachIfPossible(from view: NSView) {
            if let scrollView, scrollView.window === view.window {
                return
            }
            guard let scrollView = transcriptScrollView(near: view),
                  self.scrollView !== scrollView else { return }
            detach()
            self.scrollView = scrollView
            transitionGate.synchronize(with: synchronizedFollowsLatest)
            refreshNearBottom()

            let center = NotificationCenter.default
            observers.append(
                center.addObserver(
                    forName: NSScrollView.willStartLiveScrollNotification,
                    object: scrollView,
                    queue: .main
                ) { [weak self] _ in
                    self?.liveScrollWillStart()
                }
            )
            observers.append(
                center.addObserver(
                    forName: NSScrollView.didLiveScrollNotification,
                    object: scrollView,
                    queue: .main
                ) { [weak self] _ in
                    self?.liveScrollDidMove()
                }
            )
            observers.append(
                center.addObserver(
                    forName: NSScrollView.didEndLiveScrollNotification,
                    object: scrollView,
                    queue: .main
                ) { [weak self] _ in
                    self?.liveScrollDidEnd()
                }
            )
        }

        private func transcriptScrollView(near view: NSView) -> NSScrollView? {
            guard let root = view.window?.contentView else { return nil }
            let pointInWindow = view.convert(
                CGPoint(x: view.bounds.midX, y: view.bounds.midY),
                to: nil
            )
            return scrollViews(in: root)
                .filter { scrollView in
                    guard scrollView.hasVerticalScroller,
                          let verticalScroller = scrollView.verticalScroller,
                          !verticalScroller.isHidden
                    else { return false }
                    let frameInWindow = scrollView.convert(scrollView.bounds, to: nil)
                    return frameInWindow.contains(pointInWindow)
                }
                .max { lhs, rhs in
                    lhs.bounds.width * lhs.bounds.height < rhs.bounds.width * rhs.bounds.height
                }
        }

        private func scrollViews(in view: NSView) -> [NSScrollView] {
            var result: [NSScrollView] = []
            if let scrollView = view as? NSScrollView {
                result.append(scrollView)
            }
            for subview in view.subviews {
                result.append(contentsOf: scrollViews(in: subview))
            }
            return result
        }

        func detach() {
            attachmentGeneration += 1
            let center = NotificationCenter.default
            observers.forEach(center.removeObserver)
            observers.removeAll()
            scrollView = nil
            isBracketedLiveScroll = false
            latestNearBottom = true
            transitionGate.reset()
            pendingFollowValue = nil
            emissionScheduled = false
        }

        private func liveScrollWillStart() {
            isBracketedLiveScroll = true
            refreshNearBottom()
            scheduleFollowChange(false)
        }

        private func liveScrollDidMove() {
            refreshNearBottom()

            // Modern trackpad and scroller gestures are bracketed by start/end
            // notifications. Keep position local during those high-frequency
            // updates. Legacy mouse wheels may only emit didLiveScroll, so
            // report their threshold crossing through the same transition gate.
            if !isBracketedLiveScroll {
                scheduleFollowChange(latestNearBottom)
            }
        }

        private func liveScrollDidEnd() {
            refreshNearBottom()
            isBracketedLiveScroll = false
            scheduleFollowChange(latestNearBottom)
        }

        private func refreshNearBottom() {
            guard let documentView = scrollView?.documentView else {
                latestNearBottom = true
                return
            }
            latestNearBottom = TranscriptScrollGeometry.isNearBottom(
                documentBounds: documentView.bounds,
                visibleRect: documentView.visibleRect,
                isFlipped: documentView.isFlipped
            )
        }

        private func scheduleFollowChange(_ followsLatest: Bool) {
            pendingFollowValue = followsLatest
            guard !emissionScheduled else { return }
            emissionScheduled = true
            let generation = attachmentGeneration

            DispatchQueue.main.async { [weak self] in
                guard let self, self.attachmentGeneration == generation else { return }
                self.emissionScheduled = false
                guard let pendingFollowValue = self.pendingFollowValue else { return }
                self.pendingFollowValue = nil
                guard let value = self.transitionGate.valueToEmit(for: pendingFollowValue) else {
                    return
                }
                self.onFollowChange(value)
            }
        }
    }
}

private struct ChatEmptyState: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(spacing: 13) {
            TokenityBrandMark()
                .frame(width: 66, height: 42)
            Text(store.isChatReady ? "What would you like to explore?" : "Chat is ready when your model is")
                .font(.tokenitySectionTitle(22))
            Text(emptyDetail)
                .font(.tokenityText(13))
                .foregroundStyle(theme.secondaryText)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
        .padding(24)
    }

    private var emptyDetail: String {
        if store.isChatReady {
            return "Ask a question, paste code, or request a structured Markdown answer."
        }
        if store.phase == .running {
            return "Load a model from Models, then return here to begin a conversation."
        }
        return "Create a cluster and load a model. Your conversation history will remain available here."
    }
}

struct ChatMessageView: View {
    let message: ChatMessage
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var isHovered = false
    @State private var didCopy = false
    @State private var showsRouteDetails = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.role == .user { Spacer(minLength: 80) }
            if message.role == .assistant {
                assistantMark
            }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 9) {
                HStack(spacing: 8) {
                    Text(message.role == .user ? "You" : "Assistant")
                        .font(.tokenityText(11, weight: .semibold))
                        .foregroundStyle(theme.secondaryText)
                    if message.role == .assistant,
                       let modelName = message.routedModelID ?? message.modelName {
                        Text(modelName)
                            .font(.tokenityText(10))
                            .foregroundStyle(theme.tertiaryText)
                            .lineLimit(1)
                    }
                    if message.role == .assistant, let instanceID = message.instanceID {
                        Text("· \(instanceID)")
                            .font(.tokenityText(10))
                            .foregroundStyle(theme.tertiaryText)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }

                if message.role == .assistant {
                    assistantBody
                } else {
                    Text(message.content)
                        .font(.tokenityText(14))
                        .foregroundStyle(theme.text)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(theme.accent.opacity(0.14), in: RoundedRectangle(cornerRadius: 13, style: .continuous))
                }

                footer
            }
            .frame(maxWidth: message.role == .user ? 620 : .infinity, alignment: message.role == .user ? .trailing : .leading)
            if message.role == .assistant { Spacer(minLength: 24) }
        }
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
        .accessibilityElement(children: .contain)
    }

    private var assistantMark: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(theme.raisedSurface)
            TokenityBrandMark()
                .padding(4)
        }
        .frame(width: 30, height: 30)
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.border, lineWidth: 0.6)
        }
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private var assistantBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !message.thinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ThinkingDisclosureView(message: message)
            }

            if message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                if message.generationState?.isGenerating == true || message.generationState == .waiting {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(message.generationState == .reasoning ? "Reasoning…" : "Generating…")
                            .font(.tokenityText(13))
                            .foregroundStyle(theme.secondaryText)
                    }
                    .accessibilityLabel(message.generationState == .reasoning ? "Reasoning" : "Generating response")
                }
            } else {
                MarkdownMessageView(markdown: message.content)
                    .foregroundStyle(theme.text)
            }

            if let status = message.statusMessage,
               message.generationState != .completed,
               !message.content.hasSuffix(status) {
                generationNotice(status)
            }
        }
        .padding(16)
        .background(theme.surface, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .stroke(theme.border, lineWidth: 0.6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func generationNotice(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: statusSymbol)
            Text(text)
        }
        .font(.tokenityText(11))
        .foregroundStyle(statusColor)
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Text(message.createdAt.formatted(date: .omitted, time: .shortened))
                .font(.tokenityText(10))
                .foregroundStyle(theme.tertiaryText)

            if let metrics = message.metrics {
                metric("TTFT", metrics.firstTokenSeconds.map { String(format: "%.2fs", $0) })
                metric("Total", metrics.totalSeconds.map { String(format: "%.2fs", $0) })
                metric("Tokens", metrics.outputTokens.map(String.init))
                metric("Speed", metrics.outputTokensPerSecond.map { String(format: "%.1f tok/s", $0) })
            }

            if message.role == .assistant, message.routeReason != nil {
                Button("Why selected") {
                    showsRouteDetails.toggle()
                }
                .buttonStyle(.borderless)
                .font(.tokenityText(10, weight: .medium))
                .popover(isPresented: $showsRouteDetails, arrowEdge: .bottom) {
                    routeDetails
                }
            }

            Spacer(minLength: 4)
            messageActions
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func metric(_ label: String, _ value: String?) -> some View {
        if let value {
            Text("\(label) \(value)")
                .font(.tokenityText(10))
                .foregroundStyle(theme.tertiaryText)
        }
    }

    private var messageActions: some View {
        HStack(spacing: 4) {
            if message.generationState?.isGenerating == true {
                Button {
                    store.cancelChatGeneration()
                } label: {
                    Image(systemName: "stop.fill")
                }
                .help("Stop generation")
                .accessibilityLabel("Stop generation")
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(message.content, forType: .string)
                didCopy = true
                Task {
                    try? await Task.sleep(for: .seconds(1.2))
                    didCopy = false
                }
            } label: {
                Image(systemName: didCopy ? "checkmark" : "doc.on.doc")
            }
            .help(didCopy ? "Copied original Markdown" : "Copy original Markdown")
            .accessibilityLabel(didCopy ? "Copied" : "Copy original message")

            if message.role == .assistant, message.generationState?.isGenerating != true {
                Button {
                    store.regenerateAssistantMessage(message.id)
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(!store.isChatReady || store.isChatRunning)
                .help("Regenerate answer")
                .accessibilityLabel("Regenerate answer")
                Button {
                    store.regenerateAssistantMessageWithAnotherModel(message.id)
                } label: {
                    Image(systemName: "arrow.triangle.branch")
                }
                .disabled(!store.isChatReady || store.isChatRunning)
                .help("Regenerate with another model")
                .accessibilityLabel("Regenerate with another model")
            } else if message.role == .user {
                Button {
                    store.editChatMessage(message.id)
                } label: {
                    Image(systemName: "pencil")
                }
                .disabled(store.isChatRunning)
                .help("Edit from this message")
                .accessibilityLabel("Edit this message")
            }
        }
        .buttonStyle(.borderless)
        .foregroundStyle(theme.secondaryText)
        .opacity(isHovered ? 1 : 0.34)
    }

    private var routeDetails: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Routing decision")
                .font(.tokenityText(13, weight: .semibold))
            if let model = message.routedModelID {
                Text(model).font(.tokenityText(12, weight: .medium))
            }
            if let instanceID = message.instanceID {
                Text("Instance \(instanceID)")
                    .font(.tokenityText(10))
                    .foregroundStyle(theme.tertiaryText)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if let reason = message.routeReason {
                Text(reason)
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.secondaryText)
            }
            HStack(spacing: 10) {
                if let confidence = message.routeConfidence {
                    Text("Confidence \(Int((confidence * 100).rounded()))%")
                }
                if let latency = message.routingLatencyMilliseconds {
                    Text(String(format: "Route %.1f ms", latency))
                }
                if let wait = message.queueWaitMilliseconds {
                    Text(String(format: "Queue %.1f ms", wait))
                }
            }
            .font(.tokenityText(10))
            .foregroundStyle(theme.tertiaryText)
        }
        .padding(14)
        .frame(width: 280, alignment: .leading)
    }

    private var statusSymbol: String {
        switch message.generationState {
        case .stopped: return "stop.circle"
        case .repetitive: return "repeat.circle"
        case .lengthLimited: return "exclamationmark.circle"
        case .failed: return "xmark.circle"
        default: return "info.circle"
        }
    }

    private var statusColor: Color {
        switch message.generationState {
        case .failed, .repetitive: return theme.danger
        case .stopped, .lengthLimited: return theme.warning
        default: return theme.secondaryText
        }
    }
}

struct ThinkingDisclosureView: View {
    let message: ChatMessage
    @Environment(\.tokenityTheme) private var theme
    @State private var expansionState: ThinkingDisclosureState

    init(message: ChatMessage) {
        self.message = message
        _expansionState = State(
            initialValue: ThinkingDisclosureState(
                answerHasStarted: !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )
        )
    }

    var body: some View {
        DisclosureGroup(isExpanded: expansionBinding) {
            ScrollView(.horizontal) {
                Text(message.thinking)
                    .font(.tokenityText(12))
                    .foregroundStyle(theme.secondaryText)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
            }
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "brain.head.profile")
                Text(thinkingTitle)
                    .font(.tokenityText(12, weight: .medium))
                Spacer()
                if let count = message.reasoningTokenCount {
                    Text("≈\(count) tokens")
                }
                if let duration = message.reasoningDurationSeconds {
                    Text(String(format: "%.1fs", duration))
                }
            }
            .font(.tokenityText(10))
            .foregroundStyle(theme.secondaryText)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(theme.raisedSurface, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).stroke(theme.border, lineWidth: 0.6))
        .onChange(of: message.content.isEmpty) { wasEmpty, isEmpty in
            guard wasEmpty, !isEmpty else { return }
            withAnimation(.easeOut(duration: 0.16)) { expansionState.answerBegan() }
        }
        .accessibilityLabel(thinkingTitle)
    }

    private var expansionBinding: Binding<Bool> {
        Binding(
            get: { expansionState.isExpanded },
            set: {
                expansionState.userSetExpanded($0)
            }
        )
    }

    private var thinkingTitle: String {
        switch message.generationState {
        case .reasoning: return "Thinking"
        case .stopped: return "Thinking stopped"
        case .repetitive: return "Thinking stopped · repetition detected"
        case .lengthLimited: return "Thinking ended at token limit"
        default: return "Thinking"
        }
    }
}

struct ChatComposerView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var editorHeight: CGFloat = 46
    @State private var isEditorFocused = false

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    ChatComposerTextView(
                        text: $store.chatInput,
                        height: $editorHeight,
                        isEnabled: !store.isChatRunning && store.isChatReady,
                        focusRevision: store.chatComposerFocusRevision,
                        isFocused: $isEditorFocused,
                        onSend: send
                    )
                    .frame(height: editorHeight)
                    .accessibilityLabel("Message")
                    .accessibilityHint("Press Return to send or Shift Return for a new line")

                    if store.chatInput.isEmpty {
                        Text(composerPlaceholder)
                            .font(.tokenityText(14))
                            .foregroundStyle(theme.tertiaryText)
                            .padding(.leading, 4)
                            .padding(.top, 10)
                            .allowsHitTesting(false)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.top, 8)
                .padding(.bottom, 5)

                Rectangle()
                    .fill(theme.border.opacity(0.48))
                    .frame(height: 0.5)

                HStack(spacing: 10) {
                    ChatModelSelectionMenu()

                    if store.isAutoChatSelection {
                        Rectangle()
                            .fill(theme.border.opacity(0.55))
                            .frame(width: 0.5, height: 30)
                        ChatRoutePolicySlider()
                        ChatModelLockButton()
                    } else {
                        Label(thinkingModeTitle, systemImage: "brain.head.profile")
                            .font(.tokenityText(10, weight: .medium))
                            .foregroundStyle(theme.secondaryText)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 6)
                            .background(theme.group, in: Capsule())
                    }

                    Spacer(minLength: 4)

                    Button(action: sendOrStop) {
                        Image(systemName: store.isChatRunning ? "stop.fill" : "arrow.up")
                            .font(.system(size: 14, weight: .bold))
                            .frame(width: 36, height: 36)
                            .foregroundStyle(theme.accentText)
                            .background(sendButtonColor, in: Circle())
                            .shadow(
                                color: sendButtonColor.opacity(canSend || store.isChatRunning ? 0.14 : 0),
                                radius: 4,
                                y: 1
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!store.isChatRunning && !canSend)
                    .help(sendHelp)
                    .accessibilityLabel(store.isChatRunning ? "Stop generation" : "Send message")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
            }
            .background(theme.raisedSurface, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 15, style: .continuous)
                    .stroke(
                        isEditorFocused ? theme.accent : theme.strongBorder,
                        lineWidth: isEditorFocused ? 1.35 : 0.7
                    )
            )
            .shadow(color: .black.opacity(0.045), radius: 8, y: 2)

            HStack(spacing: 7) {
                Text(composerHint)
                Spacer()
                Text("Return to send · Shift-Return for a new line")
            }
            .font(.tokenityText(10))
            .foregroundStyle(theme.tertiaryText)
        }
        .padding(.horizontal, 20)
        .padding(.top, 9)
        .padding(.bottom, 11)
        .background(theme.window.opacity(0.92))
    }

    private var canSend: Bool {
        store.isChatReady && !store.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func send() {
        guard canSend, !store.isChatRunning else { return }
        store.beginSendingChatMessage()
    }

    private func sendOrStop() {
        if store.isChatRunning {
            store.cancelChatGeneration()
        } else {
            send()
        }
    }

    private var sendButtonColor: Color {
        if store.isChatRunning { return theme.danger }
        return canSend ? theme.accent : theme.tertiaryText.opacity(0.55)
    }

    private var sendHelp: String {
        if store.isChatRunning { return "Stop the current generation" }
        if !store.isChatReady { return "Create a cluster and load a model before sending" }
        if store.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a message to send" }
        return "Send message"
    }

    private var composerHint: String {
        if store.isChatRunning { return "The current response is streaming. Stop it before editing the next prompt." }
        if !store.isChatReady { return "Create a cluster and load a model to enable input." }
        return store.chatRoutingDisplayTitle
    }

    private var composerPlaceholder: String {
        if store.isChatRunning { return "Generating a response…" }
        if !store.isChatReady { return "Load a model to start chatting" }
        return "Message Tokenity…"
    }

    private var thinkingModeTitle: String {
        guard !store.isAutoChatSelection else { return "Router default" }
        let model = store.chatSelectedModelID
        return store.modelConfiguration(for: model).thinkingMode.title
    }
}

private struct ChatModelSelectionMenu: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        Menu {
            Button {
                store.selectChatModel("tokenity-auto")
            } label: {
                Label(
                    "Free routing",
                    systemImage: store.isAutoChatSelection ? "checkmark.circle.fill" : "sparkles"
                )
            }

            if !store.availableChatModelIDs.isEmpty {
                Divider()
            }
            ForEach(store.availableChatModelIDs, id: \.self) { modelID in
                Button {
                    store.selectChatModel(modelID)
                } label: {
                    Label(
                        modelID,
                        systemImage: store.chatSelectedModelID == modelID
                            ? "checkmark.circle.fill"
                            : "cube"
                    )
                }
            }
        } label: {
            HStack(spacing: 7) {
                Image(systemName: store.isAutoChatSelection ? "sparkles" : "cube")
                    .foregroundStyle(store.isAutoChatSelection ? theme.accent : theme.secondaryText)
                Text(selectionTitle)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(theme.tertiaryText)
            }
            .font(.tokenityText(11, weight: .medium))
            .foregroundStyle(theme.text)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .frame(minWidth: 94, maxWidth: 176, alignment: .leading)
            .background(theme.group, in: Capsule())
            .overlay(Capsule().stroke(theme.border.opacity(0.65), lineWidth: 0.5))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize(horizontal: true, vertical: false)
        .disabled(store.isChatRunning)
        .help("Choose free routing or a specific resident model")
        .accessibilityLabel("Model selection: \(selectionTitle)")
    }

    private var selectionTitle: String {
        store.isAutoChatSelection ? "Free routing" : store.chatSelectedModelID
    }
}

private struct ChatRoutePolicySlider: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(spacing: 2) {
            HStack(spacing: 4) {
                Text("Faster")
                Spacer(minLength: 4)
                Text(store.chatRoutePolicy.title)
                    .foregroundStyle(theme.accent)
                Spacer(minLength: 4)
                Text("Smarter")
            }
            .font(.tokenityText(9, weight: .medium))
            .foregroundStyle(theme.tertiaryText)

            Slider(value: policyValue, in: 0...2, step: 1)
                .controlSize(.mini)
                .tint(theme.accent)
                .accessibilityLabel("Routing policy")
                .accessibilityValue(store.chatRoutePolicy.title)
        }
        .frame(width: 188)
        .disabled(store.isChatRunning)
        .help("Move toward Faster for latency or Smarter for model quality")
    }

    private var policyValue: Binding<Double> {
        Binding(
            get: {
                switch store.chatRoutePolicy {
                case .fast: return 0
                case .balanced: return 1
                case .quality: return 2
                }
            },
            set: { value in
                let policy: ChatRoutePolicy
                switch Int(value.rounded()) {
                case 0: policy = .fast
                case 2: policy = .quality
                default: policy = .balanced
                }
                store.setChatRoutePolicy(policy)
            }
        )
    }
}

private struct ChatModelLockButton: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        Toggle(isOn: lockBinding) {
            Image(systemName: store.locksChatModel ? "lock.fill" : "lock.open")
                .frame(width: 28, height: 28)
        }
        .toggleStyle(.button)
        .buttonStyle(.plain)
        .foregroundStyle(store.locksChatModel ? theme.accent : theme.secondaryText)
        .background(
            store.locksChatModel ? theme.accent.opacity(0.13) : theme.group,
            in: Circle()
        )
        .overlay(Circle().stroke(theme.border.opacity(0.6), lineWidth: 0.5))
        .disabled(store.isChatRunning)
        .help("Keep this conversation on the same routed model")
        .accessibilityLabel("Lock routed model for this conversation")
    }

    private var lockBinding: Binding<Bool> {
        Binding(
            get: { store.locksChatModel },
            set: { store.setChatModelLocked($0) }
        )
    }
}

struct ComposerTextSyncState {
    private(set) var lastAcceptedBindingText: String
    private(set) var pendingNativeText: String?

    init(bindingText: String) {
        lastAcceptedBindingText = bindingText
    }

    mutating func nativeTextDidChange(_ text: String) {
        pendingNativeText = text
    }

    mutating func bindingDidUpdate(_ text: String, forceExternal: Bool = false) -> Bool {
        if let pendingNativeText {
            if text == pendingNativeText {
                self.pendingNativeText = nil
                lastAcceptedBindingText = text
            } else if forceExternal || text != lastAcceptedBindingText {
                self.pendingNativeText = nil
                lastAcceptedBindingText = text
                return true
            }
            return false
        }
        guard text != lastAcceptedBindingText else { return false }
        lastAcceptedBindingText = text
        return true
    }
}

private struct ChatComposerTextView: NSViewRepresentable {
    @Binding var text: String
    @Binding var height: CGFloat
    let isEnabled: Bool
    let focusRevision: Int
    @Binding var isFocused: Bool
    let onSend: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder

        let textView = ComposerNSTextView()
        textView.delegate = context.coordinator
        textView.onSend = onSend
        textView.onFocusChange = context.coordinator.updateFocus
        textView.font = .systemFont(ofSize: 14)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 2, height: 8)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        textView.string = text
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? ComposerNSTextView else { return }
        textView.onSend = onSend
        textView.onFocusChange = context.coordinator.updateFocus
        textView.isEditable = isEnabled
        textView.isSelectable = true

        // NSTextView owns the text while the user is editing. A store update from an
        // unrelated status poll can render this representable before SwiftUI has
        // propagated the latest binding value; comparing the two strings directly
        // would then write stale text back into the editor and reset its selection.
        let focusChanged = context.coordinator.focusRevision != focusRevision
        let bindingChangedExternally = context.coordinator.textSync.bindingDidUpdate(
            text,
            forceExternal: !isEnabled || focusChanged
        )
        if bindingChangedExternally, textView.string != text {
            let previousSelection = textView.selectedRange()
            context.coordinator.isApplyingBindingText = true
            textView.string = text
            context.coordinator.isApplyingBindingText = false
            let utf16Count = text.utf16.count
            let location = min(previousSelection.location, utf16Count)
            let length = min(previousSelection.length, utf16Count - location)
            textView.setSelectedRange(NSRange(location: location, length: length))
            context.coordinator.updateHeight(textView)
        }
        if focusChanged, isEnabled {
            context.coordinator.focusRevision = focusRevision
            DispatchQueue.main.async {
                textView.window?.makeFirstResponder(textView)
                textView.setSelectedRange(NSRange(location: textView.string.utf16.count, length: 0))
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ChatComposerTextView
        var focusRevision = Int.min
        var textSync: ComposerTextSyncState
        var lastReportedHeight: CGFloat
        var isApplyingBindingText = false

        init(parent: ChatComposerTextView) {
            self.parent = parent
            textSync = ComposerTextSyncState(bindingText: parent.text)
            lastReportedHeight = parent.height
        }

        func textDidChange(_ notification: Notification) {
            guard !isApplyingBindingText,
                  let textView = notification.object as? NSTextView else { return }
            let newText = textView.string
            textSync.nativeTextDidChange(newText)
            parent.text = newText
            updateHeight(textView)
        }

        func textDidBeginEditing(_ notification: Notification) {
            updateFocus(true)
        }

        func textDidEndEditing(_ notification: Notification) {
            updateFocus(false)
        }

        func updateFocus(_ focused: Bool) {
            guard parent.isFocused != focused else { return }
            parent.isFocused = focused
        }

        func updateHeight(_ textView: NSTextView) {
            guard let layoutManager = textView.layoutManager,
                  let textContainer = textView.textContainer else { return }
            layoutManager.ensureLayout(for: textContainer)
            let contentHeight = layoutManager.usedRect(for: textContainer).height + textView.textContainerInset.height * 2
            let next = min(max(46, ceil(contentHeight)), 164)
            if abs(lastReportedHeight - next) > 0.5 {
                lastReportedHeight = next
                DispatchQueue.main.async { self.parent.height = next }
            }
        }
    }
}

private final class ComposerNSTextView: NSTextView {
    var onSend: (() -> Void)?
    var onFocusChange: ((Bool) -> Void)?

    override func becomeFirstResponder() -> Bool {
        let becameFirstResponder = super.becomeFirstResponder()
        if becameFirstResponder {
            onFocusChange?(true)
        }
        return becameFirstResponder
    }

    override func resignFirstResponder() -> Bool {
        let resignedFirstResponder = super.resignFirstResponder()
        if resignedFirstResponder {
            onFocusChange?(false)
        }
        return resignedFirstResponder
    }

    override func doCommand(by selector: Selector) {
        guard selector == #selector(insertNewline(_:)) else {
            super.doCommand(by: selector)
            return
        }
        if hasMarkedText() {
            super.doCommand(by: selector)
        } else if NSEvent.modifierFlags.contains(.shift) {
            insertNewlineIgnoringFieldEditor(self)
        } else {
            onSend?()
        }
    }
}

struct ChatSidebarView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @Environment(\.colorScheme) private var colorScheme
    @State private var searchText = ""
    @State private var renameCandidate: ChatSession?
    @State private var renameDraft = ""
    @State private var deleteCandidate: ChatSession?

    var body: some View {
        VStack(spacing: 0) {
            sidebarHeader
                .frame(height: ChatWorkspaceMetrics.headerHeight)

            if store.isChatRunning {
                Label("Stop generation to switch conversations", systemImage: "lock")
                    .font(.tokenityText(10))
                    .foregroundStyle(theme.secondaryText)
                    .padding(.horizontal, 14)
                    .padding(.top, 10)
            }

            if store.isChatHistoryLoading {
                VStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Loading conversations…")
                        .font(.tokenityText(11))
                        .foregroundStyle(theme.secondaryText)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if filteredSessions.isEmpty {
                VStack(spacing: 9) {
                    Image(systemName: searchText.isEmpty ? "text.bubble" : "magnifyingglass")
                        .font(.system(size: 24, weight: .light))
                        .foregroundStyle(theme.tertiaryText)
                    Text(searchText.isEmpty ? "No conversations" : "No matching conversations")
                        .font(.tokenityText(12, weight: .medium))
                    if searchText.isEmpty {
                        Button("Start a New Chat") { store.newChatSession() }
                            .buttonStyle(.link)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(20)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 15) {
                        ForEach(ChatHistoryGroup.allCases) { group in
                            let groupSessions = sessions(in: group)
                            if !groupSessions.isEmpty {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(group.title.uppercased())
                                        .font(.tokenityText(9, weight: .semibold))
                                        .foregroundStyle(theme.tertiaryText)
                                        .tracking(0.45)
                                        .padding(.horizontal, 7)

                                    ForEach(groupSessions) { session in
                                        ChatHistoryRow(
                                            session: session,
                                            isSelected: store.activeChatSessionID == session.id
                                        ) {
                                            store.selectChatSession(session.id)
                                        }
                                        .contextMenu {
                                            Button("Rename…") { beginRename(session) }
                                            Divider()
                                            Button("Delete…", role: .destructive) { deleteCandidate = session }
                                        }
                                        .disabled(store.isChatRunning)
                                    }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 10)
                }
                .scrollIndicators(.visible)
            }
        }
        .background(theme.window)
        .sheet(item: $renameCandidate) { session in
            RenameConversationSheet(
                title: $renameDraft,
                onCancel: { renameCandidate = nil },
                onSave: {
                    store.renameChatSession(session.id, title: renameDraft)
                    renameCandidate = nil
                }
            )
        }
        .alert("Delete Conversation?", isPresented: deleteAlertBinding, presenting: deleteCandidate) { session in
            Button("Delete", role: .destructive) {
                store.deleteChatSession(session.id)
                deleteCandidate = nil
            }
            Button("Cancel", role: .cancel) { deleteCandidate = nil }
        } message: { session in
            Text("“\(session.title)” will be removed from this Mac. This cannot be undone.")
        }
    }

    private var sidebarHeader: some View {
        VStack(spacing: 10) {
            HStack {
                Text("History")
                    .font(.tokenitySectionTitle(18))
                    .foregroundStyle(theme.text)
                Spacer()
                Button {
                    store.newChatSession()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 24, height: 24)
                        .background(theme.group, in: Circle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(theme.secondaryText)
                .disabled(store.isChatRunning)
                .help("New conversation")
                .accessibilityLabel("New conversation")
            }
            .padding(.horizontal, 14)

            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(theme.tertiaryText)

                TextField("Search conversations", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.tokenityText(11))

                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(theme.tertiaryText)
                    }
                    .buttonStyle(.plain)
                    .help("Clear search")
                    .accessibilityLabel("Clear search")
                }
            }
            .padding(.horizontal, 9)
            .frame(height: 29)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(theme.group)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .stroke(theme.border.opacity(colorScheme == .dark ? 0.8 : 0.55), lineWidth: 0.5)
            }
            .padding(.horizontal, 12)
        }
        .padding(.top, 12)
        .padding(.bottom, 11)
        .background(theme.window)
    }

    private var filteredSessions: [ChatSession] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return store.chatSessions }
        return store.chatSessions.filter { $0.matches(search: query) }
    }

    private func sessions(in group: ChatHistoryGroup) -> [ChatSession] {
        filteredSessions.filter { ChatHistoryGroup.group(for: $0.updatedAt) == group }
    }

    private func beginRename(_ session: ChatSession) {
        renameDraft = session.title
        renameCandidate = session
    }

    private var deleteAlertBinding: Binding<Bool> {
        Binding(
            get: { deleteCandidate != nil },
            set: { if !$0 { deleteCandidate = nil } }
        )
    }
}

private struct ChatHistoryRow: View {
    let session: ChatSession
    let isSelected: Bool
    let action: () -> Void

    @Environment(\.tokenityTheme) private var theme
    @Environment(\.colorScheme) private var colorScheme
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Capsule()
                    .fill(isSelected ? theme.accent.opacity(0.82) : Color.clear)
                    .frame(width: 2.5, height: 30)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(session.title)
                            .font(.tokenityText(12, weight: isSelected ? .semibold : .medium))
                            .foregroundStyle(theme.text)
                            .lineLimit(1)
                        Spacer(minLength: 2)
                        Text(session.updatedAt, style: .relative)
                            .font(.tokenityText(9))
                            .foregroundStyle(theme.tertiaryText)
                    }
                    Text(session.preview)
                        .font(.tokenityText(10))
                        .foregroundStyle(theme.secondaryText)
                        .lineLimit(2)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(rowBackground)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(
                    isSelected ? theme.border.opacity(colorScheme == .dark ? 0.92 : 0.72) : Color.clear,
                    lineWidth: 0.5
                )
        }
        .onHover { isHovered = $0 }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(session.title), \(session.preview)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var rowBackground: Color {
        if isSelected {
            return theme.selection
        }
        return isHovered ? theme.hover : Color.clear
    }
}

private struct RenameConversationSheet: View {
    @Binding var title: String
    let onCancel: () -> Void
    let onSave: () -> Void
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Rename Conversation")
                .font(.tokenityText(17, weight: .semibold))
            TextField("Conversation title", text: $title)
                .textFieldStyle(.roundedBorder)
                .focused($isFocused)
                .onSubmit(onSave)
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Save", action: onSave)
                    .keyboardShortcut(.defaultAction)
                    .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 380)
        .onAppear { isFocused = true }
    }
}
