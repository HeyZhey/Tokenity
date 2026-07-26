import AppKit
import SwiftUI

struct ChatWorkspaceView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var showsHistory = true

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                ChatWorkspaceHeader(showsHistory: $showsHistory)
                Rectangle()
                    .fill(theme.border.opacity(0.55))
                    .frame(height: 0.5)
                ChatTranscriptView()
                ChatComposerView()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if showsHistory {
                Divider()
                ChatSidebarView()
                    .frame(minWidth: 244, idealWidth: 278, maxWidth: 326)
                    .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
        .background {
            LinearGradient(
                colors: [theme.window, theme.accent.opacity(0.018)],
                startPoint: .top,
                endPoint: .bottom
            )
        }
        .frame(minWidth: 780, minHeight: 620)
        .animation(.easeInOut(duration: 0.18), value: showsHistory)
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
                    .font(.tokenityText(19, weight: .semibold))
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
                .buttonStyle(.plain)
                .foregroundStyle(theme.secondaryText)
                .background(theme.group, in: Circle())
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
                .buttonStyle(.plain)
                .foregroundStyle(theme.secondaryText)
                .background(theme.group, in: Circle())
                .help(showsHistory ? "Hide conversation history" : "Show conversation history")
                .accessibilityLabel(showsHistory ? "Hide conversation history" : "Show conversation history")
            }

            HStack(spacing: 9) {
                Label(store.selectedModelName, systemImage: "cube")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(theme.group, in: Capsule())

                Label(
                    connectionLabel,
                    systemImage: store.selectedNodes.count > 1
                        ? "point.3.connected.trianglepath.dotted"
                        : "desktopcomputer"
                )
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(theme.group, in: Capsule())
                .accessibilityLabel(connectionLabel)

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
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial)
    }

    private var currentTitle: String {
        store.chatSessions.first(where: { $0.id == store.activeChatSessionID })?.title ?? "New Chat"
    }

    private var connectionLabel: String {
        if store.selectedNodes.count > 1 {
            return "\(store.selectedNodes.count) Macs · \(store.connectionMode.shortName)"
        }
        return "Single Mac"
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
    private let coordinateSpace = "tokenity-chat-transcript"

    var body: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 24) {
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
                            GeometryReader { geometry in
                                Color.clear.preference(
                                    key: TranscriptBottomPreferenceKey.self,
                                    value: geometry.frame(in: .named(coordinateSpace)).maxY
                                )
                            }
                            .frame(height: 1)
                            .id(bottomID)
                        }
                        .id(store.activeChatSessionID)
                        .frame(maxWidth: 900, alignment: .leading)
                        .padding(.horizontal, 30)
                        .padding(.vertical, 30)
                        .frame(maxWidth: .infinity)
                        .background {
                            TranscriptScrollActivityObserver {
                                followState.userDidScroll()
                            }
                        }
                    }
                    .coordinateSpace(name: coordinateSpace)
                    .scrollIndicators(.visible)
                    .background(Color.clear)
                    .onPreferenceChange(TranscriptBottomPreferenceKey.self) { bottom in
                        followState.update(bottomDistance: bottom - viewport.size.height)
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
                        .background(.regularMaterial, in: Capsule())
                        .padding(18)
                        .shadow(color: .black.opacity(0.12), radius: 7, y: 2)
                        .accessibilityLabel("Return to latest message")
                    }
                }
            }
        }
    }
}

private struct TranscriptBottomPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

private struct TranscriptScrollActivityObserver: NSViewRepresentable {
    let onUserScroll: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onUserScroll: onUserScroll)
    }

    func makeNSView(context: Context) -> ObserverView {
        let view = ObserverView()
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ nsView: ObserverView, context: Context) {
        context.coordinator.onUserScroll = onUserScroll
        nsView.coordinator = context.coordinator
        context.coordinator.attachIfPossible(from: nsView)
    }

    final class ObserverView: NSView {
        weak var coordinator: Coordinator?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.coordinator?.attachIfPossible(from: self)
            }
        }
    }

    final class Coordinator {
        var onUserScroll: () -> Void
        private weak var scrollView: NSScrollView?
        private var observers: [NSObjectProtocol] = []
        private var eventMonitor: Any?

        init(onUserScroll: @escaping () -> Void) {
            self.onUserScroll = onUserScroll
        }

        deinit {
            detach()
        }

        func attachIfPossible(from view: NSView) {
            var ancestor = view.superview
            while let candidate = ancestor, !(candidate is NSScrollView) {
                ancestor = candidate.superview
            }
            guard let scrollView = ancestor as? NSScrollView,
                  self.scrollView !== scrollView else { return }
            detach()
            self.scrollView = scrollView
            let center = NotificationCenter.default
            for name in [NSScrollView.willStartLiveScrollNotification, NSScrollView.didLiveScrollNotification] {
                observers.append(center.addObserver(forName: name, object: scrollView, queue: .main) { [weak self] _ in
                    self?.onUserScroll()
                })
            }
            eventMonitor = NSEvent.addLocalMonitorForEvents(
                matching: [.scrollWheel, .leftMouseDown, .leftMouseDragged]
            ) { [weak self, weak scrollView] event in
                guard let self, let scrollView,
                      event.window === scrollView.window
                else { return event }
                let point = scrollView.convert(event.locationInWindow, from: nil)
                let isScrollWheelInsideTranscript = event.type == .scrollWheel
                    && scrollView.bounds.contains(point)
                let isDraggingTranscriptScroller = event.type != .scrollWheel
                    && [scrollView.verticalScroller, scrollView.horizontalScroller]
                        .compactMap { $0 }
                        .contains(where: { !$0.isHidden && $0.frame.contains(point) })
                if isScrollWheelInsideTranscript || isDraggingTranscriptScroller {
                    self.onUserScroll()
                }
                return event
            }
        }

        private func detach() {
            let center = NotificationCenter.default
            observers.forEach(center.removeObserver)
            observers.removeAll()
            if let eventMonitor {
                NSEvent.removeMonitor(eventMonitor)
                self.eventMonitor = nil
            }
            scrollView = nil
        }
    }
}

private struct ChatEmptyState: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(spacing: 13) {
            Image(systemName: "sparkles")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(theme.accent)
                .accessibilityHidden(true)
            Text(store.isChatReady ? "What would you like to explore?" : "Chat is ready when your model is")
                .font(.tokenityText(20, weight: .semibold))
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
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(theme.accent.opacity(0.13))
            Image(systemName: "sparkles")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(theme.accent)
        }
        .frame(width: 28, height: 28)
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
        .background(theme.group, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(theme.border.opacity(0.8), lineWidth: 0.5))
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

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    ChatComposerTextView(
                        text: $store.chatInput,
                        height: $editorHeight,
                        isEnabled: !store.isChatRunning && store.isChatReady,
                        focusRevision: store.chatComposerFocusRevision,
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
                            .foregroundStyle(Color.white)
                            .background(sendButtonColor, in: Circle())
                            .shadow(
                                color: sendButtonColor.opacity(canSend || store.isChatRunning ? 0.28 : 0),
                                radius: 7,
                                y: 2
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
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 17, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 17, style: .continuous)
                    .stroke(theme.border.opacity(0.78), lineWidth: 0.6)
            )
            .shadow(color: .black.opacity(0.08), radius: 13, y: 4)

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
        return store.selectedModelName
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
                    "Auto route",
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
        .help("Choose automatic routing or a specific resident model")
        .accessibilityLabel("Model selection: \(selectionTitle)")
    }

    private var selectionTitle: String {
        store.isAutoChatSelection ? "Auto" : store.chatSelectedModelID
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
    @State private var searchText = ""
    @State private var renameCandidate: ChatSession?
    @State private var renameDraft = ""
    @State private var deleteCandidate: ChatSession?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("History")
                    .font(.tokenityText(14, weight: .semibold))
                Spacer()
                Button {
                    store.newChatSession()
                } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .disabled(store.isChatRunning)
                .help("New conversation")
                .accessibilityLabel("New conversation")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)

            if store.isChatRunning {
                Label("Stop generation to switch conversations", systemImage: "lock")
                    .font(.tokenityText(10))
                    .foregroundStyle(theme.secondaryText)
                    .padding(.horizontal, 14)
                    .padding(.bottom, 8)
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
                List(selection: selection) {
                    ForEach(ChatHistoryGroup.allCases) { group in
                        let sessions = sessions(in: group)
                        if !sessions.isEmpty {
                            Section(group.title) {
                                ForEach(sessions) { session in
                                    ChatHistoryRow(session: session)
                                        .tag(session.id)
                                        .contextMenu {
                                            Button("Rename…") { beginRename(session) }
                                            Divider()
                                            Button("Delete…", role: .destructive) { deleteCandidate = session }
                                        }
                                }
                            }
                        }
                    }
                }
                .listStyle(.sidebar)
                .disabled(store.isChatRunning)
            }
        }
        .background(theme.sidebar)
        .searchable(text: $searchText, placement: .sidebar, prompt: "Search conversations")
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

    private var selection: Binding<UUID?> {
        Binding(
            get: { store.activeChatSessionID },
            set: { id in
                guard let id else { return }
                store.selectChatSession(id)
            }
        )
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
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(session.title)
                    .font(.tokenityText(12, weight: .medium))
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
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(session.title), \(session.preview)")
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
