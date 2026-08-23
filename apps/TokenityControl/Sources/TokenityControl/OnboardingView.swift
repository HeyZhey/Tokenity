import SwiftUI

enum TokenityOnboardingPage: Int, CaseIterable, Identifiable {
    case welcome
    case ready

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .welcome: return "What would you like to do?"
        case .ready: return "Prepare this Mac"
        }
    }

    var eyebrow: String {
        switch self {
        case .welcome: return "WELCOME TO TOKENITY"
        case .ready: return "READINESS"
        }
    }
}

private enum TokenityFirstTask: String {
    case language
    case video

    var title: String {
        switch self {
        case .language: return "Run a language model"
        case .video: return "Generate a video"
        }
    }

    var section: AppSection { self == .language ? .models : .video }
}

struct TokenityOnboardingView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var page: TokenityOnboardingPage = .welcome
    @State private var task: TokenityFirstTask = .language
    @StateObject private var runtimeBootstrap = TokenityRuntimeBootstrapModel()

    init(initialPage: TokenityOnboardingPage = .welcome) {
        _page = State(initialValue: initialPage)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(page.eyebrow)
                            .font(.tokenityText(11, weight: .semibold))
                            .foregroundStyle(theme.accent)
                        Text(page.title)
                            .font(.tokenityTitle(28))
                            .foregroundStyle(theme.text)
                    }
                    if page == .welcome { taskPage } else { readinessPage }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.horizontal, 42)
                .padding(.vertical, 30)
            }
            .scrollIndicators(.never)
            Divider()
            footer
        }
        .frame(width: 780, height: 590)
        .background(theme.window)
        .interactiveDismissDisabled()
        .accessibilityIdentifier("tokenity.onboarding")
        .task {
            await runtimeBootstrap.refresh()
            await store.refreshSelectedNodeStatus()
        }
    }

    private var header: some View {
        HStack(spacing: 15) {
            TokenityBrandLockup().frame(width: 94, height: 72)
            VStack(alignment: .leading, spacing: 3) {
                Text("Tokenity")
                    .font(.tokenitySectionTitle(18))
                    .foregroundStyle(theme.text)
                Text("Local AI for Apple silicon")
                    .font(.tokenityText(12))
                    .foregroundStyle(theme.secondaryText)
            }
            Spacer()
            Text("\(page.rawValue + 1) of \(TokenityOnboardingPage.allCases.count)")
                .font(.tokenityText(11, weight: .medium))
                .foregroundStyle(theme.tertiaryText)
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 12)
        .background(theme.sidebar)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Tokenity. Local AI for Apple silicon.")
        .accessibilityIdentifier("tokenity-onboarding-brand")
    }

    private var taskPage: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Choose a task. Tokenity will check the required components, model, and Macs on the next screen.")
                .font(.tokenityText(15))
                .foregroundStyle(theme.secondaryText)

            HStack(spacing: 16) {
                taskButton(
                    .language,
                    symbol: "bubble.left.and.bubble.right",
                    detail: "Load a local LLM, chat, or use the OpenAI-compatible API."
                )
                taskButton(
                    .video,
                    symbol: "film.stack",
                    detail: "Generate MiniMax H3 video, follow progress, cancel, and reopen results."
                )
            }
        }
    }

    private func taskButton(
        _ value: TokenityFirstTask,
        symbol: String,
        detail: String
    ) -> some View {
        Button {
            task = value
            page = .ready
            Task {
                await runtimeBootstrap.refresh()
                if value == .video {
                    await store.refreshVideoNodes(validateRuntime: true)
                } else {
                    await store.scanModels()
                }
            }
        } label: {
            VStack(alignment: .leading, spacing: 13) {
                Image(systemName: symbol)
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(theme.accent)
                Text(value.title)
                    .font(.tokenitySectionTitle(18))
                    .foregroundStyle(theme.text)
                Text(detail)
                    .font(.tokenityText(13))
                    .foregroundStyle(theme.secondaryText)
                    .multilineTextAlignment(.leading)
                Spacer(minLength: 0)
                Label("Choose", systemImage: "arrow.right.circle.fill")
                    .font(.tokenityText(12, weight: .semibold))
                    .foregroundStyle(theme.accent)
            }
            .padding(20)
            .frame(maxWidth: .infinity, minHeight: 230, alignment: .topLeading)
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 14))
            .overlay(RoundedRectangle(cornerRadius: 14).stroke(theme.border, lineWidth: 0.8))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("tokenity.onboarding.task.\(value.rawValue)")
    }

    private var readinessPage: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(task.title)
                .font(.tokenitySectionTitle(18))
                .foregroundStyle(theme.text)

            VStack(spacing: 0) {
                readinessRow(
                    title: runtimeBootstrap.state == .installed
                        ? "This Mac is ready"
                        : "This Mac needs Tokenity components",
                    detail: runtimeBootstrap.detail,
                    ready: runtimeBootstrap.state == .installed
                ) {
                    if let action = runtimeBootstrap.actionTitle {
                        Button(action == "Reinstall Runtime" ? "Install or Repair" : action) {
                            runtimeBootstrap.performPrimaryAction()
                        }
                        .controlSize(.small)
                    }
                }

                if task == .language {
                    let hasModel = store.modelLibraryRows.contains { $0.modality == .language }
                    readinessRow(
                        title: hasModel ? "Language model found" : "Choose a model folder",
                        detail: hasModel ? "Tokenity found a compatible language model." : "Choose the folder that contains your MLX model, then scan again.",
                        ready: hasModel
                    ) {
                        Button("Open Models") {
                            store.completeOnboarding(opening: .models)
                        }
                        .controlSize(.small)
                    }
                } else {
                    let hasModelIssue = store.videoReadiness.contains { $0.state == .modelNotFound }
                    let hasRuntimeIssue = store.videoReadiness.contains {
                        $0.state == .runtimeMissing || $0.state == .componentsNeedUpdate
                    }
                    readinessRow(
                        title: hasModelIssue ? "Video model not found" : "Video model found",
                        detail: hasModelIssue ? "Choose the MiniMax H3 model folder." : "The MiniMax H3 model selection is ready to check.",
                        ready: !hasModelIssue
                    ) {
                        Button("Choose in Video") { store.completeOnboarding(opening: .video) }
                            .controlSize(.small)
                    }
                    readinessRow(
                        title: hasRuntimeIssue ? "Video runtime needs repair" : "Video runtime installed",
                        detail: hasRuntimeIssue ? "Install or repair Tokenity components, then scan again." : "Tokenity will run a full model and runtime preflight before starting.",
                        ready: !hasRuntimeIssue
                    ) {
                        if hasRuntimeIssue {
                            Button("Install or Repair") { runtimeBootstrap.performPrimaryAction() }
                                .controlSize(.small)
                        }
                    }
                }

                readinessRow(
                    title: "Optional: Add another Mac",
                    detail: "One Mac works for language models and MiniMax H3. Add a second Mac when you want more capacity.",
                    ready: true
                ) {
                    Button("Add Mac") { store.completeOnboarding(opening: .cluster) }
                        .controlSize(.small)
                }
            }
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(theme.border, lineWidth: 0.6))
        }
    }

    private func readinessRow<Action: View>(
        title: String,
        detail: String,
        ready: Bool,
        @ViewBuilder action: () -> Action
    ) -> some View {
        HStack(spacing: 13) {
            Image(systemName: ready ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(ready ? theme.success : theme.warning)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.tokenityText(13, weight: .semibold))
                Text(detail)
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.secondaryText)
                    .lineLimit(2)
            }
            Spacer(minLength: 8)
            action()
        }
        .padding(14)
        .overlay(alignment: .bottom) { Divider() }
    }

    private var footer: some View {
        HStack {
            Button("Skip") { store.completeOnboarding() }
                .buttonStyle(.plain)
                .foregroundStyle(theme.secondaryText)
                .accessibilityIdentifier("tokenity.onboarding.skip")
            Spacer()
            if page == .ready {
                Button("Back") { page = .welcome }
                    .accessibilityIdentifier("tokenity.onboarding.back")
                Button("Open \(task == .language ? "Models" : "Video")") {
                    store.completeOnboarding(opening: task.section)
                }
                .buttonStyle(.borderedProminent)
                .tint(theme.controlAccent)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("tokenity.onboarding.open-cluster")
            }
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 17)
        .background(theme.sidebar)
    }
}
