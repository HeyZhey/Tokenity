import SwiftUI

enum TokenityOnboardingPage: Int, CaseIterable, Identifiable {
    case welcome
    case install
    case workflow
    case ready

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .welcome: return "Private AI, powered by your Macs"
        case .install: return "Prepare every inference Mac"
        case .workflow: return "From cluster to conversation"
        case .ready: return "You are ready to begin"
        }
    }

    var eyebrow: String {
        switch self {
        case .welcome: return "WELCOME TO TOKENITY"
        case .install: return "ONE-TIME SETUP"
        case .workflow: return "HOW TO USE TOKENITY"
        case .ready: return "YOUR FIRST RUN"
        }
    }
}

struct TokenityOnboardingView: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var page: TokenityOnboardingPage = .welcome
    @StateObject private var runtimeBootstrap = TokenityRuntimeBootstrapModel()

    init(initialPage: TokenityOnboardingPage = .welcome) {
        _page = State(initialValue: initialPage)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                pageContent
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
        }
    }

    private var header: some View {
        HStack(spacing: 15) {
            TokenityBrandLockup()
                .frame(width: 94, height: 72)

            VStack(alignment: .leading, spacing: 3) {
                Text("Distributed AI")
                    .font(.tokenitySectionTitle(18))
                    .foregroundStyle(theme.text)
                Text("Distributed MLX inference for Apple silicon")
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
        .accessibilityLabel("Tokenity, Distributed AI. Distributed MLX inference for Apple silicon.")
        .accessibilityIdentifier("tokenity-onboarding-brand")
    }

    @ViewBuilder
    private var pageContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                Text(page.eyebrow)
                    .font(.tokenityText(11, weight: .semibold))
                    .foregroundStyle(theme.accent)
                Text(page.title)
                    .font(.tokenityTitle(28))
                    .foregroundStyle(theme.text)
            }

            switch page {
            case .welcome:
                welcomePage
            case .install:
                installPage
            case .workflow:
                workflowPage
            case .ready:
                readyPage
            }
        }
    }

    private var welcomePage: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text(
                "Tokenity turns one or more Apple-silicon Macs into a local language-model cluster. "
                    + "It coordinates model loading, distributed inference, automatic model routing, "
                    + "Chat, and an OpenAI-compatible API from one native Mac app."
            )
            .font(.tokenityText(15))
            .foregroundStyle(theme.secondaryText)
            .fixedSize(horizontal: false, vertical: true)

            HStack(alignment: .top, spacing: 14) {
                OnboardingCard(
                    symbol: "lock.shield",
                    title: "Runs locally",
                    detail: "Prompts and model traffic stay on the Macs and networks you control.",
                    color: theme.success
                )
                OnboardingCard(
                    symbol: "point.3.connected.trianglepath.dotted",
                    title: "Uses combined hardware",
                    detail: "Run MLX models across selected Macs over LAN or Thunderbolt RDMA.",
                    color: theme.accent
                )
                OnboardingCard(
                    symbol: "arrow.triangle.branch",
                    title: "Routes automatically",
                    detail: "Keep multiple models resident and let Tokenity select for speed or quality.",
                    color: theme.warning
                )
            }
        }
    }

    private var installPage: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(
                "The control app and the inference service have separate installation steps. "
                    + "This keeps a normal drag-to-Applications experience while making the privileged "
                    + "Node Agent installation explicit."
            )
            .font(.tokenityText(14))
            .foregroundStyle(theme.secondaryText)
            .fixedSize(horizontal: false, vertical: true)

            OnboardingInstruction(
                number: 1,
                title: "Install the control app",
                detail: "Drag TokenityControl.app to Applications, then open it on the Mac you will use to manage the cluster.",
                symbol: "macwindow"
            )
            OnboardingInstruction(
                number: 2,
                title: "Install the Node Agent on every inference Mac",
                detail: "Open the verified Runtime installer included with Tokenity. Installer.app requests administrator approval, installs the Node Agent, and starts its local service on port 9100.",
                symbol: "shippingbox"
            )
            OnboardingInstruction(
                number: 3,
                title: "Make models available",
                detail: "Place compatible MLX model folders under \(TokenityDeploymentConfiguration.modelRoot.path) on each participating Mac, or set TOKENITY_MODEL_ROOT.",
                symbol: "externaldrive"
            )

            HStack(spacing: 14) {
                Image(
                    systemName: runtimeBootstrap.hasFailed
                        ? "exclamationmark.triangle.fill"
                        : "shippingbox.fill"
                )
                .font(.system(size: 21, weight: .medium))
                .foregroundStyle(
                    runtimeBootstrap.hasFailed || runtimeBootstrap.state == .downloadAvailable
                        ? theme.warning
                        : theme.accent
                )
                .frame(width: 28)

                VStack(alignment: .leading, spacing: 3) {
                    Text(runtimeBootstrap.title)
                        .font(.tokenityText(13, weight: .semibold))
                        .foregroundStyle(theme.text)
                    Text(runtimeBootstrap.detail)
                        .font(.tokenityText(11))
                        .foregroundStyle(theme.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                if runtimeBootstrap.isBusy {
                    ProgressView()
                        .controlSize(.small)
                } else if let actionTitle = runtimeBootstrap.actionTitle {
                    Button(actionTitle) {
                        runtimeBootstrap.performPrimaryAction()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(theme.controlAccent)
                    .controlSize(.small)
                }
            }
            .padding(13)
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(theme.border, lineWidth: 0.6)
            )

            Label(
                "The bundled Runtime is for Apple silicon and currently requires macOS 26.2 or newer. For multi-Mac inference, run the same installer locally on every participating Mac.",
                systemImage: "info.circle"
            )
            .font(.tokenityText(12))
            .foregroundStyle(theme.secondaryText)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(theme.accent.opacity(0.08), in: RoundedRectangle(cornerRadius: 9))
        }
    }

    private var workflowPage: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text("Most first-time setups follow the same three-step path.")
                .font(.tokenityText(14))
                .foregroundStyle(theme.secondaryText)

            HStack(alignment: .top, spacing: 14) {
                OnboardingWorkflowStep(
                    number: 1,
                    symbol: "point.3.connected.trianglepath.dotted",
                    title: "Cluster",
                    detail: "Confirm the Macs are online, select the participants and connection mode, then choose Create Cluster.",
                    color: theme.accent
                )
                OnboardingWorkflowStep(
                    number: 2,
                    symbol: "cube.transparent",
                    title: "Models",
                    detail: "Scan the shared model directory, load a model, and wait for the real inference probe to report Ready.",
                    color: theme.warning
                )
                OnboardingWorkflowStep(
                    number: 3,
                    symbol: "bubble.left.and.bubble.right",
                    title: "Chat",
                    detail: "Use Auto or choose a resident model manually. Speed, Balanced, and Quality control routing priorities.",
                    color: theme.success
                )
            }

            HStack(spacing: 12) {
                Image(systemName: "network")
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(theme.accent)
                    .frame(width: 34)
                VStack(alignment: .leading, spacing: 3) {
                    Text("External apps use the Coordinator")
                        .font(.tokenityText(13, weight: .semibold))
                        .foregroundStyle(theme.text)
                    Text("Connect OpenAI-compatible clients to http://<coordinator>:9100/v1. Runtime ports such as 8000 are private implementation details.")
                        .font(.tokenityText(12))
                        .foregroundStyle(theme.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(14)
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(theme.border, lineWidth: 0.6)
            )
        }
    }

    private var readyPage: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(
                "Tokenity only reports a model as Ready after the expected ranks are healthy, "
                    + "the runtime has warmed up, and a real inference probe succeeds."
            )
            .font(.tokenityText(14))
            .foregroundStyle(theme.secondaryText)
            .fixedSize(horizontal: false, vertical: true)

            VStack(spacing: 0) {
                OnboardingChecklistRow(
                    symbol: "checkmark.circle.fill",
                    title: "Node Agents are online",
                    detail: "Cluster shows every selected Mac as reachable on port 9100."
                )
                OnboardingChecklistRow(
                    symbol: "checkmark.circle.fill",
                    title: "Network path is healthy",
                    detail: "Use Standard Network, or verify Thunderbolt RDMA is active on every selected Mac."
                )
                OnboardingChecklistRow(
                    symbol: "checkmark.circle.fill",
                    title: "At least one model is Ready",
                    detail: "Chat and the OpenAI-compatible API become available after loading completes."
                )
            }
            .background(theme.surface, in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(theme.border, lineWidth: 0.6)
            )

            Text("You can reopen this guide at any time from Settings or the Help menu.")
                .font(.tokenityText(12))
                .foregroundStyle(theme.tertiaryText)
        }
    }

    private var footer: some View {
        HStack {
            HStack(spacing: 7) {
                ForEach(TokenityOnboardingPage.allCases) { item in
                    Button {
                        page = item
                    } label: {
                        Capsule()
                            .fill(item == page ? theme.accent : theme.border)
                            .frame(width: item == page ? 22 : 7, height: 7)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Show onboarding page \(item.rawValue + 1)")
                }
            }

            Spacer()

            if page == .welcome {
                Button("Skip Guide") {
                    store.completeOnboarding()
                }
                .buttonStyle(.plain)
                .foregroundStyle(theme.secondaryText)
                .accessibilityIdentifier("tokenity.onboarding.skip")
            } else {
                Button("Back") {
                    move(by: -1)
                }
                .accessibilityIdentifier("tokenity.onboarding.back")
            }

            if page == .ready {
                Button("Open Cluster Setup") {
                    store.completeOnboarding(opening: .cluster)
                }
                .buttonStyle(.borderedProminent)
                .tint(theme.controlAccent)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("tokenity.onboarding.open-cluster")
            } else {
                Button("Continue") {
                    move(by: 1)
                }
                .buttonStyle(.borderedProminent)
                .tint(theme.controlAccent)
                .keyboardShortcut(.defaultAction)
                .accessibilityIdentifier("tokenity.onboarding.continue")
            }
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 17)
        .background(theme.sidebar)
    }

    private func move(by offset: Int) {
        let target = min(
            max(page.rawValue + offset, 0),
            TokenityOnboardingPage.allCases.count - 1
        )
        if let nextPage = TokenityOnboardingPage(rawValue: target) {
            page = nextPage
        }
    }
}

private struct OnboardingCard: View {
    let symbol: String
    let title: String
    let detail: String
    let color: Color

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(color)
            Text(title)
                .font(.tokenityText(14, weight: .semibold))
                .foregroundStyle(theme.text)
            Text(detail)
                .font(.tokenityText(12))
                .foregroundStyle(theme.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .background(theme.surface, in: RoundedRectangle(cornerRadius: 13))
        .overlay(
            RoundedRectangle(cornerRadius: 13)
                .stroke(theme.border, lineWidth: 0.6)
        )
    }
}

private struct OnboardingInstruction: View {
    let number: Int
    let title: String
    let detail: String
    let symbol: String

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(theme.accent.opacity(0.13))
                Text("\(number)")
                    .font(.tokenityText(13, weight: .bold))
                    .foregroundStyle(theme.accent)
            }
            .frame(width: 30, height: 30)

            Image(systemName: symbol)
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(theme.secondaryText)
                .frame(width: 25, height: 30)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.tokenityText(14, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(detail)
                    .font(.tokenityText(12))
                    .foregroundStyle(theme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

private struct OnboardingWorkflowStep: View {
    let number: Int
    let symbol: String
    let title: String
    let detail: String
    let color: Color

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: symbol)
                    .font(.system(size: 22, weight: .medium))
                    .foregroundStyle(color)
                Spacer()
                Text("\(number)")
                    .font(.tokenityText(11, weight: .bold))
                    .foregroundStyle(color)
                    .frame(width: 24, height: 24)
                    .background(color.opacity(0.12), in: Circle())
            }
            Text(title)
                .font(.tokenityText(16, weight: .semibold))
                .foregroundStyle(theme.text)
            Text(detail)
                .font(.tokenityText(12))
                .foregroundStyle(theme.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: 175, alignment: .topLeading)
        .background(theme.surface, in: RoundedRectangle(cornerRadius: 13))
        .overlay(
            RoundedRectangle(cornerRadius: 13)
                .stroke(theme.border, lineWidth: 0.6)
        )
    }
}

private struct OnboardingChecklistRow: View {
    let symbol: String
    let title: String
    let detail: String

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(theme.success)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.tokenityText(13, weight: .semibold))
                    .foregroundStyle(theme.text)
                Text(detail)
                    .font(.tokenityText(12))
                    .foregroundStyle(theme.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 48)
        }
    }
}
