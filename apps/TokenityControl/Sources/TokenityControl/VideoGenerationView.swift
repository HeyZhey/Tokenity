import AVFoundation
import AVKit
import AppKit
import SwiftUI

struct VideoGenerationPage: View {
    @EnvironmentObject private var store: TokenityStore
    @Environment(\.tokenityTheme) private var theme
    @State private var player: AVPlayer?
    @State private var showsRuntimeConfiguration = false

    var body: some View {
        PageScaffold(title: "Video") {
            runtimeSection
            generationSection
            outputSection
        }
        .task {
            await store.refreshVideoNodes()
            configurePlayer(for: store.generatedVideoArtifact)
        }
        .onChange(of: store.generatedVideoArtifact) { _, artifact in
            configurePlayer(for: artifact)
        }
        .onDisappear {
            player?.pause()
        }
    }

    private var runtimeSection: some View {
        InfoGroup(title: "MiniMax H3 Runtime") {
            InfoRow(label: "Status") {
                HStack(spacing: 9) {
                    StatusPill(text: store.videoRuntimeState.title, tone: runtimeTone)
                    Text(store.videoTopologySummary)
                        .foregroundStyle(theme.secondaryText)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
            }

            ForEach(Array(store.videoNodes.enumerated()), id: \.element.id) { rank, node in
                InfoRow(label: "Rank \(rank)") {
                    HStack(spacing: 9) {
                        StatusPill(text: node.isOnline ? "online" : "offline", tone: node.isOnline ? .good : .danger)
                        Text(node.displayName)
                        Text(node.agentURL)
                            .font(.tokenityMono(10))
                            .foregroundStyle(theme.tertiaryText)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer(minLength: 0)
                        if rank == 0 {
                            Text("API · media decode")
                                .foregroundStyle(theme.secondaryText)
                        } else {
                            Text("DiT worker")
                                .foregroundStyle(theme.secondaryText)
                        }
                        StatusPill(
                            text: node.rdma.rdmaEnabled ? "RDMA" : "LAN",
                            tone: node.rdma.rdmaEnabled ? .accent : .warning
                        )
                    }
                }
            }

            if let detail = store.videoRuntimeState.detail {
                InfoRow(label: "Runtime error") {
                    Text(detail)
                        .foregroundStyle(theme.danger)
                        .textSelection(.enabled)
                }
            }

            if !store.videoRuntimeReadinessIssues.isEmpty,
               store.videoRuntimeState != .ready {
                InfoRow(label: "Readiness") {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(store.videoRuntimeReadinessIssues, id: \.self) { issue in
                            Label(issue, systemImage: "exclamationmark.triangle")
                                .foregroundStyle(theme.warning)
                        }
                    }
                }
            }

            InfoRow(label: "Actions") {
                HStack(spacing: 10) {
                    Button {
                        Task { await store.startVideoRuntime() }
                    } label: {
                        Label("Start Video Runtime", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(theme.controlAccent)
                    .disabled(
                        store.videoRuntimeState == .starting
                            || store.videoRuntimeState == .stopping
                            || store.isVideoRuntimeReady
                            || !store.videoRuntimeReadinessIssues.isEmpty
                    )

                    Button(role: .destructive) {
                        Task { await store.stopVideoRuntime() }
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .disabled(store.videoRuntimeState == .stopped || store.videoRuntimeState == .stopping)

                    Button {
                        Task { await store.refreshVideoNodes() }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    Spacer(minLength: 0)
                }
            }

            InfoRow(label: "Configuration") {
                DisclosureGroup(isExpanded: $showsRuntimeConfiguration) {
                    VStack(alignment: .leading, spacing: 12) {
                        LabeledContent("Coordinator Agent") {
                            TextField("Coordinator Node Agent URL", text: $store.h3CoordinatorAgentURL)
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 420)
                        }
                        LabeledContent("Worker Agent") {
                            TextField("Worker Node Agent URL", text: $store.h3WorkerAgentURL)
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 420)
                        }
                        LabeledContent("Model path") {
                            TextField("MiniMax H3 model path", text: $store.h3ModelPath)
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 420)
                        }
                        LabeledContent("Native binary") {
                            TextField("mlx-serve path", text: $store.h3BinaryPath)
                                .textFieldStyle(.roundedBorder)
                                .frame(minWidth: 420)
                        }
                        LabeledContent("Optimization profile") {
                            Picker("Optimization profile", selection: $store.h3OptimizationProfile) {
                                ForEach(H3OptimizationProfile.allCases) { profile in
                                    Text(profile.title).tag(profile)
                                }
                            }
                            .labelsHidden()
                            .frame(width: 250)
                        }
                        Text(store.h3OptimizationProfile.detail)
                            .font(.tokenityText(11))
                            .foregroundStyle(theme.secondaryText)
                    }
                    .disabled(store.videoRuntimeState == .starting || store.isVideoRuntimeReady)
                    .padding(.top, 12)
                } label: {
                    Text("H3 Agent endpoints, paths and validated kernel profile")
                        .foregroundStyle(theme.secondaryText)
                }
            }
        }
    }

    private var generationSection: some View {
        InfoGroup(title: "Generate Video") {
            InfoRow(label: "Prompt") {
                TextEditor(text: $store.videoRequest.prompt)
                    .font(.tokenityText(13))
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .frame(minHeight: 92, maxHeight: 130)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(theme.raisedSurface)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(theme.border, lineWidth: 1)
                    )
                    .disabled(store.isVideoGenerating)
            }

            InfoRow(label: "Canvas") {
                HStack(spacing: 14) {
                    numericField("Width", value: $store.videoRequest.width, range: 128...1_024, step: 32)
                    Text("×").foregroundStyle(theme.tertiaryText)
                    numericField("Height", value: $store.videoRequest.height, range: 128...1_024, step: 32)
                    Spacer(minLength: 0)
                    Button("512 × 256") {
                        store.videoRequest.width = 512
                        store.videoRequest.height = 256
                    }
                    .buttonStyle(.borderless)
                    Button("768 × 512") {
                        store.videoRequest.width = 768
                        store.videoRequest.height = 512
                    }
                    .buttonStyle(.borderless)
                }
            }

            InfoRow(label: "Sampling") {
                HStack(spacing: 18) {
                    numericField("Frames", value: $store.videoRequest.numFrames, range: 5...345, step: 17)
                    numericField("Steps", value: $store.videoRequest.steps, range: 1...50, step: 1)
                    numericField("Seed", value: $store.videoRequest.seed, range: 0...Int.max, step: 1)
                    Toggle("Fast", isOn: $store.videoRequest.fast)
                        .toggleStyle(.switch)
                        .help("Fast uses the H3 step-cache and attention broadcast recipe. Turn it off for final-quality renders.")
                    Spacer(minLength: 0)
                }
            }

            InfoRow(label: "Progress") {
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text(store.videoProgressStage)
                        Spacer()
                        Text("\(Int((store.videoProgress * 100).rounded()))%")
                            .font(.tokenityMono(11))
                            .foregroundStyle(theme.secondaryText)
                    }
                    ProgressView(value: store.videoProgress)
                        .progressViewStyle(.linear)
                }
            }

            if let error = store.videoGenerationError {
                InfoRow(label: "Generation error") {
                    Text(error)
                        .foregroundStyle(theme.danger)
                        .textSelection(.enabled)
                }
            }

            InfoRow(label: "Actions") {
                HStack(spacing: 10) {
                    if store.isVideoGenerating {
                        Button(role: .destructive) {
                            store.cancelVideoGeneration()
                        } label: {
                            Label("Cancel", systemImage: "xmark.circle.fill")
                        }
                    } else {
                        Button {
                            store.beginVideoGeneration()
                        } label: {
                            Label("Generate Video", systemImage: "wand.and.stars")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(theme.controlAccent)
                        .disabled(!store.isVideoRuntimeReady || store.videoRequest.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    Text("Frames are snapped upward to MiniMax H3's 17k+5 ladder.")
                        .font(.tokenityText(11))
                        .foregroundStyle(theme.tertiaryText)
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private var outputSection: some View {
        InfoGroup(title: "Output") {
            if let artifact = store.generatedVideoArtifact {
                InfoRow(label: "Preview") {
                    TokenityVideoPlayer(player: player)
                        .frame(minHeight: 280, idealHeight: 360)
                        .background(Color.black)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                }
                InfoRow(label: "Media") {
                    HStack(spacing: 10) {
                        StatusPill(
                            text: artifact.hasMuxedAudio ? "Video + audio" : "Video",
                            tone: artifact.hasMuxedAudio ? .good : .warning
                        )
                        Text(
                            "\(artifact.frames) frames · \(artifact.width)×\(artifact.height) · "
                                + "\(artifact.fps) fps · \(String(format: "%.2f", artifact.durationSeconds)) s"
                        )
                        Spacer(minLength: 0)
                    }
                }
                InfoRow(label: "Files") {
                    HStack(spacing: 10) {
                        Button("Open Movie") {
                            NSWorkspace.shared.open(artifact.movieURL)
                        }
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([artifact.movieURL])
                        }
                        ShareLink(item: artifact.movieURL) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        Spacer(minLength: 0)
                        Text(artifact.directoryURL.path)
                            .font(.tokenityMono(10))
                            .foregroundStyle(theme.tertiaryText)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }
            } else {
                InfoRow(label: "Preview") {
                    VStack(spacing: 10) {
                        Image(systemName: "film.stack")
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(theme.tertiaryText)
                        Text("Generated video will appear here")
                            .foregroundStyle(theme.secondaryText)
                        Text("Tokenity saves a playable MOV, raw RGB, PCM/WAV audio and metadata for every completed request.")
                            .font(.tokenityText(11))
                            .foregroundStyle(theme.tertiaryText)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, minHeight: 180)
                }
            }
        }
    }

    private var runtimeTone: StatusPill.Tone {
        switch store.videoRuntimeState {
        case .ready: return .good
        case .starting, .stopping: return .warning
        case .failed: return .danger
        case .stopped: return .neutral
        }
    }

    private func configurePlayer(for artifact: GeneratedVideoArtifact?) {
        player?.pause()
        guard let artifact else {
            player = nil
            return
        }
        player = AVPlayer(url: artifact.movieURL)
    }

    private func numericField(
        _ label: String,
        value: Binding<Int>,
        range: ClosedRange<Int>,
        step: Int
    ) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .foregroundStyle(theme.secondaryText)
            TextField("", value: value, format: .number)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(width: label == "Seed" ? 100 : 70)
            Stepper("", value: value, in: range, step: step)
                .labelsHidden()
        }
        .disabled(store.isVideoGenerating)
    }
}

/// Uses AppKit's mature player view directly. SwiftUI.VideoPlayer is backed by
/// the private _AVKit_SwiftUI bridge, which aborts while constructing its
/// responder metadata on macOS 26.5 after a generated movie is inserted into
/// a live view hierarchy.
struct TokenityVideoPlayer: NSViewRepresentable {
    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        view.videoGravity = .resizeAspect
        view.showsFullScreenToggleButton = true
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        guard view.player !== player else { return }
        view.player?.pause()
        view.player = player
    }

    static func dismantleNSView(_ view: AVPlayerView, coordinator: Void) {
        view.player?.pause()
        view.player = nil
    }
}
