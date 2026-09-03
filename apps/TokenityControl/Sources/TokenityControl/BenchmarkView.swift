import AppKit
import SwiftUI

struct BenchmarkPage: View {
    @EnvironmentObject private var store: TokenityStore

    var body: some View {
        BenchmarkView(runner: store.benchmarkRunner)
    }
}

struct BenchmarkView: View {
    @EnvironmentObject private var store: TokenityStore
    @ObservedObject var runner: BenchmarkRunner
    @Environment(\.tokenityTheme) private var theme
    @State private var configuration: BenchmarkConfiguration

    init(
        runner: BenchmarkRunner,
        configuration: BenchmarkConfiguration = .language()
    ) {
        self.runner = runner
        _configuration = State(initialValue: configuration)
    }

    var body: some View {
        PageScaffold(title: "Benchmark") {
            benchmarkTypeSection
            targetSection
            modelSection
            configurationSection
            runSection
            resultSection
        }
        .onAppear(perform: synchronizeSelection)
        .onChange(of: store.modelLibraryRows) { _, _ in synchronizeSelection() }
    }

    private var benchmarkTypeSection: some View {
        InfoGroup(title: "Benchmark Type") {
            InfoRow(label: "Workload") {
                Picker("Workload", selection: kindBinding) {
                    ForEach(BenchmarkKind.allCases) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 430)
                .disabled(runner.isActive)
            }
        }
    }

    private var targetSection: some View {
        InfoGroup(title: "Target Environment") {
            InfoRow(label: "Topology") {
                Picker("Topology", selection: $configuration.target) {
                    ForEach(BenchmarkTarget.allCases) { target in
                        Text(target.title).tag(target)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .disabled(runner.isActive)
                .onChange(of: configuration.target) { _, _ in synchronizeSelection() }
            }

            if configuration.target == .selectedNode {
                InfoRow(label: "Selected Mac") {
                    Picker("Selected Mac", selection: targetNodeBinding) {
                        ForEach(store.selectedNodes) { node in
                            Text(node.displayName).tag(node.id)
                        }
                    }
                    .labelsHidden()
                    .disabled(runner.isActive)
                }
            }

            ForEach(Array(targetNodes.enumerated()), id: \.element.id) { index, node in
                InfoRow(label: configuration.target == .tensorParallel2 ? "Rank \(index)" : "Mac") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            StatusPill(text: node.isOnline ? "online" : "offline", tone: node.isOnline ? .good : .danger)
                            Text(node.displayName)
                            Spacer(minLength: 8)
                            Text(connectionText(for: node))
                                .foregroundStyle(theme.secondaryText)
                        }
                        HStack(spacing: 10) {
                            Text(node.memoryUsageText)
                            Text("·")
                            Text(runtimeReadyText)
                        }
                        .font(.tokenityText(11))
                        .foregroundStyle(theme.tertiaryText)
                    }
                }
            }
        }
    }

    private var modelSection: some View {
        InfoGroup(title: "Model") {
            InfoRow(label: "Inventory") {
                HStack(spacing: 10) {
                    Picker("Model", selection: $configuration.modelID) {
                        if eligibleModels.isEmpty {
                            Text("No compatible model").tag("")
                        } else {
                            ForEach(eligibleModels) { model in
                                Text(model.displayName).tag(model.id)
                            }
                        }
                    }
                    .labelsHidden()
                    .disabled(runner.isActive || eligibleModels.isEmpty)
                    Spacer(minLength: 0)
                    Button {
                        Task { await store.scanModels() }
                    } label: {
                        Label(store.isScanningModels ? "Scanning" : "Scan Models", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.borderless)
                    .disabled(store.isScanningModels || runner.isActive)
                }
            }
            if let model = selectedModel {
                InfoRow(label: "Details") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Text(model.id).font(.tokenityMono(11))
                            StatusPill(text: model.loadState.rawValue, tone: model.loadState == .loaded ? .good : .neutral)
                        }
                        Text([
                            model.sizeText,
                            model.quantization ?? "Quantization unknown",
                            model.availability,
                            model.revision.map { "revision \($0)" } ?? "revision unavailable",
                        ].joined(separator: " · "))
                        .font(.tokenityText(11))
                        .foregroundStyle(theme.secondaryText)
                    }
                }
            }
            if let issue = selectionIssue {
                InfoRow(label: "Availability") {
                    Label(issue, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(theme.warning)
                }
            }
        }
    }

    private var configurationSection: some View {
        InfoGroup(title: "Test Configuration") {
            InfoRow(label: "Profile") {
                Picker("Profile", selection: $configuration.profile) {
                    ForEach(BenchmarkProfile.allCases) { profile in
                        Text(profile.title).tag(profile)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(maxWidth: 430)
                .disabled(runner.isActive)
            }

            InfoRow(label: "Runs") {
                HStack(spacing: 12) {
                    if configuration.kind == .languageModel {
                        Text("\(configuration.warmupRuns) warmup + \(configuration.measuredRuns) measured per size")
                        StatusPill(text: "warmup excluded", tone: .neutral)
                        Text("\(configuration.totalRuns) total")
                            .font(.tokenityMono(10.5))
                            .foregroundStyle(theme.secondaryText)
                    } else {
                        Text(videoRunDescription)
                        StatusPill(text: "Cold/Warm separate", tone: .accent)
                    }
                    Spacer(minLength: 0)
                    if configuration.profile == .custom {
                        Stepper("\(configuration.customRuns) measured", value: $configuration.customRuns, in: 1...20)
                            .fixedSize()
                            .disabled(runner.isActive)
                    }
                }
            }

            if configuration.kind == .languageModel {
                InfoRow(label: "Input size") {
                    inputSizeSelector
                }
                InfoRow(label: "Generation") {
                    Text("Concurrency 1 · Temperature 0 · Max \(configuration.maximumOutputTokens) tokens")
                }
            } else {
                videoConfigurationRows
            }

            InfoRow(label: "Prompt") {
                VStack(alignment: .leading, spacing: 6) {
                    Text(configuration.kind == .languageModel
                        ? "Deterministic payload generated for every selected target size; server-reported prompt tokens are recorded as the actual input."
                        : configuration.prompt)
                        .lineLimit(4)
                        .textSelection(.enabled)
                    Text("Fixed profile: \(configuration.promptProfile)")
                        .font(.tokenityMono(10))
                        .foregroundStyle(theme.tertiaryText)
                }
            }
        }
    }

    private var inputSizeSelector: some View {
        HStack(spacing: 0) {
            ForEach(Array(BenchmarkLLMInputSize.allCases.enumerated()), id: \.element.id) { index, preset in
                let selected = configuration.selectedInputTokenSizes.contains(preset.tokenCount)
                Button {
                    toggleInputSize(preset.tokenCount)
                } label: {
                    Text(preset.title)
                        .font(.tokenityMono(10.5))
                        .foregroundStyle(selected ? Color.white : theme.secondaryText)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(selected ? theme.controlAccent : theme.raisedSurface)
                .overlay(alignment: .trailing) {
                    if index < BenchmarkLLMInputSize.allCases.count - 1 {
                        Rectangle()
                            .fill(theme.border)
                            .frame(width: 1)
                    }
                }
                .disabled(runner.isActive)
                .help("Toggle a target input of \(preset.tokenCount.formatted()) tokens")
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(theme.border, lineWidth: 1)
        }
        .frame(maxWidth: 620)
    }

    @ViewBuilder
    private var videoConfigurationRows: some View {
        InfoRow(label: "Resolution") {
            Picker("Resolution", selection: resolutionBinding) {
                ForEach(BenchmarkResolutionPreset.h3Presets) { preset in
                    Text(preset.title).tag(preset)
                }
            }
            .labelsHidden()
            .disabled(runner.isActive)
        }
        InfoRow(label: "Duration") {
            Picker("Duration", selection: durationBinding) {
                ForEach(BenchmarkDurationPreset.h3Presets) { preset in
                    Text(preset.title).tag(preset)
                }
            }
            .labelsHidden()
            .disabled(runner.isActive)
        }
        InfoRow(label: "Sampling") {
            HStack(spacing: 18) {
                Text("\(configuration.steps) steps")
                Text("Seed \(configuration.seed)")
                Text(configuration.fast ? "Fast on" : "Fast off")
                Spacer(minLength: 0)
                Toggle("Save one preview", isOn: $configuration.saveVideoPreview)
                    .toggleStyle(.switch)
                    .disabled(runner.isActive)
            }
        }
    }

    private var runSection: some View {
        InfoGroup(title: "Run") {
            InfoRow(label: "Actions") {
                HStack(spacing: 10) {
                    Button {
                        var submitted = configuration
                        submitted.modelRevision = selectedModel?.revision
                        submitted.quantization = selectedModel?.quantization
                        runner.start(submitted)
                    } label: {
                        Label(runButtonTitle, systemImage: "gauge.with.needle")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .tint(theme.controlAccent)
                    .disabled(!canRun)

                    Button(role: .destructive) {
                        runner.cancel()
                    } label: {
                        Label("Cancel", systemImage: "xmark.circle.fill")
                    }
                    .disabled(!runner.isActive || runner.state == .cancelling)

                    Spacer(minLength: 0)
                    StatusPill(text: runner.state.title, tone: phaseTone)
                }
            }
            InfoRow(label: "Current stage") {
                Text(runner.currentStage)
            }
            InfoRow(label: "Progress") {
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text("\(runner.completedRuns) / \(runner.totalRuns) runs")
                        Spacer()
                        Text(durationText(runner.elapsedSeconds))
                            .font(.tokenityMono(11))
                            .foregroundStyle(theme.secondaryText)
                    }
                    ProgressView(value: runner.progress)
                }
            }
            ForEach(runner.rankStatuses.keys.sorted(), id: \.self) { rank in
                InfoRow(label: rank) {
                    Text(runner.rankStatuses[rank] ?? "Waiting")
                }
            }
            if let error = runner.errorMessage {
                InfoRow(label: "Error") {
                    Text(error).foregroundStyle(theme.danger).textSelection(.enabled)
                }
            }
        }
    }

    private var resultSection: some View {
        InfoGroup(title: "Results") {
            if let result = runner.currentResult {
                InfoRow(label: "Summary") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("\(result.configuration.modelID) · \(result.topology) · \(result.samples.filter(\.isMeasuredSuccess).count) successful measured samples")
                        if let load = result.modelLoadMilliseconds {
                            Text("Model load \(durationText(load / 1_000)) · excluded from TTFT and end-to-end timing")
                                .font(.tokenityText(11))
                                .foregroundStyle(theme.secondaryText)
                        }
                        if let speedup = runner.matchingSpeedup(for: result) {
                            StatusPill(text: String(format: "Matched TP2 speedup %.2f×", speedup), tone: .good)
                        } else {
                            Text("Speedup unavailable until an identical single-Mac/TP2 configuration pair exists.")
                                .font(.tokenityText(11))
                                .foregroundStyle(theme.tertiaryText)
                        }
                    }
                }
                ForEach(result.summaries.indices, id: \.self) { index in
                    summaryRow(result.summaries[index], kind: result.configuration.kind)
                }
                InfoRow(label: "Runs") {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(result.samples) { sample in
                            sampleRow(sample)
                        }
                    }
                }
                InfoRow(label: "Export") {
                    HStack(spacing: 10) {
                        Button("Export JSON") { export(result, format: "json") }
                        Button("Export CSV") { export(result, format: "csv") }
                        Spacer()
                        Text("schema_version 1 · latest 20 retained")
                            .font(.tokenityText(11))
                            .foregroundStyle(theme.tertiaryText)
                    }
                }
            } else if let latest = runner.history.first {
                InfoRow(label: "Latest saved") {
                    Text("\(latest.configuration.modelID) · \(latest.topology) · \(latest.completedAt.formatted())")
                }
            } else {
                InfoRow(label: "Status") {
                    Text("Run a benchmark to see Cold and Warm results here.")
                        .foregroundStyle(theme.secondaryText)
                }
            }
        }
    }

    private func summaryRow(_ summary: BenchmarkThermalSummary, kind: BenchmarkKind) -> some View {
        let inputLabel = summary.requestedInputTokens.map {
            "\(BenchmarkLLMInputSize.title(for: $0)) · \(summary.thermalState.rawValue)"
        } ?? summary.thermalState.rawValue
        return InfoRow(label: inputLabel) {
            VStack(alignment: .leading, spacing: 6) {
                if kind == .languageModel {
                    metricLine("TTFT", summary.ttftMilliseconds, unit: "ms")
                    metricLine("Prefill", summary.prefillTokensPerSecond, unit: "tok/s")
                    metricLine("Decode", summary.decodeTokensPerSecond, unit: "tok/s")
                } else {
                    metricLine("Sampling", summary.samplingMilliseconds, unit: "ms")
                    metricLine("Generation", summary.framesPerSecond, unit: "frames/s")
                }
                metricLine("End to end", summary.totalMilliseconds, unit: "ms")
            }
        }
    }

    private func metricLine(_ name: String, _ metric: BenchmarkMetricSummary, unit: String) -> some View {
        HStack(spacing: 12) {
            Text(name).frame(width: 82, alignment: .leading)
            Text(metric.mean.map { "Mean \(format($0)) \(unit)" } ?? "Unavailable")
            Text(metric.p50.map { "p50 \(format($0))" } ?? "p50 —")
            Text(metric.p95.map { "p95 \(format($0))" } ?? "p95 needs 5 samples")
                .foregroundStyle(theme.secondaryText)
            Spacer(minLength: 0)
        }
        .font(.tokenityMono(10.5))
    }

    private func sampleRow(_ sample: BenchmarkRunSample) -> some View {
        HStack(alignment: .top, spacing: 10) {
            StatusPill(text: sample.thermalState.rawValue, tone: sample.thermalState == .cold ? .warning : .accent)
            if sample.isWarmup { StatusPill(text: "excluded", tone: .neutral) }
            Text("#\(sample.runIndex)").font(.tokenityMono(10.5))
            Text(sampleDetail(sample))
                .font(.tokenityMono(10.5))
                .foregroundStyle(sample.status == .succeeded ? theme.secondaryText : theme.danger)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }

    private var eligibleModels: [ModelLibraryRow] {
        BenchmarkModelFilter.eligibleModels(in: store.modelLibraryRows, for: configuration)
    }

    private var selectedModel: ModelLibraryRow? {
        eligibleModels.first { $0.id == configuration.modelID }
    }

    private var targetNodes: [TokenityNode] {
        switch configuration.target {
        case .currentMac:
            // TODO(BUG): Recover the Current Mac target independently of cluster selection.
            // Repro: let the Node Agent go offline until selection pruning clears the
            // coordinator, then restart it; discovery sees the healthy local Agent but
            // Benchmark remains blocked on "Choose an online Mac" until manual selection.
            return Array([store.coordinator].compactMap { $0 })
        case .selectedNode:
            return store.selectedNodes.filter { $0.id == configuration.targetNodeID }
        case .tensorParallel2:
            return Array(store.selectedNodes.prefix(2))
        }
    }

    private var selectionIssue: String? {
        if targetNodes.count != configuration.target.requiredNodeCount {
            return configuration.target == .tensorParallel2
                ? "TP2 requires exactly two selected, online Macs."
                : "Choose an online Mac."
        }
        if targetNodes.contains(where: { !$0.isOnline }) { return "Every target Mac must be online." }
        if selectedModel == nil { return "No model in the current inventory satisfies this topology." }
        return configuration.validationIssue
    }

    private var canRun: Bool {
        !runner.isActive
            && !store.isChatRunning
            && !store.isVideoGenerating
            && !store.isModelTransitioning
            && selectionIssue == nil
    }

    private var runButtonTitle: String {
        selectedModel?.loadState == .loaded ? "Run Benchmark" : "Load and Run"
    }

    private var runtimeReadyText: String {
        if configuration.kind == .videoGeneration {
            return store.isVideoRuntimeReady ? "Runtime Ready" : "Runtime will be loaded"
        }
        let ready = store.residentModelInstances.contains { instance in
            instance.modelID == configuration.modelID && (instance.isReady || instance.isBusy)
        }
        return ready ? "Runtime Ready" : "Runtime will be loaded"
    }

    private var videoRunDescription: String {
        switch configuration.profile {
        case .quick: return "1 Cold generation"
        case .standard: return "1 Cold + 3 Warm generations"
        case .custom:
            return "1 Cold + \(max(0, configuration.measuredRuns - 1)) Warm generations"
        }
    }

    private var phaseTone: StatusPill.Tone {
        switch runner.state {
        case .completed: return .good
        case .failed: return .danger
        case .preflight, .loading, .warmup, .running, .cancelling: return .warning
        case .idle: return .neutral
        }
    }

    private var kindBinding: Binding<BenchmarkKind> {
        Binding(
            get: { configuration.kind },
            set: { kind in
                let target = configuration.target
                let targetNodeID = configuration.targetNodeID
                configuration = kind == .languageModel ? .language() : .video()
                configuration.target = target
                configuration.targetNodeID = targetNodeID
                synchronizeSelection()
            }
        )
    }

    private var targetNodeBinding: Binding<String> {
        Binding(
            get: { configuration.targetNodeID ?? store.selectedNodes.first?.id ?? "" },
            set: { configuration.targetNodeID = $0 }
        )
    }

    private var resolutionBinding: Binding<BenchmarkResolutionPreset> {
        Binding(
            get: { BenchmarkResolutionPreset(width: configuration.width, height: configuration.height) },
            set: { configuration.width = $0.width; configuration.height = $0.height }
        )
    }

    private var durationBinding: Binding<BenchmarkDurationPreset> {
        Binding(
            get: {
                BenchmarkDurationPreset.h3Presets.first { $0.frames == configuration.frames }
                    ?? BenchmarkDurationPreset(name: "Custom", frames: configuration.frames)
            },
            set: { configuration.frames = $0.frames }
        )
    }

    private func toggleInputSize(_ tokenCount: Int) {
        var selected = Set(configuration.selectedInputTokenSizes)
        if selected.contains(tokenCount) {
            guard selected.count > 1 else { return }
            selected.remove(tokenCount)
        } else {
            selected.insert(tokenCount)
        }
        configuration.inputTokenSizes = selected.sorted()
    }

    private func synchronizeSelection() {
        if configuration.targetNodeID == nil {
            configuration.targetNodeID = store.selectedNodes.first?.id
        }
        if !eligibleModels.contains(where: { $0.id == configuration.modelID }) {
            configuration.modelID = eligibleModels.first?.id ?? ""
        }
    }

    private func connectionText(for node: TokenityNode) -> String {
        if configuration.target == .tensorParallel2 {
            return node.rdma.rdmaEnabled ? "Thunderbolt RDMA" : "Standard network"
        }
        return node.agentURL.isEmpty ? "Agent unavailable" : "Node Agent"
    }

    private func sampleDetail(_ sample: BenchmarkRunSample) -> String {
        if sample.status != .succeeded { return sample.error ?? sample.status.rawValue }
        if sample.kind == .languageModel {
            let prefill = sample.prefillTokensPerSecond.map { "prefill \(format($0)) tok/s" } ?? "prefill unavailable"
            let input = sample.requestedInputTokens.map {
                "\(BenchmarkLLMInputSize.title(for: $0)) target / \(sample.promptTokens?.formatted() ?? "—") actual"
            } ?? "\(sample.promptTokens?.formatted() ?? "—") input tokens"
            return "\(input) · TTFT \(sample.ttftMilliseconds.map { format($0) } ?? "—") ms · \(prefill) · decode \(sample.decodeTokensPerSecond.map { format($0) } ?? "—") tok/s · request \(sample.requestID ?? "unavailable")"
        }
        return "\(sample.width ?? 0)×\(sample.height ?? 0) · \(sample.frames ?? 0) frames @ \(sample.actualFPS ?? 0) fps · \(sample.framesPerSecond.map { format($0) } ?? "—") frames/s"
    }

    private func format(_ value: Double) -> String {
        if abs(value) >= 100 { return String(format: "%.0f", value) }
        return String(format: "%.2f", value)
    }

    private func durationText(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        return String(format: "%d:%02d", Int(seconds) / 60, Int(seconds) % 60)
    }

    private func export(_ result: BenchmarkResult, format: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = format == "json" ? [.json] : [.commaSeparatedText]
        panel.nameFieldStringValue = "tokenity-benchmark-\(result.id.uuidString).\(format)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            if format == "json" {
                try runner.exportJSON(result, to: url)
            } else {
                try runner.exportCSV(result, to: url)
            }
        } catch {
            runner.errorMessage = "Export failed: \(error.localizedDescription)"
        }
    }
}
