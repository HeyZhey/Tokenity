import SwiftUI
import Foundation

struct StatusPill: View {
    enum Tone {
        case neutral
        case good
        case warning
        case danger
        case accent
    }

    let text: String
    let tone: Tone

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(text)
                .font(.tokenityText(11, weight: .medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.13), in: Capsule())
        .foregroundStyle(color)
        .accessibilityLabel(text)
    }

    private var color: Color {
        switch tone {
        case .neutral: return theme.secondaryText
        case .good: return theme.success
        case .warning: return theme.warning
        case .danger: return theme.danger
        case .accent: return theme.accent
        }
    }
}

struct InfoGroup<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.tokenityText(11, weight: .semibold))
                .foregroundStyle(theme.tertiaryText)
            VStack(spacing: 0) {
                content
            }
            .background(theme.group, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(theme.border.opacity(0.8), lineWidth: 0.5)
            )
        }
    }
}

struct InfoRow<Trailing: View>: View {
    let label: String
    @ViewBuilder var trailing: Trailing

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        HStack(spacing: 12) {
            Text(label)
                .font(.tokenityText(13))
                .foregroundStyle(theme.secondaryText)
                .frame(width: 160, alignment: .leading)
            trailing
                .font(.tokenityText(13))
                .foregroundStyle(theme.text)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.rowSeparator)
                .frame(height: 0.5)
                .padding(.leading, 12)
        }
    }
}

struct MemoryUsageBar: View {
    let memory: MemoryStats

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(theme.border.opacity(0.35))
                    Capsule()
                        .fill(theme.secondaryText.opacity(0.22))
                        .frame(width: proxy.size.width * CGFloat(clampedPhysicalRatio))
                    Capsule()
                        .fill(toneColor)
                        .frame(width: proxy.size.width * CGFloat(clampedInUseRatio))
                }
            }
            .frame(height: 6)

            HStack(spacing: 6) {
                Text(inUseText)
                    .font(.tokenityText(11, weight: .medium))
                    .foregroundStyle(theme.secondaryText)
                Spacer(minLength: 4)
                Text(residentText)
                    .font(.tokenityText(11))
                    .foregroundStyle(theme.tertiaryText)
                    .lineLimit(1)
            }

            if !detailText.isEmpty {
                Text(detailText)
                    .font(.tokenityText(10))
                    .foregroundStyle(theme.tertiaryText)
                    .lineLimit(1)
            }
        }
        .help(helpText)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(helpText)
    }

    private var clampedInUseRatio: Double {
        min(max(memory.inUseRatio ?? memory.usedRatio ?? 0, 0), 1)
    }

    private var clampedPhysicalRatio: Double {
        min(max(memory.physicalUsedRatio ?? memory.usedRatio ?? clampedInUseRatio, clampedInUseRatio), 1)
    }

    private var toneColor: Color {
        if clampedInUseRatio >= 0.9 { return theme.danger }
        if clampedInUseRatio >= 0.75 { return theme.warning }
        return theme.success
    }

    private var inUseText: String {
        guard let ratio = memory.inUseRatio ?? memory.usedRatio else { return "Memory unknown" }
        return "In use \(formatPercent(ratio))"
    }

    private var residentText: String {
        guard memory.physicalUsedRatio != nil || memory.usedRatio != nil else { return "" }
        return "Resident \(formatPercent(clampedPhysicalRatio))"
    }

    private var detailText: String {
        guard let inUse = memory.inUseBytes ?? memory.usedBytes else { return "" }
        var value = "\(formatBytes(inUse)) in use"
        if let cache = memory.reclaimableBytes, cache > 0 {
            value += " · \(formatBytes(cache)) reclaimable cache"
        }
        return value
    }

    private var helpText: String {
        var parts = [inUseText]
        if !residentText.isEmpty { parts.append(residentText + " including cache") }
        if !detailText.isEmpty { parts.append(detailText) }
        if let pressure = memory.pressureAvailableRatio {
            parts.append("\(formatPercent(pressure)) pressure headroom")
        }
        return parts.joined(separator: ", ")
    }

    private func formatPercent(_ ratio: Double) -> String {
        "\(Int((ratio * 100).rounded()))%"
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824
        if gib >= 10 {
            return String(format: "%.0f GB", gib)
        }
        return String(format: "%.1f GB", gib)
    }
}

struct CodeBlock: View {
    let text: String

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            Text(text)
                .font(.tokenityMono(12))
                .foregroundStyle(theme.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
        .frame(minHeight: 120)
        .background(theme.code, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.border.opacity(0.9), lineWidth: 0.5)
        )
    }
}

struct OperationLog: View {
    let lines: [String]

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(theme.accent.opacity(0.65))
                            .frame(width: 6, height: 6)
                            .padding(.top, 6)
                        Text(line)
                            .font(.tokenityText(13))
                            .foregroundStyle(theme.text)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(theme.group, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(theme.border.opacity(0.9), lineWidth: 0.5)
        )
    }
}

struct PageScaffold<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(title)
                    .font(.tokenityText(20, weight: .semibold))
                Spacer()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(theme.window)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    content
                }
                .frame(maxWidth: 980, alignment: .leading)
                .padding(24)
            }
            .scrollIndicators(.visible)
            .contentShape(Rectangle())
            .background(theme.window)
        }
    }
}
