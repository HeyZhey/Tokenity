import AppKit
import Foundation
import SwiftUI

enum MarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list([MarkdownListLine])
    case quote(String)
    case divider
    case code(language: String?, text: String)
    case table(headers: [String], rows: [[String]])
}

struct MarkdownListLine: Equatable, Sendable {
    var level: Int
    var marker: String
    var text: String
}

enum StableMarkdownParser {
    static func parse(_ markdown: String) -> [MarkdownBlock] {
        guard !markdown.isEmpty else { return [] }
        let lines = markdown.components(separatedBy: .newlines)
        var blocks: [MarkdownBlock] = []
        var index = 0

        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                index += 1
                continue
            }

            if let fence = fenceStart(trimmed) {
                var codeLines: [String] = []
                index += 1
                while index < lines.count {
                    let candidate = lines[index]
                    if candidate.trimmingCharacters(in: .whitespaces).hasPrefix(fence.marker) {
                        index += 1
                        break
                    }
                    codeLines.append(candidate)
                    index += 1
                }
                blocks.append(.code(language: fence.language, text: codeLines.joined(separator: "\n")))
                continue
            }

            if let heading = heading(in: trimmed) {
                blocks.append(.heading(level: heading.level, text: heading.text))
                index += 1
                continue
            }

            if isDivider(trimmed) {
                blocks.append(.divider)
                index += 1
                continue
            }

            if index + 1 < lines.count,
               line.contains("|"),
               isTableSeparator(lines[index + 1]) {
                let headers = tableCells(line)
                index += 2
                var rows: [[String]] = []
                while index < lines.count, lines[index].contains("|"), !lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                    rows.append(tableCells(lines[index]))
                    index += 1
                }
                blocks.append(.table(headers: headers, rows: rows))
                continue
            }

            if listLine(line) != nil {
                var items: [MarkdownListLine] = []
                while index < lines.count, let item = listLine(lines[index]) {
                    items.append(item)
                    index += 1
                }
                blocks.append(.list(items))
                continue
            }

            if trimmed.hasPrefix(">") {
                var quoteLines: [String] = []
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    guard candidate.hasPrefix(">") else { break }
                    var content = String(candidate.dropFirst())
                    if content.hasPrefix(" ") { content.removeFirst() }
                    quoteLines.append(content)
                    index += 1
                }
                blocks.append(.quote(quoteLines.joined(separator: "\n")))
                continue
            }

            var paragraph: [String] = [line]
            index += 1
            while index < lines.count {
                let candidate = lines[index]
                let candidateTrimmed = candidate.trimmingCharacters(in: .whitespaces)
                if candidateTrimmed.isEmpty { break }
                if fenceStart(candidateTrimmed) != nil || heading(in: candidateTrimmed) != nil ||
                    isDivider(candidateTrimmed) || listLine(candidate) != nil || candidateTrimmed.hasPrefix(">") {
                    break
                }
                if index + 1 < lines.count, candidate.contains("|"), isTableSeparator(lines[index + 1]) {
                    break
                }
                paragraph.append(candidate)
                index += 1
            }
            blocks.append(.paragraph(paragraph.joined(separator: "\n")))
        }
        return blocks
    }

    private static func fenceStart(_ trimmed: String) -> (marker: String, language: String?)? {
        let marker: String
        if trimmed.hasPrefix("```") {
            marker = "```"
        } else if trimmed.hasPrefix("~~~") {
            marker = "~~~"
        } else {
            return nil
        }
        let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespacesAndNewlines)
        return (marker, language.isEmpty ? nil : language)
    }

    private static func heading(in trimmed: String) -> (level: Int, text: String)? {
        let level = trimmed.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level), trimmed.dropFirst(level).hasPrefix(" ") else { return nil }
        return (level, String(trimmed.dropFirst(level + 1)))
    }

    private static func isDivider(_ trimmed: String) -> Bool {
        let compact = trimmed.replacingOccurrences(of: " ", with: "")
        guard compact.count >= 3, let first = compact.first, ["-", "*", "_"].contains(String(first)) else { return false }
        return compact.allSatisfy { $0 == first }
    }

    private static func listLine(_ line: String) -> MarkdownListLine? {
        let indentation = line.prefix(while: { $0 == " " || $0 == "\t" })
        let level = indentation.reduce(0) { $0 + ($1 == "\t" ? 2 : 1) } / 2
        let trimmed = line.dropFirst(indentation.count)
        for marker in ["- ", "* ", "+ "] where trimmed.hasPrefix(marker) {
            return MarkdownListLine(level: level, marker: "•", text: String(trimmed.dropFirst(2)))
        }
        var digitCount = 0
        for character in trimmed {
            guard character.isNumber else { break }
            digitCount += 1
        }
        guard digitCount > 0 else { return nil }
        let suffix = trimmed.dropFirst(digitCount)
        guard suffix.hasPrefix(". ") else { return nil }
        let number = String(trimmed.prefix(digitCount))
        return MarkdownListLine(level: level, marker: "\(number).", text: String(suffix.dropFirst(2)))
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let cells = tableCells(line)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let compact = cell.replacingOccurrences(of: " ", with: "")
            let core = compact.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            return core.count >= 3 && core.allSatisfy { $0 == "-" }
        }
    }

    private static func tableCells(_ line: String) -> [String] {
        var trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("|") { trimmed.removeFirst() }
        if trimmed.hasSuffix("|") { trimmed.removeLast() }
        return trimmed.split(separator: "|", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
    }
}

@MainActor
final class MarkdownRenderModel: ObservableObject {
    @Published private(set) var blocks: [MarkdownBlock]
    private var source: String
    private var renderedSource: String
    private var parseTask: Task<Void, Never>?

    init(source: String) {
        self.source = source
        renderedSource = source
        blocks = StableMarkdownParser.parse(source)
    }

    func update(source: String) {
        guard source != self.source else { return }
        self.source = source
        guard parseTask == nil else { return }

        // Throttle to one parse per frame-sized interval, but never restart the
        // delay for every token. A cancelling debounce starves during a steady
        // stream and makes the answer appear only when the stream pauses.
        parseTask = Task { [weak self] in
            while let self {
                do {
                    try await Task.sleep(for: .milliseconds(33))
                } catch {
                    self.parseTask = nil
                    return
                }
                guard !Task.isCancelled else {
                    self.parseTask = nil
                    return
                }

                let snapshot = self.source
                if snapshot != self.renderedSource {
                    let parsed = await Task.detached(priority: .userInitiated) {
                        StableMarkdownParser.parse(snapshot)
                    }.value
                    guard !Task.isCancelled else {
                        self.parseTask = nil
                        return
                    }
                    self.blocks = parsed
                    self.renderedSource = snapshot
                }
                if self.source == snapshot {
                    self.parseTask = nil
                    return
                }
            }
        }
    }
}

struct MarkdownMessageView: View {
    let markdown: String
    @StateObject private var model: MarkdownRenderModel
    @Environment(\.tokenityTheme) private var theme

    init(markdown: String) {
        self.markdown = markdown
        _model = StateObject(wrappedValue: MarkdownRenderModel(source: markdown))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(model.blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onChange(of: markdown) { _, newValue in
            model.update(source: newValue)
        }
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let text):
            InlineMarkdownText(text: text)
                .font(headingFont(level))
                .padding(.top, level <= 2 ? 8 : 3)
        case .paragraph(let text):
            InlineMarkdownText(text: text)
                .font(.tokenityText(14))
                .lineSpacing(3)
        case .list(let items):
            VStack(alignment: .leading, spacing: 7) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.marker)
                            .font(.tokenityText(13, weight: .medium))
                            .foregroundStyle(theme.secondaryText)
                            .frame(minWidth: 15, alignment: .trailing)
                        InlineMarkdownText(text: item.text)
                            .font(.tokenityText(14))
                    }
                    .padding(.leading, CGFloat(item.level) * 20)
                }
            }
        case .quote(let text):
            HStack(alignment: .top, spacing: 12) {
                RoundedRectangle(cornerRadius: 1)
                    .fill(theme.accent.opacity(0.55))
                    .frame(width: 3)
                InlineMarkdownText(text: text)
                    .font(.tokenityText(14))
                    .foregroundStyle(theme.secondaryText)
                    .lineSpacing(3)
            }
            .padding(.vertical, 4)
        case .divider:
            Divider().padding(.vertical, 5)
        case .code(let language, let text):
            CodeBlockView(language: language, code: text)
        case .table(let headers, let rows):
            MarkdownTableView(headers: headers, rows: rows)
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .tokenityText(25, weight: .bold)
        case 2: return .tokenityText(21, weight: .bold)
        case 3: return .tokenityText(18, weight: .semibold)
        case 4: return .tokenityText(16, weight: .semibold)
        case 5: return .tokenityText(14, weight: .semibold)
        default: return .tokenityText(13, weight: .semibold)
        }
    }
}

private struct InlineMarkdownText: View {
    let text: String

    var body: some View {
        Text(Self.attributed(text))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
    }

    private static func attributed(_ source: String) -> AttributedString {
        var result = AttributedString()
        var remaining = source[...]
        while let opening = remaining.range(of: "~~"),
              let closing = remaining[opening.upperBound...].range(of: "~~") {
            result.append(native(String(remaining[..<opening.lowerBound])))
            var struck = native(String(remaining[opening.upperBound..<closing.lowerBound]))
            struck.strikethroughStyle = Text.LineStyle(pattern: .solid)
            result.append(struck)
            remaining = remaining[closing.upperBound...]
        }
        result.append(native(String(remaining)))
        return result
    }

    private static func native(_ source: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
        var attributed = (try? AttributedString(markdown: source, options: options)) ?? AttributedString(source)
        let codeRanges = attributed.runs.compactMap { run -> Range<AttributedString.Index>? in
            run.inlinePresentationIntent?.contains(.code) == true ? run.range : nil
        }
        for range in codeRanges {
            attributed[range].font = .system(size: 13, design: .monospaced)
            attributed[range].backgroundColor = Color(nsColor: .quaternaryLabelColor).opacity(0.13)
        }
        addAutomaticLinks(to: &attributed)
        return attributed
    }

    private static func addAutomaticLinks(to attributed: inout AttributedString) {
        let plain = String(attributed.characters)
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue) else { return }
        let range = NSRange(plain.startIndex..<plain.endIndex, in: plain)
        for match in detector.matches(in: plain, range: range) {
            guard let url = match.url,
                  let swiftRange = Range(match.range, in: plain),
                  let lower = AttributedString.Index(swiftRange.lowerBound, within: attributed),
                  let upper = AttributedString.Index(swiftRange.upperBound, within: attributed)
            else { continue }
            attributed[lower..<upper].link = url
        }
    }
}

private struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]
    @Environment(\.tokenityTheme) private var theme

    var body: some View {
        ScrollView(.horizontal) {
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                tableRow(headers, isHeader: true)
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    tableRow(row, isHeader: false, shaded: index.isMultiple(of: 2))
                }
            }
            .overlay(Rectangle().stroke(theme.border, lineWidth: 0.5))
        }
        .scrollIndicators(.visible)
    }

    private func tableRow(_ values: [String], isHeader: Bool, shaded: Bool = false) -> some View {
        GridRow {
            ForEach(0..<max(headers.count, values.count), id: \.self) { index in
                InlineMarkdownText(text: index < values.count ? values[index] : "")
                    .font(.tokenityText(13, weight: isHeader ? .semibold : .regular))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .frame(minWidth: 110, maxWidth: 260, alignment: .leading)
                    .background(isHeader ? theme.group : (shaded ? theme.group.opacity(0.55) : Color.clear))
                    .overlay(alignment: .trailing) {
                        Rectangle().fill(theme.border).frame(width: 0.5)
                    }
                    .overlay(alignment: .bottom) {
                        Rectangle().fill(theme.border).frame(height: 0.5)
                    }
            }
        }
    }
}

@MainActor
private final class CodeHighlightModel: ObservableObject {
    @Published private(set) var highlighted: AttributedString
    private var source: String
    private let language: String?

    init(code: String, language: String?) {
        source = code
        self.language = language
        highlighted = SyntaxHighlighter.highlight(code, language: language)
    }

    func update(code: String) {
        guard code != source else { return }
        source = code
        highlighted = SyntaxHighlighter.highlight(code, language: language)
    }
}

struct CodeBlockView: View {
    let language: String?
    let code: String
    @StateObject private var model: CodeHighlightModel
    @State private var didCopy = false
    @Environment(\.tokenityTheme) private var theme

    init(language: String?, code: String) {
        self.language = language
        self.code = code
        _model = StateObject(wrappedValue: CodeHighlightModel(code: code, language: language))
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text(language?.isEmpty == false ? language! : "Plain text")
                    .font(.tokenityMono(11, weight: .medium))
                    .foregroundStyle(theme.secondaryText)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    didCopy = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.4))
                        didCopy = false
                    }
                } label: {
                    Label(didCopy ? "Copied" : "Copy", systemImage: didCopy ? "checkmark" : "doc.on.doc")
                        .font(.tokenityText(11, weight: .medium))
                }
                .buttonStyle(.borderless)
                .accessibilityLabel(didCopy ? "Code copied" : "Copy code")
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 8)
            .background(theme.group)

            Divider()

            ScrollView(.horizontal) {
                Text(model.highlighted)
                    .font(.tokenityMono(12))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: true, vertical: true)
                    .padding(12)
            }
            .scrollIndicators(.visible)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(theme.code)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(theme.border, lineWidth: 0.5))
        .onChange(of: code) { _, newValue in model.update(code: newValue) }
    }
}

private enum SyntaxHighlighter {
    private static let knownLanguages: Set<String> = [
        "swift", "python", "py", "javascript", "js", "typescript", "ts", "json", "bash", "sh", "zsh", "c", "cpp", "c++", "java", "rust", "go"
    ]

    static func highlight(_ code: String, language: String?) -> AttributedString {
        var output = AttributedString(code)
        output.font = .system(size: 12, design: .monospaced)
        guard let language = language?.lowercased(), knownLanguages.contains(language) else { return output }

        apply(#"\b(class|struct|enum|protocol|extension|func|let|var|if|else|for|while|return|throw|throws|try|await|async|import|from|def|in|is|not|and|or|true|false|null|nil|self|public|private|internal|static|const|function|interface|type|package|use|fn|match)\b"#, color: .purple, to: &output, source: code)
        apply(#"\b\d+(?:\.\d+)?\b"#, color: .blue, to: &output, source: code)
        apply(#"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"#, color: .red, to: &output, source: code)
        apply(#"(?m)//.*$|#.*$|/\*[\s\S]*?\*/"#, color: .green, to: &output, source: code)
        return output
    }

    private static func apply(
        _ pattern: String,
        color: Color,
        to output: inout AttributedString,
        source: String
    ) {
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return }
        let searchRange = NSRange(source.startIndex..<source.endIndex, in: source)
        for match in expression.matches(in: source, range: searchRange) {
            guard let swiftRange = Range(match.range, in: source),
                  let lower = AttributedString.Index(swiftRange.lowerBound, within: output),
                  let upper = AttributedString.Index(swiftRange.upperBound, within: output)
            else { continue }
            output[lower..<upper].foregroundColor = color
        }
    }
}
