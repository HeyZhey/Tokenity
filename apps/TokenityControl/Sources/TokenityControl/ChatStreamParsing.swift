import Foundation

struct ChatStreamFragment: Equatable {
    var reasoning = ""
    var answer = ""

    mutating func append(_ other: ChatStreamFragment) {
        reasoning += other.reasoning
        answer += other.answer
    }
}

/// Incrementally separates `<think>...</think>` from normal content without
/// exposing a partial tag when the server splits it across SSE chunks.
struct ThinkingTagStreamParser {
    private enum Mode {
        case answer
        case reasoning
    }

    private var mode: Mode = .answer
    private var pendingTagPrefix = ""

    var isInsideThinking: Bool { mode == .reasoning }

    mutating func consume(_ chunk: String) -> ChatStreamFragment {
        var fragment = ChatStreamFragment()
        var remaining = pendingTagPrefix + chunk
        pendingTagPrefix = ""

        while !remaining.isEmpty {
            let tag = mode == .answer ? "<think>" : "</think>"
            if let range = remaining.range(of: tag) {
                append(String(remaining[..<range.lowerBound]), to: &fragment)
                remaining = String(remaining[range.upperBound...])
                mode = mode == .answer ? .reasoning : .answer
                continue
            }

            let retainedCount = longestSuffixPrefixLength(in: remaining, of: tag)
            if retainedCount > 0 {
                let splitIndex = remaining.index(remaining.endIndex, offsetBy: -retainedCount)
                append(String(remaining[..<splitIndex]), to: &fragment)
                pendingTagPrefix = String(remaining[splitIndex...])
            } else {
                append(remaining, to: &fragment)
            }
            remaining = ""
        }
        return fragment
    }

    mutating func finish() -> ChatStreamFragment {
        var fragment = ChatStreamFragment()
        append(pendingTagPrefix, to: &fragment)
        pendingTagPrefix = ""
        return fragment
    }

    static func parseComplete(_ text: String) -> ChatStreamFragment {
        var parser = ThinkingTagStreamParser()
        var result = parser.consume(text)
        result.append(parser.finish())
        return result
    }

    private mutating func append(_ text: String, to fragment: inout ChatStreamFragment) {
        guard !text.isEmpty else { return }
        switch mode {
        case .answer:
            fragment.answer += text
        case .reasoning:
            fragment.reasoning += text
        }
    }

    private func longestSuffixPrefixLength(in text: String, of tag: String) -> Int {
        let maximum = min(text.count, tag.count - 1)
        guard maximum > 0 else { return 0 }
        for length in stride(from: maximum, through: 1, by: -1) {
            if text.suffix(length) == tag.prefix(length) {
                return length
            }
        }
        return 0
    }
}

enum ChatHistoryGroup: Int, CaseIterable, Identifiable {
    case today
    case yesterday
    case previousSevenDays
    case previousThirtyDays
    case older

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .today: return "Today"
        case .yesterday: return "Yesterday"
        case .previousSevenDays: return "Previous 7 Days"
        case .previousThirtyDays: return "Previous 30 Days"
        case .older: return "Older"
        }
    }

    static func group(for date: Date, now: Date = Date(), calendar: Calendar = .current) -> ChatHistoryGroup {
        let startOfToday = calendar.startOfDay(for: now)
        let startOfDate = calendar.startOfDay(for: date)
        let days = calendar.dateComponents([.day], from: startOfDate, to: startOfToday).day ?? 0
        switch days {
        case ...0: return .today
        case 1: return .yesterday
        case 2...7: return .previousSevenDays
        case 8...30: return .previousThirtyDays
        default: return .older
        }
    }
}

struct ChatTranscriptFollowState: Equatable {
    private(set) var followsLatest = true

    mutating func userDidScroll() {
        followsLatest = false
    }

    mutating func resume() {
        followsLatest = true
    }
}

struct TranscriptFollowTransitionGate: Equatable {
    private(set) var lastEmittedValue: Bool?

    mutating func valueToEmit(for value: Bool) -> Bool? {
        guard value != lastEmittedValue else { return nil }
        lastEmittedValue = value
        return value
    }

    mutating func synchronize(with value: Bool) {
        lastEmittedValue = value
    }

    mutating func reset() {
        lastEmittedValue = nil
    }
}

enum TranscriptScrollGeometry {
    static let nearBottomThreshold: CGFloat = 72

    static func bottomDistance(
        documentBounds: CGRect,
        visibleRect: CGRect,
        isFlipped: Bool
    ) -> CGFloat {
        let distance = isFlipped
            ? documentBounds.maxY - visibleRect.maxY
            : visibleRect.minY - documentBounds.minY
        return max(0, distance)
    }

    static func isNearBottom(
        documentBounds: CGRect,
        visibleRect: CGRect,
        isFlipped: Bool,
        threshold: CGFloat = nearBottomThreshold
    ) -> Bool {
        bottomDistance(
            documentBounds: documentBounds,
            visibleRect: visibleRect,
            isFlipped: isFlipped
        ) <= threshold
    }
}

struct ThinkingDisclosureState: Equatable {
    private(set) var isExpanded: Bool
    private(set) var userOverrodeExpansion = false

    init(answerHasStarted: Bool) {
        isExpanded = !answerHasStarted
    }

    mutating func userSetExpanded(_ expanded: Bool) {
        userOverrodeExpansion = true
        isExpanded = expanded
    }

    mutating func answerBegan() {
        guard !userOverrodeExpansion else { return }
        isExpanded = false
    }
}
