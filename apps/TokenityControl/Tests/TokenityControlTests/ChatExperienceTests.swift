import XCTest
@testable import TokenityControl

final class ChatParsingTests: XCTestCase {
    func testContentOnlyNeverCreatesThinking() {
        var parser = ThinkingTagStreamParser()
        let fragment = parser.consume("A normal answer with no hidden reasoning.")

        XCTAssertEqual(fragment.reasoning, "")
        XCTAssertEqual(fragment.answer, "A normal answer with no hidden reasoning.")
        XCTAssertFalse(parser.isInsideThinking)
    }

    func testThinkTagsCanCrossEveryChunkBoundary() {
        var parser = ThinkingTagStreamParser()
        let chunks = ["Before <th", "ink>plan", " carefully</th", "ink>After"]
        var result = ChatStreamFragment()

        for chunk in chunks {
            result.append(parser.consume(chunk))
        }
        result.append(parser.finish())

        XCTAssertEqual(result.reasoning, "plan carefully")
        XCTAssertEqual(result.answer, "Before After")
    }

    func testUnclosedThinkTagKeepsReasoningSeparate() {
        var parser = ThinkingTagStreamParser()
        var result = parser.consume("<think>still working")
        result.append(parser.finish())

        XCTAssertEqual(result.reasoning, "still working")
        XCTAssertEqual(result.answer, "")
        XCTAssertTrue(parser.isInsideThinking)
    }

    func testPartialOrdinaryTagLikeTextIsNotSwallowed() {
        var parser = ThinkingTagStreamParser()
        var result = parser.consume("Use <th")
        result.append(parser.consume("em> literally"))
        result.append(parser.finish())

        XCTAssertEqual(result.answer, "Use <them> literally")
        XCTAssertEqual(result.reasoning, "")
    }

    func testUserDisclosureChoiceWinsOverAutomaticCollapse() {
        var state = ThinkingDisclosureState(answerHasStarted: false)
        state.userSetExpanded(true)
        state.answerBegan()

        XCTAssertTrue(state.isExpanded)
        XCTAssertTrue(state.userOverrodeExpansion)
    }

    func testTranscriptStopsFollowingAfterManualScrollAndCanResume() {
        var state = ChatTranscriptFollowState()
        state.userDidScroll()
        state.update(bottomDistance: 300)
        XCTAssertFalse(state.followsLatest)

        state.update(bottomDistance: 100)
        XCTAssertFalse(state.followsLatest)

        state.resume()
        XCTAssertTrue(state.followsLatest)
    }

    func testTranscriptContentGrowthDoesNotLookLikeUserScrolling() {
        var state = ChatTranscriptFollowState()

        state.update(bottomDistance: 500)

        XCTAssertTrue(state.followsLatest)
    }
}

final class MarkdownRenderingTests: XCTestCase {
    func testParsesAllHeadingLevels() {
        let source = (1...6).map { String(repeating: "#", count: $0) + " Heading \($0)" }.joined(separator: "\n")
        let blocks = StableMarkdownParser.parse(source)

        XCTAssertEqual(blocks.count, 6)
        for level in 1...6 {
            XCTAssertTrue(blocks.contains(.heading(level: level, text: "Heading \(level)")))
        }
    }

    func testParsesNestedAndOrderedLists() {
        let blocks = StableMarkdownParser.parse("- parent\n  - child\n1. first\n  2. nested")
        guard case .list(let items) = blocks.first else {
            return XCTFail("Expected a list block")
        }

        XCTAssertEqual(items.map(\.level), [0, 1, 0, 1])
        XCTAssertEqual(items.map(\.marker), ["•", "•", "1.", "2."])
    }

    func testParsesQuoteLinkInlineCodeAndRawMarkdownRemainsUnchanged() {
        let source = "> Read [Tokenity](https://example.com) and use `swift test`."
        let blocks = StableMarkdownParser.parse(source)

        XCTAssertEqual(blocks, [.quote("Read [Tokenity](https://example.com) and use `swift test`.")])
        XCTAssertEqual(source, "> Read [Tokenity](https://example.com) and use `swift test`.")
    }

    func testParsesTable() {
        let blocks = StableMarkdownParser.parse("| Name | Value |\n| --- | ---: |\n| A | 1 |")

        XCTAssertEqual(blocks, [.table(headers: ["Name", "Value"], rows: [["A", "1"]])])
    }

    func testClosedAndUnclosedCodeFencesRemainVisible() {
        let closed = StableMarkdownParser.parse("```swift\nlet value = 1\n```")
        let unclosed = StableMarkdownParser.parse("```python\nprint('streaming')")

        XCTAssertEqual(closed, [.code(language: "swift", text: "let value = 1")])
        XCTAssertEqual(unclosed, [.code(language: "python", text: "print('streaming')")])
    }

    func testIncompleteInlineMarkdownRemainsAParagraph() {
        let blocks = StableMarkdownParser.parse("A streaming **bold phrase and [link](https://example.com")

        XCTAssertEqual(blocks, [.paragraph("A streaming **bold phrase and [link](https://example.com")])
    }

    @MainActor
    func testStreamingRendererCannotBeStarvedByContinuousUpdates() async throws {
        let model = MarkdownRenderModel(source: "")

        for count in 1...12 {
            model.update(source: String(repeating: "word ", count: count))
            try await Task.sleep(for: .milliseconds(10))
        }

        XCTAssertFalse(model.blocks.isEmpty)
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(model.blocks, [.paragraph(String(repeating: "word ", count: 12))])
    }
}

@MainActor
final class ChatExperienceStoreTests: XCTestCase {
    func testComposerIgnoresAStaleBindingRenderUntilNativeTextIsAcknowledged() {
        var state = ComposerTextSyncState(bindingText: "")

        state.nativeTextDidChange("hello")
        XCTAssertFalse(state.bindingDidUpdate(""), "An unrelated render must not restore stale text")
        XCTAssertFalse(state.bindingDidUpdate("hello"), "Binding acknowledgement must not rewrite native text")
        XCTAssertTrue(state.bindingDidUpdate("replacement"), "A later external edit must still be applied")

        var immediateSend = ComposerTextSyncState(bindingText: "")
        immediateSend.nativeTextDidChange("pasted prompt")
        XCTAssertTrue(
            immediateSend.bindingDidUpdate("", forceExternal: true),
            "Sending immediately after a paste must still clear the native editor"
        )
    }

    func testContentOnlyStreamCompletesWithoutThinkingPanelData() async throws {
        let store = try await readyStore(lines: [
            #"data: {"choices":[{"delta":{"content":"Only an answer."},"finish_reason":null}]}"#,
            "data: [DONE]"
        ])
        store.chatInput = "Answer directly"

        await store.sendChatMessage()

        let message = try XCTUnwrap(store.chatMessages.last)
        XCTAssertEqual(message.thinking, "")
        XCTAssertEqual(message.content, "Only an answer.")
        XCTAssertEqual(message.generationState, .completed)
        XCTAssertNotNil(message.metrics)
    }

    func testReasoningContentCreatesThinkingAndThenAnswer() async throws {
        let store = try await readyStore(lines: [
            #"data: {"choices":[{"delta":{"reasoning_content":"Check facts."},"finish_reason":null}]}"#,
            #"data: {"choices":[{"delta":{"content":"Final answer."},"finish_reason":null}]}"#,
            "data: [DONE]"
        ])
        store.chatInput = "Think first"

        await store.sendChatMessage()

        let message = try XCTUnwrap(store.chatMessages.last)
        XCTAssertEqual(message.thinking, "Check facts.")
        XCTAssertEqual(message.content, "Final answer.")
        XCTAssertEqual(message.generationState, .completed)
        XCTAssertNotNil(message.reasoningDurationSeconds)
        XCTAssertGreaterThan(message.reasoningTokenCount ?? 0, 0)
    }

    func testCrossChunkThinkTagsNeverLeakIntoFinalAnswer() async throws {
        let store = try await readyStore(lines: [
            Self.contentLine("<thi"),
            Self.contentLine("nk>private plan"),
            Self.contentLine("</th"),
            Self.contentLine("ink>Public answer"),
            "data: [DONE]"
        ])
        store.chatInput = "Split tags"

        await store.sendChatMessage()

        let message = try XCTUnwrap(store.chatMessages.last)
        XCTAssertEqual(message.thinking, "private plan")
        XCTAssertEqual(message.content, "Public answer")
        XCTAssertFalse(message.content.contains("think"))
    }

    func testRepetitionDetectorSetsExplicitTerminationState() async throws {
        let repeated = String(repeating: "one two three four ", count: 3)
        let store = try await readyStore(lines: [Self.contentLine(repeated), "data: [DONE]"])
        store.chatInput = "Do not repeat"

        await store.sendChatMessage()

        let message = try XCTUnwrap(store.chatMessages.last)
        XCTAssertEqual(message.generationState, .repetitive)
        XCTAssertTrue(message.statusMessage?.contains("repeated output") == true)
        XCTAssertFalse(message.includeInContext)
    }

    func testUserStopMarksMessageAndRejectsLateTokens() async throws {
        var streamContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let defaults = UserDefaults(suiteName: "ChatExperienceTests.\(UUID().uuidString)")!
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream<ChatStreamEvent, Error> { continuation in
                    streamContinuation = continuation
                }
            },
            userDefaults: defaults
        )
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        store.chatInput = "Stop this"
        store.beginSendingChatMessage()
        for _ in 0..<100 where streamContinuation == nil { await Task.yield() }
        streamContinuation?.yield(.line(Self.contentLine("partial")))
        for _ in 0..<100 where !(store.chatMessages.last?.content.contains("partial") ?? false) { await Task.yield() }

        store.cancelChatGeneration()
        let stoppedContent = store.chatMessages.last?.content
        streamContinuation?.yield(.line(Self.contentLine("late")))
        streamContinuation?.finish()
        for _ in 0..<10 { await Task.yield() }

        XCTAssertEqual(store.chatMessages.last?.generationState, .stopped)
        XCTAssertEqual(store.chatMessages.last?.content, stoppedContent)
        XCTAssertFalse(store.chatMessages.last?.content.contains("late") ?? true)
        XCTAssertFalse(store.chatMessages.last?.includeInContext ?? true)
    }

    func testMetricsRemainHiddenUntilStreamingAnswerFinishes() async throws {
        var streamContinuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation?
        let defaults = UserDefaults(suiteName: "ChatExperienceTests.\(UUID().uuidString)")!
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream<ChatStreamEvent, Error> { continuation in
                    streamContinuation = continuation
                }
            },
            userDefaults: defaults
        )
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        store.chatInput = "Measure after completion"
        store.beginSendingChatMessage()
        for _ in 0..<100 where streamContinuation == nil { await Task.yield() }

        streamContinuation?.yield(.line(Self.contentLine("first token")))
        for _ in 0..<100 where store.chatMessages.last?.content != "first token" { await Task.yield() }
        XCTAssertNil(store.chatMessages.last?.metrics)
        XCTAssertEqual(store.chatMetrics, .empty)

        streamContinuation?.yield("data: [DONE]")
        streamContinuation?.finish()
        for _ in 0..<100 where store.isChatRunning { await Task.yield() }
        XCTAssertNotNil(store.chatMessages.last?.metrics)
        XCTAssertNotNil(store.chatMetrics.firstTokenSeconds)
        XCTAssertNotNil(store.chatMetrics.outputTokensPerSecond)
        XCTAssertEqual(store.chatMetrics.outputTokens, 2)
    }

    func testHistoryRenameSearchSwitchAndDeleteCurrentSelectsNeighbor() {
        let defaults = UserDefaults(suiteName: "ChatExperienceTests.\(UUID().uuidString)")!
        let store = TokenityStore(userDefaults: defaults)
        defer { defaults.removePersistentDomain(forName: defaultsSuiteName(defaults)) }
        let firstID = store.activeChatSessionID
        store.chatMessages = [ChatMessage(role: .user, content: "First topic")]
        store.newChatSession()
        let secondID = store.activeChatSessionID
        store.chatMessages = [ChatMessage(role: .user, content: "Second topic")]
        store.renameChatSession(secondID, title: "Custom Name")

        XCTAssertTrue(store.chatSessions.first(where: { $0.id == secondID })?.matches(search: "custom") == true)
        store.selectChatSession(firstID)
        XCTAssertEqual(store.chatMessages.first?.content, "First topic")

        store.deleteChatSession(firstID)
        XCTAssertEqual(store.activeChatSessionID, secondID)
        XCTAssertEqual(store.chatMessages.first?.content, "Second topic")
    }

    func testSelectingHistoryDoesNotTouchTimestampsOrReorderSessions() throws {
        let defaults = UserDefaults(suiteName: "ChatExperienceTests.\(UUID().uuidString)")!
        let olderDate = Date(timeIntervalSince1970: 1_000)
        let newerDate = Date(timeIntervalSince1970: 2_000)
        let older = ChatSession(
            id: UUID(),
            title: "Older prompt",
            createdAt: olderDate,
            updatedAt: olderDate,
            messages: [ChatMessage(role: .user, content: "Older prompt")],
            metrics: .empty,
            titleWasEdited: false
        )
        let newer = ChatSession(
            id: UUID(),
            title: "Newer prompt",
            createdAt: newerDate,
            updatedAt: newerDate,
            messages: [ChatMessage(role: .user, content: "Newer prompt")],
            metrics: .empty,
            titleWasEdited: false
        )
        defaults.set(try JSONEncoder().encode([older, newer]), forKey: "TokenityChatSessions.v1")
        let store = TokenityStore(userDefaults: defaults)

        XCTAssertEqual(store.chatSessions.map(\.id), [newer.id, older.id])
        store.selectChatSession(older.id)
        XCTAssertEqual(store.chatSessions.map(\.id), [newer.id, older.id])
        XCTAssertEqual(store.chatSessions.first(where: { $0.id == older.id })?.updatedAt, olderDate)
        XCTAssertEqual(store.chatSessions.first(where: { $0.id == newer.id })?.updatedAt, newerDate)

        store.selectChatSession(newer.id)
        XCTAssertEqual(store.chatSessions.map(\.id), [newer.id, older.id])
        XCTAssertEqual(store.chatSessions.first(where: { $0.id == older.id })?.updatedAt, olderDate)
        XCTAssertEqual(store.chatSessions.first(where: { $0.id == newer.id })?.updatedAt, newerDate)
    }

    func testHistoryGroupsBoundaries() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 7, day: 16, hour: 12)))

        XCTAssertEqual(ChatHistoryGroup.group(for: now, now: now, calendar: calendar), .today)
        XCTAssertEqual(ChatHistoryGroup.group(for: calendar.date(byAdding: .day, value: -1, to: now)!, now: now, calendar: calendar), .yesterday)
        XCTAssertEqual(ChatHistoryGroup.group(for: calendar.date(byAdding: .day, value: -7, to: now)!, now: now, calendar: calendar), .previousSevenDays)
        XCTAssertEqual(ChatHistoryGroup.group(for: calendar.date(byAdding: .day, value: -30, to: now)!, now: now, calendar: calendar), .previousThirtyDays)
        XCTAssertEqual(ChatHistoryGroup.group(for: calendar.date(byAdding: .day, value: -31, to: now)!, now: now, calendar: calendar), .older)
    }

    private func readyStore(lines: [String]) async throws -> TokenityStore {
        let defaults = UserDefaults(suiteName: "ChatExperienceTests.\(UUID().uuidString)")!
        let store = TokenityStore(
            dataTransport: Self.successfulModelTransport,
            lineStreamTransport: { _ in
                AsyncThrowingStream<ChatStreamEvent, Error> { continuation in
                    for line in lines {
                        continuation.yield(.line(line))
                    }
                    continuation.finish()
                }
            },
            userDefaults: defaults
        )
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        store.connectionMode = .ring
        store.createCluster()
        await store.loadModel(model)
        return store
    }

    private static func contentLine(_ content: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: [
            "choices": [["delta": ["content": content], "finish_reason": NSNull()]]
        ])
        return "data: " + String(data: data, encoding: .utf8)!
    }

    private static func successfulModelTransport(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let payload: String
        if path == "/v1/node/info" {
            payload = TokenityTestFixtures.basicNodeInfoPayload(for: request)
        } else if path.contains("/v1/models") {
            payload = #"{"data":[{"id":"Qwen3.5-122B-A10B-4bit"}]}"#
        } else if path.contains("/v1/chat/completions") {
            payload = "data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
        } else if path.contains("/v1/node/start") {
            payload = #"{"status":{"state":"running"}}"#
        } else {
            payload = #"{}"#
        }
        let response = HTTPURLResponse(
            url: request.url ?? URL(string: "http://127.0.0.1")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        return (Data(payload.utf8), response)
    }

    private func defaultsSuiteName(_ defaults: UserDefaults) -> String {
        // The test suite is discarded with the process; return a unique domain
        // name only to keep the cleanup call harmless across Foundation versions.
        "ChatExperienceTests.cleanup.\(ObjectIdentifier(defaults).hashValue)"
    }
}
