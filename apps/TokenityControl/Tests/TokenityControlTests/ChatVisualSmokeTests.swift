import AppKit
import SwiftUI
import XCTest
@testable import TokenityControl

@MainActor
final class ChatVisualSmokeTests: XCTestCase {
    func testChatWorkspaceRendersInLightAndDarkAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let suiteName = "ChatVisualSmokeTests.\(scheme).\(UUID().uuidString)"
            let defaults = UserDefaults(suiteName: suiteName)!
            defer { defaults.removePersistentDomain(forName: suiteName) }
            let store = TokenityStore(userDefaults: defaults)
            store.chatMessages = [
                ChatMessage(role: .user, content: "Show a Markdown and code example."),
                ChatMessage(
                    role: .assistant,
                    content: """
                    # Native Markdown

                    A paragraph with **bold**, *italic*, ~~deleted~~, and `inline code`.

                    - First item
                      - Nested item

                    | Feature | State |
                    | --- | --- |
                    | Markdown | Ready |

                    ```swift
                    struct Greeting {
                        let text = "Hello, Tokenity"
                    }
                    ```
                    """,
                    thinking: "I checked the requested structure and selected a compact example.",
                    generationState: .completed,
                    metrics: ChatMetrics(firstTokenSeconds: 0.42, totalSeconds: 2.8, outputTokensPerSecond: 31.5),
                    reasoningDurationSeconds: 1.1,
                    reasoningTokenCount: 18,
                    modelName: "Preview Model"
                )
            ]

            let root = ChatWorkspaceView()
                .environmentObject(store)
                .tokenityThemed()
                .environment(\.colorScheme, scheme)
            let hostingView = NSHostingView(rootView: root)
            hostingView.frame = NSRect(x: 0, y: 0, width: 1_200, height: 800)
            hostingView.layoutSubtreeIfNeeded()

            let representation = try XCTUnwrap(hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds))
            hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
            let data = try XCTUnwrap(representation.representation(using: .png, properties: [:]))
            let name = scheme == .light ? "light" : "dark"
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-chat-\(name).png"))
            XCTAssertGreaterThan(data.count, 10_000)
        }
    }

    func testComposerKeepsTextAndCaretDuringUnrelatedStorePublications() async throws {
        let suiteName = "ChatVisualSmokeTests.composer.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(dataTransport: Self.successfulModelTransport, userDefaults: defaults)
        store.connectionMode = .ring
        store.createCluster()
        let model = try XCTUnwrap(store.modelLibraryRows.first)
        await store.loadModel(model)
        XCTAssertTrue(store.isChatReady)

        let root = ChatWorkspaceView()
            .environmentObject(store)
            .tokenityThemed()
        let hostingView = NSHostingView(rootView: root)
        hostingView.frame = NSRect(x: 0, y: 0, width: 1_200, height: 800)
        let window = NSWindow(
            contentRect: hostingView.frame,
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.contentView = hostingView
        hostingView.layoutSubtreeIfNeeded()
        let composer = try XCTUnwrap(allTextViews(in: hostingView).first(where: \.isEditable))
        window.makeFirstResponder(composer)

        let expected = "continuous typing keeps the caret"
        for character in expected {
            composer.insertText(String(character), replacementRange: composer.selectedRange())
            store.logs.append("Unrelated status publication \(character)")
            await Task.yield()
            await Task.yield()
        }

        XCTAssertEqual(composer.string, expected)
        XCTAssertEqual(store.chatInput, expected)
        XCTAssertEqual(composer.selectedRange(), NSRange(location: expected.utf16.count, length: 0))
        withExtendedLifetime(window) {}
    }

    private func allTextViews(in view: NSView) -> [NSTextView] {
        var result: [NSTextView] = []
        if let textView = view as? NSTextView {
            result.append(textView)
        }
        for subview in view.subviews {
            result.append(contentsOf: allTextViews(in: subview))
        }
        return result
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
        } else if path.contains("/v1/readiness") {
            payload = #"{"phase":"ready"}"#
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
}
