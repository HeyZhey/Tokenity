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
            store.renameChatSession(store.activeChatSessionID, title: "Model setup notes")
            store.newChatSession()
            store.renameChatSession(store.activeChatSessionID, title: "SwiftUI review")
            store.newChatSession()
            store.renameChatSession(store.activeChatSessionID, title: "Release checklist")
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
                    modelName: "Preview Model",
                    routedModelID: "Qwen3.5-122B-A10B-4bit",
                    modelRevision: "visual-rev",
                    instanceID: "visual-instance",
                    routeReason: "Balanced policy matched this general coding request.",
                    routeConfidence: 0.91,
                    routingLatencyMilliseconds: 3.4,
                    queueWaitMilliseconds: 0.8,
                    requestID: "visual-request"
                )
            ]

            let root = ChatWorkspaceView()
                .environmentObject(store)
                .tokenityThemed()
                .environment(\.colorScheme, scheme)
                .environment(\.displayScale, 2)
            let hostingView = NSHostingView(rootView: root)
            hostingView.frame = NSRect(x: 0, y: 0, width: 1_200, height: 800)
            hostingView.layoutSubtreeIfNeeded()

            let representation = try XCTUnwrap(hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds))
            hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
            try assertChatSurfacesAreContinuous(in: representation, logicalSize: hostingView.bounds.size)
            let data = try XCTUnwrap(representation.representation(using: .png, properties: [:]))
            let name = scheme == .light ? "light" : "dark"
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-chat-\(name).png"))
            XCTAssertGreaterThan(data.count, 10_000)
        }
    }

    private func assertChatSurfacesAreContinuous(
        in representation: NSBitmapImageRep,
        logicalSize: NSSize
    ) throws {
        let pixelsPerPoint = CGFloat(representation.pixelsWide) / logicalSize.width
        let historyBoundaryX = Int(((logicalSize.width - 326) * pixelsPerPoint).rounded())
        let headerY = Int((20 * pixelsPerPoint).rounded())
        let bodyY = Int((logicalSize.height * 0.5 * pixelsPerPoint).rounded())

        let mainHeader = try XCTUnwrap(
            representation.colorAt(
                x: Int((300 * pixelsPerPoint).rounded()),
                y: headerY
            )
        )
        let historyHeader = try XCTUnwrap(
            representation.colorAt(
                x: Int(((logicalSize.width - 150) * pixelsPerPoint).rounded()),
                y: headerY
            )
        )
        assertColor(mainHeader, matches: historyHeader, message: "Chat and History headers use different surfaces")

        for y in [headerY, bodyY] {
            let reference = try XCTUnwrap(representation.colorAt(x: historyBoundaryX - 4, y: y))
            for x in (historyBoundaryX - 3)...(historyBoundaryX + 3) {
                let candidate = try XCTUnwrap(representation.colorAt(x: x, y: y))
                assertColor(
                    candidate,
                    matches: reference,
                    message: "Unexpected pixel seam at the History boundary (x: \(x), y: \(y))"
                )
            }
        }
    }

    private func assertColor(_ lhs: NSColor, matches rhs: NSColor, message: String) {
        guard let left = lhs.usingColorSpace(.deviceRGB),
              let right = rhs.usingColorSpace(.deviceRGB) else {
            XCTFail("\(message): colors could not be converted to device RGB")
            return
        }
        let tolerance = CGFloat(1.0 / 255.0)
        XCTAssertEqual(left.redComponent, right.redComponent, accuracy: tolerance, message)
        XCTAssertEqual(left.greenComponent, right.greenComponent, accuracy: tolerance, message)
        XCTAssertEqual(left.blueComponent, right.blueComponent, accuracy: tolerance, message)
        XCTAssertEqual(left.alphaComponent, right.alphaComponent, accuracy: tolerance, message)
    }

    func testSidebarNavigationRendersBrandInLightAndDarkAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let root = SidebarView(selection: .constant(.overview))
                .tokenityThemed()
                .environment(\.colorScheme, scheme)
            let hostingView = NSHostingView(rootView: root)
            hostingView.frame = NSRect(x: 0, y: 0, width: 218, height: 800)
            hostingView.layoutSubtreeIfNeeded()

            let representation = try XCTUnwrap(hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds))
            hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
            let data = try XCTUnwrap(representation.representation(using: .png, properties: [:]))
            let name = scheme == .light ? "light" : "dark"
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-sidebar-navigation-\(name).png"))
            XCTAssertGreaterThan(data.count, 4_000)
        }
    }

    func testResidentModelPoolRendersInLightAndDarkAppearances() async throws {
        let store = TokenityStore(
            dataTransport: { request in
                let path = request.url?.path ?? ""
                let payload: String
                if path == "/v1/node/info" {
                    payload = Self.residentNodeInfoPayload(for: request)
                } else if path == "/v1/gateway/routes" {
                    payload = """
                    {"data":[{"model":"Qwen3.5-122B-A10B-4bit","instance_id":"visual-instance","model_revision":"visual-rev","execution_mode":"single","state":"busy","active_request_count":1,"queue_depth":2,"api_base_url":"http://127.0.0.1:18000/v1","capabilities":{"tools":true,"json":true,"thinking":true,"modalities":["text"],"task_tags":["general","code"]},"warm_ttft_p50_ms":180.0,"warm_ttft_p95_ms":420.0}]}
                    """
                } else if path.hasSuffix("/quorum") {
                    payload = #"{"instance_id":"visual-instance","ready":true,"issues":[],"rank_quorum":"1/1","ranks":[]}"#
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
            },
            userDefaults: UserDefaults(suiteName: "ChatVisualSmokeTests.pool.\(UUID().uuidString)")!
        )
        store.connectionMode = .ring
        await store.refreshSelectedNodeStatus()
        XCTAssertEqual(store.residentModelInstances.count, 1)
        let instance = try XCTUnwrap(store.residentModelInstances.first)
        XCTAssertEqual(instance.modelRevision, "visual-rev")
        XCTAssertEqual(
            Set(instance.capabilities),
            Set(["JSON", "Task: code", "Task: general", "Text", "Thinking", "Tools"])
        )
        XCTAssertEqual(instance.warmTTFTP50Milliseconds, 180)
        XCTAssertEqual(instance.warmTTFTP95Milliseconds, 420)

        for scheme in [ColorScheme.light, .dark] {
            let root = ModelsPage()
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
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-resident-pool-\(name).png"))
            XCTAssertGreaterThan(data.count, 10_000)
        }
    }

    func testMenuBarBrandMarkRendersInLightAndDarkAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let root = TokenityMenuBarMark(level: .warning)
                .padding(8)
                .background(scheme == .light ? Color.white : Color.black)
                .environment(\.colorScheme, scheme)
            let hostingView = NSHostingView(rootView: root)
            hostingView.frame = NSRect(x: 0, y: 0, width: 40, height: 40)
            hostingView.layoutSubtreeIfNeeded()

            let representation = try XCTUnwrap(
                hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
            )
            hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
            let data = try XCTUnwrap(
                representation.representation(using: .png, properties: [:])
            )
            let name = scheme == .light ? "light" : "dark"
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-menubar-logo-\(name).png"))
            XCTAssertGreaterThan(data.count, 500)
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

    private static func residentNodeInfoPayload(for request: URLRequest) -> String {
        let base = TokenityTestFixtures.basicNodeInfoPayload(for: request)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(base.dropLast()) + """
        ,"agent_contract":{"version":1,"capabilities":["managed_instances","instance_runtimes","instance_quorum","cluster_runtime"]}
        ,"instances":[{"instance_id":"visual-instance","operation_id":"visual-operation","requested_model_id":"Qwen3.5-122B-A10B-4bit","resolved_path":"/Users/Shared/TokenityModels/Qwen3.5-122B-A10B-4bit","model_revision":"visual-rev","execution_mode":"single","selected_nodes":["mac-a"],"world_size":1,"connection_mode":"ring","coordinator":"mac-a","http_port":18000,"memory_reservation_bytes":17179869184,"actual_memory_bytes":15032385536,"state":"busy","active_request_count":1,"queued_request_count":2,"health_ready":true,"health_issues":[]}]}
        """
    }
}
