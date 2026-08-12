import AppKit
import SwiftUI
import XCTest
@testable import TokenityControl

@MainActor
final class ChatVisualSmokeTests: XCTestCase {
    func testBundledBrandAssetsAreTransparentAndUsableAtSmallSizes() throws {
        for asset in [TokenityBrandAsset.lockup, .mark] {
            var dimensions: [NSSize] = []
            var luminance: [CGFloat] = []

            for scheme in [ColorScheme.light, .dark] {
                let image = TokenityBrandAssets.image(asset, for: scheme)
                var proposedRect = NSRect(origin: .zero, size: image.size)
                let cgImage = try XCTUnwrap(
                    image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil)
                )
                XCTAssertGreaterThan(cgImage.width, 100)
                XCTAssertGreaterThan(cgImage.height, 100)
                dimensions.append(NSSize(width: cgImage.width, height: cgImage.height))

                let representation = NSBitmapImageRep(cgImage: cgImage)
                XCTAssertTrue(representation.hasAlpha)
                let corner = try XCTUnwrap(representation.colorAt(x: 0, y: 0))
                XCTAssertLessThan(corner.alphaComponent, 0.05)
                luminance.append(try averageOpaqueLuminance(in: representation))
            }

            XCTAssertEqual(dimensions[0], dimensions[1])
            XCTAssertGreaterThan(
                luminance[1],
                luminance[0] + 0.2,
                "The Dark Mode brand variant must be visibly lighter than the source-colored Light Mode asset"
            )
        }
    }

    private func averageOpaqueLuminance(in representation: NSBitmapImageRep) throws -> CGFloat {
        var total: CGFloat = 0
        var samples = 0
        for y in stride(from: 0, to: representation.pixelsHigh, by: 8) {
            for x in stride(from: 0, to: representation.pixelsWide, by: 8) {
                let color = try XCTUnwrap(representation.colorAt(x: x, y: y))
                guard color.alphaComponent > 0.5,
                      let rgb = color.usingColorSpace(.sRGB) else { continue }
                total += (0.2126 * rgb.redComponent)
                    + (0.7152 * rgb.greenComponent)
                    + (0.0722 * rgb.blueComponent)
                samples += 1
            }
        }
        XCTAssertGreaterThan(samples, 20)
        return total / CGFloat(max(samples, 1))
    }

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

    func testChatOperationalStatesRenderInLightAndDarkAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let suiteName = "ChatVisualSmokeTests.states.\(scheme).\(UUID().uuidString)"
            let defaults = UserDefaults(suiteName: suiteName)!
            defer { defaults.removePersistentDomain(forName: suiteName) }
            let store = TokenityStore(userDefaults: defaults)
            store.chatMessages = [
                ChatMessage(role: .user, content: "Summarize this long technical trace and keep the code intact."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    thinking: "Inspecting the runtime evidence and comparing rank state.",
                    generationState: .reasoning,
                    reasoningTokenCount: 42,
                    modelName: "Qwen3.5-122B-A10B-4bit"
                ),
                ChatMessage(role: .user, content: "Stop this response."),
                ChatMessage(
                    role: .assistant,
                    content: "The partial answer remains available.",
                    generationState: .stopped,
                    statusMessage: "Generation stopped by the user.",
                    modelName: "Qwen3.5-122B-A10B-4bit"
                ),
                ChatMessage(role: .user, content: "Try the failing request."),
                ChatMessage(
                    role: .assistant,
                    content: "",
                    generationState: .failed,
                    statusMessage: "The model service returned an error.",
                    modelName: "Qwen3.5-122B-A10B-4bit"
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

            let representation = try XCTUnwrap(
                hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
            )
            hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
            let data = try XCTUnwrap(
                representation.representation(using: .png, properties: [:])
            )
            let appearance = scheme == .light ? "light" : "dark"
            try data.write(to: URL(fileURLWithPath: "/tmp/tokenity-chat-states-\(appearance).png"))
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
        let mainDividerX = Int((300 * pixelsPerPoint).rounded())
        let historyDividerX = Int(((logicalSize.width - 150) * pixelsPerPoint).rounded())

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

        let dividerSearchRange = Int(
            ((ChatWorkspaceMetrics.headerHeight - 2) * pixelsPerPoint).rounded()
        )...Int(
            ((ChatWorkspaceMetrics.headerHeight + 2) * pixelsPerPoint).rounded()
        )
        var strongestDivider: (y: Int, color: NSColor, distance: CGFloat)?
        for y in dividerSearchRange {
            guard let color = representation.colorAt(x: mainDividerX, y: y) else { continue }
            let distance = colorDistance(color, mainHeader)
            if strongestDivider == nil || distance > strongestDivider!.distance {
                strongestDivider = (y, color, distance)
            }
        }
        let divider = try XCTUnwrap(strongestDivider)
        XCTAssertGreaterThan(
            divider.distance,
            0.01,
            "The shared Chat and History header divider is not visible"
        )
        let historyDivider = try XCTUnwrap(
            representation.colorAt(
                x: historyDividerX,
                y: divider.y
            )
        )
        assertColor(
            divider.color,
            matches: historyDivider,
            message: "Chat and History header dividers are vertically misaligned"
        )

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

    private func colorDistance(_ lhs: NSColor, _ rhs: NSColor) -> CGFloat {
        guard let left = lhs.usingColorSpace(.deviceRGB),
              let right = rhs.usingColorSpace(.deviceRGB) else {
            return 0
        }
        return abs(left.redComponent - right.redComponent)
            + abs(left.greenComponent - right.greenComponent)
            + abs(left.blueComponent - right.blueComponent)
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

    func testAllPagesRenderAtMinimumAndReferenceWindowSizes() throws {
        let outputDirectory = URL(fileURLWithPath: "/tmp/tokenity-page-matrix", isDirectory: true)
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        let sizes: [(name: String, value: NSSize)] = [
            ("1120x720", NSSize(width: 1_120, height: 720)),
            ("1280x820", NSSize(width: 1_280, height: 820)),
        ]

        for size in sizes {
            for scheme in [ColorScheme.light, .dark] {
                for section in AppSection.allCases {
                    let suiteName = [
                        "ChatVisualSmokeTests.pages",
                        size.name,
                        section.rawValue,
                        "\(scheme)",
                        UUID().uuidString,
                    ].joined(separator: ".")
                    let defaults = UserDefaults(suiteName: suiteName)!
                    defer { defaults.removePersistentDomain(forName: suiteName) }
                    let store = TokenityStore(userDefaults: defaults)
                    store.selectedSection = section
                    let root = HStack(spacing: 0) {
                        SidebarView(selection: .constant(section))
                            .frame(width: 218)
                        visualPage(for: section)
                    }
                        .environmentObject(store)
                        .tokenityThemed()
                        .environment(\.colorScheme, scheme)
                        .environment(\.displayScale, 1)
                    let hostingView = NSHostingView(rootView: root)
                    hostingView.frame = NSRect(origin: .zero, size: size.value)
                    hostingView.layoutSubtreeIfNeeded()

                    let representation = try XCTUnwrap(
                        hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
                    )
                    hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
                    let data = try XCTUnwrap(
                        representation.representation(using: .png, properties: [:])
                    )
                    let appearance = scheme == .light ? "light" : "dark"
                    let outputURL = outputDirectory
                        .appendingPathComponent(
                            "\(section.rawValue)-\(appearance)-\(size.name).png"
                        )
                    try data.write(to: outputURL)
                    XCTAssertGreaterThan(data.count, 4_000)
                }
            }
        }
    }

    private func visualPage(for section: AppSection) -> AnyView {
        switch section {
        case .overview: return AnyView(OverviewPage())
        case .cluster: return AnyView(ClusterPage())
        case .chat: return AnyView(ChatWorkspaceView())
        case .video: return AnyView(VideoGenerationPage())
        case .models: return AnyView(ModelsPage())
        case .network: return AnyView(NetworkPage())
        case .api: return AnyView(APIAccessPage())
        case .logs: return AnyView(LogsPage())
        case .settings: return AnyView(SettingsPage())
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
                    {"data":[
                      {"model":"Qwen3.5-122B-A10B-4bit","instance_id":"visual-instance","model_revision":"visual-rev","execution_mode":"single","state":"busy","active_request_count":1,"queue_depth":2,"api_base_url":"http://127.0.0.1:18000/v1","capabilities":{"tools":true,"json":true,"thinking":true,"modalities":["text"],"task_tags":["general","code"]},"warm_ttft_p50_ms":180.0,"warm_ttft_p95_ms":420.0},
                      {"model":"GLM-5.2-mxfp4","instance_id":"visual-instance-2","model_revision":"visual-rev-2","execution_mode":"distributed","state":"ready","active_request_count":0,"queue_depth":0,"api_base_url":"http://127.0.0.1:18001/v1","capabilities":{"tools":false,"json":true,"thinking":false,"modalities":["text"],"task_tags":["general"]},"warm_ttft_p50_ms":260.0,"warm_ttft_p95_ms":610.0}
                    ]}
                    """
                } else if path.hasSuffix("/quorum") {
                    let instanceID = path.contains("visual-instance-2")
                        ? "visual-instance-2"
                        : "visual-instance"
                    payload = """
                    {"instance_id":"\(instanceID)","ready":true,"issues":[],"rank_quorum":"1/1","ranks":[]}
                    """
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
        XCTAssertEqual(store.residentModelInstances.count, 2)
        XCTAssertEqual(store.chatRoutingDisplayTitle, "Free routing · 2 models")
        XCTAssertTrue(store.chatRoutingDisplayHelp.contains("GLM-5.2-mxfp4"))
        XCTAssertTrue(store.chatRoutingDisplayHelp.contains("Qwen3.5-122B-A10B-4bit"))
        let instance = try XCTUnwrap(
            store.residentModelInstances.first(where: { $0.id == "visual-instance" })
        )
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

            let chatRoot = ChatWorkspaceView()
                .environmentObject(store)
                .tokenityThemed()
                .environment(\.colorScheme, scheme)
                .environment(\.displayScale, 2)
            let chatHostingView = NSHostingView(rootView: chatRoot)
            chatHostingView.frame = NSRect(x: 0, y: 0, width: 1_200, height: 800)
            chatHostingView.layoutSubtreeIfNeeded()
            let chatRepresentation = try XCTUnwrap(
                chatHostingView.bitmapImageRepForCachingDisplay(in: chatHostingView.bounds)
            )
            chatHostingView.cacheDisplay(
                in: chatHostingView.bounds,
                to: chatRepresentation
            )
            let chatData = try XCTUnwrap(
                chatRepresentation.representation(using: .png, properties: [:])
            )
            try chatData.write(
                to: URL(fileURLWithPath: "/tmp/tokenity-chat-free-routing-\(name).png")
            )
            XCTAssertGreaterThan(chatData.count, 10_000)

            let clusterRoot = ClusterPage()
                .environmentObject(store)
                .tokenityThemed()
                .environment(\.colorScheme, scheme)
                .environment(\.displayScale, 2)
            let clusterHostingView = NSHostingView(rootView: clusterRoot)
            clusterHostingView.frame = NSRect(x: 0, y: 0, width: 1_200, height: 800)
            clusterHostingView.layoutSubtreeIfNeeded()
            let clusterRepresentation = try XCTUnwrap(
                clusterHostingView.bitmapImageRepForCachingDisplay(in: clusterHostingView.bounds)
            )
            clusterHostingView.cacheDisplay(
                in: clusterHostingView.bounds,
                to: clusterRepresentation
            )
            let clusterData = try XCTUnwrap(
                clusterRepresentation.representation(using: .png, properties: [:])
            )
            try clusterData.write(
                to: URL(fileURLWithPath: "/tmp/tokenity-cluster-node-padding-\(name).png")
            )
            XCTAssertGreaterThan(clusterData.count, 10_000)
        }
    }

    func testMenuBarBrandMarkRendersInLightAndDarkAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let nativeMark = TokenityBrandAssets.menuBarMarkImage(for: scheme)
            XCTAssertEqual(nativeMark.size.width, 20, accuracy: 0.01)
            XCTAssertEqual(nativeMark.size.height, 12, accuracy: 0.01)

            let intrinsicView = NSHostingView(
                rootView: TokenityMenuBarMark(level: .warning)
                    .environment(\.colorScheme, scheme)
            )
            XCTAssertLessThanOrEqual(intrinsicView.fittingSize.width, 24.01)
            XCTAssertLessThanOrEqual(intrinsicView.fittingSize.height, 18.01)

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
        await Task.yield()
        await Task.yield()
        hostingView.layoutSubtreeIfNeeded()
        XCTAssertTrue(window.firstResponder === composer)

        let focusedRepresentation = try XCTUnwrap(
            hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
        )
        hostingView.cacheDisplay(in: hostingView.bounds, to: focusedRepresentation)
        let focusedData = try XCTUnwrap(
            focusedRepresentation.representation(using: .png, properties: [:])
        )
        try focusedData.write(to: URL(fileURLWithPath: "/tmp/tokenity-composer-focused.png"))
        XCTAssertGreaterThan(focusedData.count, 10_000)

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
        ,"instances":[
          {"instance_id":"visual-instance","operation_id":"visual-operation","requested_model_id":"Qwen3.5-122B-A10B-4bit","resolved_path":"/fixtures/tokenity/models/Qwen3.5-122B-A10B-4bit","model_revision":"visual-rev","execution_mode":"single","selected_nodes":["mac-a"],"world_size":1,"connection_mode":"ring","coordinator":"mac-a","http_port":18000,"memory_reservation_bytes":17179869184,"actual_memory_bytes":15032385536,"state":"busy","active_request_count":1,"queued_request_count":2,"health_ready":true,"health_issues":[]},
          {"instance_id":"visual-instance-2","operation_id":"visual-operation-2","requested_model_id":"GLM-5.2-mxfp4","resolved_path":"/fixtures/tokenity/models/GLM-5.2-mxfp4","model_revision":"visual-rev-2","execution_mode":"distributed","selected_nodes":["mac-a","mac-b"],"world_size":2,"connection_mode":"ring","coordinator":"mac-a","http_port":18001,"memory_reservation_bytes":274877906944,"actual_memory_bytes":270582939648,"state":"ready","active_request_count":0,"queued_request_count":0,"health_ready":true,"health_issues":["Low memory headroom on worker"]}]}
        """
    }
}
