import SwiftUI
import AppKit

@main
struct TokenityControlApp: App {
    @NSApplicationDelegateAdaptor(TokenityAppDelegate.self) private var appDelegate
    @StateObject private var store = TokenityStore()

    var body: some Scene {
        WindowGroup("Tokenity") {
            TokenityRootView()
                .environmentObject(store)
                .tokenityThemed()
                .onAppear {
                    appDelegate.store = store
                }
        }
        .windowResizability(.contentSize)
    }
}

@MainActor
final class TokenityAppDelegate: NSObject, NSApplicationDelegate {
    weak var store: TokenityStore?
    private var terminationInProgress = false

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in sender.windows {
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
            }
        }
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminationInProgress, let store else {
            return .terminateNow
        }
        terminationInProgress = true
        Task {
            await store.shutdownForApplicationTermination()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}
