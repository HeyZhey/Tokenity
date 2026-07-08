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
        }
        .windowResizability(.contentSize)
    }
}

final class TokenityAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in sender.windows {
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
            }
        }
        return true
    }
}
