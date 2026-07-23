import SwiftUI
import AppKit

@main
struct TokenityControlApp: App {
    @NSApplicationDelegateAdaptor(TokenityAppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Tokenity", id: "main") {
            TokenityRootView()
                .environmentObject(appDelegate.store)
                .tokenityThemed()
        }
        .windowResizability(.contentSize)

        MenuBarExtra {
            TokenityMenuBarContentHost(store: appDelegate.store)
        } label: {
            TokenityMenuBarLabelHost(store: appDelegate.store)
        }
        .menuBarExtraStyle(.menu)
    }
}

@MainActor
final class TokenityAppDelegate: NSObject, NSApplicationDelegate {
    let store = TokenityStore(loadsChatHistorySynchronously: false)
    private var terminationInProgress = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        store.startStatusMonitoring()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

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
        guard !terminationInProgress else {
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
