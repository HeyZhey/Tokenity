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
                .sheet(
                    isPresented: Binding(
                        get: { appDelegate.store.isOnboardingPresented },
                        set: { isPresented in
                            if isPresented {
                                appDelegate.store.presentOnboarding()
                            } else {
                                appDelegate.store.completeOnboarding()
                            }
                        }
                    )
                ) {
                    TokenityOnboardingView()
                        .environmentObject(appDelegate.store)
                        .tokenityThemed()
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(after: .help) {
                Button("Show Tokenity Welcome") {
                    appDelegate.store.presentOnboarding()
                }
            }
        }

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
        store.prepareForAppLaunch()
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
