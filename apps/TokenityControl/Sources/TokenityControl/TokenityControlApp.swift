import SwiftUI
import AppKit

@main
struct TokenityControlApp: App {
    @NSApplicationDelegateAdaptor(TokenityAppDelegate.self) private var appDelegate

    var body: some Scene {
        Window("Tokenity", id: "main") {
            TokenityMainWindowView(store: appDelegate.store)
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

struct TokenityMainWindowView: View {
    @ObservedObject var store: TokenityStore

    var body: some View {
        TokenityRootView()
            .environmentObject(store)
            .tokenityThemed()
            .sheet(isPresented: onboardingPresentation) {
                TokenityOnboardingView()
                    .environmentObject(store)
                    .tokenityThemed()
            }
    }

    private var onboardingPresentation: Binding<Bool> {
        Binding(
            get: { store.isOnboardingPresented },
            set: { isPresented in
                if isPresented {
                    store.presentOnboarding()
                } else {
                    store.completeOnboarding()
                }
            }
        )
    }
}

@MainActor
final class TokenityAppDelegate: NSObject, NSApplicationDelegate {
    let store: TokenityStore
    private var terminationInProgress = false

    init(store: TokenityStore) {
        self.store = store
        store.prepareForAppLaunch()
        super.init()
    }

    override convenience init() {
        self.init(store: TokenityStore(loadsChatHistorySynchronously: false))
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        store.startNodeDiscoveryMonitoring()
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
