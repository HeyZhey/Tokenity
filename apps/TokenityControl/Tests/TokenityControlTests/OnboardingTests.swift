import AppKit
import SwiftUI
import XCTest
@testable import TokenityControl

@MainActor
final class OnboardingTests: XCTestCase {
    func testAppDelegatePreparesGuideBeforeTheFirstWindowIsBuilt() {
        let suiteName = "OnboardingTests.delegate.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(userDefaults: defaults)

        XCTAssertFalse(store.isOnboardingPresented)
        _ = TokenityAppDelegate(store: store)

        XCTAssertTrue(store.isOnboardingPresented)
    }

    func testFirstLaunchPresentsGuideAndCompletionPersists() {
        let suiteName = "OnboardingTests.persistence.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let firstStore = TokenityStore(userDefaults: defaults)
        XCTAssertFalse(firstStore.isOnboardingPresented)

        firstStore.prepareForAppLaunch()
        XCTAssertTrue(firstStore.isOnboardingPresented)

        firstStore.completeOnboarding(opening: .cluster)
        XCTAssertFalse(firstStore.isOnboardingPresented)
        XCTAssertEqual(firstStore.selectedSection, .cluster)

        let nextStore = TokenityStore(userDefaults: defaults)
        nextStore.prepareForAppLaunch()
        XCTAssertFalse(nextStore.isOnboardingPresented)
    }

    func testCompletedGuideCanBeOpenedAgainFromSettingsOrHelp() {
        let suiteName = "OnboardingTests.reopen.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(userDefaults: defaults)

        store.prepareForAppLaunch()
        store.completeOnboarding()
        store.presentOnboarding()

        XCTAssertTrue(store.isOnboardingPresented)
    }

    func testMainWindowDismissesAndReopensGuideWhenStorePresentationChanges() {
        let suiteName = "OnboardingTests.window-presentation.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = TokenityStore(userDefaults: defaults)
        store.prepareForAppLaunch()

        let hostingController = NSHostingController(
            rootView: TokenityMainWindowView(store: store)
        )
        let window = NSWindow(contentViewController: hostingController)
        window.isReleasedWhenClosed = false
        window.setContentSize(NSSize(width: 1_280, height: 820))
        window.makeKeyAndOrderFront(nil)
        defer {
            store.completeOnboarding()
            _ = waitUntil { window.attachedSheet == nil }
            window.close()
        }

        XCTAssertTrue(
            waitUntil { window.attachedSheet != nil },
            "The first-launch guide should be attached to the observed main window."
        )

        store.completeOnboarding(opening: .cluster)

        XCTAssertTrue(
            waitUntil { window.attachedSheet == nil },
            "Completing the guide should dismiss its real SwiftUI sheet."
        )
        XCTAssertEqual(store.selectedSection, .cluster)

        store.presentOnboarding()

        XCTAssertTrue(
            waitUntil { window.attachedSheet != nil },
            "Settings and Help should be able to present the guide again."
        )
    }

    func testEveryGuidePageRendersInLightAndDarkAppearances() throws {
        for page in TokenityOnboardingPage.allCases {
            for scheme in [ColorScheme.light, .dark] {
                let suiteName = "OnboardingTests.visual.\(page.id).\(scheme).\(UUID().uuidString)"
                let defaults = UserDefaults(suiteName: suiteName)!
                defer { defaults.removePersistentDomain(forName: suiteName) }
                let store = TokenityStore(userDefaults: defaults)
                let root = TokenityOnboardingView(initialPage: page)
                    .environmentObject(store)
                    .tokenityThemed()
                    .environment(\.colorScheme, scheme)
                let hostingView = NSHostingView(rootView: root)
                hostingView.frame = NSRect(x: 0, y: 0, width: 780, height: 590)
                hostingView.layoutSubtreeIfNeeded()

                let representation = try XCTUnwrap(
                    hostingView.bitmapImageRepForCachingDisplay(in: hostingView.bounds)
                )
                hostingView.cacheDisplay(in: hostingView.bounds, to: representation)
                let data = try XCTUnwrap(
                    representation.representation(using: .png, properties: [:])
                )
                let appearance = scheme == .light ? "light" : "dark"
                try data.write(
                    to: URL(
                        fileURLWithPath:
                            "/tmp/tokenity-onboarding-\(page.rawValue)-\(appearance).png"
                    )
                )
                XCTAssertGreaterThan(data.count, 10_000)
            }
        }
    }

    private func waitUntil(
        timeout: TimeInterval = 2,
        condition: () -> Bool
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() {
                return true
            }
            _ = RunLoop.main.run(
                mode: .default,
                before: Date().addingTimeInterval(0.01)
            )
        }
        return condition()
    }
}
