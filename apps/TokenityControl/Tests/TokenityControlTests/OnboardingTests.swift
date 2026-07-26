import AppKit
import SwiftUI
import XCTest
@testable import TokenityControl

@MainActor
final class OnboardingTests: XCTestCase {
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
}
