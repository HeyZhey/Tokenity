import AppKit
import SwiftUI

enum TokenityBrandAsset: String {
    case lockup = "TokenityBrandLockup"
    case mark = "TokenityBrandMark"

    func resourceName(for scheme: ColorScheme) -> String {
        scheme == .dark ? "\(rawValue)Dark" : rawValue
    }
}

enum TokenityBrandAssets {
    static let menuBarMarkSize = NSSize(width: 20, height: 12)

    static func image(_ asset: TokenityBrandAsset, for scheme: ColorScheme = .light) -> NSImage {
        let resourceName = asset.resourceName(for: scheme)
        if let url = Bundle.main.url(forResource: resourceName, withExtension: "png"),
           let image = NSImage(contentsOf: url) {
            return image
        }
        guard let url = Bundle.module.url(forResource: resourceName, withExtension: "png"),
              let image = NSImage(contentsOf: url) else {
            assertionFailure("Missing bundled Tokenity brand asset: \(resourceName).png")
            return NSImage(size: NSSize(width: 1, height: 1))
        }
        return image
    }

    static func menuBarMarkImage(for scheme: ColorScheme) -> NSImage {
        let source = image(.mark, for: scheme)
        guard let result = source.copy() as? NSImage else {
            assertionFailure("Unable to copy bundled Tokenity menu bar mark.")
            return NSImage(size: menuBarMarkSize)
        }
        // MenuBarExtra may inspect the NSImage intrinsic size before applying
        // SwiftUI layout modifiers. Keep the high-resolution bitmap, but give
        // the native image a menu-bar-sized point footprint.
        result.size = menuBarMarkSize
        return result
    }
}

struct TokenityBrandLockup: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Image(nsImage: TokenityBrandAssets.image(.lockup, for: scheme))
            .resizable()
            .interpolation(.high)
            .aspectRatio(contentMode: .fit)
            .accessibilityHidden(true)
    }
}

struct TokenityBrandMark: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Image(nsImage: TokenityBrandAssets.image(.mark, for: scheme))
            .resizable()
            .interpolation(.high)
            .aspectRatio(contentMode: .fit)
            .accessibilityHidden(true)
    }
}

struct TokenityMenuBarBrandMark: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Image(nsImage: TokenityBrandAssets.menuBarMarkImage(for: scheme))
            .interpolation(.high)
            .frame(
                width: TokenityBrandAssets.menuBarMarkSize.width,
                height: TokenityBrandAssets.menuBarMarkSize.height
            )
            .clipped()
            .accessibilityHidden(true)
    }
}
