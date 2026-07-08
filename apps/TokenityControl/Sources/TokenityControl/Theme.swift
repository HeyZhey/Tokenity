import AppKit
import SwiftUI

struct TokenityTheme {
    let window: Color
    let sidebar: Color
    let group: Color
    let border: Color
    let rowSeparator: Color
    let text: Color
    let secondaryText: Color
    let tertiaryText: Color
    let accent: Color
    let success: Color
    let warning: Color
    let danger: Color
    let code: Color

    static func resolve(_ scheme: ColorScheme) -> TokenityTheme {
        TokenityTheme(
            window: Color(nsColor: scheme == .dark ? .underPageBackgroundColor : .windowBackgroundColor),
            sidebar: Color(nsColor: .windowBackgroundColor),
            group: Color(nsColor: .labelColor).opacity(scheme == .dark ? 0.035 : 0.045),
            border: Color(nsColor: .separatorColor),
            rowSeparator: Color(nsColor: .separatorColor).opacity(0.75),
            text: Color(nsColor: .labelColor),
            secondaryText: Color(nsColor: .secondaryLabelColor),
            tertiaryText: Color(nsColor: .tertiaryLabelColor),
            accent: Color(nsColor: .controlAccentColor),
            success: Color(nsColor: .systemGreen),
            warning: Color(nsColor: .systemOrange),
            danger: Color(nsColor: .systemRed),
            code: Color(nsColor: .textBackgroundColor)
        )
    }
}

private struct TokenityThemeKey: EnvironmentKey {
    static let defaultValue = TokenityTheme.resolve(.light)
}

extension EnvironmentValues {
    var tokenityTheme: TokenityTheme {
        get { self[TokenityThemeKey.self] }
        set { self[TokenityThemeKey.self] = newValue }
    }
}

private struct TokenityThemeBinder: ViewModifier {
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content.environment(\.tokenityTheme, TokenityTheme.resolve(scheme))
    }
}

extension View {
    func tokenityThemed() -> some View {
        modifier(TokenityThemeBinder())
    }
}

extension Font {
    static func tokenityText(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }

    static func tokenityMono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

