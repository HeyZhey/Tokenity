import AppKit
import SwiftUI

struct TokenityTheme {
    let window: Color
    let sidebar: Color
    let surface: Color
    let raisedSurface: Color
    let group: Color
    let border: Color
    let strongBorder: Color
    let rowSeparator: Color
    let selection: Color
    let hover: Color
    let text: Color
    let secondaryText: Color
    let tertiaryText: Color
    let accent: Color
    let controlAccent: Color
    let accentText: Color
    let success: Color
    let warning: Color
    let danger: Color
    let code: Color

    static func resolve(_ scheme: ColorScheme) -> TokenityTheme {
        if scheme == .dark {
            return TokenityTheme(
                window: color(0x20211F),
                sidebar: color(0x1A1B1A),
                surface: color(0x272825),
                raisedSurface: color(0x2D2E2B),
                group: color(0x292A27),
                border: color(0x40423E),
                strongBorder: color(0x555852),
                rowSeparator: color(0x3B3D39),
                selection: color(0x293A40),
                hover: color(0x30322F),
                text: color(0xF1F0EA),
                secondaryText: color(0xB5B5AE),
                tertiaryText: color(0x858780),
                accent: color(0x7B9EAC),
                controlAccent: color(0x4F7485),
                accentText: color(0xFFFFFF),
                success: color(0x66B873),
                warning: color(0xE3A34F),
                danger: color(0xE56A66),
                code: color(0x191A18)
            )
        }

        return TokenityTheme(
            window: color(0xF6F4EF),
            sidebar: color(0xEFEEE8),
            surface: color(0xFCFBF8),
            raisedSurface: color(0xFFFFFF),
            group: color(0xFAF9F5),
            border: color(0xD8D6CF),
            strongBorder: color(0xC3C2BA),
            rowSeparator: color(0xE2E0DA),
            selection: color(0xDDE7E9),
            hover: color(0xE9E8E2),
            text: color(0x292B29),
            secondaryText: color(0x62645F),
            tertiaryText: color(0x8A8C85),
            accent: color(0x456B7C),
            controlAccent: color(0x456B7C),
            accentText: color(0xFFFFFF),
            success: color(0x3E8A4B),
            warning: color(0xB96E17),
            danger: color(0xB94742),
            code: color(0xF1F2EF)
        )
    }

    private static func color(_ rgb: UInt32) -> Color {
        Color(
            .sRGB,
            red: Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >> 8) & 0xFF) / 255,
            blue: Double(rgb & 0xFF) / 255,
            opacity: 1
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
        let theme = TokenityTheme.resolve(scheme)
        content
            .environment(\.tokenityTheme, theme)
            .tint(theme.accent)
    }
}

extension View {
    func tokenityThemed() -> some View {
        modifier(TokenityThemeBinder())
    }
}

extension Font {
    static func tokenityDisplay(_ size: CGFloat = 32) -> Font {
        .system(size: size, weight: .semibold, design: .serif)
    }

    static func tokenityTitle(_ size: CGFloat = 32) -> Font {
        tokenityDisplay(size)
    }

    static func tokenitySectionTitle(_ size: CGFloat = 18) -> Font {
        .system(size: size, weight: .medium, design: .serif)
    }

    static func tokenityBody(_ size: CGFloat = 13, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    static func tokenityCaption(_ size: CGFloat = 11, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    static func tokenityText(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        tokenityBody(size, weight: weight)
    }

    static func tokenityMono(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}
