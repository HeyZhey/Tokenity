#!/usr/bin/env swift

import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

private struct Raster {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    subscript(x: Int, y: Int, channel: Int) -> UInt8 {
        get { pixels[((y * width + x) * 4) + channel] }
        set { pixels[((y * width + x) * 4) + channel] = newValue }
    }
}

private struct Bounds {
    var minX: Int
    var minY: Int
    var maxX: Int
    var maxY: Int

    var width: Int { maxX - minX }
    var height: Int { maxY - minY }

    func padded(by amount: Int, width: Int, height: Int) -> Bounds {
        Bounds(
            minX: max(0, minX - amount),
            minY: max(0, minY - amount),
            maxX: min(width, maxX + amount),
            maxY: min(height, maxY + amount)
        )
    }
}

private enum BrandAssetError: Error, CustomStringConvertible {
    case invalidArguments
    case invalidSource(URL)
    case missingContent
    case failedToEncode(URL)
    case iconutilFailed(String)

    var description: String {
        switch self {
        case .invalidArguments:
            return "Usage: generate-tokenity-brand-assets.swift <repository-root>"
        case .invalidSource(let url):
            return "Could not decode the source logo at \(url.path)"
        case .missingContent:
            return "The source logo did not contain a detectable blue brand mark."
        case .failedToEncode(let url):
            return "Could not encode \(url.path)"
        case .iconutilFailed(let output):
            return "iconutil failed: \(output)"
        }
    }
}

private func decodeRGBA(_ url: URL) throws -> Raster {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw BrandAssetError.invalidSource(url)
    }

    let width = image.width
    let height = image.height
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: &pixels,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        throw BrandAssetError.invalidSource(url)
    }
    context.interpolationQuality = .none
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return Raster(width: width, height: height, pixels: pixels)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
    return sorted[middle]
}

private func deriveTransparentLogo(from source: Raster) throws -> Raster {
    var backgroundSamples = [[Double](), [Double](), [Double]()]
    var foregroundSamples = [[Double](), [Double](), [Double]()]
    let border = max(24, min(source.width, source.height) / 18)

    for y in 0..<source.height {
        for x in 0..<source.width {
            let red = Double(source[x, y, 0])
            let green = Double(source[x, y, 1])
            let blue = Double(source[x, y, 2])
            let maximum = max(red, green, blue)
            let minimum = min(red, green, blue)
            let isBorder = x < border || y < border
                || x >= source.width - border || y >= source.height - border

            if isBorder, maximum - minimum < 8, minimum > 225 {
                backgroundSamples[0].append(red)
                backgroundSamples[1].append(green)
                backgroundSamples[2].append(blue)
            }

            if blue - red > 45, blue - green > 20, maximum < 205 {
                foregroundSamples[0].append(red)
                foregroundSamples[1].append(green)
                foregroundSamples[2].append(blue)
            }
        }
    }

    let background = backgroundSamples.map(median)
    let foreground = foregroundSamples.map(median)
    guard background.allSatisfy({ $0 > 220 }),
          foregroundSamples.allSatisfy({ !$0.isEmpty })
    else {
        throw BrandAssetError.missingContent
    }

    var output = Raster(
        width: source.width,
        height: source.height,
        pixels: [UInt8](repeating: 0, count: source.width * source.height * 4)
    )

    for y in 0..<source.height {
        for x in 0..<source.width {
            let components = (0..<3).map { Double(source[x, y, $0]) }
            let coolSignal = components[2] - max(components[0], components[1])
            let blueRedSignal = components[2] - components[0]
            guard coolSignal > 0.8, blueRedSignal > 1.6 else { continue }

            var estimates: [Double] = []
            for channel in 0..<3 {
                let denominator = background[channel] - foreground[channel]
                guard denominator > 18 else { continue }
                estimates.append((background[channel] - components[channel]) / denominator)
            }
            var alpha = min(max(median(estimates), 0), 1)
            if alpha < 0.018 { continue }
            if alpha > 0.82 { alpha = 1 }

            for channel in 0..<3 {
                let recovered: Double
                if alpha >= 0.999 {
                    recovered = components[channel]
                } else {
                    recovered = (
                        components[channel] - ((1 - alpha) * background[channel])
                    ) / alpha
                }
                output[x, y, channel] = UInt8(min(max(recovered.rounded(), 0), 255))
            }
            output[x, y, 3] = UInt8((alpha * 255).rounded())
        }
    }

    return output
}

private func contentBounds(
    in raster: Raster,
    alphaThreshold: UInt8 = 8,
    maximumY: Int? = nil
) throws -> Bounds {
    var bounds: Bounds?
    let upperY = min(maximumY ?? raster.height, raster.height)
    for y in 0..<upperY {
        for x in 0..<raster.width where raster[x, y, 3] >= alphaThreshold {
            if let current = bounds {
                bounds = Bounds(
                    minX: min(current.minX, x),
                    minY: min(current.minY, y),
                    maxX: max(current.maxX, x + 1),
                    maxY: max(current.maxY, y + 1)
                )
            } else {
                bounds = Bounds(minX: x, minY: y, maxX: x + 1, maxY: y + 1)
            }
        }
    }
    guard let bounds else { throw BrandAssetError.missingContent }
    return bounds
}

private func markSplit(in raster: Raster, fullBounds: Bounds) throws -> Int {
    var occupiedRows: [Int] = []
    for y in fullBounds.minY..<fullBounds.maxY {
        var count = 0
        for x in fullBounds.minX..<fullBounds.maxX where raster[x, y, 3] >= 28 {
            count += 1
        }
        if count >= 4 { occupiedRows.append(y) }
    }
    guard occupiedRows.count > 2 else { throw BrandAssetError.missingContent }

    var candidates: [(gap: Int, split: Int)] = []
    for pair in zip(occupiedRows, occupiedRows.dropFirst()) {
        let gap = pair.1 - pair.0
        let relativePosition = Double(pair.0 - fullBounds.minY) / Double(fullBounds.height)
        if gap > 8, relativePosition > 0.35, relativePosition < 0.82 {
            candidates.append((gap, (pair.0 + pair.1) / 2))
        }
    }
    guard let candidate = candidates.max(by: { $0.gap < $1.gap }) else {
        throw BrandAssetError.missingContent
    }
    return candidate.split
}

private func crop(_ raster: Raster, to bounds: Bounds) -> Raster {
    var result = Raster(
        width: bounds.width,
        height: bounds.height,
        pixels: [UInt8](repeating: 0, count: bounds.width * bounds.height * 4)
    )
    for y in 0..<bounds.height {
        for x in 0..<bounds.width {
            for channel in 0..<4 {
                result[x, y, channel] = raster[bounds.minX + x, bounds.minY + y, channel]
            }
        }
    }
    return result
}

private func tint(_ raster: Raster, rgb: UInt32) -> Raster {
    var result = raster
    let red = UInt8((rgb >> 16) & 0xFF)
    let green = UInt8((rgb >> 8) & 0xFF)
    let blue = UInt8(rgb & 0xFF)

    for y in 0..<result.height {
        for x in 0..<result.width where result[x, y, 3] > 0 {
            result[x, y, 0] = red
            result[x, y, 1] = green
            result[x, y, 2] = blue
        }
    }
    return result
}

private func makeCGImage(_ raster: Raster) throws -> CGImage {
    let data = Data(raster.pixels) as CFData
    guard let provider = CGDataProvider(data: data) else {
        throw BrandAssetError.missingContent
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    guard let image = CGImage(
        width: raster.width,
        height: raster.height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: raster.width * 4,
        space: colorSpace,
        bitmapInfo: CGBitmapInfo(
            rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    ) else {
        throw BrandAssetError.missingContent
    }
    return image
}

private func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw BrandAssetError.failedToEncode(url)
    }
    CGImageDestinationAddImage(destination, image, [
        kCGImagePropertyHasAlpha: true,
        kCGImagePropertyPNGInterlaceType: 0,
    ] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
        throw BrandAssetError.failedToEncode(url)
    }
}

private func renderIcon(mark: CGImage, size: Int) throws -> CGImage {
    var pixels = [UInt8](repeating: 0, count: size * size * 4)
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: &pixels,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: size * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        throw BrandAssetError.missingContent
    }

    let scale = CGFloat(size) / 1024
    let plate = CGRect(x: 80 * scale, y: 80 * scale, width: 864 * scale, height: 864 * scale)
    let platePath = CGPath(
        roundedRect: plate,
        cornerWidth: 188 * scale,
        cornerHeight: 188 * scale,
        transform: nil
    )
    context.addPath(platePath)
    context.setFillColor(
        CGColor(
            srgbRed: 235 / 255,
            green: 241 / 255,
            blue: 246 / 255,
            alpha: 1
        )
    )
    context.fillPath()
    context.addPath(platePath)
    context.setStrokeColor(
        CGColor(
            srgbRed: 190 / 255,
            green: 204 / 255,
            blue: 216 / 255,
            alpha: 0.85
        )
    )
    context.setLineWidth(4 * scale)
    context.strokePath()

    let safeWidth = 560 * scale
    let aspect = CGFloat(mark.height) / CGFloat(mark.width)
    let markHeight = safeWidth * aspect
    let destination = CGRect(
        x: (CGFloat(size) - safeWidth) / 2,
        y: ((CGFloat(size) - markHeight) / 2) + (18 * scale),
        width: safeWidth,
        height: markHeight
    )
    context.interpolationQuality = .high
    context.draw(mark, in: destination)

    guard let image = context.makeImage() else {
        throw BrandAssetError.missingContent
    }
    return image
}

private func makeAppIcon(mark: CGImage, outputURL: URL) throws {
    let fileManager = FileManager.default
    let temporaryRoot = fileManager.temporaryDirectory
        .appendingPathComponent("tokenity-brand-\(UUID().uuidString)", isDirectory: true)
    let iconset = temporaryRoot.appendingPathComponent("AppIcon.iconset", isDirectory: true)
    try fileManager.createDirectory(at: iconset, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: temporaryRoot) }

    let files: [(String, Int)] = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]
    for (name, size) in files {
        try writePNG(renderIcon(mark: mark, size: size), to: iconset.appendingPathComponent(name))
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
    process.arguments = ["-c", "icns", iconset.path, "-o", outputURL.path]
    let output = Pipe()
    process.standardOutput = output
    process.standardError = output
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        let data = output.fileHandleForReading.readDataToEndOfFile()
        throw BrandAssetError.iconutilFailed(String(decoding: data, as: UTF8.self))
    }
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw BrandAssetError.invalidArguments
    }
    let fileManager = FileManager.default
    let repositoryRoot = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        .standardizedFileURL
    let sourceURL = repositoryRoot.appendingPathComponent("Logo-New.png")
    let resourceDirectory = repositoryRoot
        .appendingPathComponent("apps/TokenityControl/Sources/TokenityControl/Resources", isDirectory: true)
    let iconURL = repositoryRoot
        .appendingPathComponent("apps/TokenityControl/Resources/AppIcon.icns")

    try fileManager.createDirectory(at: resourceDirectory, withIntermediateDirectories: true)

    let source = try decodeRGBA(sourceURL)
    let transparent = try deriveTransparentLogo(from: source)
    let fullBounds = try contentBounds(in: transparent)
    let split = try markSplit(in: transparent, fullBounds: fullBounds)
    let fullPadding = max(22, fullBounds.width / 28)
    let markPadding = max(22, fullBounds.width / 24)
    let lockup = crop(
        transparent,
        to: fullBounds.padded(by: fullPadding, width: source.width, height: source.height)
    )
    let markBounds = try contentBounds(in: transparent, maximumY: split)
        .padded(by: markPadding, width: source.width, height: split)
    let mark = crop(transparent, to: markBounds)
    let darkLockup = tint(lockup, rgb: 0x73B7EA)
    let darkMark = tint(mark, rgb: 0x73B7EA)

    let lockupURL = resourceDirectory.appendingPathComponent("TokenityBrandLockup.png")
    let darkLockupURL = resourceDirectory.appendingPathComponent("TokenityBrandLockupDark.png")
    let markURL = resourceDirectory.appendingPathComponent("TokenityBrandMark.png")
    let darkMarkURL = resourceDirectory.appendingPathComponent("TokenityBrandMarkDark.png")
    let lockupImage = try makeCGImage(lockup)
    let darkLockupImage = try makeCGImage(darkLockup)
    let markImage = try makeCGImage(mark)
    let darkMarkImage = try makeCGImage(darkMark)
    try writePNG(lockupImage, to: lockupURL)
    try writePNG(darkLockupImage, to: darkLockupURL)
    try writePNG(markImage, to: markURL)
    try writePNG(darkMarkImage, to: darkMarkURL)
    try makeAppIcon(mark: markImage, outputURL: iconURL)

    print("Generated \(lockupURL.path) (\(lockup.width)x\(lockup.height))")
    print("Generated \(darkLockupURL.path) (\(darkLockup.width)x\(darkLockup.height))")
    print("Generated \(markURL.path) (\(mark.width)x\(mark.height))")
    print("Generated \(darkMarkURL.path) (\(darkMark.width)x\(darkMark.height))")
    print("Generated \(iconURL.path)")
} catch {
    FileHandle.standardError.write(Data("Brand asset generation failed: \(error)\n".utf8))
    exit(1)
}
