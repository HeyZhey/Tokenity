import CryptoKit
import Foundation
import XCTest
@testable import TokenityControl

final class RuntimeBootstrapTests: XCTestCase {
    func testCatalogDecodesPinnedArtifactIdentity() throws {
        let catalog = try JSONDecoder().decode(
            TokenityRuntimeCatalog.self,
            from: Data(Self.catalogJSON.utf8)
        )

        XCTAssertEqual(catalog.runtimeID, "runtime-test")
        XCTAssertEqual(catalog.minimumMacOS, "14.0")
        XCTAssertEqual(catalog.packages["mlx"], "0.32.0")
        XCTAssertEqual(catalog.artifact.filename, "Runtime.pkg")
        XCTAssertEqual(catalog.artifact.urls, ["https://example.invalid/Runtime.pkg"])
    }

    func testArtifactVerificationRejectsTampering() async throws {
        let temporary = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }
        let artifact = temporary.appendingPathComponent("Runtime.pkg")
        try Data("verified-runtime".utf8).write(to: artifact)
        let catalog = catalog(for: artifact)
        let service = TokenityRuntimeBootstrapService(
            cachesDirectory: temporary.appendingPathComponent("cache"),
            installedPythonPath: temporary.appendingPathComponent("missing-python").path
        )

        try await service.verifyArtifact(at: artifact, catalog: catalog)
        try Data("tampered-runtime".utf8).write(to: artifact)

        do {
            try await service.verifyArtifact(at: artifact, catalog: catalog)
            XCTFail("Expected checksum or size verification to fail")
        } catch {
            XCTAssertTrue(
                error is TokenityRuntimeBootstrapError,
                "Unexpected error: \(error)"
            )
        }
    }

    func testCachedVerifiedArtifactIsDiscoveredWithoutNetwork() async throws {
        let temporary = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }
        let seed = temporary.appendingPathComponent("seed.pkg")
        try Data("cached-runtime".utf8).write(to: seed)
        let catalog = catalog(for: seed)
        let service = TokenityRuntimeBootstrapService(
            cachesDirectory: temporary.appendingPathComponent("cache"),
            installedPythonPath: temporary.appendingPathComponent("missing-python").path
        )
        let cached = service.cachedArtifactURL(for: catalog)
        try FileManager.default.createDirectory(
            at: cached.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.copyItem(at: seed, to: cached)

        let located = try await service.locateVerifiedArtifact(catalog: catalog)

        XCTAssertEqual(located?.standardizedFileURL, cached.standardizedFileURL)
    }

    func testOperatingSystemVersionComparisonIsNumeric() {
        XCTAssertTrue(
            TokenityRuntimeBootstrapService.isVersion(
                OperatingSystemVersion(majorVersion: 26, minorVersion: 2, patchVersion: 0),
                atLeast: OperatingSystemVersion(
                    majorVersion: 26,
                    minorVersion: 2,
                    patchVersion: 0
                )
            )
        )
        XCTAssertTrue(
            TokenityRuntimeBootstrapService.isVersion(
                OperatingSystemVersion(majorVersion: 27, minorVersion: 0, patchVersion: 0),
                atLeast: OperatingSystemVersion(
                    majorVersion: 26,
                    minorVersion: 9,
                    patchVersion: 0
                )
            )
        )
        XCTAssertFalse(
            TokenityRuntimeBootstrapService.isVersion(
                OperatingSystemVersion(majorVersion: 26, minorVersion: 1, patchVersion: 9),
                atLeast: OperatingSystemVersion(
                    majorVersion: 26,
                    minorVersion: 2,
                    patchVersion: 0
                )
            )
        )
    }

    private func catalog(for artifact: URL) -> TokenityRuntimeCatalog {
        let data = try! Data(contentsOf: artifact)
        let checksum = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
        return TokenityRuntimeCatalog(
            schemaVersion: 1,
            runtimeID: "runtime-test",
            tokenityVersion: "0.1.0",
            platform: "macos",
            architecture: "arm64",
            minimumMacOS: "14.0",
            pythonVersion: "3.12.13",
            packages: ["mlx": "0.32.0", "mlx-lm": "0.31.3"],
            runtimePayloadSHA256: String(repeating: "a", count: 64),
            artifact: .init(
                filename: "Runtime.pkg",
                packageIdentifier: "ai.tokenity.runtime.test",
                sizeBytes: Int64(data.count),
                sha256: checksum,
                urls: []
            )
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TokenityRuntimeTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }

    private static let catalogJSON = """
    {
      "schema_version": 1,
      "runtime_id": "runtime-test",
      "tokenity_version": "0.1.0",
      "platform": "macos",
      "architecture": "arm64",
      "minimum_macos": "14.0",
      "python_version": "3.12.13",
      "packages": {
        "mlx": "0.32.0",
        "mlx-lm": "0.31.3"
      },
      "runtime_payload_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "artifact": {
        "filename": "Runtime.pkg",
        "package_identifier": "ai.tokenity.runtime.test",
        "size_bytes": 10,
        "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "urls": ["https://example.invalid/Runtime.pkg"]
      }
    }
    """
}
