import AppKit
import Combine
import CryptoKit
import Foundation

struct TokenityRuntimeCatalog: Codable, Equatable {
    struct Artifact: Codable, Equatable {
        let filename: String
        let packageIdentifier: String
        let sizeBytes: Int64
        let sha256: String
        let urls: [String]

        enum CodingKeys: String, CodingKey {
            case filename
            case packageIdentifier = "package_identifier"
            case sizeBytes = "size_bytes"
            case sha256
            case urls
        }
    }

    let schemaVersion: Int
    let runtimeID: String
    let tokenityVersion: String
    let platform: String
    let architecture: String
    let minimumMacOS: String
    let pythonVersion: String
    let packages: [String: String]
    let runtimePayloadSHA256: String
    let artifact: Artifact

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case runtimeID = "runtime_id"
        case tokenityVersion = "tokenity_version"
        case platform
        case architecture
        case minimumMacOS = "minimum_macos"
        case pythonVersion = "python_version"
        case packages
        case runtimePayloadSHA256 = "runtime_payload_sha256"
        case artifact
    }
}

enum TokenityRuntimeBootstrapError: LocalizedError {
    case catalogMissing
    case invalidCatalog(String)
    case incompatibleMac(String)
    case artifactMissing
    case artifactSize(expected: Int64, observed: Int64)
    case artifactChecksum
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .catalogMissing:
            return "This Tokenity build does not contain a Runtime catalog."
        case let .invalidCatalog(detail):
            return "The Runtime catalog is invalid: \(detail)"
        case let .incompatibleMac(detail):
            return detail
        case .artifactMissing:
            return "The Runtime installer is unavailable locally and has no download URL."
        case let .artifactSize(expected, observed):
            return "The Runtime installer size is invalid (expected \(expected), found \(observed))."
        case .artifactChecksum:
            return "The Runtime installer failed SHA-256 verification."
        case let .downloadFailed(detail):
            return "The Runtime download failed: \(detail)"
        }
    }
}

struct TokenityRuntimeBootstrapService {
    let bundle: Bundle
    let fileManager: FileManager
    let session: URLSession
    let cachesDirectory: URL
    let installedPythonPath: String

    init(
        bundle: Bundle = .main,
        fileManager: FileManager = .default,
        session: URLSession = .shared,
        cachesDirectory: URL? = nil,
        installedPythonPath: String = TokenityDeploymentConfiguration.runtimePythonPath
    ) {
        self.bundle = bundle
        self.fileManager = fileManager
        self.session = session
        self.cachesDirectory = cachesDirectory
            ?? fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first!
        self.installedPythonPath = installedPythonPath
    }

    var isRuntimeInstalled: Bool {
        fileManager.isExecutableFile(atPath: installedPythonPath)
    }

    func loadCatalog() throws -> TokenityRuntimeCatalog {
        guard let url = bundle.url(forResource: "RuntimeCatalog", withExtension: "json") else {
            throw TokenityRuntimeBootstrapError.catalogMissing
        }
        let catalog = try JSONDecoder().decode(
            TokenityRuntimeCatalog.self,
            from: Data(contentsOf: url)
        )
        guard catalog.schemaVersion == 1,
              catalog.platform == "macos",
              catalog.architecture == "arm64",
              catalog.artifact.sizeBytes > 0,
              catalog.artifact.sha256.count == 64
        else {
            throw TokenityRuntimeBootstrapError.invalidCatalog("unsupported fields")
        }
        return catalog
    }

    func ensureCompatibleHost(for catalog: TokenityRuntimeCatalog) throws {
        #if !arch(arm64)
            throw TokenityRuntimeBootstrapError.incompatibleMac(
                "Tokenity Runtime requires an Apple-silicon Mac."
            )
        #endif

        let observed = ProcessInfo.processInfo.operatingSystemVersion
        let required = Self.operatingSystemVersion(catalog.minimumMacOS)
        guard Self.isVersion(observed, atLeast: required) else {
            throw TokenityRuntimeBootstrapError.incompatibleMac(
                "Runtime \(catalog.runtimeID) requires macOS "
                    + "\(catalog.minimumMacOS) or newer."
            )
        }
    }

    func locateVerifiedArtifact(
        catalog: TokenityRuntimeCatalog
    ) async throws -> URL? {
        try ensureCompatibleHost(for: catalog)
        for candidate in candidateArtifactURLs(for: catalog) {
            guard fileManager.isReadableFile(atPath: candidate.path) else { continue }
            do {
                try await verifyArtifact(at: candidate, catalog: catalog)
                return candidate
            } catch {
                continue
            }
        }
        return nil
    }

    func acquireArtifact(catalog: TokenityRuntimeCatalog) async throws -> URL {
        if let local = try await locateVerifiedArtifact(catalog: catalog) {
            return local
        }
        guard !catalog.artifact.urls.isEmpty else {
            throw TokenityRuntimeBootstrapError.artifactMissing
        }

        var lastError: Error?
        for value in catalog.artifact.urls {
            guard let remoteURL = URL(string: value),
                  remoteURL.scheme?.lowercased() == "https"
            else {
                lastError = TokenityRuntimeBootstrapError.invalidCatalog(
                    "Runtime URLs must use HTTPS"
                )
                continue
            }
            do {
                let (temporaryURL, response) = try await session.download(from: remoteURL)
                if let http = response as? HTTPURLResponse,
                   !(200..<300).contains(http.statusCode) {
                    throw TokenityRuntimeBootstrapError.downloadFailed(
                        "server returned HTTP \(http.statusCode)"
                    )
                }
                try await verifyArtifact(at: temporaryURL, catalog: catalog)
                return try cacheDownloadedArtifact(
                    temporaryURL,
                    catalog: catalog
                )
            } catch {
                lastError = error
            }
        }
        throw lastError ?? TokenityRuntimeBootstrapError.artifactMissing
    }

    func verifyArtifact(
        at url: URL,
        catalog: TokenityRuntimeCatalog
    ) async throws {
        let expectedSize = catalog.artifact.sizeBytes
        let expectedSHA256 = catalog.artifact.sha256.lowercased()
        let observed = try await Task.detached(priority: .utility) {
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            let size = Int64(values.fileSize ?? 0)
            let checksum = try Self.sha256(of: url)
            return (size, checksum)
        }.value

        guard observed.0 == expectedSize else {
            throw TokenityRuntimeBootstrapError.artifactSize(
                expected: expectedSize,
                observed: observed.0
            )
        }
        guard observed.1 == expectedSHA256 else {
            throw TokenityRuntimeBootstrapError.artifactChecksum
        }
    }

    func candidateArtifactURLs(for catalog: TokenityRuntimeCatalog) -> [URL] {
        var candidates: [URL] = []
        if let resourceURL = bundle.resourceURL {
            candidates.append(
                resourceURL.appendingPathComponent(catalog.artifact.filename)
            )
        }
        candidates.append(
            bundle.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent(catalog.artifact.filename)
        )
        candidates.append(
            cachedArtifactURL(for: catalog)
        )

        if let volumes = fileManager.mountedVolumeURLs(
            includingResourceValuesForKeys: [.isDirectoryKey],
            options: [.skipHiddenVolumes]
        ) {
            candidates.append(contentsOf: volumes.map {
                $0.appendingPathComponent(catalog.artifact.filename)
            })
        }

        var unique: [URL] = []
        var seen = Set<String>()
        for candidate in candidates where seen.insert(candidate.standardizedFileURL.path).inserted {
            unique.append(candidate)
        }
        return unique
    }

    func cachedArtifactURL(for catalog: TokenityRuntimeCatalog) -> URL {
        cachesDirectory
            .appendingPathComponent("Tokenity", isDirectory: true)
            .appendingPathComponent("Runtime", isDirectory: true)
            .appendingPathComponent(catalog.runtimeID, isDirectory: true)
            .appendingPathComponent(catalog.artifact.filename)
    }

    private func cacheDownloadedArtifact(
        _ temporaryURL: URL,
        catalog: TokenityRuntimeCatalog
    ) throws -> URL {
        let destination = cachedArtifactURL(for: catalog)
        let directory = destination.deletingLastPathComponent()
        try fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let staged = directory.appendingPathComponent(".\(catalog.artifact.filename).part")
        if fileManager.fileExists(atPath: staged.path) {
            try fileManager.removeItem(at: staged)
        }
        try fileManager.moveItem(at: temporaryURL, to: staged)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: staged, to: destination)
        return destination
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            hash.update(data: data)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func operatingSystemVersion(_ value: String) -> OperatingSystemVersion {
        let pieces = value.split(separator: ".").compactMap { Int($0) }
        return OperatingSystemVersion(
            majorVersion: pieces.indices.contains(0) ? pieces[0] : 0,
            minorVersion: pieces.indices.contains(1) ? pieces[1] : 0,
            patchVersion: pieces.indices.contains(2) ? pieces[2] : 0
        )
    }

    static func isVersion(
        _ observed: OperatingSystemVersion,
        atLeast required: OperatingSystemVersion
    ) -> Bool {
        let lhs = (observed.majorVersion, observed.minorVersion, observed.patchVersion)
        let rhs = (required.majorVersion, required.minorVersion, required.patchVersion)
        if lhs.0 != rhs.0 { return lhs.0 > rhs.0 }
        if lhs.1 != rhs.1 { return lhs.1 > rhs.1 }
        return lhs.2 >= rhs.2
    }
}

@MainActor
final class TokenityRuntimeBootstrapModel: ObservableObject {
    enum State: Equatable {
        case checking
        case installed
        case bundled
        case downloadAvailable
        case downloading
        case waitingForInstaller
        case failed(String)
    }

    @Published private(set) var state: State = .checking
    @Published private(set) var catalog: TokenityRuntimeCatalog?

    private let service: TokenityRuntimeBootstrapService
    private var localArtifact: URL?

    init(service: TokenityRuntimeBootstrapService = .init()) {
        self.service = service
    }

    var title: String {
        switch state {
        case .checking: return "Checking Runtime"
        case .installed: return "Runtime already installed"
        case .bundled: return "Runtime installer included"
        case .downloadAvailable: return "Runtime available to download"
        case .downloading: return "Downloading and verifying Runtime"
        case .waitingForInstaller: return "Continue in Installer"
        case .failed: return "Runtime setup needs attention"
        }
    }

    var detail: String {
        let compatibility = catalog.map {
            "Runtime \($0.runtimeID) · Apple silicon · macOS \($0.minimumMacOS)+"
        }
        switch state {
        case .checking:
            return "Inspecting this Mac and the Tokenity installer."
        case .installed:
            return compatibility.map { "\($0). A local Runtime is already present." }
                ?? "A local Tokenity Runtime is already present."
        case .bundled:
            return compatibility.map {
                "\($0). No additional Runtime download is required."
            } ?? "No additional Runtime download is required."
        case .downloadAvailable:
            return compatibility.map {
                "\($0). Tokenity will verify the fixed package before opening it."
            } ?? "Tokenity will verify the package before opening it."
        case .downloading:
            return "The package is accepted only when its size and SHA-256 match the catalog."
        case .waitingForInstaller:
            return "Installer.app will request administrator approval. Repeat this on every inference Mac."
        case let .failed(message):
            return message
        }
    }

    var actionTitle: String? {
        switch state {
        case .bundled:
            return "Open Runtime Installer"
        case .downloadAvailable:
            return "Download Runtime Installer"
        case .installed:
            return "Reinstall Runtime"
        case .failed:
            return "Retry"
        case .checking, .downloading, .waitingForInstaller:
            return nil
        }
    }

    var isBusy: Bool {
        state == .checking || state == .downloading
    }

    var hasFailed: Bool {
        if case .failed = state {
            return true
        }
        return false
    }

    func refresh() async {
        state = .checking
        do {
            let loadedCatalog = try service.loadCatalog()
            try service.ensureCompatibleHost(for: loadedCatalog)
            catalog = loadedCatalog
            localArtifact = try await service.locateVerifiedArtifact(
                catalog: loadedCatalog
            )
            if service.isRuntimeInstalled {
                state = .installed
            } else if localArtifact != nil {
                state = .bundled
            } else {
                state = .downloadAvailable
            }
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func performPrimaryAction() {
        guard !isBusy else { return }
        Task {
            do {
                let loadedCatalog = try catalog ?? service.loadCatalog()
                catalog = loadedCatalog
                state = localArtifact == nil ? .downloading : state
                let artifact = try await service.acquireArtifact(catalog: loadedCatalog)
                localArtifact = artifact
                guard NSWorkspace.shared.open(artifact) else {
                    throw TokenityRuntimeBootstrapError.downloadFailed(
                        "Installer.app could not open the verified package"
                    )
                }
                state = .waitingForInstaller
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }
}
