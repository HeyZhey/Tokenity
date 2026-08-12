import Foundation

enum TokenityDeploymentConfiguration {
    private static var environment: [String: String] {
        ProcessInfo.processInfo.environment
    }

    private static var bundledValues: [String: String] {
        guard let url = Bundle.main.url(
            forResource: "DeploymentConfiguration",
            withExtension: "json"
        ),
        let data = try? Data(contentsOf: url),
        let values = try? JSONDecoder().decode([String: String].self, from: data)
        else { return [:] }
        return values
    }

    private static func value(_ key: String) -> String? {
        environment[key] ?? bundledValues[key]
    }

    private static func configuredPath(_ key: String) -> String? {
        guard let value = value(key), !value.isEmpty else { return nil }
        let expanded = NSString(string: value).expandingTildeInPath
        guard NSString(string: expanded).isAbsolutePath else { return nil }
        return expanded
    }

    private static func configuredURL(_ key: String, fallback: URL) -> URL {
        guard let path = configuredPath(key) else { return fallback }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static var dataRoot: URL {
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        return configuredURL(
            "TOKENITY_DATA_ROOT",
            fallback: applicationSupport.appendingPathComponent("Tokenity", isDirectory: true)
        )
    }

    static var modelRoot: URL {
        configuredURL(
            "TOKENITY_MODEL_ROOT",
            fallback: dataRoot.appendingPathComponent("Models", isDirectory: true)
        )
    }

    static var runtimeRoot: URL {
        configuredURL(
            "TOKENITY_RUNTIME_ROOT",
            fallback: dataRoot.appendingPathComponent("Runtime", isDirectory: true)
        )
    }

    static var runtimePythonPath: String {
        if let configured = configuredPath("TOKENITY_RUNTIME_PYTHON") {
            return configured
        }
        return runtimeRoot
            .appendingPathComponent("current/.venv/bin/python", isDirectory: false)
            .path
    }

    static var agentBaseURL: String {
        value("TOKENITY_AGENT_URL") ?? "http://127.0.0.1:9100"
    }

    static var nodeAgentURLs: [String] {
        guard let value = value("TOKENITY_NODE_AGENT_URLS") else {
            return [agentBaseURL]
        }
        return value
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    static var hasExplicitNodeAgentURLs: Bool {
        value("TOKENITY_NODE_AGENT_URLS") != nil
    }

    static var h3CoordinatorAgentURL: String {
        value("TOKENITY_H3_COORDINATOR_AGENT") ?? agentBaseURL
    }

    static var h3WorkerAgentURL: String {
        value("TOKENITY_H3_WORKER_AGENT") ?? ""
    }

    static var h3ModelPath: String {
        if let configured = configuredPath("TOKENITY_H3_MODEL_PATH") {
            return configured
        }
        return modelRoot.appendingPathComponent("MiniMax-H3", isDirectory: true).path
    }

    static var h3BinaryPath: String {
        if let configured = configuredPath("TOKENITY_H3_BINARY_PATH") {
            return configured
        }
        return runtimeRoot.appendingPathComponent("current/bin/mlx-serve").path
    }
}
