import Darwin
import Foundation

struct DiscoveredNodeEndpoint {
    var agentURL: String
    var info: NodeInfoResponse
}

enum TokenityNodeDiscovery {
    private static let defaultPorts = [9_100, 9_200]
    private static let maximumSubnetHosts: UInt64 = 1_024
    private static let probeBatchSize = 64

    static func discover(seedOrigins: [URL]) async -> [DiscoveredNodeEndpoint] {
        let origins = candidateOrigins(seedOrigins: seedOrigins)
        guard !origins.isEmpty else { return [] }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }

        var discoveries: [DiscoveredNodeEndpoint] = []
        for batchStart in stride(from: 0, to: origins.count, by: probeBatchSize) {
            let batchEnd = min(batchStart + probeBatchSize, origins.count)
            let batch = origins[batchStart..<batchEnd]
            let responses = await withTaskGroup(
                of: (String, Data)?.self,
                returning: [(String, Data)].self
            ) { group in
                for origin in batch {
                    group.addTask {
                        guard let baseURL = URL(string: origin) else { return nil }
                        var request = URLRequest(
                            url: baseURL.appendingPathComponent("/v1/node/info")
                        )
                        // Older Agents can return a sizeable instance and
                        // log snapshot. Give local links enough time to
                        // transfer it without making unreachable hosts
                        // serialize the scan.
                        request.timeoutInterval = 1.0
                        do {
                            let (data, response) = try await session.data(for: request)
                            guard let httpResponse = response as? HTTPURLResponse,
                                  (200..<300).contains(httpResponse.statusCode)
                            else { return nil }
                            return (origin, data)
                        } catch {
                            return nil
                        }
                    }
                }
                var values: [(String, Data)] = []
                for await value in group {
                    if let value { values.append(value) }
                }
                return values
            }
            for (origin, data) in responses {
                guard let info = try? JSONDecoder().decode(NodeInfoResponse.self, from: data)
                else { continue }
                discoveries.append(DiscoveredNodeEndpoint(agentURL: origin, info: info))
            }
        }
        return discoveries.sorted { $0.agentURL < $1.agentURL }
    }

    static func candidateOrigins(seedOrigins: [URL]) -> [String] {
        var ports = Set(defaultPorts)
        var origins = Set<String>()
        for seed in seedOrigins {
            guard seed.scheme?.lowercased() == "http", seed.host != nil else { continue }
            if let port = seed.port { ports.insert(port) }
            if let origin = normalizedOrigin(host: seed.host, port: seed.port ?? 80) {
                origins.insert(origin)
            }
        }

        for address in localSubnetAddresses() {
            for port in ports {
                if let origin = normalizedOrigin(host: address, port: port) {
                    origins.insert(origin)
                }
            }
        }
        return origins.sorted()
    }

    private static func normalizedOrigin(host: String?, port: Int) -> String? {
        guard let host, !host.isEmpty, (1...65_535).contains(port) else { return nil }
        var components = URLComponents()
        components.scheme = "http"
        components.host = host
        components.port = port
        return components.url?.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private static func localSubnetAddresses() -> Set<String> {
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return [] }
        defer { freeifaddrs(interfaces) }

        var addresses = Set<String>()
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let interface = cursor?.pointee {
            defer { cursor = interface.ifa_next }
            guard let rawAddress = interface.ifa_addr,
                  rawAddress.pointee.sa_family == UInt8(AF_INET),
                  let rawNetmask = interface.ifa_netmask
            else { continue }
            let flags = Int32(interface.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }

            let localAddress = UnsafeRawPointer(rawAddress)
                .assumingMemoryBound(to: sockaddr_in.self).pointee
            let localNetmask = UnsafeRawPointer(rawNetmask)
                .assumingMemoryBound(to: sockaddr_in.self).pointee
            let hostAddress = UInt32(bigEndian: localAddress.sin_addr.s_addr)
            let declaredMask = UInt32(bigEndian: localNetmask.sin_addr.s_addr)
            guard isPrivateOrLinkLocal(hostAddress) else { continue }

            let declaredNetwork = hostAddress & declaredMask
            let declaredBroadcast = declaredNetwork | ~declaredMask
            let declaredHostCount = UInt64(declaredBroadcast) - UInt64(declaredNetwork) + 1
            let effectiveMask = declaredHostCount > maximumSubnetHosts
                ? UInt32(0xFFFF_FF00)
                : declaredMask
            let network = hostAddress & effectiveMask
            let broadcast = network | ~effectiveMask
            guard broadcast > network + 1 else { continue }

            for candidate in (network + 1)..<broadcast {
                if let address = ipv4String(hostOrderAddress: candidate) {
                    addresses.insert(address)
                }
            }
        }
        return addresses
    }

    private static func isPrivateOrLinkLocal(_ address: UInt32) -> Bool {
        let first = (address >> 24) & 0xFF
        let second = (address >> 16) & 0xFF
        return first == 10
            || (first == 172 && (16...31).contains(second))
            || (first == 192 && second == 168)
            || (first == 169 && second == 254)
    }

    private static func ipv4String(hostOrderAddress: UInt32) -> String? {
        var address = in_addr(s_addr: hostOrderAddress.bigEndian)
        var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        guard inet_ntop(AF_INET, &address, &buffer, socklen_t(INET_ADDRSTRLEN)) != nil
        else { return nil }
        return String(cString: buffer)
    }
}
