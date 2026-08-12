# ImageGen Prompt Set

The eight designs were generated with the built-in ImageGen workflow in
`ui-mockup` mode. Every page used three references:

1. the user-supplied Tokenity lynx logo;
2. the generated Overview master design;
3. the corresponding current SwiftUI page rendered from
   `feature/auto-model-router`.

## Shared prompt

Draw a high-fidelity 16:10 native macOS SwiftUI application window for Tokenity,
a control plane for distributed MLX inference across Apple-silicon Macs.

Use the supplied logo language faithfully: two angular alpine-blue lynx/cat ear
strokes above two circular endpoint dots, paired with the Tokenity wordmark.

All pages must share the same shell:

- warm pale-stone light sidebar `#ECEAE4`, never black or dark;
- warm ivory main canvas `#F7F5F0`;
- graphite text `#25272A`;
- alpine steel-blue accent `#315F8D`;
- muted green success `#53765B`;
- serif display typography for page and section titles;
- clean macOS sans-serif controls and monospaced measurements;
- matte paper-like panels, hairline rules, 8–10 px radii, minimal shadow;
- navigation groups Cluster and Operations with Overview, Cluster, Chat, Models,
  Network / RDMA, API Access, Logs, and Settings.

Keep every screen compact, restrained, readable, and implementable in SwiftUI.
Avoid a black sidebar, dark-mode appearance, literal animals, mascots, neon,
glossy 3D icons, glassmorphism, excessive gradients, oversized cards, mobile
layouts, browser chrome, and watermarks.

## Page-specific prompts

### 01 — Overview

Show a healthy running overview with Cluster State, Resident Models, and
Selected Macs. Include Running, Auto routing healthy, Tokenity Distributed,
Thunderbolt RDMA, Qwen3.5-122B-A10B-4bit, GLM-5.2-4bit, Mango, Kiwi, memory
bars, endpoints, and Open Chat. Use a quiet node topology in Cluster State.

### 02 — Cluster

Show Cluster Builder with Mango as Coordinator, Kiwi as Worker, and Apple as
Available. Connect Mango and Kiwi through Thunderbolt RDMA. Include Cluster
Setup, Tokenity Distributed, Create Cluster, Refresh Status, Stop Cluster,
Inference Acceleration, Native MTP Off/Auto/Required, Supported, and Cluster
Plan.

### 03 — Chat

Show an active conversation titled Distributed inference overview. Preserve the
three-column relationship between app navigation, transcript, and History.
Include automatic model routing, Qwen3.5-122B-A10B-4bit, 2 Macs · Thunderbolt
RDMA, Ready, Thinking, TTFT 210 ms, Total 2.80 s, 42.8 tok/s, Why selected,
Faster/Balanced/Smarter, and the Ask Tokenity composer.

### 04 — Models

Show Model Library, Resident Model Pool, and Available Models. Include two ready
resident models, Allow Auto, Keep Resident, MLX, 4-bit, Native MTP, reserved
memory, Use in Chat, Configure, Load, Stop, and one Draft only model. Make the
auto-router state immediately scannable.

### 05 — Network / RDMA

Show RDMA Readiness with Mango and Kiwi connected over Thunderbolt RDMA,
38.4 GB/s throughput, and 8.2 µs latency. Include Node Properties, Cluster Link
Plan, Coordinator/Worker roles, compact link-quality sparklines, Run
Diagnostics, and Refresh.

### 06 — API Access

Show an available OpenAI-Compatible API with Base URL
`http://<node-a-lan-ip>:9100/v1`, model `tokenity-auto`, API key Not required,
Supported Endpoints, a short Chat Completions curl sample, Copy controls, Test
Connection, and a Client → Tokenity Gateway → Resident Model setup flow.

### 07 — Logs

Show Activity with node and level filters, search, Export, event summary pills,
and a chronological table. Include Cluster created, RDMA link established,
model loaded, auto-router selection, chat completion, Mango, Kiwi, Gateway, and
Router. Add a light monospaced Recent activity console with Live and Follow
latest.

### 08 — Settings

Show Console, Backend Status, Interface, and Getting Started. Include discovery,
model library, access URL, Tokenity Distributed Stable target, Official MLX-LM
Experimental, Appearance System/Light/Dark with Light selected, the note
“Light mode uses a pale stone sidebar; Dark mode uses graphite”, Alpine Blue,
Show Guide, and Tokenity 0.1.0.
