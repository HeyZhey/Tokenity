# Tokenity UI QA — 2026-07-30

This pass verifies the post-audit visual corrections without starting a cluster, loading a model,
restarting a remote Mac, or changing the production Agent.

## Real app captures

- `after-fixes/01` through `08`: all eight product pages in Light and Dark appearances.
- `after-fixes/09`: Onboarding in Light and Dark appearances.
- `after-fixes/06-api-copy-feedback-dark.png`: visible and accessible Copy confirmation.

The system appearance was Automatic before QA. It was temporarily changed to Light through System
Settings, then restored to Automatic after the Light pass.

## Deterministic visual-smoke captures

- `after-fixes/10-chat-markdown-*`: long Markdown, table, code, Thinking, route detail, and metrics.
- `after-fixes/11-chat-states-*`: reasoning, stopped, and failed message states.
- `after-fixes/12-chat-composer-focused-dark.png`: editable composer with the system-accent focus ring.
- `after-fixes/13-models-two-resident-*`: two resident instances, ready/busy, active/queued,
  capabilities, exact instance identity, and memory values.
- `after-fixes/14-menubar-mark-*`: 20 pt MenuBar mark with status dot.
- `after-fixes/15-sidebar-*`: full Logo lockup and sidebar selection at 218 pt.
- `after-fixes/page-matrix/`: every product page at 1120×720 and 1280×820 in Light and Dark.

The active send/stop/stream path remains covered by Swift tests using an in-memory transport rather
than a production model. The real app was intentionally kept in its existing stopped/no-model
state.
