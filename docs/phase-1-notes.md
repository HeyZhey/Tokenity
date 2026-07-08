# Phase 1 Notes

## Reference Reading

- oMLX was used only as a product and engineering reference: native
  `NavigationSplitView`, compact settings-style rows, status pills, and app-owned
  Python process lifecycle.
- Old Tokenity docs were used only to extract the multi-node MLX-LM risks,
  known A/B RDMA topology, and launch-mode requirements.

## Known A/B RDMA Topology

- Mac A: `apple@192.168.5.23`, Thunderbolt IP `192.168.0.1`, active device
  `rdma_en4`
- Mac B: `probriefing@192.168.5.75`, Thunderbolt IP `192.168.0.2`, active
  device `rdma_en5`

The known successful JACCL hostfile shape is covered by
`tests/test_hostfile.py`.

## Current Boundary

This milestone produces dry-run launch plans and local process supervision
hooks. It does not claim that official MLX-LM chat is stable for the large
target model, and the Tokenity-owned distributed OpenAI server is still a
skeleton with explicit readiness phases.

