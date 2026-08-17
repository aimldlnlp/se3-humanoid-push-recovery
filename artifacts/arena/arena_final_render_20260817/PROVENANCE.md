# Arena final render provenance

This folder is a selected local copy of the remote run `arena_final_render_20260817`. The complete run remained on the SSH worker at:

`/home/aimldl/workspaces/se3-humanoid-push-recovery-arena-20260817/source/arena_final_render_20260817`

- SSH endpoint: public worker identity redacted
- Remote hostname: `remote_ssh_worker` (public identity redacted)
- Source version declared by the run: `a854c22`
- Remote workspace: execution copy without Git metadata; GitHub/local Git is authoritative
- Command: `experiments/adaptive_recovery_arena.py --output-root arena_final_render_20260817 --duration 5`
- Seed: `0`
- Python: `3.12.3`; NumPy `2.5.2`; MuJoCo `3.11.0`; OSQP `1.1.3`
- Model path: `models/unitree_g1/scene_push_recovery.xml`
- Worker model SHA-256: `613781e1b87d4e0d028332bfec4be9f2db53e2ddb252c0157bcaf04de88c0d76`
- Manifest configuration SHA-256: `8838329461a2d383e0564352aa69db40e9ddb13807635ac7f6c28836ee1653ef`
- Raw configuration SHA-256: `49e9a2288eaa6386d3e90e332eb15081f7593e6024949599c99ef6c073e8fd15`

The worker scene uses CRLF line endings; the tracked local scene uses LF. After normalizing line endings, their UTF-8 content hashes are identical (`43912c5f880adff88355c6cd1a887caec143543c7bc578234e1a369226fb0422`). The exact worker scene and configuration snapshots are retained in `provenance/`.

The copied evidence includes the hero GIF/MP4, four compact scenario MP4s, four telemetry PNGs, the aggregate CSV, the remote manifest and summary, and representative rendered frames. The remote manifest remains the complete artifact inventory; omitted frame sequences and PDFs/SVGs remain on the worker output root.
