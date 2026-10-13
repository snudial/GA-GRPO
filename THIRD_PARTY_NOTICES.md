# Third-party components

- **Molt**: NVIDIA-NeMo/labs-molt, Apache License 2.0. The modified runtime is distributed in `third_party/molt` with the original `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`. Its source identity and file digests are recorded in `third_party/molt.lock.json`. No original project Git history or credentials are included.
- **BFCL / Gorilla**: downloaded from the official repository at the pinned revision in `env/pins.sh`. Source and benchmark data retain their upstream license.
- **AppWorld**: installed through the official package. Its protected benchmark material and derivatives are subject to the upstream protected license. Protected data and training guides are not distributed by this repository; users provide authorized local guides. https://github.com/StonyBrookNLP/appworld
- **Qwen**: model weights are downloaded at revisions recorded in `config/models.json`. Their upstream model licenses apply.

The source snapshot bundled here includes modifications to the Molt trainer, rollout collection, checkpoint/resume handling, GPU sharing, and optional distillation hooks. These changes are identified by the source lock; the original Molt copyright and license notices are preserved.
