# bfcl_protobuf_overlay

The pinned BFCL CLI venvs ship a `protobuf` too old for the vLLM 0.28 OpenAI
server they talk to in `scripts/official_pool.sh`. Rather than
upgrading a pinned venv (which would change the protocol pin), the pool puts a
one-package overlay first on `PYTHONPATH`:

```
$BFCL_PROTOBUF_OVERLAY/
  google/
    protobuf -> <a venv with a current protobuf>/lib/python3.12/site-packages/google/protobuf
```

`scripts/setup.sh` creates it under `$CACHE_ROOT/bfcl_protobuf_overlay`,
linking to the Molt venv's `google/protobuf`. It is a machine-local symlink, so
it is not stored in git; `config/env.sh` points `BFCL_PROTOBUF_OVERLAY` at it.
