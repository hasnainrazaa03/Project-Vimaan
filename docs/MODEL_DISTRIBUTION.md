# Model Distribution

Trained model checkpoints are **not** stored in the git repository. They are
distributed as **GitHub Release assets**.

- `ML/models/**` is gitignored.
- Git LFS (which previously held `*.safetensors` / `*.bin`) was **retired on
  2026-06-30**, along with a history rewrite that purged the weight blobs.

## Install a published model

```bash
scripts/fetch_model.sh            # latest model-* release
scripts/fetch_model.sh v10        # a specific version
```

This downloads and extracts the checkpoint into
`ML/models/vimaan_nlu_model_best/<version>/`, which is where the plugin,
dashboard, and `predict.py` look for it.

## Publish a model (maintainers)

```bash
scripts/publish_model.sh v10      # tars ML/models/.../v10 and uploads it
                                  # to the GitHub Release tagged "model-v10"
```

Both scripts require an authenticated `gh` CLI.

## Why Releases instead of LFS

Git LFS counts model weights against the repository's LFS **storage and
bandwidth quota** and bloats every clone. Release assets don't count against
LFS quota, aren't pulled on `git clone`, and are versioned independently of the
code. Each prod model is published under a `model-v<N>` tag.

## Note on reclaiming the old LFS storage

The history rewrite removes the weight pointers from git history, but GitHub
does **not** garbage-collect the underlying LFS objects automatically. The
~2.2 GB already uploaded to the repo's LFS store remains until either the
repository is deleted/recreated or GitHub Support is asked to purge it. New
weights published via Releases do not add to LFS usage.
