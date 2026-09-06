"""Export a trained V-Max checkpoint's policy as a submission `weights.pkl`.

V-Max checkpoints (`runs/<run>/model/model_*.pkl`) pickle the full network
params struct (e.g. SAC: policy + value + target_value, as jax arrays), which
can only be unpickled where the `vmax` package is importable. A challenge
submission needs the POLICY alone, as a plain dict of numpy arrays that loads
with nothing but `pickle` + `numpy`.

Usage (from the V-Max project, so the checkpoint classes resolve):
  uv run python scripts/export_policy_weights.py \
      runs/<run>/model/model_final.pkl <submission_dir>/weights.pkl
"""

import argparse
import pickle

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="V-Max checkpoint pickle (runs/<run>/model/model_*.pkl)")
    ap.add_argument("out_weights", help="output weights.pkl (policy-only, plain numpy dict)")
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)

    # Full network struct -> policy params; a policy-only pickle passes through.
    params = ckpt.policy if hasattr(ckpt, "policy") else ckpt
    params = jax.tree.map(np.asarray, params)

    with open(args.out_weights, "wb") as f:
        pickle.dump(params, f)

    n_leaves = len(jax.tree.leaves(params))
    print(f"exported {type(ckpt).__name__} -> {args.out_weights} ({n_leaves} arrays)")


if __name__ == "__main__":
    main()
