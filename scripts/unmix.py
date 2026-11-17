"""Evaluate DOFA features (with or without NAF upsampling) on hyperspectral unmixing.

    python scripts/unmix.py --scene apex --features naf
    python scripts/unmix.py --scene apex --features dofa     # baseline, no NAF

Results (metrics + figures) are written to ``results/<scene>_<features>/``.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsnaf.backbones import dofa_features, load_dofa, load_spectral_adapter, resize_for_dofa  # noqa: E402
from hsnaf.data import load_unmixing_scene  # noqa: E402
from hsnaf.metrics import evaluate_unmixing  # noqa: E402
from hsnaf.plots import plot_abundances, plot_endmembers  # noqa: E402
from hsnaf.unmixing import train_unmixing  # noqa: E402
from hsnaf.upsampling import build_naf, upsample_features  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="apex", choices=["apex", "jasper", "samson", "urban"])
    p.add_argument("--features", default="naf", choices=["naf", "dofa"])
    p.add_argument("--naf-checkpoint", default="checkpoints/naf_7to14.pth")
    p.add_argument("--kernel-size", type=int, default=9, help="NAF neighborhood size at inference")
    p.add_argument("--dofa-weights", default="weights/DOFA_ViT_large_e100.pth")
    p.add_argument("--adapter-weights", default="weights/spec_ViTb_mae.pth")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Y, A, E, wavelengths = load_unmixing_scene(args.scene)
    Y, A = Y.to(device), A.to(device)
    print(f"{args.scene}: {Y.shape[1]} bands, {Y.shape[-2]}x{Y.shape[-1]} pixels, {E.shape[1]} endmembers")

    dofa = load_dofa(args.dofa_weights, device)
    if args.features == "naf":
        # NAF upsamples the 14x14 DOFA features to the native image resolution
        adapter = load_spectral_adapter(args.adapter_weights, device)
        naf = build_naf(args.kernel_size, args.naf_checkpoint, device).eval()
        features = upsample_features(naf, adapter, dofa, Y, wavelengths)
    else:
        # Baseline: raw 14x14 DOFA features; unmixing is done on the 224x224 resized image
        Y, A = resize_for_dofa(Y), resize_for_dofa(A)
        features = dofa_features(dofa, Y, wavelengths)
    print(f"features: {tuple(features.shape)}")

    _, E_hat, A_hat = train_unmixing(features, Y, n_endmembers=E.shape[1], epochs=args.epochs)
    results = evaluate_unmixing(E_hat, A_hat, E, A)
    print(f"SAD  = {results['sad']:.4f}  per endmember: {[round(v, 3) for v in results['sad_per_endmember']]}")
    print(f"NMSE = {results['nmse']:.4f}  per endmember: {[round(v, 3) for v in results['nmse_per_endmember']]}")

    out = args.out_dir / f"{args.scene}_{args.features}"
    out.mkdir(parents=True, exist_ok=True)
    label = "DOFA + NAF" if args.features == "naf" else "DOFA"
    plot_endmembers(E, results, title=f"{args.scene} — {label}").savefig(out / "endmembers.png", dpi=150)
    plot_abundances(A, results, title=f"{args.scene} — {label}").savefig(out / "abundances.png", dpi=150)
    metrics = {k: v for k, v in results.items() if k not in ("E_hat", "A_hat")}
    (out / "metrics.json").write_text(json.dumps({**vars(args), **metrics}, indent=2, default=str))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
