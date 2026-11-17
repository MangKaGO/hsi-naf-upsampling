"""Train NAF to upsample DOFA features of EnMAP hyperspectral tiles.

DOFA and the spectral adapter are frozen; only NAF is trained, with an MSE loss
between the upsampled and the target DOFA features.

    python scripts/train_naf.py --mode 7to14
    python scripts/train_naf.py --mode 14to28 --epochs 20
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsnaf.backbones import load_dofa, load_spectral_adapter  # noqa: E402
from hsnaf.data import DATA_DIR, EnmapTiles, read_wavelengths  # noqa: E402
from hsnaf.upsampling import TRAINING_MODES, build_naf  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=TRAINING_MODES, default="7to14",
                   help="7to14: self-supervised (DOFA 14x14 features downsampled to 7x7); "
                        "14to28: target built from the DOFA features of the 4 image quadrants")
    p.add_argument("--enmap-dir", default=DATA_DIR / "enmap", type=Path)
    p.add_argument("--wavelengths", default=DATA_DIR / "enmap_wavelengths.csv", type=Path)
    p.add_argument("--dofa-weights", default="weights/DOFA_ViT_large_e100.pth")
    p.add_argument("--adapter-weights", default="weights/spec_ViTb_mae.pth")
    p.add_argument("--max-images", type=int, default=600)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kernel-size", type=int, default=None, help="NAF neighborhood size (default depends on mode)")
    p.add_argument("--out", type=Path, default=None, help="default: checkpoints/naf_<mode>.pth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_pair, output_size, default_kernel = TRAINING_MODES[args.mode]
    out = args.out or Path("checkpoints") / f"naf_{args.mode}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)

    wavelengths = read_wavelengths(args.wavelengths)
    dofa = load_dofa(args.dofa_weights, device)
    adapter = load_spectral_adapter(args.adapter_weights, device)
    naf = build_naf(kernel_size=args.kernel_size or default_kernel, device=device)
    naf.train()

    dataset = EnmapTiles(args.enmap_dir, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Training NAF ({args.mode}) on {len(dataset)} EnMAP tiles, device={device}")

    optimizer = torch.optim.AdamW(naf.parameters(), lr=args.lr)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for hsi in pbar:
            hsi = hsi.to(device)
            low_res, target = make_pair(dofa, hsi, wavelengths)
            with torch.no_grad():
                guidance = adapter(hsi)

            pred = naf(guidance, low_res, output_size=output_size)
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(naf.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total / len(loader)
        print(f"epoch {epoch}: mean loss {mean_loss:.4f}")
        if mean_loss < best:
            best = mean_loss
            torch.save({"model_state_dict": naf.state_dict(), "epoch": epoch, "loss": best,
                        "mode": args.mode}, out)
            print(f"  saved best model to {out}")


if __name__ == "__main__":
    main()
