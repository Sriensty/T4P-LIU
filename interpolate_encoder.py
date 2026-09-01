import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--rho", type=float, required=True)  # theta = rho*source + (1-rho)*adapted
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src = torch.load(args.source, map_location="cpu")["state_dict"]
    adp = torch.load(args.adapted, map_location="cpu")["state_dict"]

    merged = dict(adp)
    n_interp = 0
    for k, v in adp.items():
        if k.startswith("net.blocks.") or k.startswith("net.norm."):
            assert k in src, f"missing key in source ckpt: {k}"
            assert src[k].shape == v.shape, f"shape mismatch: {k}"
            merged[k] = args.rho * src[k].float() + (1 - args.rho) * v.float()
            n_interp += 1

    torch.save({"state_dict": merged}, args.out)
    print(f"[Interp] rho={args.rho} interpolated {n_interp} encoder tensors -> {args.out}")


if __name__ == "__main__":
    main()
