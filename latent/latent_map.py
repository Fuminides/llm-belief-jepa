"""Export the latent space of the latent models on the recurrent memory for an interactive map (docs/latent_map/).

For the test episodes (the last 20%, never seen in training), per memory layer: the latent states z of every turn,
their 2D maps (t-SNE and the first two principal components), zones (k-means on z), the surprise of each real
transition (latent prediction error, standardised within its turn, as in Test 6), and the weights of the predictor
and readout, so the page can apply any clue to any state. Layers: 14, the memory the LLM needs most (leaving it out
costs 11% of a clue's effect, cache_patch.py); 9, also needed; 22, as readable but not needed at all.

  python -m latent.latent_map   # writes docs/latent_map/latent_map.json
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from games import game
from latent import jepa
from latent.explain import get_model

LAYERS = [  # (data directory, layer, note shown on the page)
    ("data/qwen3.5-2b-memory-mid", 14, "the memory the LLM needs most: leaving it out costs 11% of a clue's effect"),
    ("data/qwen3.5-2b-memory-mid", 9, "also needed: leaving it out costs 6%"),
    ("data/qwen3.5-2b-memory", 22, "readable but not needed: leaving it out costs nothing"),
]
OUT = Path("docs/latent_map/latent_map.json")
ZONES = 10


def f16(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype=np.float16).tobytes()).decode()


def export_layer(data_dir, layer, note, device):
    memory_layers = list(np.load(f"{data_dir}/shard_000000.npz")["memory_layers"])
    i = memory_layers.index(layer)
    h, a, b, _ = jepa.load(data_dir, "memory", i)
    n_train = round(len(h) * 0.8)
    mu, sd = h[:n_train].mean((0, 1)), h[:n_train].std((0, 1)) + 1e-6
    model = get_model(torch.tensor((h - mu) / sd), torch.tensor(a), torch.tensor(b), n_train,
                      Path(f"{data_dir}/latent_memory_layer{i}_mlp.pt"), "memory", "mlp", 3000, device)
    with torch.no_grad():
        z = model.enc(torch.tensor((h[n_train:] - mu) / sd, device=device))  # [E, T+1, d_z]
        at = torch.tensor(a[n_train:], device=device)
        zhat = model.step(z[:, :-1], at)
        err = ((zhat - z[:, 1:]) ** 2).sum(-1).cpu().numpy()  # [E, T]
        readout = model.read(z).cpu().numpy()
    surprise = (err - err.mean(0)) / err.std(0)
    llm_logit = np.log(b[n_train:] / (1 - b[n_train:]))
    readout_r2 = jepa.r2(readout, llm_logit)
    flat = z.reshape(-1, z.shape[-1]).cpu().numpy()
    mean = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    pca = (flat - mean) @ vt[:2].T
    # Identical states (all turn-0 states, most turn-1 states) would be smeared over the map by t-SNE: embed each
    # distinct state once and put its copies on top of it.
    unique, inverse = np.unique(flat.round(5), axis=0, return_inverse=True)
    tsne = TSNE(n_components=2, perplexity=50, init="pca", random_state=0).fit_transform(unique)[inverse.ravel()]
    zones = KMeans(n_clusters=ZONES, n_init=10, random_state=0).fit_predict(flat)
    p = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    weights = {name: {"shape": list(p[key].shape), "data": f16(p[key])} for name, key in [
        ("act", "act.weight"), ("w0", "pred.0.weight"), ("b0", "pred.0.bias"), ("w1", "pred.2.weight"),
        ("b1", "pred.2.bias"), ("w2", "pred.4.weight"), ("b2", "pred.4.bias"), ("rw", "read.weight"),
        ("rb", "read.bias")]}
    print(f"layer {layer}: readout R² on test states {readout_r2:.3f}", flush=True)
    return {"layer": layer, "note": note, "readout_r2": readout_r2, "d_z": int(flat.shape[1]), "z": f16(flat),
            "tsne": np.round(tsne, 2).tolist(), "pca": np.round(pca, 3).tolist(),
            "pca_mean": mean.round(5).tolist(), "pca_axes": vt[:2].round(5).tolist(),
            "pca_variance": (s[:2] ** 2 / (s ** 2).sum()).round(4).tolist(), "zone": zones.tolist(),
            "surprise": np.round(surprise, 2).tolist(), "weights": weights}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, a, b, _ = jepa.load(LAYERS[0][0], "memory", 0)
    n_train = round(len(a) * 0.8)
    bayes = np.array([[game.bayes_belief(row[:t].tolist()) for t in range(a.shape[1] + 1)] for row in a[n_train:]])
    data = {
        "attributes": [[name, *values] for name, values in game.ATTRIBUTES.items()],
        "sources": game.SOURCE_NAMES,
        "reliability": [game.SOURCES[s] for s in game.SOURCE_NAMES],
        "actions": [{"attr": at, "value": v, "negated": bool(n), "source": s, "text": game.clue_text(k).strip()}
                    for k, (at, v, n, s) in enumerate(game.ACTIONS)],
        "episodes": {"first_id": n_train, "actions": a[n_train:].tolist(),
                     "llm": np.round(b[n_train:], 4).tolist(), "bayes": np.round(bayes, 4).tolist()},
        "layers": [export_layer(*spec, device) for spec in LAYERS],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
