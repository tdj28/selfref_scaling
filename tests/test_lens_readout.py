"""Offline tests for the descriptive Jacobian-lens readout on tiny synthetic lenses and states."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

from selfref_scaling import design
from selfref_scaling import lens_readout as L
from selfref_scaling.common import canonical, sha

D, V, LAYERS = 8, 40, 6  # tiny lens: source layers 0..5, band = middle third = [2, 3]
BAND = [2, 3]


def tiny_lens(tmp_path, layers=range(LAYERS), extra=None, name="lens.pt"):
    torch.manual_seed(0)
    J = {layer: torch.randn(D, D).to(torch.float16) for layer in layers}
    checkpoint = {"J": J, "n_prompts": 3, "source_layers": sorted(J), "d_model": D, **(extra or {})}
    path = tmp_path / name
    torch.save(checkpoint, path)
    return path, J


def tiny_head(identity_unembed=False):
    torch.manual_seed(1)
    norm = Qwen3_5MoeRMSNorm(D, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.zeros(D) if identity_unembed else torch.randn(D) * 0.3)
    unembed = torch.cat([torch.eye(D), torch.zeros(V - D, D)]) if identity_unembed else torch.randn(V, D)
    return norm.eval(), unembed


def demo2_random(J, seed):
    """Verbatim recipe of demo2.make_control_lens(kind="random") on an fp32-upcast lens."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer, M in {k: v.float() for k, v in J.items()}.items():
        R = torch.randn(M.shape[0], M.shape[0], generator=g)
        R = R * (M.float().norm() / R.norm())
        out[layer] = R.to(M.dtype)
    return out


class StubTokenizer:
    VOCAB = {" aware": [7], "aware": [8], " roleplay": [1, 2], "roleplay": [3, 4], " seem": [5, 6], "seem": [11]}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.VOCAB[text]


def q4_rows(blocks=(1, 2)):
    return [r for r in design.qwen_rows() if r["family"] == "q4" and r["block"] in blocks]


# --------------------------------------------------------------- lens file
def test_lens_file_format_roundtrip_and_validation(tmp_path):
    path, J = tiny_lens(tmp_path)
    lens = L.Lens(path, sha(path))
    assert lens.source_layers == list(range(LAYERS)) and lens.d_model == D and lens.n_prompts == 3
    assert lens.stored_dtype == "float16"
    assert lens.matrix(2).dtype == torch.float32 and torch.equal(lens.matrix(2), J[2].float())
    assert L.band_rule(lens.source_layers) == BAND
    assert L.band_rule(list(range(59))) == design.LENS["band_layers"]  # the pinned 59-layer lens
    with pytest.raises(ValueError, match="hash"):
        L.Lens(path, "0" * 64)
    bad, _ = tiny_lens(tmp_path, extra={"fit": "x"}, name="bad.pt")
    with pytest.raises(ValueError, match="checkpoint"):
        L.Lens(bad)
    torch.save({"J": {1: J[1], 0: J[0]}, "n_prompts": 1, "source_layers": [1, 0], "d_model": D}, tmp_path / "o.pt")
    with pytest.raises(ValueError, match="ascending"):
        L.Lens(tmp_path / "o.pt")


def test_spec_binding():
    assert L.check_spec(design.LENS) == ["boundary", "answer_1", "answer_2", "answer_3", "answer_4"]
    with pytest.raises(ValueError):
        L.check_spec(dict(design.LENS, transports=["lens", "identity"]))


# ---------------------------------------------------------------- readout
def test_rank_rule_counts_strictly_larger_logits():
    logits = torch.tensor([[3.0, 1.0, 3.0, 2.0], [0.0, 5.0, -1.0, 5.0]])
    assert L.ranks_from_logits(logits, [0, 1, 2, 3]).tolist() == [[1, 4, 1, 3], [3, 1, 4, 1]]


def test_identity_transport_equals_logit_lens_ranks(tmp_path):
    norm, unembed = tiny_head()
    vectors = torch.randn(5, D)
    ids = [0, 3, 17, 39]
    got = L.readout(vectors, None, norm, unembed, ids, chunk=2)
    with torch.no_grad():
        logits = torch.nn.functional.linear(norm(vectors), unembed)  # logit lens: no transport
    order = torch.argsort(torch.argsort(logits, dim=1, descending=True, stable=True), dim=1, stable=True) + 1
    assert torch.equal(got, order[:, ids])
    assert torch.equal(L.readout(vectors, torch.eye(D), norm, unembed, ids), got)
    J = torch.randn(D, D)
    with torch.no_grad():
        transported = torch.nn.functional.linear(norm(vectors @ J.T), unembed)
    assert torch.equal(L.readout(vectors, J, norm, unembed, ids), L.ranks_from_logits(transported, ids))


def test_random_transports_are_seeded_and_frobenius_matched(tmp_path):
    path, J = tiny_lens(tmp_path)
    lens = L.Lens(path)
    names = ["lens", "identity", "random_0", "random_1"]
    first = [(n, layer, m) for n, layer, m in L.transports(lens, BAND, names)]
    again = [(n, layer, m) for n, layer, m in L.transports(lens, BAND, names)]
    assert [(n, layer) for n, layer, _ in first] == [(n, layer) for n in names for layer in BAND]
    assert all((a[2] is None and b[2] is None) or torch.equal(a[2], b[2]) for a, b in zip(first, again))
    by = {(n, layer): m for n, layer, m in first}
    assert by[("identity", 2)] is None and torch.equal(by[("lens", 3)], J[3].float())
    for seed in (0, 1):
        reference = demo2_random(J, seed)
        for layer in BAND:
            matrix = by[(f"random_{seed}", layer)]
            assert torch.equal(matrix, reference[layer])  # identical to demo2's random_J recipe
            assert torch.isclose(matrix.norm(), J[layer].float().norm(), rtol=1e-5)
    assert not torch.equal(by[("random_0", 2)], by[("random_1", 2)])
    skipped = torch.randn(D, D, generator=torch.Generator().manual_seed(0))  # first draw only
    assert not torch.allclose(by[("random_0", 2)] / by[("random_0", 2)].norm(), skipped / skipped.norm())


def test_lexicon_resolution_follows_demo2():
    tokens, skipped = L.resolve_lexicons(StubTokenizer(), {"x": ["aware", "roleplay", "seem"]})
    assert tokens == {"x": {"aware": 7, "seem": 11}} and skipped == {"x": ["roleplay"]}


def captures_for(rows, value):
    """States [LAYERS + 1, k + 1, D]; value(row, capture_index, position) gives a basis index."""
    out = {}
    for i, row in enumerate(rows):
        k = 1 + i % 4
        states = torch.zeros(LAYERS + 1, k + 1, D)
        for c in range(LAYERS + 1):
            for p in range(k + 1):
                states[c, p, value(row, c, p)] = 5.0
        out[row["id"]] = {"states": states.to(torch.bfloat16),
                          "positions": ["boundary"] + [f"answer_{j}" for j in range(1, k + 1)]}
    return out


def test_capture_index_mapping_and_positions(tmp_path):
    path, _ = tiny_lens(tmp_path)
    norm, unembed = tiny_head(identity_unembed=True)  # logit of basis token j = normalized coordinate j
    rows = q4_rows()
    # capture index l + 1 holds basis 0 for band layer 2 and basis 1 for band layer 3; every other
    # capture index holds basis 7; answer positions hold basis 5.
    target = {3: 0, 4: 1}
    captures = captures_for(rows, lambda row, c, p: 5 if p else target.get(c, 7))
    tokens = {"probe": {"zero": 0, "one": 1, "five": 5, "seven": 7}}
    raw = L.compute(L.Lens(path), BAND, ["identity"], captures, norm, unembed, tokens)
    ranks = raw["ranks"]["identity"]  # [band layer, vector, token]
    col = {t: j for j, t in enumerate(raw["token_ids"])}
    boundary = [v for v, (_, position) in enumerate(raw["vectors"]) if position == "boundary"]
    answers = [v for v, (_, position) in enumerate(raw["vectors"]) if position != "boundary"]
    assert ranks[0, boundary, col[0]].tolist() == [1] * len(rows)  # band layer 2 reads capture index 3
    assert ranks[1, boundary, col[1]].tolist() == [1] * len(rows)  # band layer 3 reads capture index 4
    assert ranks[0, boundary, col[7]].tolist() == [2] * len(rows)
    assert (ranks[:, answers, col[5]] == 1).all()
    assert len(answers) == sum(len(c["positions"]) - 1 for c in captures.values())


def test_band_best_rank_medians_and_separation():
    rows = q4_rows(blocks=(1,))
    ids = {r["instruction"] + r["transcript"]: r["id"] for r in rows}
    vectors = [[ids["SS"], "boundary"], [ids["SH"], "boundary"], [ids["HS"], "boundary"], [ids["HH"], "boundary"],
               [ids["SS"], "answer_1"]]
    # ranks[layer, vector, token] for tokens 10, 11, 12 over band layers 2 and 3
    ranks = torch.tensor([[[5, 9, 4], [8, 2, 6], [30, 40, 50], [7, 7, 7], [1, 2, 3]],
                          [[3, 9, 6], [8, 1, 9], [60, 20, 10], [7, 9, 5], [4, 1, 3]]])
    raw = {"token_ids": [10, 11, 12], "vectors": vectors, "ranks": {"lens": ranks}}
    tokens = {"odd": {"a": 10, "b": 11, "c": 12}, "even": {"a": 10, "b": 11}}
    out = L.summarize(raw, BAND, tokens, rows, ["boundary", "answer_1", "answer_2"])
    words = {(w["row"], w["position"], w["lexicon"], w["word"]): (w["best_rank"], w["best_layer"]) for w in out["words"]}
    assert words[(ids["SS"], "boundary", "odd", "a")] == (3, 3)
    assert words[(ids["SS"], "boundary", "odd", "b")] == (9, 2)  # ties resolve to the first band layer
    assert words[(ids["HH"], "boundary", "odd", "c")] == (5, 3)
    medians = {(v["row"], v["position"], v["lexicon"]): v["median_best_rank"] for v in out["vectors"]}
    assert medians[(ids["SS"], "boundary", "odd")] == 4  # best ranks [3, 9, 4]
    assert medians[(ids["SS"], "boundary", "even")] == 9  # [3, 9] -> upper median, as demo2
    assert medians[(ids["HS"], "boundary", "odd")] == 20  # [30, 20, 10]
    cells = {(c["lexicon"], c["position"], c["cell"]): (c["n_rows"], c["median"]) for c in out["cells"]}
    assert cells[("odd", "answer_1", "SS")] == (1, 1) and cells[("odd", "answer_1", "HH")] == (0, None)
    assert cells[("odd", "answer_2", "SS")] == (0, None)
    sep = {(s["lexicon"], s["position"]): s for s in out["separation"]}
    boundary = sep[("odd", "boundary")]
    # S transcript: SS 4 - HS 20 = -16; H transcript: SH 6 - HH 7 = -1 (SH best [8, 1, 6] -> 6; HH [7, 7, 5] -> 7)
    assert (boundary["S_transcript"], boundary["H_transcript"], boundary["mean"]) == (-16, -1, "-17/2")
    assert sep[("odd", "answer_1")]["mean"] is None


# ---------------------------------------------------------------- inputs
def write_capture(directory, row_id, states, positions, sidecar_sha=None):
    from safetensors.torch import save_file
    path = directory / (row_id + ".safetensors")
    save_file({"states": states}, str(path), metadata={"positions": ",".join(positions), "index_0": "embeddings",
                                                       "index_i": "output of decoder layer i-1"})
    sidecar = {"id": row_id, "positions": positions, "shape": list(states.shape), "dtype": "bfloat16",
               "safetensors_sha256": sidecar_sha or sha(path)}
    (directory / (row_id + ".json")).write_text(canonical(sidecar) + "\n")


def test_load_states_reads_pod_runner_layout(tmp_path):
    plan = {"model": design.MODEL, "qwen_rows": design.qwen_rows()}
    rows = q4_rows(blocks=(1,))
    shape = (design.MODEL["architecture_record"]["layers"] + 1, 2, design.MODEL["architecture_record"]["hidden_size"])
    for row in rows[:2]:
        write_capture(tmp_path, row["id"], torch.randn(shape).to(torch.bfloat16), ["boundary", "answer_1"])
    captures, missing = L.load_states(plan, tmp_path)
    assert list(captures) == [r["id"] for r in rows[:2]] and len(missing) == 80 - 2
    assert captures[rows[0]["id"]]["positions"] == ["boundary", "answer_1"]
    write_capture(tmp_path, rows[2]["id"], torch.randn(shape).to(torch.bfloat16), ["boundary", "answer_1"], "0" * 64)
    with pytest.raises(ValueError, match="sidecar"):
        L.load_states(plan, tmp_path)
    (tmp_path / (rows[2]["id"] + ".json")).unlink()
    (tmp_path / (rows[2]["id"] + ".safetensors")).unlink()
    write_capture(tmp_path, rows[2]["id"], torch.randn(shape).to(torch.bfloat16), ["boundary", "answer_2"])
    with pytest.raises(ValueError, match="Invalid captured state"):
        L.load_states(plan, tmp_path)


def test_outputs_are_deterministic(tmp_path):
    path, _ = tiny_lens(tmp_path)
    norm, unembed = tiny_head()
    rows = q4_rows()
    torch.manual_seed(5)
    captures = {}
    for i, row in enumerate(rows):
        k = 1 + i % 4
        captures[row["id"]] = {"states": torch.randn(LAYERS + 1, k + 1, D).to(torch.bfloat16),
                               "positions": ["boundary"] + [f"answer_{j}" for j in range(1, k + 1)]}
    tokens = {"experience": {"a": 3, "b": 9, "c": 21}, "denial_tool": {"d": 5, "e": 30}}
    positions = ["boundary", "answer_1", "answer_2", "answer_3", "answer_4"]
    outputs = []
    for name in ("one", "two"):
        raw = L.compute(L.Lens(path), BAND, L.TRANSPORTS, captures, norm, unembed, tokens)
        provenance = {"band_layers": BAND, "capture_indices": [b + 1 for b in BAND]}
        outputs.append(L.write_outputs(tmp_path / name, raw, L.summarize(raw, BAND, tokens, rows, positions),
                                       provenance))
    assert outputs[0] == outputs[1]
    for file in outputs[0]["files"]:
        assert (tmp_path / "one" / file).read_bytes() == (tmp_path / "two" / file).read_bytes()
    summary = json.loads((tmp_path / "one" / "lens_summary.json").read_text())
    assert set(summary["primary_boundary_separation"]) == {f"{t}|{x}" for t in L.TRANSPORTS for x in tokens}
    assert summary["transport_matrices"]["identity"] == {"frobenius": [], "sha256": None}
    assert len(summary["transport_matrices"]["random_4"]["frobenius"]) == len(BAND)
    ranks = json.loads((tmp_path / "one" / "lens_ranks.json").read_text())
    shape = torch.tensor(ranks["ranks_by_transport_layer_vector_token"]["lens"]).shape
    assert tuple(shape) == (len(BAND), sum(len(c["positions"]) for c in captures.values()), 5)
    with pytest.raises(FileExistsError):
        L.write_outputs(tmp_path / "one", raw, L.summarize(raw, BAND, tokens, rows, positions), provenance)


# ------------------------------------------------- pinned artifacts, if cached
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "hf"
PINNED = (CACHE / "models--praxagent-org--jacobian-lens-qwen3.5-397b-a17b" / "snapshots" / design.LENS["revision"]
          / design.LENS["file"])


@pytest.mark.skipif(not PINNED.exists(), reason="pinned lens not in the local cache (no downloads in tests)")
def test_pinned_lens_format_if_cached():
    lens = L.Lens(PINNED, design.LENS["sha256"])
    assert lens.source_layers == list(range(59)) and lens.stored_dtype == "float16"
    assert (lens.n_prompts, lens.d_model) == (design.LENS["fit_prompts"], 4096)
    assert L.band_rule(lens.source_layers) == design.LENS["band_layers"]
    index = (CACHE / "models--Qwen--Qwen3.5-397B-A17B" / "snapshots" / design.MODEL["revision"]
             / "model.safetensors.index.json")
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = ("lm_head.weight", design.MODEL["text_backbone"] + ".norm.weight")
        assert {weight_map[n] for n in names} == set(L.HEAD_SHARDS)
