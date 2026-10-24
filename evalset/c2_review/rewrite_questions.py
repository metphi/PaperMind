"""Rewrite the 9 ambiguous cross questions in evalset/dev.jsonl to name the
involved models/papers.

Principle (user, 2026-08-13): a real user does NOT quote paper titles. They
refer to papers by their model/architecture name or a paper designation
("LMF", "PhaseDNN", "MscaleDNN", "Fourier features", "HiNOTE", ...). The first
rewrite expanded full titles ("LMF (Latent Modulated Function)",
"Fourier Features (NeurIPS 2020)") which reads unnatural; this version uses
plain model names.

gold_spans / gold_sources are untouched (anchored to paper text). Only
``question`` and ``gold_answer`` change. Matched by ``_meta.papers`` so it is
idempotent against the previous rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path

DEV = Path(__file__).resolve().parents[2] / "evalset" / "dev.jsonl"

# _meta.papers (tuple) -> (new question, new gold_answer | None)
REWRITES: dict[tuple[str, ...], tuple[str, str | None]] = {
    # ── 1. LMF vs Meta-SR ──────────────────────────────────────────────────
    ("LMF", "Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper"): (
        "How do LMF and Meta-SR differ in their approach to handling arbitrary scale factors in super-resolution, particularly in terms of model architecture and computational efficiency?",
        None,
    ),
    # ── 2. GaussianSR vs GSASR ─────────────────────────────────────────────
    ("GaussianSR", "GSASR"): (
        "Both GaussianSR and GSASR address the task of arbitrary-scale image super-resolution using 2D Gaussian Splatting to overcome limitations of implicit neural representations. Compare the two methods in terms of how they represent image content with Gaussians and how they ensure the model generalizes across different images and scaling factors.",
        "GaussianSR represents each pixel as a continuous Gaussian field and refines/upsamples encoded features by rendering mutually stacked Gaussian fields, but its generalization mechanism is not explicitly detailed in the abstract. In contrast, GSASR explicitly addresses the generalization challenge of Gaussian Splatting (which was originally optimization-based and scene-specific) by designing a feed-forward architecture that predicts image-conditioned Gaussians from the input low-resolution image, enabling the model to generalize to different images and scaling factors without per-scene overfitting.",
    ),
    # ── 3. PhaseDNN + Frequency-adaptive TNN + MscaleDNN (was [1][2][3]) ────
    ("A_Phase_Shift_Deep_Neural_Network_for_High-Frequency_Approximation", "Frequency-adaptive_tensor_neural_networks_for_high-dimensional_multi-scale_problems", "MscaleDNNLiuCaiXu2020"): (
        "Both PhaseDNN and MscaleDNN introduce deep learning architectures designed to overcome the spectral bias that causes standard neural nets to learn low-frequency components before high-frequency ones, yet they employ fundamentally different strategies to achieve uniform multi-scale convergence. Compare the core mechanism by which PhaseDNN decomposes the frequency learning task with the mechanism used in MscaleDNN, and explain how Frequency-adaptive TNN's proposed enhancement to tensor-based architectures addresses a related but distinct challenge in the same domain. Specifically, what frequency-domain transformation does PhaseDNN apply to its sub-networks, what spatial-domain technique does MscaleDNN use to isolate frequency bands, and what type of feature mapping does Frequency-adaptive TNN integrate to boost expressivity for multi-scale problems?",
        "PhaseDNN applies phase shifts in the frequency domain to each sub-network so that it learns a targeted high-frequency band at the same speed as low-frequency learning. MscaleDNN instead uses radial scaling in the frequency domain combined with activation functions that have compact support in the spatial domain to separate and isolate different frequency contents across its sub-networks. Frequency-adaptive TNN tackles the same spectral bias problem from a different angle by integrating random Fourier features into tensor neural networks, leveraging their intrinsic tensor structure to efficiently extract frequency features for high-dimensional multi-scale functions. All three methods address the difficulty of learning high-frequency or multi-scale content, but PhaseDNN relies on frequency-domain phase translation, MscaleDNN on spatial-domain compact-support activations paired with radial frequency scaling, and Frequency-adaptive TNN on random Fourier feature embeddings within a tensor framework.",
    ),
    # ── 4. MscaleDNN vs Fourier features ───────────────────────────────────
    ("MscaleDNNLiuCaiXu2020", "NeurIPS-2020-fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-Paper_1_"): (
        "Both MscaleDNN and Fourier features address the challenge of learning high-frequency components in neural network-based function approximation. Compare how each method proposes to overcome the difficulty of capturing high-frequency content: what specific architectural or input-transformation strategy does each method employ, and what is the core mechanism by which that strategy improves high-frequency learning?",
        None,
    ),
    # ── 5. Fourier features vs RandONet ─────────────────────────────────────
    ("NeurIPS-2020-fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-Paper_1_", "RandOnet"): (
        "How do Fourier features and RandONet differ in their treatment of the input representation for neural networks, and what shared goal do they pursue despite using different architectural strategies?",
        "Fourier features proposes augmenting the input space by passing raw coordinates through a Fourier feature mapping before feeding them into a multilayer perceptron, thereby reshaping the network's implicit kernel to overcome its bias against high-frequency signals. RandONet takes a different route: it employs shallow networks whose first layer consists of fixed random projections, using these as a lightweight, interpretable way to approximate linear and nonlinear operators governing dynamical systems. Despite their distinct mechanisms — one relying on structured, frequency-domain feature lifting for regression tasks and the other on randomized shallow architectures for operator learning — both share the overarching goal of enhancing a neural network's ability to capture complex, high-dimensional functional relationships by deliberately engineering the input representation rather than relying solely on depth or standard training.",
    ),
    # ── 6. Fourier features vs MscaleFNO ───────────────────────────────────
    ("NeurIPS-2020-fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-Paper_1_", "multiscale_FNO"): (
        "Both Fourier features and MscaleFNO address the challenge of learning high-frequency components in neural networks. How does the Fourier features approach to overcoming spectral bias differ fundamentally from the MscaleFNO approach in terms of architectural strategy, and what specific problem domain does each method target?",
        "Fourier features addresses spectral bias by applying a Fourier feature mapping as a preprocessing step to transform input points before feeding them into a standard multilayer perceptron, effectively modifying the neural tangent kernel to have tunable bandwidth for high-frequency learning in low-dimensional regression tasks relevant to computer vision and graphics. In contrast, MscaleFNO tackles spectral bias through an architectural modification by constructing a multi-scale Fourier neural operator that consists of multiple parallel normal FNOs with scaled inputs, targeting high-frequency wave scattering problems and learning mappings between oscillatory functions in partial differential equations. The key difference is that Fourier features uses input transformation with a fixed architecture, while MscaleFNO modifies the network architecture itself with parallel multi-scale components.",
    ),
    # ── 7. PhaseDNN vs MscaleDNN ───────────────────────────────────────────
    ("A_Phase_Shift_Deep_Neural_Network_for_High-Frequency_Approximation", "MscaleDNNLiuCaiXu2020"): (
        "Both PhaseDNN and MscaleDNN address the challenge of training neural networks to approximate high-frequency functions by converting the problem into one that involves lower-frequency learning. Compare the specific mechanisms each method uses to achieve this frequency conversion, and explain how each approach ensures that different frequency components are handled appropriately.",
        "PhaseDNN leverages phase shifts in the frequency domain to train a series of moderately sized DNNs, each targeting a specific high-frequency range at the speed typically associated with low-frequency learning. This allows the method to effectively convert high-frequency learning into low-frequency learning. MscaleDNN proposes MscaleDNNs, which use radial scaling in the frequency domain combined with activation functions that have compact support. The radial scaling transforms high-frequency approximation into lower-frequency learning, while the compact support activation functions help separate the frequency contents of the target function, enabling each DNN to focus on its designated frequency range. Both methods achieve uniform convergence across multiple scales but employ distinct strategies: PhaseDNN relies on phase manipulation and sequential training, while MscaleDNN uses scaling and specialized activation functions for frequency separation.",
    ),
    # ── 8. CiaoSR vs LMF ───────────────────────────────────────────────────
    ("CiaoSR", "LMF"): (
        "How do CiaoSR and LMF differ in their strategies for addressing the limitations of local feature aggregation in continuous image representation for arbitrary-scale super-resolution, and what specific mechanisms do they each introduce to overcome these limitations?",
        None,
    ),
    # ── 9. HiNOTE vs NeurOp-Diff ───────────────────────────────────────────
    ("HiNOTE", "Xu_NeurOp-Diff_Continuous_Remote_Sensing_Image_Super-Resolution_via_Neural_Operator_Diffusion_ICCV_2025_paper"): (
        "Both HiNOTE and NeurOp-Diff address arbitrary-scale super-resolution using neural operators, but they adopt fundamentally different generative frameworks. Compare how each method leverages neural operators: what specific mechanism does HiNOTE use to enable resolution-invariant function-space mappings, and what role do neural operators play in NeurOp-Diff's diffusion-based pipeline?",
        "HiNOTE uses a hierarchical neural operator with a Galerkin-type self-attention mechanism to directly learn mappings between function spaces in a resolution-invariant manner, with sinc filters ensuring representation equivalence across hierarchy levels. In contrast, NeurOp-Diff uses neural operators not as the primary super-resolution model but as a feature encoder: they learn resolution representations at arbitrary scales to encode low-resolution images into high-dimensional features, which then serve as prior conditions to guide a diffusion model for denoising. Thus, while both employ neural operators for arbitrary-scale SR, HiNOTE uses them as the core mapping architecture, whereas NeurOp-Diff uses them as conditioning encoders within a diffusion framework.",
    ),
}


def main() -> None:
    pairs = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_papers = {tuple(p["_meta"].get("papers") or []): p for p in pairs}

    applied = 0
    for papers, (new_q, new_answer) in REWRITES.items():
        p = by_papers.get(papers)
        if p is None:
            print("NOT FOUND:", papers)
            continue
        assert p.get("cross_paper"), f"target not cross: {papers}"
        p["question"] = new_q
        if new_answer is not None:
            p["gold_answer"] = new_answer
        applied += 1
        print(f"rewrote: {papers}")

    if applied != len(REWRITES):
        raise SystemExit(1)
    print(f"applied {applied}/{len(REWRITES)} rewrites")

    out = "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n"
    DEV.write_text(out, encoding="utf-8")
    print(f"wrote {DEV}")


if __name__ == "__main__":
    main()
