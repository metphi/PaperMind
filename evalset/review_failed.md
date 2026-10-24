# 失败对审查清单（14 对）

> 生成完 100 对，86 ok 已入 `dev.jsonl`，这 14 对 stage-2 没锚到 ≥2 个 verbatim span。
> 失败 ≠ 答案错：多数答案忠实复述了摘要，失败主因是**锚定措辞与正文对不上**（连字符、摘要用词≠正文用词、或 Qwen span 偏差）。
> 请在"处理"列标注：**留**（补锚定）/ **剔**（答案查无实据）/ **重写**（问题或答案改措辞）。

> ✅ **已全部解决（2026-08-12）**：14 对已用"候选段落 + 强化 prompt"全部重锚定成功（`scripts/evalset_reanchor.py`），无需人工筛选。本清单留档备查。

| # | pair_id | 难度 | 处理 | 我的可疑点 |
|---|---|---|---|---|
| 1 | single:HiNOTE:definition | easy | | 答案"Galerkin-type self-attention"与摘要一致，内容应真实。疑锚定措辞问题 |
| 2 | single:HiNOTE:method | medium | | 答案"sinc filters 传递层级信息"与摘要一致，内容应真实 |
| 3 | single:HiNOTE:synthesis | medium | | 同上，sinc filters 跨层信息传递，摘要有 |
| 4 | single:HiNOTE:challenge | hard | | 答案"learnable prior + spectral resizing"，摘要"frequency-aware loss prior"，措辞可能略偏 |
| 5 | single:SRNO:definition | easy | | 答案"kernel integral mechanism 迭代逼近"，摘要原文有，内容应真实 |
| 6 | single:SRNO:method | medium | | **答案称"Galerkin-type attention"——SRNO 摘要无 Galerkin，疑与 HiNOTE 混淆/幻觉，重点查** |
| 7 | single:SRNO:challenge | hard | | 同 6，Galerkin 疑点 |
| 8 | single:MscaleDNN:definition | easy | | 答案"radial scaling + compact support"，摘要原文有，内容应真实 |
| 9 | single:MscaleDNN:synthesis | medium | | 答案"优于 fully connected + mesh-free"，摘要"Summary"有，内容应真实 |
| 10 | single:MscaleDNN:challenge | hard | | 同 9，Poisson-Boltzmann 复杂域，内容应真实 |
| 11 | single:LocalImplicit:definition | easy | | 答案"LINF 命名"由摘要可证，内容应真实 |
| 12 | single:LocalImplicit:method | medium | | 答案"normalizing flow 建模纹理分布"，摘要有，内容应真实 |
| 13 | single:DiffuFNO_SR:challenge | hard | | 答案"Mode Re-balancing in WFNO 提升高频细节"，图/正文有，内容应真实 |
| 14 | single:GaussianSR:challenge | hard | | 答案"discrete latent codes 限制表达能力"，摘要原文有，内容应真实 |

---

## 详细内容

### HiNOTE（4 对，全挂——疑锚定措辞系统性没对上）

**definition** (easy)
- Q: What is the core component of the proposed hierarchical neural operator transformer model, and what mechanism does it use to enable efficient learning of mappings between function spaces?
- A: The core of the model is a hierarchical neural operator that leverages a Galerkin-type self-attention mechanism, enabling efficient learning of mappings between function spaces.
- 判定：答案=摘要原句改写，**内容真实**。

**method** (medium)
- Q: What filtering technique does the proposed model employ to maintain consistent feature representations when passing information between different levels of its multi-level architecture?
- A: The model uses sinc filters to facilitate information transfer across different levels in the hierarchy, ensuring representation equivalence in the neural operator.
- 判定：答案=摘要原句改写，**内容真实**。

**synthesis** (medium)
- Q: How does the proposed architecture ensure consistent feature representation when transferring information between different hierarchical levels, and what specific filtering technique enables this cross-level communication?
- A: The model employs Sinc filters to facilitate information transfer across different levels in the hierarchy, which ensures representation equivalence within the neural operator framework.
- 判定：与 method 高度同源，可能二选一。

**challenge** (hard)
- Q: What mechanism does the paper employ to dynamically balance gradient contributions during training, and how is this mechanism derived from the input data's spectral properties?
- A: The paper introduces a model-agnostic learnable prior structure derived from spectral resizing of the input data, which dynamically adjusts pixel contribution weighting to effectively balance gradients across the model.
- 判定：摘要"learnable frequency-aware loss prior"；"spectral resizing / 平衡梯度"为正文机制，措辞可能略偏，**需查正文**。

### SRNO（3 对）

**definition** (easy)
- Q: What mechanism does SRNO use to iteratively approximate the implicit image function after embedding the low-resolution input into a higher-dimensional latent space?
- A: SRNO iteratively approximates the implicit image function with a kernel integral mechanism.
- 判定：摘要原文有"kernel integral mechanism"，**内容真实**。

**method** (medium)
- Q: What specific attention-based mechanism does the proposed framework use to implement kernel integrals across its layers, and what spatial property does this mechanism provide that benefits continuous-domain processing?
- A: The framework implements kernel integrals via Galerkin-type attention, which provides non-local properties in the spatial domain that benefit grid-free continuum processing.
- ⚠️ **Galerkin 疑点**：SRNO 摘要/正文可能无 Galerkin（HiNOTE 才有）。若查无 → 答案幻觉，**剔或重写**。

**challenge** (hard)
- Q: What specific architectural component does SRNO use to implement kernel integrals, and what advantage does this provide for grid-free continuous super-resolution?
- A: SRNO implements kernel integrals via Galerkin-type attention, which provides non-local properties in the spatial domain that benefit grid-free continuum processing.
- ⚠️ 同 method，Galerkin 疑点。

### MscaleDNN（3 对）

**definition** (easy)
- Q: What is the core idea behind the multi-scale deep neural networks (MscaleDNNs) proposed in the paper for solving PDEs with high-frequency contents?
- A: The core idea is to use radial scaling in the frequency domain to convert the approximation of high-frequency contents into learning lower-frequency functions, combined with activation functions that have compact support.
- 判定：摘要原文，"radial scaling + compact support"，**内容真实**。

**synthesis** (medium)
- Q: What experimental evidence demonstrates that the frequency-domain transformation technique combined with localized activation functions enables the proposed architecture to outperform standard fully connected networks on equations with complex spectral characteristics?
- A: The MscaleDNNs are shown to be superior to traditional fully connected DNNs and be an effective mesh-less numerical method for Poisson-Boltzmann equations with ample frequency contents over complex and singular domains.
- 判定：摘要"Summary"有，"mesh-less / superior to FC"应真实，**内容真实**。

**challenge** (hard)
- Q: When applied to solving electrostatic potential models characterized by complex geometries and a broad spectrum of oscillatory components, how did the authors' proposed architecture perform relative to standard dense feed-forward networks, and what specific computational advantage did it demonstrate?
- A: The MscaleDNNs outperformed conventional fully connected neural networks and proved to be an effective mesh-free numerical approach for Poisson-Boltzmann equations that contain rich frequency content on intricate and singular domains.
- 判定：与 synthesis 高度同源，可能二选一。

### LocalImplicit（2 对）

**definition** (easy)
- Q: What is the name of the method proposed in the paper that uses normalizing flow to model texture details across different scaling factors for arbitrary-scale image super-resolution?
- A: The proposed method is called Local Implicit Normalizing Flow (LINF).
- 判定：命名可由摘要证，**内容真实**（但问题极简，可能有词面重叠风险）。

**method** (medium)
- Q: What core mechanism does the proposed approach use to capture fine-grained visual patterns across varying magnification levels?
- A: The method employs normalizing flow to model the distribution of texture details under different scaling factors, enabling the generation of photorealistic high-resolution images with rich texture details at arbitrary scale.
- 判定：摘要"model the distribution of HR images with normalizing flow"改写，**内容真实**。

### DiffuFNO（1 对）

**challenge** (hard)
- Q: What empirical advantage does the proposed framework demonstrate over existing methods, and what specific component is credited for enhancing the preservation of fine-grained visual elements?
- A: The proposed framework achieves state-of-the-art performance across extensive experiments, with the Mode Re-balancing mechanism in the Weighted Fourier Neural Operator specifically credited for significantly improving the reconstruction of high-frequency image details.
- 判定：图/正文有 WFNO Mode Re-balancing（见第 1 页 Figure 1 说明），**内容应真实**。

### GaussianSR（1 对）

**challenge** (hard)
- Q: What specific constraint do the authors identify in prior encoder-decoder architectures that utilize discrete latent representations for arbitrary-scale image upscaling?
- A: The authors state that the performance of existing INR-based ASSR networks is constrained by the limited representation ability of discrete latent codes in the encoded features.
- 判定：摘要原文"constrained by the limited representation ability of discrete latent codes"，**内容真实**。

---

## 给审查的建议顺序

1. **大概率可直接留**：HiNOTE×3（def/method/synthesis）、MscaleDNN×3、LocalImplicit×2、GaussianSR、DiffuFNO、SRNO:definition —— 答案都能从摘要/正文直接对上，只是 stage-2 没锚到 span。
2. **重点查**：SRNO:method / SRNO:challenge 的 **Galerkin**（疑与 HiNOTE 混淆）。
3. **同源二选一**：HiNOTE:synthesis vs method、MscaleDNN:synthesis vs challenge —— 若都留，注意检索测评里答案内容重叠。

> 留下的处理方式：手动从 `evalset/corpus/<slug>.json` 页文本里挑 2-3 段 verbatim 支撑（或放宽 stage-2 重跑）。剔除的直接在 `dev.jsonl` 里删。
