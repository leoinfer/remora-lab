# Prior-art map

This is a working classification, not a novelty claim. “Related” means the
mechanism overlaps; “complementary” means it can be used inside Remora; “same”
is reserved for a claim that should not be presented as novel. The distinctive
Remora question is the combination of a versioned common bus, factual
assembly/lineage state, cluster-aware lifetime evidence, and experimentally
verified local replacement across changing surrounding systems.

| Area | Representative primary work | Classification for Remora | What Remora must still test |
| --- | --- | --- | --- |
| Knowledge distillation | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | RELATED | Whether donor behavior becomes a replaceable bus module with provenance, not just a smaller student |
| Intermediate representation transfer | [FitNets](https://arxiv.org/abs/1412.6550) | RELATED | Whether a learned donor port transfers beyond the prompts used to fit it |
| Function-preserving surgery | [Net2Net](https://arxiv.org/abs/1511.05641) | COMPLEMENTARY | Which compatibility contracts make surgery safe across Remora generations |
| Parameter-efficient adaptation | [LoRA](https://arxiv.org/abs/2106.09685) | COMPLEMENTARY | Whether local islands retain old capability and can later consolidate |
| Continual learning / forgetting | [Overcoming Catastrophic Forgetting](https://arxiv.org/abs/1612.00796) | RELATED | Retention under repeated modular replacement, not only task sequence fine-tuning |
| Progressive modular systems | [Progressive Neural Networks](https://arxiv.org/abs/1606.04671) | RELATED | Whether old modules can eventually be replaced rather than only kept as frozen columns |
| Sparse expert routing | [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538), [Switch Transformers](https://arxiv.org/abs/2101.03961) | RELATED | Specialization, load balance, and surgical expert replacement under a shared bus |
| Fast weights / learned updates | [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) | RELATED | Whether fast plasticity plus slow consolidation improves capability per new token |
| Model editing | [MEND](https://arxiv.org/abs/2110.11309), [MEMIT](https://arxiv.org/abs/2210.07229) | RELATED | Provenance-linked edits, contradiction handling, and retention under repeated edits |
| Weight-space merging | [Task Arithmetic](https://arxiv.org/abs/2212.04089) | RELATED BUT RISKY | Whether alignment is measured before any merge; cross-family copying is not a default |
| Learned optimizers | [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) | RELATED | Meta-learning of Remora's local update policies |
| Neural architecture search | [DARTS](https://arxiv.org/abs/1806.09055) | COMPLEMENTARY | External promotion, resurrection, and lineage controls around candidate search |
| World models / latent dynamics | [DreamerV3](https://arxiv.org/abs/2301.04104) | RELATED | Separating inherited prior, medium-term beliefs, and episodic operational evidence |
| Long-context recurrent/linear state | [Qwen3 technical report](https://arxiv.org/abs/2505.09388) | RELATED | Whether donor recurrent mechanisms can be imported through response/activation ports |

The donor-specific design and license/runtime constraints are in
`docs/donors/RESIDENT_MODEL_IMPORT.md`. The v0 code records measured overlap
tests in the experiment ledger; it does not infer novelty from different names.
