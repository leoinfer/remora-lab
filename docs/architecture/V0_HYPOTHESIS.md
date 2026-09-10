# Remora-v0 hypothesis and falsification gates

## Primary hypothesis

At a matched small scale, a language model with explicit module boundaries,
shared latent communication, replaceable specialists, and local plasticity can
retain more accumulated capability per adaptation unit than a monolithic causal
Transformer, while still learning ordinary next-token structure from scratch.

This is a hypothesis about update economics and recoverability, not a claim of
better initial perplexity.

## Operational claims

1. **Scratch learning.** Both models reduce held-out next-token loss on a fixed
   synthetic text/code/math stream without pretrained weights.
2. **Specialization.** Routed modules develop non-uniform activity or task
   selectivity that exceeds a shuffled-router control.
3. **Evidence separation.** Inherited and experienced evidence remain separately
   inspectable; a conditional experience can override a prior only under its
   supported condition.
4. **Independence.** Duplicating one evidence lineage does not produce the same
   confidence increase as adding independent clusters.
5. **Surgical consolidation.** After local consolidation, the target task can be
   solved without retrieval with a small parameter-update fraction and without
   large unrelated-task loss.
6. **Replacement.** Replacing a meaningful expert and training only that expert
   plus declared ports can recover target performance while preserving a useful
   fraction of old capability.
7. **Lineage.** The factual manual answers dependency and ancestry questions from
   the graph; a learned reader must not exceed graph-supported facts.
8. **Resurrection.** A candidate that failed under state S1 receives higher
   retest priority after the recorded failure condition is changed.

## Falsification conditions

The v0 thesis is weakened or falsified for a mechanism if any of these persist
after debugging and matched reruns:

- Remora cannot fit a tiny batch while the baseline can;
- local updates change more than 50% of trainable parameters or destroy the
  held-out old-task score relative to full-model adaptation;
- duplicate evidence produces materially the same effective evidence as truly
  independent evidence;
- removing retrieval after consolidation causes target accuracy to collapse;
- replacing a used module has no causal effect, or requires broad retraining;
- graph-backed manual answers contain unsupported dependencies;
- resurrection priority does not respond to a changed surrounding state.

Thresholds are gates for this prototype, not universal laws. Every experiment
records its own expected result and falsification condition before execution.
