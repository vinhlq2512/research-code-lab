# Continual Relation Extraction Lab

Domain terminology for continual learning and relation extraction benchmark experiments.

## Language

**Sequential Fine-Tuning (B0)**:
The continual learning lower-bound baseline where a single model instance is continually fine-tuned on a sequence of tasks with zero memory replay (M=0) and no anti-forgetting regularization.
_Avoid_: Naive Fine-Tuning, Standard Fine-Tuning

**Seen-Class Masking**:
The constraint where logit outputs for relation classes not yet seen up to current stage t are masked out with large negative values (-1e9) during both training and evaluation.
_Avoid_: Local Classification Head, Unconstrained Softmax

**Lower Bound Purity**:
The strict protocol constraint ensuring baseline B0 contains zero anti-forgetting components (no memory replay, prototypes, prompts, router, or distillation) to prevent underestimating Catastrophic Forgetting.
_Avoid_: Strong Baseline, Augmented Baseline

**Catastrophic Forgetting (AF)**:
The phenomenon where a neural network precipitously drops performance on previously learned tasks upon learning new tasks.
_Avoid_: Memory drift, degradation
