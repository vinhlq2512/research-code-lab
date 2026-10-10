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

**Joint Training (Upper Bound)**:
The theoretical and empirical upper-bound benchmark where all tasks, classes, and training samples are trained simultaneously in a single multitask session without sequential task shifts or temporal isolation.
_Avoid_: Multitask baseline, Full-data baseline

**Sequential Linear Probing (Frozen Backbone)**:
A continual learning diagnostic baseline where the neural feature extractor (BERT encoder) is 100% frozen, and only a shared linear classification head is trained sequentially across tasks to isolate classifier interference from representation drift.
_Avoid_: Linear probing without task structure, Static BERT

**Nearest Prototype Classifier (Frozen Backbone)**:
A non-parametric continual learning diagnostic baseline where the neural feature extractor (BERT encoder) is 100% frozen, each relation class is represented by the mean embedding (centroid) of its support samples, and test samples are assigned to the nearest prototype via metric distance (e.g., Cosine similarity) over observed classes.
_Avoid_: K-NN classifier, Dynamic prototype replay

**Cross-Task Prototype Ambiguity**:
The phenomenon in non-parametric continual learning where accuracy on past tasks decreases purely because new class prototypes populate previously unoccupied regions of the static embedding space, creating geometric overlap with representations of past classes without any neural weight degradation.
_Avoid_: Prototype drift, Centroid forgetting

