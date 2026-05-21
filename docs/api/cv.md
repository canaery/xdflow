# Cross-Validation API

XDFlow validators run the evaluation loop. They build folds, apply split policies from named coordinates, reuse fold-invariant work, clone and refit stateful steps, score predictions, and keep outputs aligned with the source data.

Use these classes instead of handwritten sklearn split loops when validation depends on metadata, pipeline state, or reusable preprocessing.

## `xdflow.cv.base`

Primary class:

- `CrossValidator`: abstract orchestration layer for train/validation/holdout evaluation

Key responsibilities:

- build holdout and CV folds from `DataContainer` inputs
- split, group, or stratify by named coordinates
- run fold-invariant stateless prefixes once when legal
- clone and refit stateful or split-dependent suffixes inside each fold
- resolve scoring functions
- store out-of-fold predictions and holdout results with metadata intact

## `xdflow.cv.kfold`

Primary classes:

- `KFoldValidator`
- `GroupedKFoldValidator`

Use these for repeated train/validation splitting with optional shuffling, holdout sets, grouping, and stratification by named coordinates.

`KFoldValidator` can also keep repeated `orig_trial` groups together when augmented samples share the same original-trial coordinate.

## `xdflow.cv.domain`

Primary class:

- `SampledDomainKFoldValidator`

Use this for few-shot transfer evaluation. Validation folds are built from target-domain trials, while training folds include all source-domain trials plus a sampled subset of target-domain trials. Domain, label, and sample-count policy are explicit validator inputs rather than side-channel split logic.

## `xdflow.cv.leave_group_out`

Primary classes:

- `LeaveGroupOutValidator`
- `LeaveSessionOutValidator`
- `LeaveAnimalOutValidator`

These validators are designed for experimental designs where entire groups must be held out together.
