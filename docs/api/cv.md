# Cross-Validation API

## `xdflow.cv.base`

Primary class:

- `CrossValidator`: abstract orchestration layer for train/validation/holdout evaluation

Key responsibilities:

- splitting holdout and CV folds
- separating stateless and stateful pipeline steps
- resolving scoring functions
- storing out-of-fold predictions and holdout results

## `xdflow.cv.kfold`

Primary classes:

- `KFoldValidator`
- `GroupedKFoldValidator`

Use these for repeated train/validation splitting with optional shuffling, holdout sets, and stratification.

`KFoldValidator` can also keep repeated `orig_trial` groups together when augmented samples share the same original-trial coordinate.

## `xdflow.cv.domain`

Primary class:

- `SampledDomainKFoldValidator`

Use this for few-shot transfer evaluation. Validation folds are built from target-domain trials, while training folds include all source-domain trials plus a sampled subset of target-domain trials.

## `xdflow.cv.leave_group_out`

Primary classes:

- `LeaveGroupOutValidator`
- `LeaveSessionOutValidator`
- `LeaveAnimalOutValidator`

These validators are designed for experimental designs where entire groups must be held out together.
