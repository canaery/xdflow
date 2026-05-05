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

## `xdflow.cv.leave_group_out`

Primary classes:

- `LeaveGroupOutValidator`
- `LeaveSessionOutValidator`
- `LeaveAnimalOutValidator`

These validators are designed for experimental designs where entire groups must be held out together.
