# Cross-Validation API

XDFlow validators run the evaluation loop. They build folds, apply split policies from named coordinates, reuse fold-invariant work, clone and refit stateful steps, score predictions, and keep outputs aligned with the source data.

Use these classes instead of handwritten sklearn split loops when validation depends on metadata, pipeline state, or reusable preprocessing.

## Base Validator

::: xdflow.cv.base.CrossValidator

## K-Fold Validators

::: xdflow.cv.kfold.KFoldValidator

::: xdflow.cv.kfold.GroupedKFoldValidator

## Domain Sampling

::: xdflow.cv.domain.SampledDomainKFoldValidator

## Leave-Group-Out Validators

::: xdflow.cv.leave_group_out.LeaveGroupOutValidator

::: xdflow.cv.leave_group_out.LeaveSessionOutValidator

::: xdflow.cv.leave_group_out.LeaveAnimalOutValidator

## Sklearn Adapter

::: xdflow.cv.sklearn_adapter.SklearnCVAdapter

::: xdflow.cv.sklearn_adapter.set_cv_container
