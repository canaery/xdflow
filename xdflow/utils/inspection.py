from __future__ import annotations

from inspect import Parameter, signature
from typing import Any


def _get_declared_init_params(cls: type) -> set[str]:
    """Return explicit __init__ parameter names for a class."""
    params: set[str] = set()
    init = getattr(cls, "__init__", None)
    if init is None:
        return params

    try:
        sig = signature(init)
    except (TypeError, ValueError):
        return params

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        params.add(name)

    extra = getattr(cls, "_cooperative_init_kwarg_names", None)
    if extra:
        params.update(extra)

    return params


def collect_super_init_param_names(child_cls: type[Any], stop_class: type[Any]) -> set[str]:
    """
    Collect the names of constructor parameters consumed by cooperative superclasses.

    Args:
        child_cls: The most-derived class participating in cooperative init.
        stop_class: The class whose direct superclasses should be inspected.

    Returns:
        set[str]: Names of keyword arguments expected by the remaining classes in the MRO.
    """
    mro = child_cls.mro()
    if stop_class not in mro:
        raise ValueError(f"{stop_class.__name__} is not in the MRO of {child_cls.__name__}.")

    collect = False
    params: set[str] = set()
    for cls in mro:
        if cls is stop_class:
            collect = True
            continue
        if not collect or cls is object:
            continue
        params.update(_get_declared_init_params(cls))
    return params
