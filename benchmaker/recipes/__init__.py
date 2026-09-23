"""Recipe registry.

A recipe is a named, self-contained benchmark scenario exposed as a
``benchmaker <recipe> --args`` subcommand. Concrete recipes self-register at
import time by calling :func:`register`; ``benchmaker.cli`` then builds one
click command per registered recipe via ``recipes._factory.make_command``.
"""

from __future__ import annotations

from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts, run_recipe

_REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    """Register a recipe instance under ``recipe.name``."""
    if not recipe.name:
        raise ValueError("recipe must define a non-empty name")
    if recipe.name in _REGISTRY:
        raise ValueError(f"duplicate recipe {recipe.name!r}")
    _REGISTRY[recipe.name] = recipe
    return recipe


def get(name: str) -> Recipe:
    return _REGISTRY[name]


def all_recipes() -> list[Recipe]:
    return list(_REGISTRY.values())


# Import the concrete recipe modules so they self-register. Kept at the bottom
# to avoid a circular import (each module imports from this package).
from benchmaker.recipes import http, llm, sglang, sandbox, swebench, swebench_replay, agentic, pareval, tracelab, agentx  # noqa: E402,F401

__all__ = [
    "Recipe",
    "SharedOpts",
    "BuildResult",
    "run_recipe",
    "register",
    "get",
    "all_recipes",
]
