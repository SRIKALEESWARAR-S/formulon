"""One mechanism for every optional framework ("backend switch").

A *domain* (quantum, autodiff, ...) has a fixed set of choices, an ``auto``
order, an environment variable and a thread-local scope.  Resolution priority is
always:  per-call argument  >  ``use_*`` scope  >  ``set_*``  >  env var  >  auto.

An explicit choice never silently falls back: asking for ``torch`` when torch is
missing raises ``ImportError`` with the install command.  ``auto`` picks the first
installed choice from ``auto_order``; the last entry must always be available.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import threading

__all__ = ["BackendSwitch"]


class BackendSwitch:
    def __init__(self, domain: str, choices: tuple, env_var: str, auto_order: tuple,
                 modules: dict, hints: dict):
        self.domain = domain
        self.choices = tuple(choices)
        self.env_var = env_var
        self.auto_order = tuple(auto_order)
        self.modules = dict(modules)      # choice -> importable module name (None = always available)
        self.hints = dict(hints)
        self._global = None
        self._local = threading.local()

    # -- discovery -------------------------------------------------------------
    def available(self) -> list:
        out = []
        for c in self.auto_order:
            mod = self.modules.get(c)
            if mod is None or importlib.util.find_spec(mod) is not None:
                out.append(c)
        return out

    def _check(self, name: str) -> str:
        if not isinstance(name, str):
            raise ValueError(f"{self.domain} backend must be a string")
        name = name.lower()
        if name not in self.choices + ("auto",):
            raise ValueError(f"unknown {self.domain} backend {name!r}; choose one of {self.choices + ('auto',)}")
        return name

    # -- configuration ---------------------------------------------------------
    def set(self, name: str) -> None:
        self._global = self._check(name)

    def reset(self) -> None:
        self._global = None
        self._local.name = None

    def get(self) -> str:
        """Configured preference (may be 'auto')."""
        local = getattr(self._local, "name", None)
        if local:
            return local
        if self._global:
            return self._global
        return self._check(os.environ.get(self.env_var, "auto"))

    @contextlib.contextmanager
    def use(self, name: str):
        prev = getattr(self._local, "name", None)
        self._local.name = self._check(name)
        try:
            yield
        finally:
            self._local.name = prev

    def resolve(self, name: str | None = None) -> str:
        """Concrete installed backend for this call."""
        name = self._check(name) if name else self.get()
        if name == "auto":
            return self.available()[0]
        if name not in self.available():
            raise ImportError(f"{self.domain} backend {name!r} is not installed. {self.hints.get(name, '')}".strip())
        return name
