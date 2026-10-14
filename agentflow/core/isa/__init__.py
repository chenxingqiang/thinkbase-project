"""ISA package initialization.

Both public names are resolved lazily: importing the package must stay
cheap because ``agentflow.core.config`` (and therefore almost every
framework entry point) imports ``agentflow.core.isa.instruction``, which
triggers this package's ``__init__``.  Eagerly importing ``.manager``
here would drag in Ray (~2s) for every process, even ones that never
touch the ISA machinery.
"""

__all__ = ["ISAManager", "InstructionSelector"]

_Lazy = {"ISAManager": ".manager", "InstructionSelector": ".selector"}


def __getattr__(name):
    if name in _Lazy:
        import importlib

        module = importlib.import_module(_Lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
