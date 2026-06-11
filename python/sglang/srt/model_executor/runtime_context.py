"""Process control context — the Global Context Object (unified PoC).

One container, ``RuntimeContext``, accessed via ``get_context()``. It collapses
the ~30 process-level globals scattered across ~15 files into a single object
expressed as ``context = f(model, server_args, forward_batch)``. It is organized
by **subsystem**, not by kind (there is no ``.config`` / ``.resources`` split a
developer must reason about):

  ctx.parallel : the parallel topology — every dim's size + rank AND the group
                 handles. ``tp_group`` is the PoC exemplar handle (a long-lived
                 resource read during forward); the other 8 groups stay in
                 ``parallel_state`` and migrate in Stage-B P4.
  ctx.flags    : flags, grouped under a subsystem wrapper when they form a family
                 (``flags.attn.*`` / ``flags.moe.*``) and kept flat when they are
                 a lone general property (``flags.is_extend_in_batch``). Static
                 flags (resolved from model/server_args, frozen after ``freeze()``)
                 and per-forward flags (set each forward from forward_batch) live
                 in the same namespace; the reader does ``get_flags().<…>``
                 regardless of tier.
  ctx.buffers  : lazily-materialized static buffers (``get_buffer()`` allocates on
                 first access and caches; ``release()`` drops the cache).

A developer never decides "runtime vs forward context": there is one context. A
flag is added to ``Flags`` and read/written via plain attribute access, symmetric:
``get_flags().attn.use_mla_backend`` to read, ``... = value`` to write — wherever
its inputs are known (at init / ``freeze()`` for static, each forward for
per-forward). Static groups enforce freeze on write; there are no setter
functions and no flag-routing registry.

Lifecycle: published once at ``ModelRunner.__init__`` with the engine parallel
dims; the attention dims + ``tp_group`` are filled IN PLACE after
``initialize_dp_attention`` (mutable container → no ``dataclasses.replace``);
``freeze()`` materializes static derived flags and closes the static write
surface; per forward, ``set_flags(...)`` updates the per-forward flags;
``reset_context()`` is the single teardown point.

Concurrency: ``_context`` is a plain module-level global (mirrors the
``_global_server_args`` / ``forward_context`` precedents), safe because a worker
process resolves and runs synchronously on one thread.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Parallel topology: sizes + ranks + long-lived group handles, co-located
# (the "parallel" subsystem owns both the scalars and the groups).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParallelContext:
    """Resolved parallel topology of this process. All fields default so the
    container can be published early (engine dims) and filled IN PLACE later
    (attention dims + ``tp_group`` after ``initialize_dp_attention``) — no
    ``dataclasses.replace``, no two-stage re-publish.

    ``*_rank`` fields that the main (tp_worker) path does not pass stay ``None``.
    """

    # --- engine side (known at ModelRunner.__init__) ---
    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    dp_size: int = 1  # folded by enable_dp_attention
    dp_rank: Optional[int] = None
    moe_ep_size: int = 1
    moe_ep_rank: int = 0
    moe_dp_size: int = 1
    moe_dp_rank: Optional[int] = None
    attn_cp_size: int = 1
    attn_cp_rank: Optional[int] = None
    # --- attention side (filled in place after initialize_dp_attention) ---
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    attn_dp_size: Optional[int] = None
    attn_dp_rank: Optional[int] = None
    local_attn_dp_size: Optional[int] = None
    local_attn_dp_rank: Optional[int] = None
    # --- group handles (long-lived resources read during forward) ---
    # PoC exemplar: the TP GroupCoordinator (read in all-reduce on every layer).
    # The other 8 groups stay in parallel_state and migrate in Stage-B P4; the
    # alias-aware destroy contract is NOT touched here.
    tp_group: Optional[Any] = None


# ---------------------------------------------------------------------------
# Flags: subsystem wrapper for a family, flat for a lone general property.
# ---------------------------------------------------------------------------


class _StaticFlags:
    """Mixin for a static flag group: plain attribute reads/writes (symmetric),
    writable until ``freeze()``, frozen after (G3). A scoped test/debug change
    goes through ``override()``; there are NO per-flag / per-group setter
    functions and no routing registry. Deliberately not a dataclass itself so the
    ``__dict__``-based ``__setattr__`` guard works on the subclass."""

    _frozen = False  # class default; flipped per-instance by freeze()

    def __setattr__(self, name, value):
        if name != "_frozen":
            if self.__dict__.get("_frozen"):
                raise RuntimeError(
                    f"{name!r} is a frozen static flag; use override() on this "
                    "flag group for a scoped test/debug change"
                )
            if name not in self.__dataclass_fields__:  # typo-safety (no slots)
                raise AttributeError(f"{type(self).__name__} has no flag {name!r}")
        object.__setattr__(self, name, value)

    def freeze(self) -> None:
        self._frozen = True

    @contextmanager
    def override(self, **kwargs):
        """Scoped set-with-restore that bypasses ``freeze`` (the only sanctioned
        post-freeze mutation), restoring on exception too:
        ``with get_flags().attn.override(use_mla_backend=True): ...``"""
        saved = []
        for name, value in kwargs.items():
            if name not in self.__dataclass_fields__:
                raise AttributeError(f"{type(self).__name__} has no flag {name!r}")
            saved.append((name, getattr(self, name)))
            object.__setattr__(self, name, value)  # bypass freeze guard
        try:
            yield
        finally:
            for name, old in saved:
                object.__setattr__(self, name, old)


@dataclass
class AttnFlags(_StaticFlags):
    """Attention-subsystem static flags (a family — more land here in Stage-B)."""

    # resolved from the model at init (the G3 declarative model override).
    use_mla_backend: bool = False


@dataclass
class MoeFlags(_StaticFlags):
    """MoE-subsystem static flags (a family — should_use_dp_reduce_scatterv etc.
    land here in Stage-B P3)."""

    # materialized once at freeze() from the existing predicate.
    use_cutlass_fp4_allgather: bool = False


@dataclass(slots=True)
class Flags:
    """The flags namespace. A subsystem family gets a wrapper (``attn`` / ``moe``);
    a lone general property stays flat (``is_extend_in_batch``). Reads and writes
    are symmetric attribute access — ``get_flags().attn.use_mla_backend`` to read,
    ``... = value`` to write — with no setter functions. The static sub-structs
    enforce freeze; the flat per-forward field is freely writable each forward."""

    attn: AttnFlags = field(default_factory=AttnFlags)
    moe: MoeFlags = field(default_factory=MoeFlags)
    # per-forward, flat — a batch-level property, not a moe/attn family member.
    is_extend_in_batch: bool = False


# ---------------------------------------------------------------------------
# Buffers: lazy materialize-at-get + cache + release (unchanged from M0.4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """Metadata for a lazily-materialized static buffer — shape/dtype/device,
    NOT the physical tensor (G1: generate at get_buffer time, not pre-allocated)."""

    name: str  # layered namespace key, e.g. "trtllm.mha_workspace"
    size_bytes: int
    dtype: "torch.dtype"
    device: "torch.device"


class BufferStore:
    """Lazy-materialize-at-get + cache + release. ``register`` records metadata
    only; the tensor is allocated on the FIRST ``get_buffer(name)`` and cached, so
    every later caller shares one object (data_ptr stable), equivalent to the old
    ``global ... if None`` init-once. ``release`` (folded into reset) drops the
    cache so a fresh engine / test re-allocates."""

    def __init__(self) -> None:
        self._specs: dict = {}
        self._cache: dict = {}

    def register(self, spec: "BufferSpec") -> None:
        # pure addition: record metadata, never allocate here (idempotent).
        self._specs[spec.name] = spec

    def get_buffer(self, name: str):
        buf = self._cache.get(name)
        if buf is None:  # lazy: allocate on first access, then cache
            import torch

            spec = self._specs[name]
            buf = torch.zeros(spec.size_bytes, dtype=spec.dtype, device=spec.device)
            self._cache[name] = buf
        return buf

    def release(self) -> None:
        # drop cached tensors; next get_buffer re-allocates. Wired only into reset.
        self._cache.clear()


# ---------------------------------------------------------------------------
# The one context object.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Metrics:
    """Runtime measurements captured at init (not config). ``pre_model_load_memory``
    is the available GPU memory before model load (measured by
    init_torch_distributed), read later by init_memory_pool to size the KV cache —
    so it lives on the context instead of being threaded through call signatures."""

    pre_model_load_memory: Optional[float] = None


@dataclass(slots=True)
class RuntimeContext:
    parallel: ParallelContext = field(default_factory=ParallelContext)
    flags: Flags = field(default_factory=Flags)
    buffers: BufferStore = field(default_factory=BufferStore)
    metrics: Metrics = field(default_factory=Metrics)
    _frozen: bool = False

    @classmethod
    def from_model_runner(cls, model_runner) -> "RuntimeContext":
        """Build the context from a fully-initialized ModelRunner — call ONCE,
        AFTER ``init_torch_distributed``. By then every parallel dim (engine +
        attention) and the TP group are resolved, so the whole context is built
        in one shot — no early-publish + in-place fill, no ``dataclasses.replace``.
        The 12+ field enumeration lives here (the construction site), not at the
        call-site.

        Engine sizes/ranks are read off the model_runner (already resolved — e.g.
        ``dp_size`` is folded); the attention dims + ``tp_group`` come from the
        ``dp_attention`` / ``distributed`` getters. ``server_args`` alone is not
        enough: per-process ranks come from the distributed env, not server_args."""
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_rank,
            get_attention_cp_size,
            get_attention_dp_rank,
            get_attention_dp_size,
            get_attention_tp_rank,
            get_attention_tp_size,
            get_local_attention_dp_rank,
            get_local_attention_dp_size,
        )

        mr = model_runner
        return cls(
            parallel=ParallelContext(
                tp_size=mr.tp_size,
                tp_rank=mr.tp_rank,
                pp_size=mr.pp_size,
                pp_rank=mr.pp_rank,
                dp_size=mr.dp_size,
                dp_rank=mr.dp_rank,
                moe_ep_size=mr.moe_ep_size,
                moe_ep_rank=mr.moe_ep_rank,
                moe_dp_size=mr.moe_dp_size,
                moe_dp_rank=mr.moe_dp_rank,
                attn_cp_size=get_attention_cp_size(),
                attn_cp_rank=get_attention_cp_rank(),
                attn_tp_size=get_attention_tp_size(),
                attn_tp_rank=get_attention_tp_rank(),
                attn_dp_size=get_attention_dp_size(),
                attn_dp_rank=get_attention_dp_rank(),
                local_attn_dp_size=get_local_attention_dp_size(),
                local_attn_dp_rank=get_local_attention_dp_rank(),
                tp_group=get_tp_group(),  # the .resources forward-read exemplar
            ),
            # G3 declarative model override (static attn flag, set pre-freeze).
            flags=Flags(attn=AttnFlags(use_mla_backend=mr.use_mla_backend)),
        )

    def freeze(self) -> None:
        """Materialize static derived flags and close the static write surface.

        Lands after ``init_memory_pool`` per resolution cycle — i.e. after
        ``initialize_dp_attention`` + ``initialize_moe_config`` — so the predicate
        inputs are resolved. A second ModelRunner re-publishes a fresh (unfrozen)
        context, so multi-runner is safe. After this, ``set_flags`` on a static
        flag raises (use ``override`` for a scoped test/debug change)."""
        # G4 static-side materialize: the predicate stays the single source of
        # truth (clauses not reimplemented); we cache its result for one named read.
        # This is the LAST static write — do it before freezing the moe group.
        from sglang.srt.layers.moe.utils import (
            should_use_flashinfer_cutlass_moe_fp4_allgather,
        )

        self.flags.moe.use_cutlass_fp4_allgather = (
            should_use_flashinfer_cutlass_moe_fp4_allgather()
        )
        self.flags.attn.freeze()
        self.flags.moe.freeze()
        self._frozen = True


# ---------------------------------------------------------------------------
# Module-global lifecycle (single publish / single teardown — G1).
# ---------------------------------------------------------------------------

_context: Optional[RuntimeContext] = None


def set_context(ctx: RuntimeContext) -> None:
    """Low-level publish primitive (overwrite the module global). Used by
    ``init_context`` and by tests that publish a hand-built context. A second
    ModelRunner re-publishes a fresh, unfrozen context, so multi-runner
    (EAGLE draft+target) is safe."""
    global _context
    _context = ctx


def init_context(model_runner) -> None:
    """Resolve and publish the process context — the single runtime entry point.

    Runs ``model_runner.init_torch_distributed()`` (which sets up the process
    groups + dp-attention and measures the available GPU memory before model
    load), builds the context from the now-resolved topology, stashes that
    measurement on ``ctx.metrics``, and publishes. Returns nothing — consumers
    read ``get_context()`` (e.g. init_memory_pool reads
    ``get_context().metrics.pre_model_load_memory``)."""
    pre_model_load_memory = model_runner.init_torch_distributed()
    ctx = RuntimeContext.from_model_runner(model_runner)
    ctx.metrics.pre_model_load_memory = pre_model_load_memory
    set_context(ctx)


def get_context() -> RuntimeContext:
    if _context is None:
        raise ValueError("RuntimeContext not initialized — call init_context() first")
    return _context


def has_context() -> bool:
    return _context is not None


def reset_context() -> None:
    """Single teardown point (G1) — the legacy globals have no such primitive
    (multi-engine / test-isolation hazard). Drops cached buffers + the tp_group
    reference; kept independent of destroy_model_parallel()'s alias-aware group
    destroy (the group's real destroy stays there). P8 folds these together."""
    global _context
    if _context is not None:
        _context.buffers.release()
        _context.parallel.tp_group = None
    _context = None


# ---------------------------------------------------------------------------
# Flags read / write surface (one getter, one generic setter).
# ---------------------------------------------------------------------------


def get_flags() -> Flags:
    """The flags namespace — the single read surface for flags. Reads and writes
    are symmetric attribute access: ``get_flags().attn.use_mla_backend`` /
    ``get_flags().moe.use_cutlass_fp4_allgather`` / ``get_flags().is_extend_in_batch``.
    Writing a static flag after ``freeze()`` raises; per-forward flat flags are
    freely writable; a scoped test change uses ``<group>.override(...)``.

    There is intentionally NO per-flag ``get_<name>()`` accessor (mirrors the
    no-per-flag-setter rule) — read the leaf off ``get_flags()`` directly."""
    return get_context().flags


# ---------------------------------------------------------------------------
# Parallel accessors — thin reads off ctx.parallel.<field>. New names; they do
# NOT alias the existing parallel_state group readers (those stay).
# ---------------------------------------------------------------------------


def _parallel() -> ParallelContext:
    return get_context().parallel


def get_tp_size() -> int:
    return _parallel().tp_size


def get_tp_rank() -> int:
    return _parallel().tp_rank


def get_pp_size() -> int:
    return _parallel().pp_size


def get_pp_rank() -> int:
    return _parallel().pp_rank


def get_dp_size() -> int:
    return _parallel().dp_size


def get_dp_rank() -> Optional[int]:
    return _parallel().dp_rank


def get_moe_ep_size() -> int:
    return _parallel().moe_ep_size


def get_moe_ep_rank() -> int:
    return _parallel().moe_ep_rank


def get_moe_dp_size() -> int:
    return _parallel().moe_dp_size


def get_moe_dp_rank() -> Optional[int]:
    return _parallel().moe_dp_rank


def get_attn_cp_size() -> int:
    return _parallel().attn_cp_size


def get_attn_cp_rank() -> Optional[int]:
    return _parallel().attn_cp_rank


def get_attn_tp_size() -> Optional[int]:
    return _parallel().attn_tp_size


def get_attn_tp_rank() -> Optional[int]:
    return _parallel().attn_tp_rank


def get_attn_dp_size() -> Optional[int]:
    return _parallel().attn_dp_size


def get_attn_dp_rank() -> Optional[int]:
    return _parallel().attn_dp_rank


def get_local_attn_dp_size() -> Optional[int]:
    return _parallel().local_attn_dp_size


def get_local_attn_dp_rank() -> Optional[int]:
    return _parallel().local_attn_dp_rank
