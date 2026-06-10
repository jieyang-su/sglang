from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    set_mamba_track_indices_from_reqs,
)
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_alloc_reserve_per_decode,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    create_extend_after_decode_spec_info,
    generate_simulated_accept_index,
)
from sglang.srt.speculative.triton_ops.cache_locs import (
    assign_draft_cache_locs_contiguous,
    assign_extend_cache_locs_func,
)
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2
from sglang.srt.utils.async_probe import maybe_detect_nan, maybe_detect_oob

if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )

_is_npu = is_npu()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _draft_runner_of(worker):
    """Draft model_runner accessor across worker shapes.

    v2 draft workers (`EagleDraftWorker` and subclasses) expose the draft
    model_runner as `draft_runner`; fall back to `model_runner` for workers
    that run the draft model directly.
    """
    return (
        worker.draft_runner if hasattr(worker, "draft_runner") else worker.model_runner
    )


def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """Copy the prefix partial-tail page into each branch's first-page holes (page>1 + topk>1).

    The draft-decode expand pass reads each branch's own draft page by block id
    (cache_loc // page_size), so branch b>=1's hole slots [0, last_page) must hold the
    real prefix tail (branch 0's first page already is it). Mirrors V1 #7725.
    """
    if topk <= 1:
        return
    bs = rows.shape[0]
    page_off = torch.arange(page_size, device=rows.device, dtype=torch.int64)
    branches = torch.arange(1, topk, device=rows.device, dtype=torch.int64).view(
        1, topk - 1, 1
    )
    # Source: the prefix tail page [prefix_base, prefix_base + page_size), one per branch.
    src_pos = (prefix_base.view(bs, 1, 1) + page_off.view(1, 1, page_size)).expand(
        bs, topk - 1, page_size
    )
    # Target: branch b's first page [prefix_base + b*num_new_pages*page, + page_size).
    tgt_pos = (
        prefix_base.view(bs, 1, 1)
        + branches * (num_new_pages.view(bs, 1, 1) * page_size)
        + page_off.view(1, 1, page_size)
    )
    # Only [0, last_page) holds real prefix KV; [last_page, page_size) are the branch's
    # own draft slots and must not be overwritten.
    vmask = (page_off.view(1, 1, page_size) < last_page.view(bs, 1, 1)).expand(
        bs, topk - 1, page_size
    )
    src_slots = torch.gather(rows, 1, src_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    tgt_slots = torch.gather(rows, 1, tgt_pos.reshape(bs, -1)).reshape(
        bs, topk - 1, page_size
    )[vmask]
    if src_slots.numel() > 0:
        token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)


@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    # Shape info for padding
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def prepare_for_v2_verify(
        self,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            maybe_detect_oob(
                batch.input_ids,
                0,
                batch.model_config.vocab_size,
                "v2 prepare_for_verify input_ids",
            )
            device = batch.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )

            if get_global_server_args().enable_mamba_extra_buffer():
                set_mamba_track_indices_from_reqs(batch)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

            # TBO's split_spec_info reads these; no-verify-sync leaves both None.
            self.seq_lens_cpu = batch.seq_lens_cpu
            self.seq_lens_sum = (
                int(batch.seq_lens_cpu.sum())
                if batch.seq_lens_cpu is not None
                else None
            )

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        capture_mode = (
            CaptureHiddenMode.NULL
            if target_worker.model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.capture_hidden_mode = capture_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.replay_prepare(
                verify_forward_batch
            )
            verify_forward_batch.mark_forward_metadata_ready()
        # Non-cuda-graph: defer init to forward_extend, which runs after
        # `_forward_raw -> prepare_mlp_sync_batch` pads the batch. Initing
        # here would use pre-pad shapes and trip DSv4 indexer shape match.

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        device = batch.device
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=device)
            num_correct_drafts = torch.empty(0, dtype=torch.int32, device=device)
            accept_index = torch.empty(0, dtype=torch.int32, device=device)
            return predict, num_correct_drafts, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits

        # Apply penalty
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after softmax")
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_k_renorm")
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            maybe_detect_nan(target_probs, "v2 verify: target_probs after top_p_renorm")
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=num_correct_drafts,  # mutable
                candidates=candidates,
                # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # `num_correct_drafts` stays drafts-only inside this function; the returned
        # tensor includes the trailing/bonus token via out-of-place +1 so the
        # name no longer flips semantics mid-function (naming doc C2).
        return predict, num_correct_drafts + 1, accept_index


@dataclass
class EagleDraftInput(SpecInput):
    # For idle stubs use `create_idle_input`, not the bare ctor: `filter_batch`
    # / `merge_batch` slice / cat `topk_p` / `topk_index` / `hidden_states` /
    # `bonus_tokens` unconditionally.

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size) - one hidden per req, consumed by `draft` forward.
    # None when the spec algorithm's draft doesn't read hidden_states
    # (e.g., STANDALONE — vanilla LLM draft).
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Per-req bonus token (the "+1" target prediction at end of each accept
    # chain). Written by `EagleDraftExtendInput.prepare_extend_after_decode`;
    # the worker copies it here for next iter's draft.
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # V2 overlap worker only: req_pool_indices used as buf slot keys.
    future_indices: Optional[torch.Tensor] = None
    # V2 reuses `EagleDraftInput` across phases (V1 has a separate
    # `EagleDraftExtendInput` for these). Set during V2's draft-extend.
    num_correct_drafts: Optional[torch.Tensor] = None
    num_accept_tokens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Decode-phase `hidden_states` width: draft self-chain output
        (draft model writes its own last hidden back via `capture_for_decode`
        and the draft loop). Returns None when the draft architecture doesn't
        consume the field (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.spec_hidden_size

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            self.bonus_tokens = self.bonus_tokens[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
            self.bonus_tokens = self.bonus_tokens[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        # Detect idle stub by `topk_index` length (idle inputs have
        # shape[0] == 0 across all fields). Don't use `hidden_states is None`:
        # for STANDALONE all non-idle inputs also have None hidden_states.
        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
            self.bonus_tokens = spec_info.bonus_tokens
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if len(spec_info.topk_index) == 0:
            return
        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])

    def prepare_for_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        double_alloc = get_alloc_reserve_per_decode()

        cur_kv_lens = [0] * bs
        nxt_kv_lens = [0] * bs
        num_needed_tokens = 0
        for i, r in enumerate(batch.reqs):
            cur = r.kv_allocated_len
            # max(cur, ...) clamps so adaptive downswitch (smaller alloc_len_per_decode)
            # cannot make nxt < cur and corrupt allocator state. kv_committed_len lags
            # batch.seq_lens by ~1 verify in overlap mode, so we react to adaptive
            # switches one batch later than a seq_lens-based baseline; the 2*alloc
            # over-allocation buffer absorbs that lag.
            nxt = max(cur, r.kv_committed_len + double_alloc)
            cur_kv_lens[i] = cur
            nxt_kv_lens[i] = nxt
            num_needed_tokens += nxt - cur
            r.kv_allocated_len = nxt
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")

        # Fail fast if the page>1 + topk>1 draft over-allocation
        # (get_alloc_reserve_per_decode) outgrows the req_to_token row: the write below
        # would OOB and free would leak KV. The row is widened to hold it in _init_pools
        # (PR #26972); fail here with a clear error, not on a later cryptic CUDA assert.
        from sglang.srt.server_args import get_global_server_args

        if page_size > 1 and (get_global_server_args().speculative_eagle_topk or 1) > 1:
            max_alloc_len = int(nxt_kv_lens_cpu.max())
            row_width = batch.req_to_token_pool.req_to_token.shape[1]
            assert max_alloc_len <= row_width, (
                f"spec v2 page>1 topk>1 draft over-allocation ({max_alloc_len}) exceeds "
                f"req_to_token row width ({row_width}); page_size={page_size}. Widen the "
                f"row to hold committed + get_alloc_reserve_per_decode (PR #26972)."
            )

        # non_blocking H2D: a blocking .to() syncs the schedule stream, which the WAR
        # barrier has chained to the prev forward -> host stalls a full forward.
        cur_kv_lens_device = cur_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        nxt_kv_lens_device = nxt_kv_lens_cpu.to(device=batch.device, non_blocking=True)
        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens_device,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens_device,
                cur_kv_lens_cpu,
                nxt_kv_lens_device,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_device,
            nxt_kv_lens_device,
            out_cache_loc,
            bs,
        )

    def prepare_for_v2_draft(
        self,
        req_to_token_pool: ReqToTokenPool,
        batch: ScheduleBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # Assign cache locations (draft-write targets).
            page_size = batch.token_to_kv_pool_allocator.page_size
            if page_size == 1 or topk == 1:
                batch.out_cache_loc = torch.empty(
                    (bs * topk * num_steps,),
                    dtype=torch.int64,
                    device=batch.device,
                )
                # FIXME(lsyin): align with the default code path
                assign_draft_cache_locs_contiguous[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )
            else:
                # page_size > 1 + topk > 1: per-branch page-aligned draft pages.
                # Reduce out_cache_loc from the page-aligned tree region down to the
                # dense draft slots (skip each branch's duplicated prefix-tail slots
                # and trailing padding), matching generate_draft_decode_kv_indices'
                # paged read formula: prefix_base + t*num_new_pages*page + last_page + s.
                # base is batch.seq_lens (== KV-ready committed prefix at draft time;
                # the bonus is the tree root written by verify, not part of [0:seq_lens]).
                rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
                seq_lens = batch.seq_lens.to(torch.int64)
                last_page = seq_lens % page_size
                prefix_base = seq_lens - last_page
                num_new_pages = (last_page + num_steps + page_size - 1) // page_size
                topk_ids = torch.arange(
                    topk, device=rows.device, dtype=torch.int64
                ).view(1, topk)
                starts = (
                    prefix_base.view(bs, 1)
                    + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                    + last_page.view(bs, 1)
                )
                steps = torch.arange(
                    num_steps, device=rows.device, dtype=torch.int64
                ).view(1, 1, num_steps)
                pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
                batch.out_cache_loc = (
                    torch.gather(rows, 1, pos).reshape(-1).contiguous()
                )

                # Each branch's page-aligned region starts with `last_page` hole slots
                # overlapping the prefix tail page; duplicate the real prefix-tail KV
                # into them so whole-page reads stay coherent (see helper docstring).
                duplicate_prefix_tail_to_draft_branches(
                    draft_model_runner.token_to_kv_pool,
                    rows,
                    prefix_base,
                    last_page,
                    num_new_pages,
                    topk,
                    page_size,
                )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        bs = len(batch.seq_lens)
        extend_num_tokens = bs * num_draft_tokens
        # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
        gpu_only = batch.seq_lens_cpu is None

        batch.spec_info = self
        batch.input_ids = predict
        maybe_detect_oob(
            batch.input_ids,
            0,
            batch.model_config.vocab_size,
            "v2 prepare_for_extend_to_fill_draft_kvcache input_ids",
        )
        # init_new requires both list or both Tensor;
        # gpu_only emits device tensors to skip H2D.
        if gpu_only:
            batch.prefix_lens = batch.seq_lens.to(torch.int32)
            batch.extend_lens = torch.full(
                (bs,), num_draft_tokens, dtype=torch.int32, device=batch.seq_lens.device
            )
        else:
            batch.prefix_lens = batch.seq_lens_cpu.tolist()
            batch.extend_lens = [num_draft_tokens] * bs
        batch.extend_num_tokens = extend_num_tokens
        capture_mode = (
            CaptureHiddenMode.NULL
            if draft_model_runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        batch.capture_hidden_mode = capture_mode
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        # Forward sees post-write length (draft extend writes num_draft_tokens
        # slots); mutation stays on forward_batch to preserve SB.seq_lens.
        forward_batch.seq_lens = forward_batch.seq_lens + num_draft_tokens
        if not gpu_only:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens_cpu + num_draft_tokens
            forward_batch.seq_lens_sum = int(forward_batch.seq_lens_cpu.sum())
        else:
            # Supply CPU mirror (extend_seq_lens are all num_draft_tokens) so
            # backend max() reads from list without a per-iter D2H sync.
            forward_batch.extend_seq_lens_cpu = [num_draft_tokens] * bs
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
            # Planned pre-pad; do NOT opt into post-pad re-plan. DSA's indexer
            # cannot rebuild its deep_gemm schedule_meta on a DP-padded batch
            # (the `_batch_size == batch_size` assertion, see #27091); the
            # marked pre-pad metadata is used as-is, matching the proven
            # skip_attn_backend_init=True behavior.
            forward_batch.mark_forward_metadata_ready()
        return forward_batch


@dataclass
class EagleDraftExtendInput(SpecInput):
    """Inputs to the draft-extend forward (the per-accepted-token pass after verify).

    Carries the post-verify accept bookkeeping (`num_correct_drafts` /
    `num_accept_tokens`) and sliced hidden states into the draft-extend
    forward; built by `EAGLEDraftExtendCudaGraphRunner`.
    """

    # shape: (total_accepted, hidden_size). Sliced from verify-time hidden_states
    # by accept_index; consumed by the draft-extend forward. None when the spec
    # algorithm's draft doesn't read hidden_states (e.g., STANDALONE).
    hidden_states: Optional[torch.Tensor] = None

    # Per-req accept counts. `num_accept_tokens = num_correct_drafts + 1`.
    # Both kept for cuda-graph buffer indexing and the
    # `create_extend_after_decode_spec_info` kernel.
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    # CPU view, read by attention backends during the extend forward.
    num_accept_tokens_cpu: List[int] = None

    # Batch-state slices for the draft-extend forward. Set by verify (sliced to
    # reqs continuing into next iter). `prepare_extend_after_decode` copies
    # these onto `batch.{input_ids, seq_lens, seq_lens_cpu, req_pool_indices}`.
    #   - input_ids:        accept tokens flat over surviving reqs
    #   - seq_lens / _cpu:  per-req sequence length (post-accept)
    #   - req_pool_indices: per-req kv-pool slot
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    # Set by `prepare_extend_after_decode`:
    #   - positions: kernel-written, shape `[total_accepted]`.
    #   - bonus_tokens: kernel-written, shape `[bs]`. The worker reads this
    #     post-extend to populate next iter's `EagleDraftInput.bonus_tokens`.
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Extend-phase `hidden_states` width: target's `spec_hidden_size`,
        widened to `num_aux * target_hidden` for EAGLE-3 aux mode. Returns
        None when the draft architecture doesn't consume the field
        (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        target_cfg = worker.target_worker.model_runner.model_config
        if not (
            worker.speculative_algorithm.is_eagle3()
            and worker.eagle_use_aux_hidden_state
        ):
            return target_cfg.spec_hidden_size

        hf_config = target_cfg.hf_config

        # `num_aux` resolution: explicit attr > eagle_config layer_ids > default 3.
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_config = getattr(hf_config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            num_aux = len(layer_ids) if layer_ids else 3

        target_hidden = getattr(hf_config, "target_hidden_size", target_cfg.hidden_size)
        return target_hidden * num_aux

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return worker.target_worker.model_runner.model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        # Caller must have installed `self` as `batch.spec_info` before calling.
        assert batch.spec_info is self
        if batch.forward_mode.is_idle():
            return

        # The kernel below populates `self.positions` and `self.bonus_tokens`;
        # the worker reads `self.bonus_tokens` to construct next iter's
        # `EagleDraftInput`.
        batch.input_ids = self.input_ids
        batch.extend_lens = self.num_accept_tokens_cpu
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = self.seq_lens
        batch.seq_lens_cpu = self.seq_lens_cpu
        batch.req_pool_indices = self.req_pool_indices
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.bonus_tokens = torch.empty_like(self.num_accept_tokens, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.num_accept_tokens,
            self.positions,
            self.bonus_tokens,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.num_accept_tokens, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None


@dataclass
class EagleVerifyOutput:
    # Next iter's draft-extend input, installed as `batch.spec_info` for the
    # draft-extend forward.
    draft_extend_input: EagleDraftExtendInput
    # Logit outputs from target worker.
    logits_output: LogitsProcessorOutput
    # All accepted tokens flat across all reqs incl. those that finished this
    # step. Includes the bonus token. Used for output processing.
    accept_tokens: torch.Tensor
    # Accepted token length per sequence in a batch in CPU (full set).
    num_correct_drafts_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accept_indices: torch.Tensor
    # Whether the target verify forward ran a captured cuda graph. Set by
    # the worker after `EagleVerifyInput.sample` returns; default kept so
    # idle / direct constructions don't have to pass it.
    can_run_cuda_graph: bool = False

    @classmethod
    def create_idle(
        cls,
        *,
        draft_extend_input: EagleDraftExtendInput,
        logits_output: LogitsProcessorOutput,
        device: torch.device,
        spec_steps: int,
    ) -> "EagleVerifyOutput":
        return cls(
            draft_extend_input=draft_extend_input,
            logits_output=logits_output,
            accept_tokens=torch.empty(0, dtype=torch.long, device=device),
            num_correct_drafts_per_req_cpu=[],
            accept_indices=torch.full(
                (0, spec_steps + 1), -1, dtype=torch.int32, device=device
            ),
        )
