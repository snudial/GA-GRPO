# exp2: EMA self-teacher + privileged boundary OPSD (optional; not used in the paper).
#
# Verifier-gated hybrid: groups with reward variation train through the normal
# group-relative PG path (see the "verifier_gated" advantage estimator); groups
# where every rollout failed get this stage instead — a forward-KL pull of the
# student toward its own EMA copy scoring the SAME boundary action tokens from
# a privileged context (verifier failure certificate; never the test traces).
#
# Everything is opt-in via EXP2_EMA_BETA; without it PolicyModelActor behaves
# exactly as upstream. Kept in one file so the upstream diff stays 3 lines.
import os
import sys

import torch


def _hook_dir():
    return os.environ.get("EXP2_HOOK_DIR", "")


def _load_hooks():
    """Import the experiment-side prompt/bank code (projects/exp2)."""
    d = _hook_dir()
    if d and d not in sys.path:
        sys.path.insert(0, d)
    import banks  # noqa: PLC0415
    import opsd_data  # noqa: PLC0415
    return banks, opsd_data


class Exp2OpsdMixin:
    """Bound into PolicyModelActor. Assumes self.actor / self.strategy /
    self.actor_optim / self.actor_scheduler / self.tokenizer exist."""

    # ---------------------------------------------------------------- EMA

    def exp2_setup(self):
        beta = os.environ.get("EXP2_EMA_BETA")
        self._exp2_beta = float(beta) if beta else None
        self._exp2_ema = None
        if self._exp2_beta is None:
            return
        dev = os.environ.get("EXP2_EMA_DEVICE", "cpu")
        ema = {}
        for name, p in self.actor.model.named_parameters():
            loc = p.detach()
            if hasattr(loc, "to_local"):
                loc = loc.to_local()
            ema[name] = loc.to(dev, copy=True)
        self._exp2_ema = ema

    def _exp2_local(self, p):
        loc = p.detach()
        return loc.to_local() if hasattr(loc, "to_local") else loc

    def _exp2_reshard(self):
        """Force FSDP2 modules back to the sharded state. A forward leaves the
        root module's params gathered (reshard_after_forward=False), so
        to_local() would return the FULL tensor and shard-shaped copies fail
        (measured: 151936 vs 75968 rows on the embedding)."""
        for m in self.actor.model.modules():
            if hasattr(m, "reshard"):
                try:
                    m.reshard()
                except Exception:  # noqa: BLE001 — non-FSDP .reshard duck types
                    pass

    def _exp2_ema_step(self):
        """theta_bar <- beta*theta_bar + (1-beta)*theta, on local shards.
        Called right after every actor optimizer step."""
        if not getattr(self, "_exp2_ema", None):
            return
        self._exp2_reshard()
        with torch.no_grad():
            for name, p in self.actor.model.named_parameters():
                loc = self._exp2_local(p)
                e = self._exp2_ema[name]
                # Blocking D2H copy: a non_blocking one returns the pinned buffer
                # before the queued optimizer kernels and the copy have run, so
                # the CPU lerp mixes stale buffer contents into the teacher.
                e.lerp_(loc.to(e.device, e.dtype), 1.0 - self._exp2_beta)

    def _exp2_swap_in(self):
        """Copy EMA values into the live bf16 module params, returning the
        saved originals. The optimizer holds fp32 master weights, so mutating
        the bf16 params between forwards is safe as long as we restore them
        before the next training forward."""
        self._exp2_reshard()
        saved = {}
        with torch.no_grad():
            for name, p in self.actor.model.named_parameters():
                loc = self._exp2_local(p)
                saved[name] = loc.clone()
                loc.copy_(self._exp2_ema[name].to(loc.device, loc.dtype,
                                                  non_blocking=True))
        return saved

    def _exp2_swap_back(self, saved):
        self._exp2_reshard()
        with torch.no_grad():
            for name, p in self.actor.model.named_parameters():
                self._exp2_local(p).copy_(saved[name])

    # --------------------------------------------------------------- OPSD

    def _exp2_span_logprobs(self, ids, ctx_len, target_len, need_grad):
        """Run the actor on one (ctx + target) sequence and return fp32
        log-probs over the vocab at every target position. logits[j] predicts
        token j+1, so target token i lives at logit index ctx_len - 1 + i."""
        device = torch.cuda.current_device()
        seq = ids.unsqueeze(0).to(device)
        att = torch.ones_like(seq)
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx:
            out = self.actor(seq, None, att)
            logits = out["logits"]
            if hasattr(logits, "full_tensor"):  # TP-sharded DTensor
                logits = logits.full_tensor()
            span = logits[0, ctx_len - 1: ctx_len - 1 + target_len, :]
            return torch.log_softmax(span.float(), dim=-1)

    def opsd_accumulate(self, uids, lam=1.0):
        """Accumulate the gradient of lam * KL(EMA teacher, privileged || student)
        on the boundary actions of all-zero groups. It takes no optimizer step of
        its own: the cycle's policy-gradient step applies it, so an OPSD arm makes
        exactly the baseline's Adam updates. Returns metrics."""
        if not getattr(self, "_exp2_ema", None) or not uids:
            return {}
        banks, opsd_data = _load_hooks()
        records = list(banks.load_opsd_records(uids).values())
        max_len = int(getattr(self.strategy.args.data, "max_len", 32768))
        pairs = opsd_data.build_pairs(records, self.tokenizer, max_len)
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        if not pairs:
            return {"opsd/num_pairs": 0.0}
        # Every rank must run the same number of forward/backward calls or the
        # FSDP collectives deadlock, so give each rank exactly `per` pairs,
        # cycling when there are fewer pairs than slots.
        per = max(1, len(pairs) // world)
        mine = [pairs[(rank * per + i) % len(pairs)] for i in range(per)]

        # Teacher pass: swap the EMA weights in once for all pairs.
        saved = self._exp2_swap_in()
        try:
            # The EMA must stay near the student: at lr 1e-6, beta >= 0.99 lags
            # ~1e-2 relative L2 at most, so a far or non-finite teacher is corrupt.
            sq = torch.zeros(2, device=torch.cuda.current_device())
            with torch.no_grad():
                for name, p in self.actor.model.named_parameters():
                    sq[0] += (self._exp2_local(p).float() - saved[name].float()).square().sum()
                    sq[1] += saved[name].float().square().sum()
            torch.distributed.all_reduce(sq)
            ema_rel_dist = (sq[0] / sq[1]).sqrt().item()
            if not ema_rel_dist <= 0.05:
                raise RuntimeError(f"EMA teacher is {ema_rel_dist:.3g} relative L2 from the student")
            self.actor.eval()
            teacher = []
            for pr in mine:
                lp = self._exp2_span_logprobs(
                    pr["teacher_ids"], pr["teacher_ctx_len"],
                    pr["target_len"], need_grad=False)
                teacher.append(lp)
        finally:
            self._exp2_swap_back(saved)

        # Student pass + forward KL(teacher || student) on the boundary span.
        # BFCL boundary actions are the FIRST action of an officially-failed
        # turn — often degenerate text — so raw token KLs can spike (measured:
        # 23 -> 38 -> nan over 15 interventions). Clip per-token KL and skip
        # non-finite pairs; every rank still runs `per` backward calls (zero
        # loss for a skipped pair) so the FSDP collectives stay aligned.
        kl_clip = float(os.environ.get("EXP2_OPSD_KL_CLIP", "10.0"))
        self.actor.train()
        total = torch.zeros((), device=torch.cuda.current_device())
        kl_report = 0.0
        kl_max = 0.0
        skipped = 0
        for i, pr in enumerate(mine):
            s_lp = self._exp2_span_logprobs(
                pr["student_ids"], pr["student_ctx_len"],
                pr["target_len"], need_grad=True)
            t_lp = teacher[i]
            w = pr["weights"].to(s_lp.device)
            p_t = t_lp.exp()
            kl_tok = (p_t * (t_lp - s_lp)).sum(-1)          # (target_len,)
            kl_tok = kl_tok.clamp(max=kl_clip)
            kl = (kl_tok * w).sum() / w.sum().clamp(min=1e-6)
            if not torch.isfinite(kl):
                skipped += 1
                kl = (s_lp.sum() * 0.0)  # keeps the backward/collective schedule
            loss = float(lam) * kl / per
            self.strategy.backward(
                loss, self.actor, self.actor_optim, name="actor",
                accumulate=True, scale_loss_by_accumulation=False,
                sync_gradients=(i == per - 1))
            total = total + loss.detach()
            kl_report += float(kl.detach().item())
            kl_max = max(kl_max, float(kl_tok.max().item()) if torch.isfinite(kl_tok.max()) else kl_max)
        # Norm of the OPSD gradient alone (the step adds the policy gradient later),
        # to compare with that step's reported grad_norm.
        grad_sq = torch.zeros((), device=torch.cuda.current_device())
        with torch.no_grad():
            for p in self.actor.model.parameters():
                if p.grad is not None:
                    grad_sq += self._exp2_local(p.grad).float().square().sum()
        torch.distributed.all_reduce(grad_sq)
        grad_norm = grad_sq.sqrt().item()
        if rank == 0:
            # Intervention ledger for the "did OPSD restore reward variation"
            # analysis: next time the curriculum revisits these tasks, compare
            # their group outcomes against this record.
            banks.append("opsd_interventions.jsonl", {
                "policy_version": banks.policy_version(),
                "tasks": sorted({p["task_id"] for p in pairs if p.get("task_id")}),
                "num_pairs": len(pairs),
                "kl": kl_report / per,
                "grad_norm": grad_norm,
            })
        # Hybrid mode wakes the vLLM engines right after this stage; the
        # teacher/student passes leave a large allocator cache that the
        # engines' re-allocation cannot get past (each OPSD firing at 9B
        # OOMed the next wake_up), so hand it back to the driver first.
        torch.cuda.empty_cache()
        return {"opsd/num_pairs": float(len(pairs)),
                "opsd/pairs_per_rank": float(per),
                "opsd/kl": kl_report / per,
                "opsd/kl_tok_max": kl_max,
                "opsd/skipped_pairs": float(skipped),
                "opsd/ema_rel_dist": ema_rel_dist,
                "opsd/loss": total.item(),
                "opsd/grad_norm": grad_norm}

    def opsd_release(self):
        """Drop OPSD gradients that no optimizer step consumed (the cycle's
        policy-gradient batch stepped zero times), so they never join a later
        cycle's update."""
        self.actor_optim.zero_grad(set_to_none=True)
