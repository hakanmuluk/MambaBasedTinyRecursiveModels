from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)

    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # If a valid_mask is provided, convert non-valid positions to ignore_index
    if valid_mask is not None:
        labels = labels.clone()
        labels[~valid_mask] = ignore_index

    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        # We need previous_carry to compute step-to-step corrected/damaged
        previous_carry = model_kwargs.get("carry", None)

        # Forward model
        new_carry, outputs = self.model(**model_kwargs)

        labels = new_carry.current_data["labels"]
        inputs = new_carry.current_data["inputs"]

        # build_sudoku_dataset.py does arr+1, so blank(0)->1
        BLANK_TOKEN_ID = 1

        with torch.no_grad():
            # Preds
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            # Only evaluate on blanks (givens excluded from loss/metrics)
            mask = (inputs == BLANK_TOKEN_ID)  # (B,L) bool

            # counts per example
            loss_counts = mask.sum(-1)  # (B,)
            loss_divisor = loss_counts.clamp_min(1).to(torch.float32).unsqueeze(-1)  # (B,1)

            # correctness on blanks only
            is_correct = mask & (preds == labels)            # (B,L)
            seq_is_correct = (is_correct.sum(-1) == loss_counts)  # (B,)

            # Metrics (halted-only, same as your original behavior)
            valid_metrics = new_carry.halted
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),

                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0.0,
                ).sum(),

                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

            # ------------------------------------------------------------
            # Step-to-step corrected / damaged (blanks only):
            # computed for CONTINUING examples only (exclude newly loaded),
            # and reported every step (not gated by new_carry.halted).
            # ------------------------------------------------------------
            prev_preds = getattr(previous_carry, "prev_preds", None) if previous_carry is not None else None

            if previous_carry is not None and prev_preds is not None and prev_preds.shape == preds.shape:
                continuing = ~previous_carry.halted  # (B,) True => not newly loaded this step

                prev_correct = mask & (prev_preds == labels)  # (B,L)
                now_correct = is_correct                      # (B,L)

                corrected = (~prev_correct) & now_correct     # wrong -> correct on blanks
                damaged = prev_correct & (~now_correct)       # correct -> wrong on blanks
                flipped = mask & (prev_preds != preds)        # changed value on blanks

                corrected_step = corrected.sum(-1).to(torch.float32)  # (B,)
                damaged_step = damaged.sum(-1).to(torch.float32)      # (B,)
                flipped_step = flipped.sum(-1).to(torch.float32)      # (B,)

                corrected_sum = torch.where(continuing, corrected_step, 0.0).sum()
                damaged_sum = torch.where(continuing, damaged_step, 0.0).sum()
                net_sum = torch.where(continuing, corrected_step - damaged_step, 0.0).sum()
                flipped_sum = torch.where(continuing, flipped_step, 0.0).sum()

                metrics.update({
                    "corrected_cells_step": corrected_sum,
                    "damaged_cells_step": damaged_sum,
                    "net_cells_step": net_sum,
                    "flipped_cells_step": flipped_sum,
                })

                # Optional: averages per continuing example (nice for W&B)
                denom = continuing.sum().clamp_min(1).to(torch.float32)
                metrics.update({
                    "corrected_cells_step_avg": corrected_sum / denom,
                    "damaged_cells_step_avg": damaged_sum / denom,
                    "net_cells_step_avg": net_sum / denom,
                    "flipped_cells_step_avg": flipped_sum / denom,
                })
            else:
                z = torch.tensor(0.0, device=preds.device)
                metrics.update({
                    "corrected_cells_step": z,
                    "damaged_cells_step": z,
                    "net_cells_step": z,
                    "flipped_cells_step": z,
                    "corrected_cells_step_avg": z,
                    "damaged_cells_step_avg": z,
                    "net_cells_step_avg": z,
                    "flipped_cells_step_avg": z,
                })

            # Save preds for next step comparison
            new_carry.prev_preds = preds.detach()

        # Losses (only blanks)
        lm_loss = (
            self.loss_fn(
                outputs["logits"],
                labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=mask,
            )
            / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
