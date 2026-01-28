from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from torch import nn
from pydantic import BaseModel
import random

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    LinearSwish,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding

# 🔹 Mamba import
from mamba_ssm.modules.mamba_simple import Mamba

IGNORE_LABEL_ID = -100


@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # "Transformer" / hidden config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    no_ACT_continue: bool = True  # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # 🔹 Mamba-specific hyperparameters (for L-level reasoning blocks)
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: float = 2.0
    mamba_dt_rank: str = "auto"  # passed directly to Mamba (e.g. "auto")

    mamba_two_stage: bool = False
    mamba_bimamba_v2_with_transformer: bool = False  # BiMambaV2 then self-attn then FFN
    mamba_bimamba_with_transformer_and_nn: bool = False

    # NEW: Multi-head BiMamba (like attention heads) + mid-MLP + self-attn + final MLP
    mamba_bimamba_heads_transformer_nn: bool = False
    mamba_bimamba_heads: int = 4
    mamba_bimamba_heads_out_proj: bool = True  # optional mixing after concat (adds params)

    mamba_bimamba_v2: bool = False       # if True → use BiMamba v2 block
    mamba_bimamba_v2_dropout: bool = True  # NEW: BiMamba v2 + dropout on residual branch
    mamba_dropout_p: float = 0.1            # NEW: dropout prob

    mamba_if_divide_out: bool = True     # match Vision Mamba's /2 behavior
    state_dep_dual_mamba: bool = False

    # 🔹 BiLSTM option (per-layer): BiLSTM -> downproj -> (final SwiGLU MLP)
    bilstm_with_nn: bool = True
    bilstm_hidden_size: int = 0      # 0 => use hidden_size (per direction)
    bilstm_num_layers: int = 1
    bilstm_downproj_bias: bool = False




class BiMambaV2(nn.Module):
    """
    Bi-directional Mamba v2 block implemented with PyTorch ops + selective_scan_fn
    (no dependency on causal_conv1d_cuda).

    Input:  hidden_states (B, L, D)
    Output: hidden_states (B, L, D)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        if_divide_out: bool = True,
        init_layer_scale: float | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_divide_out = if_divide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model), **factory_kwargs))

        # In-projection: [B, L, D] -> [B, L, 2 * d_inner]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Depthwise convs (forward & backward)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # Projections for dt, B, C (forward)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Projections for dt, B, C (backward)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt_proj like Vision Mamba
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Bias init so softplus(dt_bias) ∈ [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True

        self.dt_proj.bias._no_weight_decay = True
        self.dt_proj_b.bias._no_weight_decay = True

        # S4D A matrices (forward & backward)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.A_b_log._no_weight_decay = True

        # D (skip) parameters
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b._no_weight_decay = True

        # Final output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _ssm_pass(
        self,
        x_conv: torch.Tensor,
        z: torch.Tensor,
        A_log: torch.Tensor,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        One directional SSM pass.

        x_conv: [B, d_inner, L]
        z:      [B, d_inner, L]
        A_log:  [d_inner, d_state]
        """
        Bsz, d_inner, L = x_conv.shape
        d_state = self.d_state

        # Activation applied to conv output
        x_conv = self.act(x_conv)

        # x_dbl: [B*L, dt_rank + 2*d_state]
        x_dbl = x_proj(rearrange(x_conv, "b d l -> (b l) d"))
        dt, B_, C_ = torch.split(x_dbl, [self.dt_rank, d_state, d_state], dim=-1)

        # dt: [B*L, dt_rank] -> [B*L, d_inner] -> [B, d_inner, L]
        dt = dt_proj(dt)
        dt = rearrange(dt, "(b l) d -> b d l", b=Bsz, l=L)

        # B, C: [B, d_state, L]
        B_ = rearrange(B_, "(b l) n -> b n l", b=Bsz, l=L).contiguous()
        C_ = rearrange(C_, "(b l) n -> b n l", b=Bsz, l=L).contiguous()

        A = -torch.exp(A_log.float())

        y = selective_scan_fn(
            x_conv,
            dt,
            A,
            B_,
            C_,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        return y

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        """
        hidden_states: (B, L, D)
        returns: (B, L, D)
        """
        if inference_params is not None:
            raise NotImplementedError("BiMambaV2: inference cache path not implemented.")

        Bsz, L, D = hidden_states.shape
        assert D == self.d_model

        # In-projection: [B, L, D] -> [B, L, 2*d_inner] -> [B, 2*d_inner, L]
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d2 -> b d2 l")
        x, z = xz.chunk(2, dim=1)

        # ----- FORWARD DIRECTION -----
        x_conv_f = self.conv1d(x)[..., :L]
        y_f = self._ssm_pass(
            x_conv=x_conv_f,
            z=z,
            A_log=self.A_log,
            x_proj=self.x_proj,
            dt_proj=self.dt_proj,
            D=self.D,
        )

        # ----- BACKWARD DIRECTION -----
        x_b, z_b = x.flip(-1), z.flip(-1)
        x_conv_b = self.conv1d_b(x_b)[..., :L]
        y_b = self._ssm_pass(
            x_conv=x_conv_b,
            z=z_b,
            A_log=self.A_b_log,
            x_proj=self.x_proj_b,
            dt_proj=self.dt_proj_b,
            D=self.D_b,
        )
        y_b = y_b.flip(-1)

        # Combine directions
        if self.if_divide_out:
            y = (y_f + y_b) / 2.0
        else:
            y = y_f + y_b

        # Final projection to d_model
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class MultiHeadBiMambaV2(nn.Module):
    """
    Multi-head BiMambaV2:
      - Split hidden dim D into H heads (D_head = D/H)
      - Run independent BiMambaV2 per head
      - Concat back to (B, L, D)
      - Optional output projection to mix heads (like attention's o_proj)
    """
    def __init__(
        self,
        d_model: int,
        heads: int,
        d_state: int,
        d_conv: int,
        expand: float,
        dt_rank: int | str,
        if_divide_out: bool,
        dtype,
        use_out_proj: bool = True,
    ) -> None:
        super().__init__()
        assert heads > 0
        if d_model % heads != 0:
            raise ValueError(f"hidden_size ({d_model}) must be divisible by mamba_bimamba_heads ({heads})")
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.use_out_proj = use_out_proj

        self.mambas = nn.ModuleList(
            [
                BiMambaV2(
                    d_model=self.d_head,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    device=None,
                    dtype=dtype,
                    if_divide_out=if_divide_out,
                )
                for _ in range(heads)
            ]
        )

        if self.use_out_proj:
            self.out_proj = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        else:
            self.out_proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, D)
        B, L, D = hidden_states.shape
        assert D == self.d_model

        chunks = torch.chunk(hidden_states, self.heads, dim=-1)  # tuple of (B, L, d_head)
        outs = []
        for m, xh in zip(self.mambas, chunks):
            outs.append(m(xh.contiguous()))
        y = torch.cat(outs, dim=-1)  # (B, L, D)

        if self.out_proj is not None:
            y = self.out_proj(y)
        return y





class state_dependent_mamba(nn.Module):
    """
    State-dependent Mamba (pure PyTorch):
      Δ_t depends on running SSM state h_{t-1} (channelwise).

    BF16-friendly version:
      - Activations / run_state / y_out are bf16 (low memory)
      - softplus + exp discretization are computed in fp32 for stability
      - A_log, D, dt_state_weight stored in fp32 as "master params"

    Notes:
      - Still slower than fused selective_scan.
      - This is the "best effort" bf16 version without writing a custom fused kernel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        layer_idx=None,  # signature compat; unused
        device=None,
        dtype=None,      # pass torch.bfloat16 if you want weights in bf16
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        bias = False  # match original Mamba style

        # (These weights can be bf16 if dtype=torch.bfloat16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # dt_proj init
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias init so softplus(dt_bias) in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt.to(self.dt_proj.bias.dtype))
        self.dt_proj.bias._no_reinit = True
        self.dt_proj.bias._no_weight_decay = True

        # --- fp32 master params (keep stable) ---
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # fp32
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=torch.float32))  # fp32
        self.D._no_weight_decay = True

        self.dt_state_weight = nn.Parameter(
            torch.zeros(self.d_inner, d_state, device=device, dtype=torch.float32)  # fp32
        )
        nn.init.normal_(self.dt_state_weight, mean=0.0, std=1e-3)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def _state_term(self, run_state_bf16: torch.Tensor) -> torch.Tensor:
        """
        run_state_bf16: (B, d_inner, d_state) bf16
        returns:        (B, d_inner) bf16

        We compute tanh + projection in fp32, then cast back to bf16.
        """
        h_fp32 = torch.tanh(run_state_bf16.float())  # (B,d_inner,d_state) fp32
        st_fp32 = torch.einsum("bdn,dn->bd", h_fp32, self.dt_state_weight)  # fp32
        return st_fp32.to(run_state_bf16.dtype)  # bf16

    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("state_dependent_mamba: inference cache not implemented.")

        B, L, D = hidden_states.shape
        assert D == self.d_model

        # We expect bf16 model usage, but this will also work for fp16/fp32.
        model_dtype = hidden_states.dtype
        device = hidden_states.device

        # in_proj: (B,L,D) -> (B,2*d_inner,L)
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d2 -> b d2 l")
        x, z = xz.chunk(2, dim=1)  # (B,d_inner,L)

        # conv + silu in model dtype
        x = self.conv1d(x)[..., :L]
        x = self.act(x)

        # token-conditioned dt,B,C in model dtype
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_raw, B_raw, C_raw = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_linear = self.dt_proj(dt_raw)  # (B*L,d_inner) model dtype
        dt_linear = rearrange(dt_linear, "(b l) d -> b d l", b=B, l=L)

        B_seq = rearrange(B_raw, "(b l) n -> b n l", b=B, l=L).contiguous()
        C_seq = rearrange(C_raw, "(b l) n -> b n l", b=B, l=L).contiguous()

        # --- prepare fp32 constants ---
        A_fp32 = -torch.exp(self.A_log)               # (d_inner,d_state) fp32
        dt_bias_fp32 = self.dt_proj.bias.float()      # (d_inner,) fp32
        Dskip_fp32 = self.D                           # (d_inner,) fp32

        # --- bf16 recurrent state (memory saver) ---
        run_state = torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=model_dtype)

        # output buffer in bf16
        y_out = torch.empty(B, L, self.d_inner, device=device, dtype=model_dtype)

        for t in range(L):
            u_t = x[:, :, t]          # (B,d_inner) bf16
            z_t = z[:, :, t]          # (B,d_inner) bf16
            B_t = B_seq[:, :, t]      # (B,d_state) bf16
            C_t = C_seq[:, :, t]      # (B,d_state) bf16

            st = self._state_term(run_state)          # bf16

            # dt_eff computed in fp32 for stability
            dt_eff_fp32 = F.softplus(
                dt_linear[:, :, t].float()
                #+ dt_bias_fp32.unsqueeze(0)
                + st.float()
            )  # (B,d_inner) fp32

            # discretization in fp32, then cast to bf16 for state update
            # e = dt_eff * A  -> (B,d_inner,d_state)
            e_fp32 = torch.einsum("bd,dn->bdn", dt_eff_fp32, A_fp32)
            dA = torch.exp(e_fp32).to(model_dtype)  # bf16

            dB_fp32 = torch.einsum("bd,bn->bdn", dt_eff_fp32, B_t.float())
            dB = dB_fp32.to(model_dtype)            # bf16

            # state update in bf16
            run_state = run_state * dA + u_t.unsqueeze(-1) * dB

            # output: do the reduction in fp32, then cast back
            y_t_fp32 = torch.einsum("bdn,bn->bd", run_state.float(), C_t.float()) + Dskip_fp32.unsqueeze(0) * u_t.float()
            y_t_fp32 = y_t_fp32 * self.act(z_t.float())
            y_out[:, t, :] = y_t_fp32.to(model_dtype)

        return self.out_proj(y_out)




class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        self.norm_eps = config.rms_norm_eps
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.dropout = None


        # Decide mode
        if self.config.mlp_t:
            self.mode = "mlp_t"
        elif self.config.bilstm_with_nn:
            self.mode = "bilstm_with_nn"
        elif self.config.mamba_bimamba_heads_transformer_nn:
            self.mode = "bimamba_heads_transformer_nn"
        elif self.config.mamba_bimamba_v2_with_transformer:
            self.mode = "bimamba_v2_with_transformer"
        
        elif self.config.mamba_bimamba_v2_dropout:     # NEW
            self.mode = "bimamba_v2_dropout"


        elif self.config.mamba_bimamba_v2:
            self.mode = "bimamba_v2"
        elif self.config.mamba_bimamba_with_transformer_and_nn:
            self.mode = "bimamba_with_transformer_and_nn"
        elif self.config.state_dep_dual_mamba:
            self.mode = "state_dep_dual_mamba"
        elif self.config.mamba_two_stage:
            self.mode = "mamba_two_stage"
        else:
            self.mode = "simple_mamba"

        # ------------------------------------------------------------------
        # 1) mlp_t mode (original TRM)
        # ------------------------------------------------------------------
        if self.mode == "mlp_t":
            self.puzzle_emb_len = (
                -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                if self.config.puzzle_emb_len == 0
                else self.config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
            self.mamba = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

        elif self.mode == "bimamba_v2_dropout":
            print("GATING")
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = BiMambaV2(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
                if_divide_out=self.config.mamba_if_divide_out,
            )

            self.dropout = nn.Dropout(p=getattr(config, "mamba_dropout_p", 0.1))

            # ------------------------------------------------------------
            # NEW: gated residual for BiMambaV2 (conditioned on [x; Δ])
            #   α(x,Δ): tokenwise scalar (B,L,1) via 2-layer MLP
            #   m(x,Δ): tokenwise-channel mask (B,L,D) via 1-layer
            # ------------------------------------------------------------
            """ D = config.hidden_size
            H = D // 2  # gate hidden width (you can change to D//4 or D)

            self.gate_act = nn.SiLU()

            # α MLP: (2D -> H -> 1)
            self.alpha_fc1 = nn.Linear(2 * D, H, bias=True, dtype=self.forward_dtype)
            self.alpha_fc2 = nn.Linear(H, 1, bias=True, dtype=self.forward_dtype)

            # mask: (2D -> D)
            self.mask_fc = nn.Linear(2 * D, D, bias=True, dtype=self.forward_dtype)

            # Init: start mostly closed so TRM steps don't thrash
            with torch.no_grad():
                # Make alpha initially small: sigmoid(-2) ≈ 0.12
                self.alpha_fc2.bias.fill_(-2.0)
                # Make mask initially small-ish too
                self.mask_fc.bias.fill_(-2.0) """


        elif self.mode == "bilstm_with_nn":
            self.mlp_t = None
            self.mamba = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            # per-direction hidden size
            h = config.hidden_size if getattr(config, "bilstm_hidden_size", 0) in (0, None) else config.bilstm_hidden_size

            # NOTE: LSTM kernels can be picky with bf16 on some setups.
            # Safest: keep LSTM weights in fp32 and cast inputs/outputs in forward.
            self.bilstm = nn.LSTM(
                input_size=config.hidden_size,
                hidden_size=h,
                num_layers=config.bilstm_num_layers,
                bidirectional=True,
                batch_first=True,
            )

            # BiLSTM output is 2*h; we down-project to hidden_size
            self.bilstm_downproj = nn.Linear(
                2 * h,
                config.hidden_size,
                bias=config.bilstm_downproj_bias,
            )


        # ------------------------------------------------------------------
        # 2) NEW: Multi-head BiMamba + mid-MLP + Transformer self-attn
        # ------------------------------------------------------------------
        elif self.mode == "bimamba_heads_transformer_nn":
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = MultiHeadBiMambaV2(
                d_model=config.hidden_size,
                heads=config.mamba_bimamba_heads,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                if_divide_out=config.mamba_if_divide_out,
                dtype=self.forward_dtype,
                use_out_proj=config.mamba_bimamba_heads_out_proj,
            )

            self.mlp_mid = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )

            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        # ------------------------------------------------------------------
        # 2a) BiMamba v2 mode (one block, v2 SSMs inside)
        # ------------------------------------------------------------------
        elif self.mode == "bimamba_v2":
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = BiMambaV2(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
                if_divide_out=self.config.mamba_if_divide_out,
            )

    



        # ------------------------------------------------------------------
        # 2b) BiMamba v2 + mid MLP + Transformer self-attn (single BiMamba)
        # ------------------------------------------------------------------
        elif self.mode == "bimamba_with_transformer_and_nn":
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = BiMambaV2(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
                if_divide_out=self.config.mamba_if_divide_out,
            )

            self.mlp_mid = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )

            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        # ------------------------------------------------------------------
        # 2d) State-dependent dual Mamba (forward/backward) + downproj
        # ------------------------------------------------------------------
        elif self.mode == "state_dep_dual_mamba":
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None
            self.mamba = None

            self.state_dep_fwd = state_dependent_mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
            )
            self.state_dep_bwd = state_dependent_mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
            )
            self.state_dep_downproj = nn.Linear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=self.forward_dtype,
            )
            self.dropout = nn.Dropout(p=getattr(config, "mamba_dropout_p", 0.1))


        # ------------------------------------------------------------------
        # 2c) BiMamba v2 + Transformer (self-attn) in the same layer
        # ------------------------------------------------------------------
        elif self.mode == "bimamba_v2_with_transformer":
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = BiMambaV2(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                device=None,
                dtype=self.forward_dtype,
                if_divide_out=self.config.mamba_if_divide_out,
            )

            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        # ------------------------------------------------------------------
        # 3) Two-stage Mamba (your previous design)
        # ------------------------------------------------------------------
        elif self.mode == "mamba_two_stage":
            self.mlp_t = None

            self.mamba_first = Mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                layer_idx=None,
                device=None,
                dtype=self.forward_dtype,
            )

            self.dir_fwd = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
            self.dir_bwd = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )

            self.mamba_second = Mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                layer_idx=None,
                device=None,
                dtype=self.forward_dtype,
            )

            self.mamba = None

        # ------------------------------------------------------------------
        # 4) Default: single shared Mamba + manual bi-direction via flip
        # ------------------------------------------------------------------
        else:
            self.mlp_t = None
            self.mamba_first = None
            self.mamba_second = None
            self.dir_fwd = None
            self.dir_bwd = None

            self.mamba = Mamba(
                d_model=config.hidden_size,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                layer_idx=None,
                device=None,
                dtype=self.forward_dtype,
            )

        # final MLP over hidden_size for all modes
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )





    def _apply_bimamba_v2_dropout_gates(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, L, D)
        delta: (B, L, D)  proposed update from BiMambaV2

        Returns gated_delta: (B, L, D) = α(x,delta) * (m(x,delta) ⊙ delta)
        with α tokenwise scalar (B,L,1), m tokenwise-channel (B,L,D).
        """
        u = torch.cat([x, delta], dim=-1)  # (B, L, 2D)

        # α: 2-layer MLP -> sigmoid -> (B,L,1)
        h = self.gate_act(self.alpha_fc1(u))  # (B,L,H)
        alpha = torch.sigmoid(self.alpha_fc2(h).float()).to(x.dtype)  # (B,L,1)

        # m: 1-layer -> sigmoid -> (B,L,D)
        mask = torch.sigmoid(self.mask_fc(u).float()).to(x.dtype)  # (B,L,D)

        return alpha * (mask * delta)


    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, D]

        # ---------------- mlp_t ----------------
        if self.mode == "mlp_t":
            hidden_states = hidden_states.transpose(1, 2)  # [B, D, L]
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)  # [B, L, D]

        # ---------------- BiMamba v2 + dropout ----------------
        elif self.mode == "bimamba_v2_dropout": 
            x = hidden_states.contiguous()
            delta = self.mamba(x)  # proposed update Δ

            # Apply α(x,Δ) and m(x,Δ)
            # y = self._apply_bimamba_v2_dropout_gates(x, delta)

            # Keep your dropout on the residual branch
            y = self.dropout(delta)

            hidden_states = rms_norm(x + y, variance_epsilon=self.norm_eps)



        elif self.mode == "bilstm_with_nn":
            # Author-style post-norm residual around the recurrent module.
            # BiLSTM expects (B, L, D) already since batch_first=True.

            x = hidden_states.contiguous()

            # Run LSTM in fp32 for stability/compatibility; then project back and cast.
            lstm_out, _ = self.bilstm(x.to(torch.float32))                 # (B, L, 2*h)
            proj = self.bilstm_downproj(lstm_out)                          # (B, L, D)
            proj = proj.to(x.dtype)

            hidden_states = rms_norm(x + proj, variance_epsilon=self.norm_eps)


        # ---------------- Multi-head BiMamba + mid-MLP + self-attn ----------------
        elif self.mode == "bimamba_heads_transformer_nn":
            hidden_states = hidden_states.contiguous()

            # (1) Multi-head BiMamba (post-norm, author style)
            hidden_states = rms_norm(
                hidden_states + self.mamba(hidden_states),
                variance_epsilon=self.norm_eps
            )

            # (2) mid MLP (post-norm)
            hidden_states = rms_norm(
                hidden_states + self.mlp_mid(hidden_states),
                variance_epsilon=self.norm_eps
            )

            # (3) self-attn (EXACT author pattern)
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps
            )

        # ---------------- BiMamba v2 (single) ----------------
        elif self.mode == "bimamba_v2":
            x = hidden_states.contiguous()
            y = self.mamba(x)
            hidden_states = rms_norm(x + y, variance_epsilon=self.norm_eps)

        elif self.mode == "bimamba_with_transformer_and_nn":
            x = hidden_states.contiguous()
        
            # (1) BiMambaV2
            y = self.mamba(x)
            hidden_states = rms_norm(x + y, variance_epsilon=self.norm_eps)
        
            # (2) MLP in the middle
            mid = self.mlp_mid(hidden_states)
            hidden_states = rms_norm(hidden_states + mid, variance_epsilon=self.norm_eps)
        
            # (3) Transformer self-attention
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        elif self.mode == "state_dep_dual_mamba":
            x = hidden_states.contiguous()

            y_fwd = self.state_dep_fwd(x)
            rev_x = torch.flip(x, dims=[1])
            y_bwd = self.state_dep_bwd(rev_x)
            y_bwd = torch.flip(y_bwd, dims=[1])

            y = torch.cat([y_fwd, y_bwd], dim=-1)
            y = self.state_dep_downproj(y)

            y = self.dropout(y)

            hidden_states = rms_norm(x + y, variance_epsilon=self.norm_eps)

        # ---------------- two-stage Mamba ----------------
        elif self.mode == "mamba_two_stage":
            x = hidden_states.contiguous()

            y1_fwd = self.mamba_first(x)
            rev_x = torch.flip(x, dims=[1])
            y1_bwd = self.mamba_first(rev_x)
            y1_bwd = torch.flip(y1_bwd, dims=[1])

            y1_fwd = self.dir_fwd(y1_fwd)
            y1_bwd = self.dir_bwd(y1_bwd)

            y1 = y1_fwd + y1_bwd
            x = rms_norm(x + y1, variance_epsilon=self.norm_eps)

            y2_fwd = self.mamba_second(x)
            rev_x2 = torch.flip(x, dims=[1])
            y2_bwd = self.mamba_second(rev_x2)
            y2_bwd = torch.flip(y2_bwd, dims=[1])

            y2 = y2_fwd + y2_bwd
            hidden_states = rms_norm(x + y2, variance_epsilon=self.norm_eps)

        elif self.mode == "bimamba_v2_with_transformer":
            x = hidden_states.contiguous()

            # (1) BiMambaV2
            y = self.mamba(x)
            hidden_states = rms_norm(x + y, variance_epsilon=self.norm_eps)

            # (2) Transformer self-attention (same logic as original transformer block)
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # ---------------- simple shared Mamba + flip ----------------
        else:
            hidden_states = hidden_states.contiguous()
            y_fwd = self.mamba(hidden_states)
            reversed_input = torch.flip(hidden_states, dims=[1])
            y_bwd = self.mamba(reversed_input)
            y_bwd = torch.flip(y_bwd, dims=[1])
            y = 0.5 * (y_fwd + y_bwd)
            hidden_states = rms_norm(hidden_states + y, variance_epsilon=self.norm_eps)

        out = self.mlp(hidden_states)
        if self.dropout is not None:
            out = self.dropout(out)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)


        return hidden_states


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0
            else self.config.puzzle_emb_len
        )  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            pass

        # Reasoning Layers (L-level stack)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings (if learned)
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad (bootstrapping / recurrent rollouts)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Last cycle with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)  # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reset in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return (
            TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
