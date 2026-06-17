from dataclasses import dataclass, field, replace

import torch

from ltx_audio_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_audio_core.model.transformer.attention import (
    Attention,
    AttentionCallable,
    AttentionFunction,
    AttentionOps,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from ltx_audio_core.model.transformer.feed_forward import FeedForward
from ltx_audio_core.model.transformer.ops import (
    AdaZeroCallable,
    GatedAttentionCallable,
    PostSACallable,
    PreAttentionCallable,
    PytorchAdaZeroFunction,
    PytorchGatedAttention,
    PytorchPostSAFunction,
    PytorchPreAttention,
)
from ltx_audio_core.model.transformer.rope import LTXRopeType
from ltx_audio_core.model.transformer.transformer_args import TransformerArgs
from ltx_audio_core.utils import rms_norm


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass(frozen=True)
class TransformerOpsConfig:
    attention_ops: AttentionOps = field(default_factory=AttentionOps)
    ada_zero_function: AdaZeroCallable = field(default_factory=PytorchAdaZeroFunction)
    post_sa_function: PostSACallable = field(default_factory=PytorchPostSAFunction)

    @classmethod
    def from_functions(
        cls,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.AUTOMATIC,
        masked_attention: MaskedAttentionFunction | MaskedAttentionCallable = MaskedAttentionFunction.AUTOMATIC,
        preattention: PreAttentionCallable | None = None,
        gated_attention: GatedAttentionCallable | None = None,
        ada_zero: AdaZeroCallable | None = None,
        post_sa: PostSACallable | None = None,
    ) -> "TransformerOpsConfig":
        attention_callable = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        masked_callable = (
            masked_attention.to_callable()
            if isinstance(masked_attention, MaskedAttentionFunction)
            else masked_attention
        )
        attention_ops = AttentionOps(
            attention_function=attention_callable,
            masked_attention_function=masked_callable,
            preattention_function=preattention if preattention is not None else PytorchPreAttention(),
            gated_attention_function=(gated_attention if gated_attention is not None else PytorchGatedAttention()),
        )
        return cls(
            attention_ops=attention_ops,
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        audio: TransformerConfig,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
    ):
        super().__init__()

        if ops is None:
            ops = TransformerOpsConfig()

        self.ada_zero_function = ops.ada_zero_function
        self.post_sa_function = ops.post_sa_function

        self.audio_attn1 = Attention(
            query_dim=audio.dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            context_dim=None,
            rope_type=rope_type,
            norm_eps=norm_eps,
            ops=ops.attention_ops,
            apply_gated_attention=audio.apply_gated_attention,
        )
        self.audio_attn2 = Attention(
            query_dim=audio.dim,
            context_dim=audio.context_dim,
            heads=audio.heads,
            dim_head=audio.d_head,
            rope_type=rope_type,
            norm_eps=norm_eps,
            ops=ops.attention_ops,
            apply_gated_attention=audio.apply_gated_attention,
        )
        self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)

        audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        self.cross_attention_adaln = audio.cross_attention_adaln
        if self.cross_attention_adaln:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(self, audio: TransformerArgs) -> TransformerArgs:
        ax = audio.x
        run_ax = audio.enabled and ax.numel() > 0

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(0, 3),
            )
            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa

            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )
            ax = ax + ax_msa_out * agate_msa
            del agate_msa, norm_ax, ax_msa_out

            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )

            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table,
                ax.shape[0],
                audio.timesteps,
                slice(3, 6),
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(audio, x=ax)


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
