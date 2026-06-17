from enum import Enum

import torch

from ltx_audio_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_audio_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_audio_core.model.transformer.modality import Modality
from ltx_audio_core.model.transformer.rope import LTXRopeType
from ltx_audio_core.model.transformer.transformer import (
    DEFAULT_TRANSFORMER_OPS,
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from ltx_audio_core.model.transformer.transformer_args import (
    BlockPerturbationsProcessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_audio_core.utils import to_denoised


class LTXModelType(Enum):
    AudioOnly = "ltx audio only model"


class LTXModel(torch.nn.Module):
    """
    Audio-only LTX transformer implementation.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioOnly,
        num_layers: int = 48,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
        positional_embedding_theta: float = 10000.0,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()

        self._enable_gradient_checkpointing = False
        self.model_type = model_type
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta

        if audio_positional_embedding_max_pos is None:
            audio_positional_embedding_max_pos = [20]

        self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim

        self._init_audio(
            in_channels=audio_in_channels,
            out_channels=audio_out_channels,
            norm_eps=norm_eps,
            caption_projection=audio_caption_projection,
        )

        self._init_preprocessors()
        self._init_transformer_blocks(
            num_layers=num_layers,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
        )

        self.block_input_processor = BlockPerturbationsProcessor()

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_preprocessors(self) -> None:
        self.audio_args_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            use_middle_indices_grid=self.use_middle_indices_grid,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            caption_projection=getattr(self, "audio_caption_projection", None),
            prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
        )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops: TransformerOpsConfig,
        apply_gated_attention: bool,
    ) -> None:
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=audio_attention_head_dim,
            context_dim=audio_cross_attention_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=self.cross_attention_adaln,
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                )
                for _ in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        audio: TransformerArgs,
        perturbations: BatchedPerturbationConfig | None,
    ) -> TransformerArgs:
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(audio.x.shape[0])

        for block_idx, block in enumerate(self.transformer_blocks):
            audio = self.block_input_processor(
                audio,
                perturbations,
                block_idx,
                self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
            )

            if self._enable_gradient_checkpointing and self.training:
                audio = torch.utils.checkpoint.checkpoint(
                    block,
                    audio,
                    use_reentrant=False,
                )
            else:
                audio = block(audio=audio)

        return audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        audio: Modality,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> torch.Tensor:
        audio_args = self.audio_args_preprocessor.prepare(audio)
        audio_out = self._process_transformer_blocks(
            audio=audio_args,
            perturbations=perturbations,
        )

        ax = self._process_output(
            self.audio_scale_shift_table,
            self.audio_norm_out,
            self.audio_proj_out,
            audio_out.x,
            audio_out.embedded_timestep,
        )
        return ax


class LegacyX0Model(torch.nn.Module):
    """
    Audio-only legacy X0 wrapper.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        audio: Modality,
        perturbations: BatchedPerturbationConfig | None,
        sigma: float,
    ) -> torch.Tensor:
        ax = self.velocity_model(audio, perturbations)
        return to_denoised(audio.latent, ax, sigma)


class X0Model(torch.nn.Module):
    """
    Audio-only X0 wrapper.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        audio: Modality,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> torch.Tensor:
        ax = self.velocity_model(audio, perturbations)
        return to_denoised(audio.latent, ax, audio.timesteps)
