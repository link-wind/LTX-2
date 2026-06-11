import torch

from ltx_audio_core.loader.sd_ops import SDOps
from ltx_audio_core.model.model_protocol import ModelConfigurator
from ltx_audio_core.model.transformer.model import LTXModel, LTXModelType
from ltx_audio_core.model.transformer.rope import LTXRopeType
from ltx_audio_core.model.transformer.text_projection import create_caption_projection
from ltx_audio_core.model.transformer.transformer import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig
from ltx_audio_core.utils import check_config_value


class LTXModelConfigurator(ModelConfigurator[LTXModel]):
    """
    Build the audio-only transformer from a checkpoint config dict.
    """

    @classmethod
    def from_config(
        cls,
        config: dict,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
    ) -> LTXModel:
        audio_caption_projection = _build_audio_caption_projection(config)

        config = config.get("transformer", {})

        _check_config_value_if_present(config, "dropout", 0.0)
        _check_config_value_if_present(config, "attention_bias", True)
        _check_config_value_if_present(config, "num_vector_embeds", None)
        _check_config_value_if_present(config, "activation_fn", "gelu-approximate")
        _check_config_value_if_present(config, "num_embeds_ada_norm", 1000)
        _check_config_value_if_present(config, "use_linear_projection", False)
        _check_config_value_if_present(config, "only_cross_attention", False)
        _check_config_value_if_present(config, "cross_attention_norm", True)
        _check_config_value_if_present(config, "double_self_attention", False)
        _check_config_value_if_present(config, "upcast_attention", False)
        _check_config_value_if_present(config, "standardization_norm", "rms_norm")
        _check_config_value_if_present(config, "norm_elementwise_affine", False)
        _check_config_value_if_present(config, "qk_norm", "rms_norm")
        _check_config_value_if_present(config, "positional_embedding_type", "rope")
        _check_config_value_if_present(config, "use_middle_indices_grid", True)

        return LTXModel(
            model_type=LTXModelType.AudioOnly,
            num_layers=config.get("num_layers", 48),
            norm_eps=config.get("norm_eps", 1e-06),
            ops=ops,
            positional_embedding_theta=config.get("positional_embedding_theta", 10000.0),
            timestep_scale_multiplier=config.get("timestep_scale_multiplier", 1000),
            use_middle_indices_grid=config.get("use_middle_indices_grid", True),
            audio_num_attention_heads=config.get(
                "audio_num_attention_heads",
                config.get("num_attention_heads", 32),
            ),
            audio_attention_head_dim=config.get(
                "audio_attention_head_dim",
                config.get("attention_head_dim", 64),
            ),
            audio_in_channels=config.get(
                "audio_in_channels",
                config.get("in_channels", 128),
            ),
            audio_out_channels=config.get(
                "audio_out_channels",
                config.get("out_channels", 128),
            ),
            audio_cross_attention_dim=config.get(
                "audio_cross_attention_dim",
                config.get("cross_attention_dim", 2048),
            ),
            audio_positional_embedding_max_pos=config.get(
                "audio_positional_embedding_max_pos",
                config.get("positional_embedding_max_pos", [20]),
            ),
            rope_type=LTXRopeType(config.get("rope_type", "split")),
            double_precision_rope=config.get("frequencies_precision", False) == "float64",
            apply_gated_attention=config.get("apply_gated_attention", False),
            audio_caption_projection=audio_caption_projection,
            cross_attention_adaln=config.get("cross_attention_adaln", False),
        )


def _build_audio_caption_projection(config: dict) -> torch.nn.Module | None:
    transformer_config = config.get("transformer", {})
    if "caption_channels" not in transformer_config:
        return None
    if transformer_config.get("caption_proj_before_connector", False):
        return None

    with torch.device("meta"):
        return create_caption_projection(transformer_config, audio=True)


def _check_config_value_if_present(config: dict, key: str, expected: object) -> None:
    if key in config:
        check_config_value(config, key, expected)


LTX_AUDIO_MODEL_COMFY_RENAMING_MAP = (
    SDOps("LTX_AUDIO_MODEL_COMFY_RENAMING_MAP")
    .with_matching(prefix="model.diffusion_model.")
    .with_replacement("model.diffusion_model.", "")
)
