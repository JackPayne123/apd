"""Config classes of various types"""

from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.log import logger
from spd.types import ModelPath, Probability


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )
    pretrained_model_path: ModelPath  # e.g. wandb:spd-tms/runs/si0zbfxf


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_mlp"] = "residual_mlp"
    feature_probability: Probability
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    pretrained_model_path: ModelPath  # e.g. wandb:spd-resid-mlp/runs/j9kmavzi


class LMTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["lm"] = "lm"
    model_size: str  # e.g. "1.25M"
    max_seq_len: PositiveInt = 512
    buffer_size: PositiveInt = 1000
    dataset_name: str = "lennart-finke/SimpleStories"
    train_data_split: str = "train"
    eval_data_split: str = "test"
    n_eval_steps: PositiveInt = 100
    # List of fnmatch patterns for nn.Linear modules to decompose
    target_module_patterns: list[str] = ["transformer.h.*.mlp.*_proj"]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: PositiveInt | None = None
    image_on_first_step: bool = True
    save_freq: PositiveInt | None = None
    lr: PositiveFloat
    out_recon_coeff: NonNegativeFloat | None = None
    act_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    masked_recon_coeff: NonNegativeFloat | None = None
    random_mask_recon_coeff: NonNegativeFloat | None = None
    layerwise_recon_coeff: NonNegativeFloat | None = None
    layerwise_random_recon_coeff: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat
    schatten_coeff: NonNegativeFloat | None = None
    pnorm: PositiveFloat
    m: PositiveInt
    n_random_masks: PositiveInt
    init_from_target_model: bool = False
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    unit_norm_matrices: bool = False
    attribution_type: Literal["gradient"] = "gradient"
    n_gate_hidden_neurons: PositiveInt | None = None
    task_config: TMSTaskConfig | ResidualMLPTaskConfig | LMTaskConfig = Field(
        ..., discriminator="task_name"
    )
    embedding_recon_coeff: float | None = None
    is_embed_unembed_recon: bool = False

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = []
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {}

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]
            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Warn if neither masked_recon_coeff nor lp_sparsity_coeff is set
        if not self.masked_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither masked_recon_coeff nor lp_sparsity_coeff is set")

        # Give a warning if both out_recon_coeff and param_match_coeff are > 0
        if (
            self.param_match_coeff is not None
            and self.param_match_coeff > 0
            and self.out_recon_coeff is not None
            and self.out_recon_coeff > 0
        ):
            logger.warning(
                "Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one."
            )

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.masked_recon_coeff == 0:
            logger.warning(f"masked_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert self.lr_exponential_halflife is not None, (
                "lr_exponential_halflife must be set if lr_schedule is exponential"
            )
        # Schatten norm schould be null unless the model is an LM
        if self.task_config.task_name != "lm":
            assert self.schatten_coeff is None, (
                "schatten_coeff should be null unless the model is an LM"
            )
        return self
