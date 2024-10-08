import torch
from torch.utils.data import DataLoader

from spd.experiments.bigrams.model import BigramDataset, BigramModel
from spd.func_spd import optimize
from spd.run_spd import Config, MinimalTaskConfig

# Parameters
A_vocab_size = 100  # A ranges from 0 to 99
B_vocab_size = 5  # B ranges from 0 to 4
embedding_dim = 20
hidden_dim = 50
batch_size = 1024

dataset = BigramDataset(A_vocab_size, B_vocab_size)
new_model = BigramModel(dataset.n_A, dataset.n_B, embedding_dim, hidden_dim)
new_model.load_state_dict(torch.load("bigram_model.pt"))
# Evaluate model
batch_size = 12
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


minimal_config = Config(
    wandb_project="spd-bigrams",
    wandb_run_name="decompose",
    wandb_run_name_prefix="bigrams",
    full_rank=True,
    seed=0,
    topk=1,
    batch_topk=False,
    steps=1000,
    print_freq=100,
    lr=0.005,
    task_config=MinimalTaskConfig(k=B_vocab_size),
    batch_size=batch_size,
    topk_param_attrib_coeff=0.0,
    orthog_coeff=0.0,
    out_recon_coeff=None,
    param_match_coeff=0.0,
    topk_recon_coeff=0.0,
    topk_l2_coeff=0.0,
    lp_sparsity_coeff=None,
    pnorm=None,
    pnorm_end=None,
    lr_schedule="constant",
    sparsity_loss_type="jacobian",
    sparsity_warmup_pct=0.0,
    unit_norm_matrices=False,
    ablation_attributions=False,
    initialize_spd="xavier",
)


# model_config = ConfigDict(extra="forbid", frozen=True)
# wandb_project: str | None = None
# wandb_run_name: str | None = None
# wandb_run_name_prefix: str = ""
# full_rank: bool = False
# seed: int = 0
# topk: PositiveFloat | None = None
# batch_topk: bool = True
# batch_size: PositiveInt
# steps: PositiveInt
# print_freq: PositiveInt
# image_freq: PositiveInt | None = None
# slow_images: bool = False
# save_freq: PositiveInt | None = None
# lr: PositiveFloat
# topk_param_attrib_coeff: NonNegativeFloat | None = None
# orthog_coeff: NonNegativeFloat | None = None
# out_recon_coeff: NonNegativeFloat | None = None
# param_match_coeff: NonNegativeFloat | None = 1.0
# topk_recon_coeff: NonNegativeFloat | None = None
# topk_l2_coeff: NonNegativeFloat | None = None
# lp_sparsity_coeff: NonNegativeFloat | None = None
# pnorm: PositiveFloat | None = None
# pnorm_end: PositiveFloat | None = None
# lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
# lr_exponential_halflife: PositiveFloat | None = None
# lr_warmup_pct: Probability = 0.0
# sparsity_loss_type: Literal["jacobian"] = "jacobian"
# sparsity_warmup_pct: Probability = 0.0
# unit_norm_matrices: bool = True
# ablation_attributions: bool = False
# initialize_spd: Literal["xavier", "oldSPD", "fullcopies"] = "xavier"
# task_config: DeepLinearConfig | PiecewiseConfig | TMSConfig | MinimalTaskConfig = Field(
#     ..., discriminator="task_name"
# )

optimize(
    model=None,
    config=minimal_config,
    device="cpu",
    dataloader=dataloader,
    pretrained_model=new_model,
)
