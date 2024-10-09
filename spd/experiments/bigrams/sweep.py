import wandb

from spd.experiments.bigrams.train import train

wandb.require("core")
# Define the sweep configuration
sweep_config = {
    "method": "grid",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [0.001]},
        "batch_size": {"values": [500]},
        "dim": {"values": [50, 100, 200, 500]},
        "activation": {"values": [True, False]},
    },
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="train-bigrams")


# Define the training function
def training_run():
    run = wandb.init()
    config = run.config

    run_name = (
        f"act={config.activation}_"
        f"dim={config.dim}_"
        f"lr={config.learning_rate:.2e}_"
        f"bs={config.batch_size}"
    )
    wandb.run.name = run_name

    train(
        embedding_dim=config.dim,
        hidden_dim=config.dim,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        log=wandb.log,
        epochs=2000,
        activation=config.activation,
    )

    run.finish()


# Run the sweep
wandb.agent(sweep_id, function=training_run, count=100)
