import wandb
from games.liars_dice.dice_game_nODE import main

# Stop the sweep agents:
# wandb sweep --stop chemical_reaction_networks/MFG_ICML/rl2m1pr8


def wandb_run():
    wandb.login()
    parameters = {
            # a flat distribution between 0 and 0.1
            "lr": {'distribution': 'uniform', 'min': 0, 'max': 0.1},
            "width": {"values": [4, 8, 16, 32]},
            "depth": {"values": [3, 4]},
            "dataset_size": {"values": [50, 100, 500, 1000]},
            "optimizer": {"values": ["adabelief", "sgd"]}
    }

    sweep_config = {"method": "bayes",
                    "metric": {"goal": "minimize", "name": "validation_loss"},
                    "parameters": parameters}

    sweep_id = wandb.sweep(
        # Set the project where this run will be logged
        project="MFG_ICML",
        # Track hyperparameters and run metadata
        sweep=sweep_config,
    )
    wandb.agent(sweep_id, function=sweep, count=30)


def sweep(config=None):
    wandb.init(config=config, tags=["liars_dice_architecture_sweep"])
    main(steps=200, suppress=True, weights_and_biases=True, **wandb.config)
    wandb.finish()


if __name__ == '__main__':
    wandb_run()
