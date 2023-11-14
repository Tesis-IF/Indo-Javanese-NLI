import wandb

from decouple import config

def configure_sweep(logger):
    logger.info("configure_sweep: Configuring wandb hyperparameters sweep...")

    wandb.login(key=config('API_KEY_WANDB_JALAL'))

    # method
    sweep_config = {
        'method': 'grid',
        'name': 'sweep-transferlearning-experiment-xlmr-distillation'
    }

    # hyperparameters
    parameters_dict = {
        'epochs': {'values': [6, 10]},
        'learning_rate': {'values': [2e-4, 3e-6]},
        'lambda_kld': {'values': [0.5, 0.015]}, # between 0.01-0.5
        'batch_size': {'values': [2, 16]}
    }

    # metrics
    metrics_goal = {
        'goal': 'maximize',
        'name': 'test/f1_score'
    }

    sweep_config['parameters'] = parameters_dict
    sweep_config['metric'] = metrics_goal

    sweep_id = wandb.sweep(sweep_config, project='javanese_nli')

    logger.info("configure_sweep: Done configuring wandb hyperparameters sweep.")

    return sweep_id, sweep_config