import wandb

from ..models.model import init_model
from ..dataset import CreateDataLoader
from ..sequence.seq import training_sequence, testing_sequence

def start_sweeping_seq(
        tok,
        train_cmp,
        valid_cmp,
        test_cmp,
        model_name,
        teacher_name,
        student_name,
        hf_upload_name,
        hf_api_token,
        logger,
        device,
        configs=None
    ):

    logger.info("start_sweeping_seq: Starting hyperparameters sweep sequence...")

    with wandb.init(
        config=configs,
        project="javanese_nli",
        tags=["transferlearning", "bandyopadhyay"]
    ):
        # set sweep configuration
        configs = wandb.config #if configs is None else configs

        train_dataloader = CreateDataLoader(train_cmp, logger, configs.batch_size)
        valid_dataloader = CreateDataLoader(valid_cmp, logger, configs.batch_size)
        test_dataloader = CreateDataLoader(test_cmp, logger, configs.batch_size)
        
        ret_model = init_model(
            tok,
            device,
            model_name,
            configs.lambda_kld,
            configs.learning_rate,
            teacher_name,
            student_name,
            hf_upload_name
        )

        training_result = training_sequence(
            ret_model, 
            train_dataloader, 
            valid_dataloader, 
            configs.epochs,
            device, 
            configs.batch_size,
            save_model_path="",
            huggingface_token=hf_api_token,
            save_model = False,
            upload_model = True
            )
        
        testing_result = testing_sequence(
            ret_model, 
            test_dataloader,
            configs.batch_size,
            device
        )