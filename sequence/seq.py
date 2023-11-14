from tqdm import tqdm

import logging
import wandb

from huggingface_hub import login, logout

from .train import train
from .test import validate, test

def training_sequence(the_model, 
                      train_data, 
                      valid_data, 
                      epochs, 
                      device: str,
                      batch_size,
                      save_model_path: str,
                      huggingface_token: str = None,
                      save_model=True, 
                      upload_model=True):
    track_train_loss = []
    track_val_loss = []

    logging.info("training_sequence: Starting training sequence...")
    
    pbar_format = "{l_bar}{bar} | Epoch: {n:.2f}/{total_fmt} [{elapsed}<{remaining}]"
    with tqdm(total=epochs, colour="blue", leave=True, position=0, bar_format=pbar_format) as t:
        for ep in range(epochs):
            training_loss = train(the_model, train_data, t, batch_size, device)
            t.set_description(f"Evaluating... Train loss: {training_loss:.3f}")
            valid_loss, _ = validate(the_model, valid_data, batch_size, device)

            track_train_loss.append(training_loss)
            track_val_loss.append(valid_loss)

            t.set_description(f"Train loss: {training_loss:.3f} Valid loss: {valid_loss:.3f}")

            if valid_loss < min(track_val_loss) or ep + 1 == 1:
                if save_model:
                    the_model.save_model(save_model_path)

            wandb.log({
                "train_loss/epoch": training_loss,
                "validation_loss/epoch": valid_loss
            })

            logging.info(f"training_sequence: epoch: {ep + 1} training loss: {training_loss} validation loss: {valid_loss}")
        
        if upload_model:
            try:
                login(token=huggingface_token)
                the_model.upload_to_huggingface()
                logout()
            except Exception as e:
                logging.error(f"training_sequence: Error cannot upload model to Huggingface: {e}")

    logging.info("training_sequence: Done training the model.")
        
    return {
        "training_loss": track_train_loss,
        "validation_loss": track_val_loss
    }

def testing_sequence(
        the_model,
        testing_data,
        batch_size,
        device: str
):
    track_test_loss = []

    logging.info("testing_sequence: Starting testing sequence...")

    test_loss, _ = test(the_model, testing_data, batch_size, device)

    track_test_loss.append(test_loss)

    logging.info(f"testing_sequence: test loss: {test_loss}")

    logging.info("testing_sequence: Done testing sequence.")

    wandb.log({
        "test_loss/epoch": test_loss
    })

    return {
        "testing_loss": track_test_loss
    }