import logging
import wandb

def train(the_model, train_data, pgb, batch_size, device):
    the_model.train()

    batch_loss = 0
    logging.info("train: Start training model...")
    
    for batch, data in enumerate(train_data):
        # Clear accumulated gradients
        the_model.clear_grad()
        
        input_ids_teacher = data["input_ids_teacher"].to(device)
        attention_mask_teacher = data["attention_mask_teacher"].to(device)
        lbl_teacher = data["lbl_teacher"].to(device)
        input_ids_student = data["input_ids_student"].to(device)
        attention_mask_student = data["attention_mask_student"].to(device)
        lbl_student = data["lbl_student"].to(device)
        
        output = the_model(
            input_ids_teacher = input_ids_teacher, 
            attention_mask_teacher = attention_mask_teacher,
            lbl_teacher = lbl_teacher,
            input_ids_student = input_ids_student, 
            attention_mask_student = attention_mask_student, 
            lbl_student = lbl_student
        )
        
        loss_model = output["loss"]
        batch_loss += loss_model
        
        # Backpropagation
        # the_model.update_param_student_model(loss_model) # uncomment to use ordinary backpro
        ## now using gradient accumulation technique
        the_model.backpro_compute(loss_model) # backward pass and gradient accumulation
        
        # Accumulate gradients for the desired number of mini-batches
        if(batch+1) % batch_size == 0:
            # update weights
            the_model.update_std_weights_and_clear_grad()
        
        pgb.update(1 / len(train_data))
    
    # Make sure to update the weights for any remaining accumulated gradients
    if (batch+1) % batch_size != 0:
        the_model.update_std_weights()
        
    training_loss = batch_loss / batch_size

    logging.info(f"train: Training loss: {training_loss}")
    wandb.log({"train/loss": training_loss})
    
    return training_loss