import numpy as np
import torch

import logging
import wandb

from ..metrics.computation import compute_metrics

def validate(the_model, valid_data, batch_size, device):
    the_model.eval()
    
    batch_loss = 0
    
    eval_f1 = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []

    logging.info("validate: Validating model...")
    
    with torch.no_grad():
        for batch, data in enumerate(valid_data):
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

            logits = output["logits"].cpu().detach().numpy()
            packed_val = logits, lbl_student.cpu().detach().numpy()
            metrics = compute_metrics(packed_val)
            
            eval_f1.append(metrics["f1_score"])
            eval_accuracy.append(metrics["accuracy"])
            eval_precision.append(metrics["precision"])
            eval_recall.append(metrics["recall"])
            
            loss_model = output["loss"]
            batch_loss += loss_model
    
        eval_loss = batch_loss / batch_size        

        wandb.log({
            "eval/loss": eval_loss, 
            "eval/f1_score": np.average(eval_f1), 
            "eval/accuracy": np.average(eval_accuracy),
            "eval/precision": np.average(eval_precision),
            "eval/recall": np.average(eval_recall)
        })
    
    out_metrics = {
        "eval/loss": eval_loss, 
        "eval/f1_score": np.average(eval_f1), 
        "eval/accuracy": np.average(eval_accuracy),
        "eval/precision": np.average(eval_precision),
        "eval/recall": np.average(eval_recall)
    }

    logging.info(f"validate: loss: {eval_loss} f1_score: {np.average(eval_f1)} accuracy: {np.average(eval_accuracy)} precision: {np.average(eval_precision)} recall: {np.average(eval_recall)}")
    
    return eval_loss, out_metrics

def test(the_model, test_data, batch_size, device):
    the_model.eval()
    
    batch_loss = 0
    
    eval_f1 = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []

    logging.info("test: Testing model...")
    
    with torch.no_grad():
        for batch, data in enumerate(test_data):
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

            logits = output["logits"].cpu().detach().numpy()
            packed_val = logits, lbl_student.cpu().detach().numpy()
            metrics = compute_metrics(packed_val)
            
            eval_f1.append(metrics["f1_score"])
            eval_accuracy.append(metrics["accuracy"])
            eval_precision.append(metrics["precision"])
            eval_recall.append(metrics["recall"])
            
            loss_model = output["loss"]
            batch_loss += loss_model
    
        eval_loss = batch_loss / batch_size
        wandb.log({
            "test/loss": eval_loss, 
            "test/f1_score": np.average(eval_f1), 
            "test/accuracy": np.average(eval_accuracy),
            "test/precision": np.average(eval_precision),
            "test/recall": np.average(eval_recall)
        })
    
    out_metrics = {
        "test/loss": eval_loss, 
        "test/f1_score": np.average(eval_f1), 
        "test/accuracy": np.average(eval_accuracy),
        "test/precision": np.average(eval_precision),
        "test/recall": np.average(eval_recall)
    }

    logging.info(f"validate: loss: {eval_loss} f1_score: {np.average(eval_f1)} accuracy: {np.average(eval_accuracy)} precision: {np.average(eval_precision)} recall: {np.average(eval_recall)}")
    
    return eval_loss, out_metrics