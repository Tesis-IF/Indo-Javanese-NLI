import logging
from decouple import config
import sys

from transformers import PretrainedConfig

from .kld import KnowledgeDistillation

def init_model(
        the_tokenizer,
        device: str = "",
        model_name: str = "mbert-kld",
        lambda_kld: float = 0.5, # from 0.01 - 0.5
        learning_rate: float = 3e-6,
        teacher_model_type: str = "",
        student_model_type: str = "",
        huggingface_model_name: str = ""
    ):
    logging.info("init_model: Initializing model...")

    if(the_tokenizer == None):
        logging.error("init_model: Error when initializing model: Tokenizer must be initialized.")
        sys.exit()
    if(device == None or device == ""):
        logging.error("init_model: Error when initializing model: Device must be initialized.")
        sys.exit()

    model_type = model_name.split("-")[0]

    config_pretrained_model = PretrainedConfig(
        problem_type = "single_label_classification",
        id2label = {
            "0": "ENTAIL",
            "1": "NEUTRAL",
            "2": "CONTRADICTION"
        },
        label2id = {
            "ENTAIL": 0,
            "NEUTRAL": 1,
            "CONTRADICTION": 2
        },
        num_labels = 3,
        hidden_size = 768,
        name_or_path = f"indojavanesenli-transfer-learning-{model_name.lower()}",
        finetuning_task = "indonesian-javanese natural language inference"
    )

    if(model_name[-3:].lower() == "kld"):
        model = KnowledgeDistillation(
            configs = config_pretrained_model,
            lambda_kld = lambda_kld, 
            learningrate_student = learning_rate,
            tokenizer = the_tokenizer,
            mod_teacher_type = teacher_model_type, 
            batchnorm_epsilon = 1e-5, 
            mod_type = model_type, 
            mod_student_type = student_model_type,
            mod_name_for_hf = huggingface_model_name
        )
        model = model.to(device)
    
    logging.info("init_model: Model initialized.")
    return model