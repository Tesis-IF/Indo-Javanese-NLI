import os

import torch
import torch.nn as nn

from transformers import AdamW
from transformers import PreTrainedModel
from transformers import XLMRobertaModel, BertModel

class KnowledgeDistillation(PreTrainedModel):
    def __init__(
            self, 
            configs, 
            lambda_kld, 
            learningrate_student, 
            tokenizer, 
            mod_teacher_type, 
            batchnorm_epsilon = 1e-5, 
            mod_type="mbert", 
            mod_student_type="bert-base-multilingual-cased",
            mod_name_for_hf = ""
            ):
        super(KnowledgeDistillation, self).__init__(configs)

        self.tokenizer = tokenizer

        self.mod_name_for_hf = mod_name_for_hf
        
        self.xlmr_model_teacher = XLMRobertaModel.from_pretrained(
            mod_teacher_type, # using pretrained mBERT in INA language
            num_labels = 3,
            output_hidden_states=True
        )
        
        # Freeze teacher mBERT parameters
        for params_teacher in self.xlmr_model_teacher.parameters():
            params_teacher.requires_grad = False

        self.xlmr_model_student = XLMRobertaModel.from_pretrained(
            mod_student_type,
            num_labels = 3,
            output_hidden_states=True
        )
    
        if mod_type.lower() == "xlmr":
            self.xlmr_model_student = XLMRobertaModel.from_pretrained(
                mod_student_type,
                num_labels = 3,
                output_hidden_states=True
            )
        elif mod_type.lower() == "mbert":
            self.xlmr_model_student = BertModel.from_pretrained(
                mod_student_type,
                num_labels = 3,
                output_hidden_states=True
            )
        
        # Unfreeze student mBERT parameters
        for params_student in self.xlmr_model_student.parameters():
            params_student.requires_grad = True
        
        self.optimizer_student = AdamW(
            self.xlmr_model_student.parameters(), 
            lr=learningrate_student
        )
        
        self.linear = nn.Linear(configs.hidden_size, 3)  # Linear layer
        self.batchnorm = nn.BatchNorm1d(configs.hidden_size, eps=batchnorm_epsilon)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kld = nn.KLDivLoss(reduction='batchmean')
        
        # Initialize the weights of the linear layer
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        
        self.lambda_kld = lambda_kld
    
    def forward(self, input_ids_teacher, attention_mask_teacher, lbl_teacher, input_ids_student, attention_mask_student, lbl_student):
        # the label is already one-hot encoded 
        self.xlmr_model_teacher.eval()
        self.xlmr_model_student.eval()
        
        lbl_teacher = lbl_teacher[:, 0, :]
        lbl_student = lbl_student[:, 0, :]
        
        with torch.no_grad():
            # Taking CLS token out of XLMR last hidden state
            outputs_teacher = self.xlmr_model_teacher(
                input_ids=input_ids_teacher, 
                attention_mask=attention_mask_teacher
            )
        
            # take CLS token of the last hidden state
            pooled_output_teacher = outputs_teacher.last_hidden_state[:, 0, :]
        
        # taking CLS token out of the student data without deleting the gradient
        outputs_student = self.xlmr_model_student(
            input_ids=input_ids_student, 
            attention_mask=attention_mask_student 
        )
        
        pooled_output_student = outputs_student.last_hidden_state[:, 0, :]
        
        # FFNN
        batchnormed_logits = self.batchnorm(pooled_output_student)
        linear_output = self.linear(batchnormed_logits) # the output's logits
        softmax_linear_output = F.log_softmax(linear_output, dim=1)
        
        lbl_student = lbl_student.float()
        softmax_linear_output = softmax_linear_output.float()
        
        # Loss Computation
        cross_entropy_loss = self.cross_entropy(softmax_linear_output, lbl_student)
        total_kld = self.kld(F.log_softmax(pooled_output_student, dim=1), F.softmax(pooled_output_teacher, dim=1))
        joint_loss = cross_entropy_loss + (self.lambda_kld * total_kld )
        
        return {"loss": joint_loss, "logits": softmax_linear_output}
    
    def clear_grad(self):
        self.xlmr_model_student.train()
        self.optimizer_student.zero_grad()
    
    def backpro_compute(self, loss):
        loss.backward()
        
    def update_std_weights_and_clear_grad(self):
        self.optimizer_student.step()
        self.optimizer_student.zero_grad()
    
    def update_std_weights(self):
        self.optimizer_student.step()
    
    def update_param_student_model(self, loss):
        # Doing customized backpropagation for student's model
        self.xlmr_model_student.train()
        
        self.optimizer_student.zero_grad()
        loss.backward()
        self.optimizer_student.step()

    def save_model(self, model_name):
        if len(model_name) > 0:
            # save
            cur_dir = os.getcwd() if os.getcwd()[:-1] == '/' else os.getcwd() + '/'
            cur_dir = cur_dir + model_name if model_name[0] != '/' else cur_dir + model_name[1:]
            print(f"Saving model to {cur_dir}")
            self.xlmr_model_student.save_pretrained(save_directory=cur_dir)
        
    def upload_to_huggingface(self):
        self.xlmr_model_student.push_to_hub(self.mod_name_for_hf)
        self.tokenizer.push_to_hub(self.mod_name_for_hf)