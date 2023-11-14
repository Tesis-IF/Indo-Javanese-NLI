import subprocess
import os
import pandas as pd
from decouple import config

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def check_dataset(dataset_type: str = "train"):
    dataset_filename = "datasets/" + dataset_type.lower() + ".csv"
    if(os.path.exists(dataset_filename)):
        return True
    else:
        return False

def PrepareDataset(
    logger,
    uri_repo = None
):
    # Preparing dataset
    logger.info("PrepareDataset: Preparing dataset folder...")
    if not os.path.exists("datasets/"):
        os.makedirs("datasets/")
    
    repo_url = "https://github.com/jalalAzhmatkhan/indojavanese-nli.git" if uri_repo == None else uri_repo

    # Download dataset
    logger.info("PrepareDataset: Downloading training data...")
    try:
        subprocess.run('git', 'clone', repo_url, "datasets/")
    except Exception as e:
        logger.error("PrepareDataset: Error while cloning dataset.")
        logger.error(str(e))

def LoadDataset(
    logger
):
    logger.info("LoadDataset: Loading train dataset...")
    df_train = pd.read_csv("datasets/indojavanesenli-train.csv", sep='\t')
    df_train = df_train.sample(frac=1).reset_index(drop=True) #shuffle the data

    df_train_student = pd.DataFrame()
    df_train_student["premise"] = df_train["premise"]
    df_train_student["hypothesis"] = df_train["jv_hypothesis"]
    df_train_student["label"] = df_train["label"]
    df_train_student.head()

    logger.info("LoadDataset: Loading validation dataset...")
    df_valid = pd.read_csv("datasets/indojavanesenli-valid.csv", sep='\t')
    df_valid = df_valid.sample(frac=1).reset_index(drop=True) #shuffle the data

    df_valid_student = pd.DataFrame()
    df_valid_student["premise"] = df_valid["premise"]
    df_valid_student["hypothesis"] = df_valid["jv_hypothesis"]
    df_valid_student["label"] = df_valid["label"]
    df_valid_student.head()

    logger.info("LoadDataset: Loading testing dataset...")
    df_test = pd.read_csv("datasets/indojavanesenli-test.csv", sep='\t')
    df_test = df_test.sample(frac=1).reset_index(drop=True) #shuffle the data

    df_test_student = pd.DataFrame()
    df_test_student["premise"] = df_test["premise"]
    df_test_student["premise"] = df_test_student["premise"].astype(str)
    df_test_student["hypothesis"] = df_test["jv_hypothesis"]
    df_test_student["hypothesis"] = df_test_student["hypothesis"].astype(str)
    df_test_student["label"] = df_test["label"]
    df_test_student.head()


    # Prepare Dataset for Teacher
    # Dataset from teacher will be from "IndoNLI", and using Indonesian only.
    logger.info("LoadDataset: Preparing Dataset for teacher...")

    df_train_t = pd.DataFrame()
    df_train_t["premise"] = df_train["premise"]
    df_train_t["hypothesis"] = df_train["hypothesis"]
    df_train_t["label"] = df_train["label"]
    df_train_t = df_train_t.sample(frac=1).reset_index(drop=True)

    logger.info("LoadDataset: Count per class train:") 
    logger.info(df_train_t['label'].value_counts())


    df_valid_t = pd.DataFrame()
    df_valid_t["premise"] = df_valid["premise"]
    df_valid_t["hypothesis"] = df_valid["hypothesis"]
    df_valid_t["label"] = df_valid["label"]
    df_valid_t = df_valid_t.sample(frac=1).reset_index(drop=True)


    logger.info("LoadDataset: Count per class valid:") 
    logger.info(df_valid_t['label'].value_counts())


    df_test_t = pd.DataFrame()
    df_test_t["premise"] = df_test["premise"]
    df_test_t["hypothesis"] = df_test["hypothesis"]
    df_test_t["label"] = df_test["label"]
    df_test_t = df_test_t.sample(frac=1).reset_index(drop=True)


    logger.info("LoadDataset: Count per class test:") 
    logger.info(df_test_t['label'].value_counts())

    return df_train_t, df_train_student, df_valid_t, df_valid_student, df_test_t, df_test_student

def CreateDataLoader(dataframe, logger, batch_sz: int = 1):
    try:
        data = DataLoader(dataframe, batch_size=batch_sz)

        return data
    except Exception as e:
        logger.error("CreateDataLoader: Exception during creating PyTorch DataLoader:" + e.message)

class CompDataset(Dataset):
    def __init__(self, df_teacher, df_student, max_len, tokenizer):
        self.df_data_teacher = df_teacher
        self.df_data_student = df_student
        self.tokenizer = tokenizer
        self.max_length = max_len
        
    def __getitem__(self, index):
        # Teacher
        sentence_teacher_1 = self.df_data_teacher.loc[index, 'premise']
        sentence_teacher_2 = self.df_data_teacher.loc[index, 'hypothesis']
        
        encoded_dict_teacher = self.tokenizer.encode_plus(
            sentence_teacher_1,
            sentence_teacher_2,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation='longest_first',
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        padded_token_list_teacher = encoded_dict_teacher['input_ids'][0]
        att_mask_teacher = encoded_dict_teacher['attention_mask'][0]
        
        target_teacher = torch.tensor([self.df_data_teacher.loc[index, 'label']])
        lt_target_teacher = torch.LongTensor(target_teacher)
        onehot_encoded_lbl_teacher = F.one_hot(lt_target_teacher, num_classes=3) # 3 classes: entails, neutral, contradict
        
        # Student
        sentence_student_1 = self.df_data_student.loc[index, 'premise']
        sentence_student_2 = self.df_data_student.loc[index, 'hypothesis']
        
        encoded_dict_student = self.tokenizer.encode_plus(
            sentence_student_1,
            sentence_student_2,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation='longest_first',
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        padded_token_list_student = encoded_dict_student['input_ids'][0]
        att_mask_student = encoded_dict_student['attention_mask'][0]
        
        target_student = torch.tensor([self.df_data_student.loc[index, 'label']])
        lt_target_student = torch.LongTensor(target_student)
        onehot_encoded_lbl_student = F.one_hot(lt_target_student, num_classes=3) # 3 classes: entails, neutral, contradict
        
        output = {
            "input_ids_teacher": padded_token_list_teacher, 
            "attention_mask_teacher": att_mask_teacher,
            "lbl_teacher": onehot_encoded_lbl_teacher,
            "input_ids_student": padded_token_list_student, 
            "attention_mask_student": att_mask_student,
            "lbl_student": onehot_encoded_lbl_student
        }
        
        return output
    
    def __len__(self):
        return len(self.df_data_teacher)