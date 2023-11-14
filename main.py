import torch
from transformers import XLMRobertaTokenizer, BertTokenizer
from decouple import config
import sys, getopt

from libinstaller import installer
from dataset import PrepareDataset, LoadDataset, CreateDataLoader, CompDataset
from logger.logger import init_logger
from sequence.seq import training_sequence, testing_sequence
from sweep.configuration import configure_sweep
from sweep.sweep_sequence import start_sweeping_seq

def main(argv):
    opts, args = getopt.getopt(argv,"h:e:k:b:l:lr:u:t:y:tc:", ["help", "epoch=", "lambda_kld=", "batch_size=", "max_len=", "learning_rate=", "max_length=", "used_model=", "task=", "type=", "teacher="])

    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCH = 6
    STUDENT_LRATE = 3e-6
    BATCH_NORM_EPSILON = 1e-5
    USED_MODEL = "XLMR"
    TASK = "train"
    MOD_TYPE = "kld"
    TEACHER_MDL_NAME = "jalaluddin94/nli_mbert"
    LAMBDA_KLD = 0.5
    the_tokenizer = None

    logger = init_logger()

    if len(opts) == 0:
        logger.error("main: You must provide at least one argument.")
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            filename = str(__file__)
            if " " in filename:
                filename = '"' + filename[0:len(filename)] + '"'
            
            logger.error(f"""To run experiment for IndoJavaneseNLI, please run the following command:
            python {filename} --max_len=MAX_LENGTH --used_model=MBERT/XLMR --task=TASK""")
            sys.exit()
        elif opt in ("-b", "--batch_size"):
            BATCH_SIZE = int(arg)
        elif opt in ("-e", "--epoch"):
            EPOCH = int(arg)
        elif opt in ("-k", "--lambda_kld"):
            LAMBDA_KLD = float(arg)
        elif opt in ("-t", "--task"):
            TASK = str(arg)
        elif opt in ("-m", "--max_len", "max_length"):
            MAX_LEN = int(arg)
        elif opt in ("-lr", "--learning_rate"):
            STUDENT_LRATE = float(str(arg))
        elif opt in ("-u", "--used_model"):
            USED_MODEL = str(arg)
        elif opt in ("-y", "--type", "--model_type"):
            MOD_TYPE = str(arg)
        elif opt in ("-tc", "--teacher"):
            TEACHER_MDL_NAME = str(arg)
        else:
            logger.error(f"Unknown option {str(arg)}")
            sys.exit()

        if LAMBDA_KLD == 0 or LAMBDA_KLD > 0.5:
            logger.error("main: LAMBDA_KLD cannot be zero or greater than 0.5.")
            sys.exit()
        
        if USED_MODEL.lower() not in ("xlmr", "mbert"):
            logger.error(f'main: {USED_MODEL} not recognized. Please enter "XLMR" or "MBERT" as model used.')
            sys.exit()
        if TASK.lower() not in ("train", "training", "sweep", "gridsearch"):
            logger.error(f'main: {TASK} not recognized. Please enter "TRAIN" or "SWEEP" as the task.')
            sys.exit()
        elif TASK.lower() in ("train", "training"):
            TASK = "train"
        elif TASK.lower() in ("sweep", "gridsearch"):
            TASK = "sweep"

        if MAX_LEN <= 0 or MAX_LEN == None:
            logger.error("main: MAX_LEN must be a positive number and greater than 0.")
            sys.exit()
        if BATCH_SIZE <= 0 or BATCH_SIZE == None:
            logger.error("main: BATCH_SIZE must be a positive number and greater than 0.")
            sys.exit()
        if MOD_TYPE == None or MOD_TYPE =="" or MOD_TYPE not in ("kld", "kld_wo_softmax"):
            logger.error("main: MOD_TYPE must be fill in with 'kld' or 'kld_wo_softmax'.")
            sys.exit()

    # Check installed packages
    installer(logger)

    PrepareDataset(logger)
    df_train_t, df_train_student, df_valid_t, df_valid_student, df_test_t, df_test_student = LoadDataset(logger)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"main: Using device: {device}")

    if USED_MODEL.lower() == "xlmr":
        TOKENIZER_TYPE = 'xlm-roberta-base'
        MBERT_TYPE = 'xlm-roberta-base'
        the_tokenizer = XLMRobertaTokenizer.from_pretrained(TOKENIZER_TYPE)
    elif USED_MODEL.lower() == "mbert":
        TOKENIZER_TYPE = 'bert-base-multilingual-cased'
        MBERT_TYPE = 'bert-base-multilingual-cased'
        the_tokenizer = BertTokenizer.from_pretrained(TOKENIZER_TYPE)

    model_name = USED_MODEL.lower() + "-" + MOD_TYPE.lower()

    train_data_cmp = CompDataset(df_train_t, df_train_student, MAX_LEN, the_tokenizer)
    valid_data_cmp = CompDataset(df_valid_t, df_valid_student, MAX_LEN, the_tokenizer)
    test_data_cmp = CompDataset(df_test_t, df_test_student, MAX_LEN, the_tokenizer)

    if TASK.lower() == "train":
        train_dataloader = CreateDataLoader(train_data_cmp, BATCH_SIZE)
        valid_dataloader = CreateDataLoader(valid_data_cmp, BATCH_SIZE)
        test_dataloader = CreateDataLoader(test_data_cmp, BATCH_SIZE)

        logger.info(f"main: EXPERIMENT - {USED_MODEL.upper()}-KLD - Epoch {EPOCH} Learning Rate Student {STUDENT_LRATE}, Batch size {BATCH_SIZE}, Lambda KLD: {LAMBDA_KLD}")

    elif TASK.lower() == "sweep":
        sw_id, sw_conf = configure_sweep(logger)
        start_sweeping_seq(
            the_tokenizer,
            train_data_cmp,
            valid_data_cmp,
            test_data_cmp,
            model_name, # "mbert-kld", "xlmr-kld", "mbert-kld_wo_softmax", "xlmr-kld_wo_softmax"
            TEACHER_MDL_NAME,
            MBERT_TYPE,
            None,
            config("HF_TOKEN_JALAL"),
            logger,
            device,
            sw_conf
        )
        
if __name__ == "__main__":
    main(sys.argv[1:])