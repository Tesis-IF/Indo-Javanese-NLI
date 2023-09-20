# Indo-Javanese NLI

A Cross-lingual Natural Language Inference from Indonesian to Javanese.

## How to run code
1. On the terminal, navigate to the .py directory
2. Run this code:
```python "NLI Transfer Learning - XLMR-KLD.py" --epoch=6 --batch_size=2 --max_len=512 --std_lr=3e-6 --lambda_kld=0.5 --used_model=XLMR```
3. You may change the hyperparameter on the arguments above.
4. ```--used_model``` argument currently only support ```XLMR``` or ```mBERT```.

## How to use sweeper
1. On the terminal, navigate to the .py directory
2. Run this code:
```python "NLI Transfer Learning - Hyperparameter Sweeper.py" --max_len=512 --used_model=XLMR```
3. You may change the hyperparameter on the arguments above.
4. ```--used_model``` argument currently only support ```XLMR``` or ```mBERT```.