### **Requirements**

Install the necessary libraries using the following commands:

```bash
pip install transformers
pip install gpytorch
```

> **Note:** Ensure you have Python and pip installed before running the commands.

### **Run the Code**

Use the following command to execute the script with the specified hyperparameters:

```bash
python run.py \
    --max_length 256 \
    --dataname TACRED \
    --encoder_epochs 30 \
    --encoder_lr 2e-5 \
    --prompt_pool_epochs 25 \
    --prompt_pool_lr 1e-4 \
    --classifier_epochs 250 \
    --seed 2021 \
    --bert_path bert-base-uncased \
    --data_path datasets \
    --prompt_length 8 \
    --prompt_top_k 4 \
    --batch_size 16 \
    --prompt_pool_size 20 \
    --replay_s_e_e 100 \
    --replay_epochs 200 \
    --classifier_lr 5e-5 \
    --prompt_type only_prompt
```

### **Parameter Variations**

- **Seeds**: `2021`, `2121`, `2221`, `2321`, `2421`
- **Dataname Options**: `TACRED`, `FewRel`

> **Tip:** To run experiments with different seeds or datasets, modify the `--seed` and `--dataname` arguments accordingly. 
