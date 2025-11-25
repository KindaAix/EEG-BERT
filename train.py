from transformers import Trainer, TrainingArguments
from models.bert_arch import BERT_arch as bert
from torch.optim import AdamW

model = bert()

training_args = TrainingArguments(
    output_dir="./results",

    learning_rate=2e-5,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    weight_decay_rate=0.01,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    eval_steps=100,
    logging_steps=50,
    save_steps=500,
    save_total_limit=5,
    max_steps=10000,
    num_train_epochs=10,
    logging_dir="./logs",

    eval_strategy="steps",
    eval_accumulation_steps=5,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_first_step=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=None,  # Replace with your training dataset
    eval_dataset=None,    # Replace with your evaluation dataset
    compute_loss_func=None,  # Replace with your custom loss function if needed
    compute_metrics=None,  # Replace with your custom metrics function if needed
    optimizers=(AdamW(model.parameters(), lr=2e-5, weight_decay=0.01), None)
)





