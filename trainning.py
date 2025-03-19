from transformers import TrainingArguments, Trainer
import numpy as np

training_args = TrainingArguments(
    output_dir="./pet_ner_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # GTX 1060显存有限
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,  # 混合精度加速
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    from seqeval.metrics import precision_score, recall_score, f1_score

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("./pet_ner_model")
tokenizer.save_pretrained("./pet_ner_model")
