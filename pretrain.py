from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
import pandas

# from sklearn.metrics import accuracy_score
import json

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-chinese", num_labels=15
)
# 定义标签
label_list = [
    "O",
    "B-品种",
    "I-品种",
    "B-颜色",
    "I-颜色",
    "B-价格",
    "I-价格",
    "B-年龄",
    "I-年龄",
    "B-疫苗",
    "I-疫苗",
    "B-身长",
    "I-身长",
    "B-健康状况",
    "I-健康状况",
]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
model.config.label2id = label2id
model.config.id2label = id2label

# 加载数据
with open("data/pet-marked.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转为Dataset格式
dataset = Dataset.from_list(data)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding=True, is_split_into_words=False
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 分词后的词与原词对齐
        aligned_labels = [
            -100 if word_id is None else label2id[label[word_id]]
            for word_id in word_ids
        ]  # -100表示忽略（CLS、SEP等）
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 处理数据集
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir="./pet_ner_results",  # 输出目录
    eval_strategy="epoch",  # 每轮评估
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=4,  # 批次大小（适应显存）
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # 训练3轮
    weight_decay=0.01,  # 权重衰减
    save_steps=500,  # 每500步保存
    save_total_limit=2,  # 最多保存2个检查点
    logging_dir="./logs",  # 日志目录
    logging_steps=10,
    fp16=True,  # 混合精度训练（加速，节省显存）
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


# 划分数据集（80%训练，20%验证）
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./pet_ner_model")
tokenizer.save_pretrained("./pet_ner_model")


def test_model():
    # 加载微调后的模型
    ner_pipeline = pipeline(
        "ner",
        model="./pet_ner_model",
        tokenizer="./pet_ner_model",
        aggregation_strategy="simple",
    )

    # 测试
    text = "这是一只黑色的波斯猫，价格1000元，1岁"
    results = ner_pipeline(text)
    for entity in results:
        print(
            f"实体: {entity['word']}, 类型: {entity['entity_group']}, 置信度: {entity['score']:.4f}"
        )

    organized = organize_results(results)
    df = pandas.DataFrame([organized])
    df.to_csv("organized_pet_data.csv", index=False)


def organize_results(results):
    pet_info = {
        "品种": None,
        "颜色": None,
        "价格": None,
        "年龄": None,
        "疫苗": None,
        "身长": None,
        "健康状况": None,
    }
    for entity in results:
        label = entity["entity_group"].split("-")[-1]  # 提取B-品种中的“品种”
        if label in pet_info:
            pet_info[label] = entity["word"]
    return pet_info
