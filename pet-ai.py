import torch
from transformers import BertTokenizerFast, BertForTokenClassification

print(torch.__version__)  # 如 2.0.1+cu117
print(torch.cuda.is_available())  # 输出True
print(torch.cuda.get_device_name(0))  # 输出GTX 1060

# # 加载模型和分词器
# # bert-base-chinese适合中文任务
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
# model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=15)
#
# label_list = [
#     "O",
#     "B-品种",
#     "I-品种",
#     "B-颜色",
#     "I-颜色",
#     "B-价格",
#     "I-价格",
#     "B-年龄",
#     "I-年龄",
#     "B-疫苗",
#     "I-疫苗",
#     "B-身长",
#     "I-身长",
#     "B-健康状况",
#     "I-健康状况",
# ]
# label2id = {label: idx for idx, label in enumerate(label_list)}
# id2label = {idx: label for label, idx in label2id.items()}
# model.config.label2id = label2id
# model.config.id2label = id2label
