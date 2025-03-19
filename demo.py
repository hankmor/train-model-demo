import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import json
from datasets import Dataset

# 加载模型
# model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=15)


def pretrain():

# 信息提取
def extract_info(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = [tokenizer.decode([token_id]) for token_id in predictions[0]]

    extracted_info = {}
    for token, entity in zip(tokens, entities):
        if entity != "O":
            extracted_info[token] = entity
    return extracted_info


# 数据整理
def format_info(extracted_info):
    formatted_data = {
        "品种": extracted_info.get("金毛犬", ""),
        "颜色": extracted_info.get("金色", ""),
        "年龄": extracted_info.get("3岁", ""),
        "价格": extracted_info.get("5000元", ""),
        "疫苗接种": extracted_info.get("已接种疫苗", ""),
        "健康状况": extracted_info.get("良好", ""),
        "体长": extracted_info.get("60cm", ""),
        "高度": extracted_info.get("50cm", ""),
    }
    return formatted_data


def organize_results():
    # 批量处理
    data = pd.read_json("data/pet.json")
    formatted_results = []
    for text in data["text"]:
        extracted_info = extract_info(text)
        formatted_data = format_info(extracted_info)
        formatted_results.append(formatted_data)

    # 保存结果
    with open("formatted_pet_data.json", "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)


# text = "这是一只金毛犬，颜色是金色，年龄3岁，价格5000元，已接种疫苗，健康状况良好，体长60cm，高度50cm。"
# extracted_info = extract_info(text)
# print(extracted_info)
organize_results()
