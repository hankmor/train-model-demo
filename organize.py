from transformers import pipeline
import pandas as pd

ner_pipeline = pipeline(
    "ner",
    model="./pet_ner_model",
    tokenizer="./pet_ner_model",
    aggregation_strategy="simple",
)
text = "这是一只黑色的波斯猫，价格1000元，1岁"
results = ner_pipeline(text)


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
        label = entity["entity_group"].split("-")[-1]
        if label in pet_info:
            pet_info[label] = entity["word"]
    return pet_info


organized = organize_results(results)
df = pd.DataFrame([organized])
df.to_csv("pet_data.csv", index=False)
