# download_models.py
from src.bert_runner import download_model_if_needed

if __name__ == "__main__":
    download_model_if_needed("roberta-base")
    print("All models ready.")