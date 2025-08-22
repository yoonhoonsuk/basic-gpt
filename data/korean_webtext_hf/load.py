from datasets import load_dataset

def load_korean_webtext():
    dataset = load_dataset("HAERAE-HUB/KOREAN-WEBTEXT",split="train")

    # Remove auto-index column if present
    if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])
    
    dataset = dataset.select_columns(["text"])
    return dataset
