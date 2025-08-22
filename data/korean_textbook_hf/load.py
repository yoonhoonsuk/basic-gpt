from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

def load_korean_textbook():
    dataset_name = "maywell/korean_textbooks"
    configs = get_dataset_config_names(dataset_name)

    all_splits = []
    for cfg in configs:
        ds = load_dataset(dataset_name, cfg, split="train")

        if "text" in ds.column_names:
            ds = ds.select_columns(["text"])
        
        all_splits.append(ds)

    dataset = concatenate_datasets(all_splits)
    return dataset
