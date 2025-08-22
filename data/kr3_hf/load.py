from datasets import load_dataset

def load_kr3():
    dataset = load_dataset("leey4n/KR3",split="train")

    if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])

    dataset = dataset.filter(lambda example: example['Rating'] != 2)
    dataset = dataset.rename_column("Review", "text")

    return dataset