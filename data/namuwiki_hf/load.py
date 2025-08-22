from datasets import load_dataset

def load_namuwiki():
    dataset = load_dataset("heegyu/namuwiki-sentences", split="train[:2_500_000]")
    if "sentence" not in dataset.column_names:
        raise ValueError("Expected 'sentence' column not found")
    dataset = dataset.rename_column("sentence", "text").select_columns(["text"])
    return dataset
