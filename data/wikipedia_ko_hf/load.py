from datasets import load_dataset

def load_wikipedia_ko():
    dataset = load_dataset("Cohere/wikipedia-22-12-ko-embeddings",split="train")

    if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])

    dataset = dataset.select_columns(["text"])

    return dataset