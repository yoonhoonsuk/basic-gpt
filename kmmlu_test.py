# import torch
# from preprocess.tokenizer_bbpe import get_tokenizer
# from model.decoder_transformer import GPT
# from util.file_utils import parse_filename
# from torch.nn.functional import softmax
# from tqdm import tqdm

# def load_model(model_path, vocab_size, device, max_len):
#     config = parse_filename(model_path)
#     state_dict = torch.load(model_path, map_location=device)
#     if "model" in state_dict:
#         state_dict = state_dict["model"]
#     model = GPT(
#         vocab_size=vocab_size,
#         d_model=config["d_model"],
#         d_ffn=config["d_ffn"],
#         n_layers=config["n_layers"],
#         max_len=max_len
#     ).to(device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def main(pt_path, model_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = get_tokenizer()
#     data = torch.load(pt_path, map_location="cpu")
#     x = data["x"]
#     y = data["y"]
#     vocab_size = data["vocab_size"]

#     answer_token_ids = [tokenizer.encode(str(i))[0] for i in range(1, 5)]
#     model = load_model(model_path, vocab_size, device, x.shape[1])

#     correct = 0
#     total = 0

#     for i in tqdm(range(len(x)), desc="Evaluating"):
#         input_ids = x[i:i+1].to(device)
#         with torch.no_grad():
#             logits = model(input_ids)
#             next_logits = logits[:, -1, :] # [B, vocab]
#             answer_logits = next_logits[:, answer_token_ids].squeeze(0) # [4]
#             probs = softmax(answer_logits, dim=-1).cpu()
#             pred_idx = probs.argmax().item()
#             pred_token = answer_token_ids[pred_idx]

#         # Ground truth: last non -100 value in y[i]
#         gt_ids = [t.item() for t in y[i] if t != -100]
#         if not gt_ids:
#             continue  # skip
#         gt_token = gt_ids[0]

#         if pred_token == gt_token:
#             correct += 1
#         total += 1

#     print(f"Total: {total}")
#     print(f"Correct: {correct}")
#     print(f"Wrong: {total - correct}")
#     print(f"Accuracy: {correct / total:.4f}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--pt_paths", type=str, nargs="+", required=True, help="List of .pt files to test")
#     args = parser.parse_args()

#     for pt_path in args.pt_paths:
#         print(f"\n==== Testing {pt_path} ====")
#         main(pt_path, args.model_path)



import torch
import openai
import time
import argparse
import random
import json
from preprocess.tokenizer_bbpe import get_tokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed


client = openai.OpenAI(api_key= "sk-proj-2tQydgAaOYN7gc83zCDVT3BlbkFJVJyUpkmMnvdDAiUImA1e")

label_map = {1: "A", 2: "B", 3: "C", 4: "D"}

def build_5shot_prompt(data, idx, shots=5):
    idxs = list(range(len(data)))
    idxs.remove(idx)
    shot_idxs = random.sample(idxs, shots)
    prompt = ""
    for j in shot_idxs:
        demo_q = data[j]["prompt"].strip()
        demo_ans = label_map[data[j]["output"]]
        prompt += f"{demo_q}\n답: {demo_ans}\n\n"
    tgt_q = data[idx]["prompt"].strip()
    prompt += f"{tgt_q}\n답:"
    return prompt

def get_openai_answer(prompt, model="gpt-5"):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # max_completion_tokens=10000
        )
        # Print the entire API response for debugging
        # print(f"[RAW RESPONSE] {resp}")

        # Extract and print the raw answer
        if not resp.choices or not resp.choices[0].message or not resp.choices[0].message.content:
            # print("[DEBUG] No completion found in response.")
            return None

        ans = resp.choices[0].message.content.strip().upper()
        # print(f"[RAW] {ans}")

        # Try to extract A/B/C/D from answer
        for c in "ABCD":
            if c in ans:
                return c
        # Try to extract 1/2/3/4 as start or anywhere
        for num, letter in zip("1234", "ABCD"):
            if ans.startswith(num):
                return letter
        for num, letter in zip("1234", "ABCD"):
            if num in ans:
                return letter

        return None
    except Exception as e:
        print(f"[ERROR] OpenAI API exception: {e}")
        return None


def evaluate_file(jsonl_path, model="gpt-5", max_workers=10, shots=5):
    print(f"\n==== Results for {jsonl_path} ====")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    correct, total = 0, 0

    # Prepare all prompts and ground-truths
    prompts = [build_5shot_prompt(data, i, shots=shots) for i in range(len(data))]
    gt_labels = [label_map[item["output"]] for item in data]

    # Threaded querying
    preds = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(get_openai_answer, prompts[i], model): i
            for i in range(len(prompts))
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                preds[i] = future.result()
                # print(f"[DEBUG] Prompt #{i+1}/{len(prompts)} | Answer: {preds[i]}")
            except Exception as e:
                preds[i] = None
                print(f"[DEBUG] Prompt #{i+1}/{len(prompts)} | ERROR: {e}")

    # Accuracy calc
    for i in range(len(prompts)):
        total += 1
        if preds[i] == gt_labels[i]:
            correct += 1

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_paths", type=str, nargs="+", required=True, help="List of .jsonl files to test")
    parser.add_argument("--model_name", type=str, default="gpt-5")
    parser.add_argument("--max_workers", type=int, default=100)
    parser.add_argument("--shots", type=int, default=5)
    args = parser.parse_args()

    for jsonl_path in args.jsonl_paths:
        evaluate_file(jsonl_path, model=args.model_name, max_workers=args.max_workers, shots=args.shots)