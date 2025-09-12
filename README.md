# 개요

**한국어 음식점 리뷰 데이터셋(KR3)**을 활용하여 GPT‑스타일 언어 모델을 학습하고, 프롬프트에 이어지는 토큰을 생성합니다.  
데이터 전처리 → 분산 학습(DDP) → 검증/추론까지의 워크플로를 **Poetry + Docker** 환경에서 실행합니다.

---

## 1) 실행 환경 (Docker 컨테이너)

```bash
docker build -t <docker_image_name> .
```

```bash
docker run -it --rm --gpus '"device=0,1,2,3"' <docker_image_name> bash
```
---

## 2) Dependency 설치

```bash
poetry install
```

---

## 3) 데이터 준비

### `load.py` — 텍스트 병합/분할

```bash
poetry run python load.py
# 현재 기본값: output_dir='data', val_ratio=0.1
```

**load.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| output_dir |  | str |  | data |  |  |  |  | 출력 디렉터리. 각 소스별 `*_train.txt`, `*_val.txt` 및 최종 `train.txt`, `validation.txt`를 생성합니다. |
| val_ratio |  | float |  | 0.1 |  |  |  |  | 검증 데이터 비율. |


### `tokenization.py` — 토크나이즈(배치 처리)

```bash
poetry run python tokenization.py 
```

**tokenization.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| data_dir | --data_dir | str |  | data |  |  |  |  |  |
| model_name | --model_name | str |  | skt/kogpt2-base-v2 |  |  |  |  |  |
| chunk_size | --chunk_size | int |  | 256 |  |  |  |  |  |
| batch_size | --batch_size | int |  | 5000 |  |  |  |  |  |


---

## 4) 모델 학습 (DDP + Cosine 스케줄러) (DDP + Cosine 스케줄러)

```bash
poetry run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
  train.py \
  --data_dir <data_directory> --batch_size <batch_size> --epochs <num_epochs> \
  --chunk_size <chunk_size> --d_model <model_dim> --d_ffn <ffn_dim> \
  --n_layers <num_layers> --lr <learning_rate> --mode <tokenizer|manual> \
  [--arch decoder|ffn] [--dropout <p>] [--nhead <int>] [--n_tokens <int>]
```

### torchrun 옵션
| parameter | 설명 |
|---|---|
| `--standalone` | 단일 노드 실행 |
| `--nnodes=1` | 노드 수 |
| `--nproc_per_node=<num_gpus>` | 노드당 프로세스(=GPU) 수 |

### **train.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| arch | --arch | str |  | decoder | ["ffn", "decoder"] |  |  |  | Choose 'ffn' (MLP-only) or 'decoder' (Transformer). |
| data_dir | --data_dir | str |  | data |  |  |  |  |  |
| batch_size | --batch_size | int |  | 64 |  |  |  |  |  |
| epochs | --epochs | int |  | 1 |  |  |  |  |  |
| chunk_size | --chunk_size | int |  | 256 |  |  |  |  |  |
| d_model | --d_model | int |  | 256 |  |  |  |  |  |
| d_ffn | --d_ffn | int |  | 1024 |  |  |  |  |  |
| n_layers | --n_layers | int |  | 6 |  |  |  |  |  |
| dropout | --dropout | float |  | 0.1 |  |  |  |  |  |
| lr | --lr | float |  | 0.001 |  |  |  |  |  |
| mode | --mode | str |  | tokenizer | ["tokenizer", "manual"] |  |  |  |  |
| nhead | --nhead | int |  |  |  |  |  |  | Number of attention heads (defaults to ~d_model/64). |
| n_tokens | --n_tokens | int |  | 30 |  |  |  |  | Train up to this many million tokens PER EPOCH. |

---

## 5) Validation

```bash
poetry run python validation.py \
  --data_dir <data_directory> --batch_size <batch_size> --chunk_size <chunk_size> \
  --d_model <model_dim> --d_ffn <ffn_dim> --n_layers <num_layers> \
  --model_path <checkpoint_path> [--arch decoder|ffn] [--n_tokens <int>]
```

**validation.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| data_dir | --data_dir | str |  | data |  |  |  |  |  |
| batch_size | --batch_size | int |  | 64 |  |  |  |  |  |
| chunk_size | --chunk_size | int |  | 256 |  |  |  |  |  |
| d_model | --d_model | int |  | 256 |  |  |  |  |  |
| d_ffn | --d_ffn | int |  | 1024 |  |  |  |  |  |
| n_layers | --n_layers | int |  | 6 |  |  |  |  |  |
| model_path | --model_path | str |  | output/gpt_model_ddp.pt |  |  |  |  |  |
| arch | --arch | str |  |  | ["ffn", "decoder"] |  |  |  | Force model type; if omitted, auto-detect from checkpoint. |
| n_tokens | --n_tokens | int |  |  |  |  |  |  | Limit validation to this many million tokens. |

---

## 6) Predict

```bash
poetry run python predict.py \
  --model_path <checkpoint_path> \
  --d_model <model_dim> --d_ffn <ffn_dim> --n_layers <num_layers> \
  --chunk_size <chunk_size> --max_token <max_generation_tokens> \
  [--mode <tokenizer|manual>] [--arch decoder|ffn]
```

**predict.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| model_path | --model_path | str |  | output/gpt_model_ddp.pt |  |  |  |  |  |
| arch | --arch | str |  |  | ["ffn", "decoder"] |  |  |  |  |
| d_model | --d_model | int |  | 256 |  |  |  |  |  |
| d_ffn | --d_ffn | int |  | 1024 |  |  |  |  |  |
| n_layers | --n_layers | int |  | 6 |  |  |  |  |  |
| chunk_size | --chunk_size | int |  | 256 |  |  |  |  |  |
| max_token | --max_token | int |  | 50 |  |  |  |  |  |

---

## 7) Test

```bash
poetry run python test.py \
  --data_dir <data_directory> --batch_size <batch_size> \
  --chunk_size <chunk_size> --d_model <model_dim> --d_ffn <ffn_dim> \
  --n_layers <num_layers> --mode <tokenizer|manual> \
  --model_path <checkpoint_path> [--arch decoder|ffn]
```

**test.py**

| Argument | Flags | Type | Required | Default | Choices | Action | Nargs | Metavar | Help |
|---|---|---|---:|---|---|---|---|---|---|
| data_dir | --data_dir | str |  | data |  |  |  |  |  |
| batch_size | --batch_size | int |  | 64 |  |  |  |  |  |
| chunk_size | --chunk_size | int |  | 256 |  |  |  |  |  |
| d_model | --d_model | int |  | 256 |  |  |  |  |  |
| d_ffn | --d_ffn | int |  | 1024 |  |  |  |  |  |
| n_layers | --n_layers | int |  | 6 |  |  |  |  |  |
| mode | --mode | str |  | tokenizer | ["tokenizer", "manual"] |  |  |  |  |
| model_path | --model_path | str |  | output/gpt_model_ddp.pt |  |  |  |  |  |
| arch | --arch | str |  |  | ["ffn", "decoder"] |  |  |  | Force model type; if omitted, auto-detect from checkpoint. |

---

## 8) 실행 순서 요약

1. 컨테이너 실행
2. `poetry install`
3. 데이터 전처리: `load.py``tokenization.py`
4. 학습: `train.py` (DDP, Cosine 스케줄)
5. 검증: `validation.py`
6. 추론: `predict.py`
7. 테스트: `test.py`

---
