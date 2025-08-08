# 개요

**한국어 음식점 리뷰 데이터셋(KR3)**을 활용하여, 간단한 GPT‑스타일 언어 모델을 학습하고, 사용자가 예시 문장을 입력하면 이어지는 단어들을 예측합니다.
데이터 전처리부터 모델 학습, 그리고 대화형 추론까지의 전 과정을 포함하며, 모든 명령어는 **Poetry** 환경과 **Docker 컨테이너**에서 실행됩니다.

---

## 1. 실행 환경 (Docker 컨테이너)

```bash
docker run -it --rm   --gpus '"device=0,1,2,3"'   -v $(pwd):/app   -w /app   <docker_image_name>   bash
```

| 옵션 | 설명 |
|------|------|
| `-it` | 인터랙티브 모드와 TTY 할당 |
| `--rm` | 컨테이너 종료 시 자동 삭제 |
| `--gpus '"device=0,1,2,3"'` | 사용할 GPU ID 지정 (예: 0–3번) |
| `-v $(pwd):/app` | 현재 디렉터리를 컨테이너 `/app`에 마운트 |
| `-w /app` | 컨테이너 작업 디렉터리를 `/app`으로 설정 |
| `<docker_image_name>` | 사용할 Docker 이미지 이름 |
| `bash` | 컨테이너에서 Bash 셸 실행 |

---

## 2. Dependency 설치

```bash
poetry install
```

> `pyproject.toml`에 정의된 모든 Python 패키지를 설치합니다.

---

## 3. 데이터 준비

```bash
poetry run python load.py --output_path <output_file_path> --val_ratio <validation_ratio>
```

**load.py 주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--output_path` | `data/dataset.pt` | directory to save processed data |
| `--val_ratio` | `0.05` | fraction of data reserved for validation |

---

## 4. 모델 학습

```bash
poetry run torchrun   --standalone   --nnodes=1   --nproc_per_node=<num_gpus>   train.py --batch_size <batch_size> --epochs <num_epochs>
```

**torchrun 옵션**

| 옵션 | 설명 |
|------|------|
| `--standalone` | 별도 렌데부 서버 없이 단일 노드 실행 |
| `--nnodes=1` | 사용 노드 수 |
| `--nproc_per_node=<num_gpus>` | 노드당 프로세스(=GPU) 수 |

**train.py 주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data_path` | `data/dataset.pt` | 학습 데이터 경로 |
| `--batch_size` | `64` | 미니배치 크기 |
| `--epochs` | `1` | 학습 에폭 수 |
| `--chunk_size` | `256` | 토큰 시퀀스 길이 |
| `--d_ffn` | `1024` | FFN(Feed Forward Network) 차원 |
| `--n_layers` | `6` | Transformer 레이어 수 |
| `--lr` | `1e-3` | 학습률 |
| `--mode` | `tokenizer` | 토크나이저 선택 (`tokenizer` 또는 `manual`) |

---

## 5. 추론 실행

```bash
poetry run python predict.py   --model_path <trained_model_path>   --chunk_size <chunk_size>   --n_layers <num_layers>
```

**predict.py 주요 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model_path` | `output/gpt_model_ddp.pt` | 학습된 모델 파라미터 경로 |
| `--chunk_size` | `256` | 입력 시퀀스 길이 (학습 시 설정과 동일) |
| `--n_layers` | `6` | 모델 레이어 수 (학습 시 설정과 동일) |

> 실행 후 Prompt 입력 대기 상태가 되며, 문장을 입력하면 모델이 이어서 문장을 생성합니다.  
> `exit` 입력 시 종료됩니다.

---

## 6. 요약 실행 흐름

1. **Docker 컨테이너 실행**
2. **Poetry로 의존성 설치**
3. **데이터 다운로드 및 정제 (`load.py`)**
4. **분산 학습 (`train.py` with `torchrun`)**
5. **대화형 추론 (`predict.py`)**

---
