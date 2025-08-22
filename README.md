
# 개요

**한국어 음식점 리뷰 데이터셋(KR3)**으로 간단한 GPT‑스타일 언어 모델을 학습하고, 프롬프트에 이어지는 토큰을 생성합니다.  
데이터 전처리 → 분산 학습(DDP) → 검증/추론까지의 워크플로를 **Poetry + Docker** 환경에서 실행합니다.

---

## 1) 실행 환경 (Docker 컨테이너)

```bash
docker run -it --rm   --gpus '"device=0,1,2,3"'   -v $(pwd):/app -w /app   <docker_image_name> bash
```

| 옵션 | 설명 |
|---|---|
| `-it` | 인터랙티브 모드/TTY |
| `--rm` | 컨테이너 종료 시 자동 삭제 |
| `--gpus '"device=0,1,2,3"'` | 사용할 GPU ID 지정 |
| `-v $(pwd):/app` | 현재 디렉터리를 컨테이너 `/app`에 마운트 |
| `-w /app` | 작업 디렉터리 설정 |
| `<docker_image_name>` | 사용할 이미지 이름 |

---

## 2) 의존성 설치

```bash
poetry install
```

`pyproject.toml`에 정의된 패키지를 설치합니다.

---

## 3) 데이터 준비

```bash
poetry run python load.py --output_path <output_file_path> --val_ratio <validation_split_ratio>
```

**load.py 주요 인자**

| parameter | 기본값 | 설명 |
|---|---|---|
| `--output_path` | `data/dataset.pt` | 저장 경로(내부적으로 `train.pt`, `validation.pt` 생성) |
| `--val_ratio` | `0.05` | 검증 데이터 비율 |

---

## 4) 모델 학습 (DDP + Cosine 스케줄러)

```bash
poetry run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus>   train.py   --data_dir <data_directory> --batch_size <batch_size> --epochs <num_epochs>   --chunk_size <chunk_size> --d_model <model_dim> --d_ffn <ffn_dim>   --n_layers <num_layers> --lr <learning_rate> --mode <tokenizer|manual>   [--arch decoder|ffn] [--dropout <p>] [--nhead <int>]
```

### torchrun 옵션
| parameter | 설명 |
|---|---|
| `--standalone` | 단일 노드 실행(별도 rendezvous 서버 불필요) |
| `--nnodes=1` | 노드 수 |
| `--nproc_per_node=<num_gpus>` | 노드당 프로세스(=GPU) 수 |

### **train.py **

| parameter | 기본값 | 설명 |
|---|---|---|
| `--arch` | `decoder` | 모델 아키텍처: `decoder`(Transformer) 또는 `ffn`(MLP‑only) |
| `--data_dir` | `data` | 학습/검증 데이터 디렉터리 |
| `--batch_size` | `64` | 배치 크기 |
| `--epochs` | `1` | 에폭 수 |
| `--chunk_size` | `256` | 입력 시퀀스 길이 |
| `--d_model` | `256` | 임베딩/히든 차원 |
| `--d_ffn` | `1024` | FFN 내부 차원 |
| `--n_layers` | `6` | 블록 수 |
| `--dropout` | `0.1` | 드롭아웃 비율 |
| `--lr` | `1e-3` | 학습률(AdamW) |
| `--mode` | `tokenizer` | 토크나이저: `tokenizer`(BBPE) 또는 `manual` |
| `--nhead` | *(자동)* | 멀티헤드 수. 미지정 시 `≈ d_model / 64`에 맞춰 **`d_model`로 나누어떨어지는 최대값**으로 조정 |

> 예) `E12_N24_512_2048_4.500.pt`

---

## 5) 검증 (파일명 파싱 지원)

```bash
poetry run python validate.py   --data_dir <data_directory> --batch_size <batch_size>   --chunk_size <chunk_size> --d_model <model_dim> --d_ffn <ffn_dim>   --n_layers <num_layers> --mode <tokenizer|manual>   --model_path <checkpoint_path> [--arch decoder|ffn]
```

**validate.py**

| parameter | 기본값 | 설명 |
|---|---|---|
| `--data_dir` | `data` | 검증 데이터 경로 |
| `--batch_size` | `64` | 배치 크기 |
| `--chunk_size` | `256` | 입력 길이 |
| `--d_model` | `256` | 모델 차원 |
| `--d_ffn` | `1024` | FFN 차원 |
| `--n_layers` | `6` | 블록 수 |
| `--mode` | `tokenizer` | 토크나이저 모드 |
| `--model_path` | (필수) | 체크포인트 경로 |
| `--arch` | *(auto)* | `decoder`/`ffn` 강제 지정. 미지정 시 **체크포인트 키로 자동 판별** |

- 패턴: `E{epochs}_N{n_layers}_{d_model}_{d_ffn}*.pt`  
- 예: `E12_N24_512_2048_4.5739.pt` → `n_layers=24, d_model=512, d_ffn=2048`을 자동으로 덮어씀
- DDP 저장(`module.` prefix)도 자동 제거 후 로드

출력: `Validation Loss`, `Perplexity`

---

## 6) 추론 (predict.py, 파일명 파싱 동일 적용)

```bash
poetry run python predict.py   --model_path <checkpoint_path>   --d_model <model_dim> --d_ffn <ffn_dim> --n_layers <num_layers>   --chunk_size <chunk_size> --max_token <max_generation_tokens>   [--mode <tokenizer|manual>] [--arch decoder|ffn]
```

**predict.py**

| parameter | 기본값 | 설명 |
|---|---|---|
| `--model_path` | (필수) | 학습된 모델 경로 |
| `--d_model` | `256` | 모델 차원 |
| `--d_ffn` | `1024` | FFN 차원 |
| `--n_layers` | `6` | 블록 수 |
| `--chunk_size` | `256` | 입력 길이 |
| `--max_token` | `50` | 최대 생성 토큰 수 |
| `--mode` | `tokenizer` | 토크나이저 모드 |
| `--arch` | *(auto)* | `decoder`/`ffn` 강제 지정(미지정 시 체크포인트로 자동 판별) |

- **validate.py와 동일한 파일명 파싱**을 사용하여 `n_layers/d_model/d_ffn`을 자동 설정
- 체크포인트 키로 `decoder`(Self‑Attention 키 존재) vs `ffn` 자동 판별

---

## 7) 테스트 스크립트 (test.py, 파일명 파싱 동일 적용)

`test.py` 역시 `validate.py`와 **동일한 파일명 파싱 규칙**을 적용합니다.

- 입력 체크포인트 이름이 `E{epochs}_N{layers}_{d_model}_{d_ffn}*.pt` 형식을 따를 경우,
  **명령행 인자로 전달한 값보다 파일명에서 파싱한 값이 우선 적용**됩니다.
- `--arch` 미지정 시, **체크포인트 키 스캔**으로 `decoder/ffn`을 자동 선택합니다.
- DDP 저장본(`module.` prefix)은 자동 정리 후 로드합니다.

---

## 8) 실행 순서 요약

1. 컨테이너 실행
2. `poetry install`
3. 데이터 전처리: `load.py`
4. 학습: `train.py` (DDP, Cosine 스케줄)
5. 검증: `validate.py` (파일명 파싱 지원)
6. 추론: `predict.py` (파일명 파싱 지원)
