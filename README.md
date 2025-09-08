# Seg-M2F: Mask2Former Segmentation Tool

GPU 서버에서 실행 가능한 Mask2Former 기반 COCO panoptic segmentation 도구입니다. 도로 구별에 최적화되어 있습니다.

## 🚀 설치

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 폴더 구조 생성 (선택사항)
```bash
mkdir -p data/input data/output
```

## 📁 프로젝트 구조
```
seg_m2f/
├── seg_m2f.py                    # 메인 실행 파일
├── hf_model_manager.py            # 모델 관리
├── hf_segmentation_processor.py   # 세그멘테이션 처리
├── config/
│   └── hf_model_config.yaml      # 설정 파일
├── requirements.txt              # 의존성
├── README.md                     # 사용법
└── data/                         # 데이터 폴더 (선택)
    ├── input/                    # 입력 이미지/비디오
    └── output/                   # 출력 결과
```

## 🖥️ 사용법

### 1. 단일 이미지 처리
```bash
# 오버레이 결과 (기본)
python seg_m2f.py image input.jpg -o output.jpg

# 세그멘테이션 결과만
python seg_m2f.py image input.jpg -o output.jpg --output-mode result

# 둘 다 저장
python seg_m2f.py image input.jpg -o output.jpg --output-mode both
```

### 2. 이미지 폴더 전체 처리
```bash
# 폴더 내 모든 이미지 처리
python seg_m2f.py folder ./data/input -o ./data/output

# 세그멘테이션 결과만 저장
python seg_m2f.py folder ./data/input -o ./data/output --output-mode result
```

### 3. 비디오 처리
```bash
# 비디오 파일 처리
python seg_m2f.py video input.mp4 -o output.mp4

# 세그멘테이션만 (오버레이 없음)
python seg_m2f.py video input.mp4 -o output.mp4 --output-mode result
```

### 4. 커스텀 설정 사용
```bash
python seg_m2f.py image input.jpg -o output.jpg -c custom_config.yaml
```

## ⚙️ 설정

`config/hf_model_config.yaml`에서 모델 설정을 변경할 수 있습니다:

```yaml
model:
  # 기본 모델 (COCO panoptic segmentation - 도로 구별 최적화)
  name: "facebook/mask2former-swin-small-coco-panoptic"
  device: "auto"  # auto, cpu, cuda
  confidence_threshold: 0.3
```

### 사용 가능한 모델들:
- **Panoptic Segmentation (COCO)**: 객체+배경 분할 ⭐ 도로용 추천!
  - `facebook/mask2former-swin-small-coco-panoptic` (기본)
  - `facebook/mask2former-swin-base-coco-panoptic`
  - `facebook/mask2former-swin-large-coco-panoptic`

- **Instance Segmentation (COCO)**: 객체별 분할만
  - `facebook/mask2former-swin-small-coco-instance`
  - `facebook/mask2former-swin-base-coco-instance`
  
- **Semantic Segmentation (ADE20K)**: 모든 픽셀 분류
  - `facebook/mask2former-swin-small-ade-semantic`
  - `facebook/mask2former-swin-base-ade-semantic`

## 🎨 색상 매핑

COCO-Stuff panoptic segmentation에서 주요 클래스별 고정 색상 (도로 최우선!):

### 🛣️ 도로 관련 (매우 눈에 띄는 색상):
- **road (100)**: 밝은 노란색 🟡 - 가장 눈에 띔!
- **pavement-merged (123)**: 핫 마젠타 🟣 - 매우 눈에 띔!
- **railroad (98)**: 순수 빨간색 🔴
- **playingfield (97)**: 주황색 🟠

### 🚗 교통수단:
- **car (2)**: 딥핑크 🩷
- **bicycle (1)**: 스프링그린 🟢
- **bus (5)**: 주황색 🟠
- **truck (7)**: 오렌지 🧡

### 🌳 자연물:
- **tree-merged (116)**: 초록색 🟢
- **grass-merged (125)**: 밝은 라임 🟢
- **sky-other-merged (119)**: 하늘색 🔵
- **sea (103)**: 시안 🩵

### 🏢 건물:
- **building-other-merged (129)**: 파란색 🔵
- **house (91)**: 핑크 🩷

### 👤 사람:
- **person (0)**: 노란색 🟡

## 📊 출력 정보

실행 시 콘솔에 다음 정보가 표시됩니다:
- 감지된 객체 수
- 클래스별 객체 개수
- 처리 진행 상황 (폴더/비디오 처리 시)
- GPU/CPU 사용 정보

## 🔧 GPU 최적화

GPU가 있는 경우 자동으로 감지하여 사용합니다:
```
🚀 GPU detected: NVIDIA GeForce RTX 3090
✅ HuggingFace model loaded successfully on cuda
```

CPU만 있는 경우:
```
💻 No GPU detected, using CPU
✅ HuggingFace model loaded successfully on cpu
```

## 📈 성능 팁

1. **GPU 사용**: CUDA가 설치된 환경에서 실행하면 속도가 크게 향상됩니다.
2. **배치 처리**: 폴더 모드로 여러 이미지를 한 번에 처리하세요.
3. **모델 선택**: small < base < large 순으로 정확도가 높지만 속도는 느려집니다.

## 🐛 문제 해결

### 모델 로딩 실패
```bash
# 인터넷 연결 확인 후 재시도
python seg_m2f.py image test.jpg -o result.jpg
```

### CUDA 메모리 부족
```yaml
# config에서 더 작은 모델로 변경
model:
  name: "facebook/mask2former-swin-small-ade-semantic"
```

### 이미지 로드 실패
- 지원되는 형식: JPG, JPEG, PNG, BMP, TIFF
- 파일 경로에 한글이나 특수문자가 있는지 확인하세요.