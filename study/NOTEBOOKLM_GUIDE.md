# 🎙️ Google NotebookLM 오디오 오버뷰(딥다이브 팟캐스트) 기획 가이드 (`NOTEBOOKLM_GUIDE.md`)

> **[필수] NotebookLM 업로드 소스 파일 체크리스트**
> Google NotebookLM은 `.py` 소스 파일 직접 업로드를 지원하지 않으므로, 아래 5개 문서를 소스로 업로드하세요.
> - [x] `NOTEBOOKLM_FULL_SOURCE.txt` *(테스트·프롬프트가 제외된 순수 핵심 구현 소스 코드 전체 통합본)*
> - [x] `스터디.md` *(원본 과제 전문, 1:1 매핑표, 3개월 차 비유 및 아키텍처 다이어그램 해설)*
> - [x] `REPORT.md` *(아키텍처 의사결정, 엣지 케이스 방어 내역, 벤치마크 및 검증 결과)*
> - [x] `PLAN.md` *(모듈 설계 규격 및 예외 처리 명세)*
> - [x] `study/핵심_코드_스터디.md` *(핵심 코드 블록 단어 분해 & 비유 학습서)*

---

## 1. 호스트 2인 배경 스토리 및 대화 유도 질문 리스트

### 호스트 설정
- **호스트 A (시니어 AI 아키텍트)**: PyTorch나 TensorFlow 없이 오직 NumPy만으로 신경망의 수학적 뼈대(선형대수, 미적분, 역전파, 최적화, MLE)를 직접 구현한 이유와 설계 철학을 짚어주는 역할.
- **호스트 B (3개월 차 주니어 멘토)**: *"행렬식 `det(A)`가 왜 도형 넓이 배율이 되나요?"*, *"왜 `lr=1.05`면 경사하강법이 폭발하나요?"* 같은 주니어의 가려운 곳을 긁어주며 일상 비유로 풀어주는 역할.

### 대화 유도 질문 리스트
1. *"요즘 PyTorch에서 `loss.backward()` 한 줄이면 끝나는데, 굳이 `src/linear_algebra.py`와 `src/calculus.py`를 NumPy로 한 땀 한 땀 짠 이유가 뭔가요?"*
2. *"단위 원에 회전·스케일링·전단 행렬을 곱했을 때 `|det(A)|`와 사선 공식(Shoelace) 면적 비율이 소수점 아래까지 일치한다는 게 기하학적으로 어떤 의미인가요?"*
3. *"64×64 이미지를 SVD로 압축할 때 `k=10`에서 `k=50`으로 올리면 왜 화질이 거의 원본과 똑같아지나요?"*
4. *"타원형 함수 $f(x, y) = x^2 + 10y^2$에서 `VanillaGD`가 지그재그로 헤맬 때 `Momentum(beta=0.9)`은 어떻게 볼링공처럼 부드럽게 골짜기로 내려가나요?"*
5. *"결국 MSE 손실 함수는 정규분포의 MLE이고, Cross-Entropy는 베르누이/카테고리 분포의 MLE라는 게 AI 모델 설계에 어떤 통찰을 주나요?"*

---

## 2. 핵심 기술적 결정 3가지 & 일상 비유 모음

1. **Zero-Autograd 원칙 (PyTorch/Sklearn 배제 & NumPy 순수 구현)**
   - **비유**: 자동변속기(AutoGrad) 스포츠카를 타기 전에, 클러치와 기어(행렬 곱, 중심차분, 연쇄 법칙)가 어떻게 맞물려 바퀴가 굴러가는지 직접 조립해보는 정비소 실습입니다.
2. **수치 안정성 방어 (Softmax `max` 빼기 & 중심차분 `h` 경계 검증)**
   - **비유**: Softmax에서 최댓값을 먼저 빼주는 것은 **큰 숫자들이 `exp()`를 만나 폭발(Overflow)하기 전에 키높이를 나란히 낮춰주는 안전밸브**입니다.
3. **고정 시드(`np.random.seed(42)`) & 다중 엣지 케이스 가드**
   - **비유**: 실험실의 온도·습도를 고정해두고(시드 고정), 비커에 0이나 음수 같은 잘못된 시약(`h <= 0`, 영행렬, `k <= 0`)이 들어오면 즉시 경고등을 울리는 안전장치입니다.

---

## 3. 3개월 차 주니어가 반드시 깨달아야 할 청취 포인트 3가지

1. **선형 변환은 "공간의 왜곡"이고, 행렬식(`det`)은 "넓이 배율"이다.**
2. **Gradient는 항상 등고선과 수직(90도)이며, 경사하강법은 그 반대 방향으로 걷는다.**
3. **손실 함수(MSE, Cross-Entropy)는 하늘에서 떨어진 공식이 아니라, 데이터 분포(정규분포, 베르누이 분포)를 가정한 MLE의 자연스러운 결과다.**

---

## 4. NotebookLM Customize Audio Overview 전용 시스템 프롬프트

### 🇰🇷 한국어 전용 프롬프트 (복사용)
```text
당신들은 시니어 AI 아키텍트와 3개월 차 주니어 개발자 멘토입니다.
업로드된 소스 코드(NOTEBOOKLM_FULL_SOURCE.txt)와 학습 문서(스터디.md, REPORT.md)를 바탕으로,
PyTorch 없이 NumPy만으로 구현한 AI 수학 기반 프로젝트(선형대수, 중심차분 수치미분, 2층 신경망 역전파, VanillaGD vs Momentum, MSE/Cross-Entropy의 MLE 유도)를
출퇴근길에 귀로 들으며 직관적으로 이해할 수 있도록 재미있는 일상 비유를 곁들여 대담 형식으로 설명해주세요.
특히 행렬식과 면적 비율, 타원형 골짜기에서의 Momentum 효과, 손실 함수와 확률 분포의 연결을 중점적으로 다뤄주세요.
```

### 🇺🇸 English Prompt (For Copy-Paste)
```text
You are two hosts: a Senior AI Architect and a Junior Developer Mentor.
Based on the uploaded pure NumPy implementation (NOTEBOOKLM_FULL_SOURCE.txt) and engineering docs (스터디.md, REPORT.md),
host an engaging, intuitive deep-dive podcast explaining the mathematical foundations of AI built from scratch without PyTorch:
1) Matrix transformations & determinant as area scaling factor, Power Iteration, and SVD image compression;
2) Central-difference numerical gradients orthogonal to contour lines;
3) 2-layer neural network backpropagation via chain rule with explicit tensor shapes;
4) VanillaGD vs. Momentum on elliptical valleys (x^2 + 10y^2);
5) How MSE and Cross-Entropy naturally emerge from Maximum Likelihood Estimation (MLE).
Use vivid everyday analogies suitable for a developer with 3 months of experience.
```
