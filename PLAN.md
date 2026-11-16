# 📋 프로젝트 계획서 (PLAN.md) — AI 수학 기반 구현 및 시각화 (Codyssey A2-01)

## 1. 프로젝트 개요
- **프로젝트명**: AI 수학 기반 구현 및 시각화 (`codyssey-a2-01`)
- **핵심 목표**: 선형대수(행렬 변환·행렬식·Power Iteration·SVD), 미적분(중심차분 수치 미분·Gradient 등고선 시각화), 역전파(2층 신경망 연쇄 법칙 수식 유도 및 손계산 검증), 최적화(VanillaGD vs Momentum 수렴 경로 비교), 확률/통계(정규·베르누이 분포, Softmax, MSE/Cross-Entropy와 MLE 연결)를 **NumPy와 Matplotlib/Seaborn만으로** 직접 구현하고 시각화한다.
- **개발 환경**: Python 3.11.5 (3.8+ 호환), NumPy 1.24.3, Matplotlib 3.7.2, Seaborn 0.12.2, `np.random.seed(42)` 시드 고정.
- **절대 제약 사항**:
  - PyTorch, TensorFlow, JAX 등 자동 미분 라이브러리 사용 금지
  - scikit-learn의 PCA, 최적화 함수 사용 금지
  - `np.linalg.eig`는 오직 Power Iteration 결과 검증 목적에만 제한적으로 사용

---

## 2. 모듈 설계 및 인터페이스 규격

### 2.1 `src/linear_algebra.py` (선형대수 모듈)
| 함수명 | 역할 | 입력 / 출력 | 수학적 검증 기준 |
| :--- | :--- | :--- | :--- |
| `rotation_matrix(theta)` | 2D 회전 행렬 $R(\theta)$ 생성 | `theta: float` → `(2, 2) ndarray` | $\det(R) = 1$ |
| `scaling_matrix(sx, sy)` | 2D 스케일링 행렬 $S(s_x, s_y)$ 생성 | `sx, sy: float` → `(2, 2) ndarray` | $\det(S) = s_x \cdot s_y$ |
| `shear_matrix(kx, ky=0.0)` | 2D 전단 행렬 $Sh(k_x, k_y)$ 생성 | `kx, ky: float` → `(2, 2) ndarray` | $\det(Sh) = 1 - k_x k_y$ |
| `compute_polygon_area(points)` | 사선 공식(Shoelace)으로 2D 폐곡선 면적 계산 | `points: (2, N) ndarray` → `float` | 단위 원 근사 면적 $\approx \pi$ |
| `verify_determinant_area(matrix, n_points=100)` | $\lvert\det(A)\rvert$와 변환 전후 도형 면적 비율 비교 | `matrix: (2, 2)` → `dict` (det, area_ratio, rel_error) | 상대 오차 `< 1%` |
| `visualize_matrix_transforms(output_dir="outputs")` | 회전, 스케일링, 전단 변환 전후를 각각/통합 시각화 | `output_dir: str` → 저장된 PNG 경로 목록 | `outputs/la_transforms.png` |
| `power_iteration(matrix, num_iters=1000, tol=1e-9)` | Power Iteration으로 최대 고유값·고유벡터 계산 | `matrix: (N, N)` → `(eigenvalue, eigenvector)` | `np.linalg.eig` 대비 오차 `< 5%` |
| `svd_compress_reconstruct(image, k)` | `np.linalg.svd`로 Rank-$k$ 이미지 압축 및 복원 | `image: (H, W)`, `k: int` → `(reconstructed, ratio, mse)` | `k=10, 50, 100` 지원 |
| `visualize_svd_compression(image=None, ks=(10, 50, 100), output_dir="outputs")` | 64×64 이하 Grayscale 이미지 생성/압축/비교 출력 | `output_dir: str` → PNG 경로 | `outputs/la_svd_compression.png` |

### 2.2 `src/calculus.py` (미적분 모듈)
| 함수명 | 역할 | 입력 / 출력 | 수학적 검증 기준 |
| :--- | :--- | :--- | :--- |
| `numerical_derivative(func, x, h=1e-5)` | 중심차분법 기반 1변수 수치 미분 | `func: Callable`, `x: float` → `float` | $f(x)=x^2, x=3$에서 오차 `< 1e-4` |
| `numerical_gradient(func, point, h=1e-5)` | 중심차분법 기반 다변수 함수 Gradient 벡터 계산 | `func: Callable`, `point: (D,)` → `(D,) ndarray` | 해석적 Gradient $\nabla f$와 일치 |
| `visualize_gradient_field(output_dir="outputs")` | $f(x,y)=x^2+y^2$ 등고선 + Gradient 화살표(`quiver`) | `output_dir: str` → PNG 경로 | 등고선 수직 확인 (`equal` aspect), `outputs/calc_gradient.png` |

### 2.3 `src/optimizer.py` (최적화 모듈)
| 클래스/함수 | 역할 | 주요 속성/메서드 | 검증 기준 |
| :--- | :--- | :--- | :--- |
| `VanillaGD(lr=0.1)` | 기본 경사하강법 옵티마이저 | `step(params, grads)`, `optimize(grad_fn, init, steps)` | $f=x^2+y^2$, `lr=0.1`, 100회 후 반경 `< 0.1` |
| `Momentum(lr=0.1, beta=0.9)` | 모멘텀 기반 경사하강법 옵티마이저 | `velocity`, `step(...)`, `optimize(...)` | 타원형 $x^2+10y^2$에서 진동 완화 |
| `visualize_gd_convergence(output_dir="outputs")` | $x^2+y^2$ 수렴 경로(점선) & `lr>=0.5` 발산 그래프 | `output_dir: str` → PNG 경로 | `outputs/opt_gd_divergence.png` |
| `visualize_optimizer_comparison(output_dir="outputs")` | 원형($x^2+y^2$) & 타원형($x^2+10y^2$) VanillaGD vs Momentum 비교 | `output_dir: str` → PNG 경로 목록 | `outputs/opt_comparison_circle.png`, `outputs/opt_comparison_ellipse.png` |

### 2.4 주피터 노트북 (`notebooks/`)
1. **`notebooks/backprop_derivation.ipynb`**:
   - 2층 신경망($2 \to 2 \to 1$, Sigmoid, BCE) 순전파/역전파 수식 유도 (Markdown + LaTeX)
   - 고정 예제(`x=[1,0]`, `W1=[[0.1,0.2],[0.3,0.4]]`, `b1=[0,0]`, `W2=[0.5,0.6]`, `b2=0`, `y_true=1`) 손계산 과정 + 텐서 shape 표기 + 중간값(`z1, a1, z2, y_pred`) + 기울기($\partial L/\partial y_{pred}, \partial L/\partial z_2, \partial L/\partial W_2, \partial L/\partial a_1, \partial L/\partial z_1, \partial L/\partial W_1$)
   - NumPy 코드 검증 (소수점 4자리까지 일치 확인)
2. **`notebooks/probability_loss.ipynb`**:
   - 정규분포 $N(0,1)$ vs $N(2, 0.5)$ PDF 비교 시각화 (`outputs/prob_normal_pdf.png`)
   - 베르누이 분포 $B(0.3)$ vs $B(0.7)$ PMF 막대그래프 비교 시각화 (`outputs/prob_bernoulli_pmf.png`)
   - 수치 안정적 Softmax 구현 및 합 `1.0`과의 오차 `< 1e-6` 검증
   - MSE 최소화 = 정규분포 가정 하의 MLE 수식 유도
   - Cross-Entropy 최소화 = 베르누이/카테고리 분포 가정 하의 MLE 수식 유도

---

## 3. 예외 처리 및 엣지 케이스 규격
- **수치 미분 `h <= 0`**: `ValueError` 발생으로 0 나눗셈 및 역부호 방지.
- **Power Iteration 영행렬 / 비정방 행렬**: 비정방 행렬 입력 시 `ValueError`, 영벡터 도달 시 안전한 정규화 및 경고/예외 처리.
- **SVD 압축 `k` 범위**: `k <= 0`이면 `ValueError`, `k > min(H, W)`이면 최대 랭크로 안전하게 동작하며 경고 없이 유효 복원 수행.
- **Softmax 오버플로우**: `x - np.max(x)` 적용 및 빈 배열 입력 방어.
- **Windows 콘솔 UTF-8 인코딩**: 검증/생성 스크립트에서 `sys.stdout.reconfigure(encoding="utf-8")` 적용.
- **마크다운 이미지 상대경로**: 모든 문서(`README.md`, `스터디.md`, `REPORT.md`)에서 `diagrams/*.png`, `outputs/*.png` 상대경로만 사용.

---

## 4. 8단계 실행 마일스톤
1. **Step 1**: `PLAN.md` 및 `requirements.txt` 작성
2. **Step 2**: `src/linear_algebra.py`, `src/calculus.py`, `src/optimizer.py` 구현 및 `notebooks/*.ipynb` 작성/실행, `outputs/*.png` 생성
3. **Step 3**: `tests/test_math_modules.py` 작성 및 `pytest` 100% 통과 검증
4. **Step 4**: `스터디.md` 작성 및 `scripts/generate_diagrams_matplotlib.py`로 `diagrams/*.png` 생성 및 상대경로 임베딩
5. **Step 5**: `REPORT.md`, `scripts/verify_requirements.py` 작성 및 `README.md` 업데이트
6. **Step 6**: `scripts/build_notebooklm_source.py`, `NOTEBOOKLM_FULL_SOURCE.txt`, `NOTEBOOKLM_GUIDE.md` 작성
7. **Step 7**: `src/*.py` 전 라인 1차/2차 2단 주석 내재화 및 `NOTEBOOKLM_FULL_SOURCE.txt` 재빌드
8. **Step 8**: `study/핵심_코드_스터디.md` 작성 및 전체 테스트/검증 최종 확인
