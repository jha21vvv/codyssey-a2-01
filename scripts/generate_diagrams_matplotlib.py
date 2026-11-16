"""Matplotlib 기반 고해상도 아키텍처 다이어그램 생성 스크립트 (Step 4).

생성 파일:
- diagrams/01_system_architecture.png
- diagrams/02_linear_algebra_pipeline.png
- diagrams/03_backprop_chain_rule.png
- diagrams/04_optimizer_mle_flow.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Windows 콘솔 UTF-8 출력 방어
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#eaf2f8",
    ec: str = "#2980b9",
    fontsize: int = 10,
    bold: bool = False,
) -> None:
    """둥근 모서리 박스 및 중앙 텍스트를 그립니다."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.8,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1c2833",
    )


def _draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str = "#34495e") -> None:
    """두 좌표 사이에 방향 화살표를 그립니다."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=color, lw=1.8),
    )


def build_system_architecture(out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Codyssey A2-01: AI 수학 기반 전체 시스템 아키텍처", ha="center", va="center", fontsize=15, fontweight="bold")

    # 상단: 5대 핵심 수학 모듈
    modules = [
        (0.03, 0.62, 0.17, 0.22, "1. 선형대수\nsrc/linear_algebra.py\n\n• 행렬 변환 (R, S, Sh)\n• det(A) 면적 검증\n• Power Iteration\n• SVD 이미지 압축", "#ebf5fb", "#2e86c1"),
        (0.22, 0.62, 0.17, 0.22, "2. 미적분\nsrc/calculus.py\n\n• 중심차분 수치 미분\n• 다변수 Gradient\n• 등고선 직교성\n  시각화", "#eafaf1", "#27ae60"),
        (0.41, 0.62, 0.17, 0.22, "3. 역전파 유도\nnotebooks/backprop\n\n• 2층 신경망 수식\n• 연쇄 법칙 손계산\n• Tensor Shape 표기\n• NumPy 1e-4 검증", "#fef9e7", "#f39c12"),
        (0.60, 0.62, 0.17, 0.22, "4. 최적화\nsrc/optimizer.py\n\n• VanillaGD 클래스\n• Momentum(β=0.9)\n• 수렴/발산 경로\n• 타원형 함수 비교", "#fdedec", "#e74c3c"),
        (0.79, 0.62, 0.18, 0.22, "5. 확률/통계\nnotebooks/probability\n\n• 정규분포 PDF 2종\n• 베르누이 PMF 2종\n• Softmax 안정화\n• MSE/CE = MLE 유도", "#f4ecf7", "#8e44ad"),
    ]
    for x, y, w, h, txt, fc, ec in modules:
        _draw_box(ax, x, y, w, h, txt, fc=fc, ec=ec, fontsize=9)

    # 중간: 검증 및 시각화 레이어
    _draw_box(ax, 0.08, 0.30, 0.25, 0.16, "시각화 산출물\noutputs/*.png\n(변환, SVD, Gradient, 최적화, 확률)", fc="#f8f9f9", ec="#5d6d7e", fontsize=9.5, bold=True)
    _draw_box(ax, 0.38, 0.30, 0.25, 0.16, "초고속 단위 테스트\ntests/test_math_modules.py\n(pytest 12개 케이스 100% 통과)", fc="#f8f9f9", ec="#5d6d7e", fontsize=9.5, bold=True)
    _draw_box(ax, 0.68, 0.30, 0.25, 0.16, "평가자 원클릭 검증 러너\nscripts/verify_requirements.py\n(터미널 테이블 + JSON 출력)", fc="#f8f9f9", ec="#5d6d7e", fontsize=9.5, bold=True)

    for mx in [0.115, 0.305, 0.495, 0.685, 0.88]:
        _draw_arrow(ax, mx, 0.60, 0.205, 0.48, "#7f8c8d")
        _draw_arrow(ax, mx, 0.60, 0.505, 0.48, "#7f8c8d")
        _draw_arrow(ax, mx, 0.60, 0.805, 0.48, "#7f8c8d")

    # 하단: 학습/문서화 레이어
    _draw_box(ax, 0.15, 0.06, 0.70, 0.14, "학습 & 엔지니어링 문서화 레이어\nPLAN.md  |  스터디.md  |  REPORT.md  |  NOTEBOOKLM_FULL_SOURCE.txt  |  study/핵심_코드_스터디.md", fc="#e8f8f5", ec="#16a085", fontsize=10, bold=True)
    _draw_arrow(ax, 0.205, 0.28, 0.35, 0.22, "#16a085")
    _draw_arrow(ax, 0.505, 0.28, 0.50, 0.22, "#16a085")
    _draw_arrow(ax, 0.805, 0.28, 0.65, 0.22, "#16a085")

    save_path = out_dir / "01_system_architecture.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def build_linear_algebra_pipeline(out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "선형대수(Linear Algebra) 핵심 연산 파이프라인", ha="center", va="center", fontsize=15, fontweight="bold")

    # 트랙 A: 행렬 변환 & 행렬식
    _draw_box(ax, 0.04, 0.66, 0.18, 0.16, "단위 원 & 2D 격자\n(theta=0~2π, 100점)", fc="#ebf5fb", ec="#2980b9", bold=True)
    _draw_box(ax, 0.29, 0.66, 0.22, 0.16, "선형 변환 행렬 곱 A@X\nR(π/4), S(2, 0.5), Sh(k)", fc="#d4e6f1", ec="#2471a3")
    _draw_box(ax, 0.58, 0.66, 0.18, 0.16, "면적 변화율 검증\n|det(A)| vs Shoelace", fc="#eafaf1", ec="#27ae60")
    _draw_box(ax, 0.82, 0.66, 0.15, 0.16, "오차 < 1% 통과\nla_transforms.png", fc="#d5f5e3", ec="#1e8449", bold=True)
    _draw_arrow(ax, 0.24, 0.74, 0.27, 0.74)
    _draw_arrow(ax, 0.53, 0.74, 0.56, 0.74)
    _draw_arrow(ax, 0.78, 0.74, 0.80, 0.74)

    # 트랙 B: Power Iteration
    _draw_box(ax, 0.04, 0.38, 0.18, 0.16, "정방 행렬 A (N×N)\n초기 단위벡터 b0", fc="#fef9e7", ec="#d68910", bold=True)
    _draw_box(ax, 0.29, 0.38, 0.22, 0.16, "반복 정규화 루프\nb_{k+1} = Ab_k / ||Ab_k||", fc="#fdebd0", ec="#ca6f1e")
    _draw_box(ax, 0.58, 0.38, 0.18, 0.16, "Rayleigh Quotient\nλ_max = b^T A b", fc="#fef5e7", ec="#b9770e")
    _draw_box(ax, 0.82, 0.38, 0.15, 0.16, "np.linalg.eig 검증\n오차 < 5% 통과", fc="#fcf3cf", ec="#b7950b", bold=True)
    _draw_arrow(ax, 0.24, 0.46, 0.27, 0.46)
    _draw_arrow(ax, 0.53, 0.46, 0.56, 0.46)
    _draw_arrow(ax, 0.78, 0.46, 0.80, 0.46)

    # 트랙 C: SVD 이미지 압축
    _draw_box(ax, 0.04, 0.10, 0.18, 0.16, "64×64 Grayscale\n테스트 이미지 A", fc="#f4ecf7", ec="#8e44ad", bold=True)
    _draw_box(ax, 0.29, 0.10, 0.22, 0.16, "특이값 분해 (SVD)\nA = U Σ V^T", fc="#ebdef0", ec="#7d3c98")
    _draw_box(ax, 0.58, 0.10, 0.18, 0.16, "Rank-k 절단 복원\nk = 10, 50, 100", fc="#e8daef", ec="#6c3483")
    _draw_box(ax, 0.82, 0.10, 0.15, 0.16, "MSE & 압축률 비교\nla_svd_compression", fc="#d2b4de", ec="#5b2c6f", bold=True)
    _draw_arrow(ax, 0.24, 0.18, 0.27, 0.18)
    _draw_arrow(ax, 0.53, 0.18, 0.56, 0.18)
    _draw_arrow(ax, 0.78, 0.18, 0.80, 0.18)

    save_path = out_dir / "02_linear_algebra_pipeline.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def build_backprop_chain_rule(out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "2층 신경망 순전파(Forward) & 역전파(Backward) 연쇄 법칙 흐름도", ha="center", va="center", fontsize=15, fontweight="bold")

    # 순전파 (상단)
    ax.text(0.05, 0.80, "▶ 순전파 (Forward Pass)", fontsize=11, fontweight="bold", color="#1f618d")
    fwd_nodes = [
        (0.04, 0.58, 0.13, 0.16, "입력 x\nshape: (2,)\n[1, 0]"),
        (0.20, 0.58, 0.14, 0.16, "z1 = xW1+b1\nshape: (2,)\n[0.1, 0.2]"),
        (0.37, 0.58, 0.14, 0.16, "a1 = σ(z1)\nshape: (2,)\n[0.525, 0.550]"),
        (0.54, 0.58, 0.14, 0.16, "z2 = a1·W2+b2\nshape: ()\n0.5924"),
        (0.71, 0.58, 0.13, 0.16, "y_pred = σ(z2)\nshape: ()\n0.6439"),
        (0.87, 0.58, 0.10, 0.16, "BCE Loss\nshape: ()\n0.4402"),
    ]
    for x, y, w, h, txt in fwd_nodes:
        _draw_box(ax, x, y, w, h, txt, fc="#ebf5fb", ec="#2980b9", fontsize=9)
    for x1, x2 in [(0.175, 0.195), (0.345, 0.365), (0.515, 0.535), (0.685, 0.705), (0.845, 0.865)]:
        _draw_arrow(ax, x1, 0.66, x2, 0.66, "#2980b9")

    # 역전파 (하단)
    ax.text(0.05, 0.36, "◀ 역전파 (Backward Pass — Chain Rule)", fontsize=11, fontweight="bold", color="#c0392b")
    bwd_nodes = [
        (0.04, 0.12, 0.16, 0.18, "∂L/∂W1\nshape: (2, 2)\nx^T ⊗ ∂L/∂z1"),
        (0.23, 0.12, 0.16, 0.18, "∂L/∂z1\nshape: (2,)\n∂L/∂a1 ⊙ σ'(z1)"),
        (0.42, 0.12, 0.16, 0.18, "∂L/∂a1\nshape: (2,)\n∂L/∂z2 · W2"),
        (0.61, 0.12, 0.17, 0.18, "∂L/∂z2 & ∂L/∂W2\n(y_pred - y)\nW2 shape: (2,)"),
        (0.81, 0.12, 0.16, 0.18, "∂L/∂y_pred\nshape: ()\n-1.5530"),
    ]
    for x, y, w, h, txt in bwd_nodes:
        _draw_box(ax, x, y, w, h, txt, fc="#fdedec", ec="#c0392b", fontsize=9)
    for x_right, x_left in [(0.805, 0.785), (0.605, 0.585), (0.415, 0.395), (0.225, 0.205)]:
        _draw_arrow(ax, x_right, 0.21, x_left, 0.21, "#c0392b")

    # Loss에서 역전파 시작 화살표
    _draw_arrow(ax, 0.92, 0.56, 0.89, 0.32, "#8e44ad")

    save_path = out_dir / "03_backprop_chain_rule.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def build_optimizer_mle_flow(out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "최적화(VanillaGD vs Momentum) & 확률-손실함수(MLE) 연결 구조", ha="center", va="center", fontsize=15, fontweight="bold")

    # 좌측: 확률 & 손실 함수
    _draw_box(ax, 0.04, 0.62, 0.38, 0.20, "정규분포 가정 N(y_hat, σ²)\n음의 로그 우도(NLL) 최소화\n⟹ MSE Loss 도출", fc="#f4ecf7", ec="#8e44ad", fontsize=10, bold=True)
    _draw_box(ax, 0.04, 0.28, 0.38, 0.20, "베르누이/카테고리 분포 가정\n음의 로그 우도(NLL) 최소화\n⟹ Cross-Entropy Loss 도출", fc="#fef9e7", ec="#f39c12", fontsize=10, bold=True)

    # 중앙: 수치 미분 / Gradient
    _draw_box(ax, 0.48, 0.45, 0.20, 0.22, "Gradient 계산\n∇f(θ) = (∂f/∂x, ∂f/∂y)\n(등고선에 수직 방향)", fc="#eafaf1", ec="#27ae60", fontsize=10, bold=True)
    _draw_arrow(ax, 0.43, 0.72, 0.47, 0.58, "#8e44ad")
    _draw_arrow(ax, 0.43, 0.38, 0.47, 0.52, "#f39c12")

    # 우측: 옵티마이저 2종
    _draw_box(ax, 0.74, 0.62, 0.22, 0.20, "VanillaGD\nθ ← θ - lr · ∇f\n(타원 계곡에서 지그재그)", fc="#ebf5fb", ec="#2980b9", fontsize=9.5)
    _draw_box(ax, 0.74, 0.28, 0.22, 0.20, "Momentum (β=0.9)\nv ← βv + ∇f\nθ ← θ - lr · v\n(관성으로 진동 상쇄)", fc="#fdedec", ec="#e74c3c", fontsize=9.5)
    _draw_arrow(ax, 0.69, 0.58, 0.73, 0.71, "#2980b9")
    _draw_arrow(ax, 0.69, 0.52, 0.73, 0.39, "#e74c3c")

    _draw_box(ax, 0.15, 0.06, 0.70, 0.12, "최종 결과: lr=0.1 수렴(반경<0.1) vs lr>=0.5 발산 확인 & 타원형(x²+10y²) Momentum 우수성 입증", fc="#f8f9f9", ec="#34495e", fontsize=10, bold=True)

    save_path = out_dir / "04_optimizer_mle_flow.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)


def main() -> None:
    out_dir = Path("diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        build_system_architecture(out_dir),
        build_linear_algebra_pipeline(out_dir),
        build_backprop_chain_rule(out_dir),
        build_optimizer_mle_flow(out_dir),
    ]
    for p in paths:
        print(f"Generated diagram: {p}")


if __name__ == "__main__":
    main()
