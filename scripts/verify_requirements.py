"""원클릭 과제 요구사항 검증 러너 (Step 5).

평가자가 터미널에서 즉시 실행하여 모든 수학 모듈 및 요구사항 충족 여부를
시각적 테이블 또는 JSON(--json) 형태로 확인할 수 있습니다.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

# 프로젝트 루트를 sys.path에 추가하여 어디서 실행해도 src import 보장
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.linear_algebra import (
    create_sample_grayscale_image,
    power_iteration,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    svd_compress_reconstruct,
    verify_determinant_area,
    visualize_matrix_transforms,
    visualize_svd_compression,
)
from src.calculus import (
    numerical_derivative,
    visualize_gradient_field,
)
from src.optimizer import (
    Momentum,
    VanillaGD,
    grad_circle,
    grad_ellipse,
    visualize_gd_convergence,
    visualize_optimizer_comparison,
)


def run_all_verifications() -> List[Dict[str, Any]]:
    """모든 과제 요구사항을 실행하고 검증 결과 리스트를 반환합니다."""
    np.random.seed(42)
    results: List[Dict[str, Any]] = []

    # 1. 행렬식 vs 면적 비율 (오차 < 1%)
    for name, mat in [
        ("Rotation R(pi/4)", rotation_matrix(np.pi / 4.0)),
        ("Scaling S(2, 0.5)", scaling_matrix(2.0, 0.5)),
        ("Shear Sh(1.0)", shear_matrix(1.0)),
    ]:
        stats = verify_determinant_area(mat, n_points=100)
        passed = stats["rel_error"] < 0.01
        results.append({
            "id": f"LA-DET-{name.split()[0]}",
            "requirement": f"{name} |det(A)| vs 면적비 오차 < 1%",
            "metric": f"rel_error={stats['rel_error']:.6f}",
            "passed": bool(passed),
        })

    # 2. Power Iteration vs np.linalg.eig (오차 < 5%)
    mat_eig = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], dtype=np.float64)
    eigval, _ = power_iteration(mat_eig, seed=42)
    ref_vals, _ = np.linalg.eig(mat_eig)
    ref_max = float(np.max(np.abs(ref_vals)))
    eig_err = abs(abs(eigval) - ref_max) / ref_max
    results.append({
        "id": "LA-EIG",
        "requirement": "Power Iteration 최대 고유값 오차 < 5%",
        "metric": f"eig={eigval:.6f}, ref={ref_max:.6f}, err={eig_err:.2e}",
        "passed": bool(eig_err < 0.05),
    })

    # 3. SVD 압축 복원 (k=10, 50, 100)
    img = create_sample_grayscale_image(64)
    _, _, mse10 = svd_compress_reconstruct(img, 10)
    _, _, mse50 = svd_compress_reconstruct(img, 50)
    _, _, mse100 = svd_compress_reconstruct(img, 100)
    results.append({
        "id": "LA-SVD",
        "requirement": "SVD 64x64 압축 복원 (k=10, 50, 100)",
        "metric": f"MSE(10)={mse10:.2f}, MSE(50)={mse50:.2e}, MSE(100)={mse100:.2e}",
        "passed": bool(mse10 > mse50 and mse100 < 1e-6),
    })

    # 4. 중심차분 수치 미분 (f(x)=x^2, x=3 -> 오차 < 1e-4)
    deriv = numerical_derivative(lambda x: x ** 2, 3.0)
    deriv_err = abs(deriv - 6.0)
    results.append({
        "id": "CALC-DERIV",
        "requirement": "f(x)=x^2, x=3 수치 미분 오차 < 1e-4",
        "metric": f"approx={deriv:.8f}, err={deriv_err:.2e}",
        "passed": bool(deriv_err < 1e-4),
    })

    # 5. VanillaGD 수렴 (lr=0.1, 100 steps -> 반경 < 0.1 및 < 1.0) & 발산 (lr >= 0.5)
    traj_conv = VanillaGD(lr=0.1).optimize(grad_circle, np.array([4.0, 3.0]), num_steps=100)
    final_radius = float(np.linalg.norm(traj_conv[-1]))
    results.append({
        "id": "OPT-GD-CONV",
        "requirement": "VanillaGD lr=0.1, 100회 반경 < 0.1 (< 1.0)",
        "metric": f"final_radius={final_radius:.2e}",
        "passed": bool(final_radius < 0.1 and final_radius < 1.0),
    })

    init_div = np.array([1.0, 0.8])
    traj_div = VanillaGD(lr=1.05).optimize(grad_circle, init_div, num_steps=12)
    diverged = float(np.linalg.norm(traj_div[-1])) > float(np.linalg.norm(init_div))
    results.append({
        "id": "OPT-GD-DIV",
        "requirement": "VanillaGD lr>=0.5 (1.05) 발산 확인",
        "metric": f"init_norm={np.linalg.norm(init_div):.2f} -> end_norm={np.linalg.norm(traj_div[-1]):.2f}",
        "passed": bool(diverged),
    })

    # 6. Momentum(beta=0.9) 타원형 함수 수렴
    traj_mom = Momentum(lr=0.04, beta=0.9).optimize(grad_ellipse, np.array([8.0, 2.0]), num_steps=100)
    mom_radius = float(np.linalg.norm(traj_mom[-1]))
    results.append({
        "id": "OPT-MOM",
        "requirement": "Momentum(beta=0.9) 타원형 x^2+10y^2 수렴",
        "metric": f"final_radius={mom_radius:.4f}",
        "passed": bool(mom_radius < 0.5),
    })

    # 7. 역전파 고정 예제 손계산 vs NumPy 4자리 일치
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    x = np.array([1.0, 0.0])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    W2 = np.array([0.5, 0.6])
    a1 = sig(x @ W1)
    y_pred = float(sig(a1 @ W2))
    dW1 = np.outer(x, ((y_pred - 1.0) * W2) * a1 * (1.0 - a1))
    bp_match = bool(np.isclose(y_pred, 0.643913, atol=1e-4) and np.allclose(dW1[0], [-0.044400, -0.052882], atol=1e-4))
    results.append({
        "id": "BP-CHAIN",
        "requirement": "2층 신경망 역전파 손계산 vs NumPy 1e-4 일치",
        "metric": f"y_pred={y_pred:.6f}, dW1[0]={np.round(dW1[0], 6).tolist()}",
        "passed": bp_match,
    })

    # 8. Softmax 합 1.0과의 차이 < 1e-6
    logits = np.array([2.0, 1.0, 0.1, -0.5, 3.2])
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    sm_err = abs(float(np.sum(probs)) - 1.0)
    results.append({
        "id": "PROB-SOFTMAX",
        "requirement": "Softmax 출력 합 1.0과의 오차 < 1e-6",
        "metric": f"sum={np.sum(probs):.10f}, diff={sm_err:.2e}",
        "passed": bool(sm_err < 1e-6),
    })

    # 9. 시각화 출력 파일 및 노트북 생성 확인
    out_dir = ROOT_DIR / "outputs"
    visualize_matrix_transforms(output_dir=str(out_dir))
    visualize_svd_compression(output_dir=str(out_dir))
    visualize_gradient_field(output_dir=str(out_dir))
    visualize_gd_convergence(output_dir=str(out_dir))
    visualize_optimizer_comparison(output_dir=str(out_dir))

    expected_files = [
        "outputs/la_transforms.png",
        "outputs/la_svd_compression.png",
        "outputs/calc_gradient.png",
        "outputs/opt_gd_divergence.png",
        "outputs/opt_comparison_circle.png",
        "outputs/opt_comparison_ellipse.png",
        "outputs/prob_normal_pdf.png",
        "outputs/prob_bernoulli_pmf.png",
        "notebooks/backprop_derivation.ipynb",
        "notebooks/probability_loss.ipynb",
    ]
    missing = [f for f in expected_files if not (ROOT_DIR / f).exists()]
    results.append({
        "id": "ARTIFACTS",
        "requirement": "필수 PNG 시각화 및 노트북 산출물 존재",
        "metric": f"checked={len(expected_files)}, missing={len(missing)}",
        "passed": len(missing) == 0,
    })

    return results


def print_table(results: List[Dict[str, Any]]) -> None:
    """검증 결과를 컬러 터미널 테이블로 출력합니다."""
    green = "\033[92m"
    red = "\033[91m"
    bold = "\033[1m"
    reset = "\033[0m"

    print(f"\n{bold}=== Codyssey A2-01 요구사항 실시간 검증 리포트 ==={reset}")
    print(f"{'ID':<14} | {'상태':<6} | {'요구사항':<40} | {'측정 결과'}")
    print("-" * 105)
    for r in results:
        status = f"{green}PASS{reset}" if r["passed"] else f"{red}FAIL{reset}"
        print(f"{r['id']:<14} | {status:<15} | {r['requirement']:<40} | {r['metric']}")

    passed_cnt = sum(1 for r in results if r["passed"])
    total_cnt = len(results)
    print("-" * 105)
    print(f"{bold}총 {total_cnt}개 검증 항목 중 {passed_cnt}개 통과 ({passed_cnt/total_cnt*100:.1f}%){reset}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Codyssey A2-01 요구사항 검증 러너")
    parser.add_argument("--json", action="store_true", help="결과를 JSON 포맷으로 출력")
    args = parser.parse_args()

    results = run_all_verifications()
    all_passed = all(r["passed"] for r in results)

    if args.json:
        payload = {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "all_passed": all_passed,
            "results": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_table(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
