"""Codyssey A2-01 수학 모듈 단위 테스트 (Step 3)."""

from pathlib import Path
import numpy as np
import pytest

from src.linear_algebra import (
    compute_polygon_area,
    create_sample_grayscale_image,
    generate_unit_circle,
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
    numerical_gradient,
    quadratic_2d,
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


# ---------------------------------------------------------------------------
# 1. 선형대수 (Linear Algebra) 테스트
# ---------------------------------------------------------------------------
def test_matrix_determinant_and_area_ratio_within_1_percent():
    """회전, 스케일링, 전단 행렬의 |det(A)|와 면적 비율 오차가 1% 이내인지 검증."""
    matrices = [
        rotation_matrix(np.pi / 4.0),
        scaling_matrix(2.0, 0.5),
        shear_matrix(1.2),
    ]
    for mat in matrices:
        res = verify_determinant_area(mat, n_points=100)
        assert res["rel_error"] < 0.01


def test_power_iteration_matches_linalg_eig_within_5_percent():
    """Power Iteration 최대 고유값이 np.linalg.eig 결과와 5% 이내인지 검증."""
    mat = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]], dtype=np.float64)
    eigval, eigvec = power_iteration(mat, num_iters=1000, tol=1e-9, seed=42)

    # 검증 목적 전용으로만 np.linalg.eig 사용
    ref_vals, _ = np.linalg.eig(mat)
    ref_max = float(np.max(np.abs(ref_vals)))
    rel_err = abs(abs(eigval) - ref_max) / ref_max
    assert rel_err < 0.05
    assert np.isclose(np.linalg.norm(eigvec), 1.0, atol=1e-6)


def test_svd_compression_reconstruction_k_10_50_100():
    """64x64 이미지에 대해 k=10, 50, 100 SVD 압축/복원이 정상 동작하고 MSE가 감소하는지 검증."""
    img = create_sample_grayscale_image(64)
    assert img.shape == (64, 64)

    _, _, mse_10 = svd_compress_reconstruct(img, 10)
    _, _, mse_50 = svd_compress_reconstruct(img, 50)
    recon_100, _, mse_100 = svd_compress_reconstruct(img, 100)

    assert recon_100.shape == img.shape
    assert mse_10 > mse_50
    assert mse_50 + 1e-12 >= mse_100
    assert mse_100 < 1e-6


def test_linear_algebra_edge_cases():
    """선형대수 모듈의 잘못된 입력에 대한 예외 처리 검증."""
    with pytest.raises(ValueError):
        generate_unit_circle(2)
    with pytest.raises(ValueError):
        compute_polygon_area(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        verify_determinant_area(np.eye(3))
    with pytest.raises(ValueError):
        power_iteration(np.ones((2, 3)))
    with pytest.raises(ValueError):
        power_iteration(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        create_sample_grayscale_image(65)
    with pytest.raises(ValueError):
        svd_compress_reconstruct(np.zeros((10, 10)), 0)


# ---------------------------------------------------------------------------
# 2. 미적분 (Calculus) 테스트
# ---------------------------------------------------------------------------
def test_numerical_derivative_x_squared_at_3():
    """f(x) = x^2, x=3에서 중심차분 수치 미분 오차가 1e-4 이내인지 검증."""
    approx = numerical_derivative(lambda x: x ** 2, 3.0)
    assert abs(approx - 6.0) < 1e-4


def test_numerical_gradient_2d():
    """f(x, y) = x^2 + y^2에서 Gradient가 (2x, 2y)와 일치하는지 검증."""
    pt = np.array([2.5, -1.5], dtype=np.float64)
    grad = numerical_gradient(quadratic_2d, pt)
    assert np.allclose(grad, np.array([5.0, -3.0]), atol=1e-6)


def test_calculus_edge_cases():
    """미소 변화량 h <= 0 및 잘못된 좌표 입력 예외 검증."""
    with pytest.raises(ValueError):
        numerical_derivative(lambda x: x ** 2, 3.0, h=0.0)
    with pytest.raises(ValueError):
        numerical_gradient(quadratic_2d, np.array([1.0, 2.0]), h=-1e-5)
    with pytest.raises(ValueError):
        numerical_gradient(quadratic_2d, np.array([]))


# ---------------------------------------------------------------------------
# 3. 최적화 (Optimizer) 테스트
# ---------------------------------------------------------------------------
def test_vanilla_gd_converges_within_radius_0_1():
    """VanillaGD(lr=0.1)가 100회 반복 후 원점 반경 0.1 이내로 수렴하는지 검증."""
    gd = VanillaGD(lr=0.1)
    traj = gd.optimize(grad_circle, np.array([4.0, 3.0]), num_steps=100)
    final_norm = float(np.linalg.norm(traj[-1]))
    assert final_norm < 0.1


def test_vanilla_gd_diverges_when_lr_ge_0_5():
    """lr >= 0.5(예: 1.05)에서 최적화 경로가 발산하는지 검증."""
    init = np.array([1.0, 0.8], dtype=np.float64)
    gd = VanillaGD(lr=1.05)
    traj = gd.optimize(grad_circle, init, num_steps=15)
    assert np.linalg.norm(traj[-1]) > np.linalg.norm(init)


def test_momentum_converges_on_circle_and_ellipse():
    """Momentum(beta=0.9)이 원형 및 타원형 함수에서 수렴하는지 검증."""
    mom_c = Momentum(lr=0.1, beta=0.9)
    traj_c = mom_c.optimize(grad_circle, np.array([4.0, 3.0]), num_steps=100)
    assert np.linalg.norm(traj_c[-1]) < 0.1

    mom_e = Momentum(lr=0.04, beta=0.9)
    traj_e = mom_e.optimize(grad_ellipse, np.array([8.0, 2.0]), num_steps=100)
    assert np.linalg.norm(traj_e[-1]) < 0.5


def test_optimizer_edge_cases():
    """옵티마이저 하이퍼파라미터 예외 검증."""
    with pytest.raises(ValueError):
        VanillaGD(lr=0.0)
    with pytest.raises(ValueError):
        Momentum(lr=-0.1, beta=0.9)
    with pytest.raises(ValueError):
        Momentum(lr=0.1, beta=1.0)
    with pytest.raises(ValueError):
        VanillaGD(lr=0.1).optimize(grad_circle, np.array([1.0, 1.0]), num_steps=0)


# ---------------------------------------------------------------------------
# 4. 시각화 PNG 생성 테스트 (임시 디렉토리)
# ---------------------------------------------------------------------------
def test_visualizations_create_pngs(tmp_path: Path):
    """모든 시각화 함수가 지정된 폴더에 정상 PNG 파일을 생성하는지 검증."""
    out_dir = str(tmp_path)
    paths = []
    paths.extend(visualize_matrix_transforms(output_dir=out_dir))
    paths.append(visualize_svd_compression(output_dir=out_dir))
    paths.append(visualize_gradient_field(output_dir=out_dir))
    paths.append(visualize_gd_convergence(output_dir=out_dir))
    paths.extend(visualize_optimizer_comparison(output_dir=out_dir))

    for p in paths:
        file_obj = Path(p)
        assert file_obj.exists()
        assert file_obj.stat().st_size > 0
