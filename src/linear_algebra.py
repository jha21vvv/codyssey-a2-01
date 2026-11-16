"""선형대수(Linear Algebra) 핵심 구현 및 시각화 모듈.

포함 기능:
- 2D 행렬 변환(회전, 스케일링, 전단) 생성 및 단위 원/격자 시각화
- 행렬식 det(A)와 변환 전후 도형 면적 비율 비교 검증 (오차 1% 이내)
- Power Iteration 기반 최대 고유값 및 대응 고유벡터 계산
- SVD 기반 64x64 이하 Grayscale 이미지 압축 및 복원 (k=10, 50, 100)
"""

# [1차]: 파일 및 디렉토리 경로 처리를 위해 pathlib의 Path 클래스를 임포트합니다.
# [2차]: 결과 이미지들을 저장할 폴더 주소를 다루기 위해 주소록 도구를 꺼냅니다.
from pathlib import Path
# [1차]: 타입 힌트(Dict, List, Optional, Sequence, Tuple)를 위해 typing 모듈에서 임포트합니다.
# [2차]: 함수가 어떤 자료형의 데이터를 받고 돌려주는지 안내하는 이름표를 준비합니다.
from typing import Dict, List, Optional, Sequence, Tuple

# [1차]: Matplotlib 핵심 패키지를 임포트합니다.
# [2차]: 도형과 이미지를 그릴 도화지 세트를 꺼냅니다.
import matplotlib
# [1차]: GUI 창을 띄우지 않고 파일로만 렌더링하도록 'Agg' 백엔드를 지정합니다.
# [2차]: 화면에 창을 띄우지 않고 조용히 사진 파일(.png)로만 인화하게 설정합니다.
matplotlib.use("Agg")
# [1차]: pyplot 모듈을 plt 별칭으로 임포트합니다.
# [2차]: 그래프를 그리는 간편한 리모컨을 plt라고 부릅니다.
import matplotlib.pyplot as plt
# [1차]: 수치 연산 라이브러리 NumPy를 np 별칭으로 임포트합니다.
# [2차]: 행렬 곱셈, 삼각함수, SVD를 빠르게 처리하는 공학용 계산기를 불러옵니다.
import numpy as np


# [1차]: 회전 각도 theta(라디안)를 받아 (2, 2) 회전 변환 행렬을 생성하는 함수입니다.
# [2차]: 도화지 위의 점들을 원점 기준으로 theta만큼 빙글 돌려주는 회전판을 만듭니다.
def rotation_matrix(theta: float) -> np.ndarray:
    """2D 회전 변환 행렬 R(theta)를 생성합니다.

    수식:
        R(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]]

    Args:
        theta: 반시계 방향 회전 각도 (라디안).

    Returns:
        (2, 2) 크기의 회전 변환 행렬.
    """
    # [1차]: 코사인 값 cos(theta)를 float로 계산합니다.
    # [2차]: 회전 후 가로 성분이 얼마나 남는지 비율을 잽니다.
    c = float(np.cos(theta))
    # [1차]: 사인 값 sin(theta)를 float로 계산합니다.
    # [2차]: 회전 후 세로 방향으로 얼마나 꺾이는지 비율을 잽니다.
    s = float(np.sin(theta))
    # [1차]: [[c, -s], [s, c]] 2x2 float64 배열을 생성해 반환합니다.
    # [2차]: 네 칸짜리 회전판 행렬을 조립해서 돌려줍니다.
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# [1차]: x축 배율 sx와 y축 배율 sy를 받아 (2, 2) 스케일링 변환 행렬을 생성합니다.
# [2차]: 고무판을 가로로 sx배, 세로로 sy배 늘리거나 줄이는 확대/축소기를 만듭니다.
def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """2D 스케일링 변환 행렬 S(sx, sy)를 생성합니다.

    수식:
        S(sx, sy) = [[sx, 0],
                     [0, sy]]

    Args:
        sx: x축 방향 배율.
        sy: y축 방향 배율.

    Returns:
        (2, 2) 크기의 스케일링 변환 행렬.
    """
    # [1차]: 주대각선에 sx, sy를 배치한 (2, 2) float64 대각행렬을 반환합니다.
    # [2차]: 가로는 sx배, 세로는 sy배로 바꾸는 확대/축소 행렬을 돌려줍니다.
    return np.array([[float(sx), 0.0], [0.0, float(sy)]], dtype=np.float64)


# [1차]: x축 전단 계수 kx와 y축 전단 계수 ky를 받아 (2, 2) 전단 행렬을 생성합니다.
# [2차]: 네모난 카드 덱을 옆으로 비스듬히 밀어 평행사변형으로 기울이는 전단기를 만듭니다.
def shear_matrix(kx: float, ky: float = 0.0) -> np.ndarray:
    """2D 전단(Shear) 변환 행렬 Sh(kx, ky)를 생성합니다.

    수식:
        Sh(kx, ky) = [[1,  kx],
                      [ky, 1 ]]

    Args:
        kx: x축 방향 전단 계수.
        ky: y축 방향 전단 계수 (기본값 0.0).

    Returns:
        (2, 2) 크기의 전단 변환 행렬.
    """
    # [1차]: 주대각선이 1이고 비대각 성분이 kx, ky인 (2, 2) 전단 행렬을 반환합니다.
    # [2차]: 높이는 그대로 두고 옆으로 비스듬히 밀어주는 기울이기 행렬을 돌려줍니다.
    return np.array([[1.0, float(kx)], [float(ky), 1.0]], dtype=np.float64)


# [1차]: n_points개의 점을 등간격으로 배치하여 (2, n_points) 단위 원 좌표를 생성합니다.
# [2차]: 반지름이 1인 동그란 고무줄 위에 점 100개를 똑같은 간격으로 찍어냅니다.
def generate_unit_circle(n_points: int = 100) -> np.ndarray:
    """단위 원 위의 점들을 (2, n_points) 배열로 생성합니다.

    수식:
        x = cos(theta), y = sin(theta), theta in [0, 2*pi]

    Args:
        n_points: 생성할 점의 개수 (최소 3 이상).

    Returns:
        (2, n_points) 크기의 좌표 배열.

    Raises:
        ValueError: n_points가 3 미만일 때 발생.
    """
    # [1차]: 다각형 면적 근사를 위해 점 개수가 3개 미만인지 검사합니다.
    # [2차]: 점이 2개 이하이면 선분일 뿐 넓이가 있는 도형이 안 되므로 막습니다.
    if n_points < 3:
        # [1차]: n_points < 3일 때 ValueError 예외를 발생시킵니다.
        # [2차]: "최소 3개 이상의 점이 필요합니다"라고 안내합니다.
        raise ValueError("단위 원 근사를 위해 n_points는 3 이상이어야 합니다.")
    # [1차]: 0부터 2*pi까지 n_points개의 균등 각도 배열 theta를 생성합니다.
    # [2차]: 0도부터 360도(2π)까지 100개의 각도 눈금을 만듭니다.
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    # [1차]: cos(theta) 행과 sin(theta) 행을 수직 결합(vstack)해 (2, N) 배열로 반환합니다.
    # [2차]: 첫 번째 줄엔 x좌표, 두 번째 줄엔 y좌표를 나란히 쌓아 원 좌표표를 돌려줍니다.
    return np.vstack([np.cos(theta), np.sin(theta)])


# [1차]: 사선 공식(Shoelace Formula)으로 (2, N) 꼭짓점 배열의 다각형 면적을 계산합니다.
# [2차]: 운동화 끈을 엇갈려 묶는 공식으로 도형이 변환 전후 몇 평방미터인지 정확히 잽니다.
def compute_polygon_area(points: np.ndarray) -> float:
    """사선 공식(Shoelace Formula)을 사용하여 2D 다각형의 면적을 계산합니다.

    수식:
        Area = 0.5 * |sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)|

    Args:
        points: (2, N) 크기의 2D 꼭짓점 좌표 배열.

    Returns:
        계산된 다각형의 양수 면적.

    Raises:
        ValueError: points의 차원이 (2, N) 형태가 아니거나 N < 3일 때 발생.
    """
    # [1차]: 입력 points를 float64형 NumPy 배열로 변환합니다.
    # [2차]: 들어온 점들을 계산 가능한 숫자 표로 정리합니다.
    pts = np.asarray(points, dtype=np.float64)
    # [1차]: 배열이 2차원이며 첫 번째 축이 2(x, y), 점 개수가 3 이상인지 검증합니다.
    # [2차]: (2, N) 모양이 아니거나 점이 3개 미만이면 넓이를 잴 수 없으므로 거부합니다.
    if pts.ndim != 2 or pts.shape[0] != 2 or pts.shape[1] < 3:
        # [1차]: 잘못된 좌표 배열 형태에 대해 ValueError를 발생시킵니다.
        # [2차]: 올바른 (2, N) 좌표표를 넣으라고 알립니다.
        raise ValueError("points는 (2, N) 형태이며 N >= 3이어야 합니다.")
    # [1차]: 0번 행(모든 x좌표)을 슬라이싱합니다.
    # [2차]: x좌표 목록만 따로 꺼냅니다.
    x = pts[0]
    # [1차]: 1번 행(모든 y좌표)을 슬라이싱합니다.
    # [2차]: y좌표 목록만 따로 꺼냅니다.
    y = pts[1]
    # [1차]: np.roll로 x 배열을 왼쪽으로 1칸 순환 이동시켜 다음 점의 x좌표를 얻습니다.
    # [2차]: 옆 사람(다음 꼭짓점)의 x좌표를 내 앞으로 한 칸씩 당겨 짝을 짓습니다.
    x_next = np.roll(x, -1)
    # [1차]: np.roll로 y 배열을 왼쪽으로 1칸 순환 이동시켜 다음 점의 y좌표를 얻습니다.
    # [2차]: 다음 꼭짓점의 y좌표도 내 앞으로 한 칸 당겨 짝을 짓습니다.
    y_next = np.roll(y, -1)
    # [1차]: 교차 곱의 차를 모두 합산한 뒤 절댓값의 0.5배를 float로 반환합니다.
    # [2차]: X자로 엇갈려 곱하고 뺀 값을 전부 더해 반으로 나누어 정확한 넓이를 돌려줍니다.
    return float(0.5 * np.abs(np.sum(x * y_next - x_next * y)))


# [1차]: (2, 2) 행렬의 |det(A)|와 단위 원 변환 전후 면적 비율의 상대 오차를 계산해 반환합니다.
# [2차]: "행렬식이 진짜 도형 넓이의 배율과 같은가?"를 실제 넓이를 재서 1% 이내인지 시험합니다.
def verify_determinant_area(
    matrix: np.ndarray,
    n_points: int = 100,
) -> Dict[str, float]:
    """행렬식 |det(A)|와 변환 전후 단위 원 면적 비율의 일치 여부를 검증합니다.

    수식:
        |det(A)| == Area(A @ Circle) / Area(Circle)

    Args:
        matrix: (2, 2) 변환 행렬.
        n_points: 단위 원 근사에 사용할 점 개수.

    Returns:
        abs_det, area_before, area_after, area_ratio, rel_error 를 담은 딕셔너리.

    Raises:
        ValueError: matrix가 (2, 2) 형태가 아닐 때 발생.
    """
    # [1차]: 입력 matrix를 float64형 NumPy 배열로 변환합니다.
    # [2차]: 변환 행렬을 숫자 표로 정리합니다.
    mat = np.asarray(matrix, dtype=np.float64)
    # [1차]: 행렬 크기가 (2, 2)인지 검사합니다.
    # [2차]: 2D 평면 변환용 2x2 행렬이 아니면 오류를 냅니다.
    if mat.shape != (2, 2):
        # [1차]: (2, 2)가 아닌 경우 ValueError를 발생시킵니다.
        # [2차]: 2x2 행렬만 넣으라고 안내합니다.
        raise ValueError("2D 면적 검증을 위해 matrix는 (2, 2) 형태여야 합니다.")

    # [1차]: n_points개의 점으로 구성된 단위 원 좌표를 생성합니다.
    # [2차]: 기준이 될 동그란 고무판(단위 원)을 만듭니다.
    circle = generate_unit_circle(n_points=n_points)
    # [1차]: 행렬 곱셈(@)으로 단위 원의 모든 점을 한 번에 선형 변환합니다.
    # [2차]: 고무판 위의 모든 점에 행렬을 곱해 찌그러뜨리거나 돌립니다.
    transformed = mat @ circle

    # [1차]: 변환 전 단위 원의 다각형 면적을 계산합니다.
    # [2차]: 변형 전 원래 동그라미의 넓이를 잽니다.
    area_before = compute_polygon_area(circle)
    # [1차]: 변환 후 도형의 다각형 면적을 계산합니다.
    # [2차]: 변형 후 찌그러진 도형의 넓이를 잽니다.
    area_after = compute_polygon_area(transformed)
    # [1차]: 변환 전 대비 변환 후 면적 비율을 계산합니다.
    # [2차]: 넓이가 실제로 몇 배 변했는지(나중 넓이 ÷ 처음 넓이) 계산합니다.
    area_ratio = area_after / area_before
    # [1차]: np.linalg.det로 행렬식을 구한 뒤 절댓값을 취합니다.
    # [2차]: 행렬식 공식이 예측한 넓이 배율(|det(A)|)을 계산합니다.
    abs_det = float(np.abs(np.linalg.det(mat)))

    # [1차]: 행렬식이 0(특이 행렬)인 경우 0 나눗셈을 피하기 위해 절대 오차를 사용합니다.
    # [2차]: 도형이 납작한 선으로 찌그러져 넓이가 0이 된 특수 상황이면 0으로 나누지 않고 차이만 잽니다.
    if abs_det == 0.0:
        # [1차]: 면적 비율과 0의 절대 차이를 계산합니다.
        # [2차]: 절대 오차를 기록합니다.
        rel_error = float(np.abs(area_ratio - abs_det))
    else:
        # [1차]: |area_ratio - |det(A)|| / |det(A)| 상대 오차를 계산합니다.
        # [2차]: 예측 배율과 실제 넓이 배율이 몇 % 차이나는지 계산합니다.
        rel_error = float(np.abs(area_ratio - abs_det) / abs_det)

    # [1차]: 검증 지표 5가지를 담은 딕셔너리를 반환합니다.
    # [2차]: 행렬식, 전후 넓이, 비율, 오차를 한눈에 보기 좋은 보고서 상자에 담아 돌려줍니다.
    return {
        "abs_det": abs_det,
        "area_before": area_before,
        "area_after": area_after,
        "area_ratio": area_ratio,
        "rel_error": rel_error,
    }


# [1차]: 회전, 스케일링, 전단 변환 결과를 개별 및 통합 Figure로 그려 PNG로 저장합니다.
# [2차]: 동그라미와 격자가 회전·확대·기울어짐을 겪으며 어떻게 변하는지 사진 4장으로 인화합니다.
def visualize_matrix_transforms(
    output_dir: str = "outputs",
    theta: float = np.pi / 4.0,
    shear_k: float = 1.0,
) -> List[str]:
    """회전, 스케일링, 전단 변환 전후의 단위 원과 격자를 시각화하여 PNG로 저장합니다.

    Args:
        output_dir: 결과 PNG 파일을 저장할 디렉토리 경로.
        theta: 회전 행렬 각도 (라디안, 기본 pi/4).
        shear_k: 전단 행렬 계수 (기본 1.0).

    Returns:
        생성된 PNG 파일 경로 문자열 목록.
    """
    # [1차]: 출력 경로 Path 객체를 생성합니다.
    # [2차]: 사진을 넣을 보관함 주소를 준비합니다.
    out_path = Path(output_dir)
    # [1차]: 디렉토리가 없으면 생성합니다.
    # [2차]: 보관함 폴더가 없으면 새로 만듭니다.
    out_path.mkdir(parents=True, exist_ok=True)
    # [1차]: 음수 눈금 깨짐을 방지합니다.
    # [2차]: 마이너스 기호가 네모로 깨지지 않게 설정합니다.
    plt.rcParams["axes.unicode_minus"] = False

    # [1차]: 단위 원 좌표를 생성합니다.
    # [2차]: 동그라미 점 100개를 만듭니다.
    circle = generate_unit_circle(n_points=100)
    # [1차]: -1.2 ~ 1.2 범위의 7x7 2D 격자를 생성합니다.
    # [2차]: 공간이 어떻게 휘어지는지 보기 위해 바둑판 점 49개를 만듭니다.
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 7), np.linspace(-1.2, 1.2, 7))
    # [1차]: 격자 좌표를 (2, 49) 형태로 평탄화(ravel)해 수직 결합합니다.
    # [2차]: 49개의 바둑판 점을 행렬 곱셈하기 좋게 두 줄 표로 나열합니다.
    grid_pts = np.vstack([gx.ravel(), gy.ravel()])

    # [1차]: 시각화할 3가지 변환(제목, 행렬, 파일명) 튜플 리스트를 정의합니다.
    # [2차]: 회전, 스케일링, 전단 3가지 실험 목록을 순서대로 적어둡니다.
    transforms = [
        ("Rotation R(pi/4)", rotation_matrix(theta), "la_rotation.png"),
        ("Scaling S(2, 0.5)", scaling_matrix(2.0, 0.5), "la_scaling.png"),
        (f"Shear Sh({shear_k})", shear_matrix(shear_k), "la_shear.png"),
    ]

    # [1차]: 생성된 PNG 경로를 저장할 빈 리스트를 초기화합니다.
    # [2차]: 완성된 사진 주소들을 담을 바구니를 만듭니다.
    saved_paths: List[str] = []

    # [1차]: 3개 변환을 한눈에 비교할 1x3 통합 Figure를 생성합니다.
    # [2차]: 세 가지 변환을 나란히 붙여서 볼 큰 파노라마 도화지를 펼칩니다.
    fig_all, axes_all = plt.subplots(1, 3, figsize=(15, 5))
    # [1차]: 변환 목록을 인덱스와 함께 순회합니다.
    # [2차]: 회전, 스케일링, 전단을 하나씩 차례로 그립니다.
    for idx, (title, mat, fname) in enumerate(transforms):
        # [1차]: 단위 원에 변환 행렬을 곱합니다.
        # [2차]: 동그라미 점들에 행렬을 적용해 새 모양을 만듭니다.
        transformed_circle = mat @ circle
        # [1차]: 격자 점들에 변환 행렬을 곱합니다.
        # [2차]: 바둑판 점들도 같이 변형시킵니다.
        transformed_grid = mat @ grid_pts
        # [1차]: 행렬식과 면적 비율 검증 통계를 계산합니다.
        # [2차]: 이 변환의 넓이 배율(|det|)과 실제 면적 비율을 잽니다.
        stats = verify_determinant_area(mat)

        # [1차]: 개별 변환용 6x6 Figure를 생성합니다.
        # [2차]: 각 변환을 크게 따로 저장할 단독 도화지를 펼칩니다.
        fig_single, ax_s = plt.subplots(figsize=(6, 6))
        # [1차]: 변환 전 격자 점들을 회색 산점도로 그립니다.
        # [2차]: 원래 바둑판 점들을 흐린 회색으로 찍습니다.
        ax_s.scatter(grid_pts[0], grid_pts[1], color="#bdc3c7", s=12, alpha=0.5, label="Original Grid")
        # [1차]: 변환 후 격자 점들을 주황색 산점도로 그립니다.
        # [2차]: 밀려나간 바둑판 점들을 주황색으로 찍습니다.
        ax_s.scatter(transformed_grid[0], transformed_grid[1], color="#f39c12", s=12, alpha=0.5, label="Transformed Grid")
        # [1차]: 변환 전 단위 원을 파란 점선('b--')으로 그립니다.
        # [2차]: 원래 동그라미를 파란 점선으로 그립니다.
        ax_s.plot(circle[0], circle[1], "b--", linewidth=2, label="Unit Circle (Before)")
        # [1차]: 변환 후 도형을 빨간 실선('r-')으로 겹쳐 그립니다.
        # [2차]: 변형된 타원이나 기울어진 모양을 빨간 실선으로 겹쳐 그립니다.
        ax_s.plot(transformed_circle[0], transformed_circle[1], "r-", linewidth=2.5, label="Transformed (After)")
        # [1차]: x축 기준선(y=0)을 그립니다.
        # [2차]: 가로 중심선을 긋습니다.
        ax_s.axhline(0, color="gray", linewidth=0.6)
        # [1차]: y축 기준선(x=0)을 그립니다.
        # [2차]: 세로 중심선을 긋습니다.
        ax_s.axvline(0, color="gray", linewidth=0.6)
        # [1차]: 가로세로 비율을 1:1로 고정합니다.
        # [2차]: 원이 찌그러져 보이지 않도록 비율을 1:1로 맞춥니다.
        ax_s.set_aspect("equal", adjustable="box")
        # [1차]: 점선 격자를 표시합니다.
        # [2차]: 모눈종이를 켭니다.
        ax_s.grid(True, linestyle=":", alpha=0.6)
        # [1차]: 행렬식과 면적 비율을 포함한 제목을 설정합니다.
        # [2차]: 제목에 |det(A)| 값과 실제 면적 비율을 적어 증명합니다.
        ax_s.set_title(
            f"{title}\n|det(A)|={stats['abs_det']:.4f}, Area Ratio={stats['area_ratio']:.4f}"
        )
        # [1차]: 우측 상단에 범례를 표시합니다.
        # [2차]: 설명표를 붙입니다.
        ax_s.legend(loc="upper right", fontsize=8)
        # [1차]: 개별 파일 저장 경로를 조합합니다.
        # [2차]: 단독 사진 파일 이름을 정합니다.
        single_path = out_path / fname
        # [1차]: 여백을 자동 정리합니다.
        # [2차]: 글자가 잘리지 않게 여백을 맞춥니다.
        fig_single.tight_layout()
        # [1차]: 150 DPI PNG로 저장합니다.
        # [2차]: 사진 파일로 인화합니다.
        fig_single.savefig(single_path, dpi=150)
        # [1차]: 개별 Figure를 닫아 메모리를 해제합니다.
        # [2차]: 다 쓴 단독 도화지를 치웁니다.
        plt.close(fig_single)
        # [1차]: 저장 경로를 리스트에 추가합니다.
        # [2차]: 사진 주소를 바구니에 담습니다.
        saved_paths.append(str(single_path))

        # [1차]: 통합 Figure의 idx번째 서브플롯을 선택합니다.
        # [2차]: 파노라마 도화지의 해당 칸에 똑같이 그릴 준비를 합니다.
        ax_a = axes_all[idx]
        # [1차]: 변환 전 단위 원을 파란 점선으로 그립니다.
        # [2차]: 원래 동그라미를 그립니다.
        ax_a.plot(circle[0], circle[1], "b--", linewidth=2, label="Before")
        # [1차]: 변환 후 도형을 빨간 실선으로 그립니다.
        # [2차]: 변형된 도형을 그립니다.
        ax_a.plot(transformed_circle[0], transformed_circle[1], "r-", linewidth=2.5, label="After")
        # [1차]: x축 중심선을 그립니다.
        # [2차]: 가로 중심선을 긋습니다.
        ax_a.axhline(0, color="gray", linewidth=0.6)
        # [1차]: y축 중심선을 그립니다.
        # [2차]: 세로 중심선을 긋습니다.
        ax_a.axvline(0, color="gray", linewidth=0.6)
        # [1차]: 1:1 비율을 고정합니다.
        # [2차]: 비율을 1:1로 맞춥니다.
        ax_a.set_aspect("equal", adjustable="box")
        # [1차]: 격자를 켭니다.
        # [2차]: 모눈종이를 켭니다.
        ax_a.grid(True, linestyle=":", alpha=0.6)
        # [1차]: 서브플롯 제목에 요약 수치를 표시합니다.
        # [2차]: 각 칸 위에 행렬식과 비율을 적습니다.
        ax_a.set_title(f"{title}\n|det|={stats['abs_det']:.3f}, Ratio={stats['area_ratio']:.3f}")
        # [1차]: 범례를 표시합니다.
        # [2차]: 설명표를 붙입니다.
        ax_a.legend(loc="upper right", fontsize=8)

    # [1차]: 통합 이미지 저장 경로를 설정합니다.
    # [2차]: 파노라마 사진 이름을 정합니다.
    all_path = out_path / "la_transforms.png"
    # [1차]: 통합 Figure 상단 제목을 설정합니다.
    # [2차]: 맨 위에 큰 제목을 적습니다.
    fig_all.suptitle("2D Linear Transformations & Determinant Area Ratio", fontsize=13, fontweight="bold")
    # [1차]: 여백을 정리합니다.
    # [2차]: 여백을 맞춥니다.
    fig_all.tight_layout()
    # [1차]: 통합 PNG 파일을 저장합니다.
    # [2차]: 파노라마 사진을 인화합니다.
    fig_all.savefig(all_path, dpi=150)
    # [1차]: 통합 Figure를 닫습니다.
    # [2차]: 도화지를 치웁니다.
    plt.close(fig_all)
    # [1차]: 통합 이미지 경로를 리스트 맨 앞에 삽입합니다.
    # [2차]: 파노라마 사진 주소를 목록 맨 앞에 추가합니다.
    saved_paths.insert(0, str(all_path))
    # [1차]: 저장된 모든 PNG 경로 리스트를 반환합니다.
    # [2차]: 완성된 사진 주소 목록을 돌려줍니다.
    return saved_paths


# [1차]: Power Iteration으로 정방 행렬의 최대 고유값과 대응 단위 고유벡터를 반복 계산합니다.
# [2차]: 행렬을 계속 곱하고 길이를 1로 줄이는 과정을 반복해 '가장 세게 늘어나는 주력 방향과 배율'을 찾습니다.
def power_iteration(
    matrix: np.ndarray,
    num_iters: int = 1000,
    tol: float = 1e-9,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Power Iteration 알고리즘으로 정방 행렬의 최대 고유값과 대응 고유벡터를 계산합니다.

    수식:
        b_{k+1} = (A @ b_k) / ||A @ b_k||_2
        lambda_max = (b_k^T @ A @ b_k) / (b_k^T @ b_k)  (Rayleigh quotient)

    Args:
        matrix: (N, N) 정방 행렬.
        num_iters: 최대 반복 횟수.
        tol: 수렴 판정 임계값.
        seed: 초기 벡터 생성을 위한 난수 시드.

    Returns:
        (최대 고유값, 단위 고유벡터) 튜플.

    Raises:
        ValueError: 정방 행렬이 아니거나, num_iters <= 0 이거나, 영행렬인 경우.
    """
    # [1차]: 입력 matrix를 float64형 NumPy 배열로 변환합니다.
    # [2차]: 행렬을 계산 가능한 숫자 표로 정리합니다.
    mat = np.asarray(matrix, dtype=np.float64)
    # [1차]: 비어 있지 않은 2D 정방 행렬(행 수 == 열 수)인지 검사합니다.
    # [2차]: 가로·세로 길이가 똑같은 정사각형 행렬이 아니면 거부합니다.
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        # [1차]: 정방 행렬이 아닐 때 ValueError를 발생시킵니다.
        # [2차]: 정사각형 행렬만 넣으라고 알립니다.
        raise ValueError("Power Iteration에는 비어 있지 않은 정방 행렬이 필요합니다.")
    # [1차]: 최대 반복 횟수가 1 이상인지 확인합니다.
    # [2차]: 0번 이하로 반복할 순 없으니 막습니다.
    if num_iters <= 0:
        # [1차]: num_iters <= 0이면 ValueError를 발생시킵니다.
        # [2차]: 최소 1회 이상 반복하라고 안내합니다.
        raise ValueError("num_iters는 1 이상이어야 합니다.")
    # [1차]: 모든 원소가 0인 영행렬인지 검사합니다.
    # [2차]: 전부 0뿐인 행렬은 곱하면 화살표가 사라지므로 미리 차단합니다.
    if np.allclose(mat, 0.0):
        # [1차]: 영행렬 입력 시 ValueError를 발생시킵니다.
        # [2차]: "영행렬에는 적용할 수 없습니다"라고 알립니다.
        raise ValueError("영행렬(Zero Matrix)에는 Power Iteration을 적용할 수 없습니다.")

    # [1차]: 실험 재현성을 위해 난수 시드를 고정합니다.
    # [2차]: 언제 실행해도 똑같은 출발 화살표가 나오도록 주사위를 고정합니다.
    np.random.seed(seed)
    # [1차]: 행렬의 차원 크기 N을 추출합니다.
    # [2차]: 화살표의 칸 수(차원)를 확인합니다.
    n = mat.shape[0]
    # [1차]: [0, 1) 범위의 난수로 초기 벡터 b를 생성합니다.
    # [2차]: 아무 방향이나 가리키는 임시 화살표를 하나 뽑습니다.
    b = np.random.rand(n)
    # [1차]: 초기 벡터의 L2 노름(길이)을 계산합니다.
    # [2차]: 화살표 길이를 잽니다.
    norm_b = np.linalg.norm(b)
    # [1차]: 우연히 영벡터가 나왔을 경우 [1, 1, ...] 벡터로 대체하는 방어 로직입니다.
    # [2차]: 혹시 길이가 0짜리 화살표가 뽑히면 안전한 [1, 1, ...] 화살표로 바꿉니다.
    if norm_b == 0.0:
        # [1차]: 모든 성분이 1인 벡터를 생성합니다.
        # [2차]: 안전한 화살표를 새로 만듭니다.
        b = np.ones(n, dtype=np.float64)
        # [1차]: 새 벡터의 길이를 다시 계산합니다.
        # [2차]: 길이를 다시 잽니다.
        norm_b = np.linalg.norm(b)
    # [1차]: 초기 벡터를 단위 벡터(||b||=1)로 정규화합니다.
    # [2차]: 길이를 딱 1로 줄여 표준 화살표로 만듭니다.
    b = b / norm_b

    # [1차]: 이전 스텝 고유값 비교를 위한 초기값 0.0을 설정합니다.
    # [2차]: 직전 배율 기록을 0으로 초기화합니다.
    eigenvalue = 0.0
    # [1차]: 최대 num_iters회 Power Iteration 루프를 실행합니다.
    # [2차]: 화살표가 주력 방향에 딱 멈출 때까지 반복합니다.
    for _ in range(num_iters):
        # [1차]: 행렬-벡터 곱 A @ b를 수행합니다.
        # [2차]: 화살표에 행렬 바람을 한 번 불어줍니다.
        b_next = mat @ b
        # [1차]: 변환된 벡터의 L2 노름을 계산합니다.
        # [2차]: 늘어난 화살표의 길이를 잽니다.
        norm_next = np.linalg.norm(b_next)
        # [1차]: 변환 후 영벡터가 되면 0 나눗셈을 막기 위해 루프를 종료합니다.
        # [2차]: 화살표 길이가 0으로 쪼그라들면 안전하게 멈춥니다.
        if norm_next == 0.0:
            break
        # [1차]: 길이를 1로 정규화해 다음 단위 벡터 b_next를 얻습니다.
        # [2차]: 숫자가 폭발하지 않게 다시 길이 1짜리 표준 화살표로 줄입니다.
        b_next = b_next / norm_next
        # [1차]: Rayleigh Quotient (b^T @ A @ b)로 현재 고유값을 추정합니다.
        # [2차]: 이 화살표가 원래 방향으로 몇 배 늘어났는지 배율(고유값)을 잽니다.
        new_eigenvalue = float(b_next.T @ mat @ b_next)

        # [1차]: 고유값 변화와 고유벡터 변화가 모두 tol 미만이면 수렴으로 판정해 조기 종료합니다.
        # [2차]: 화살표 방향과 배율이 거의 변하지 않고 딱 멈추면 정답을 찾은 것이므로 종료합니다.
        if np.abs(new_eigenvalue - eigenvalue) < tol and np.linalg.norm(b_next - b) < tol:
            # [1차]: 최종 벡터와 고유값을 갱신하고 루프를 탈출합니다.
            # [2차]: 최종 값을 확정 짓고 반복문을 빠져나갑니다.
            b = b_next
            eigenvalue = new_eigenvalue
            break

        # [1차]: 다음 반복을 위해 현재 벡터와 고유값을 갱신합니다.
        # [2차]: 방금 구한 화살표와 배율을 현재 값으로 업데이트합니다.
        b = b_next
        eigenvalue = new_eigenvalue

    # [1차]: 수렴된 최대 고유값과 단위 고유벡터를 튜플로 반환합니다.
    # [2차]: 찾은 최대 배율(고유값)과 주력 방향(고유벡터)을 함께 돌려줍니다.
    return eigenvalue, b


# [1차]: 외부 파일 없이 재현 가능한 64x64 이하 Grayscale 테스트 이미지를 생성합니다.
# [2차]: 인터넷에서 사진을 받지 않고도 SVD 압축 실험을 할 수 있는 64x64 흑백 그림을 직접 그립니다.
def create_sample_grayscale_image(size: int = 64) -> np.ndarray:
    """재현 가능한 64x64 이하 Grayscale 테스트 이미지를 생성합니다.

    Args:
        size: 이미지 한 변의 픽셀 크기 (1 ~ 64).

    Returns:
        0~255 범위의 float64 (size, size) Grayscale 이미지 배열.

    Raises:
        ValueError: size가 1 미만이거나 64 초과일 때 발생.
    """
    # [1차]: 과제 제약 조건(64x64 이하)을 검증합니다.
    # [2차]: 크기가 1 미만이거나 64를 넘으면 규격 위반이므로 막습니다.
    if size < 1 or size > 64:
        # [1차]: 범위를 벗어난 size에 대해 ValueError를 발생시킵니다.
        # [2차]: "64x64 이하만 가능합니다"라고 알립니다.
        raise ValueError("이미지는 64x64 이하(1 <= size <= 64)여야 합니다.")

    # [1차]: 재현성을 위해 시드 42를 고정합니다.
    # [2차]: 항상 동일한 테스트 그림이 나오도록 시드를 맞춥니다.
    np.random.seed(42)
    # [1차]: 브로드캐스팅 가능한 열린 격자 인덱스(ogrid)를 생성합니다.
    # [2차]: 세로 줄번호(y)와 가로 칸번호(x) 자를 준비합니다.
    y_idx, x_idx = np.ogrid[:size, :size]
    # [1차]: 이미지 중심 좌표를 계산합니다.
    # [2차]: 캔버스 한가운데 좌표를 찾습니다.
    center = size / 2.0
    # [1차]: 중앙 원 패턴의 반지름을 계산합니다.
    # [2차]: 가운데 그릴 동그라미의 반지름을 정합니다.
    radius = size / 3.0

    # [1차]: 대각선 방향 그라데이션(0~140 밝기)을 생성합니다.
    # [2차]: 왼쪽 위에서 오른쪽 아래로 서서히 밝아지는 배경을 칠합니다.
    gradient = (x_idx + y_idx) / (2.0 * size) * 140.0
    # [1차]: 중심으로부터 거리가 radius 이하인 픽셀들의 불리언 마스크를 만듭니다.
    # [2차]: 가운데 동그라미 영역에 해당하는 픽셀만 골라내는 스텐실(틀)을 만듭니다.
    circle_mask = ((x_idx - center) ** 2 + (y_idx - center) ** 2) <= radius ** 2
    # [1차]: 8픽셀 간격의 체커보드(바둑판) 명암 패턴을 생성합니다.
    # [2차]: 8칸마다 흑백이 교차하는 체스판 무늬를 만듭니다.
    checker = (((x_idx // 8) + (y_idx // 8)) % 2) * 55.0

    # [1차]: 배경 그라데이션과 체커보드 패턴을 합칩니다.
    # [2차]: 배경 위에 체스판 무늬를 은은하게 덧칠합니다.
    img = gradient + checker
    # [1차]: 원형 마스크 영역의 밝기를 반전시켜 뚜렷한 원형 피사체를 그립니다.
    # [2차]: 가운데 동그라미 안쪽 색을 반전시켜 또렷한 도형을 넣습니다.
    img[circle_mask] = 220.0 - 0.5 * img[circle_mask]
    # [1차]: 픽셀 값을 0~255 범위로 클리핑해 float64 배열로 반환합니다.
    # [2차]: 밝기가 0~255 범위를 벗어나지 않게 다듬어 완성된 흑백 그림을 돌려줍니다.
    return np.clip(img, 0.0, 255.0).astype(np.float64)


# [1차]: np.linalg.svd로 이미지를 분해한 뒤 상위 k개 특이값만으로 이미지를 복원하고 오차를 계산합니다.
# [2차]: 그림에서 가장 중요한 핵심 붓터치 k개만 남기고 나머지는 버려서 압축한 뒤 다시 복원해봅니다.
def svd_compress_reconstruct(
    image: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, float, float]:
    """SVD를 사용하여 Grayscale 이미지를 상위 k개 특이값으로 압축 및 복원합니다.

    수식:
        A = U @ Sigma @ V^T
        A_k = U[:, :k] @ diag(S[:k]) @ V^T[:k, :]

    Args:
        image: (H, W) 2D Grayscale 이미지 배열.
        k: 유지할 특이값 개수 (1 이상).

    Returns:
        (복원 이미지, 압축률(원본 대비 저장 파라미터 비율), 복원 MSE) 튜플.

    Raises:
        ValueError: image가 2D가 아니거나 k <= 0일 때 발생.
    """
    # [1차]: 입력 image를 float64형 NumPy 배열로 변환합니다.
    # [2차]: 그림 데이터를 숫자 표로 정리합니다.
    img = np.asarray(image, dtype=np.float64)
    # [1차]: 이미지가 2차원 흑백 배열인지 검사합니다.
    # [2차]: 가로×세로 2차원 흑백 사진이 아니면 거부합니다.
    if img.ndim != 2:
        # [1차]: 2D가 아닐 때 ValueError를 발생시킵니다.
        # [2차]: 2D 흑백 이미지로 넣으라고 알립니다.
        raise ValueError("SVD 압축을 위해 image는 2D (H, W) 배열이어야 합니다.")
    # [1차]: 유지할 특이값 개수 k가 1 이상인지 검증합니다.
    # [2차]: 0개 이하의 붓터치로는 그림을 그릴 수 없으므로 막습니다.
    if k <= 0:
        # [1차]: k <= 0일 때 ValueError를 발생시킵니다.
        # [2차]: 최소 1개 이상의 k를 입력하라고 알립니다.
        raise ValueError("유지할 특이값 개수 k는 1 이상이어야 합니다.")

    # [1차]: 이미지의 높이(h)와 너비(w)를 추출합니다.
    # [2차]: 캔버스의 세로·가로 픽셀 수를 확인합니다.
    h, w = img.shape
    # [1차]: 축소형 SVD(full_matrices=False)로 U, 특이값 S, V^T를 분해합니다.
    # [2차]: 그림을 '가로 패턴(u)', '중요도 점수(s)', '세로 패턴(vt)' 세 서랍으로 쪼갭니다.
    u, s, vt = np.linalg.svd(img, full_matrices=False)
    # [1차]: 요청한 k가 최대 특이값 개수 len(s)를 넘지 않도록 보정합니다.
    # [2차]: 전체가 64개뿐인데 100개를 요청하면 있는 64개 전부만 쓰도록 안전하게 맞춥니다.
    k_eff = min(k, len(s))

    # [1차]: 상위 k_eff개 성분만 슬라이싱·브로드캐스팅 곱셈해 근사 행렬을 복원합니다.
    # [2차]: 점수 높은 상위 k개 붓터치만 골라 다시 캔버스에 겹쳐 그림을 복원합니다.
    reconstructed = (u[:, :k_eff] * s[:k_eff]) @ vt[:k_eff, :]
    # [1차]: 복원된 픽셀 값을 유효 범위 [0, 255]로 클리핑합니다.
    # [2차]: 밝기가 0~255를 살짝 넘은 픽셀을 정상 범위로 다듬습니다.
    reconstructed = np.clip(reconstructed, 0.0, 255.0)

    # [1차]: 원본 이미지의 총 픽셀 수를 계산합니다.
    # [2차]: 원본 사진이 차지하는 숫자 개수(예: 64×64 = 4096개)를 셉니다.
    original_params = float(h * w)
    # [1차]: Rank-k 저장에 필요한 파라미터 수 k*(h + w + 1)를 계산합니다.
    # [2차]: 압축 후 저장해야 할 핵심 숫자 개수를 셉니다.
    compressed_params = float(k_eff * (h + w + 1))
    # [1차]: 원본 대비 압축 파라미터 비율을 계산합니다.
    # [2차]: 원본 대비 몇 배 크기인지 저장 비율을 구합니다.
    storage_ratio = compressed_params / original_params
    # [1차]: 원본과 복원 이미지 간의 평균 제곱 오차(MSE)를 계산합니다.
    # [2차]: 원본과 복원 그림이 픽셀당 평균 얼마나 차이 나는지(화질 손실)를 잽니다.
    mse = float(np.mean((img - reconstructed) ** 2))
    # [1차]: (복원 이미지, 저장 비율, MSE) 튜플을 반환합니다.
    # [2차]: 복원된 그림, 압축 비율, 화질 오차 세 가지를 한꺼번에 돌려줍니다.
    return reconstructed, storage_ratio, mse


# [1차]: 원본과 k=10, 50, 100 복원 결과를 1행 서브플롯으로 비교해 PNG로 저장합니다.
# [2차]: 원본 사진 옆에 k=10, 50, 100으로 복원한 사진을 나란히 붙여 화질 차이를 보여줍니다.
def visualize_svd_compression(
    image: Optional[np.ndarray] = None,
    ks: Sequence[int] = (10, 50, 100),
    output_dir: str = "outputs",
) -> str:
    """SVD 이미지 압축 및 복원 결과(k=10, 50, 100)를 하나의 Figure에 비교 출력합니다.

    Args:
        image: (H, W) Grayscale 이미지 (None이면 64x64 기본 이미지 생성).
        ks: 비교할 k 값 시퀀스.
        output_dir: PNG 저장 디렉토리.

    Returns:
        저장된 PNG 파일 경로 문자열.
    """
    # [1차]: 출력 디렉토리 Path 객체를 생성합니다.
    # [2차]: 저장 폴더 주소를 준비합니다.
    out_path = Path(output_dir)
    # [1차]: 폴더가 없으면 생성합니다.
    # [2차]: 보관함 폴더가 없으면 새로 만듭니다.
    out_path.mkdir(parents=True, exist_ok=True)

    # [1차]: image가 None이면 기본 64x64 이미지를 생성하고, 아니면 float64 배열로 변환합니다.
    # [2차]: 외부 그림이 없으면 기본 64x64 테스트 그림을 만들어 씁니다.
    img = create_sample_grayscale_image(64) if image is None else np.asarray(image, dtype=np.float64)
    # [1차]: 원본(1) + 비교할 k 개수만큼 전체 서브플롯 수를 계산합니다.
    # [2차]: 원본 1칸 + 압축 버전 3칸, 총 4칸짜리 액자를 준비합니다.
    n_plots = 1 + len(ks)
    # [1차]: 1행 n_plots열 크기의 Figure와 Axes 배열을 생성합니다.
    # [2차]: 4장의 사진을 가로로 나란히 걸 액자를 펼칩니다.
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4.5))

    # [1차]: 첫 번째 칸에 원본 이미지를 흑백(gray) 컬러맵으로 출력합니다.
    # [2차]: 맨 왼쪽 첫 번째 액자에 원본 사진을 걸어둡니다.
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    # [1차]: 원본 이미지의 해상도를 제목으로 표시합니다.
    # [2차]: "원본 (64x64)" 이름표를 붙입니다.
    axes[0].set_title(f"Original ({img.shape[0]}x{img.shape[1]})")
    # [1차]: 이미지 출력 시 불필요한 축 눈금을 숨깁니다.
    # [2차]: 사진 테두리의 숫자 눈금을 깔끔히 지웁니다.
    axes[0].axis("off")

    # [1차]: k 시퀀스를 1번 인덱스부터 순회하며 압축·복원 이미지를 그립니다.
    # [2차]: k=10, 50, 100 버전을 순서대로 액자에 채웁니다.
    for idx, k in enumerate(ks, start=1):
        # [1차]: 해당 k로 SVD 압축 및 복원을 수행합니다.
        # [2차]: 상위 k개 붓터치만으로 그림을 복원하고 오차를 잽니다.
        recon, ratio, mse = svd_compress_reconstruct(img, k)
        # [1차]: idx번째 칸에 복원 이미지를 출력합니다.
        # [2차]: 해당 액자에 복원된 사진을 겁니다.
        axes[idx].imshow(recon, cmap="gray", vmin=0, vmax=255)
        # [1차]: k값, MSE, 저장 비율을 제목에 표시합니다.
        # [2차]: "k=10일 때 오차(MSE)는 얼마"라고 이름표를 붙입니다.
        axes[idx].set_title(f"SVD k={k}\nMSE={mse:.2f}, Ratio={ratio:.2f}")
        # [1차]: 축 눈금을 숨깁니다.
        # [2차]: 테두리 눈금을 지웁니다.
        axes[idx].axis("off")

    # [1차]: 저장 경로(output_dir/la_svd_compression.png)를 조합합니다.
    # [2차]: 완성된 비교 사진의 파일 이름을 정합니다.
    save_path = out_path / "la_svd_compression.png"
    # [1차]: 전체 Figure 상단 제목을 설정합니다.
    # [2차]: 맨 위에 큰 제목 간판을 달아줍니다.
    fig.suptitle("SVD Grayscale Image Compression (k=10, 50, 100)", fontsize=13, fontweight="bold")
    # [1차]: 서브플롯 간 여백을 자동 조정합니다.
    # [2차]: 사진들이 겹치지 않게 간격을 맞춥니다.
    fig.tight_layout()
    # [1차]: 150 DPI PNG로 저장합니다.
    # [2차]: 사진 파일로 인화합니다.
    fig.savefig(save_path, dpi=150)
    # [1차]: Figure를 닫아 메모리를 해제합니다.
    # [2차]: 다 그린 도화지를 치웁니다.
    plt.close(fig)
    # [1차]: 저장된 파일 경로 문자열을 반환합니다.
    # [2차]: 사진 주소를 알려주며 마칩니다.
    return str(save_path)
