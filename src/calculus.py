"""미적분(Calculus) 수치 미분 및 Gradient 시각화 모듈.

포함 기능:
- 중심차분법(Central Difference) 기반 1변수 수치 미분 함수
- 중심차분법 기반 다변수 함수 Gradient 벡터 계산 함수
- f(x, y) = x^2 + y^2 등고선 및 Gradient 화살표 수직성 시각화
"""

# [1차]: 파일 시스템 경로를 객체지향적으로 다루기 위해 pathlib의 Path 클래스를 임포트합니다.
# [2차]: 문자열 경로("outputs/...") 대신 폴더와 파일을 안전하게 조립하는 만능 주소록 도구를 꺼냅니다.
from pathlib import Path
# [1차]: 함수 타입 힌트(Callable)를 명시하기 위해 typing 모듈에서 임포트합니다.
# [2차]: "이 자리에는 숫자나 글자가 아니라 실행할 수 있는 함수가 들어온다"는 표지판을 준비합니다.
from typing import Callable

# [1차]: Matplotlib 핵심 패키지를 임포트합니다.
# [2차]: 수학 그래프를 그릴 도화지와 붓 세트를 책상 위에 올려둡니다.
import matplotlib
# [1차]: GUI 창을 띄우지 않고 파일로만 렌더링하도록 백엔드를 'Agg'로 설정합니다.
# [2차]: 모니터 화면에 창을 팝업으로 띄우지 않고 조용히 사진 파일(.png)로만 인화하도록 설정합니다.
matplotlib.use("Agg")
# [1차]: 그래프 생성을 위한 pyplot 서브모듈을 plt 별칭으로 임포트합니다.
# [2차]: 도화지에 선을 긋고 제목을 적는 간편한 그림판 리모컨을 plt라는 짧은 이름으로 부릅니다.
import matplotlib.pyplot as plt
# [1차]: 수치 연산 라이브러리 NumPy를 np 별칭으로 임포트합니다.
# [2차]: 벡터와 행렬 계산을 순식간에 처리하는 공학용 계산기를 np라는 이름으로 불러옵니다.
import numpy as np


# [1차]: 스칼라 함수 func, 지점 x, 미소 간격 h(기본 1e-5)를 받아 float를 반환하는 함수를 정의합니다.
# [2차]: 어떤 곡선이든 x 위치에서 앞뒤로 살짝 움직여 기울기를 재는 '만능 경사 측정기'를 만듭니다.
def numerical_derivative(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """중심차분법(Central Difference)으로 1변수 함수의 수치 미분 값을 계산합니다.

    수식:
        f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)

    Args:
        func: 미분할 스칼라 함수 f(x).
        x: 미분 계수를 구할 지점.
        h: 미소 변화량 (양수, 기본값 1e-5).

    Returns:
        x에서의 근사 미분 계수 f'(x).

    Raises:
        ValueError: h <= 0 일 때 발생.
    """
    # [1차]: 보폭 h가 0 이하인지 검사하여 0 나눗셈 및 부호 반전을 방지합니다.
    # [2차]: 보폭이 0이거나 마이너스면 경사를 잴 수 없으므로 입구에서 즉시 경고벨을 울립니다.
    if h <= 0.0:
        # [1차]: 잘못된 인자 입력에 대해 ValueError 예외를 발생시킵니다.
        # [2차]: "h는 반드시 양수여야 합니다!"라고 안내문을 띄우며 작업을 중단합니다.
        raise ValueError("미소 변화량 h는 0보다 큰 양수여야 합니다.")
    # [1차]: 입력 x를 배정밀도 실수(float)로 명시적 형변환합니다.
    # [2차]: 정수 3이 들어와도 소수점 계산이 매끄럽도록 실수 3.0으로 깔끔히 다듬습니다.
    x_val = float(x)
    # [1차]: (f(x+h) - f(x-h)) / (2*h) 중심차분 공식을 계산해 float로 반환합니다.
    # [2차]: 현재 위치에서 앞(x+h)과 뒤(x-h)의 높이 차이를 양쪽 보폭 합(2h)으로 나누어 정확한 기울기를 구합니다.
    return float((func(x_val + h) - func(x_val - h)) / (2.0 * h))


# [1차]: 다변수 함수 func, 1D 좌표 배열 point, 미소 간격 h를 받아 Gradient 벡터를 반환합니다.
# [2차]: 산의 2차원·3차원 좌표 위에서 동서남북 어느 쪽이 가장 가파른 오르막인지 화살표를 구하는 함수입니다.
def numerical_gradient(
    func: Callable[[np.ndarray], float],
    point: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """중심차분법으로 다변수 스칼라 함수의 Gradient 벡터를 계산합니다.

    수식:
        grad_i f(p) ≈ (f(p + h * e_i) - f(p - h * e_i)) / (2 * h)

    Args:
        func: D차원 벡터를 입력받아 스칼라를 반환하는 함수.
        point: Gradient를 계산할 1D 좌표 배열 (D,).
        h: 미소 변화량 (양수, 기본값 1e-5).

    Returns:
        입력 point와 동일한 shape을 가진 Gradient 벡터.

    Raises:
        ValueError: h <= 0 이거나 point가 비어 있는 경우.
    """
    # [1차]: h가 양수인지 검증하는 가드 조건문입니다.
    # [2차]: 보폭이 0 이하이면 측정이 불가능하므로 먼저 차단합니다.
    if h <= 0.0:
        # [1차]: h <= 0일 때 ValueError를 발생시킵니다.
        # [2차]: 잘못된 보폭 입력 시 즉시 오류 메시지를 전달합니다.
        raise ValueError("미소 변화량 h는 0보다 큰 양수여야 합니다.")
    # [1차]: 입력 point를 float64형 NumPy 배열로 변환하고 원본 보호를 위해 복사(.copy())합니다.
    # [2차]: 원본 좌표 지도가 훼손되지 않도록 똑같은 사본을 한 장 복사해둡니다.
    p = np.asarray(point, dtype=np.float64).copy()
    # [1차]: p가 1차원 벡터가 아니거나 원소 수가 0개인 경우를 검사합니다.
    # [2차]: 좌표가 아예 비어 있거나 1줄짜리 벡터가 아니면 계산을 거부합니다.
    if p.ndim != 1 or p.size == 0:
        # [1차]: 유효하지 않은 벡터 형태에 대해 ValueError를 발생시킵니다.
        # [2차]: "비어 있지 않은 1차원 좌표만 넣으세요"라고 알려줍니다.
        raise ValueError("point는 비어 있지 않은 1차원 벡터여야 합니다.")

    # [1차]: p와 동일한 크기(shape)의 0으로 채워진 float64 배열 grad를 생성합니다.
    # [2차]: 각 방향별 기울기를 적어 넣을 빈 성적표(0으로 초기화)를 준비합니다.
    grad = np.zeros_like(p, dtype=np.float64)
    # [1차]: 0부터 벡터 차원 수(p.size) - 1까지 인덱스를 순회합니다.
    # [2차]: x축, y축 등 각 방향을 한 번씩 순서대로 점검합니다.
    for i in range(p.size):
        # [1차]: i번째 좌표의 원래 값을 임시 변수 orig에 백업합니다.
        # [2차]: 발을 옮기기 전 현재 서 있는 위치를 분필로 표시해둡니다.
        orig = p[i]
        # [1차]: i번째 원소를 +h만큼 이동시킵니다.
        # [2차]: 해당 축으로 아주 조금(+h) 앞으로 발을 내딛습니다.
        p[i] = orig + h
        # [1차]: 이동된 좌표에서 함수값을 평가하여 float로 저장합니다.
        # [2차]: 앞쪽 지점의 산 높이를 잽니다.
        f_plus = float(func(p))
        # [1차]: i번째 원소를 기준점으로부터 -h만큼 이동시킵니다.
        # [2차]: 이번엔 뒤쪽(-h)으로 똑같은 보폭만큼 발을 내딛습니다.
        p[i] = orig - h
        # [1차]: 음의 방향 좌표에서 함수값을 평가하여 float로 저장합니다.
        # [2차]: 뒤쪽 지점의 산 높이를 잽니다.
        f_minus = float(func(p))
        # [1차]: 다음 축 계산에 영향을 주지 않도록 i번째 원소를 원래 값으로 복원합니다.
        # [2차]: 다음 축을 재기 전에 반드시 원래 제자리로 돌아옵니다.
        p[i] = orig
        # [1차]: 중심차분 공식으로 i번째 편미분 값을 계산해 grad[i]에 저장합니다.
        # [2차]: (앞 높이 - 뒤 높이) ÷ 2h로 이 방향의 기울기를 성적표에 기록합니다.
        grad[i] = (f_plus - f_minus) / (2.0 * h)
    # [1차]: 모든 축의 편미분이 채워진 Gradient 벡터 grad를 반환합니다.
    # [2차]: 각 축의 기울기를 모은 최종 '가장 가파른 오르막 화살표'를 돌려줍니다.
    return grad


# [1차]: 2차원 좌표 배열 point를 받아 x^2 + y^2 스칼라 값을 반환하는 테스트 함수입니다.
# [2차]: 원점이 가장 낮고 바깥으로 갈수록 둥글게 높아지는 밥그릇 모양 함수입니다.
def quadratic_2d(point: np.ndarray) -> float:
    """테스트용 2변수 2차 함수 f(x, y) = x^2 + y^2."""
    # [1차]: 첫 번째 원소의 제곱과 두 번째 원소의 제곱을 더해 float로 반환합니다.
    # [2차]: x좌표 제곱과 y좌표 제곱을 합쳐 현재 위치의 높이를 계산합니다.
    return float(point[0] ** 2 + point[1] ** 2)


# [1차]: 등고선 그래프와 Gradient 화살표를 그려 output_dir에 PNG로 저장하는 함수입니다.
# [2차]: 등고선 지도 위에 화살표를 그려서 기울기가 등고선과 90도 수직임을 눈으로 확인시켜 줍니다.
def visualize_gradient_field(output_dir: str = "outputs") -> str:
    """f(x, y) = x^2 + y^2 등고선과 등고선 위 여러 점의 Gradient 화살표를 시각화합니다.

    등고선과 Gradient 벡터가 항상 직교(수직)함을 확인하기 위해 축 비율을 1:1로 고정합니다.

    Args:
        output_dir: 결과 PNG 파일을 저장할 디렉토리 경로.

    Returns:
        저장된 PNG 파일 경로 문자열.
    """
    # [1차]: 출력 디렉토리 문자열을 Path 객체로 변환합니다.
    # [2차]: 그림을 저장할 폴더 주소를 Path 객체로 만듭니다.
    out_path = Path(output_dir)
    # [1차]: 폴더가 없으면 상위 폴더까지 포함해 생성하고, 이미 있어도 에러를 내지 않습니다.
    # [2차]: 보관함 폴더가 없으면 새로 만들고, 이미 있으면 그대로 씁니다.
    out_path.mkdir(parents=True, exist_ok=True)
    # [1차]: 그래프에서 마이너스(-) 기호가 깨지지 않도록 유니코드 마이너스 옵션을 끕니다.
    # [2차]: 음수 눈금이 네모(□)로 깨지지 않게 안전장치를 겁니다.
    plt.rcParams["axes.unicode_minus"] = False

    # [1차]: -3.0부터 3.0까지 200개의 균등 간격 좌표를 생성합니다.
    # [2차]: 가로·세로 눈금을 촘촘하게 200칸으로 나눕니다.
    grid_vals = np.linspace(-3.0, 3.0, 200)
    # [1차]: 1차원 좌표 배열로 2차원 평면 격자(gx, gy)를 생성합니다.
    # [2차]: 바둑판처럼 가로줄과 세로줄이 만나는 모든 교차점 좌표를 만듭니다.
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    # [1차]: 각 격자점에서의 함숫값 z = x^2 + y^2를 벡터화 연산으로 계산합니다.
    # [2차]: 바둑판 위 모든 점의 높이(x²+y²)를 한 번에 계산합니다.
    gz = gx ** 2 + gy ** 2

    # [1차]: 7x7 인치 크기의 Figure와 Axes 객체를 생성합니다.
    # [2차]: 정사각형 7x7 크기의 도화지를 펼칩니다.
    fig, ax = plt.subplots(figsize=(7, 7))
    # [1차]: 등고선으로 표시할 높이 레벨 리스트를 정의합니다.
    # [2차]: 높이가 1, 2, 4, 6, 9인 지점을 따라 등고선 테두리를 그릴 준비를 합니다.
    levels = [1.0, 2.0, 4.0, 6.0, 9.0]
    # [1차]: 등고선 그래프를 그리고 반환 객체를 contour에 저장합니다.
    # [2차]: 지도에 둥근 등고선들을 진한 남색 선으로 그립니다.
    contour = ax.contour(gx, gy, gz, levels=levels, colors="#2c3e50", linewidths=1.5)
    # [1차]: 각 등고선 위에 높이 숫자 라벨을 인라인으로 표시합니다.
    # [2차]: 등고선마다 "여기는 높이 f=4"라고 꼬리표를 붙여줍니다.
    ax.clabel(contour, inline=True, fontsize=9, fmt="f=%.0f")

    # [1차]: 0부터 2*pi까지 12등분한 각도 배열을 생성합니다(끝점 2*pi 제외).
    # [2차]: 시계 문자판 12시간처럼 원 둘레를 12방향으로 나눕니다.
    sample_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    # [1차]: 화살표를 그릴 두 개의 반지름(1.0, 2.0) 리스트를 정의합니다.
    # [2차]: 안쪽 원(반지름 1)과 바깥쪽 원(반지름 2) 위에서 화살표를 그려볼 것입니다.
    sample_radii = [1.0, 2.0]
    # [1차]: 점 좌표(px, py)와 기울기 벡터 성분(ux, uy)을 담을 빈 리스트 4개를 초기화합니다.
    # [2차]: 화살표 시작점과 방향을 기록할 빈 바구니 4개를 준비합니다.
    px_list, py_list, ux_list, uy_list = [], [], [], []
    # [1차]: 각 반지름 r에 대해 반복합니다.
    # [2차]: 안쪽 원부터 바깥쪽 원 순서로 돕니다.
    for r in sample_radii:
        # [1차]: 각 각도 ang에 대해 반복합니다.
        # [2차]: 12시 방향부터 시계 방향으로 한 점씩 방문합니다.
        for ang in sample_angles:
            # [1차]: 극좌표 (r, ang)를 직교좌표 [r*cos, r*sin] NumPy 배열로 변환합니다.
            # [2차]: 각도와 반지름을 (x, y) 위치 좌표로 바꿉니다.
            pt = np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float64)
            # [1차]: 해당 좌표 pt에서의 수치 Gradient 벡터를 계산합니다.
            # [2차]: 이 점에 서 있을 때 가장 가파른 오르막 방향 화살표를 계산합니다.
            grad = numerical_gradient(quadratic_2d, pt)
            # [1차]: x좌표를 px_list에 추가합니다.
            # [2차]: 화살표 시작 x위치를 기록합니다.
            px_list.append(pt[0])
            # [1차]: y좌표를 py_list에 추가합니다.
            # [2차]: 화살표 시작 y위치를 기록합니다.
            py_list.append(pt[1])
            # [1차]: Gradient의 x성분을 ux_list에 추가합니다.
            # [2차]: 화살표의 가로 방향 크기를 기록합니다.
            ux_list.append(grad[0])
            # [1차]: Gradient의 y성분을 uy_list에 추가합니다.
            # [2차]: 화살표의 세로 방향 크기를 기록합니다.
            uy_list.append(grad[1])

    # [1차]: quiver 함수를 사용해 시작점 (px, py)에서 방향 (ux, uy)의 벡터 화살표를 그립니다.
    # [2차]: 저장해둔 모든 점 위에 빨간색 오르막 화살표를 한꺼번에 꽂아 넣습니다.
    ax.quiver(
        px_list,
        py_list,
        ux_list,
        uy_list,
        color="#e74c3c",
        angles="xy",
        scale_units="xy",
        scale=3.5,
        width=0.006,
        label="Gradient Vector (∇f)",
    )
    # [1차]: 샘플 좌표 위치에 파란색 산점도 마커를 찍습니다.
    # [2차]: 화살표가 시작되는 위치에 파란 압정을 콕콕 찍어 표시합니다.
    ax.scatter(px_list, py_list, color="#2980b9", s=25, zorder=5, label="Sample Points")
    # [1차]: x축과 y축의 1단위 길이를 동일하게(1:1) 고정해 직교성이 왜곡되지 않게 합니다.
    # [2차]: 도화지가 찌그러져 90도가 80도처럼 보이지 않도록 가로세로 비율을 딱 1:1로 맞춥니다.
    ax.set_aspect("equal", adjustable="box")
    # [1차]: x축 표시 범위를 -3.2 ~ 3.2로 설정합니다.
    # [2차]: 가로 화면 범위를 -3.2에서 3.2까지로 자릅니다.
    ax.set_xlim(-3.2, 3.2)
    # [1차]: y축 표시 범위를 -3.2 ~ 3.2로 설정합니다.
    # [2차]: 세로 화면 범위도 똑같이 맞춥니다.
    ax.set_ylim(-3.2, 3.2)
    # [1차]: 점선 형태의 반투명 격자선을 켭니다.
    # [2차]: 모눈종이 눈금을 은은하게 깔아줍니다.
    ax.grid(True, linestyle=":", alpha=0.6)
    # [1차]: x축 라벨을 'x'로 설정합니다.
    # [2차]: 가로축 이름표를 붙입니다.
    ax.set_xlabel("x")
    # [1차]: y축 라벨을 'y'로 설정합니다.
    # [2차]: 세로축 이름표를 붙입니다.
    ax.set_ylabel("y")
    # [1차]: 그래프 상단 제목을 설정합니다.
    # [2차]: 그림 맨 위에 제목 간판을 달아줍니다.
    ax.set_title("Contour Plot & Orthogonal Gradient Vectors of f(x, y) = x^2 + y^2")
    # [1차]: 우측 상단에 범례 박스를 표시합니다.
    # [2차]: 빨간 화살표와 파란 점이 무엇인지 설명표를 붙여줍니다.
    ax.legend(loc="upper right")

    # [1차]: 저장할 파일 경로(output_dir/calc_gradient.png)를 조합합니다.
    # [2차]: 완성된 사진의 파일 이름을 정합니다.
    save_path = out_path / "calc_gradient.png"
    # [1차]: 여백이 겹치지 않도록 레이아웃을 자동 정렬합니다.
    # [2차]: 제목과 눈금이 잘리지 않게 여백을 깔끔히 정리합니다.
    fig.tight_layout()
    # [1차]: 150 DPI 고해상도로 PNG 파일로 저장합니다.
    # [2차]: 선명한 화질로 사진 파일에 인화합니다.
    fig.savefig(save_path, dpi=150)
    # [1차]: 메모리 누수를 막기 위해 Figure 객체를 닫습니다.
    # [2차]: 다 그린 도화지를 치워 책상(메모리)을 비웁니다.
    plt.close(fig)
    # [1차]: 저장된 파일 경로를 문자열로 반환합니다.
    # [2차]: 사진이 저장된 주소를 알려주며 끝냅니다.
    return str(save_path)
