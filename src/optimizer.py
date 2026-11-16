"""최적화(Optimization) 알고리즘 및 수렴 경로 시각화 모듈.

포함 기능:
- VanillaGD (기본 경사하강법) 옵티마이저 클래스
- Momentum (관성 기반 경사하강법) 옵티마이저 클래스
- f(x, y) = x^2 + y^2 수렴(lr=0.1) 및 발산(lr>=0.5) 시각화
- f(x, y) = x^2 + y^2 및 타원형 함수 f(x, y) = x^2 + 10y^2 수렴 경로 비교 시각화
"""

# [1차]: 파일 경로 처리를 위해 pathlib의 Path 클래스를 임포트합니다.
# [2차]: 결과 그림을 저장할 폴더 주소를 다루기 위해 주소록 도구를 꺼냅니다.
from pathlib import Path
# [1차]: 타입 힌트(Callable, List, Optional, Tuple)를 위해 typing 모듈에서 임포트합니다.
# [2차]: 함수의 입력과 출력 형태를 명확히 알려주는 이름표들을 준비합니다.
from typing import Callable, List, Optional, Tuple

# [1차]: Matplotlib 핵심 패키지를 임포트합니다.
# [2차]: 그래프를 그릴 도화지 세트를 꺼냅니다.
import matplotlib
# [1차]: 화면 출력 없이 파일로만 렌더링하도록 'Agg' 백엔드를 지정합니다.
# [2차]: 모니터에 창을 띄우지 않고 바로 그림 파일(.png)로만 인화하게 설정합니다.
matplotlib.use("Agg")
# [1차]: pyplot 모듈을 plt 별칭으로 임포트합니다.
# [2차]: 그래프를 그리는 간편한 리모컨을 plt라고 부릅니다.
import matplotlib.pyplot as plt
# [1차]: 수치 연산 라이브러리 NumPy를 np 별칭으로 임포트합니다.
# [2차]: 벡터 뺄셈과 노름 계산을 빠르게 해주는 공학용 계산기를 불러옵니다.
import numpy as np


# [1차]: 기본 경사하강법(Vanilla Gradient Descent)을 구현한 클래스를 정의합니다.
# [2차]: 오직 현재 발밑의 경사만 보고 한 걸음씩 내려가는 '정직한 등산객' 클래스입니다.
class VanillaGD:
    """기본 경사하강법(Vanilla Gradient Descent) 옵티마이저.

    수식:
        theta_{t+1} = theta_t - lr * grad(theta_t)
    """

    # [1차]: 생성자 메서드 __init__에서 학습률 lr(기본 0.1)을 받아 초기화합니다.
    # [2차]: 등산객이 출발하기 전에 한 걸음 보폭(lr)을 얼마로 할지 정합니다.
    def __init__(self, lr: float = 0.1) -> None:
        """VanillaGD 옵티마이저를 초기화합니다.

        Args:
            lr: 학습률(Learning Rate, 양수).

        Raises:
            ValueError: lr <= 0 일 때 발생.
        """
        # [1차]: 학습률이 0 이하인지 검사하여 역방향 이동이나 정지를 방지합니다.
        # [2차]: 보폭이 0이거나 음수면 산을 내려갈 수 없으므로 입구에서 막습니다.
        if lr <= 0.0:
            # [1차]: 잘못된 학습률에 대해 ValueError 예외를 발생시킵니다.
            # [2차]: "보폭은 양수여야 합니다"라고 안내문을 띄웁니다.
            raise ValueError("학습률 lr은 0보다 큰 양수여야 합니다.")
        # [1차]: 학습률을 float로 변환하여 인스턴스 속성 self.lr에 저장합니다.
        # [2차]: 정해진 보폭 값을 등산객의 수첩(self.lr)에 적어둡니다.
        self.lr = float(lr)

    # [1차]: 현재 파라미터 params와 기울기 grads를 받아 한 스텝 업데이트된 파라미터를 반환합니다.
    # [2차]: 현재 위치에서 경사 반대 방향으로 딱 한 걸음 이동한 다음 좌표를 계산합니다.
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """단일 업데이트 스텝을 수행합니다.

        Args:
            params: 현재 파라미터 벡터.
            grads: 현재 파라미터에 대한 기울기 벡터.

        Returns:
            업데이트된 파라미터 벡터.
        """
        # [1차]: params를 float64형 NumPy 배열로 변환합니다.
        # [2차]: 현재 좌표를 계산 가능한 숫자 배열로 정리합니다.
        p = np.asarray(params, dtype=np.float64)
        # [1차]: grads를 float64형 NumPy 배열로 변환합니다.
        # [2차]: 현재 위치의 경사(기울기)도 숫자 배열로 정리합니다.
        g = np.asarray(grads, dtype=np.float64)
        # [1차]: theta - lr * grad 수식을 벡터화 연산으로 계산해 반환합니다.
        # [2차]: 오르막 방향(g)의 반대쪽(-g)으로 보폭(lr)만큼 내려간 새 위치를 돌려줍니다.
        return p - self.lr * g

    # [1차]: 기울기 함수 grad_fn을 이용해 init_params부터 num_steps회 반복 최적화하고 궤적을 반환합니다.
    # [2차]: 출발점부터 골짜기 바닥까지 100걸음 걸어가는 동안 밟은 발자국을 모두 지도에 기록합니다.
    def optimize(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        init_params: np.ndarray,
        num_steps: int = 100,
        clip_bound: Optional[float] = 1e6,
    ) -> np.ndarray:
        """주어진 기울기 함수에 대해 num_steps만큼 최적화를 수행하고 궤적을 반환합니다.

        Args:
            grad_fn: 파라미터를 입력받아 기울기 벡터를 반환하는 함수.
            init_params: 초기 파라미터 벡터.
            num_steps: 반복 횟수.
            clip_bound: 발산 시 오버플로우 방지를 위한 궤적 절단 상한 (None이면 미적용).

        Returns:
            (T+1, D) 형태의 최적화 파라미터 궤적 배열.

        Raises:
            ValueError: num_steps <= 0 일 때 발생.
        """
        # [1차]: 반복 횟수 num_steps가 1 이상인지 검증합니다.
        # [2차]: 0번 이하로 걷겠다고 하면 의미가 없으므로 오류를 냅니다.
        if num_steps <= 0:
            # [1차]: num_steps <= 0이면 ValueError를 발생시킵니다.
            # [2차]: 최소 한 걸음 이상 걷도록 안내합니다.
            raise ValueError("num_steps는 1 이상이어야 합니다.")

        # [1차]: 초기 파라미터를 float64 배열로 복사해 현재 위치 변수 current를 초기화합니다.
        # [2차]: 출발점 좌표를 복사해 등산객의 현재 위치로 삼습니다.
        current = np.asarray(init_params, dtype=np.float64).copy()
        # [1차]: 발자국 궤적을 저장할 리스트 history에 시작점을 복사해 넣습니다.
        # [2차]: 발자국 일기장의 첫 번째 줄에 출발 좌표를 적습니다.
        history: List[np.ndarray] = [current.copy()]

        # [1차]: num_steps 횟수만큼 파라미터 업데이트 루프를 반복합니다.
        # [2차]: 정해진 걸음 수만큼 한 걸음씩 내려갑니다.
        for _ in range(num_steps):
            # [1차]: 현재 위치에서 grad_fn을 호출해 기울기 벡터를 구합니다.
            # [2차]: 지금 서 있는 곳의 경사가 어느 쪽으로 얼마나 가파른지 잽니다.
            grads = np.asarray(grad_fn(current), dtype=np.float64)
            # [1차]: step 메서드를 호출해 다음 위치로 current를 갱신합니다.
            # [2차]: 경사 반대 방향으로 한 걸음 이동합니다.
            current = self.step(current, grads)
            # [1차]: 발산 시 무한대(inf)로 치솟는 것을 막기 위해 상한(clip_bound) 초과 여부를 검사합니다.
            # [2차]: 보폭이 너무 커서 산 밖으로 100만 미터 이상 날아가면 안전을 위해 기록을 멈춥니다.
            if clip_bound is not None and np.linalg.norm(current) > clip_bound:
                # [1차]: 이탈 직전 위치를 history에 추가하고 루프를 조기 종료합니다.
                # [2차]: 마지막 발산 위치를 일기장에 남기고 하산을 중단합니다.
                history.append(current.copy())
                break
            # [1차]: 정상 업데이트된 위치를 history에 복사 추가합니다.
            # [2차]: 이동한 새 좌표를 발자국 일기장에 차곡차곡 기록합니다.
            history.append(current.copy())

        # [1차]: 리스트에 모인 궤적을 (T+1, D) 2D NumPy 배열로 변환해 반환합니다.
        # [2차]: 지금까지 모은 발자국 일기장을 깔끔한 표로 묶어 돌려줍니다.
        return np.array(history, dtype=np.float64)


# [1차]: 관성(Momentum) 기반 경사하강법 옵티마이저 클래스를 정의합니다.
# [2차]: 이전 속도를 기억하며 묵직하게 굴러 내려가는 '볼링공' 옵티마이저 클래스입니다.
class Momentum:
    """모멘텀(Momentum) 기반 경사하강법 옵티마이저.

    수식:
        v_{t+1} = beta * v_t + grad(theta_t)
        theta_{t+1} = theta_t - lr * v_{t+1}
    """

    # [1차]: 학습률 lr과 관성 계수 beta(기본 0.9)를 받아 Momentum 객체를 초기화합니다.
    # [2차]: 볼링공의 보폭(lr)과 이전 속도를 몇 % 유지할지(관성 90%) 설정합니다.
    def __init__(self, lr: float = 0.1, beta: float = 0.9) -> None:
        """Momentum 옵티마이저를 초기화합니다.

        Args:
            lr: 학습률(Learning Rate, 양수).
            beta: 관성 계수(Momentum Coefficient, 0 <= beta < 1).

        Raises:
            ValueError: lr <= 0 이거나 beta가 [0, 1) 범위를 벗어날 때 발생.
        """
        # [1차]: 학습률이 0 이하인지 확인합니다.
        # [2차]: 보폭이 0 이하이면 움직일 수 없으므로 막습니다.
        if lr <= 0.0:
            # [1차]: lr <= 0일 때 ValueError를 발생시킵니다.
            # [2차]: 양수 보폭만 넣도록 안내합니다.
            raise ValueError("학습률 lr은 0보다 큰 양수여야 합니다.")
        # [1차]: 관성 계수 beta가 0 이상 1 미만 구간에 있는지 확인합니다.
        # [2차]: 관성이 100% 이상이면 멈추지 않고 폭주하므로 0 이상 1 미만만 허용합니다.
        if not (0.0 <= beta < 1.0):
            # [1차]: 유효 범위를 벗어난 beta에 대해 ValueError를 발생시킵니다.
            # [2차]: "관성 계수는 0과 1 사이여야 합니다"라고 알립니다.
            raise ValueError("관성 계수 beta는 0 <= beta < 1 범위여야 합니다.")
        # [1차]: 학습률 lr을 float로 저장합니다.
        # [2차]: 보폭을 수첩에 적습니다.
        self.lr = float(lr)
        # [1차]: 관성 계수 beta를 float로 저장합니다.
        # [2차]: 관성 비율(예: 0.9)을 수첩에 적습니다.
        self.beta = float(beta)
        # [1차]: 누적 속도 벡터 self.velocity를 초기 상태 None으로 설정합니다.
        # [2차]: 출발 전에는 아직 굴러간 속도가 없으므로 비워둡니다.
        self.velocity: Optional[np.ndarray] = None

    # [1차]: 누적된 속도 벡터 self.velocity를 None으로 되돌리는 초기화 메서드입니다.
    # [2차]: 새로운 산에서 다시 출발할 때 이전 볼링공 속도를 0으로 리셋합니다.
    def reset(self) -> None:
        """누적된 속도(velocity) 상태를 초기화합니다."""
        # [1차]: self.velocity를 None으로 재설정합니다.
        # [2차]: 속도 기억 장치를 깨끗이 비웁니다.
        self.velocity = None

    # [1차]: 이전 속도에 beta를 곱하고 현재 기울기 grads를 더해 파라미터를 업데이트합니다.
    # [2차]: 기존에 굴러가던 관성 + 지금 발밑의 경사를 합쳐서 한 걸음 크게 굴러갑니다.
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """속도를 누적하여 단일 파라미터 업데이트 스텝을 수행합니다.

        Args:
            params: 현재 파라미터 벡터.
            grads: 현재 파라미터에 대한 기울기 벡터.

        Returns:
            업데이트된 파라미터 벡터.
        """
        # [1차]: params를 float64 배열로 변환합니다.
        # [2차]: 현재 좌표를 숫자 배열로 다듬습니다.
        p = np.asarray(params, dtype=np.float64)
        # [1차]: grads를 float64 배열로 변환합니다.
        # [2차]: 현재 경사를 숫자 배열로 다듬습니다.
        g = np.asarray(grads, dtype=np.float64)
        # [1차]: 첫 스텝이거나 파라미터 shape이 달라진 경우 0 벡터로 속도를 초기화합니다.
        # [2차]: 첫 출발이라면 속도를 0으로 맞춰 준비합니다.
        if self.velocity is None or self.velocity.shape != p.shape:
            # [1차]: p와 같은 모양의 0 배열을 생성해 self.velocity에 할당합니다.
            # [2차]: 정지 상태의 속도 벡터를 만듭니다.
            self.velocity = np.zeros_like(p, dtype=np.float64)

        # [1차]: v_{t+1} = beta * v_t + grad 수식으로 속도를 갱신합니다.
        # [2차]: 이전 속도의 90%를 유지한 채 현재 경사의 힘을 보태 새 속도를 만듭니다.
        self.velocity = self.beta * self.velocity + g
        # [1차]: 갱신된 속도 방향으로 lr만큼 차감한 새 파라미터를 반환합니다.
        # [2차]: 완성된 속도 방향으로 보폭만큼 굴러간 새 좌표를 돌려줍니다.
        return p - self.lr * self.velocity

    # [1차]: init_params에서 시작해 num_steps회 모멘텀 업데이트를 수행하고 궤적을 반환합니다.
    # [2차]: 볼링공을 굴려 출발점부터 골짜기 바닥까지의 이동 경로를 기록합니다.
    def optimize(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        init_params: np.ndarray,
        num_steps: int = 100,
        clip_bound: Optional[float] = 1e6,
    ) -> np.ndarray:
        """주어진 기울기 함수에 대해 num_steps만큼 모멘텀 최적화를 수행하고 궤적을 반환합니다.

        Args:
            grad_fn: 파라미터를 입력받아 기울기 벡터를 반환하는 함수.
            init_params: 초기 파라미터 벡터.
            num_steps: 반복 횟수.
            clip_bound: 발산 시 오버플로우 방지를 위한 상한.

        Returns:
            (T+1, D) 형태의 최적화 파라미터 궤적 배열.

        Raises:
            ValueError: num_steps <= 0 일 때 발생.
        """
        # [1차]: 반복 횟수 num_steps가 양수인지 검증합니다.
        # [2차]: 0 이하의 걸음 수는 허용하지 않습니다.
        if num_steps <= 0:
            # [1차]: 잘못된 반복 횟수에 대해 ValueError를 발생시킵니다.
            # [2차]: 최소 1회 이상 반복하도록 안내합니다.
            raise ValueError("num_steps는 1 이상이어야 합니다.")

        # [1차]: 새 최적화 시작 전 이전 궤적의 속도가 남지 않도록 reset()을 호출합니다.
        # [2차]: 출발 전 볼링공 속도를 깔끔히 0으로 초기화합니다.
        self.reset()
        # [1차]: 초기 좌표를 복사해 current에 저장합니다.
        # [2차]: 출발점 사본을 만들어 현재 위치로 삼습니다.
        current = np.asarray(init_params, dtype=np.float64).copy()
        # [1차]: 궤적 리스트에 시작 좌표를 추가합니다.
        # [2차]: 발자국 일기장의 첫 줄에 출발점을 적습니다.
        history: List[np.ndarray] = [current.copy()]

        # [1차]: num_steps만큼 반복하며 모멘텀 스텝을 진행합니다.
        # [2차]: 정해진 횟수만큼 볼링공을 굴립니다.
        for _ in range(num_steps):
            # [1차]: 현재 좌표에서의 기울기를 계산합니다.
            # [2차]: 지금 위치의 경사를 잽니다.
            grads = np.asarray(grad_fn(current), dtype=np.float64)
            # [1차]: 모멘텀 step으로 다음 좌표를 계산합니다.
            # [2차]: 관성을 실어 다음 위치로 이동합니다.
            current = self.step(current, grads)
            # [1차]: 노름이 clip_bound를 초과하면 발산으로 간주하고 중단합니다.
            # [2차]: 너무 멀리 날아가면 안전을 위해 기록을 멈춥니다.
            if clip_bound is not None and np.linalg.norm(current) > clip_bound:
                # [1차]: 마지막 좌표를 기록하고 반복을 탈출합니다.
                # [2차]: 이탈 지점을 남기고 종료합니다.
                history.append(current.copy())
                break
            # [1차]: 정상 좌표를 history에 추가합니다.
            # [2차]: 이동한 위치를 일기장에 적습니다.
            history.append(current.copy())

        # [1차]: 전체 이동 궤적을 2D NumPy 배열로 변환해 반환합니다.
        # [2차]: 모든 발자국 기록을 표로 묶어 돌려줍니다.
        return np.array(history, dtype=np.float64)


# [1차]: 원형 함수 f(x, y) = x^2 + y^2의 해석적 Gradient (2x, 2y)를 반환합니다.
# [2차]: 동그란 밥그릇 지형에서 현재 위치의 경사를 바로 알려주는 공식 함수입니다.
def grad_circle(p: np.ndarray) -> np.ndarray:
    """f(x, y) = x^2 + y^2의 해석적 Gradient (2x, 2y)."""
    # [1차]: [2*p[0], 2*p[1]] 형태의 float64 배열을 생성해 반환합니다.
    # [2차]: x경사는 2x, y경사는 2y로 계산해 돌려줍니다.
    return np.array([2.0 * p[0], 2.0 * p[1]], dtype=np.float64)


# [1차]: 타원형 함수 f(x, y) = x^2 + 10y^2의 해석적 Gradient (2x, 20y)를 반환합니다.
# [2차]: y축 방향이 10배 더 가파른 좁은 계곡 지형에서 경사를 알려주는 함수입니다.
def grad_ellipse(p: np.ndarray) -> np.ndarray:
    """타원형 함수 f(x, y) = x^2 + 10y^2의 해석적 Gradient (2x, 20y)."""
    # [1차]: [2*p[0], 20*p[1]] 형태의 float64 배열을 생성해 반환합니다.
    # [2차]: x경사는 완만하게(2x), y경사는 가파르게(20y) 계산해 돌려줍니다.
    return np.array([2.0 * p[0], 20.0 * p[1]], dtype=np.float64)


# [1차]: VanillaGD의 수렴(lr=0.1)과 발산(lr=1.05 >= 0.5) 경로를 나란히 시각화합니다.
# [2차]: 적당한 보폭(0.1)일 때는 얌전히 골짜기에 도착하지만, 너무 큰 보폭(1.05)이면 산 밖으로 튕겨 나가는 모습을 그림으로 비교합니다.
def visualize_gd_convergence(output_dir: str = "outputs") -> str:
    """VanillaGD의 수렴(lr=0.1)과 발산(lr=1.05 >= 0.5) 경로를 등고선 위에 시각화합니다.

    f(x, y) = x^2 + y^2 에서:
    - lr = 0.1, 100회 반복 -> 원점 근처(반경 0.1 이내) 수렴
    - lr = 1.05 (lr >= 0.5 영역) -> 매 스텝 |1 - 2*lr| = 1.1 > 1 이므로 발산

    Args:
        output_dir: 결과 PNG 저장 디렉토리.

    Returns:
        저장된 PNG 파일 경로 문자열.
    """
    # [1차]: 출력 폴더 Path 객체를 생성합니다.
    # [2차]: 그림을 넣을 보관함 주소를 준비합니다.
    out_path = Path(output_dir)
    # [1차]: 폴더가 없으면 생성합니다.
    # [2차]: 보관함이 없으면 새로 만듭니다.
    out_path.mkdir(parents=True, exist_ok=True)
    # [1차]: 마이너스 기호 깨짐을 방지합니다.
    # [2차]: 음수 눈금이 깨지지 않게 설정합니다.
    plt.rcParams["axes.unicode_minus"] = False

    # [1차]: 수렴 테스트용 초기 좌표 [4.0, 3.0]을 정의합니다.
    # [2차]: 출발점 (4, 3)을 정합니다.
    init_pt = np.array([4.0, 3.0], dtype=np.float64)
    # [1차]: lr=0.1로 100회 반복한 수렴 궤적을 계산합니다.
    # [2차]: 적당한 보폭(0.1)으로 100걸음 걸어간 발자국을 얻습니다.
    gd_conv = VanillaGD(lr=0.1).optimize(grad_circle, init_pt, num_steps=100)
    # [1차]: lr=1.05(0.5 이상)로 12회 반복한 발산 궤적을 계산합니다.
    # [2차]: 너무 큰 보폭(1.05)으로 12걸음 성큼성큼 뛰다가 튕겨 나가는 발자국을 얻습니다.
    gd_div = VanillaGD(lr=1.05).optimize(grad_circle, np.array([1.0, 0.8]), num_steps=12)

    # [1차]: 1행 2열의 서브플롯(13x5.5 인치)을 생성합니다.
    # [2차]: 왼쪽엔 수렴, 오른쪽엔 발산을 나란히 그릴 두 칸짜리 도화지를 펼칩니다.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # [1차]: 등고선용 x, y 격자를 생성합니다.
    # [2차]: 왼쪽 지도용 바둑판 격자를 만듭니다.
    gx, gy = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
    # [1차]: 격자점 높이 z = x^2 + y^2를 계산합니다.
    # [2차]: 밥그릇 지형의 높이를 계산합니다.
    gz = gx ** 2 + gy ** 2
    # [1차]: 왼쪽 패널에 등고선을 그립니다.
    # [2차]: 지도에 둥근 등고선을 그립니다.
    axes[0].contour(gx, gy, gz, levels=[0.5, 2, 5, 10, 20, 30], colors="#7f8c8d", alpha=0.6)
    # [1차]: 수렴 경로를 파란 점선과 원형 마커('b--o')로 그립니다.
    # [2차]: 등산객의 발자국을 파란 점선으로 이어 그립니다.
    axes[0].plot(gd_conv[:, 0], gd_conv[:, 1], "b--o", markersize=3, linewidth=1.8, label="VanillaGD (lr=0.1)")
    # [1차]: 원점 (0, 0)에 초록색 별 마커를 표시합니다.
    # [2차]: 목표 지점인 골짜기 바닥(0, 0)에 초록 별 깃발을 꽂습니다.
    axes[0].scatter([0], [0], color="green", marker="*", s=160, zorder=6, label="Minimum (0, 0)")
    # [1차]: 가로세로 축 비율을 1:1로 맞춥니다.
    # [2차]: 원이 찌그러지지 않게 비율을 고정합니다.
    axes[0].set_aspect("equal", adjustable="box")
    # [1차]: 최종 거리(norm)를 포함한 제목을 설정합니다.
    # [2차]: 100걸음 뒤 원점에 얼마나 가까워졌는지 제목에 적어줍니다.
    axes[0].set_title(f"Convergence (lr=0.1, 100 steps)\nFinal norm={np.linalg.norm(gd_conv[-1]):.2e}")
    # [1차]: x축 라벨을 설정합니다.
    # [2차]: 가로축 이름표를 붙입니다.
    axes[0].set_xlabel("x")
    # [1차]: y축 라벨을 설정합니다.
    # [2차]: 세로축 이름표를 붙입니다.
    axes[0].set_ylabel("y")
    # [1차]: 점선 격자를 켭니다.
    # [2차]: 모눈종이 선을 켭니다.
    axes[0].grid(True, linestyle=":", alpha=0.6)
    # [1차]: 범례를 표시합니다.
    # [2차]: 설명표를 붙입니다.
    axes[0].legend()

    # [1차]: 발산 그래프용 격자를 생성합니다.
    # [2차]: 오른쪽 지도용 바둑판 격자를 만듭니다.
    gx2, gy2 = np.meshgrid(np.linspace(-4, 4, 150), np.linspace(-4, 4, 150))
    # [1차]: 발산 그래프용 높이 z를 계산합니다.
    # [2차]: 밥그릇 높이를 계산합니다.
    gz2 = gx2 ** 2 + gy2 ** 2
    # [1차]: 오른쪽 패널에 등고선을 그립니다.
    # [2차]: 등고선을 그립니다.
    axes[1].contour(gx2, gy2, gz2, levels=[1, 3, 6, 10, 16], colors="#7f8c8d", alpha=0.6)
    # [1차]: 발산 경로를 빨간 점선 마커('r--o')로 그립니다.
    # [2차]: 바깥으로 점점 커지며 튕겨 나가는 발자국을 빨간 점선으로 그립니다.
    axes[1].plot(gd_div[:, 0], gd_div[:, 1], "r--o", markersize=4, linewidth=1.8, label="VanillaGD (lr=1.05 >= 0.5)")
    # [1차]: 원점에 초록색 별 마커를 찍습니다.
    # [2차]: 목표 깃발을 꽂습니다.
    axes[1].scatter([0], [0], color="green", marker="*", s=160, zorder=6, label="Minimum (0, 0)")
    # [1차]: 축 비율을 1:1로 고정합니다.
    # [2차]: 비율을 1:1로 맞춥니다.
    axes[1].set_aspect("equal", adjustable="box")
    # [1차]: x축 범위를 -4 ~ 4로 제한합니다.
    # [2차]: 화면 가로 범위를 고정합니다.
    axes[1].set_xlim(-4.0, 4.0)
    # [1차]: y축 범위를 -4 ~ 4로 제한합니다.
    # [2차]: 화면 세로 범위를 고정합니다.
    axes[1].set_ylim(-4.0, 4.0)
    # [1차]: 오른쪽 패널 제목을 설정합니다.
    # [2차]: "보폭이 너무 커서 발산했다"는 제목을 붙입니다.
    axes[1].set_title("Divergence (lr=1.05 >= 0.5 on x^2 + y^2)")
    # [1차]: x축 라벨을 설정합니다.
    # [2차]: 가로축 이름표를 붙입니다.
    axes[1].set_xlabel("x")
    # [1차]: y축 라벨을 설정합니다.
    # [2차]: 세로축 이름표를 붙입니다.
    axes[1].set_ylabel("y")
    # [1차]: 점선 격자를 켭니다.
    # [2차]: 모눈종이 선을 켭니다.
    axes[1].grid(True, linestyle=":", alpha=0.6)
    # [1차]: 범례를 표시합니다.
    # [2차]: 설명표를 붙입니다.
    axes[1].legend()

    # [1차]: 저장 파일 경로를 조합합니다.
    # [2차]: 저장할 사진 파일 이름을 정합니다.
    save_path = out_path / "opt_gd_divergence.png"
    # [1차]: 전체 Figure 상단 제목을 설정합니다.
    # [2차]: 두 그림을 아우르는 큰 제목을 굵게 적습니다.
    fig.suptitle("VanillaGD Convergence vs Divergence", fontsize=13, fontweight="bold")
    # [1차]: 레이아웃 여백을 자동 조정합니다.
    # [2차]: 글자가 겹치지 않게 정리합니다.
    fig.tight_layout()
    # [1차]: 150 DPI PNG로 저장합니다.
    # [2차]: 사진 파일로 인화합니다.
    fig.savefig(save_path, dpi=150)
    # [1차]: Figure를 닫아 메모리를 해제합니다.
    # [2차]: 도화지를 치웁니다.
    plt.close(fig)
    # [1차]: 저장 경로 문자열을 반환합니다.
    # [2차]: 사진 주소를 돌려줍니다.
    return str(save_path)


# [1차]: 원형(x^2+y^2) 및 타원형(x^2+10y^2) 함수에서 VanillaGD와 Momentum 경로를 겹쳐 시각화합니다.
# [2차]: 둥근 그릇과 좁은 타원 계곡에서 일반 등산객과 볼링공(Momentum)의 이동 경로를 나란히 비교합니다.
def visualize_optimizer_comparison(output_dir: str = "outputs") -> List[str]:
    """원형(x^2+y^2) 및 타원형(x^2+10y^2) 함수에서 VanillaGD와 Momentum 비교 시각화.

    Args:
        output_dir: 결과 PNG 저장 디렉토리.

    Returns:
        생성된 PNG 파일 경로 목록.
    """
    # [1차]: 출력 디렉토리 Path 객체를 생성합니다.
    # [2차]: 저장 폴더 주소를 준비합니다.
    out_path = Path(output_dir)
    # [1차]: 폴더가 없으면 생성합니다.
    # [2차]: 보관함이 없으면 새로 만듭니다.
    out_path.mkdir(parents=True, exist_ok=True)
    # [1차]: 마이너스 기호 깨짐을 방지합니다.
    # [2차]: 음수 눈금이 깨지지 않게 설정합니다.
    plt.rcParams["axes.unicode_minus"] = False

    # [1차]: 생성된 파일 경로를 담을 빈 리스트를 초기화합니다.
    # [2차]: 완성된 사진 주소들을 모아둘 바구니를 만듭니다.
    saved_paths: List[str] = []

    # [1차]: 원형 함수 테스트용 시작점 [4.0, 3.0]을 정의합니다.
    # [2차]: 첫 번째 실험의 출발점을 정합니다.
    init_circle = np.array([4.0, 3.0], dtype=np.float64)
    # [1차]: 원형 함수에서 VanillaGD 50스텝 궤적을 구합니다.
    # [2차]: 일반 등산객의 50걸음 발자국을 기록합니다.
    path_gd_c = VanillaGD(lr=0.1).optimize(grad_circle, init_circle, num_steps=50)
    # [1차]: 원형 함수에서 Momentum 50스텝 궤적을 구합니다.
    # [2차]: 볼링공의 50스텝 굴러간 경로를 기록합니다.
    path_mom_c = Momentum(lr=0.1, beta=0.9).optimize(grad_circle, init_circle, num_steps=50)

    # [1차]: 원형 함수 등고선용 격자를 생성합니다.
    # [2차]: 바둑판 좌표를 만듭니다.
    gx, gy = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
    # [1차]: 원형 함수 높이 z를 계산합니다.
    # [2차]: 밥그릇 높이를 계산합니다.
    gz_c = gx ** 2 + gy ** 2
    # [1차]: 7x6 인치 Figure를 생성합니다.
    # [2차]: 첫 번째 비교 그림 도화지를 펼칩니다.
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    # [1차]: 등고선을 그립니다.
    # [2차]: 둥근 등고선을 그립니다.
    ax1.contour(gx, gy, gz_c, levels=[0.5, 2, 5, 10, 20, 30], colors="#95a5a6", alpha=0.6)
    # [1차]: VanillaGD 경로를 파란 점선으로 그립니다.
    # [2차]: 등산객 발자국을 파란 점선으로 그립니다.
    ax1.plot(path_gd_c[:, 0], path_gd_c[:, 1], "b--o", markersize=3, linewidth=1.8, label="VanillaGD (lr=0.1)")
    # [1차]: Momentum 경로를 빨간 실선과 사각 마커로 그립니다.
    # [2차]: 볼링공 경로를 빨간 실선으로 겹쳐 그립니다.
    ax1.plot(path_mom_c[:, 0], path_mom_c[:, 1], "r-s", markersize=3, linewidth=1.8, label="Momentum (lr=0.1, beta=0.9)")
    # [1차]: 원점에 초록색 별을 표시합니다.
    # [2차]: 목표 깃발을 꽂습니다.
    ax1.scatter([0], [0], color="green", marker="*", s=160, zorder=6, label="Minimum (0, 0)")
    # [1차]: 축 비율을 1:1로 맞춥니다.
    # [2차]: 비율을 고정합니다.
    ax1.set_aspect("equal", adjustable="box")
    # [1차]: 그래프 제목을 설정합니다.
    # [2차]: 제목을 붙입니다.
    ax1.set_title("VanillaGD vs Momentum on f(x, y) = x^2 + y^2")
    # [1차]: x축 라벨을 설정합니다.
    # [2차]: 가로축 이름표를 붙입니다.
    ax1.set_xlabel("x")
    # [1차]: y축 라벨을 설정합니다.
    # [2차]: 세로축 이름표를 붙입니다.
    ax1.set_ylabel("y")
    # [1차]: 격자선을 켭니다.
    # [2차]: 모눈종이를 켭니다.
    ax1.grid(True, linestyle=":", alpha=0.6)
    # [1차]: 범례를 표시합니다.
    # [2차]: 설명표를 붙입니다.
    ax1.legend()
    # [1차]: 저장 경로를 지정합니다.
    # [2차]: 첫 번째 사진 이름을 정합니다.
    p1 = out_path / "opt_comparison_circle.png"
    # [1차]: 여백을 정리합니다.
    # [2차]: 여백을 맞춥니다.
    fig1.tight_layout()
    # [1차]: PNG 파일로 저장합니다.
    # [2차]: 사진 파일로 인화합니다.
    fig1.savefig(p1, dpi=150)
    # [1차]: Figure를 닫습니다.
    # [2차]: 도화지를 치웁니다.
    plt.close(fig1)
    # [1차]: 저장된 경로를 목록에 추가합니다.
    # [2차]: 주소 바구니에 담습니다.
    saved_paths.append(str(p1))

    # [1차]: 타원형 함수 테스트용 시작점 [8.0, 2.0]을 정의합니다.
    # [2차]: 두 번째 실험(좁은 계곡) 출발점을 정합니다.
    init_ellipse = np.array([8.0, 2.0], dtype=np.float64)
    # [1차]: 타원형 함수에서 VanillaGD(lr=0.085) 50스텝 궤적을 구합니다.
    # [2차]: 일반 등산객이 좁은 계곡 벽에 부딪히며 지그재그로 내려가는 경로를 기록합니다.
    path_gd_e = VanillaGD(lr=0.085).optimize(grad_ellipse, init_ellipse, num_steps=50)
    # [1차]: 타원형 함수에서 Momentum(lr=0.04, beta=0.9) 50스텝 궤적을 구합니다.
    # [2차]: 볼링공이 위아래 진동을 상쇄하고 계곡 바닥을 따라 부드럽게 굴러가는 경로를 기록합니다.
    path_mom_e = Momentum(lr=0.04, beta=0.9).optimize(grad_ellipse, init_ellipse, num_steps=50)

    # [1차]: 타원형 함수 등고선용 x, y 격자를 생성합니다.
    # [2차]: 가로로 긴 계곡 지도 격자를 만듭니다.
    gx_e, gy_e = np.meshgrid(np.linspace(-9, 9, 200), np.linspace(-3, 3, 200))
    # [1차]: 타원형 함수 높이 z = x^2 + 10*y^2를 계산합니다.
    # [2차]: 계곡의 높이를 계산합니다.
    gz_e = gx_e ** 2 + 10.0 * (gy_e ** 2)
    # [1차]: 8x5 인치 Figure를 생성합니다.
    # [2차]: 두 번째 비교 그림 도화지를 펼칩니다.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # [1차]: 타원형 등고선을 그립니다.
    # [2차]: 길쭉한 타원형 등고선을 그립니다.
    ax2.contour(gx_e, gy_e, gz_e, levels=[1, 5, 15, 30, 60, 100], colors="#95a5a6", alpha=0.6)
    # [1차]: VanillaGD의 지그재그 경로를 파란 점선으로 그립니다.
    # [2차]: 등산객의 지그재그 발자국을 파란 점선으로 그립니다.
    ax2.plot(path_gd_e[:, 0], path_gd_e[:, 1], "b--o", markersize=3, linewidth=1.8, label="VanillaGD (lr=0.085)")
    # [1차]: Momentum의 부드러운 경로를 빨간 실선으로 그립니다.
    # [2차]: 볼링공의 매끄러운 경로를 빨간 실선으로 겹쳐 그립니다.
    ax2.plot(path_mom_e[:, 0], path_mom_e[:, 1], "r-s", markersize=3, linewidth=1.8, label="Momentum (lr=0.04, beta=0.9)")
    # [1차]: 원점에 초록색 별을 표시합니다.
    # [2차]: 목표 깃발을 꽂습니다.
    ax2.scatter([0], [0], color="green", marker="*", s=160, zorder=6, label="Minimum (0, 0)")
    # [1차]: 그래프 제목을 설정합니다.
    # [2차]: 제목을 붙입니다.
    ax2.set_title("VanillaGD vs Momentum on Elliptical f(x, y) = x^2 + 10y^2")
    # [1차]: x축 라벨을 설정합니다.
    # [2차]: 가로축 이름표를 붙입니다.
    ax2.set_xlabel("x")
    # [1차]: y축 라벨을 설정합니다.
    # [2차]: 세로축 이름표를 붙입니다.
    ax2.set_ylabel("y")
    # [1차]: 격자선을 켭니다.
    # [2차]: 모눈종이를 켭니다.
    ax2.grid(True, linestyle=":", alpha=0.6)
    # [1차]: 범례를 표시합니다.
    # [2차]: 설명표를 붙입니다.
    ax2.legend()
    # [1차]: 두 번째 파일 저장 경로를 지정합니다.
    # [2차]: 두 번째 사진 이름을 정합니다.
    p2 = out_path / "opt_comparison_ellipse.png"
    # [1차]: 여백을 정리합니다.
    # [2차]: 여백을 맞춥니다.
    fig2.tight_layout()
    # [1차]: PNG로 저장합니다.
    # [2차]: 사진 파일로 인화합니다.
    fig2.savefig(p2, dpi=150)
    # [1차]: Figure를 닫아 메모리를 해제합니다.
    # [2차]: 도화지를 치웁니다.
    plt.close(fig2)
    # [1차]: 저장된 경로를 목록에 추가합니다.
    # [2차]: 주소 바구니에 담습니다.
    saved_paths.append(str(p2))

    # [1차]: 생성된 PNG 경로 목록을 반환합니다.
    # [2차]: 완성된 두 장의 사진 주소를 돌려줍니다.
    return saved_paths
