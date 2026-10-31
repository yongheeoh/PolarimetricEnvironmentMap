import time
import requests
from requests.adapters import HTTPAdapter, Retry


class MechaController:
    def __init__(
        self, mecha_ip: str, auth_info=None, settle_delay_large=1.0, settle_delay_small=0.2
    ):
        self.MECHA_IP = mecha_ip
        self.auth = auth_info
        self.settle_delay_large = settle_delay_large
        self.settle_delay_small = settle_delay_small

        self.session = requests.Session()
        retries = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def call_i(self, params, large_move=False):
        """RAW /i 호출 (큰 이동이면 더 길게 쉬어줌)"""
        try:
            r = self.session.get(
                f"{self.MECHA_IP}/i", params=params, auth=self.auth, timeout=(3, 30)
            )
            r.raise_for_status()
            txt = r.text.strip()
            # print(f"/i {params} -> {r.status_code} {txt}")
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(f"Stage move timed out: {params}")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Stage connection failed: {e}") from e
        # 이동 후 안정 대기
        time.sleep(self.settle_delay_large if large_move else self.settle_delay_small)
        # 촬영 지연은 호출부에서 별도 수행

    def move_abs(self, yaw=None, pitch=None, large_move=False):
        params = {}
        if yaw is not None:
            params["ad1"] = yaw
        if pitch is not None:
            params["ad2"] = pitch
        if params:
            self.call_i(params, large_move=large_move)
