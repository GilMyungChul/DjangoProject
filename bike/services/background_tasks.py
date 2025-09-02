import joblib
import pandas as pd
from ..model_holder import MlModelHolder
from .registry import get_model_spec

def run_train_in_background(model_key, alpha, drop_leak, df):
    # 이 함수는 새로운 프로세스에서 실행됩니다.
    try:
        spec = get_model_spec(model_key)
        train_fn = spec["train"]
        
        # 실제 모델 학습/예측 함수 호출 (이제 DB에 접근하지 않습니다)
        metrics, out_csv, model_paths = train_fn(df, alpha=alpha, drop_leak=drop_leak)

        # 모델 홀더에 모델 강제 로드
        if isinstance(model_paths, (list, tuple)) and len(model_paths) == 2:
            rent_path, return_path = model_paths
            MlModelHolder.force_load(rent_path, return_path)
        elif isinstance(model_paths, str):
            MlModelHolder.force_load(model_paths, model_paths)
        
        print("모델 학습/예측 완료")

    except Exception as e:
        print(f"백그라운드 학습 프로세스 에러: {e}")