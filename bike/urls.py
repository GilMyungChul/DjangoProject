from django.urls import path, register_converter
from bike import views

# 한글 대여소 이름을 URL 매개변수로 처리하기 위한 커스텀 변환기
class LocationNameConverter:
    regex = r'[^/]+'
    def to_python(self, value):
        return value
    def to_url(self, value):
        return value

register_converter(LocationNameConverter, 'location_name')

app_name = "bike"
urlpatterns = [
    path("ml/", views.ml_home, name="ml_home"),
    # path("ml/run/", views.ml_train_predict, name="ml_train_predict"),  # 학습/예측 실행
    path("ml/select/", views.ml_select, name="ml_select"),                 # 모델 선택 페이지 (신규)
    path("ml/train_predict/", views.ml_train_predict, name="ml_train_predict"),  # 학습/예측 실행 (기존 이름 재사용)

    path("forecast/select/", views.forecast_select, name="forecast_select"),  # 모델 선택(미래 예측)
    path("forecast/",         views.forecast_home,   name="forecast_home"),   # 업로드 & 예측 실행

    # === 예측 지도를 위한 최종 URL 패턴 ===
    path("predict/map/", views.predict_map_view, name="predict_map"),
    path("predict/map/<str:location>/", views.predict_map_view, name="predict_map_with_loc"),
]