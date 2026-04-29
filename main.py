import pandas as pd
import numpy as np
import shap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib
matplotlib.use('Agg') # 서버 환경용 설정
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

from utils import load_dataset, load_model
from config import args

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# --- 설정 변수 ---
# KDD99 / InSDN / UNSW_NB15 / CIC2018
PHASE = args.phase
MODEL_NAME = args.model
DATASET_NAME = args.dataset
TOP_K = args.top_k
FS = 1

if PHASE == 1:
    TOP_N = 10
elif PHASE == 2:
    TOP_N = 5
elif PHASE == 3:
    TOP_N = 3
else:
    TOP_N = 20
    FS = 0

# FS_COUNT = args.fs_count
# FS = args.fs

if TOP_K:
    RESULT_DIR = f"{MODEL_NAME}_{DATASET_NAME}_results_top{TOP_K}"
else:
    RESULT_DIR = f"{MODEL_NAME}_{DATASET_NAME}_results_p{PHASE}"

#-----------------

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

X_train_nd, X_test_nd, X_train, X_test, y_train, y_test, class_names = load_dataset(MODEL_NAME, DATASET_NAME, FS, PHASE)
 
# 2. 모델 학습
print(f"모델: {MODEL_NAME} | 학습 시작 (총 {len(class_names)}개 클래스)...")
# model = load_model(MODEL_NAME, X_train.shape[1], len(class_names))

train_acc = 0.0
test_acc = 0.0
if MODEL_NAME in ["RF", "DT", "SVM", "KNN"]:
    model = load_model(MODEL_NAME)
    model.fit(X_train, y_train)
    # train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
elif MODEL_NAME == "ANN":
    model = load_model(MODEL_NAME, X_train.shape[1], len(class_names))
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1) # ANN은 학습이 빠르므로 Epoch를 좀 더 줘도 됨
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
else: # CNN
    model = load_model(MODEL_NAME, X_train_nd.shape[1], len(class_names))
    model.fit(X_train_nd, y_train, epochs=50, batch_size=128, verbose=1)
    train_acc = model.evaluate(X_train_nd, y_train, verbose=0)
    loss, test_acc = model.evaluate(X_test_nd, y_test, verbose=0)
    y_probs = model.predict(X_test_nd)
    y_pred = np.argmax(y_probs, axis=1)

# print(f"모델 정확도: {train_acc:.8f} | {test_acc:.8f}")

# 3. 예측 및 성능 평가
present_labels = np.unique(y_test)
present_names = [class_names[i] for i in present_labels]

# 리포트 생성
report = classification_report(
    y_test, y_pred, 
    labels=present_labels, 
    target_names=present_names,
    digits=6
    )

print("\n[ 공격 유형별 분류 성능 리포트 ]")
print(report)

# [추가] 결과 리포트를 .txt 파일로 저장
report_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_{DATASET_NAME}_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Accuracy: {test_acc:.8f}\n")
    f.write("-" * 50 + "\n")
    f.write(report)
print(f"성능 리포트 저장 완료: {report_path}")

shap_bool = args.shap
if shap_bool:
    # 4. 혼동 행렬 시각화
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_names, yticklabels=present_names)
    plt.title(f"{DATASET_NAME} Classification - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(RESULT_DIR, f"{MODEL_NAME}_{DATASET_NAME}_cm.png"), bbox_inches='tight')
    plt.close()

    # 5. XAI (SHAP) 해석
    print("SHAP 계산 중...")

    if MODEL_NAME in ["RF", "DT"]:
        explainer = shap.TreeExplainer(model)
        sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample_X)
    elif MODEL_NAME in ["SVM", "KNN"]:
        background_X = X_train.sample(n=min(100, len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background_X)
        sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample_X)
    elif MODEL_NAME == "ANN":
        background_data = X_train.sample(n=min(100, len(X_train)), random_state=42).values.astype('float32')
        explainer = shap.DeepExplainer(model, background_data)
        
        sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)
        sample_X_nd = sample_X.values.astype('float32')
        
        shap_values = explainer.shap_values(sample_X_nd)
        
        if isinstance(shap_values, list):
            shap_values = [np.squeeze(s) for s in shap_values]
    else: #CNN
        # 3D ndarray
        background_X = X_train.sample(n=min(100, len(X_train)), random_state=42).values
        background_X = background_X.reshape(background_X.shape[0], background_X.shape[1], 1).astype('float32')
        
        explainer = shap.DeepExplainer(model, background_X)
        
        # 샘플 데이터 준비 (시각화용 Pandas + 계산용 3D ndarray)
        sample_X = X_test.sample(n=min(200, len(X_test)), random_state=42)
        sample_X_nd = sample_X.values.reshape(sample_X.shape[0], sample_X.shape[1], 1).astype('float32')
        
        # CNN 전용 SHAP 계산
        shap_values = explainer.shap_values(sample_X_nd)
        
        # [중요] DeepExplainer 차원 압축 처리
        if isinstance(shap_values, list):
            shap_values = [np.squeeze(s) for s in shap_values]
        elif len(shap_values.shape) == 4:
            shap_values = np.squeeze(shap_values, axis=2)

    print("SHAP 결과 저장 중...")
    # shap_values의 마지막 차원 크기(클래스 수) 확인
    num_output_classes = len(present_labels)

    for i in range(num_output_classes):
        # 모델이 내뱉은 i번째 결과가 실제 어떤 클래스인지 매칭
        class_idx = present_labels[i] 
        class_name = class_names[class_idx]
        
        # [수정] SHAP 값 인덱싱 로직 최적화
        if isinstance(shap_values, list):
            # 다중 분류 트리 모델은 보통 클래스별 리스트로 반환됨
            current_shap_values = shap_values[class_idx]
        else:
            # 배열 형태(samples, features, classes)로 올 경우
            current_shap_values = shap_values[:, :, class_idx]
        
        shap.summary_plot(current_shap_values, sample_X, show=False)
        
        # 파일명 특수문자 제거
        safe_name = class_name.replace(' ', '_').replace('/', '_').replace('.', '')
        plt.title(f"Feature Importance for: {class_name}")
        plt.savefig(os.path.join(RESULT_DIR, f"shap_{safe_name}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"저장 완료: shap_{safe_name}.png")

    all_top_features = []
    top_n_to_pick = TOP_N  # 각 유형별 상위 20개 피처를 추출 기준으로 설정

    # 7. 개별 유형별 TOP 20 피처의 출현 빈도 전수 조사
    print(f"모든 공격 유형별 TOP {top_n_to_pick} 피처의 출현 빈도 집계 중...")

    # 각 클래스(공격 유형)를 순회하며 TOP 20 피처 수집
    for i in range(num_output_classes):
        # 실제 데이터에 존재하는 라벨인지 확인 (IndexError 방지)
        if i >= len(present_labels):
            continue
            
        class_idx = present_labels[i]
        
        # SHAP 값 추출 (리스트/배열 형태 대응)
        if isinstance(shap_values, list):
            current_shap_values = shap_values[class_idx]
        else:
            current_shap_values = shap_values[:, :, class_idx]
        
        # 해당 공격 유형에서 영향력이 높은 상위 20개 피처 이름 추출
        feature_importance_local = np.abs(current_shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance_local)[-top_n_to_pick:]
        top_features = [X_test.columns[idx] for idx in top_indices]
        
        all_top_features.extend(top_features)

    # 피처별 등장 횟수 카운트 (전체)
    feature_counts = Counter(all_top_features)

    # 데이터프레임 변환 (빈도수 기준 내림차순 정렬)
    # 상위 개수를 자르지 않고 발견된 모든 피처를 포함합니다.
    common_features_df = pd.DataFrame(
        feature_counts.items(), 
        columns=['feature', 'occurrence_count']
    ).sort_values(by='occurrence_count', ascending=False)

    # 결과 표 저장 (.csv)
    full_list_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_{DATASET_NAME}_feature_frequency_full.csv")
    common_features_df.to_csv(full_list_path, index=False)

    print(f"전체 피처 빈도수 표 저장 완료: {full_list_path}")
    print(f"집계된 고유 피처 개수: {len(common_features_df)}")