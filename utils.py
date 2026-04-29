import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Mechine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# CNNs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from config import dataset_configs, args

TOP_K = args.top_k

def load_dataset(model_name, dataset_name, fs, phase,
                base_path="../IDS_Datasets/"
                ):
    load_path=f"./{model_name}_phase{phase - 1}/" # 수정가능

    if dataset_name not in dataset_configs:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset_name}")

    conf = dataset_configs[dataset_name]
    dataset_dir = os.path.join(base_path, conf["path"])
    df_list = [] # 리스트 이름을 df에서 df_list로 변경 (가독성)

    # 1. 데이터 로딩 (정상 + 공격)
    norm_path = os.path.join(dataset_dir, conf["normal"])
    if os.path.exists(norm_path):
        df_n = pd.read_csv(norm_path, nrows=100000)
        df_n['target'] = len(conf["labels"]) - 1 
        df_list.append(df_n)

    for i in conf["anomaly_range"]:
        anom_path = os.path.join(dataset_dir, conf["anomaly_pattern"].format(i))
        if os.path.exists(anom_path):
            df_a = pd.read_csv(anom_path, nrows=50000)
            df_a['target'] = i 
            df_list.append(df_a)

    # 데이터 통합 (컬럼 필터링을 위해 리스트를 먼저 합칩니다)
    if not df_list:
        raise ValueError(f"데이터를 찾지 못했습니다: {dataset_dir}")
    
    full_df = pd.concat(df_list, axis=0).reset_index(drop=True)

    if TOP_K:
        load_path=f"./{model_name}_phase0/" 
        tk_file_path = os.path.join(load_path, f"{model_name}_{dataset_name}_results_p0",
                                    f"{model_name}_{dataset_name}_feature_frequency_full.csv")

        if os.path.exists(tk_file_path):
            print(f"[{dataset_name}] TOP {TOP_K} Feature Selection 적용 중...")
            tk_df = pd.read_csv(tk_file_path)
            selected_features = tk_df['feature'].head(TOP_K).tolist()

            # target 컬럼 유지 로직
            if 'target' not in selected_features:
                selected_features.append('target')

            # 실제 데이터프레임에 존재하는 컬럼만 필터링 (에러 방지)
            available_features = [f for f in selected_features if f in full_df.columns]
            full_df = full_df[available_features]

            print(f"[{dataset_name}] TOP {TOP_K} 적용 완료: {len(available_features) - 1}개 피처 선택")
        else:
            print(f"Warning: TOP_K 파일을 찾지 못했습니다: {tk_file_path}")

    # 2. Feature Selection (fs가 1일 때만 수행)
    if fs:
        # 파일 경로 설정 (이전에 저장한 빈도수 CSV)
        fs_file_path = os.path.join(load_path, f"{model_name}_{dataset_name}_results_p{phase - 1}",
                                    f"{model_name}_{dataset_name}_feature_frequency_full.csv")
        
        if os.path.exists(fs_file_path):
            print(f"[{dataset_name}] Feature Selection 적용 중...")
            
            # 빈도수 표 로드
            fs_df = pd.read_csv(fs_file_path)
            
            print(f"[{dataset_name}] Feature Selection: CSV 내 모든 피처({len(fs_df)}개) 사용")
            selected_features = fs_df['feature'].tolist()

            # target 컬럼은 학습에 반드시 필요하므로 유지
            if 'target' not in selected_features:
                selected_features.append('target')
                
            # 데이터셋에서 해당 열만 추출 (존재하지 않는 열은 제외)
            available_features = [f for f in selected_features if f in full_df.columns]
            full_df = full_df[available_features]
            
            print(f"선택된 피처 수: {len(available_features) - 1} (target 제외)")
        else:
            print(f"Warning: FS 파일을 찾지 못해 모든 피처를 유지합니다: {fs_file_path}")
    
    return split_dataset(full_df, dataset_name)

def split_dataset(full_df, dataset_name): # dataset_name 인자 추가
    """
    이미 통합된 데이터프레임을 받아 전처리, 수치형 추출, 스케일링 및 분할을 수행합니다.
    """
    # 1. 전처리 (NaN, Inf 처리)
    df = full_df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # 2. 수치형 데이터만 선택 및 X, y 분리
    # [수정] pd.concat 삭제: 이미 df가 데이터프레임이므로 바로 사용합니다.
    X = df.drop(columns=['target'], errors='ignore').select_dtypes(include=[np.number])
    y = df['target']

    print(f"\n[Data Info] Dataset: {dataset_name} | Total Rows: {X.shape[0]} | Total Features: {X.shape[1]}")

    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Scaling (Min-Max)
    scaler = MinMaxScaler()
    
    # Train 데이터로 피팅하고 Test 데이터는 변환만 수행 (Data Leakage 방지)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 다시 데이터프레임 형태로 복구 (컬럼명 유지를 위해)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 5. 해당 데이터셋의 라벨 정보 반환
    conf = dataset_configs[dataset_name]
    
    X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train_reshaped, X_test_reshaped, X_train, X_test, y_train, y_test, conf["labels"]

def load_model(MODEL_NAME, input_dim=None, num_classes=None):
    model = None
    if MODEL_NAME == "RF":
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=18, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
    elif MODEL_NAME == "DT":
        model = DecisionTreeClassifier(
            max_depth=18,             # RF와 동일한 깊이 설정
            random_state=42,
            class_weight='balanced'   # DT는 subsample 옵션이 없으므로 balanced 사용
        )
    elif MODEL_NAME == "SVM":
        model = SVC(
            kernel='rbf',  # 실험용으로 속도를 위해 가벼운 설정을 추천 (kernel='linear' 또는 'rbf')
            probability=True, 
            random_state=42, 
            class_weight='balanced'
        )
    elif MODEL_NAME == "CNN":
        model = Sequential([
            # 1D Convolution: 네트워크 패킷 특징의 인접 패턴을 추출
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
            MaxPooling1D(pool_size=2, padding='same'),
            Dropout(0.2),
            
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2, padding='same'),
            Dropout(0.2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax') # 다중 분류를 위한 Softmax
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    elif MODEL_NAME == "ANN":
        model = Sequential([
            # 첫 번째 Dense 레이어에 input_shape를 직접 지정 (튜플 형태 주의!)
            Dense(128, activation='relu', input_shape=(input_dim,)), 
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    elif MODEL_NAME == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=15, 
            metric="manhattan",
            weights="uniform",      # default 
            n_jobs=-1
        )
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {MODEL_NAME}")
    
    return model