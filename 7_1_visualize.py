import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 경로 설정 (사용자 입력)
BASE_DIR = "/home/seoi0215/trend/SAE/data_classification"
SAVED_MODEL_DIR = f"{BASE_DIR}/Verifed_code/output/7_optimized_eval"
PIPELINE_PATH = f"{SAVED_MODEL_DIR}/uncertainty_detector_pipeline.pkl"
CONFIG_PATH = f"{SAVED_MODEL_DIR}/rag_circuit_config.json"
RAW_DATA_PATH = f"{BASE_DIR}/output/hotpot_qa_llama3_results_train_no_context_evaluated.csv"
OUTPUT_DIR = f"{BASE_DIR}/Verifed_code/output/10_visualization"

# 모델 및 SAE 설정
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# 파라미터
BATCH_SIZE = 128  # 메모리 상황에 따라 조절
SAMPLE_SIZE = 1000 # 클래스별 최대 샘플 수

REFUSAL_KEYWORDS = [
        "don't know", "not sure", "sorry", "uncertain", "no information", 
        "i cannot", "apologize", "i can't", "coudln't", 
        "unable to verify", "cannot verify", "could not verify", 
        "unable to confirm", "not able to confirm", "cannot confirm", 
        "unable to validate", "unable to identify", "unable to determine", 
        "couldn't find", "could not find", "do not have information", 
        "do not have enough information", "do not have access", 
        "not aware of", "not familiar with", "wasn't able to", 
        "need more information", "need more context"
]

# 결과 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 2. 리소스 로드 (Loading Resources)
# ==========================================
def load_resources():
    print(f"[{DEVICE}] Loading Resources...")
    
    # 1. Tokenizer & Model
    print(f"Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map=DEVICE, 
        torch_dtype=torch.float16
    )
    model.eval()

    # 2. Pipeline
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_PATH}")
    print(f"Loading Pipeline from: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)

    # 3. Config
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    print(f"Loading Config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    return model, tokenizer, pipeline, config

def prepare_dataset(path):
    print(f"Loading Dataset from: {path}")
    df = pd.read_csv(path)
    
    # Label이 'Correct'인 경우 (Known/Confident)
    df_known = df[df['label'] == 'Correct'].copy()
    
    # Refusal 키워드가 포함된 경우 (Unknown/Uncertain)
    if 'is_refusal' not in df.columns:
        df['is_refusal'] = df['prediction'].apply(
            lambda x: any(k in str(x).lower() for k in REFUSAL_KEYWORDS)
        )
    df_refusal = df[df['is_refusal'] == True].copy()
    
    # 밸런싱 (Under-sampling)
    n_sample = min(len(df_known), len(df_refusal), SAMPLE_SIZE)
    print(f"Found - Known: {len(df_known)}, Refusal: {len(df_refusal)}. Sampling {n_sample} each.")
    
    df_balanced = pd.concat([
        df_known.sample(n_sample, random_state=42),
        df_refusal.sample(n_sample, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Target 설정 (0: Known, 1: Unknown/Refusal)
    df_balanced['target'] = df_balanced['label'].apply(lambda x: 0 if x == 'Correct' else 1)
    
    print(f"Final Dataset Size: {len(df_balanced)}")
    return df_balanced

# ==========================================
# 3. 피처 추출 (Feature Extraction)
# ==========================================
def extract_sae_features(model, tokenizer, df_data, config):
    prompts = df_data['question'].tolist()
    labels = df_data['target'].values
    
    layer_indices = config['layer_indices']
    feature_indices = config['feature_indices']
    
    # 결과 저장용 배열 초기화
    X = np.zeros((len(prompts), len(feature_indices)))
    
    # 레이어별로 그룹화하여 처리 (SAE 로딩 최소화)
    feat_df = pd.DataFrame({'layer': layer_indices, 'feature_idx': feature_indices})
    feat_df['col_idx'] = feat_df.index
    grouped = feat_df.groupby('layer')
    
    print(f"\nExtracting Features using SAE ({len(layer_indices)} features total)...")
    
    for layer, group in tqdm(grouped, desc="Layer Scan"):
        layer = int(layer)
        
        # SAE 로드
        try:
            # sae_lens 최신 버전은 (sae, cfg, sparsity) 튜플 반환
            sae, _, _ = SAE.from_pretrained(
                release=SAE_RELEASE, 
                sae_id=SAE_ID_TEMPLATE.format(layer=layer), 
                device=DEVICE
            )
        except Exception as e:
            print(f"Warning: Failed to load SAE for layer {layer} with standard method. Retrying... Error: {e}")
            # 예외 처리: 구형 버전 대응 등
            sae = SAE.from_pretrained(
                release=SAE_RELEASE, 
                sae_id=SAE_ID_TEMPLATE.format(layer=layer), 
                device=DEVICE
            )
            if isinstance(sae, tuple): sae = sae[0]
            
        sae.eval()
        
        target_feats = group['feature_idx'].astype(int).values
        target_cols = group['col_idx'].values
        
        # 배치 처리
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                # LLM Forward Pass
                out = model(**inputs, output_hidden_states=True)
                
                # Hidden State 추출 (Layer L의 출력은 hidden_states[L+1])
                # 마지막 토큰 기준
                resid = out.hidden_states[layer + 1][:, -1, :]
                
                # SAE Encoding
                feature_acts = sae.encode(resid)
                
                # 필요한 Feature만 추출하여 저장
                X[i:i+len(batch_prompts), target_cols] = feature_acts[:, target_feats].cpu().numpy()
        
        # 메모리 정리
        del sae
        torch.cuda.empty_cache()
        
    return X, labels

# ==========================================
# 4. 시각화 (Visualization)
# ==========================================
def visualize_results(X, y, pipeline, config, output_dir):
    print("\nStarting Visualization Process...")
    
    # 1. Test Split 및 Preprocessing
    # 시각화는 학습에 사용되지 않은(것으로 가정하는) Test Set으로 수행
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # [중요] Pipeline 입력 전 Log 변환 수행
    # 기존 코드 로직 유지: Log1p 변환 후 Pipeline에 주입
    X_test_log = np.log1p(X_test)
    
    # 예측 확률 계산
    y_probs = pipeline.predict_proba(X_test_log)[:, 1]
    
    # ---------------------------------------------------------
    # (A) Feature Importance Plot
    # ---------------------------------------------------------
    print(" - Plotting Feature Importance...")
    try:
        # Pipeline 단계 이름 확인 (보통 'logisticregressioncv' 또는 'logisticregression')
        if 'logisticregressioncv' in pipeline.named_steps:
            regressor = pipeline.named_steps['logisticregressioncv']
        elif 'logisticregression' in pipeline.named_steps:
            regressor = pipeline.named_steps['logisticregression']
        else:
            # 스텝 이름을 모를 경우 마지막 스텝 사용
            regressor = pipeline.steps[-1][1]
            
        coeffs = regressor.coef_[0]
        feature_names = [f"L{l}-F{f}" for l, f in zip(config['layer_indices'], config['feature_indices'])]
        
        imp_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeffs})
        imp_df['Abs_Coeff'] = imp_df['Coefficient'].abs()
        top_features = imp_df.nlargest(20, 'Abs_Coeff') # Top 20
        
        plt.figure(figsize=(12, 8))
        # 색상 매핑: 양수(Red, Refusal 유발) / 음수(Blue, Known 유발)
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_features['Coefficient']]
        
        sns.barplot(x='Coefficient', y='Feature', data=top_features, hue='Feature', palette=colors, legend=False)
        
        plt.title('Top 20 SAE Features for Uncertainty Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Logistic Regression Coefficient', fontsize=14)
        plt.ylabel('SAE Feature ID (Layer-Feature)', fontsize=14)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 캡션
        plt.figtext(0.5, 0.02, 
                    "Red (>0): Increases Probability of Refusal/Uncertainty\nBlue (<0): Increases Probability of Correct Answer", 
                    ha="center", fontsize=11, bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"gray"})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1]) # 캡션 공간 확보
        plt.savefig(f"{output_dir}/feature_importance_top20.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

    # ---------------------------------------------------------
    # (B) Probability Distribution Plot
    # ---------------------------------------------------------
    print(" - Plotting Probability Distribution...")
    plt.figure(figsize=(10, 6))
    
    probs_known = y_probs[y_test == 0]
    probs_refusal = y_probs[y_test == 1]
    
    sns.kdeplot(probs_known, color='#1f77b4', fill=True, label='Known (Correct)', alpha=0.3)
    sns.kdeplot(probs_refusal, color='#d62728', fill=True, label='Unknown (Refusal)', alpha=0.3)
    
    plt.title('Detector Confidence Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Probability of "Unknown"', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12, loc='upper center')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Separation Metric (예: AUPR/AUROC 등을 제목에 넣을 수도 있음)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_distribution.png", dpi=300)
    plt.close()

# ==========================================
# 5. 실행 (Main)
# ==========================================
def main():
    print("=== Starting SAE Visualization Pipeline ===")
    
    # 1. 로드
    model, tokenizer, pipeline, config = load_resources()
    
    # 2. 데이터 준비
    df_data = prepare_dataset(RAW_DATA_PATH)
    
    # 3. 피처 추출
    X, y = extract_sae_features(model, tokenizer, df_data, config)
    
    # 4. 시각화
    visualize_results(X, y, pipeline, config, OUTPUT_DIR)
    
    print(f"\nAll tasks completed. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()