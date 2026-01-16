import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm
import os
import gc
import json
import joblib  # [추가] 모델 저장을 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0"

# [Input] T-Test 결과 파일
STAT_RESULT_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_sae_t_test_results_0108.csv"
RAW_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context_evaluated.csv"

OUTPUT_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/7_optimized_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# [부스터 1] 피처 개수 대폭 증가
TOP_K_FEATURES = 500  
SAMPLE_SIZE = 2000 
BATCH_SIZE = 32

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

# ==========================================
# 2. 피처 선정 및 데이터 준비
# ==========================================
def select_top_features():
    print(f"Loading T-Test Stats from {STAT_RESULT_PATH}...")
    df = pd.read_csv(STAT_RESULT_PATH)
    
    # 절대값 T-Score 기준 정렬
    df['abs_t_score'] = df['t_score'].abs()
    top_features = df.nlargest(TOP_K_FEATURES, 'abs_t_score').reset_index(drop=True)
    
    print(f"Selected Top {TOP_K_FEATURES} features based on T-Score.")
    return top_features

def prepare_clean_dataset(path):
    print("Preparing Dataset...")
    df = pd.read_csv(path)
    
    df_known = df[df['label'] == 'Correct'].copy()
    
    if 'is_refusal' not in df.columns:
        df['is_refusal'] = df['prediction'].apply(lambda x: any(k in str(x).lower() for k in REFUSAL_KEYWORDS))
    
    # Unknown: 확실한 거절만 (Refusal)
    df_refusal = df[df['is_refusal'] == True].copy()
    
    print(f"Pool - Known: {len(df_known)}, Refusal: {len(df_refusal)}")
    
    # 최대한 많이 쓰되 밸런스 맞춤
    n_sample = min(len(df_known), len(df_refusal))
    if SAMPLE_SIZE > 0:
        n_sample = min(n_sample, SAMPLE_SIZE)
        
    df_balanced = pd.concat([
        df_known.sample(n_sample, random_state=42),
        df_refusal.sample(n_sample, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_balanced['target'] = df_balanced['label'].apply(lambda x: 0 if x == 'Correct' else 1)
    
    print(f"Final Eval Dataset: {len(df_balanced)} samples (Balanced)")
    return df_balanced

# ==========================================
# 3. 피처 추출 (Batch Processing)
# ==========================================
def extract_features(model, tokenizer, df_data, feature_list_df):
    prompts = df_data['question'].tolist()
    labels = df_data['target'].values
    
    X = np.zeros((len(prompts), len(feature_list_df)))
    grouped = feature_list_df.groupby('layer')
    
    print("\nExtracting Features from SAEs...")
    for layer, group in tqdm(grouped, desc="Layer Scan"):
        layer = int(layer)
        try:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)[0]
        except:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)
        sae.eval()
        
        feat_indices = group['feature_idx'].astype(int).values
        col_indices = group.index.values
        
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_p = prompts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                resid = out.hidden_states[layer + 1][:, -1, :]
                acts = sae.encode(resid)
                
                # 추출
                X[i:i+len(batch_p), col_indices] = acts[:, feat_indices].cpu().numpy()
        
        del sae
        torch.cuda.empty_cache()
        
    return X, labels

# ==========================================
# 4. 평가, 최적화 및 저장 (Evaluation & Saving)
# ==========================================
def evaluate_optimized_and_save(X, y, feature_df, output_dir):
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n[Optimization Step]")
    
    # [부스터 2] 로그 변환 (SAE 피처는 Long-tail 분포이므로 로그를 취하면 성능 급상승)
    print("Applying Log Transformation (log1p)...")
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    
    # [부스터 3] LogisticRegressionCV
    print("Training LogisticRegression with Cross-Validation...")
    
    clf = make_pipeline(
        RobustScaler(), 
        LogisticRegressionCV(
            Cs=10,               
            cv=5,                
            class_weight='balanced', 
            max_iter=5000,       
            scoring='roc_auc',
            n_jobs=-1            
        )
    )
    
    clf.fit(X_train_log, y_train)
    
    # Best C 확인
    best_c = clf.named_steps['logisticregressioncv'].C_[0]
    print(f"Best Regularization C: {best_c}")
    
    # Predict
    y_pred = clf.predict(X_test_log)
    y_prob = clf.predict_proba(X_test_log)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f" [Optimized Performance]")
    print(f" AUROC: {auc:.4f}")
    print(f" Accuracy: {acc:.4f}")
    print("="*30)
    print(classification_report(y_test, y_pred, target_names=['Known', 'Refusal']))
    
    # Visualization
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Optimized ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title('Performance after Log-Transform & Hyperparam Tuning')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = f"{output_dir}/optimized_roc.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved ROC plot to {save_path}")

    # ==========================================
    # [추가] 모델 및 파라미터 저장 로직
    # ==========================================
    print("\n[Saving Model for RAG Inference]")
    
    # 1. 전체 파이프라인 저장 (Python용)
    pipeline_path = f"{output_dir}/uncertainty_detector_pipeline.pkl"
    joblib.dump(clf, pipeline_path)
    print(f"- Full pipeline saved to: {pipeline_path}")
    
    # 2. 경량화 JSON Config 저장 (RAG 시스템 이식용)
    # 파이프라인 내부 요소 추출
    scaler = clf.named_steps['robustscaler']
    regressor = clf.named_steps['logisticregressioncv']
    
    circuit_config = {
        # 어떤 피처를 뽑아야 하는지 (Layer, Feature Index)
        "layer_indices": feature_df['layer'].tolist(),
        "feature_indices": feature_df['feature_idx'].tolist(),
        
        # 학습된 모델 가중치 (Weights & Bias)
        "weights": regressor.coef_[0].tolist(),
        "bias": regressor.intercept_[0],
        
        # 스케일러 파라미터 (Log변환 후 적용될 값들)
        "scale_center": scaler.center_.tolist(),
        "scale_scale": scaler.scale_.tolist(),
        
        # 메타 정보
        "auroc": auc,
        "accuracy": acc
    }
    
    json_path = f"{output_dir}/rag_circuit_config.json"
    with open(json_path, "w") as f:
        json.dump(circuit_config, f, indent=4)
        
    print(f"- Lightweight RAG config saved to: {json_path}")
    print("Done.")

# ==========================================
# 5. Main
# ==========================================
def main():
    # Model Load
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16)
    model.eval()

    # 1. Select Top 500 Features
    top_features_df = select_top_features()
    
    # 2. Data Prep
    df_data = prepare_clean_dataset(RAW_DATA_PATH)
    
    # 3. Extract
    X, y = extract_features(model, tokenizer, df_data, top_features_df)
    
    # 4. Optimized Eval & Save (feature_df 전달 추가)
    evaluate_optimized_and_save(X, y, top_features_df, OUTPUT_DIR)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()