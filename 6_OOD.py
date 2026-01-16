import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm
import os
import gc
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0"

# [Input 1] 저장된 모델 및 설정 경로 (이전 코드 output 폴더)
SAVED_MODEL_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/7_optimized_eval"
PIPELINE_PATH = f"{SAVED_MODEL_DIR}/uncertainty_detector_pipeline.pkl"
CONFIG_PATH = f"{SAVED_MODEL_DIR}/rag_circuit_config.json"

# [Input 2] 새로운 평가용 데이터셋 (TriviaQA - Excel 파일)
NEW_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/trivia_qa_llama3_results_train_no_context_evaluated.xlsx"

# [Output] 결과 저장 경로
OUTPUT_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/9_cross_domain_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# [실험 파라미터]
TEST_SAMPLE_SIZE = 1000  # 평가할 샘플 수
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
# 2. 리소스 및 데이터 로드
# ==========================================
def load_resources():
    print("Loading LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16)
    model.eval()
    
    print(f"Loading Saved Detector Pipeline from {PIPELINE_PATH}...")
    pipeline = joblib.load(PIPELINE_PATH)
    
    print(f"Loading Feature Config from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    return model, tokenizer, pipeline, config

def prepare_new_dataset(path, sample_size):
    print(f"Loading New Dataset from {path}...")
    # Excel 파일 로드
    df = pd.read_excel(path)
    
    # 데이터 정제 (Known vs Refusal)
    df_known = df[df['label'] == 'Correct'].copy()
    
    if 'is_refusal' not in df.columns:
        # prediction 컬럼이 문자열이 아닐 경우를 대비해 str() 변환
        df['is_refusal'] = df['prediction'].apply(lambda x: any(k in str(x).lower() for k in REFUSAL_KEYWORDS))
    
    df_refusal = df[df['is_refusal'] == True].copy()
    
    print(f"Pool - Known: {len(df_known)}, Refusal: {len(df_refusal)}")
    
    if len(df_refusal) == 0:
        raise ValueError("No refusal samples found in the new dataset! Check keywords or data.")

    # 밸런스 샘플링 (각 500개씩, 부족하면 최대치)
    n_sample = min(sample_size // 2, len(df_known), len(df_refusal))
    
    df_balanced = pd.concat([
        df_known.sample(n_sample, random_state=42),
        df_refusal.sample(n_sample, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_balanced['target'] = df_balanced['label'].apply(lambda x: 0 if x == 'Correct' else 1)
    
    print(f"Final Test Dataset: {len(df_balanced)} samples (Balanced)")
    return df_balanced

# ==========================================
# 3. 피처 추출 (Extract based on Config)
# ==========================================
def extract_features_from_config(model, tokenizer, df_data, config):
    prompts = df_data['question'].tolist()
    labels = df_data['target'].values
    
    # Config에서 필요한 피처 정보 복원
    layer_indices = config['layer_indices']
    feature_indices = config['feature_indices']
    
    # 결과를 담을 행렬 (Samples x Features)
    # 주의: 컬럼 순서가 학습 때와 정확히 일치해야 함
    X = np.zeros((len(prompts), len(feature_indices)))
    
    # 처리를 위해 DataFrame으로 변환하여 그룹화
    feat_df = pd.DataFrame({'layer': layer_indices, 'feature_idx': feature_indices})
    # 원래 순서(col_idx)를 기억해야 함
    feat_df['col_idx'] = feat_df.index 
    
    grouped = feat_df.groupby('layer')
    
    print("\nExtracting Features (Cross-Domain)...")
    for layer, group in tqdm(grouped, desc="Layer Scan"):
        layer = int(layer)
        try:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)[0]
        except:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)
        sae.eval()
        
        target_feats = group['feature_idx'].astype(int).values
        target_cols = group['col_idx'].values
        
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_p = prompts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                resid = out.hidden_states[layer + 1][:, -1, :]
                acts = sae.encode(resid)
                
                # 추출 및 저장
                X[i:i+len(batch_p), target_cols] = acts[:, target_feats].cpu().numpy()
        
        del sae
        torch.cuda.empty_cache()
        
    return X, labels

# ==========================================
# 4. 평가 (Evaluation)
# ==========================================
def evaluate_generalization(X, y, pipeline, output_dir):
    print("\nEvaluating Generalization Performance...")
    
    # [중요] 학습할 때와 동일한 전처리를 수동으로 해줘야 함
    # 저장된 파이프라인은 RobustScaler부터 시작하므로, np.log1p는 여기서 수행
    print("Applying Log Transformation (log1p)...")
    X_log = np.log1p(X)
    
    # 예측
    y_pred = pipeline.predict(X_log)
    y_prob = pipeline.predict_proba(X_log)[:, 1]
    
    # 지표 계산
    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    
    print("\n" + "="*40)
    print(f" [TriviaQA Generalization Result]")
    print(f" AUROC: {auc:.4f}")
    print(f" Accuracy: {acc:.4f}")
    print("="*40)
    print(classification_report(y, y_pred, target_names=['Known', 'Refusal']))
    
    # 시각화
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.plot(fpr, tpr, color='green', lw=2, label=f'TriviaQA (OOD) AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title('Generalization Performance on Unseen Dataset (TriviaQA)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = f"{output_dir}/trivia_qa_generalization_roc.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved ROC plot to {save_path}")

# ==========================================
# 5. Main
# ==========================================
def main():
    # 1. Load Resources
    model, tokenizer, pipeline, config = load_resources()
    
    # 2. Load New Data (TriviaQA)
    df_new = prepare_new_dataset(NEW_DATA_PATH, TEST_SAMPLE_SIZE)
    
    # 3. Extract Features (Using saved config schema)
    X, y = extract_features_from_config(model, tokenizer, df_new, config)
    
    # 4. Evaluate
    evaluate_generalization(X, y, pipeline, OUTPUT_DIR)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()