import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import textwrap

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0"

# [Input 1] 검증된 후보 피처 파일 (CSV)
CANDIDATE_FEATURES_PATH = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/4/final_verified_features.csv"

# [Input 2] 데이터셋 경로
RAW_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context_evaluated.csv"

TARGET_N_FEATURES = 10  # 최종 선택할 피처 개수 (Top 10)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# [Output] 결과 저장 경로
OUTPUT_DIR = f"/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/10_visualization_optimized_top_{TARGET_N_FEATURES}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
SAMPLE_SIZE = 2000 

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
def load_model_and_tokenizer():
    print("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    # 메모리 절약을 위해 float16 사용 (상황에 따라 bfloat16 권장)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16)
    model.eval()
    return model, tokenizer

def load_candidate_features(csv_path):
    print(f"Loading Candidate Features from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['layer'] = df['layer'].astype(int)
    df['feature_idx'] = df['feature_idx'].astype(int)
    
    candidates = {}
    feature_list = []
    for _, row in df.iterrows():
        l, f = int(row['layer']), int(row['feature_idx'])
        if l not in candidates:
            candidates[l] = []
        candidates[l].append(f)
        feature_list.append(f"L{l}-F{f}")
        
    print(f"Found {len(feature_list)} candidate features across {len(candidates)} layers.")
    return candidates, feature_list

def prepare_dataset(path):
    print("Preparing Dataset...")
    df = pd.read_csv(path)
    df_known = df[df['label'] == 'Correct'].copy()
    
    if 'is_refusal' not in df.columns:
        df['is_refusal'] = df['prediction'].apply(lambda x: any(k in str(x).lower() for k in REFUSAL_KEYWORDS))
    df_refusal = df[df['is_refusal'] == True].copy()
    
    n_sample = min(len(df_known), len(df_refusal), SAMPLE_SIZE)
    df_balanced = pd.concat([
        df_known.sample(n_sample, random_state=42),
        df_refusal.sample(n_sample, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_balanced['target'] = df_balanced['label'].apply(lambda x: 0 if x == 'Correct' else 1)
    print(f"Dataset Size: {len(df_balanced)}")
    return df_balanced

# ==========================================
# 3. 피처 추출 (CSV 후보군 기반)
# ==========================================
def extract_candidate_features(model, tokenizer, df_data, candidates_dict, feature_names_list):
    prompts = df_data['question'].tolist()
    labels = df_data['target'].values
    
    total_features = len(feature_names_list)
    X = np.zeros((len(prompts), total_features))
    feat_to_col = {name: i for i, name in enumerate(feature_names_list)}
    
    print("\nExtracting Features from Candidates...")
    sorted_layers = sorted(candidates_dict.keys())
    
    for layer in tqdm(sorted_layers, desc="Layer Scan"):
        target_feats = candidates_dict[layer]
        if not target_feats: continue
        
        try:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)[0]
        except:
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)
        sae.eval()
        
        current_layer_col_indices = [feat_to_col[f"L{layer}-F{f}"] for f in target_feats]
        
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_p = prompts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                resid = out.hidden_states[layer + 1][:, -1, :]
                acts = sae.encode(resid)
                X[i:i+len(batch_p), current_layer_col_indices] = acts[:, target_feats].cpu().numpy()
        
        del sae
        torch.cuda.empty_cache()
        
    return X, labels

# ==========================================
# 4. 최적 조합 선택 및 재학습 (The Fix)
# ==========================================
def optimize_feature_selection(X, y, feature_names):
    """
    RFE를 사용하여 상위 피처를 선택한 뒤, 
    ***선택된 피처만으로 경량 파이프라인을 재학습***하여 저장합니다.
    """
    print("\n" + "="*50)
    print(f"Optimizing Feature Selection (RFE - Top {TARGET_N_FEATURES})")
    print("="*50)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Log 변환
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    
    # 1. RFE 수행 (전체 피처 중 중요한 것 선별)
    base_model = LogisticRegression(
        penalty='l1', solver='liblinear', random_state=42, max_iter=1000, C=1.0 
    )
    
    # Scaler + RFE 파이프라인 (Selection 용도)
    selector_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selection', RFE(estimator=base_model, n_features_to_select=TARGET_N_FEATURES, step=1))
    ])
    
    print(f"Running RFE to select top {TARGET_N_FEATURES} features from {X.shape[1]} candidates...")
    selector_pipeline.fit(X_train_log, y_train)
    
    # 선택된 피처 마스크 추출
    rfe_step = selector_pipeline.named_steps['selection']
    selected_mask = rfe_step.support_
    selected_features = np.array(feature_names)[selected_mask]
    
    print(f" -> RFE Selected {len(selected_features)} features.")
    
    # =========================================================
    # [핵심 수정] 선택된 피처만으로 "새로운" 파이프라인 재학습 (Inference 최적화)
    # =========================================================
    print(f"\nTraining Inference Optimized Model (Expects {TARGET_N_FEATURES} inputs)...")
    
    # 선택된 컬럼만 추출
    X_train_selected = X_train_log[:, selected_mask]
    X_test_selected = X_test_log[:, selected_mask]
    
    # 새 파이프라인 정의 (RFE 단계 없이 Scaler -> LR 직결)
    inference_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l1', solver='liblinear', random_state=42, max_iter=1000, C=1.0
        ))
    ])
    
    # 재학습
    inference_pipeline.fit(X_train_selected, y_train)
    
    # 성능 검증
    y_pred_prob = inference_pipeline.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, (y_pred_prob > 0.5).astype(int))
    
    print(f" -> Optimized Model AUC: {auc:.4f}")
    print(f" -> Optimized Model Accuracy: {acc:.4f}")
    
    # ---------------------------------------------------------
    # 저장 1: 모델 파일 (.pkl)
    # RAG 시스템은 이 파일을 로드하며, 입력으로 정확히 10개의 값만 받습니다.
    # ---------------------------------------------------------
    model_save_path = f"{OUTPUT_DIR}/optimized_pipeline_top{TARGET_N_FEATURES}.pkl"
    joblib.dump(inference_pipeline, model_save_path)
    print(f" -> Saved Inference Model to {model_save_path}")
    
    # ---------------------------------------------------------
    # 저장 2: 피처 정보 파일 (.csv)
    # ---------------------------------------------------------
    # 모델의 계수 추출
    final_model = inference_pipeline.named_steps['classifier']
    final_coefs = final_model.coef_[0]
    
    top_features_data = []
    # 중요: selected_features의 순서와 final_coefs의 순서는 X_train_selected의 컬럼 순서와 일치함
    for feat_name, coef in zip(selected_features, final_coefs):
        # "L{layer}-F{idx}" 형식 파싱
        parts = feat_name.split('-') 
        layer = int(parts[0][1:])
        feature_idx = int(parts[1][1:])
        
        top_features_data.append({
            'layer': layer,
            'feature_idx': feature_idx,
            'coefficient': coef
        })
    
    df_top_features = pd.DataFrame(top_features_data)
    csv_save_path = f"{OUTPUT_DIR}/final_selected_features_top{TARGET_N_FEATURES}.csv"
    df_top_features.to_csv(csv_save_path, index=False)
    print(f" -> Saved Top {TARGET_N_FEATURES} features info to {csv_save_path}")

    # 시각화를 위한 정보 패키징
    selection_info = {
        'mask': selected_mask,          # 전체 중 어디가 선택되었는지 (시각화용 원본 X 참조 위해)
        'features': selected_features,
        'coefs': final_coefs,
        'auc': auc
    }
    
    return inference_pipeline, selection_info, X_test_log, y_test, selected_mask

# ==========================================
# 5. 시각화 및 해석
# ==========================================
def visualize_and_interpret(pipeline, selection_info, X_full, df_data, output_dir, selected_mask):
    feature_names = selection_info['features']
    coeffs = selection_info['coefs']
    
    imp_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeffs})
    imp_df['Abs_Coeff'] = imp_df['Coefficient'].abs()
    # 이미 Top N개이므로 정렬만 수행
    top_features = imp_df.sort_values(by='Abs_Coeff', ascending=False)

    # (1) Feature Importance Plot
    print("\nGenerating Feature Importance Plot...")
    plt.figure(figsize=(12, 8))
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_features['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=top_features, palette=colors, orient='h')
    plt.title(f'Top {len(top_features)} Optimized Features (AUC: {selection_info["auc"]:.3f})', fontsize=16)
    plt.xlabel('Importance (Coefficient)', fontsize=14)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimized_feature_importance.png", dpi=300)
    
    # (2) Probability Distribution
    print("Generating Probability Distribution...")
    # 전체 데이터에서 선택된 피처만 추출하여 예측
    X_log = np.log1p(X_full)
    X_selected = X_log[:, selected_mask]
    
    y_probs = pipeline.predict_proba(X_selected)[:, 1]
    labels = df_data['target'].values
    
    plt.figure(figsize=(10, 6))
    sns.histplot(y_probs[labels==0], color='blue', label='Known', kde=True, stat="density", alpha=0.4)
    sns.histplot(y_probs[labels==1], color='red', label='Refusal', kde=True, stat="density", alpha=0.4)
    plt.title('Probability Distribution of Optimized Model')
    plt.legend()
    plt.savefig(f"{output_dir}/optimized_prob_distribution.png", dpi=300)

    # (3) Qualitative Analysis
    print("Generating Qualitative Analysis Plot...")
    
    # 원본 X에서의 인덱스 찾기
    all_indices = np.arange(X_full.shape[1])
    original_col_indices = all_indices[selected_mask]
    
    n_feats = len(top_features)
    fig, axes = plt.subplots(n_feats, 2, figsize=(20, n_feats * 3), gridspec_kw={'width_ratios': [1, 3]})
    if n_feats == 1: axes = [axes] # 1개일 경우 처리
    fig.suptitle(f"Qualitative Analysis of Top {n_feats} Features", fontsize=20, y=1.02)

    interp_results = []

    for i, (idx, row) in enumerate(top_features.iterrows()):
        feat_name = row['Feature']
        coeff = row['Coefficient']
        role = "Refusal (Uncertainty)" if coeff > 0 else "Known (Confidence)"
        color = '#d62728' if coeff > 0 else '#1f77b4'
        
        # --- Bar ---
        ax_bar = axes[i][0] if n_feats > 1 else axes[0]
        sns.barplot(x=[coeff], y=[feat_name], ax=ax_bar, color=color, orient='h')
        ax_bar.set_xlim(top_features['Coefficient'].min()*1.1, top_features['Coefficient'].max()*1.1)
        ax_bar.axvline(0, color='black', linewidth=0.8)
        ax_bar.set_xlabel("Coefficient")
        ax_bar.set_title(f"{role}")

        # --- Examples ---
        ax_text = axes[i][1] if n_feats > 1 else axes[1]
        ax_text.axis('off')
        
        # 현재 피처의 원래 컬럼 인덱스를 찾아 activation 가져오기
        feat_loc_in_selected = np.where(feature_names == feat_name)[0][0]
        original_col_idx = original_col_indices[feat_loc_in_selected]
        
        acts = X_full[:, original_col_idx]
        top_k = np.argsort(acts)[::-1][:3]
        
        example_texts = []
        for j, top_i in enumerate(top_k):
            val = acts[top_i]
            if val <= 0: continue
            text = df_data.loc[top_i, 'question']
            wrapped_text = textwrap.fill(text, width=100) 
            example_texts.append(f"[{j+1}] (Act: {val:.2f}) {wrapped_text}")
            
        full_text = "\n\n".join(example_texts) if example_texts else "No significant activation found."
        
        ax_text.text(0, 0.5, full_text, fontsize=12, va='center', ha='left', wrap=True, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.5'))

        interp_results.append({"feature": feat_name, "coef": coeff, "examples": example_texts})

    plt.tight_layout()
    plt.savefig(f"{output_dir}/qualitative_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved qualitative analysis to {output_dir}/qualitative_analysis.png")
        
    with open(f"{output_dir}/optimized_interpretation.json", "w") as f:
        json.dump(interp_results, f, indent=4)

# ==========================================
# 6. Main
# ==========================================
def main():
    model, tokenizer = load_model_and_tokenizer()
    df_data = prepare_dataset(RAW_DATA_PATH)
    candidates_dict, feature_names_list = load_candidate_features(CANDIDATE_FEATURES_PATH)
    
    # 1. 모든 후보 피처 추출 (ex: 50개)
    X_full, y_full = extract_candidate_features(model, tokenizer, df_data, candidates_dict, feature_names_list)
    
    # 2. 최적 피처 선택 및 재학습 (결과: 10개 입력을 받는 모델)
    pipeline, selection_info, _, _, selected_mask = optimize_feature_selection(X_full, y_full, np.array(feature_names_list))
    
    # 3. 시각화
    visualize_and_interpret(pipeline, selection_info, X_full, df_data, OUTPUT_DIR, selected_mask)
    
    print("\nOptimization & Visualization Complete!")

if __name__ == "__main__":
    main()