import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

# XGBoost 설치 필요 (pip install xgboost)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Skipping XGBoost.")

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = "cuda:3"

# [Input] T-Test 결과 파일 & 원본 데이터
STAT_RESULT_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_sae_t_test_results_0108.csv"
RAW_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context_evaluated.csv"

OUTPUT_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/8_model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# [실험 파라미터]
TOP_K_FEATURES = 500   # 피처 개수 넉넉하게
SAMPLE_SIZE = 1000     # 데이터 샘플 수
BATCH_SIZE = 128

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
# 2. 데이터 준비 함수 (기존과 동일)
# ==========================================
def select_top_features():
    print(f"Loading Stats from {STAT_RESULT_PATH}...")
    df = pd.read_csv(STAT_RESULT_PATH)
    df['abs_t_score'] = df['t_score'].abs()
    # T-Score 절대값 기준 상위 K개
    top_features = df.nlargest(TOP_K_FEATURES, 'abs_t_score').reset_index(drop=True)
    return top_features

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
    return df_balanced

def extract_features(model, tokenizer, df_data, feature_list_df):
    prompts = df_data['question'].tolist()
    labels = df_data['target'].values
    X = np.zeros((len(prompts), len(feature_list_df)))
    
    grouped = feature_list_df.groupby('layer')
    print("\nExtracting Features...")
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
                X[i:i+len(batch_p), col_indices] = acts[:, feat_indices].cpu().numpy()
        del sae
        torch.cuda.empty_cache()
    return X, labels

# ==========================================
# 3. 모델 비교 및 시각화 (핵심)
# ==========================================
def compare_models(X, y, output_dir):
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # [중요] Log Transformation (SAE 피처 분포 보정)
    X_train = np.log1p(X_train)
    X_test = np.log1p(X_test)
    
    # 비교할 모델 리스트 정의
    models = {
        "Logistic Regression": make_pipeline(RobustScaler(), LogisticRegression(class_weight='balanced', max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "MLP (Neural Net)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42))
    }
    
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    results = []
    plt.figure(figsize=(10, 8))
    
    print("\n[Starting Model Comparison]")
    
    # 각 모델 학습 및 평가
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({"Model": name, "AUROC": auc, "Accuracy": acc})
        
        # ROC Curve 그리기
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

    # ROC Plot 스타일링
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performance Comparison: Linear vs Non-Linear Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = f"{output_dir}/model_comparison_roc.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved ROC Comparison to {save_path}")
    
    # Bar Chart (Metric Comparison)
    results_df = pd.DataFrame(results)
    print("\n[Evaluation Results]")
    print(results_df)
    results_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    melted_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Accuracy & AUROC by Model Type")
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    bar_path = f"{output_dir}/model_comparison_bar.png"
    plt.savefig(bar_path, dpi=300)
    print(f"Saved Bar Chart to {bar_path}")

# ==========================================
# 4. Main
# ==========================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16)
    model.eval()

    top_features_df = select_top_features()
    df_data = prepare_dataset(RAW_DATA_PATH)
    X, y = extract_features(model, tokenizer, df_data, top_features_df)
    
    compare_models(X, y, OUTPUT_DIR)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()