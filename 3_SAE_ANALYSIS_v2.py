import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm
import os
import gc
import torch.multiprocessing as mp
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
NUM_GPUS = 4

INPUT_PATH = "/home/seoi0215/trend/SAE/data_classification/output/images/hotpot_qa_llama3_results_train_no_context_evaluated.csv"
OUTPUT_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_sae_t_test_results_0108.csv"
ANALYSIS_DIR = "/home/seoi0215/trend/SAE/data_classification/analysis_results_final_0108"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"
TARGET_LAYERS = list(range(32))

BATCH_SIZE = 64  # HF 모델은 메모리를 좀 더 쓸 수 있으므로 배치 사이즈 조절 필요시 수정
TEXT_COLUMN = "question"
LABEL_COLUMN = "label"
FUZZY_SCORE_COLUMN = "fuzzy_score"
CORRECT_LABEL = "Correct"

TOP_K_ANALYSIS = 10 

# ==========================================
# 2. 유틸리티 함수 (기존 유지)
# ==========================================
def balance_binary_tensors(y_true_np, predictions_np):
    class_0_indices = np.where(y_true_np == 0)[0]
    class_1_indices = np.where(y_true_np == 1)[0]
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = np.random.choice(class_0_indices, min_class_size, replace=False)
    else:
        class_1_indices = np.random.choice(class_1_indices, min_class_size, replace=False)
    
    balanced_indices = np.sort(np.concatenate([class_0_indices, class_1_indices]))
    return y_true_np[balanced_indices], predictions_np[balanced_indices]

def find_optimal_threshold(y_true, scores):
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = [f1_score(y_true, (scores >= threshold).astype(int)) for threshold in thresholds]
    optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_roc_idx = np.argmax(j_scores)
    optimal_roc_threshold = roc_thresholds[optimal_roc_idx]
    
    return optimal_f1_threshold, optimal_roc_threshold

# ==========================================
# 3. Phase 1: T-test 통계 집계 (Worker)
# ==========================================
def process_chunk(rank, gpu_ids, chunk_df, return_dict):
    try:
        device = f"cuda:{rank}"
        print(f"[GPU {rank}] Starting process... Samples: {len(chunk_df)}")

        # 라벨링 (0: Known/Correct, 1: Unknown/Incorrect)
        labels = chunk_df[LABEL_COLUMN].astype(str).str.strip().tolist()
        is_known_mask = torch.tensor([label == CORRECT_LABEL for label in labels], device=device)
        
        # [수정] Hugging Face 모델 로드
        print(f"[GPU {rank}] Loading HF Model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map=device, 
            torch_dtype=torch.float16,
            output_hidden_states=True  # [핵심] Hidden State 추출 활성화
        )
        model.eval()

        prompts = chunk_df[TEXT_COLUMN].tolist()
        layer_stats = {}

        # 각 레이어별로 SAE 로드 및 통계 계산
        for layer in tqdm(TARGET_LAYERS, desc=f"GPU {rank} Layers", position=rank):
            try:
                # SAE 로드
                sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=device)[0]
                sae.to(dtype=torch.float16)
                sae.eval()
            except Exception as e:
                print(f"[GPU {rank}] SAE Load Error Layer {layer}: {e}")
                continue

            d_sae = sae.cfg.d_sae
            
            # 통계량 텐서 초기화
            stats = {
                'known_sum': torch.zeros(d_sae, device=device, dtype=torch.float32),
                'known_sq': torch.zeros(d_sae, device=device, dtype=torch.float32),
                'known_count': 0,
                'unknown_sum': torch.zeros(d_sae, device=device, dtype=torch.float32),
                'unknown_sq': torch.zeros(d_sae, device=device, dtype=torch.float32),
                'unknown_count': 0
            }
            
            # 배치 처리
            for i in range(0, len(prompts), BATCH_SIZE):
                batch_prompts = prompts[i : i + BATCH_SIZE]
                batch_mask = is_known_mask[i : i + BATCH_SIZE]
                
                # [수정] Chat Template 적용 및 토크나이징
                # Llama-3-Instruct는 템플릿 적용이 필수적임
                formatted_batch = []
                for p in batch_prompts:
                    msgs = [{"role": "user", "content": p}]
                    formatted_batch.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

                inputs = tokenizer(formatted_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                
                with torch.no_grad():
                    # HF Model Forward
                    outputs = model(**inputs)
                    
                    # Hidden State 추출
                    # outputs.hidden_states는 (emb, layer_0, ..., layer_31) 순서 튜플
                    # 따라서 Layer N의 Output(resid_post)은 index N+1에 해당
                    layer_hidden = outputs.hidden_states[layer + 1]
                    
                    # 마지막 토큰(Last Token) 추출 (Left Padding 고려)
                    # padding_side='left'이므로 마지막 토큰은 항상 -1 인덱스
                    resid_acts = layer_hidden[:, -1, :] # [batch, d_model]
                    
                    # SAE Encoding
                    feature_acts = sae.encode(resid_acts) # [batch, d_sae]
                    
                    # Known 그룹 집계
                    if batch_mask.sum() > 0:
                        acts = feature_acts[batch_mask]
                        stats['known_sum'] += acts.sum(dim=0)
                        stats['known_sq'] += (acts ** 2).sum(dim=0)
                        stats['known_count'] += acts.shape[0]
                    
                    # Unknown 그룹 집계
                    if (~batch_mask).sum() > 0:
                        acts = feature_acts[~batch_mask]
                        stats['unknown_sum'] += acts.sum(dim=0)
                        stats['unknown_sq'] += (acts ** 2).sum(dim=0)
                        stats['unknown_count'] += acts.shape[0]
                    
                    del resid_acts, feature_acts, outputs, layer_hidden
            
            # CPU로 이동하여 결과 저장
            layer_stats[layer] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
            del sae
            torch.cuda.empty_cache()
            
        return_dict[rank] = layer_stats
        print(f"[GPU {rank}] Finished.")

    except Exception as e:
        print(f"[GPU {rank}] Critical Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = None

# ==========================================
# 4. Phase 2: 심층 분석 (Top Feature Re-eval)
# ==========================================
def deep_dive_analysis(top_features_df, raw_df):
    print("\n[Phase 2] Starting Deep Dive Analysis on Top Features...")
    
    df_known = raw_df[raw_df[LABEL_COLUMN] == CORRECT_LABEL]
    df_unknown = raw_df[raw_df[LABEL_COLUMN] != CORRECT_LABEL]
    
    n_sample = min(100, len(df_known), len(df_unknown))
    balanced_df = pd.concat([
        df_known.sample(n_sample, random_state=42),
        df_unknown.sample(n_sample, random_state=42)
    ])
    
    prompts = balanced_df[TEXT_COLUMN].tolist()
    y_true = (balanced_df[LABEL_COLUMN] != CORRECT_LABEL).astype(int).values 
    
    device = "cuda:0"
    
    # [수정] HF Model 로드
    print("Loading HF Model for Deep Dive...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map=device, 
        torch_dtype=torch.float16, 
        output_hidden_states=True
    )
    model.eval()
        
    analysis_results = []
    unique_layers = top_features_df['layer'].unique()
    
    # 분석 데이터 전처리 (한 번만 수행)
    formatted_prompts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    
    # 전체 데이터 토크나이징 (메모리 괜찮으면 한번에)
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    for layer in unique_layers:
        print(f"Analyzing Layer {layer}...")
        sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=device)[0]
        sae.to(dtype=torch.float16)
        
        target_feats = top_features_df[top_features_df['layer'] == layer]
        
        with torch.no_grad():
            outputs = model(**inputs)
            resid = outputs.hidden_states[layer + 1][:, -1, :] # [batch, d_model]
            acts = sae.encode(resid) # [batch, d_sae]
            
            for _, row in target_feats.iterrows():
                feat_idx = int(row['feature_idx'])
                feat_scores = acts[:, feat_idx].cpu().float().numpy()
                
                # 메트릭 계산
                y_bal, scores_bal = balance_binary_tensors(y_true, feat_scores)
                
                # 예외 처리: 모든 점수가 0인 경우 (Dead Feature)
                if scores_bal.max() == 0:
                    print(f"Feature {feat_idx} is dead (all zeros). Skipping.")
                    continue

                opt_f1_thresh, opt_roc_thresh = find_optimal_threshold(y_bal, scores_bal)
                auroc = roc_auc_score(y_bal, scores_bal)
                y_pred = (scores_bal >= opt_f1_thresh).astype(int)
                
                res = {
                    'layer': layer,
                    'feature_idx': feat_idx,
                    't_score': row['t_score'],
                    'auroc': auroc,
                    'f1_score': f1_score(y_bal, y_pred),
                    'optimal_threshold': opt_f1_thresh,
                    'precision': precision_score(y_bal, y_pred),
                    'recall': recall_score(y_bal, y_pred)
                }
                analysis_results.append(res)
                
                # 시각화
                plt.figure(figsize=(6, 4))
                sns.histplot(x=feat_scores[y_true==0], color='blue', label='Known', alpha=0.5, bins=20)
                sns.histplot(x=feat_scores[y_true==1], color='red', label='Unknown', alpha=0.5, bins=20)
                plt.axvline(opt_f1_thresh, color='green', linestyle='--', label=f'Thresh {opt_f1_thresh:.2f}')
                plt.title(f"L{layer}-F{feat_idx} (AUROC: {auroc:.3f})")
                plt.legend()
                plt.savefig(f"{ANALYSIS_DIR}/hist_L{layer}_F{feat_idx}.png")
                plt.close()

    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv(f"{ANALYSIS_DIR}/deep_dive_metrics.csv", index=False)
    print(f"Deep dive analysis saved to {ANALYSIS_DIR}/deep_dive_metrics.csv")
    print(results_df.sort_values('auroc', ascending=False).head())

# ==========================================
# 5. 메인 함수 (기존 유지)
# ==========================================
def main():
    mp.set_start_method('spawn', force=True)
    
    # 1. 데이터 로드 및 엄격 필터링
    print(f"Loading Dataset...")
    df = pd.read_csv(INPUT_PATH)
    
    df[FUZZY_SCORE_COLUMN] = pd.to_numeric(df[FUZZY_SCORE_COLUMN], errors='coerce').fillna(0)
    cond_known = (df[LABEL_COLUMN] == CORRECT_LABEL) & (df[FUZZY_SCORE_COLUMN] >= 80)
    cond_unknown = (df[LABEL_COLUMN] != CORRECT_LABEL) & (df[FUZZY_SCORE_COLUMN] > 0) & (df[FUZZY_SCORE_COLUMN] <= 20)
    
    df_filtered = df[cond_known | cond_unknown].copy()
    print(f"Filtered Data: {len(df_filtered)} samples")
    
    # 2. Phase 1: 멀티 GPU 통계 집계
    indices = np.array_split(df_filtered.index, NUM_GPUS)
    chunks = [df_filtered.loc[idx].reset_index(drop=True) for idx in indices]
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(NUM_GPUS):
        if len(chunks[rank]) == 0: continue
        p = mp.Process(target=process_chunk, args=(rank, list(range(NUM_GPUS)), chunks[rank], return_dict))
        p.start()
        processes.append(p)
    
    for p in processes: p.join()
    
    # 3. 결과 집계 및 T-test 계산
    print("Aggregating statistics...")
    final_rows = []
    
    for layer in tqdm(TARGET_LAYERS):
        agg = {'k_sum': 0, 'k_sq': 0, 'k_cnt': 0, 'u_sum': 0, 'u_sq': 0, 'u_cnt': 0}
        
        for rank in range(NUM_GPUS):
            res = return_dict.get(rank, {}).get(layer)
            if res:
                agg['k_sum'] += res['known_sum']
                agg['k_sq'] += res['known_sq']
                agg['k_cnt'] += res['known_count']
                agg['u_sum'] += res['unknown_sum']
                agg['u_sq'] += res['unknown_sq']
                agg['u_cnt'] += res['unknown_count']
        
        if agg['k_cnt'] == 0 or agg['u_cnt'] == 0: continue
        
        mean_k = agg['k_sum'] / agg['k_cnt']
        mean_u = agg['u_sum'] / agg['u_cnt']
        var_k = (agg['k_sq'] / agg['k_cnt']) - (mean_k ** 2)
        var_u = (agg['u_sq'] / agg['u_cnt']) - (mean_u ** 2)
        
        epsilon = 1e-8
        t_score = (mean_k - mean_u) / torch.sqrt((var_k / agg['k_cnt']) + (var_u / agg['u_cnt']) + epsilon)
        
        layer_res = pd.DataFrame({
            'layer': layer,
            'feature_idx': range(len(mean_k)),
            'mean_known': mean_k.numpy(),
            'mean_unknown': mean_u.numpy(),
            't_score': t_score.numpy(),
            'abs_t_score': t_score.abs().numpy()
        })
        final_rows.append(layer_res)

    if not final_rows:
        print("No results.")
        return

    full_df = pd.concat(final_rows, ignore_index=True)
    full_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Phase 1 Complete. Results saved to {OUTPUT_PATH}")
    
    # 4. Phase 2: Top Feature 선정 및 심층 분석
    top_features = full_df.nlargest(TOP_K_ANALYSIS, 'abs_t_score')
    deep_dive_analysis(top_features, df_filtered)

if __name__ == "__main__":
    main()