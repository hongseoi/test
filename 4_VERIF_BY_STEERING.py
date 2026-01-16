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

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0"

# [Input] 코드1에서 생성된 T-Test 결과 파일 경로
STAT_RESULT_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_sae_t_test_results_0108.csv"

# [Input] 원본 데이터 (Steering용 질문 추출용)
RAW_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context_evaluated.csv"

# [Output] 최종 결과 저장 경로
OUTPUT_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV_PATH = f"{OUTPUT_DIR}/final_verified_features.csv"

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# [실험 파라미터]
TOP_K_CANDIDATES = 50   # T-Score 기준 상위 50개만 정밀 검증
STEERING_COEFF = 150.0  # 조작 강도
TEST_SAMPLE_SIZE = 100   # 검증할 질문 개수 (N=50 이상 권장 for ICML graphs)

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
# 2. 데이터 및 모델 로드
# ==========================================
def load_resources():
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=DEVICE, torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

def load_candidates():
    print(f"Loading Statistical Results from {STAT_RESULT_PATH}...")
    df = pd.read_csv(STAT_RESULT_PATH)
    
    # [필터링 로직]
    # 1. T-Score가 양수여야 함 (Unknown일 때 활성화가 더 큰 피처)
    # 2. T-Score가 가장 높은 순서대로 Top-K 추출
    # (옵션: AUROC 기준으로 뽑고 싶다면 't_score' 대신 'auroc' 사용 가능)
    
    df_filtered = df[df['t_score'] > 0].copy()
    top_candidates = df_filtered.nlargest(TOP_K_CANDIDATES, 't_score').reset_index(drop=True)
    
    print(f"Selected Top {TOP_K_CANDIDATES} candidates based on T-Score.")
    print(top_candidates[['layer', 'feature_idx', 't_score']].head())
    return top_candidates

def prepare_steering_prompts(path):
    print("Preparing 'Known' Prompts for Steering...")
    df = pd.read_csv(path)
    # 모델이 이미 정답을 맞춘(Correct) 데이터만 추출 -> 여기에 피처를 넣어서 거절하게 만들어야 함
    df_known = df[df['label'] == 'Correct'].copy()
    prompts = df_known['question'].sample(min(TEST_SAMPLE_SIZE, len(df_known)), random_state=42).tolist()
    return prompts

# ==========================================
# 3. Steering Hook
# ==========================================
class SteeringHook:
    def __init__(self, sae, feature_idx, coeff):
        self.sae = sae
        self.feature_idx = feature_idx
        self.coeff = coeff 

    def __call__(self, module, inputs, outputs):
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        target_dir = self.sae.W_dec[self.feature_idx]
        # Residual Stream에 방향 벡터 주입
        hidden_states[:, -1, :] += self.coeff * target_dir
        return (hidden_states,) + outputs[1:] if isinstance(outputs, tuple) else hidden_states

# ==========================================
# 4. 시각화 (ICML Paper Style)
# ==========================================
def plot_icml_figures(df, output_dir):
    sns.set(style="whitegrid", font_scale=1.2)
    
    # --- Figure 1: Statistical Significance vs Causal Effect ---
    # X축: T-Score (통계적 유의성), Y축: Steering Success Rate (인과적 효과)
    plt.figure(figsize=(10, 7))
    
    scatter = sns.scatterplot(
        data=df, 
        x='t_score', 
        y='steering_success_rate', 
        hue='layer', 
        palette='viridis', 
        size='mean_unknown', # 점 크기는 '모를 때의 활성화 정도'
        sizes=(50, 200),
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.axhline(0.2, color='red', linestyle='--', linewidth=1.5, label='Min. Success Threshold')
    plt.title("Decoupling Correlation from Causality: T-Score vs Steering Effect")
    plt.xlabel("T-Statistic (Separability of Known/Unknown)")
    plt.ylabel("Causal Refusal Rate (Steering Success)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Layer")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_tscore_vs_causality.png", dpi=300)
    print(f"Saved Figure 1 to {output_dir}")

    # --- Figure 2: The "Golden Layer" Identification ---
    # 성공한 피처들이 어느 레이어에 분포하는지 확인
    success_df = df[df['steering_success_rate'] >= 0.2]
    
    if not success_df.empty:
        plt.figure(figsize=(10, 5))
        layer_counts = success_df['layer'].value_counts().sort_index()
        # 0~31까지 빈 곳 채우기
        layer_counts = layer_counts.reindex(range(32), fill_value=0)
        
        sns.barplot(x=layer_counts.index, y=layer_counts.values, color="#4c72b0")
        plt.title("Distribution of Causal Uncertainty Features (Valid Detectors)")
        plt.xlabel("Layer Index")
        plt.ylabel("Count of Verified Features")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig2_causal_layer_distribution.png", dpi=300)
        print(f"Saved Figure 2 to {output_dir}")

# ==========================================
# 5. 메인 실행 루프
# ==========================================
def main():
    # 리소스 로드
    model, tokenizer = load_resources()
    candidates_df = load_candidates()
    test_prompts = prepare_steering_prompts(RAW_DATA_PATH)
    
    results = []
    
    # Layer별로 그룹화하여 SAE 로딩 최소화
    grouped = candidates_df.groupby('layer')
    
    print("\n[Start Verification] Testing Causal Effect on Top Candidates...")
    
    for layer, group in tqdm(grouped, desc="Processing Layers"):
        try:
            # SAE 로드
            sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID_TEMPLATE.format(layer=layer), device=DEVICE)[0]
            sae.eval()
        except Exception as e:
            print(f"Skip Layer {layer}: {e}")
            continue
            
        for _, row in group.iterrows():
            feat_idx = int(row['feature_idx'])
            
            # Hook 등록
            hook = SteeringHook(sae, feat_idx, coeff=STEERING_COEFF)
            handle = model.model.layers[layer].register_forward_hook(hook)
            
            refusal_count = 0
            try:
                # 배치 처리가 빠르지만, 생성(Generate)은 루프로 돌리는 게 안전
                for p in test_prompts:
                    inputs = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)
                    
                    # 40토큰 생성
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
                    
                    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    
                    # 거절 키워드 체크
                    if any(k in gen_text.lower() for k in REFUSAL_KEYWORDS):
                        refusal_count += 1
                        
            except Exception as e:
                print(f"Error steering L{layer}-F{feat_idx}: {e}")
            finally:
                handle.remove()
            
            success_rate = refusal_count / len(test_prompts)
            
            # 결과 저장
            res_dict = row.to_dict()
            res_dict['steering_success_rate'] = success_rate
            results.append(res_dict)
            
            # (옵션) 로그 출력 - 진행 상황 확인용
            # if success_rate > 0:
            #     print(f"Hit! L{layer}-F{feat_idx} (T={row['t_score']:.1f}) -> Rate: {success_rate:.2f}")

        # 메모리 정리
        del sae
        torch.cuda.empty_cache()
        gc.collect()
        
    # --- 결과 저장 및 시각화 ---
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nVerification Complete. Saved to {OUTPUT_CSV_PATH}")
    
    # 시각화 함수 호출
    plot_icml_figures(final_df, OUTPUT_DIR)

if __name__ == "__main__":
    main()