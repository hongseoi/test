import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
import os
from tqdm import tqdm
import torch.nn.functional as F
import gc

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0"

# 파일 경로
INPUT_DATA_PATH = "/home/seoi0215/trend/SAE/data_classification/output/images/hotpot_qa_llama3_results_train_no_context_evaluated.csv"
METRICS_PATH = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/4/final_verified_features.csv" 
OUTPUT_PATH = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/4_1/steering_output.xlsx" 

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE = "l{layer}r_8x"

# 실험 파라미터
STEERING_COEFFS = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150]
N_SAMPLES = 100
MAX_NEW_TOKENS = 30

# ==========================================
# 2. Hook 클래스 정의 (변경 없음)
# ==========================================

class SteeringHook:
    def __init__(self, feature_vector, coeff):
        self.feature_vector = feature_vector.to(DEVICE)
        self.coeff = coeff

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if hidden_states.dim() == 3:
            hidden_states[:, -1, :] += self.coeff * self.feature_vector
        elif hidden_states.dim() == 2:
            hidden_states += self.coeff * self.feature_vector
        
        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return hidden_states

class AblationHook:
    def __init__(self, sae, feature_idx):
        self.sae = sae
        self.feature_idx = feature_idx

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
            
        if hidden_states.dim() == 3:
            x_input = hidden_states[:, -1, :]
        elif hidden_states.dim() == 2:
            x_input = hidden_states
        else:
            return outputs

        # SAE Encode
        feature_acts = self.sae.encode(x_input) 
        target_act = feature_acts[:, self.feature_idx]
        decoder_direction = self.sae.W_dec[self.feature_idx]
        ablation_vector = torch.outer(target_act, decoder_direction)
        
        # Subtract
        if hidden_states.dim() == 3:
            hidden_states[:, -1, :] -= ablation_vector
        elif hidden_states.dim() == 2:
            hidden_states -= ablation_vector
            
        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return hidden_states

# ==========================================
# 3. 모델 및 데이터 로드 (수정됨)
# ==========================================
def load_base_model():
    print(f"Loading HF Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map=DEVICE, 
        torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

def load_sae_for_layer(layer_idx):
    # [Fix] layer_idx를 정수로 강제 변환하여 포맷팅 문제 해결 (15.0 -> 15)
    layer_idx = int(layer_idx)
    print(f"Loading SAE for Layer {layer_idx}...")
    
    sae = SAE.from_pretrained(
        release=SAE_RELEASE, 
        sae_id=SAE_ID_TEMPLATE.format(layer=layer_idx), 
        device=DEVICE
    )[0]
    sae.eval()
    return sae

def prepare_data():
    print("Loading Dataset...")
    if INPUT_DATA_PATH.endswith('.xlsx'):
        df = pd.read_excel(INPUT_DATA_PATH)
    else:
        df = pd.read_csv(INPUT_DATA_PATH)
    
    df_known = df[(df['label'] == 'Correct') & (df['fuzzy_score'] >= 80)]
    if len(df_known) > N_SAMPLES:
        df_known = df_known.sample(N_SAMPLES, random_state=42)
    
    df_unknown = df[(df['label'] == 'Incorrect') & (df['fuzzy_score'] <= 20)]
    if len(df_unknown) > N_SAMPLES:
        df_unknown = df_unknown.sample(N_SAMPLES, random_state=42)
    
    print(f"Known samples: {len(df_known)}, Unknown samples: {len(df_unknown)}")
    return df_known, df_unknown

# ==========================================
# 4. 개별 Feature 실험 함수 (변경 없음)
# ==========================================
def run_single_feature_experiment(model, tokenizer, sae, layer_idx, feature_idx, df_known, df_unknown):
    results = []
    
    refusal_keywords = [
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

    # [Fix] 모델 레이어 인덱싱을 위해 int 변환 (안전장치)
    layer_idx = int(layer_idx)
    target_module = model.model.layers[layer_idx]

    # ---------------------------------------------------------
    # Exp 1: Positive Steering (Known Questions)
    # ---------------------------------------------------------
    prompts_k = df_known['question'].tolist()
    feature_vector = sae.W_dec[feature_idx].detach().clone()
    
    for coeff in STEERING_COEFFS:
        hook_fn = SteeringHook(feature_vector, coeff)
        handle = target_module.register_forward_hook(hook_fn)
        
        try:
            generated_texts = []
            for prompt in prompts_k:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=MAX_NEW_TOKENS, 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                generated_texts.append(gen_text)
        finally:
            handle.remove()
            
        for i, (index, row) in enumerate(df_known.iterrows()):
            res_dict = row.to_dict()
            res_dict.update({
                'target_layer': layer_idx,
                'target_feature': feature_idx,
                'experiment': 'Positive_Steering',
                'prompt_type': 'Known',
                'coeff': coeff,
                'generation': generated_texts[i],
                'is_refusal': any(k in generated_texts[i].lower() for k in refusal_keywords),
                'generation_baseline': row['prediction']
            })
            results.append(res_dict)

    # ---------------------------------------------------------
    # Exp 2: Feature Ablation (Unknown Questions)
    # ---------------------------------------------------------
    prompts_u = df_unknown['question'].tolist()
    
    hook_fn = AblationHook(sae, feature_idx)
    handle = target_module.register_forward_hook(hook_fn)
    
    generations_abl = []
    try:
        for prompt in prompts_u:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(inputs, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            generations_abl.append(gen_text)
    finally:
        handle.remove()

    for i, (index, row) in enumerate(df_unknown.iterrows()):
        res_dict = row.to_dict()
        res_dict.update({
            'target_layer': layer_idx,
            'target_feature': feature_idx,
            'experiment': 'Ablation',
            'prompt_type': 'Unknown',
            'coeff': 'Ablated',
            'generation': generations_abl[i],
            'generation_baseline': row['prediction']
        })
        results.append(res_dict)

    return results

# ==========================================
# 5. 메인 함수 (수정됨)
# ==========================================
def main():
    # 1. 모델 및 데이터 로드
    model, tokenizer = load_base_model()
    df_known, df_unknown = prepare_data()
    
    # 2. 타겟 Feature 목록 로드
    print(f"Loading Target Features from {METRICS_PATH}...")
    metrics_df = pd.read_csv(METRICS_PATH)
    
    # [Fix] layer 컬럼을 정수형(int)으로 변환하여 float(15.0) 문제 원천 차단
    if 'layer' in metrics_df.columns:
        metrics_df['layer'] = metrics_df['layer'].astype(int)
    
    if 'feature_idx' in metrics_df.columns:
        metrics_df['feature_idx'] = metrics_df['feature_idx'].astype(int)

    # 중복 제거
    target_list = metrics_df[['layer', 'feature_idx']].drop_duplicates()
    
    # Layer별로 그룹화
    grouped_targets = target_list.groupby('layer')['feature_idx'].apply(list).to_dict()
    
    all_results = []
    
    # 3. Layer별 순회
    sorted_layers = sorted(grouped_targets.keys())
    print(f"Targets found in {len(sorted_layers)} layers: {sorted_layers}")
    
    for layer_idx in tqdm(sorted_layers, desc="Iterating Layers"):
        # [Fix] layer_idx가 확실히 int인지 확인 (위에서 변환했지만 이중 안전장치)
        layer_idx = int(layer_idx)
        
        # 해당 Layer의 SAE 로드
        sae = load_sae_for_layer(layer_idx)
        
        features = grouped_targets[layer_idx]
        print(f" >> Layer {layer_idx}: Processing {len(features)} features...")
        
        # 해당 Layer의 Feature별 순회
        for feature_idx in tqdm(features, desc=f"Features in L{layer_idx}", leave=False):
            feature_idx = int(feature_idx)
            
            # 실험 수행
            results = run_single_feature_experiment(
                model, tokenizer, sae, 
                layer_idx, feature_idx, 
                df_known, df_unknown
            )
            all_results.extend(results)
        
        # 메모리 정리
        del sae
        torch.cuda.empty_cache()
        gc.collect()

    # 4. 최종 저장
    final_df = pd.DataFrame(all_results)
    final_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved combined results to {OUTPUT_PATH}")
    print(f"Total experiment rows: {len(final_df)}")

if __name__ == "__main__":
    main()