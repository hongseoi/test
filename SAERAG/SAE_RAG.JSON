# backup본에서 데이터 다양하게 사용하도록 변경
import os
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset

###### h-parms (설정 변경 부분)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 사용하려는 데이터셋 이름을 선택하세요: "hotpot_qa", "trivia_qa", "nq_open"
DATASET_NAME = "nq_open"  # <--- 여기를 "nq_open" 등으로 변경

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 파일명도 데이터셋에 따라 자동으로 바뀌게 설정하거나 직접 수정하세요
OUTPUT_FILE = f"/home/seoi0215/trend/SAE/data_classification/output/{DATASET_NAME}_llama3_results_train_no_context.xlsx"
NUM_GPUS = 1

#########
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 출력 디렉토리 생성
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 2. 데이터셋 로드 및 처리 로직 분기
print(f"Loading dataset: {DATASET_NAME}...")

if DATASET_NAME == "hotpot_qa":
    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
elif DATASET_NAME == "trivia_qa":
    # rc.nocontext는 컨텍스트 없이 질문-답변만 빠르게 로드할 때 좋습니다.
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="train") 
elif DATASET_NAME == "nq_open":
    # Natural Questions는 raw 버전보다 nq_open이 QA 태스크에 적합합니다.
    dataset = load_dataset("nq_open", split="train")
else:
    raise ValueError("지원하지 않는 데이터셋입니다.")

# (테스트용)
# dataset = dataset.select(range(10))

# 3. 프롬프트 구성 및 정답 추출
prompts = []
ground_truths = [] # 정답을 미리 저장해두는 리스트

print("Processing prompts...")
for item in dataset:
    # --- 데이터셋별 필드 처리 ---
    if DATASET_NAME == "hotpot_qa":
        question = item['question']
        answer = item['answer'] # 문자열
        
    elif DATASET_NAME == "trivia_qa":
        question = item['question']
        # TriviaQA는 answer가 Dict 형태 {'value': '정답', 'aliases': [...]}
        answer = item['answer']['value'] 
        
    elif DATASET_NAME == "nq_open":
        question = item['question']
        # NQ Open은 answer가 List 형태 ['정답1', '정답2'...]
        answer = item['answer'][0] 
    # -------------------------

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant. Answer the question concisely based on your own knowledge.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    prompts.append(prompt)
    ground_truths.append(answer)

# 4. vLLM 모델 로드
print(f"Loading vLLM model with {NUM_GPUS} GPUs...")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=NUM_GPUS, 
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    max_model_len=8192
)

# 5. 샘플링 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    stop_token_ids=[128001, 128009]
)

# 6. 일괄 생성
print(f"Generating responses for {len(prompts)} samples...")
outputs = llm.generate(prompts, sampling_params)

# 7. 결과 정리 및 저장
results = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    results.append({
        "question": dataset[i]['question'], # 원본 질문 다시 참조
        "ground_truth": ground_truths[i],   # 위에서 처리해둔 정답 사용
        "prediction": generated_text
    })

df = pd.DataFrame(results)
df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
print(f"Done! Saved to {OUTPUT_FILE}")