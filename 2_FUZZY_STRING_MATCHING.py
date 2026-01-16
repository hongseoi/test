import pandas as pd
import os
from thefuzz import fuzz

# ==========================================
# 1. 파일 경로 설정
# ==========================================
# 이전 코드에서 저장한 파일 경로
INPUT_FILE = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context.xlsx"
# 평가 결과를 저장할 새로운 파일 경로
OUTPUT_FILE = "/home/seoi0215/trend/SAE/data_classification/output/hotpot_qa_llama3_results_train_no_context_evaluated.xlsx"

INPUT_FILE = "/home/seoi0215/trend/SAE/data_classification/output/trivia_qa_llama3_results_train_no_context.xlsx"
# 평가 결과를 저장할 새로운 파일 경로
OUTPUT_FILE = "/home/seoi0215/trend/SAE/data_classification/output/trivia_qa_llama3_results_train_no_context_evaluated.xlsx"


# ==========================================
# 2. 평가 로직 함수 정의
# ==========================================
def evaluate_sample(row, threshold=80):
    """
    행(row)을 받아 Correct, Incorrect, Refusal 및 점수를 반환합니다.
    """
    # 문자열 전처리 (소문자 변환, 양끝 공백 제거)
    ground_truth = str(row['ground_truth']).lower().strip()
    prediction = str(row['prediction']).lower().strip()
    
    # 1) Refusal (거절) 감지 키워드 리스트
    # 모델이 답변을 거부하거나 모른다고 할 때 자주 쓰는 패턴들
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
    
    # 예측값에 거절 키워드가 포함되어 있고, 길이가 너무 길지 않은 경우(설명하다가 우연히 들어간게 아닌 경우) Refusal로 처리
    # 다만, Llama3는 답변 앞부분에 거절 의사를 밝히는 경우가 많으므로 단순 포함 여부로 체크합니다.
    if any(keyword in prediction for keyword in refusal_keywords):
        # 만약 정답이 거절 키워드 자체일 확률은 매우 낮으므로 Refusal 처리
        return "Refusal", 0

    # 2) Fuzzy String Matching (유사도 측정)
    # fuzz.token_set_ratio: 단어 순서가 바뀌거나, "The answer is..." 같은 문구가 추가되어도 
    # 핵심 단어(정답 엔티티)가 포함되어 있으면 높은 점수를 줍니다.
    score = fuzz.token_set_ratio(ground_truth, prediction)
    
    # 3) 점수에 따른 분류
    if score >= threshold:
        return "Correct", score
    else:
        return "Incorrect", score

# ==========================================
# 3. 데이터 로드 및 평가 실행
# ==========================================
print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    print("Error: 입력 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

print("Evaluating answers using Fuzzy Matching...")

# apply 함수를 통해 평가 수행 (반환값이 튜플이므로 zip으로 분리)
evaluation_results = df.apply(lambda row: evaluate_sample(row), axis=1)
df['label'] = [res[0] for res in evaluation_results]
df['fuzzy_score'] = [res[1] for res in evaluation_results]

# ==========================================
# 4. 통계 출력 및 저장
# ==========================================
# 분류별 개수 계산
summary = df['label'].value_counts()
total = len(df)

print("\n" + "="*30)
print("Evaluation Summary")
print("="*30)
print(f"Total Samples: {total}")
print(f"Correct:       {summary.get('Correct', 0)} ({summary.get('Correct', 0)/total*100:.2f}%)")
print(f"Incorrect:     {summary.get('Incorrect', 0)} ({summary.get('Incorrect', 0)/total*100:.2f}%)")
print(f"Refusal:       {summary.get('Refusal', 0)} ({summary.get('Refusal', 0)/total*100:.2f}%)")
print("="*30)

# 결과 저장
df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
print(f"\nSaved evaluated results to: {OUTPUT_FILE}")