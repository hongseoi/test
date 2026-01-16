import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ==========================================
# 1. 설정 및 데이터 준비 (하드코딩)
# ==========================================
OUTPUT_DIR = "/home/seoi0215/trend/SAE/data_classification/Verifed_code/output/10_visualization_optimized_top10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 방금 분석한 Top 10 결과에서 '충돌'이 가장 심한 두 가지 케이스를 추출했습니다.
data = [
    # Case 1: Shania Twain (Famous) vs Joshua Ray (Obscure)
    {
        "Case": "Case A: Famous vs. Obscure Entity\n(Shania Twain vs. Joshua Ray)",
        "Feature": "L25-F2806", 
        "Role": "Known (Confidence)", 
        "Activation": 35.60, 
        "Coefficient": -1.46,
        "Description": "Pop Culture\nRecognizer"
    },
    {
        "Case": "Case A: Famous vs. Obscure Entity\n(Shania Twain vs. Joshua Ray)",
        "Feature": "L28-F9966", 
        "Role": "Refusal (Uncertainty)", 
        "Activation": 36.70, 
        "Coefficient": 1.06,
        "Description": "Conflict\nDetector"
    },
    {
        "Case": "Case A: Famous vs. Obscure Entity\n(Shania Twain vs. Joshua Ray)",
        "Feature": "L27-F4360", 
        "Role": "Refusal (Uncertainty)", 
        "Activation": 23.76, 
        "Coefficient": 1.06,
        "Description": "Asymmetry\nFlag"
    },

    # Case 2: Oliver Stone (Famous) vs James Cunningham (Less Famous)
    {
        "Case": "Case B: Comparison of Asymmetric Fame\n(Oliver Stone vs. James Cunningham)",
        "Feature": "L26-F16001", 
        "Role": "Known (Confidence)", 
        "Activation": 22.76, 
        "Coefficient": -1.10,
        "Description": "Fact Retrieval\nCircuit"
    },
    {
        "Case": "Case B: Comparison of Asymmetric Fame\n(Oliver Stone vs. James Cunningham)",
        "Feature": "L27-F7872", 
        "Role": "Refusal (Uncertainty)", 
        "Activation": 26.86, 
        "Coefficient": 0.67,
        "Description": "Obscure Entity\nDetector"
    },
    {
        "Case": "Case B: Comparison of Asymmetric Fame\n(Oliver Stone vs. James Cunningham)",
        "Feature": "L21-F18562", 
        "Role": "Refusal (Uncertainty)", 
        "Activation": 29.36, 
        "Coefficient": 0.56,
        "Description": "Uncertainty\nSignal"
    }
]

df = pd.DataFrame(data)

# ==========================================
# 2. 시각화 (Feature Interaction Diagram)
# ==========================================
def plot_interaction_diagram():
    sns.set_theme(style="whitegrid")
    
    # 두 개의 케이스를 별도의 서브플롯으로 그립니다.
    cases = df['Case'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    palette = {"Known (Confidence)": "#1f77b4", "Refusal (Uncertainty)": "#d62728"}
    
    for i, case in enumerate(cases):
        ax = axes[i]
        case_data = df[df['Case'] == case]
        
        # Bar Plot
        sns.barplot(
            data=case_data, 
            x="Feature", 
            y="Activation", 
            hue="Role", 
            palette=palette, 
            ax=ax,
            edgecolor="black",
            linewidth=1.5,
            dodge=False # 막대를 겹치지 않고 정위치에
        )
        
        # 디자인 다듬기
        ax.set_title(case, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("SAE Feature ID", fontsize=12)
        if i == 0:
            ax.set_ylabel("Activation Strength (Magnitude)", fontsize=12)
        else:
            ax.set_ylabel("")
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 막대 위에 설명 텍스트 추가 (Description)
        for p, desc in zip(ax.patches, case_data['Description']):
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2., 
                height + 1, 
                desc, 
                ha="center", va="bottom", fontsize=10, color='black', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

    # 범례 정리
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=12, title="Feature Role")
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    
    plt.tight_layout()
    
    # 저장
    save_path = f"{OUTPUT_DIR}/feature_interaction_conflict.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to: {save_path}")

if __name__ == "__main__":
    plot_interaction_diagram()