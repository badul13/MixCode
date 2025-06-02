import os
import zipfile
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm

# 1. ZIP 해제 함수
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"📂 {zip_path} 압축 해제 완료 → {extract_to}")

# 2. 중첩 ZIP 해제
def unzip_nested_zip_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_to = os.path.splitext(zip_path)[0]

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    print(f"📦 중첩 ZIP 해제: {zip_path} → {extract_to}")
                except Exception as e:
                    print(f"⚠️ {zip_path} 해제 실패: {e}")

# 3. JSON 로드 및 본문 추출 함수
def extract_newscontent_from_json(directory):
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        content = data.get('sourceDataInfo', {}).get('newsContent', '')
                        if not content:
                            sentences = data.get('labeledDataInfo', {}).get('referSentenceInfo', [])
                            if isinstance(sentences, list):
                                content = " ".join(
                                    s.get("sentenceContent", "") for s in sentences if isinstance(s, dict)
                                )

                        if content.strip():
                            result.append({'text': content})
                except Exception as e:
                    print(f"⚠️ JSON 로드 실패: {file}, 이유: {e}")
    return result

# 4. 메인 함수
def main():
    # 압축 파일 경로
    clickbait_zip = "/content/drive/MyDrive/ColabNotebooks/Clickbait.zip"
    nonclickbait_zip = "/content/drive/MyDrive/ColabNotebooks/NonClickbait.zip"
    clickbait_dir = "/content/Clickbait"
    nonclickbait_dir = "/content/NonClickbait"

    # 압축 해제
    print("🔽 ZIP 압축 해제 중...")
    unzip_file(clickbait_zip, clickbait_dir)
    unzip_file(nonclickbait_zip, nonclickbait_dir)

    print("🔽 중첩 ZIP 파일 해제 중...")
    unzip_nested_zip_files(clickbait_dir)
    unzip_nested_zip_files(nonclickbait_dir)

    # Clickbait (가짜 뉴스)
    print("📂 Clickbait → 가짜 뉴스 로드 중...")
    fake_data = extract_newscontent_from_json(clickbait_dir)
    if not fake_data:
        raise ValueError("❌ Clickbait 데이터 로드 실패.")
    fake_df = pd.DataFrame(fake_data)
    fake_df['label'] = 1
    print("✅ 가짜 뉴스 DataFrame 생성 완료:", fake_df.shape)

    # NonClickbait (진짜 뉴스)
    print("📂 NonClickbait → 진짜 뉴스 로드 중...")
    real_data = extract_newscontent_from_json(nonclickbait_dir)
    if not real_data:
        raise ValueError("❌ NonClickbait 데이터 로드 실패.")
    real_df = pd.DataFrame(real_data)
    real_df['label'] = 0
    print("✅ 진짜 뉴스 DataFrame 생성 완료:", real_df.shape)

    # 정보 출력
    print("📰 가짜 뉴스 샘플:\n", fake_df.head())
    print("📰 진짜 뉴스 샘플:\n", real_df.head())
    print("📊 라벨 분포 (가짜):\n", fake_df['label'].value_counts())
    print("📊 라벨 분포 (진짜):\n", real_df['label'].value_counts())

    # 병합 및 저장
    final_df = pd.concat([fake_df, real_df], ignore_index=True)
    final_df = final_df[final_df['text'].str.strip() != '']
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = "/content/outputdata.csv"
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 최종 CSV 저장 완료: {output_path}")
    print(final_df.head())

# 실행
if __name__ == "__main__":
    main()
