import os
import zipfile
import json
import pandas as pd

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
def extract_all_json_files(path):
    json_data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            json_data.extend(data)
                        elif isinstance(data, dict):
                            json_data.append(data)
                except Exception as e:
                    print(f"❗ JSON 로드 실패: {file_path} - {e}")
    return json_data

def extract_newscontent_from_json(directory):
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        content = ""
                        if isinstance(data, dict):
                            # 1. 기존 newsContent 키 먼저 시도
                            content = data.get('labeledDataInfo', {}).get('newsContent', '')

                            # 2. 없으면 referSentenceInfo를 이어붙임
                            if not content:
                                sentences = data.get('labeledDataInfo', {}).get('referSentenceInfo', [])
                                if isinstance(sentences, list):
                                    content = " ".join(s.get("sentenceContent", "") for s in sentences if isinstance(s, dict))

                        if content:
                            result.append({'content': content})
                except Exception as e:
                    print(f"⚠️ JSON 로드 실패: {file}, 이유: {e}")
    return result

# 4. 메인 함수
def main():
    # 경로 설정
    zip1 = "/content/drive/MyDrive/ColabNotebooks/opendata1.zip"
    zip2 = "/content/drive/MyDrive/ColabNotebooks/opendata2.zip"
    extract_path1 = "/content/data1"
    extract_path2 = "/content/data2"

    # 압축 해제
    print("🔽 ZIP 압축 해제 중...")
    unzip_file(zip1, extract_path1)
    unzip_file(zip2, extract_path2)

    print("🔽 중첩 ZIP 파일 해제 중...")
    unzip_nested_zip_files(extract_path1)
    unzip_nested_zip_files(extract_path2)

    print("🔽 JSON 파일 로드 중...")
    fake_data1 = extract_newscontent_from_json(extract_path1)
    fake_data2 = extract_newscontent_from_json(extract_path2)
    all_fake_data = fake_data1 + fake_data2

    if not all_fake_data:
        raise ValueError("❌ 가짜 뉴스 데이터 로드 실패.")

    fake_df = pd.DataFrame(all_fake_data)
    fake_df['label'] = 1
    print("✅ 가짜 뉴스 DataFrame 생성 완료:", fake_df.shape)

    # 진짜 뉴스 로드
    print("🔽 진짜 뉴스 CSV 로드 중...")
    newsdata_path = "/content/newsdata.csv"
    real_df = pd.read_csv(newsdata_path, encoding='cp949')

    text_col_real = None
    for col in real_df.columns:
        if real_df[col].apply(lambda x: isinstance(x, str)).mean() > 0.8:
            text_col_real = col
            break
    if text_col_real is None:
        raise ValueError("❌ 진짜 뉴스 텍스트 컬럼을 찾을 수 없습니다.")
    real_df = real_df.rename(columns={text_col_real: 'content'})
    real_df = real_df[['content']].dropna()
    real_df['label'] = 0

    # 병합 및 저장
    final_df = pd.concat([fake_df, real_df], ignore_index=True)
    final_df = final_df[final_df['content'].str.strip() != ''] 
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_path = "/content/opendata_output.csv"
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 최종 CSV 저장 완료: {output_path}")
    print(final_df.head())

# 실행
if __name__ == "__main__":
    main()
