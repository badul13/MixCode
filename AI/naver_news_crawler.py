import os
import re
import requests
from typing import List
from urllib.parse import quote
from bs4 import BeautifulSoup

class NaverNewsCrawler:
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }

    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[가-힣]{2,}\b', text)
        return list(set(words))[:5]

    def search_news_urls(self, keywords: List[str]) -> List[str]:
        if not keywords:
            return []

        query = quote(" ".join(keywords))
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=5&sort=sim"

        try:
            res = requests.get(url, headers=self.headers, timeout=5)
            if res.status_code != 200:
                print("🔴 네이버 API 요청 실패:", res.status_code, res.text)
                return []

            items = res.json().get("items", [])
            links = [item["link"] for item in items if "news.naver.com" in item["link"]]
            return list(set(links))

        except Exception as e:
            print(f"🔴 네이버 뉴스 검색 실패: {e}")
            return []

    def crawl_articles(self, urls: List[str]) -> List[str]:
        articles = []
        for url in urls:
            try:
                res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                soup = BeautifulSoup(res.text, "html.parser")

                selectors = [
                    "div#newsct_article",   # 일반 뉴스 (가장 흔함)
                    "div#articleBodyContents",  # 옛날 모바일 페이지
                    "div#articeBody",       # 일부 구형 뉴스
                    "div#newsEndContents",  # 기타
                    "div#dic_area",         # 정치/오피니언 등
                    "div.article_body"      # 네이버 스포츠 등
                ]

                article_text = ""
                for sel in selectors:
                    container = soup.select_one(sel)
                    if container:
                        article_text = container.get_text(separator=" ", strip=True)
                        break

                if len(article_text) > 100:
                    articles.append(article_text)
                else:
                    print(f"❗ 본문 너무 짧음 or 추출 실패: {url}")

            except Exception as e:
                print(f"🔴 기사 크롤링 실패 {url}: {e}")
                continue

        return articles
