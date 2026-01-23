import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 设置目标页面和请求头
url = "https://www.crick.ac.uk/research/platforms-and-facilities/worldwide-influenza-centre/annual-and-interim-reports"
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

# 发送初始页面请求
response = requests.get(url, headers=headers)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

# 创建保存目录
save_dir = "data/raw/WHOCC/"
os.makedirs(save_dir, exist_ok=True)

# 提取所有 PDF 链接
pdf_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.endswith(".pdf"):
        full_url = urljoin(url, href)
        pdf_links.append(full_url)

print(f"共找到 {len(pdf_links)} 个 PDF 文件，开始下载...\n")

# 下载每个 PDF，增加随机等待间隔
for idx, link in enumerate(pdf_links, 1):
    filename = link.split("/")[-1]
    file_path = os.path.join(save_dir, filename)
    print(f"[{idx}/{len(pdf_links)}] 下载中：{filename}")

    try:
        pdf_response = requests.get(link, headers=headers, stream=True)
        pdf_response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)

        # 添加随机等待间隔（1~4秒）
        wait_time = random.uniform(1, 4)
        print(f"完成，等待 {wait_time:.2f} 秒...\n")
        time.sleep(wait_time)

    except Exception as e:
        print(f" 下载失败：{filename}，错误：{e}\n")


