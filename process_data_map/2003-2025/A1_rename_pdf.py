# 重新命名乱七八糟的报告文件

import os
import re

folder = "data/raw/WHOCC/"

month_map = {
    "january": "jan", "february": "feb", "march": "mar",
    "april": "apr", "may": "may", "june": "jun",
    "july": "jul", "august": "aug", "september": "sep",
    "october": "oct", "november": "nov", "december": "dec",
    "jan": "jan", "feb": "feb", "mar": "mar",
    "apr": "apr", "jun": "jun", "jul": "jul",
    "aug": "aug", "sep": "sep", "sept": "sep",
    "oct": "oct", "nov": "nov", "dec": "dec"
}

for filename in os.listdir(folder):
    if not filename.lower().endswith(".pdf"):
        continue

    fname = filename.lower()

    match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)[^0-9]{0,5}(\d{4})', fname)

    if not match:
        match = re.search(r'(\d{4})[^a-z]{0,5}(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)', fname)

    if match:
        month_raw = match.group(1) if match.group(1).isalpha() else match.group(2)
        year = match.group(2) if match.group(1).isalpha() else match.group(1)
        month_std = month_map.get(month_raw[:3].lower())

        if month_std and year:
            new_name = f"{year}-{month_std}.pdf"
            src = os.path.join(folder, filename)
            dst = os.path.join(folder, new_name)
            try:
                os.rename(src, dst)
                print(f"✅ Renamed: {filename} -> {new_name}")
            except FileExistsError:
                print(f"⚠️ 文件已存在，跳过: {new_name}")
        else:
            print(f"❌ Skipped (无法标准化): {filename}")
    else:
        print(f"❌ Skipped (无匹配): {filename}")