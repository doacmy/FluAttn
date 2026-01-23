import re
import os
import pdfplumber
import pandas as pd
from datetime import datetime

# 说明：
# 本脚本用于解析WHO历年的 H1N1 报告 PDF 中的HI滴度表格，
# 将交叉中和（中和滴度）矩阵提取为长表数据：列为 Test Virus、Reference Virus、Titre。
#
# 主要步骤：
# 1) 遍历 PDF 按“Table ”分段，初步判断是否为目标表格；
# 2) 通过大量规则修复 OCR/排版带来的病毒名与数值异常；
# 3) 从表头区识别参考毒株列（Reference Virus，列名）并映射到全文中的完整病毒名；
# 4) 从数据区逐行解析测试毒株（Test Virus）对应的各列滴度值；
# 5) 合并所有页的结果，并可与已存在的 CSV 对比验证。

def is_table(text: str, threshold=10) -> float:
    """判断文本块是否为目标表格。

    依据：统计常见滴度值（20,40,80,160,320,640,1280,2560,5120）的出现次数，
    并要求包含关键字“REFERENCE VIRUSES”，排除与本任务无关的表格（如 PRN、swine、seasonal）。

    参数：
        text: 文本块
        threshold: 滴度值数量阈值（默认 10）
    返回：
        True/False
    """
    COMMON_TITERS = {20, 40, 80, 160, 320, 640, 1280, 2560, 5120}
    numbers = re.findall(r"\b\d+\b", text)
    numbers = [int(n) for n in numbers]
    count = sum(1 for n in numbers if n in COMMON_TITERS)
    if count > threshold and 'REFERENCE VIRUSES' in text and 'Plaque Reduction Neutralisation' not in text and 'swine' not in text and 'seasonal' not in text:
        return True
    else:
        return False


def main(filename, verlog=False):
    """解析单个 PDF 文件，返回长表 DataFrame。

    参数：
        filename: PDF 文件名（相对 pdf_folder）
        verlog: 是否打印调试信息/中间 DataFrame
    返回：
        包含列 ["Test Virus", "Reference Virus", "Titre"] 的长表 DataFrame
    """
    # 匹配完整病毒名的模式：
    # 形如 A/Region[/Subregion...]/ID/yyyy，允许中间有空格、横杠、点、单引号等
    virus_pattern = re.compile(r'(?:[A]/[A-Za-z][A-Za-z\'\.-]*(?: [A-Za-z][A-Za-z\'\.-]*)*/[0-9A-Za-z._-]+/\d{4})')

    pdf_path = os.path.join(pdf_folder, filename)

    virus_df = pd.DataFrame(columns=columns)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 同一页可能包含多个“Table ”分段，这里按关键字切分
            texts = page.extract_text().split('Table ')
            for text in texts:

                # 有些 PDF 会把“REFERENCE VIRUSES”误作“COPY AND PASTE VALUES”，统一替换回来
                text = text.replace('COPY AND PASTE VALUES', 'REFERENCE VIRUSES\n')

                if is_table(text):
                    if 'H1N1' in text:
                        # 针对 OCR/排版问题的集中修复（病毒名/年份等）
                        text = text.replace(" A/ BurkinaFaso/", " A/BurkinaFaso/")
                        text = re.sub(r'(A/SouthAfrica/[^/]+)/24\b', r'\1/2024', text)
                        text = text.replace('A/Burgos55/2022', 'A/Burgos/55/2022')
                        text = text.replace('A/Guangong-Maonan', 'A/Guangdong-Maonan')
                        text = text.replace('A/Guangong-Maona', 'A/Guangdong-Maonan')
                        text = text.replace('A/New Jersey/8/76', 'A/New Jersey/8/2006')
                        text = text.replace("A/Iowa/15/30", "A/Iowa/15/2003")

                        # 从整段文本中抓取可能出现的完整病毒名，后续用于列名映射
                        all_virus = list(set(re.findall(virus_pattern, text)))

                        # 针对个别文件/页，列出缺失但实际应出现的参考毒株名
                        if filename == '2020-sep.pdf':
                            if page.page_number == 25:
                                all_virus.append("A/Denmark/3280/2019")
                            elif page.page_number == 35 or page.page_number == 37:
                                all_virus.append("A/Hong Kong/4394/2019")
                        elif filename == '2020-feb.pdf':
                            if page.page_number == 24:
                                all_virus.append("A/Hong Kong/110/2019")
                        elif filename == '2019-feb.pdf':
                            all_virus.append("A/Michigan/45/2015")
                            all_virus.append("A/California/7/2009")
                        elif filename == '2018-feb.pdf':
                            all_virus.append("A/South Africa/3626/2013")
                            all_virus.append("A/Michigan/45/2015")
                        elif filename == '2016-sep.pdf':
                            all_virus.append("A/Yokohama/94/2015")

                        # 将文本分割为表头区（列名所在）和数据区（数值所在）
                        header_block = re.search(r'(.*)REFERENCE VIRUSES', text, re.S).group(1)
                        main_block = re.search(r'REFERENCE VIRUSES(.*)', text, re.S).group(1)

                        # 个别页面直接给定列名以避免歧义
                        if filename == '2023-sep.pdf' and page.page_number == 25:
                            col_virus = ['A/Guangdong-Maonan/SWL1536/2019', 'A/Guangdong-Maonan/SWL1536/2019', 'A/Ghana/1894/2021', 'A/Lyon/820/2021', 'A/Denmark/3280/2019', 'A/Victoria/2570/2019', 'A/Sydney/5/2021', 'A/Sydney/5/2021', 'A/Norway/25089/2022', 'A/Norway/31694/2022', 'A/Norway/31694/2022', 'A/Catalonia/NSVH161512065/2022']
                        else:
                            # 统一表头中的缩写/别名，便于后续 token 抽取与匹配完整病毒名
                            header_block = header_block.replace("IVR-238", "A/Victoria/ ")
                            header_block = header_block.replace("A/Vic/4897/22", " 4897/2022")
                            header_block = header_block.replace("IVR-215", "A/Victoria/")
                            header_block = header_block.replace("A/Vic/2570/19", "2570/2019")
                            header_block = header_block.replace("A/G-M A/G-M A/GhanaA/Lyon", "A/Guangdong A/Guangdong A/Ghana A/Lyon")
                            header_block = header_block.replace("A/SydneyA/Norway", "A/Sydney A/Norway")
                            header_block = header_block.replace("  ", " ")
                            header_block = header_block.replace("A/G-M A/G-MA/GhanaA/Lyon A/Denmark A/Victoria/ A/SydneyA/Sydney", "A/Guangdong A/Guangdong A/Ghana A/Lyon A/Denmark A/Victoria/ A/Sydney A/Sydney")
                            header_block = header_block.replace("SWL1536/19SWL1536/19", "SWL1536/19 SWL1536/19")
                            header_block = header_block.replace("1894/21820/21", "1894/21 820/21")
                            header_block = header_block.replace('5/2125089/22', '5/21 25089/22')
                            header_block = header_block.replace("A/G-M ", "A/Guangdong ")
                            header_block = header_block.replace("A/SydneyA/Sydney", "A/Sydney A/Sydney")
                            header_block = header_block.replace("820/21 3280/19", "820/2021 3280/2019")
                            header_block = header_block.replace("A/HK", "A/Hong")
                            header_block = header_block.replace("CNIC-1909", "A/Guangdong")
                            header_block = header_block.replace("A/G-M/1536/2019", "1536/2019")
                            header_block = header_block.replace("A/Sth Afr", "A/South")
                            header_block = header_block.replace("A/Chch", "A/Christchurch")
                            header_block = header_block.replace("Yoko", "A/Yokohama")
                            header_block = header_block.replace("X-243", "A/South")
                            header_block = header_block.replace("A/DR", "A/Dominican")
                            header_block = header_block.replace("A/C'church", "A/Christchurch")
                            header_block = header_block.replace("A/C'ch", "A/Christchurch")
                            header_block = header_block.replace("A/H-X", "A/Heilongjiang")
                            header_block = header_block.replace("A/NJ2", "A/New")
                            header_block = header_block.replace("A/NJ", "A/New")
                            header_block = header_block.replace("A/sw/Iowa", "A/Iowa")
                            header_block = header_block.replace("07606/202402-1057/2024", '07606/2024 02-1057/2024')
                            header_block = header_block.replace('5142/2024PP15052/2024', '5142/2024 BFC-IPP15052/2024')
                            header_block = header_block.replace('6849/20250003050/2025', '6849/2025 25220003050/2025')
                            header_block = header_block.replace('A/NorwayA//Switzerland/', 'A/Norway A/Switzerland/')

                            # 从表头中提取列名的缩略 token，并在随后的行中补齐年份等关键信息
                            
                            tokens = []
                            lines = header_block.split('\n')
                            for line in lines:
                                if line.count("A/") >= 3:
                                    # 第一行：出现多个 A/，用正则提取 token：
                                    #  - 形如 A/XXX（可能带/）
                                    #  - 或者形如 CODE-123 这类编号
                                    pattern = r'(A/[A-Za-z0-9]+/?)|([A-Za-z]+-\d+)'
                                    matches = re.findall(pattern, line)
                                    tokens = [m[0] or m[1] for m in matches]
                                    continue
                                if tokens:
                                    # 第二行：可能包含 2 位年份，统一转成 4 位年份（20yy）
                                    tpattern = r'\b([A-Za-z0-9]+)/(\d{2})\b'
                                    line = re.sub(tpattern, lambda m: f"{m.group(1)}/20{m.group(2)}", line)

                                    if filename == '2015-feb.pdf' and page.page_number == 18:
                                        # 针对该页的标题缺漏，补齐一个 3626/2013，确保与 tokens 对齐
                                        line = line.replace("date History 7/2009 69/2009 N6/2009 16/2010 3934/2011 1/2011 27/2011 100/2011 5659/2012 3626/2013", "date History 7/2009 69/2009 N6/2009 16/2010 3934/2011 1/2011 27/2011 100/2011 5659/2012 3626/2013 3626/2013")
                                    vinfo = line.split(' ')[-len(tokens):]
                                    break
                            col_virus = []
                            if len(tokens) and len(vinfo):
                                assert len(tokens) == len(vinfo)
                                for i, tk in enumerate(tokens):
                                    # 将 token 与全文中抓取的完整病毒名比对：
                                    matches = [virus_name for virus_name in all_virus if virus_name.startswith(tk)]

                                    if len(matches) == 0:
                                        raise ValueError(f"Cannot find full name for {tk} in {filename} page {page.page_number}")
                                    elif len(matches) == 1:
                                        col_virus.append(matches[0])
                                    else:
                                        # 多匹配时，结合下一行中的年份/编号等关键信息 disambiguate
                                        for virus_name in matches:
                                            if vinfo[i] in virus_name:
                                                col_virus.append(virus_name)
                                                break
                            assert len(col_virus) == len(tokens)

                        if filename == '2015-feb.pdf' and page.page_number == 24:
                            # 该页表头多出一个旧参考毒株，手动移除保证列数匹配
                            col_virus.remove('A/California/7/2009')

                        # 统一数据区的异常与 OCR 错误，保证后续行解析稳定
                        main_block = main_block.replace("SIAT2/MDCK3 1280 1289", "SIAT2/MDCK3 1280 1280")
                        main_block = main_block.replace("640 242788 A/Slovenia/268/2024", "640\n 242788 A/Slovenia/268/2024")
                        main_block = main_block.replace("10-10", "AA")
                        main_block = main_block.replace("ND", "*")
                        main_block = main_block.replace("NT", "*")
                        main_block = main_block.replace(">5120", "5120")
                        main_block = main_block.replace("2560 1280 1280 1280 A/Norway/33307/2022 5a.2a.1", "2560 1280 1280 1280\n A/Norway/33307/2022 5a.2a.1")
                        main_block = main_block.replace("2560 2560 2560 2560 A/Romania/544748/2023 5a.2a", "2560 2560 2560 2560\n A/Romania/544748/2023 5a.2a")
                        main_block = main_block.replace("25660", "2560")
                        main_block = main_block.replace("12580", "1280")
                        main_block = main_block.replace("1280 1281 640", "1280 1280 640")
                        main_block = main_block.replace("25620", "2560")
                        main_block = main_block.replace("<<", '< <')

                        data = []
                        lines = main_block.split('\n')
                        for line in lines:
                            # 每一行包含一个测试毒株和一行滴度数据
                            virus_match = virus_pattern.search(line)
                            if virus_match:
                                virus_name = virus_match.group()

                                # 将形如 "a-b" 的范围值转为平均值（仅当 a、b 都为 10 的整数倍）
                                def replacer(match):
                                    a, b = int(match.group(1)), int(match.group(2))
                                    if a % 10 == 0 and b % 10 == 0:
                                        avg = (a + b) // 2
                                        return str(avg)
                                    else:
                                        return match.group(0)

                                pattern = re.compile(r'(\d+)-(\d+)')
                                elem = pattern.sub(replacer, line).split(" ")

                                # 仅保留数值/符号型 token：纯数字、含 "<" 的阈值、"*" 的缺失值
                                tmp = [s for s in elem if re.fullmatch(r'[0-9<\*]+', s)]
                                elem = []
                                for e in tmp:
                                    if "<" in e or "*" in e:
                                        elem.append(e)
                                    elif int(e) % 10 == 0 and int(e) < 10000:
                                        # 合法滴度：10 的倍数且小于 10000
                                        elem.append(e)

                                if filename == '2015-feb.pdf' and page.page_number == 24:
                                    # 该页右侧可能有多余数字，截取到与列数一致
                                    if len(elem) != len(col_virus):
                                        elem = elem[-len(col_virus):]

                                assert len(col_virus) == len(elem)

                                # 合并为一行：测试毒株 + 所有参考毒株的滴度
                                data.append([virus_name] + elem)

                        assert len(data) > 0

                        df = pd.DataFrame(data)
                        df.columns = ['Test Virus'] + col_virus

                        if verlog:
                            # 调试：打印每页解析出的宽表和列名
                            print(f'Page:{page.page_number}')
                            print(df)
                            print(df.columns.tolist())

                        # 宽表转成长表格式
                        long_df = df.melt(id_vars=["Test Virus"], var_name="Reference Virus", value_name="Titre")

                        # 累积到总结果中
                        virus_df = pd.concat([virus_df, long_df], ignore_index=True)
    return virus_df


def equal_csv(df1, file2):
    """比较内存结果与磁盘 CSV 的等价性（基于三列的分组计数）。

    若不一致，打印差异（两边计数）。
    """
    df2 = pd.read_csv(file2, dtype=str)
    vc1 = df1.value_counts(columns).sort_index()
    vc2 = df2.value_counts(columns).sort_index()

    result = vc1.equals(vc2)

    if result == False:
        all_index = vc1.index.union(vc2.index)

        vc1 = vc1.reindex(all_index, fill_value=0)
        vc2 = vc2.reindex(all_index, fill_value=0)

        diff = vc1[vc1 != vc2].to_frame("df1_count")
        diff["df2_count"] = vc2[vc1 != vc2]
        print(diff)
    return result

def verify(filename):
    """运行解析并与对应 CSV 对比，失败则抛出异常。"""
    virus_df = main(filename, verlog=False)
    output_file = f'{out_folder}{filename.replace(".pdf", "")}.csv'
    if os.path.exists(output_file):
        if not equal_csv(virus_df, output_file):
            raise ValueError(f"Failed: {filename}")
        else:
            print(f"Passed: {filename}")


if __name__ == "__main__":
    # 单文件解析与批量校验入口
    filename = '2025-sep.pdf'
    processed = ['2010-feb.pdf', '2010-sep.pdf', '2011-feb.pdf', '2011-sep.pdf', '2012-feb.pdf', '2012-sep.pdf', '2013-feb.pdf', '2013-sep.pdf', '2014-feb.pdf', '2014-sep.pdf', '2015-feb.pdf', '2015-sep.pdf', '2016-feb.pdf', '2016-sep.pdf', '2017-feb.pdf', '2017-sep.pdf', '2018-feb.pdf', '2019-feb.pdf', '2019-sep.pdf', '2020-feb.pdf', '2020-sep.pdf', '2022-feb.pdf', '2022-sep.pdf', '2023-feb.pdf', '2023-sep.pdf', '2024-feb.pdf', '2024-sep.pdf', '2025-feb.pdf']

    out_folder = "data/raw/H1N1/csv/"
    pdf_folder = "data/raw/WHOCC/"
    columns = ["Test Virus","Reference Virus", "Titre"]

    virus_df = main(filename, verlog=True)
    output_file = f'{out_folder}{filename.replace(".pdf", "")}.csv'

    if os.path.exists(output_file):
        if not equal_csv(virus_df, output_file):
            raise ValueError(f"Failed: {filename}")
    else:
        virus_df.to_csv(output_file, index=False)

    print('Verifying:')
    for filename in processed:
        verify(filename)

