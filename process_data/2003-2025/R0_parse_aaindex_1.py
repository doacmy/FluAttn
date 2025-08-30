import re
import json

def parse_i_record_to_dict(lines):
    aa_pairs = ["A/L", "R/K", "N/M", "D/F", "C/P", "Q/S", "E/T", "G/W", "H/Y", "I/V"]
    aa_list = [aa for pair in aa_pairs for aa in pair.split('/')]

    values = []
    for line in lines:
        if 'NA' in line:
            return None
        values += [float(val) for val in line.strip().split()]

    if len(aa_list) != len(values):
        return None

    return dict(zip(aa_list, values))


def extract_all_i_records(file_path):
    all_dicts = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("H "):
            record_id = line.strip().split()[1]
        if line.startswith("I"):
            i_values = []
            i += 1
            while i < len(lines) and not lines[i].startswith("//"):
                i_values.append(lines[i])
                i += 1
            parsed = parse_i_record_to_dict(i_values)
            if parsed:
                all_dicts[record_id] = parsed
        else:
            i += 1
    return all_dicts


# 主程序入口
if __name__ == "__main__":
    aaindex_dicts = extract_all_i_records("data/raw/AAIndex/aaindex1.txt")

    # 保存为 JSON 文件
    with open("data/prd/aaindex1_dicts.json", "w") as json_file:
        json.dump(aaindex_dicts, json_file, indent=4)

    print("已保存为 aaindex1_dicts.json")
