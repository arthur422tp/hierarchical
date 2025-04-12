"""
管理小雜物function的地方
"""
import os
import pandas as pd
import glob
import re

class DataDealer:
    def __init__(self):
        pass


    def convert_md_to_txt(input_dir, output_dir):
        """
        將md.轉txt.
        範例:
        input_directory =  # 來源目錄
        output_directory =   # 目標目錄
        convert_md_to_txt(input_directory, output_directory)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith('.md'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace('.md', '.txt'))

                with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                    lines = infile.readlines()
                    for i, line in enumerate(lines):
                        if i + 1 < len(lines) and re.match(r'^-+$', lines[i + 1].strip()):
                            outfile.write(f'*{line.strip()}\n')
                        else:
                            outfile.write(line)

        print(f"轉換完成。輸出文件保存在 {output_dir} 目錄中。")

    def read_txt_files_to_dataframe(directory):
        """
        讀取txt.
        """
        data = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    title = os.path.splitext(filename)[0]
                    data.append({'Title': title, 'Content': content})
        
        df = pd.DataFrame(data)
        
        return df

    def concatenate_text(examples):
        return {
            "text": examples["Title"]
            + " \n "
            + examples["Content"]

        }
