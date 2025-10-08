import re
import os

def extract_tables_from_file(file_path, separator='\n\n'):
    """
    从指定的文本文件中提取所有格式化的表格文本块。

    Args:
        file_path (str): 输入的txt文件路径。
        separator (str, optional): 用于分隔提取出的文本块的字符串。
                                 默认为两个换行符（一个空行）。

    Returns:
        str: 包含所有提取的文本块，并由分隔符隔开的字符串。
             如果未找到匹配项，则返回空字符串。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None, f"错误：输入文件 '{file_path}' 未找到。"
    except Exception as e:
        return None, f"读取文件时发生错误: {e}"

    # 正则表达式
    pattern = re.compile(r"^\s*\|   tst   \|[^\n]*\n(?:^\s*\|  Dom \d  \|[^\n]*\n?)+", re.MULTILINE)

    found_blocks = pattern.findall(content)

    if found_blocks:
        # 清理每个块并用分隔符连接
        result_text = separator.join([block.strip() for block in found_blocks])
        return result_text, f"成功提取 {len(found_blocks)} 个文本块。"
    else:
        return None, "在文件中未找到匹配的文本块。"

def save_string_to_file(content, output_path):
    """
    将字符串内容写入指定的文件。
    """
    if not content:
        print("没有内容需要保存。")
        return

    try:
        # 确保输出文件所在的目录存在，如果不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")

        # 以写入模式 ('w') 打开文件，这会覆盖已有文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 结果已成功写入文件: {output_path}")

    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")




input_log_name = "./log/BEST/Pretrain-BEST-20251006-210259.txt"      
output_file_name = "./result/BEST/Pretrain-BEST-20251006-210259.txt" 





my_separator = "\n\n" + "="*100 +"\n\n"

print(f"开始从 {input_log_name} 提取...")
extracted_content, message = extract_tables_from_file(input_log_name, separator=my_separator)

print(message) 


if extracted_content:
    # (可选) 在控制台预览一下
    # print("--- 预览提取内容 ---")
    # print(extracted_content)
    # print("--- 预览结束 ---\n")

    
    save_string_to_file(extracted_content, output_file_name)
else:
    print("因提取失败或无内容，未执行保存操作。")