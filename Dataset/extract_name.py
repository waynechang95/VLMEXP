import json

# 讀取 JSON 文件
with open('2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定義一個函數來萃取所有的 name


def extract_names(data):
    names = []
    for entity in data['financial_entities']:
        for item in entity['items']:
            names.append(item['name'])
    return names


# 萃取所有的 name
names = extract_names(data)

output = {
    "Image": '2.jpg',
    "Entity": names
}

# 將 names 列表轉換成 JSON 格式並寫入到新的 JSON 文件中
with open('extracted_names.json', 'w', encoding='utf-8') as outfile:
    json.dump(output, outfile, ensure_ascii=False, indent=4)
