import json
file_path = "C:/Users/user/.keras/keras.json"
with open(file_path, 'r') as file:
    data = json.load(file)
    print(type(data))
    print(data)

# 데이터 수정
data['backend'] = 'tensorflow'
#data["Olivia"]["hobby"].append("take a picture")

# 기존 json 파일 덮어쓰기
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent="\t")
