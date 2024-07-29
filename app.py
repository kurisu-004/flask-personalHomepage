from flask import Flask, request, jsonify
from flask_cors import CORS
from flask.json.provider import JSONProvider

from components.Taikyoku_loader import Taikyoku_loader
from components.encoder import Encoder
from components.Bert_like import Bert_like
from components.dict import pai_dict, encode_dict
import os, orjson, torch
import numpy as np
from enum import Enum


class OrJsonProvider(JSONProvider):
    def dumps(self, obj, *, option=None, **kwargs):
        if option is None:
            option = orjson.OPT_APPEND_NEWLINE | orjson.OPT_NAIVE_UTC

        def default(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"{obj!r} is not JSON serializable")
        
        return orjson.dumps(obj, option=option).decode()

    def loads(self, s, **kwargs):
        return orjson.loads(s)

app = Flask(__name__)
app.debug = True
app.json = OrJsonProvider(app)

app.config['UPLOAD_FOLDER'] = 'E:/Code/flask-personalHomepage/log'  # 设置文件上传的目标文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
ALLOWED_EXTENSIONS = {'log'}  # 允许的文件扩展名


cors = CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000"}})

path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log')
filename = '10001.gz'
loader = Taikyoku_loader(os.path.join(path, filename))
loader.reset(3)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OPTIONS request received!'})  # 用于预检请求
    if request.method == 'POST':
        # 检查是否有文件被上传
        print(request.files)
        return jsonify({'message': 'POST request received!'})
    
@app.route('/api/step_forward', methods=['GET'])
def step_forward():
    loader.step_forward()
    data = loader.export_info()

    # 将手牌信息转换为前端所需的格式
    for i in range(4):
        tehai = data[f'player{i}']['tehai']
        temp = []
        for hai in tehai:
            temp.append(pai_dict[hai])
        data[f'player{i}']['tehai'] = temp

        for item in data[f'player{i}']['sutehai']:
           if item['hai'] in pai_dict:
               item['hai'] = pai_dict[item['hai']]
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = {}  # 用于存储返回的数据
    # 获取前端传来的数据

    try:
        data = request.get_json()
        # print(data['player0']['tehai'])
        # print(data['player0']['sutehai'])
        # print(data['player1']['naki'])
        # print(type(data))


        # 将数据转化为模型所需的格式

        encoder = Encoder(data)
        ori_data, masked_data = encoder.encode(mask_type=1)
        input = torch.tensor(ori_data.T, dtype=torch.float32).reshape(1, -1, 55)
        masked_data = masked_data.T

        # masked_data形状为(seq_len, 1)，其中seq_len为ori_data的列数，即ori_data的行数
        # 使用repeat函数将masked_data的列数扩展为ori_data的列数 
        masked_data = np.repeat(masked_data, 55, axis=1).reshape(1, -1, 55)

        mask = torch.tensor(masked_data, dtype=torch.float32)

        # 将mask为1的位置替换为2
        input = torch.where(mask == 1, torch.tensor(2, dtype=torch.float32), input)

        # 调用模型进行预测
        model = Bert_like()
        checkpoint_file = './model/predict.tar'
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            output = model(input)

        print(output)
        print(output.shape)

        output_list = []
        # 仅提取预测结果中mask为1的位置的值
        for i in range(output.shape[1]):
            if mask[0, i, 0] == 1:
                tile = torch.argmax(output[0, i, :]).item()
                output_list.append(tile)
        print(output_list)
        tile_list = [encode_dict['int_2hai'][x] for x in output_list]
        print(tile_list)
        # print(len(output_list))


        # 将预测结果转化为前端所需的格式
    

        return jsonify({"status": "success", "data": tile_list})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    finally:
        pass


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)