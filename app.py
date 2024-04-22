from flask import Flask, request, jsonify
from flask_cors import CORS
from flask.json.provider import JSONProvider

from components.Taikyoku_loader import Taikyoku_loader
from components.dict import pai_dict
import os, orjson
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
filename = 'test.log'
loader = Taikyoku_loader(os.path.join(path, filename))


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


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)