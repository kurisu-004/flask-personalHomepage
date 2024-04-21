from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.debug = True
app.config['UPLOAD_FOLDER'] = 'E:/Code/flask-personalHomepage/log'  # 设置文件上传的目标文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
ALLOWED_EXTENSIONS = {'log'}  # 允许的文件扩展名


cors = CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8000"}})

@app.route('/api', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello, World!'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'OPTIONS request received!'})  # 处理跨域请求
    elif request.method == 'POST':
        return jsonify({'message': 'POST request received!'})



if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)