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
        return jsonify({'message': 'OPTIONS request received!'})  # 用于预检请求
    if request.method == 'POST':
        # 检查是否有文件被上传
        print(request.files)
        return jsonify({'message': 'POST request received!'})
    
@app.route('/api/step_forward', methods=['GET'])
def step_forward():
    return jsonify({
        'taikyoku_info': {
                'bakaze': 0,
                'kyoku': 0,
                'oya': 0,
                'honba': 0,
                'reach_stick': 0,
                'kyotaku': 0,
                'dora': ['3s'],
                'remain_draw': 70,
            },
            'player0': {
                'tehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'naki': [],
                'sutehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'point': 25000,
                'isReach': False,
            },
            'player1': {
                'tehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'naki': [],
                'sutehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'point': 25000,
                'isReach': False,
            },
            'player2': {
                'tehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'naki': [],
                'sutehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'point': 25000,
                'isReach': False,
            },
            'player3': {
                'tehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'naki': [],
                'sutehai': ['2m', '4m', '7m', '8m', '6p', '7p', '8p', '3s', '5s', '8s', '9s', '9s', '6z'],
                'point': 25000,
                'isReach': False,
            }
    })



if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)