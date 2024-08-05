from flask import Flask, request, jsonify
from flask_cors import CORS
from flask.json.provider import JSONProvider

from components.Taikyoku_loader import Taikyoku_loader
from components.encoder import Encoder
from components.Bert_like import Bert_like
from components.dict import pai_dict, encode_dict, Action, decode_dict
import os, orjson, torch
import numpy as np
from enum import Enum
from components.DecisionTransformer import TrainableDT, DTDataCollator
from typing import List, Dict


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

# 调用模型进行预测
predictModel = Bert_like()
checkpoint_file = './model/predict.tar'
checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
predictModel.load_state_dict(checkpoint['model_state_dict'])
predictModel.eval()

# config = DecisionTransformerConfig(state_dim=32,
#                                     act_dim=666,
#                                     hidden_size=128,
#                                     action_tanh=False,
#                                     n_positions=512*3,
#                                     n_heads=4)
# myDTmodel = TrainableDT(config)
DT_checkpoint_file = './model/checkpoint-30780'
myDTmodel = TrainableDT.from_pretrained(DT_checkpoint_file)
myDTmodel.eval()

# 记录当前的状态和动作
state_list = []
action_list = []

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
    if loader.current_tag_index == len(loader.log[loader.current_kyoku_index])-1:
        info = loader.step_forward()

    else:
        # 检查是否是主视角玩家的特定行动
        check = loader._pre_haddle_tag(loader.log[loader.current_kyoku_index][loader.current_tag_index+1])
        # 判断是否是主视角行动,且动作不是摸牌和立直成功
        if check['player'] == 0 and check['ouput_state']:
            # 获取当前时刻局的信息
            data_now = loader.export_info()

            # 修改data_now中的格式
            for i in range(4):
                tehai = data_now[f'player{i}']['tehai']
                temp = []
                for hai in tehai:
                    temp.append(pai_dict[hai])
                data_now[f'player{i}']['tehai'] = temp

                for item in data_now[f'player{i}']['sutehai']:
                    if item['hai'] in pai_dict:
                        item['hai'] = pai_dict[item['hai']]
            # data_now = jsonify(data_now)


            output, _ = getPredictOutput(data_now, getLogit=True)
            # print(output.shape)
            state_list.append(output[0, 0, :].detach().numpy())
            # print("stateList is :", state_list)
            # print("len of stateList is :", len(state_list))
            info = loader.step_forward()
            action_list.append({'action':info['action'], 'tile':info['tile']})
            # print("actionList is :", action_list)
        else:
            info = loader.step_forward()
            # print(info)


    if info is None:
        print('出现异常标签')
        return jsonify({'status': 'error', 'message': '出现异常标签'})

    data_next = loader.export_info()

    # 将手牌信息转换为前端所需的格式
    for i in range(4):
        tehai = data_next[f'player{i}']['tehai']
        temp = []
        for hai in tehai:
            temp.append(pai_dict[hai])
        data_next[f'player{i}']['tehai'] = temp

        for item in data_next[f'player{i}']['sutehai']:
           if item['hai'] in pai_dict:
               item['hai'] = pai_dict[item['hai']]
    return jsonify(data_next)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = {}  # 用于存储返回的数据
    # 获取前端传来的数据

    try:
        data = request.get_json()

        output, mask = getPredictOutput(data)


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


@app.route('/api/getAction', methods=['GET'])
def getAction():
    action = action_tile_2_onehot(action_list)

    # 把action的最后一列替换为0用于预测
    action[-1] = np.zeros(action.shape[1])
    data = [{
        'state': state_list,
        'action': action,
        # 长度与state_list相同
        'rtg': [4 for _ in range(len(state_list))]
    }]
    collator = DTDataCollator(data)

    # 检查数据是否合法
    input = collator.__call__(data)
    # print(type(input))
    # print(input.keys())
    # print(input['states'].shape)
    # print(input['actions'].shape)
    # print(input['rewards'].shape)
    # print(input['returns_to_go'].shape)
    # print(input['timesteps'].shape)
    # print(input['attention_mask'].shape)
    # # print(input['states'][-1])
    # print(input['rewards'][0][-5:-1])
    # print(input['returns_to_go'][0][-5:-1])
    # print(action_list, state_list)

    # 调用模型进行预测
    with torch.no_grad():
        state_preds, action_preds, return_preds = myDTmodel.original_forward(**input, return_dict=False)

    # 返回最高的5个动作
    # action_pred = int(action_preds[0,-1].argmax())
    action_pred = action_preds[0,-1]
    top_value, top_index = torch.topk(action_pred, 20)

    for i, a in enumerate(top_index):
        print(decode_dict[int(a)],"概率：", top_value[i])

    return jsonify({'topAction': [decode_dict[int(a)] for a in top_index],
                    'topValue': [float(v) for v in top_value]})





def getPredictOutput(data, model=predictModel, getLogit=False):
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

        with torch.no_grad():
            if not getLogit:
                output = model(input)
            else:
                output = model.get_hidden_state(input)

        return output, mask


def action_tile_2_onehot(actions: List[Dict[str, object]]) -> np.ndarray:

    # 参数:
    # actions: 一个列表，其中每个元素是一个字典，包含 'tile' 和 'action' 两个键。
    #          'action' 键对应 Action 枚举类型


    # 每一行代表一种动作，最后一行表示执行动作的玩家
    num_action = len(Action) - 1 # 排除 Action.NONE
    num_tile = 37 # 34种牌 + 3张红宝牌
    encoded_action = np.zeros((len(actions), num_action*num_tile), dtype=np.int8)

    for i, action in enumerate(actions):

        try:
            if action['action'] == Action.REACH_declear:
                tile =  0
            else:
                tile = encode_dict['hai_2int'][pai_dict[action['tile']]]
                
            encoded_action[i, action['action'].value*num_tile + tile] = 1
        except:
            print(f"Action: {action['action']}, Tile: {action['tile']}")
            raise

    return encoded_action

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)