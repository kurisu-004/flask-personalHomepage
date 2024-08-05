from enum import Enum, auto
# 0~135表示136种牌
pai_dict = {
    # 0~35: 万
    0: '1m', 1: '1m', 2: '1m', 3: '1m', 4: '2m', 5: '2m', 6: '2m', 7: '2m', 8: '3m', 9: '3m', 10: '3m', 11: '3m',
    12: '4m', 13: '4m', 14: '4m', 15: '4m', 16: '0m', 17: '5m', 18: '5m', 19: '5m', 20: '6m', 21: '6m', 22: '6m',
    23: '6m', 24: '7m', 25: '7m', 26: '7m', 27: '7m', 28: '8m', 29: '8m', 30: '8m', 31: '8m', 32: '9m', 33: '9m',
    34: '9m', 35: '9m',
    # 36~71: 筒
    36: '1p', 37: '1p', 38: '1p', 39: '1p', 40: '2p', 41: '2p', 42: '2p', 43: '2p', 44: '3p', 45: '3p', 46: '3p',
    47: '3p', 48: '4p', 49: '4p', 50: '4p', 51: '4p', 52: '0p', 53: '5p', 54: '5p', 55: '5p', 56: '6p', 57: '6p',
    58: '6p', 59: '6p', 60: '7p', 61: '7p', 62: '7p', 63: '7p', 64: '8p', 65: '8p', 66: '8p', 67: '8p', 68: '9p',
    69: '9p', 70: '9p', 71: '9p',
    # 72~107: 条
    72: '1s', 73: '1s', 74: '1s', 75: '1s', 76: '2s', 77: '2s', 78: '2s', 79: '2s', 80: '3s', 81: '3s', 82: '3s',
    83: '3s', 84: '4s', 85: '4s', 86: '4s', 87: '4s', 88: '0s', 89: '5s', 90: '5s', 91: '5s', 92: '6s', 93: '6s',
    94: '6s', 95: '6s', 96: '7s', 97: '7s', 98: '7s', 99: '7s', 100: '8s', 101: '8s', 102: '8s', 103: '8s', 104: '9s',
    105: '9s', 106: '9s', 107: '9s',
    # 108~135: 字
    108: '1z', 109: '1z', 110: '1z', 111: '1z', 112: '2z', 113: '2z', 114: '2z', 115: '2z', 116: '3z', 117: '3z',
    118: '3z', 119: '3z', 120: '4z', 121: '4z', 122: '4z', 123: '4z', 124: '5z', 125: '5z', 126: '5z', 127: '5z',
    128: '6z', 129: '6z', 130: '6z', 131: '6z', 132: '7z', 133: '7z', 134: '7z', 135: '7z'
}
player_dict = {
    '0': '自家',
    '1': '下家',
    '2': '对家',
    '3': '上家'
}
kyoku_dict = {
    '0': '东一局',
    '1': '东二局',
    '2': '东三局',
    '3': '东四局',
    '4': '南一局',
    '5': '南二局',
    '6': '南三局',
    '7': '南四局',
    '8': '西一局',
    '9': '西二局',
    '10': '西三局',
    '11': '西四局',
    '12': '北一局',
    '13': '北二局',
    '14': '北三局',
    '15': '北四局'
}
draw_dict = {
    'T': '自家摸牌',
    'U': '下家摸牌',
    'V': '对家摸牌',
    'W': '上家摸牌'
}
discard_dict = {
    'D': '自家弃牌',
    'E': '下家弃牌',
    'F': '对家弃牌',
    'G': '上家弃牌'
}
draw_discard_dict_int = {
    'T': 0,
    'U': 1,
    'V': 2,
    'W': 3,
    'D': 0,
    'E': 1,
    'F': 2,
    'G': 3
}

color_dict = {
    0: 'm',
    1: 'p',
    2: 's',
}
number_dict = {
    0: '123',
    1: '234',
    2: '345',
    3: '456',
    4: '567',
    5: '678',
    6: '789'
}

encode_dict = {
    'action': {

    },

    'hai_2int': {
        '1m': 0, '2m': 1, '3m': 2, '4m': 3, '5m': 4, '6m': 5, '7m': 6, '8m': 7, '9m': 8,
        '1p': 9, '2p': 10, '3p': 11, '4p': 12, '5p': 13, '6p': 14, '7p': 15, '8p': 16, '9p': 17,
        '1s': 18, '2s': 19, '3s': 20, '4s': 21, '5s': 22, '6s': 23, '7s': 24, '8s': 25, '9s': 26,
        '1z': 27, '2z': 28, '3z': 29, '4z': 30, '5z': 31, '6z': 32, '7z': 33,
        '0m': 34, '0p': 35, '0s': 36
    },
    'int_2hai': {
        0: '1m', 1: '2m', 2: '3m', 3: '4m', 4: '5m', 5: '6m', 6: '7m', 7: '8m', 8: '9m',
        9: '1p', 10: '2p', 11: '3p', 12: '4p', 13: '5p', 14: '6p', 15: '7p', 16: '8p', 17: '9p',
        18: '1s', 19: '2s', 20: '3s', 21: '4s', 22: '5s', 23: '6s', 24: '7s', 25: '8s', 26: '9s',
        27: '1z', 28: '2z', 29: '3z', 30: '4z', 31: '5z', 32: '6z', 33: '7z',
        34: '0m', 35: '0p', 36: '0s'
    },

}

# 定义场风类
class Bakaze(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

# 定义对局结束原因类
class EndKyoku(Enum):
    TSUMO = auto()
    RON = auto()
    RYUUKYOKU = auto()

# 定义动作类
class Action(Enum):
    DRAW = 0               # 摸牌
    DISCARD_tsumokiri = 1  # 摸切
    DISCARD_tegiri = 2     # 手切
    CHI_top = 3            # 吃的牌为顶张
    CHI_middle = 4         # 吃的牌为中间张
    CHI_bottom = 5         # 吃的牌为底张
    PON_fromShimo = 6      # 碰下家的牌
    PON_fromToimen = 7     # 碰对家的牌
    PON_fromKami = 8       # 碰上家的牌
    MINKAN_fromShimo = 9   # 明杠下家的牌
    MINKAN_fromToimen = 10 # 明杠对家的牌
    MINKAN_fromKami = 11   # 明杠上家的牌
    ANKAN = 12             # 暗杠
    KAKAN = 13             # 加杠
    REACH_declear = 14     # 立直宣言
    REACH_success = 15     # 立直成功
    AGARI = 16             # 和牌
    PASS = 17              # 过
    NONE = auto()          # 无动作

# 定义副露类
class Naki(Enum):
    CHI = auto()
    PON = auto()
    MINKAN = auto()
    ANKAN = auto()
    KAKAN = auto()


# 定义动作名
action_names = {
    Action.DRAW: "DRAW",
    Action.DISCARD_tsumokiri: "DISCARD_tsumokiri",
    Action.DISCARD_tegiri: "DISCARD_tegiri",
    Action.CHI_top: "CHI_top",
    Action.CHI_middle: "CHI_middle",
    Action.CHI_bottom: "CHI_bottom",
    Action.PON_fromShimo: "PON_fromShimo",
    Action.PON_fromToimen: "PON_fromToimen",
    Action.PON_fromKami: "PON_fromKami",
    Action.MINKAN_fromShimo: "MINKAN_fromShimo",
    Action.MINKAN_fromToimen: "MINKAN_fromToimen",
    Action.MINKAN_fromKami: "MINKAN_fromKami",
    Action.ANKAN: "ANKAN",
    Action.KAKAN: "KAKAN",
    Action.REACH_declear: "REACH_declear",
    Action.REACH_success: "REACH_success",
    Action.AGARI: "AGARI",
    Action.PASS: "PASS"
}

# 构建解码字典
decode_dict = {}
num_tile = 37

for action in Action:
    if action == Action.NONE:
        continue
    for tile, tile_index in encode_dict['hai_2int'].items():
        index = action.value * num_tile + tile_index
        decode_dict[index] = f"{action_names[action]}_{tile}"

# # 打印解码字典
# for k, v in decode_dict.items():
#     print(f"{k}: {v}")
