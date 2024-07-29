# 将xml格式的牌谱数据转换为编码格式
import re, gzip
from bs4 import BeautifulSoup
from components.dict import *
from components.player import Player
from components.taikyoku_info import *
from typing import Dict


class Taikyoku_loader(object):

    def __init__(self, file, doubleRon = False ) -> None:

        # 初始化牌谱数据
        self.file = file
        self.data = self._load_file(file) # 存放牌谱数据
        self.log = self._get_kyoku(doubleRon)

        # 引入Taikyoku_info类，存放对局信息
        # 引入Player类，存放玩家信息
        self.taikyoku_info = Taikyoku_info()
        self.players = {
            0: Player('0'),
            1: Player('1'),
            2: Player('2'),
            3: Player('3')
        } # 玩家id为键，Player类为值

        # 默认初始化第一局的玩家信息和对局信息
        self.current_kyoku_index = 0    # 当前局的索引
        self.current_tag_index = 0      # 当前标签的索引
        self._init_player_taikyoku_info()

        pass

    # 初始化第num局的玩家信息, num=0表示默认初始化第一局
    def _init_player_taikyoku_info(self, num=0):
        try:
            soup = BeautifulSoup(self.log[num][0], 'lxml')
        except:
            print("对局不存在！或未安装lxml库！")
            sys.exit()

        if soup.find('init'):
            tehai = self._get_hai(soup.find('init'))
            oya = int(soup.init.get('oya'))
            ten = soup.init.get('ten')
            ten = ten.strip().split(',')
            seed = soup.init.get('seed')
            seed = seed.strip().split(',')
            for i in range(4):
                self.players[i].reset_next_kyoku()
                self.players[i].tehai = tehai[i]
                self.players[i].point = int(ten[i])*100

                if i == oya:
                    self.players[i].isOya = True
            self.taikyoku_info.oya = oya

            # 根据seed信息确定场风
            if int(seed[0]) < 4:
                self.taikyoku_info.bakaze = Bakaze.EAST
            elif int(seed[0]) < 8:
                self.taikyoku_info.bakaze = Bakaze.SOUTH
            elif int(seed[0]) < 12:
                self.taikyoku_info.bakaze = Bakaze.WEST
            elif int(seed[0]) < 16:
                self.taikyoku_info.bakaze = Bakaze.NORTH
            else:
                print("场风错误！")
                return

            # 更新本场数，立直棒数，拱托数，宝牌
            self.taikyoku_info.kyoku = int(seed[0]) % 4 + 1
            self.taikyoku_info.honba = int(seed[1])
            self.taikyoku_info.reach_stick = int(seed[2])
            self.taikyoku_info.calc_kyotaku()
            self.taikyoku_info.set_dora(int(seed[-1]))
            self.taikyoku_info.set_yama(tehai)
            self.taikyoku_info.set_remain_draw(70)
            self.taikyoku_info.reset_record()

        
        else:
            print("初始化信息不存在！")
            return
        
        pass

    def _load_file(self, file):
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            data = f.read()
        return data
    
    # 用正则表达式匹配所有的对局数据，并转化为实际的对局信息并返回
    def _get_kyoku(self, doubleRon):
        # 存放每一局的数据
        kyoku_all = []
        # 将每一局的数据提取出来
        pattern_segment = re.compile(r'(<INIT.*?>)(.*?)(?=<INIT|$)', re.DOTALL)
        # 判断是否双响
        pattern_agari_2 = re.compile(r'(<AGARI.*?><AGARI.*?>)', re.DOTALL)
        # 分割标签
        pattern_tag = re.compile(r'<.*?/>', re.DOTALL)

        matches_kyoku = pattern_segment.finditer(self.data)
        index = 0
        for match_kyoku in matches_kyoku:
            index += 1
            log = [match_kyoku.group(1)]
            match_agari_2 = pattern_agari_2.findall(match_kyoku.group(2))
            matches_tag = pattern_tag.findall(match_kyoku.group(2))
            # print(matches_tag)
            if match_agari_2 and doubleRon:
                print("出现双响！")
                print("对局文件为：", self.file,"第", index, "局")
                print("=====================================")
            for tag in matches_tag:
                log.append(tag)

            
            kyoku_all.append(log)               

        return kyoku_all

    def _get_hai(self, tag):
        hai = [[], [], [], []]
        integer_hai = [[], [], [], []]
        for i in range(4):
            hai[i] = tag.get(f'hai{i}')
            hai[i] = hai[i].split(',')
            integer_hai[i] = [int(x) for x in hai[i]]
            integer_hai[i].sort()
        return integer_hai


    #_handdle_tag仅处理标签，具体的玩家信息和对局信息由自身的函数处理
    # 返回值的格式为：
    # {
    #     'player': player,
    #     'action': action,
    #     'tile': tile，
    #     'reward': reward 
    #}
    def _haddle_tag(self, tag, forward: bool, print_info=False):
        reward = [0, 0, 0, 0]
        soup = BeautifulSoup(tag, 'lxml')
        pattern_draw = re.compile(r'<([TUVW])(\d{1,3})/>')
        pattern_discard = re.compile(r'<([DEFG])(\d{1,3})/>')
        if forward:
            if pattern_draw.match(tag):
                # 解析标签
                match = pattern_draw.match(tag)

                player = draw_discard_dict_int[match.group(1)]
                tile = int(match.group(2))

                # 更新玩家信息
                action = self.players[player].draw(tile)
                self.taikyoku_info.yama.remove(tile)
                self.taikyoku_info.remain_draw -= 1

                # 记录
                self.taikyoku_info.record.append({'player': player, 'action': action, 'tile': tile})

                if print_info:
                    print("player", player, "action", action, "tile", pai_dict[tile])

                return {'player': player, 'action': action, 'tile': tile, 'reward': reward, 'done': False}

            elif pattern_discard.match(tag):
                match = pattern_discard.match(tag)

                player = draw_discard_dict_int[match.group(1)]
                tile = int(match.group(2))
                action = self.players[player].discard(tile)

                # 记录
                self.taikyoku_info.record.append({'player': player, 'action': action, 'tile': tile})

                if print_info:
                    print("player", player, "action", action, "tile", pai_dict[tile])

                return {'player': player, 'action': action, 'tile': tile, 'reward': reward, 'done': False}

            elif soup.find('reach'):
                step = int(soup.find('reach').get('step'))
                who = int(soup.find('reach').get('who'))

                action = self.players[who].reach(step)

                # 记录
                self.taikyoku_info.record.append({'player': who, 'action': action, 'tile': None})

                if print_info:
                    print("player", who, "立直", "step", step)

                return {'player': who, 'action': action, 'tile': None, 'reward': reward, 'done': False}


            elif soup.find('dora'):
                self.taikyoku_info.append_dora(int(soup.dora.get('hai')))

                if print_info:
                    print("新宝牌为：", pai_dict[int(soup.dora.get('hai'))])

                return {'player': None, 'action': None, 'tile': None, 'reward': reward, 'done': False}
            elif soup.find('n'):
                m = soup.find('n').get('m')
                who = int(soup.find('n').get('who'))

                naki_hai, result, action_type= self.players[who].handle_naki(m, print_info=print_info)
                if print_info:
                    print(action_type)
                    print("player", who, "副露")
                    print("副露牌为：", [pai_dict[x] for x in result])
                    print("副露类型：", action_type)

                # 记录
                self.taikyoku_info.record.append({'player': who, 'action': action_type, 'tile': naki_hai})
                return {'player': who, 'action': action_type, 'tile': naki_hai, 'reward': reward, 'done': False}
            
            # elif soup.find('bye'):
            #     return {'bye': True, 'done': True}

            elif soup.find('agari'):
                done = False
                who = int(soup.agari.get('who'))
                fromWho = int(soup.agari.get('fromwho'))
                action = self.players[who].agari()
                machi = int(soup.agari.get('machi'))
                # 判断是否有owari
                # result = [0, 0, 0, 0]
                if soup.agari.get('owari'):
                    done = True
                    owari = soup.agari.get('owari')
                    owari = owari.strip().split(',')
                    owari = [float(x) for x in owari]
                    result = [int(owari[i*2]) for i in range(4)]
                    # 根据result计算reward，第一名为4,第二名为3，第三名为2，第四名为1
                    sorted_indices = sorted(range(len(result)), key=lambda k: result[k], reverse=True)
                    for rank, i in enumerate(sorted_indices):
                        reward[i] = 4 - rank

                    if print_info:
                        print("终局")

                sc_temp = soup.agari.get('sc')
                sc = sc_temp.strip().split(',')
                sc = [int(x) for x in sc]

                if print_info:
                    print("player", who, "agari", "from player", fromWho)
                    print(sc)

                for i in range(4):
                    self.players[i].point += sc[i*2+1]*100

                return {'player': who, 'action': action, 'tile': machi, 'reward': reward, 'done': done}
            
            elif soup.find('ryuukyoku'):

                sc_temp = soup.ryuukyoku.get('sc')
                sc = sc_temp.strip().split(',')
                sc = [int(x) for x in sc]
                done = False

                if soup.ryuukyoku.get('owari'):
                    done = True
                    owari = soup.ryuukyoku.get('owari')
                    owari = owari.strip().split(',')
                    owari = [float(x) for x in owari]
                    result = [int(owari[i*2]) for i in range(4)]
                    # 根据result计算reward，第一名为4,第二名为3，第三名为2，第四名为1
                    sorted_indices = sorted(range(len(result)), key=lambda k: result[k], reverse=True)
                    for rank, i in enumerate(sorted_indices):
                        reward[i] = 4 - rank

                for i in range(4):
                    self.players[i].point += sc[i*2+1]*100

                if print_info:
                    print("流局")
                    ryuukyoku_type = soup.ryuukyoku.get('type')
                    if ryuukyoku_type:
                        print("流局类型：", ryuukyoku_type)
                    print(sc)

                return {'player': 0, 'action': None, 'tile': None, 'reward': reward, 'done': done}
            
            elif soup.find('init'):
                return {'player': None, 'action': None, 'tile': None, 'reward': reward, 'done': False}

        else:
            print("后退")
        pass

    def _pre_haddle_tag(self, tag):
        soup = BeautifulSoup(tag, 'lxml')
        pattern_discard = re.compile(r'<([DEFG])(\d{1,3})/>')
        player = None
        action = None

        ouput_state = False

        if pattern_discard.match(tag):
            match = pattern_discard.match(tag)
            player = draw_discard_dict_int[match.group(1)]
            action = Action.DISCARD_tegiri # 手摸切无关紧要
            ouput_state = True

        elif soup.find('reach'):
            step = int(soup.find('reach').get('step'))
            player = int(soup.find('reach').get('who'))
            ouput_state = True

            if step == 1:
                action = Action.REACH_declear
            elif step == 2:
                action = Action.REACH_success
        
        elif soup.find('n'):
            player = int(soup.find('n').get('who'))
            action = Action.PON_fromKami
            ouput_state = True

        elif soup.find('agari'):
            player = int(soup.agari.get('who'))
            action = Action.AGARI
            ouput_state = True

        elif soup.find('ryuukyoku'):
            ouput_state = True
            action = Action.NONE

        return {"player": player, "action": action, "ouput_state": ouput_state}
            


    # 每执行一次step函数，读取一个xml标签，更新玩家信息和对局信息
    def step_forward(self, print_info=False):
        self.current_tag_index += 1
        if self.current_tag_index >= len(self.log[self.current_kyoku_index]):
            if print_info:
                print("对局结束！")
            if self.current_kyoku_index == len(self.log)-1:
                # 回到第一局的第一步
                self.reset(num=1)
                tag = self.log[self.current_kyoku_index][self.current_tag_index]
                return self._haddle_tag(tag, forward=True, print_info=print_info)
            else:
                self.reset(self.current_kyoku_index+2)# 重置到下一局
                tag = self.log[self.current_kyoku_index][self.current_tag_index]
                return self._haddle_tag(tag, forward=True, print_info=print_info)
            
        else:
            tag = self.log[self.current_kyoku_index][self.current_tag_index]
            data = self._haddle_tag(tag, forward=True, print_info=print_info)
            if print_info:
                print(data)

            return data

    # TODO: 后退一步
    def step_backward(self):
        if self.current_tag_index == 0:
            if self.current_kyoku_index == 0:
                # 到最后一局的第一步
                self.reset(num=len(self.log))    
            else:
                # 到上一局开始
                self.reset(self.current_kyoku_index)
            
        else:
            tag = self.log[self.current_kyoku_index][self.current_tag_index]
            self._haddle_tag(tag, forward=False)
            self.current_tag_index -= 1
        pass

    # 回到第num局初始化的状态
    def reset(self, num=1):
        self._init_player_taikyoku_info(num-1)
        self.current_kyoku_index = num-1
        self.current_tag_index = 0
        pass

    def print_loader(self):
        self.taikyoku_info.print_info()
        for i in range(4):
            self.players[i].print_player()

    # 导出当前状态的信息，用于Encoder的encode函数
    def export_info(self)-> Dict[str, object]:
        info = {
            'taikyoku_info': {
                'bakaze': self.taikyoku_info.bakaze,
                'kyoku': self.taikyoku_info.kyoku,
                'oya': self.taikyoku_info.oya,
                'honba': self.taikyoku_info.honba,
                'reach_stick': self.taikyoku_info.reach_stick,
                'kyotaku': self.taikyoku_info.kyotaku,
                'dora': self.taikyoku_info.dora,
                'remain_draw': self.taikyoku_info.remain_draw,
                'record': self.taikyoku_info.record
            },
            'player0': {
                'tehai': self.players[0].tehai,
                'naki': self.players[0].naki,
                'sutehai': self.players[0].sutehai,
                'point': self.players[0].point,
                'isReach': self.players[0].isReach,
                # TODO: 将来加入听牌类型检测
                # 'machi': self.players[0].machi
            },
            'player1': {
                'tehai': self.players[1].tehai,
                'naki': self.players[1].naki,
                'sutehai': self.players[1].sutehai,
                'point': self.players[1].point,
                'isReach': self.players[1].isReach,
            },
            'player2': {
                'tehai': self.players[2].tehai,
                'naki': self.players[2].naki,
                'sutehai': self.players[2].sutehai,
                'point': self.players[2].point,
                'isReach': self.players[2].isReach,
            },
            'player3': {
                'tehai': self.players[3].tehai,
                'naki': self.players[3].naki,
                'sutehai': self.players[3].sutehai,
                'point': self.players[3].point,
                'isReach': self.players[3].isReach,
            }
        }
        return info