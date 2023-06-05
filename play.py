# coding:utf-8

import torch
import torch.nn.functional as F
from torch import  cuda

from alphazero.alpha_zero_mcts import AlphaZeroMCTS
from alphazero.chess_board import ChessBoard

# model_path='model/best_policy_value_net_6400.pth'
# c_puct=4
# n_mcts_iters=500
# board_len=9
# n_feature_planes=4
# chess_board = ChessBoard(board_len, n_feature_planes)

# best_model = torch.load(model_path)  # type:PolicyValueNet
# best_model.eval()
# best_model.set_device(True)
# mcts = AlphaZeroMCTS(best_model, c_puct, n_mcts_iters)
# mcts_human = AlphaZeroMCTS(best_model, c_puct, n_mcts_iters)

class BetaGobang:
    def __init__(self, board_len=9, n_self_plays=1500, n_mcts_iters=500,
                 n_feature_planes=6, c_puct=4, is_use_gpu=True, **kwargs):
        self.c_puct = c_puct
        self.is_use_gpu = is_use_gpu
        self.n_self_plays = n_self_plays
        self.n_mcts_iters = n_mcts_iters
        self.device = torch.device(
            'cuda:0' if is_use_gpu and cuda.is_available() else 'cpu')
        self.chess_board = ChessBoard(board_len, n_feature_planes)
    
    def __do_mcts_action(self,mcts):
        """ 获取AI动作 """
        print("AI is computing...")
        action = mcts.get_action(self.chess_board)
        self.chess_board.do_action(action)
        print("AI's output:"+str(action))
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner
    def __do_human_action(self):
        """ 获取动作 """
        action = input("輸入放置位置:")
        action = int(action)
        # action = mcts_human.get_human_action(chess_board)
        self.chess_board.do_action(action)
        is_over, winner = self.chess_board.is_game_over()
        return is_over, winner

    def start_game(self):
        model_path='model/best_policy_value_net_6400.pth'
        best_model = torch.load(model_path)
        best_model.eval()
        best_model.set_device(self.is_use_gpu)
        mcts = AlphaZeroMCTS(best_model, self.c_puct, self.n_mcts_iters)
        # 开始比赛
        print('🩺 正在测试模型...')
        self.chess_board.clear_board()
        mcts.reset_root()
        while True:
            # 当前模型走一步
            is_over, winner = self.__do_human_action()
            if is_over: 
                break
            # 历史最优模型走一步
            is_over, winner = self.__do_mcts_action(mcts)
            if is_over:
                break
play_config = {
    'c_puct': 3,
    'board_len': 9,
    'batch_size': 500,
    'is_use_gpu': True,
    'n_feature_planes': 6,
}
train_model = BetaGobang(**play_config)
train_model.start_game()