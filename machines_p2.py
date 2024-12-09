#Q-learning으로 구현
import numpy as np
import random
from itertools import product
from collections import defaultdict

import os

class P2():
    def __init__(self, board, available_pieces, epsilon=0.05, num_episodes=10000, alpha=0.1, gamma=0.9, q_table_file='q2_table.npy'):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.q_table_file = q_table_file
        self.Q = self.load_q_table()

    def save_q_table(self):
        """Q-테이블 저장"""
        try:
            # 경로가 없으면 생성
            directory = os.path.dirname(self.q_table_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            # Q-테이블 저장
            np.save(self.q_table_file, dict(self.Q))
            #print(f"Q-테이블이 {os.path.abspath(self.q_table_file)}에 저장되었습니다.")
        except Exception as e:
            print(f"Q-테이블 저장 중 오류 발생: {e}")

    def load_q_table(self):
        """Q-테이블 불러오기"""
        try:
            if os.path.exists(self.q_table_file):
                q_table_data = np.load(self.q_table_file, allow_pickle=True).item()
                #print(f"[DEBUG] Q-테이블 로드 완료: {self.q_table_file}")
                return defaultdict(float, q_table_data)
            else:
                #print(f"[DEBUG] 저장된 Q-테이블이 없습니다. 새로운 테이블을 생성하고 저장합니다.")
                q_table = defaultdict(float)
                self.Q = q_table
                self.save_q_table()  # 즉시 저장
                return q_table
        except Exception as e:
            #print(f"[ERROR] Q-테이블 로드 중 오류 발생: {e}")
            return defaultdict(float)

    def update_q_table_and_save(self, state, action, reward, next_state, done):
        """Q-테이블 업데이트 및 저장"""
        self.update_q_value(state, action, reward, next_state, done)
        self.save_q_table()

    def select_piece(self):
        """상대방(P1)이 놓을 피스를 선택"""
        if self.is_board_empty():
            selected_piece = random.choice(self.available_pieces)
            #print(f"P2: 초기 상태에서 랜덤으로 선택한 피스: {selected_piece}")
            return selected_piece

        state = tuple(self.available_pieces)
        
        # 시뮬레이션을 통해 Q 값 업데이트
        for piece in self.available_pieces:
            self.simulate_episodes_for_piece(state, piece)

        # 최적 Q 값에 따라 행동 선택
        max_value = -float('inf')
        selected_piece = None
        for piece in self.available_pieces:
            q_value = self.Q[(state, piece)]
            #print(f"P2: 피스 {piece}의 Q 값: {q_value}")
            if q_value > max_value:
                max_value = q_value
                selected_piece = piece
        if selected_piece is None:
            selected_piece = random.choice(self.available_pieces)
        #print(f"P2: 선택한 피스: {selected_piece}")
        return selected_piece

    def place_piece(self, selected_piece):
        """상대가 골라준 피스를 보드에 놓기"""
        if self.is_board_empty():
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
            chosen_location = random.choice(available_locs)
            #print(f"P2: 초기 상태에서 랜덤으로 선택한 위치: {chosen_location}")
            return chosen_location

        state = (tuple(self.board.flatten()), selected_piece)
        
        # 시뮬레이션을 통해 Q 값 업데이트
        for loc in [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]:
            self.simulate_episodes_for_location(state, selected_piece, loc)

        # 최적 Q 값에 따라 행동 선택
        max_value = -float('inf')
        chosen_location = None
        for loc in [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]:
            q_value = self.Q[(state, loc)]
            #print(f"P2: 위치 {loc}의 Q 값: {q_value}")
            if q_value > max_value:
                max_value = q_value
                chosen_location = loc
        if chosen_location is None:
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
            chosen_location = random.choice(available_locs)
        #print(f"P2: 선택한 위치: {chosen_location}")
        return chosen_location

    def simulate_episodes_for_piece(self, state, piece):
        """피스 선택에 대한 시뮬레이션 수행 및 Q 값 업데이트"""
        total_reward = 0
        for _ in range(self.num_episodes):
            reward = self.play_full_episode(state, piece, is_piece_selection=True)
            total_reward += reward
        avg_reward = total_reward / self.num_episodes
        self.update_q_value(state, piece, avg_reward, state, done=False)

    def simulate_episodes_for_location(self, state, piece, loc):
        """피스를 특정 위치에 놓는 경우에 대한 시뮬레이션 수행 및 Q 값 업데이트"""
        total_reward = 0
        for _ in range(self.num_episodes):
            reward = self.play_full_episode(state, piece, initial_location=loc, is_piece_selection=False)
            total_reward += reward
        avg_reward = total_reward / self.num_episodes
        self.update_q_value(state, loc, avg_reward, state, done=False)

    def update_q_value(self, state, action, reward, next_state, done):
        """Q 테이블 업데이트"""
        current_q = self.Q[(state, action)]
        max_next_q = max([self.Q[(next_state, a)] for a in self.available_pieces]) if not done else 0
        self.Q[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def play_full_episode(self, state, piece, initial_location=None, is_piece_selection=True):
        board_copy = np.copy(self.board)
        available_pieces_copy = self.available_pieces[:]
        
        if is_piece_selection:
            current_piece = piece
        else:
            board_copy[initial_location[0], initial_location[1]] = self.pieces.index(piece) + 1
            available_pieces_copy.remove(piece)
            if self.check_win(board_copy):
                return 1  # 즉시 승리 시 보상
            current_piece = random.choice(available_pieces_copy) if available_pieces_copy else None

        player_turn = False
        while not self.is_board_full(board_copy) and current_piece is not None:
            if random.random() < self.epsilon:
                available_locs = [(row, col) for row, col in product(range(4), range(4)) if board_copy[row][col] == 0]
                location = random.choice(available_locs)
                #print(f"P2 시뮬레이션 탐험: 위치 {location}")
            else:
                available_locs = [(row, col) for row, col in product(range(4), range(4)) if board_copy[row][col] == 0]
                location = random.choice(available_locs)
                #print(f"P2 시뮬레이션 탐사: 위치 {location}")

            board_copy[location[0], location[1]] = self.pieces.index(current_piece) + 1
            if self.check_win(board_copy):
                return 1 if player_turn else -1  # 승리 또는 패배 보상

            available_pieces_copy.remove(current_piece)
            current_piece = random.choice(available_pieces_copy) if available_pieces_copy else None
            player_turn = not player_turn

        return 0.5  # 무승부 시 보상

    def check_win(self, board):
        def check_line(line):
            if 0 in line:
                return False
            characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
            for i in range(4):
                if len(set(characteristics[:, i])) == 1:
                    return True
            return False

        def check_2x2_subgrid():
            for r in range(3):
                for c in range(3):
                    subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]
                    if 0 not in subgrid:
                        characteristics = [self.pieces[idx - 1] for idx in subgrid]
                        for i in range(4):
                            if len(set(char[i] for char in characteristics)) == 1:
                                return True
            return False

        for col in range(4):
            if check_line([board[row][col] for row in range(4)]):
                return True

        for row in range(4):
            if check_line([board[row][col] for col in range(4)]):
                return True

        if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3 - i] for i in range(4)]):
            return True

        return check_2x2_subgrid()

    def is_board_full(self, board):
        return not any(cell == 0 for row in board for cell in row)

    def is_board_empty(self):
        return all(cell == 0 for row in self.board for cell in row)
