import sys
import pygame
import numpy as np

from machines_p1 import P1
from machines_p2 import P2
import time

players = {
    1: P1,
    2: P2
}

pygame.init()

# Colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)  # Highlight color for selected piece

# Proportions & Sizes
WIDTH = 400
HEIGHT = 700  # Increased height to display all 16 pieces
LINE_WIDTH = 5
BOARD_ROWS = 4
BOARD_COLS = 4
SQUARE_SIZE = WIDTH // BOARD_COLS
PIECE_SIZE = SQUARE_SIZE // 2  # Size for the available pieces

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('MBTI Quarto')
screen.fill(BLACK)

# Initialize board and pieces
board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)

# MBTI Pieces (Binary Encoding: I/E = 0/1, N/S = 0/1, T/F = 0/1, P/J = 0/1)
pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
available_pieces = pieces[:]

# Global variable for selected piece
selected_piece = None

# Helper functions
def draw_lines(color=WHITE):
    for i in range(1, BOARD_ROWS):
        pygame.draw.line(screen, color, (0, SQUARE_SIZE * i), (WIDTH, SQUARE_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, color, (SQUARE_SIZE * i, 0), (SQUARE_SIZE * i, WIDTH), LINE_WIDTH)

def draw_pieces():
    font = pygame.font.Font(None, 40)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] != 0:
                piece_idx = board[row][col] - 1
                piece = pieces[piece_idx]
                piece_text = f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"
                text_surface = font.render(piece_text, True, WHITE)
                screen.blit(text_surface, (col * SQUARE_SIZE + 10, row * SQUARE_SIZE + 10))

def draw_available_pieces():
    global selected_piece  # Declare that we are using the global variable
    font = pygame.font.Font(None, 30)
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, WIDTH, WIDTH, HEIGHT - WIDTH))
    
    for idx, piece in enumerate(available_pieces):
        col = idx % 4
        row = idx // 4
        piece_text = f"{'I' if piece[0] == 0 else 'E'}{'N' if piece[1] == 0 else 'S'}{'T' if piece[2] == 0 else 'F'}{'P' if piece[3] == 0 else 'J'}"
        if selected_piece == piece:
            text_surface = font.render(piece_text, True, YELLOW)
        else:
            text_surface = font.render(piece_text, True, BLUE)
        x_pos = col * SQUARE_SIZE + 10
        y_pos = WIDTH + (row * PIECE_SIZE) + 10
        screen.blit(text_surface, (x_pos, y_pos))

def available_square(row, col):
    return board[row][col] == 0

def is_board_full():
    return not any(0 in row for row in board)

def check_line(line):
    if 0 in line:
        return False  # Incomplete line
    characteristics = np.array([pieces[piece_idx - 1] for piece_idx in line])
    for i in range(4):  # Check each characteristic (I/E, N/S, T/F, P/J)
        if len(set(characteristics[:, i])) == 1:  # All share the same characteristic
            return True
    return False

def check_win():
    for col in range(BOARD_COLS):
        if check_line([board[row][col] for row in range(BOARD_ROWS)]):
            return True
    for row in range(BOARD_ROWS):
        if check_line([board[row][col] for col in range(BOARD_COLS)]):
            return True
    if check_line([board[i][i] for i in range(BOARD_ROWS)]) or check_line([board[i][BOARD_ROWS - i - 1] for i in range(BOARD_ROWS)]):
        return True
    return False

def auto_play(num_games=10):
    """자동으로 num_games만큼 게임 실행"""
    global board, available_pieces, selected_piece, turn, flag, game_over

    results = {1: 0, 2: 0, "draw": 0}  # 승리/무승부 결과 저장

    for game in range(num_games):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        available_pieces = pieces[:]
        selected_piece = None
        turn = 1
        flag = "select_piece"
        game_over = False

        # P1 초기화 (Q-테이블 경로 설정)
        player1 = P1(board=board, available_pieces=available_pieces, q_table_file="./data/q1_table.npy")
        player2 = P2(board=board, available_pieces=available_pieces, q_table_file="./data/q2_table.npy")

        while not game_over:
            player = player1 if turn == 1 else player2

            if flag == "select_piece":
                selected_piece = player.select_piece()
                flag = "place_piece"

            elif flag == "place_piece":
                board_row, board_col = player.place_piece(selected_piece)
                if available_square(board_row, board_col):
                    board[board_row][board_col] = pieces.index(selected_piece) + 1
                    available_pieces.remove(selected_piece)
                    selected_piece = None

                    if check_win():
                        game_over = True
                        results[turn] += 1
                    elif is_board_full():
                        game_over = True
                        results["draw"] += 1
                    else:
                        turn = 3 - turn
                        flag = "select_piece"

        # Q-테이블 저장 (P1일 경우)
        player1.save_q_table()
        player2.save_q_table()

        print(f"게임 {game + 1}/{num_games} 완료")

    print("\n=== 게임 결과 ===")
    print(f"Player 1 승리: {results[1]}회")
    print(f"Player 2 승리: {results[2]}회")
    print(f"무승부: {results['draw']}회")
    return results

if __name__ == "__main__":
    mode = input("모드를 선택하세요 ('auto' 또는 'manual'): ").strip().lower()

    if mode == "auto":
        num_games_to_play = int(input("실행할 게임 수를 입력하세요: ").strip())
        auto_play(num_games=num_games_to_play)
    elif mode == "manual":
        turn = 1
        flag = "select_piece"
        game_over = False
        selected_piece = None

        # P1 초기화 (Q-테이블 경로 설정)
        player1 = P1(board=board, available_pieces=available_pieces, q_table_file="./data/q1_table.npy")
        player2 = P2(board=board, available_pieces=available_pieces, q_table_file="./data/q2_table.npy")

        draw_lines()
        draw_available_pieces()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                player = player1 if turn == 1 else player2

                if event.type == pygame.KEYDOWN and flag == "select_piece" and not game_over:
                    selected_piece = player.select_piece()
                    flag = "place_piece"
                    
                    player.save_q_table() 

                elif event.type == pygame.KEYDOWN and flag == "place_piece" and not game_over:
                    board_row, board_col = player.place_piece(selected_piece)

                    if available_square(board_row, board_col):
                        board[board_row][board_col] = pieces.index(selected_piece) + 1
                        available_pieces.remove(selected_piece)
                        selected_piece = None

                        if check_win():
                            game_over = True
                        elif is_board_full():
                            game_over = True
                        else:
                            turn = 3 - turn
                            flag = "select_piece"

                   
                    player.save_q_table()  

                draw_pieces()
                draw_available_pieces()
                pygame.display.update()
    else:
        print("잘못된 입력입니다. 프로그램을 종료합니다.")