# CodSoft Internship - Task 2
# Tic-Tac-Toe AI using Minimax Algorithm

import math

# Print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print("|".join(row))
        print("-" * 5)

# Check if moves are left on the board
def is_moves_left(board):
    for row in board:
        if "_" in row:
            return True
    return False

# Evaluate board state
def evaluate(board):
    # Check rows
    for row in board:
        if row.count(row[0]) == 3 and row[0] != "_":
            return 10 if row[0] == "O" else -10

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != "_":
            return 10 if board[0][col] == "O" else -10

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "_":
        return 10 if board[0][0] == "O" else -10

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "_":
        return 10 if board[0][2] == "O" else -10

    return 0

# Minimax algorithm
def minimax(board, depth, is_max):
    score = evaluate(board)

    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if not is_moves_left(board):
        return 0

    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "_":
                    board[i][j] = "O"
                    best = max(best, minimax(board, depth+1, not is_max))
                    board[i][j] = "_"
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "_":
                    board[i][j] = "X"
                    best = min(best, minimax(board, depth+1, not is_max))
                    board[i][j] = "_"
        return best

# Find the best move for AI
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == "_":
                board[i][j] = "O"
                move_val = minimax(board, 0, False)
                board[i][j] = "_"
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    return best_move

# Main Game Loop
def play_game():
    board = [["_"]*3 for _ in range(3)]
    print("Welcome to Tic-Tac-Toe! You are X, AI is O.")
    print_board(board)

    for turn in range(9):
        if turn % 2 == 0:  # Human move
            x, y = map(int, input("Enter your move (row col): ").split())
            if board[x][y] == "_":
                board[x][y] = "X"
            else:
                print("Invalid move! Try again.")
                continue
        else:  # AI move
            print("AI is making a move...")
            move = find_best_move(board)
            board[move[0]][move[1]] = "O"

        print_board(board)

        score = evaluate(board)
        if score == 10:
            print("AI wins!")
            return
        elif score == -10:
            print("You win!")
            return

    print("It's a draw!")

if __name__ == "__main__":
    play_game()
