import numpy as np
import random
import json
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

class ConnectFourAI:
    def __init__(self, difficulty='medium'):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1  
        self.ai_player = 2 
        self.difficulty = difficulty

    def is_valid_move(self, column):
        """Check if the selected column has an open space"""
        return self.board[5, column] == 0

    def get_valid_locations(self):
        """Return list of valid column locations"""
        return [col for col in range(7) if self.is_valid_move(col)]

    def drop_piece(self, board, column, player):
        """Drop a piece in the specified column on a given board"""
        for row in range(6):
            if board[row, column] == 0:
                board[row, column] = player
                return board
        return board

    def check_winner(self, board, player):
        """Check for a winning condition"""
        for c in range(4):
            for r in range(6):
                if (board[r, c] == player and 
                    board[r, c+1] == player and 
                    board[r, c+2] == player and 
                    board[r, c+3] == player):
                    return True
        for c in range(7):
            for r in range(3):
                if (board[r, c] == player and 
                    board[r+1, c] == player and 
                    board[r+2, c] == player and 
                    board[r+3, c] == player):
                    return True

        for c in range(4):
            for r in range(3):
                if (board[r, c] == player and 
                    board[r+1, c+1] == player and 
                    board[r+2, c+2] == player and 
                    board[r+3, c+3] == player):
                    return True

        for c in range(4):
            for r in range(3, 6):
                if (board[r, c] == player and 
                    board[r-1, c+1] == player and 
                    board[r-2, c+2] == player and 
                    board[r-3, c+3] == player):
                    return True

        return False

    def is_board_full(self, board):
        """Check if the board is completely full"""
        return np.all(board != 0)

    def evaluate_window(self, window, player):
        """Evaluate the score of a window of 4 pieces"""
        score = 0
        opponent = 1 if player == 2 else 2

        if window.count(player) == 4:
            score += 100
        elif window.count(player) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(player) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def score_position(self, board, player):
        """Score the entire board position"""
        score = 0

        center_array = [int(i) for i in list(board[:, 3])]
        center_count = center_array.count(player)
        score += center_count * 3

        for r in range(6):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(4):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, player)

        for c in range(7):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, player)

        for r in range(3):
            for c in range(4):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        for r in range(3, 6):
            for c in range(4):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        valid_locations = [col for col in range(7) if board[5, col] == 0]
        
        # Terminal conditions
        if depth == 0 or len(valid_locations) == 0:
            return (None, self.score_position(board, self.ai_player))
        
        if self.check_winner(board, 1):
            return (None, float('-inf'))
        
        if self.check_winner(board, self.ai_player):
            return (None, float('inf'))
        
        if self.is_board_full(board):
            return (None, 0)

        if maximizing_player:
            value = float('-inf')
            column = random.choice(valid_locations)
            
            for col in valid_locations:
                board_copy = board.copy()
                temp_board = self.drop_piece(board_copy, col, self.ai_player)
                
                new_score = self.minimax(temp_board, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    column = col

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            
            return column, value

        else:
            value = float('inf')
            column = random.choice(valid_locations)
            
            for col in valid_locations:
                board_copy = board.copy()
                temp_board = self.drop_piece(board_copy, col, 1)
                
                new_score = self.minimax(temp_board, depth-1, alpha, beta, True)[1]
                
                if new_score < value:
                    value = new_score
                    column = col
   
                beta = min(beta, value)
                if alpha >= beta:
                    break
            
            return column, value

    def get_ai_move(self):
        """Determine AI's move based on difficulty"""
        if self.difficulty == 'easy':
            valid_locations = self.get_valid_locations()
            return random.choice(valid_locations)
        
        elif self.difficulty == 'medium':
            column, _ = self.minimax(self.board.copy(), 4, float('-inf'), float('inf'), True)
            return column
        
        else:  
            column, _ = self.minimax(self.board.copy(), 6, float('-inf'), float('inf'), True)
            return column

    def make_move(self, column, player):
        """Make a move and return the updated board"""
        if self.is_valid_move(column):
            self.board = self.drop_piece(self.board, column, player)
            return True
        return False

    def board_to_list(self):
        """Convert numpy board to list for JSON serialization"""
        return self.board.tolist()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a9e98bf7e2f641fda1d2e91d9c1f2a1e'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/')
def index():
    """Render the main game page"""
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new game with selected difficulty"""
    difficulty = request.json.get('difficulty', 'medium')
    game = ConnectFourAI(difficulty)
    session['game'] = {
        'board': game.board_to_list(),
        'difficulty': difficulty
    }
    return jsonify({
        'board': game.board_to_list(),
        'difficulty': difficulty
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    """Handle player's move and AI's response"""
    game_data = session.get('game')
    if not game_data:
        return jsonify({'error': 'No active game'}), 400

    game = ConnectFourAI(game_data['difficulty'])
    game.board = np.array(game_data['board'])

    column = request.json.get('column')

    if not game.is_valid_move(column):
        return jsonify({'error': 'Invalid move'}), 400

    game.make_move(column, 1)

    if game.check_winner(game.board, 1):
        session.pop('game', None)
        return jsonify({
            'board': game.board_to_list(),
            'status': 'player_win'
        })

    if game.is_board_full(game.board):
        session.pop('game', None)
        return jsonify({
            'board': game.board_to_list(),
            'status': 'draw'
        })

    ai_column = game.get_ai_move()
    game.make_move(ai_column, 2)

    if game.check_winner(game.board, 2):
        session.pop('game', None)
        return jsonify({
            'board': game.board_to_list(),
            'status': 'ai_win',
            'ai_column': ai_column
        })

    if game.is_board_full(game.board):
        session.pop('game', None)
        return jsonify({
            'board': game.board_to_list(),
            'status': 'draw'
        })

    session['game'] = {
        'board': game.board_to_list(),
        'difficulty': game.difficulty
    }

    return jsonify({
        'board': game.board_to_list(),
        'ai_column': ai_column,
        'status': 'continue'
    })

if __name__ == '__main__':
    app.run(debug=True)
