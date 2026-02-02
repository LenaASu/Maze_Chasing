'''
Agents get game_state from web via websockets. 
We can use information in game_state to create agents.

Rerir: the name of my pacman.
Enemies: ghosts.

According to script.js (getCurrentGameState()), 
game_state contains:
    1. pacman_pos: {r: pacmanCurrentRow, c: pacmanCurrentCol, direction: currentDirection},
    2. is_werewolf_mode: isWerewolfMode,

    3. ghosts: ghostsData,
    const ghostsData = [];
    for (const name in CHARACTERS) {
        const char = CHARACTERS[name];
        ghostsData.push({
            name: name,
            r: char.r,
            c: char.c,
            direction: char.direction,
            state: char.state // States: "escape", "chase", "frightened"
        });
    }

    4. map_state: currentMazeState, 
    WALL = 0; // Rerir cannot move on this tile
    PATH = 1; // Rerir can move on this tile
    PELLET = 2; // Rerir can eat them to earn scores
    HEART_FRAGMENT = 3; // Rerir eats them and will werewolf in 8 seconds
    // During Rerir's werewolf state Rerir can eat enemies. Enemies are in "frighten" state and run away from Rerir.
    GHOST_CAGE = 4; // Enemies' staring points
    PACMAN_START = 5; // Rerir's starting point

    5. score: score, 
    6. lives: lives
    
'''

import asyncio
import websockets
import json
import random
import math
import time

# Global variables
moves = ['up', 'down', 'left', 'right']
DIRECTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Helper functions
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_moves(state):
    pacman = state['pacman_pos']
    curr_r, curr_c = pacman['r'], pacman['c']
    
    map_state = state['map_state']
    legal_moves = []
    
    for move, (dr, dc) in DIRECTIONS.items():
        new_r = curr_r + dr
        new_c = curr_c + dc
        
        if (0 <= new_r < len(map_state) and 
            0 <= new_c < len(map_state[0]) and 
            map_state[new_r][new_c] != 0):
            legal_moves.append(move)
    
    return legal_moves

def apply_move(state, move):
    pacman = state['pacman_pos']
    curr_r, curr_c = pacman['r'], pacman['c']
    
    dr, dc = DIRECTIONS[move]
    new_r = curr_r + dr
    new_c = curr_c + dc
    
    new_state = {
        'pacman_pos': {
            'r': new_r,
            'c': new_c,
            'direction': move
        },
        'is_werewolf_mode': state.get('is_werewolf_mode', False),
        'ghosts': state.get('ghosts', []),
        'map_state': state['map_state'],
        'score': state.get('score', 0),
        'lives': state.get('lives', 3)
    }
    
    return new_state

# Count the number of pellets and heart fragments in the map state
# They are used to evaluate the reward of the final state and check if the game ends
def count_pellets(map_state):
    count = 0
    for row in map_state:
        for cell in row:
            if cell == 2:
                count += 1
    return count

def count_heart_fragments(map_state):
    count = 0
    for row in map_state:
        for cell in row:
            if cell == 3:
                count += 1
    return count

def evaluate_ghost_threat(pac_pos, ghosts, werewolf_mode=False):
    """
    Evaluate the maximum threat from enemy to Rerir.
    If werewolf_mode is True, frightened ghosts are considered eatable for Rerir.
    """
    if not ghosts:
        return 0.0
    
    pac_r, pac_c = pac_pos
    max_threat = 0.0
    
    for ghost in ghosts:
        ghost_r, ghost_c = ghost['r'], ghost['c']
        distance = manhattan_distance(pac_pos, (ghost_r, ghost_c))
        
        if distance == 0:
            return 1.0
        
        ghost_state = ghost.get('state', 'chase')
        
        # If enemy is in frightened mode, it can be eaten by Rerir
        if ghost_state == 'frightened':
            threat = max(0, 0.3 - distance * 0.1) 
             
        # If enemy is in chase mode, it will chase Rerir
        else:  
            threat = min(0.5, 0.4 / (distance + 0.1))
        
        # If Rerir is in the same row or column as a enemy, increase threat
        if pac_r == ghost_r or pac_c == ghost_c:
            threat *= 1.2
        
        max_threat = max(max_threat, threat)
    
    return min(1.0, max_threat)

class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.5):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = get_moves(state)
        self._is_terminal = self.check_terminal(state)
    
    def check_terminal(self, state):
        # Lose: lives <= 0 
        if state.get('lives', 3) <= 0:
            return True
        
        # Win: all pellets and heart fragments are eaten
        if (count_pellets(state['map_state']) == 0 and 
            count_heart_fragments(state['map_state']) == 0):
            return True
        return False
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        return self._is_terminal
    
    @property
    def value(self):
        if self.visits == 0:
            return 0
        return self.total_reward / self.visits

class MTCSAgent:
    def __init__(self, exploration_weight = 1.5, time_limit = 0.08, 
                 max_simulation_depth = 25, discount_factor = 0.95):
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.max_simulation_depth = max_simulation_depth
        self.discount_factor = discount_factor
        
        # Record the position history for exploration bonus calculation
        self.position_history = set()
    
    def search(self, game_state):
        root = MCTSNode(game_state)
        start_time = time.perf_counter()
        iterations = 0
        
        # Record the initial position of Rerir for exploration bonus calculation
        pac_pos = (game_state['pacman_pos']['r'], game_state['pacman_pos']['c'])
        self.position_history.add(pac_pos)
        
        while time.perf_counter() - start_time < self.time_limit:
            node = self.selection(root)
            
            if not node.is_terminal():
                if not node.is_fully_expanded():
                    node = self.expansion(node)
                
                reward = self.simulation(node.state)
                self.backpropagation(node, reward)
            
            iterations += 1

        best_move = self.get_best_move(root)
        
        return best_move if best_move else random.choice(get_moves(game_state))
    
    def selection(self, node):
        current = node
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            current = self.best_child(current)
        return current
    
    def best_child(self, node):
        """Use PUCT formula to select the best child node"""
        best_score = -float('inf')
        best_child = None
        
        total_visits = node.visits
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 1.0
        
        for move, child in node.children.items():
            if child.visits == 0:
                score = child.prior * sqrt_total
            else:
                exploit = child.value
                explore = (self.exploration_weight * child.prior * 
                          sqrt_total / (1 + child.visits))
                score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child else random.choice(list(node.children.values()))
    
    def expansion(self, node):
        if not node.untried_moves:
            return node
        
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        new_state = apply_move(node.state, move)
        prior = self.cal_prior_prob(node.state, move)
        
        child_node = MCTSNode(new_state, parent=node, move=move, prior=prior)
        node.children[move] = child_node
        
        return child_node
    
    def cal_prior_prob(self, state, move):
        """Calculate the prior probability of a move"""
        pacman = state['pacman_pos']
        curr_r, curr_c = pacman['r'], pacman['c']
        dr, dc = DIRECTIONS[move]
        new_pos = (curr_r + dr, curr_c + dc)
        
        score = 0.0
        max_possible_score = 0.0
        
        # 1. Eat items（40%）
        tile_value = state['map_state'][new_pos[0]][new_pos[1]]
        if tile_value == 2:  # pellet
            score += 0.4
            max_possible_score += 0.4
        elif tile_value == 3:  # heart-fragment
            score += 0.6
            max_possible_score += 0.6
        
        # 2. Enemy threat（40%）
        threat = evaluate_ghost_threat(
            new_pos, 
            state.get('ghosts', [])
        )
        safety = 1.0 - threat
        score += safety * 0.4
        max_possible_score += 0.4
        
        # 3. Exploration bonus（20%）
        exploration_value = self.get_exploration_value(state, new_pos)
        score += exploration_value * 0.2
        max_possible_score += 0.2
        
        # Normalize the score
        if max_possible_score > 0:
            normalized = 0.05 + 0.9 * (score / max_possible_score)
        else:
            normalized = 0.5
        
        return max(0.05, min(0.95, normalized))
    
    def get_exploration_value(self, state, position):
        r, c = position
        
        # If the position has been visited before, give a lower exploration value
        if position in self.position_history:
            return 0.2
        
        # If the position is close to items (pellet/heart-fragment), give a higher exploration value
        map_state = state['map_state']
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if (0 <= nr < len(map_state) and 
                    0 <= nc < len(map_state[0]) and 
                    map_state[nr][nc] in [2, 3]):
                    distance = abs(dr) + abs(dc)
                    if distance <= 2:
                        return 0.8
        
        return 0.5
    
    def simulation(self, state):
        current_state = state
        depth = 0
        total_reward = 0.0
        discount = self.discount_factor
        
        while (depth < self.max_simulation_depth and 
               not self.is_terminal_state(current_state)):
            
            legal_moves = get_moves(current_state)
            if not legal_moves:
                break
            
            move = self.select_move_for_simulation(current_state, legal_moves)
            current_state = apply_move(current_state, move)
            
            step_reward = self.immediate_reward(current_state, move)
            total_reward += step_reward * (discount ** depth)
            
            depth += 1
        
        final_reward = self.final_state_reward(current_state)
        total_reward += final_reward * (discount ** depth)
        
        return total_reward
    
    def select_move_for_simulation(self, state, legal_moves):
        """Use heuristic to select the best move for simulation"""
        pacman = state['pacman_pos']
        curr_r, curr_c = pacman['r'], pacman['c']
        
        best_move = None
        best_score = -float('inf')
        
        for move in legal_moves:
            score = 0
            dr, dc = DIRECTIONS[move]
            new_pos = (curr_r + dr, curr_c + dc)
            
            # Eat items
            tile_value = state['map_state'][new_pos[0]][new_pos[1]]
            if tile_value == 2:
                score += 50
            elif tile_value == 3:
                score += 100
            
            # Enemy threat
            for ghost in state.get('ghosts', []):
                ghost_pos = (ghost['r'], ghost['c'])
                distance = manhattan_distance(new_pos, ghost_pos)
                if distance == 0 and ghost.get('state', 'chase') != 'frightened':
                    score -= 1000
                elif distance < 5:
                    score -= 100 / (distance + 0.1)
            
            # Avoid oscillation
            if 'direction' in pacman:
                old_dir = pacman['direction']
                if ((old_dir == 'left' and move == 'right') or
                    (old_dir == 'right' and move == 'left') or
                    (old_dir == 'up' and move == 'down') or
                    (old_dir == 'down' and move == 'up')):
                    score -= 30
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def is_terminal_state(self, state):
        """Check if the current state is a terminal state"""
        if state.get('lives', 3) <= 0:
            return True
        
        pac_pos = (state['pacman_pos']['r'], state['pacman_pos']['c'])
        for ghost in state.get('ghosts', []):
            ghost_pos = (ghost['r'], ghost['c'])
            if (pac_pos == ghost_pos and 
                ghost.get('state', 'chase') != 'frightened'):
                return True
        
        return False
    
    def immediate_reward(self, state, move):
        reward = 0
        
        pacman = state['pacman_pos']
        curr_r, curr_c = pacman['r'], pacman['c']
        
        # Eat items
        tile_value = state['map_state'][curr_r][curr_c]
        if tile_value == 2:
            reward += 100
        elif tile_value == 3:
            reward += 200
        
        # Eat enemy or penalize being eaten/get too close
        pac_pos = (curr_r, curr_c)
        for ghost in state.get('ghosts', []):
            ghost_pos = (ghost['r'], ghost['c'])
            distance = manhattan_distance(pac_pos, ghost_pos)
            
            if distance == 0:
                if ghost.get('state', 'chase') == 'frightened':
                    reward += 300  # Eat enemy
                else:
                    reward -= 500  # Eaten by enemy
            elif distance < 5:
                reward -= 150 / (distance + 0.1)
        
        # Avoid oscillation
        if 'direction' in pacman:
            old_dir = pacman['direction']
            if ((old_dir == 'left' and move == 'right') or
                (old_dir == 'right' and move == 'left') or
                (old_dir == 'up' and move == 'down') or
                (old_dir == 'down' and move == 'up')):
                reward -= 30
        
        return reward
    
    def final_state_reward(self, state):
        '''Calculate the reward by the end of the state'''
        reward = 0
        reward += state.get('lives', 0) * 500
        reward += state.get('score', 0) * 0.5
        
        pellets_left = count_pellets(state['map_state'])
        fragments_left = count_heart_fragments(state['map_state'])
        reward -= (pellets_left * 10 + fragments_left * 20)
        
        return reward
    
    def backpropagation(self, node, reward):
        current = node
        discount = self.discount_factor
        
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            reward *= discount
            current = current.parent
    
    def get_best_move(self, root):
        if not root.children:
            return None
        
        best_move = max(root.children.items(), 
                       key=lambda x: x[1].visits)[0]
        
        return best_move

def mcts_navi(game_state):
    try:
        agent = MTCSAgent(
            exploration_weight = 1.5,
            time_limit = 0.08,
            max_simulation_depth = 20,
            discount_factor = 0.95
        )
        
        best_move = agent.search(game_state)
        print(f"[MCTS] Selected move: {best_move}")
        return best_move
        
    except Exception as e:
        print(f"[MCTS Error] {e}")
        import traceback
        traceback.print_exc()
        
        legal_moves = get_moves(game_state)
        return random.choice(legal_moves) if legal_moves else 'up'


def a_star_navi(game_state):
    print("[A* Algorithm] Currently returning random move")
    return random.choice(moves)

def dqn_navi(game_state):
    print("[DQN Algorithm] Currently returning random move")
    return random.choice(moves)

def ai_navi(game_state, algorithm):
    if algorithm == 'mcts':
        return mcts_navi(game_state)
    elif algorithm == 'a_star':
        return a_star_navi(game_state)
    elif algorithm == 'dqn':
        return dqn_navi(game_state)
    else:
        print(f"Unknown algorithm: {algorithm}. Using random move.")
        return random.choice(moves)


async def web_connection(websocket):
    global current_algorithm
    valid_algorithms = ['a_star', 'mcts', 'dqn']
    current_algorithm = "mcts"
    
    try:
        print("New game client connected. Waiting for messages...")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Received message, algorithm: {current_algorithm}")
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error: {e}")
                continue

            if isinstance(data, dict) and data.get('command') == 'set_algorithm':
                new_algorithm = data.get('name')
                if new_algorithm in valid_algorithms:
                    current_algorithm = new_algorithm
                    print(f"Algorithm set to: {current_algorithm.upper()}")
                continue

            if isinstance(data, dict) and 'pacman_pos' in data:
                print(f"Pacman position: {data['pacman_pos']}")
                print(f"Processing game state with {current_algorithm}")
                move = ai_navi(data, current_algorithm)
                print(f"Sending move: {move}")
                await websocket.send(move)
            else:
                print(f"Unknown message format")

    except websockets.ConnectionClosed:
        print("Game client connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

async def main():
    start_server = websockets.serve(
        web_connection, 
        "localhost",
        8765
    )
    print("AI WebSocket server started, listening on ws://localhost:8765...")
    
    async with start_server:
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")