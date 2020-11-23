import numpy as np
from time import perf_counter
from game import game_loop, Paddle, Ball
from ai import build_model
from chaser_ai import pong_ai


def self_play(model, games=100, verbose=0):
    x, r = init_game(model, model, verbose=verbose)
    for _ in range(games-1):
        x1, r1 = init_game(model, model, verbose=verbose)
        x, r = np.vstack((x, x1)), np.vstack((r, r1))
        del x1, r1
    return x, r

def self_play_2(model, games=100, verbose=0):
    x, r = init_game_2(model)
    for _ in range(games-1):
        x1, r1 = init_game_2(model)
        x, r = np.vstack((x, x1)), np.vstack((r, r1))
        del x1, r1
    return x, r
    
def init_game(model1, model2, verbose=1):
    """Initializes the game

    Args:
        model1 (tf.keras.models.Model): model that plays as player 1
        model2 (tf.keras.models.Model): model that plays as player 2
    
    Returns:
        x (None, 8,): Game values
        r (None,): Discounted Rewards
    """
    start = perf_counter()

    table_size = (440, 280)
    paddle_size = (10, 70)
    ball_size = (15, 15)
    paddle_speed = 1
    max_angle = 45
    paddle_bounce = 1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 2

    paddles = [Paddle((20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 1),
               Paddle((table_size[0]-20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 0)]
    ball = Ball(table_size, ball_size, paddle_bounce,
                wall_bounce, dust_error, init_speed_mag)

    player1 = DataGen(model1, 212.5/table_size[0], 132.5/table_size[1], verbose=verbose)
    player2 = DataGen(model2, 212.5/table_size[0], 132.5/table_size[1], verbose=verbose)

    paddles[0].move_getter = player1.player
    paddles[1].move_getter = player2.player

    [player1.reward, player2.reward] = [-1 if i == 0 else i for i in game_loop(paddles, ball, table_size)]
    end = perf_counter()
    print(f"Time: {(end-start)*1000:.1f}ms.")

    x1, r1 = player1.get_game_array()
    x2, r2 = player2.get_game_array()
    return np.vstack((x1, x2)), np.vstack((r1, r2))

def init_game_2(model):
    """Initializes the game

    Args:
        model1 (tf.keras.models.Model): model that plays as player 1
        model2 (tf.keras.models.Model): model that plays as player 2
    
    Returns:
        x (None, 8,): Game values
        r (None,): Discounted Rewards
    """
    start = perf_counter()

    table_size = (440, 280)
    paddle_size = (10, 70)
    ball_size = (15, 15)
    paddle_speed = 1
    max_angle = 45
    paddle_bounce = 1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 2

    paddles = [Paddle((20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 1),
               Paddle((table_size[0]-20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 0)]
    ball = Ball(table_size, ball_size, paddle_bounce,
                wall_bounce, dust_error, init_speed_mag)

    player1 = DataGen(model, 212.5/table_size[0], 132.5/table_size[1])

    paddles[0].move_getter = player1.player
    paddles[1].move_getter = pong_ai
    if game_loop(paddles, ball, table_size)[0] == 1:
        player1.reward = 1
    else:
        player1.reward = -1
    
    end = perf_counter()
    print(f"Time: {(end-start)*1000:.1f}ms. Player 1: {player1.reward}.")

    return player1.get_game_array()
    

class DataGen:
    """Generate the game data.
    """
    def __init__(self, model, init_x, init_y, gamma=0.99, verbose=0, train=True):
        self.model = model
        self.ball_x_1 = init_x
        self.ball_x_2 = init_x
        self.ball_y_1 = init_y
        self.ball_y_2 = init_y
        self.reward = 0
        self.game_array = []
        self.gamma = gamma
        self.verbose = verbose
        self.actions = []
        self.train = train


    def refresh(self, ball_x, ball_y, paddle_y, other_paddle_y):
        """Refresh the information from the board to update the game array
        """
        d = np.array([ball_x, ball_y, self.ball_x_1, self.ball_y_1, self.ball_x_2, self.ball_y_2, other_paddle_y, paddle_y], dtype=np.float32)

        self.ball_x_2, self.ball_y_2 = self.ball_x_1, self.ball_y_1
        self.ball_x_1, self.ball_y_1 = ball_x, ball_y

        self.game_array.append(d)
        return d


    def get_game_array(self):
        """Return the game array and the discounted rewards.
        """
        # [gamma^n, gamma^n-1.....gamma^2, gamma^1, gamma^0]
        reward_scaler = self.reward*np.power(self.gamma, np.arange(len(self.game_array)-2, -1, -1))
        rewards = np.zeros(shape=(reward_scaler.shape[0], 2))
        for i in range(len(self.game_array)-1):
            rewards[i, self.actions[i]] = reward_scaler[i]

        return np.array(self.game_array[:-1]), rewards


    def player(self, paddle_frect, other_paddle_frect, ball_frect, table_size):
        """Player that is used in the pong game."""
        paddle_y = paddle_frect.pos[1]+paddle_frect.size[1]/2
        other_paddle_y = other_paddle_frect.pos[1]+other_paddle_frect.size[1]/2
        ball_y = ball_frect.pos[1]+ball_frect.size[1]/2

        if paddle_frect.pos[0] > table_size[0]/2:
            ball_x = table_size[0]-ball_frect.pos[0]+ball_frect.size[0]/2
        else:
            ball_x = ball_frect.pos[0]+ball_frect.size[0]/2
        
        ball_x /= table_size[0]
        ball_y /= table_size[1]
        paddle_y /= table_size[1]
        other_paddle_y /= table_size[1]
        
        inp = self.refresh(ball_x, ball_y, paddle_y, other_paddle_y)
        prediction = self.model(inp.reshape(1,8,), training=False)[0].numpy()
        
        # print(prediction)
        # print(f"Paddle coordinates: {paddle_frect.pos[1]}")
        # print(f"Other paddle coords: {other_paddle_frect.pos[1]}")
        # print(f"Ball coordinates: {ball_frect.pos[0]:.1f}, {ball_frect.pos[1]:.1f}")
        if self.train:
            action = np.random.choice([0,1], p=prediction)
        else:
            action = np.argmax(prediction)
        self.actions.append(action)
        return ['up', 'down'][action]

def evaluate_against_chaser(model, score_to_win=10):
    table_size = (440, 280)
    paddle_size = (10, 70)
    ball_size = (15, 15)
    paddle_speed = 1
    max_angle = 45
    paddle_bounce = 1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 2

    paddles = [Paddle((20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 1),
               Paddle((table_size[0]-20, table_size[1]/2), paddle_size, paddle_speed, max_angle, 0)]
    ball = Ball(table_size, ball_size, paddle_bounce,
                wall_bounce, dust_error, init_speed_mag)

    import chaser_ai
    
    player1 = DataGen(model, 212.5/table_size[0], 132.5/table_size[1], train=False)
    paddles[0].move_getter = player1.player
    paddles[1].move_getter = chaser_ai.pong_ai
    
    print(game_loop(paddles, ball, table_size, score_to_win=score_to_win))

if __name__ == "__main__":
    model = build_model(2, 128, 64)
    x, r = self_play(model, games=5, verbose=0)
    print("bruh")
