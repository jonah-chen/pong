import numpy as np

from game import game_loop, Paddle, Ball
from ai import build_model

def init_game(model1, model2):
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

    player1 = DataGen(model1, 212.5/table_size[0], 132.5/table_size[1])
    player2 = DataGen(model2, 212.5/table_size[0], 132.5/table_size[1])
    paddles[0].move_getter = player1.player
    paddles[1].move_getter = player2.player

    print(game_loop(paddles, ball, table_size))
    print("bruh")


class DataGen:
    """Generate the game data.
    """
    def __init__(self, model, init_x, init_y):
        self.model = model
        self.ball_x_1 = init_x
        self.ball_x_2 = init_x
        self.ball_y_1 = init_y
        self.ball_y_2 = init_y
        self.reward = 0
        self.game_array = []


    def refresh(self, ball_x, ball_y, paddle_y, other_paddle_y):
        """Refresh the information from the board to update the game array
        """
        d = np.array([ball_x, ball_y, self.ball_x_1, self.ball_y_1, self.ball_x_2, self.ball_y_2, other_paddle_y, paddle_y], dtype=np.float32)

        self.ball_x_2, self.ball_y_2 = self.ball_x_1, self.ball_y_1
        self.ball_x_1, self.ball_y_1 = ball_x, ball_y

        self.game_array.append(d)
        return d


    def get_game_array(self, numpy=True):
        """Return the game array.
        """
        if numpy:
            return np.array(self.game_array[:-1])
        return self.game_array[:-1]


    def player(self, paddle_frect, other_paddle_frect, ball_frect, table_size, verbose=1):
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
        prediction = self.model.predict(inp.reshape(1,8,))[0]
        
        if verbose:
            print(prediction)
            print(f"Paddle coordinates: {paddle_frect.pos[0]}, {paddle_frect.pos[1]}")
            print(f"Other paddle coords: {other_paddle_frect.pos[0]}, {other_paddle_frect.pos[1]}")
            print(f"Ball coordinates: {ball_frect.pos[0]:.1f}, {ball_frect.pos[1]:.1f}")
        
        return np.random.choice(['up', 'down', 0], p=prediction)

if __name__ == "__main__":
    model = build_model(2, 128, 64)
    init_game(model, model)
