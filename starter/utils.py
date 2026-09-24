"""Plotting helpers for the starter notebooks.

Nothing here is needed to produce a submission -- baseline.py writes one with numpy alone. This is
for looking at the data, which is the fastest way to understand what the tracking is and is not
telling you.

    pip install -r requirements-starter.txt

matplotlib is imported inside the functions rather than at module scope, so importing this module
costs nothing in an environment that only wants the loaders.
"""

import numpy as np

# Params
# Metres. The ATD metadata says 105 x 68; games.py carries 105.3 x 68.0 measured from the
# corrected tracking. The difference matters only if you are computing distances in metres.
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
HOME_COLOUR, AWAY_COLOUR, BALL_COLOUR = "#d62728", "#1f77b4", "#ffffff"
GRASS, LINE = "#3f8f4f", "#ffffff"


def draw_pitch(ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """A plain football pitch in metres, origin at a corner. Returns the axes.

    Drawn here with matplotlib rather than pulled from a pitch-plotting package, because it is
    thirty lines and it means the notebooks need one less dependency to install and one less
    unmaintained package to age badly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.add_patch(Rectangle((0, 0), length, width, facecolor=GRASS, edgecolor=LINE, zorder=0))
    ax.plot([length / 2, length / 2], [0, width], color=LINE, zorder=1)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, edgecolor=LINE, zorder=1))

    for side in (0, 1):
        # Penalty area 16.5 m deep and 40.3 m wide, six-yard box 5.5 m by 18.3 m.
        for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
            x = 0 if side == 0 else length - depth
            ax.add_patch(Rectangle((x, (width - box_width) / 2), depth, box_width,
                                   fill=False, edgecolor=LINE, zorder=1))

    ax.set_xlim(-3, length + 3)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("metres")
    return ax


def plot_frame(dataset, frame_index, ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """Draw one tracking frame: both teams and the ball.

    The ball is drawn last and outlined so it stays visible in a crowd -- and when it is missing,
    which happens on roughly four frames in ten, the title says so rather than leaving you to
    wonder why there is no white dot.
    """
    ax = draw_pitch(ax, length, width)
    frame = dataset.frames[frame_index]
    home_team = dataset.metadata.teams[0]

    for player, position in frame.players_coordinates.items():
        if position is None:
            continue
        ax.scatter(position.x * length, position.y * width, s=90, zorder=3,
                   color=HOME_COLOUR if player.team == home_team else AWAY_COLOUR,
                   edgecolors="black", linewidths=0.5)

    ball = frame.ball_coordinates
    has_ball = ball is not None and ball.x is not None and not np.isnan(ball.x)
    if has_ball:
        ax.scatter(ball.x * length, ball.y * width, s=70, zorder=4,
                   color=BALL_COLOUR, edgecolors="black", linewidths=1.2)

    ax.set_title("frame {}{}".format(frame_index, "" if has_ball else "   -- ball not tracked"))
    return ax


def animate_frames(dataset, start_frame, end_frame, length=PITCH_LENGTH, width=PITCH_WIDTH):
    """A matplotlib animation of a span of tracking frames.

    In a notebook:

        from IPython.display import HTML
        HTML(animate_frames(dataset, 11185, 11435).to_html5_video())

    Needs ffmpeg for to_html5_video(); to_jshtml() works without it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, ax = plt.subplots(figsize=(10.5, 6.8))
    draw_pitch(ax, length, width)
    home = ax.scatter([], [], s=90, color=HOME_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    away = ax.scatter([], [], s=90, color=AWAY_COLOUR, edgecolors="black", linewidths=0.5, zorder=3)
    ball = ax.scatter([], [], s=70, color=BALL_COLOUR, edgecolors="black", linewidths=1.2, zorder=4)
    title = ax.set_title("")
    plt.close(figure)

    home_team = dataset.metadata.teams[0]

    def positions(frame):
        """(home_xy, away_xy, ball_xy) for one frame, each an (n, 2) array."""
        sides = {True: [], False: []}
        for player, position in frame.players_coordinates.items():
            if position is not None:
                sides[player.team == home_team].append(
                    (position.x * length, position.y * width))
        point = frame.ball_coordinates
        visible = point is not None and point.x is not None and not np.isnan(point.x)
        return (np.array(sides[True]).reshape(-1, 2), np.array(sides[False]).reshape(-1, 2),
                np.array([[point.x * length, point.y * width]]) if visible
                else np.empty((0, 2)))

    def update(step):
        index = start_frame + step
        home_xy, away_xy, ball_xy = positions(dataset.frames[index])
        home.set_offsets(home_xy)
        away.set_offsets(away_xy)
        # set_offsets on an empty array is how the ball simply disappears on frames where the
        # tracking lost it, which is the honest way to draw a missing observation.
        ball.set_offsets(ball_xy)
        title.set_text("frame {}{}".format(index, "" if len(ball_xy) else "   -- no ball"))
        return home, away, ball, title

    return FuncAnimation(figure, update, frames=end_frame - start_frame, interval=40, blit=False)
