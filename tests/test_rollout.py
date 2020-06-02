import os
import unittest

from utils.rollout import Controller


class TestRollout(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()

    def test_rollouts(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.controller.render_rollout(horizon=5, path=path, name='trajectory')
        # cleanup
        full_video_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'trajectory.mp4')
        if os.path.exists(full_video_path):
            os.remove(full_video_path)


if __name__ == '__main__':
    unittest.main()
