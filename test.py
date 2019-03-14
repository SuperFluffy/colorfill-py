#!/usr/bin/env python

import colorfill
import numpy as np
import unittest

class TestColorfillGame(unittest.TestCase):
    def setUp(self):
        self.board = np.array([
            [1,1,2,3],
            [2,3,4,1],
            [2,3,3,1],
            [2,3,1,1]
        ])

        self.game = colorfill.Game(self.board)

    def test_greedy_onestep(self):
        game_gen = self.game.run_strategy(colorfill.greedy) 

        board_after = np.array([
            [2,2,2,3],
            [2,3,4,1],
            [2,3,3,1],
            [2,3,1,1]
        ])
        _, g = next(game_gen)
        self.assertTrue(np.all(board_after == g.board))

    def test_greedy_twostep(self):
        game_gen = self.game.run_strategy(colorfill.greedy) 

        board_after = np.array([
            [3,3,3,3],
            [3,3,4,1],
            [3,3,3,1],
            [3,3,1,1]
        ])
        _ = next(game_gen)
        _, g = next(game_gen)
        self.assertTrue(np.all(board_after == g.board))

    def test_greedy_chooses_lower_color(self):
        board = np.array([
            [1,1,2,3],
            [2,1,4,1],
            [2,4,3,1],
            [2,3,3,1]
        ])

        c, _ = next(self.game.run_strategy(colorfill.greedy))

        self.assertEqual(c, 2)


    def test_greedy_correct_sequence(self):
        steps = []
        for c, _ in self.game.run_strategy(colorfill.greedy):
            steps.append(c)
        correct_sequence = [2, 3, 1, 4]

        self.assertTrue(c, correct_sequence)

    def test_greedy_fully_covered(self):
        for _ in self.game.run_strategy(colorfill.greedy):
            continue

        self.assertTrue(self.game.is_fully_flooded())

    def test_smart_fully_covered(self):
        for _ in self.game.run_strategy(colorfill.smart):
            continue

        self.assertTrue(self.game.is_fully_flooded())

if __name__ == "__main__":
    unittest.main()
