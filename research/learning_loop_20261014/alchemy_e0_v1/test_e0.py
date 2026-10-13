"""Behavioral contracts for the observable policy boundary and real simulator."""

import collections
import inspect
import unittest

import numpy as np
from dm_alchemy import symbolic_alchemy as sa
from dm_alchemy import symbolic_alchemy_trackers as trackers
from dm_alchemy.ideal_observer import precomputed_maps
from policies import VisibleBayesGreedy, VisibleRandom, decode

LEVEL = "alchemy/perceptual_mapping_randomized_with_random_bottleneck"


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = precomputed_maps.load_from_level_name(LEVEL)

    def environment(self):
        return sa.get_symbolic_alchemy_level(LEVEL, seed=9001)

    def test_observation_decoder_matches_public_perceived_items(self):
        env = self.environment()
        obs = env.reset().observation
        stones, potions = decode(obs)
        self.assertEqual(
            collections.Counter(stones.values()), collections.Counter(env.perceived_stones())
        )
        self.assertEqual(
            collections.Counter(potions.values()), collections.Counter(env.perceived_potions())
        )
        with self.assertRaises(ValueError):
            decode({**obs, "ground_truth_chemistry": np.zeros(1)})

    def test_visible_posterior_matches_official_tracker_from_same_real_events(self):
        env = self.environment()
        tracker = trackers.BeliefStateTracker(self.maps, env)
        env.add_trackers({"belief_state": tracker})
        timestep = env.reset()
        bot = VisibleBayesGreedy(self.maps, True)
        bot.begin_trial(timestep.observation)
        for action in [2, 3, 4, 5, 6]:
            before = timestep.observation
            timestep = env.step(action)
            bot.observe(before, action, timestep.observation, False)
            left = bot.belief.belief_state.world_state_distribution
            right = tracker.belief_state.belief_state.world_state_distribution
            np.testing.assert_array_equal(left.poss_world_states, right.poss_world_states)
            self.assertEqual(left.potion_map_possible, right.potion_map_possible)
            self.assertEqual(left.stone_map_possible, right.stone_map_possible)

    def test_cross_trial_reset_is_explicit_and_carry_keeps_same_belief(self):
        obs = self.environment().reset().observation
        for carry in [False, True]:
            bot = VisibleBayesGreedy(self.maps, carry)
            bot.begin_trial(obs)
            initial = bot.belief
            bot.begin_trial(obs)
            self.assertEqual(bot.belief is initial, carry)

    def test_boundary_outcome_never_uses_next_trials_stone(self):
        env = self.environment()
        obs = env.reset().observation
        bot = VisibleBayesGreedy(self.maps, True)
        bot.begin_trial(obs)
        before = bot.belief.belief_state.world_state_distribution.poss_world_states.copy()
        bot.observe(obs, 2, obs, True)
        np.testing.assert_array_equal(
            before, bot.belief.belief_state.world_state_distribution.poss_world_states
        )
        self.assertEqual(bot.outcome_updates, 0)
        self.assertEqual(bot.skipped_boundary_potions, 1)

    def test_production_boundary_is_native_20_steps_and_new_instance(self):
        env = self.environment()
        first = env.reset().observation["symbolic_obs"].copy()
        chem = env._chemistry
        for _ in range(20):
            timestep = env.step(0)
        self.assertEqual(env.trial_number, 1)
        self.assertIs(env._chemistry, chem)
        self.assertFalse(np.array_equal(first, timestep.observation["symbolic_obs"]))
        self.assertEqual(env.max_steps_per_trial, 20)
        self.assertEqual(env._num_trials, 10)

    def test_visible_policy_receives_no_environment_or_identifier(self):
        self.assertEqual(
            list(inspect.signature(VisibleBayesGreedy.act).parameters),
            ["self", "observation", "step"],
        )
        self.assertEqual(
            list(inspect.signature(VisibleRandom.act).parameters), ["self", "observation", "step"]
        )
        self.assertNotIn("env", vars(VisibleBayesGreedy(self.maps, True)))
        obs = self.environment().reset().observation
        a, b = VisibleRandom(77), VisibleRandom(77)
        self.assertEqual(a.act(obs, 0), b.act(obs, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
