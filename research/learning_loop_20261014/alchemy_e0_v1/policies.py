"""Visible-only E0 policies; privileged reference is a separate driver function."""

import collections
import copy

import numpy as np
from dm_alchemy import symbolic_alchemy as sa
from dm_alchemy.ideal_observer import ideal_observer as io
from dm_alchemy.types import stones_and_potions as sp


def decode(observation):
    if set(observation) != {"symbolic_obs"}:
        raise ValueError("Only the production symbolic observation is allowed")
    raw = np.asarray(observation["symbolic_obs"])
    if raw.shape != (39,) or raw.dtype != np.float32:
        raise ValueError("Unexpected production observation shape/dtype")
    stones = {}
    potions = {}
    for i, row in enumerate(raw[:15].reshape(3, 5)):
        if row[4] == 0:
            stones[i] = sp.PerceivedStone(int(round(float(row[3]) * 3)), row[:3].copy())
    for i, row in enumerate(raw[15:].reshape(12, 2)):
        if row[1] == 0:
            index = int(round((float(row[0]) + 1) * 3))
            if not 0 <= index < 6:
                raise ValueError("Invalid visible potion index")
            potions[i] = sp.perceived_potion_from_index(index)
    return stones, potions


def integer_action(stone, potion):
    return sa.tuple_action_to_int((stone, potion), slot_based=True, end_trial_action=False)


def cash_value(reward):
    return 15.0 if reward == 3 else max(0.0, float(reward))


def settle_action(stones, potions, step):
    """Common visible settlement, reserving actual actions for positive stones."""
    maximum = sorted(i for i, s in stones.items() if s.reward == 3)
    if maximum:
        return integer_action(maximum[0], -1)
    positive = sorted(i for i, s in stones.items() if s.reward > 0)
    if positive and (20 - step <= len(positive) or not potions):
        return integer_action(positive[0], -1)
    return None


class VisibleRandom:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def begin_trial(self, observation):
        decode(observation)

    def act(self, observation, step):
        stones, potions = decode(observation)
        settle = settle_action(stones, potions, step)
        if settle is not None:
            return settle
        if not stones or not potions:
            return 0
        return integer_action(
            int(self.rng.choice(sorted(stones))), int(self.rng.choice(sorted(potions)))
        )

    def observe(self, before, action, after, changed_trial):
        decode(before)
        decode(after)


class VisibleBayesGreedy:
    """Official Bayesian filter with authored one-step expected cash-value score.

    This is not the official full-horizon IdealObserverBot or a trained learner.
    No environment, tracker, hidden chemistry or episode identifier is received.
    """

    def __init__(self, precomputed, carry):
        self.precomputed = precomputed
        self.carry = carry
        self.belief = None
        self.outcome_updates = 0
        self.skipped_boundary_potions = 0

    def begin_trial(self, observation):
        if self.belief is None or not self.carry:
            self.belief = io.BeliefStateWithRotation(self.precomputed)
        self.refresh_items(observation)

    def refresh_items(self, observation):
        stones, potions = decode(observation)
        if stones:
            self.belief.new_trial(
                collections.Counter(stones.values()), collections.Counter(potions.values())
            )

    def act(self, observation, step):
        stones, potions = decode(observation)
        settle = settle_action(stones, potions, step)
        if settle is not None:
            return settle
        if not stones or not potions:
            return 0
        self.refresh_items(observation)
        distribution = self.belief.belief_state.world_state_distribution
        best = (0.0, None)
        by_type = {}
        for si, stone in sorted(stones.items()):
            stone_index = sp.align(stone, self.belief.rotation).index()
            for pi, potion in sorted(potions.items()):
                key = (stone_index, potion.index())
                if key not in by_type:
                    outcomes, _ = distribution.possible_outcomes(
                        potion.index(), stone_index, self.precomputed
                    )
                    expected, probability = 0.0, 0.0
                    mask = io.stone_potion_bit_mask(stone_index, potion.index(), self.precomputed)
                    for outcome in sorted(set(outcomes)):
                        branch = copy.deepcopy(distribution)
                        prob = branch.action_and_outcome(
                            stone_index, potion.index(), outcome, self.precomputed, mask
                        )
                        probability += prob
                        expected += prob * cash_value(self.precomputed.stone_to_reward[outcome])
                    if not np.isclose(probability, 1.0, atol=1e-9):
                        raise ValueError("Outcome probabilities do not cover the posterior")
                    by_type[key] = expected - cash_value(stone.reward)
                gain = by_type[key]
                if gain > best[0] + 1e-10:
                    best = (gain, integer_action(si, pi))
        if best[1] is not None:
            return best[1]
        positive = sorted(i for i, s in stones.items() if s.reward > 0)
        return integer_action(positive[0], -1) if positive else 0

    def observe(self, before, action, after, changed_trial):
        stones, potions = decode(before)
        new_stones, _ = decode(after)
        si, pi = sa.int_action_to_tuple(action, slot_based=True, end_trial_action=False)
        if si < 0 or pi < 0:
            return
        if changed_trial:
            # The native API has already emitted the next trial. Never use its
            # new stone, or an internal tracker, as the old action's outcome.
            self.skipped_boundary_potions += 1
            return
        if si not in stones or pi not in potions or si not in new_stones:
            raise ValueError("Cannot reconstruct a real potion outcome")
        prob = self.belief.action_and_outcome(
            stones[si], potions[pi], new_stones[si], self.precomputed
        )
        if prob <= 0:
            raise ValueError("Visible outcome has zero posterior probability")
        self.outcome_updates += 1


def privileged_greedy_action(env, observation, step):
    """Actual hidden graph/latent items; local reference, not an optimal ceiling."""
    stones, potions = decode(observation)
    settle = settle_action(stones, potions, step)
    if settle is not None:
        return settle
    graph = env._chemistry.graph
    best = (0.0, None)
    for stone in env.game_state.existing_stones():
        si = env.game_state.get_stone_ind(stone_inst=stone.idx)
        start = graph.node_list.get_node_by_coords(list(stone.latent))
        for potion in env.game_state.existing_potions():
            pi = env.game_state.get_potion_ind(potion_inst=potion.idx)
            ends = [
                node
                for node, edge in graph.edge_list.edges.get(start, {}).items()
                if potion.same_effect(edge[1])
            ]
            target = ends[0] if ends else start
            gain = cash_value(sum(target.coords)) - cash_value(stones[si].reward)
            if gain > best[0] + 1e-10:
                best = (gain, integer_action(si, pi))
    if best[1] is not None:
        return best[1]
    positive = sorted(i for i, s in stones.items() if s.reward > 0)
    return integer_action(positive[0], -1) if positive else 0
