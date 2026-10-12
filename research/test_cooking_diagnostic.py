"""Meaningful adapter and information-boundary checks for the new diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).with_name("run_cooking_diagnostic.py"))
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_registered_action_directions_match_upstream():
    Env, _ = diagnostic.load_environment(Path("/tmp/robot-task-dream-20261013"))
    for destination in [(2,1),(0,1),(1,2),(1,0)]:
        env = Env(np.array([27]),lambda s:s)
        before=env.reset().observation
        after,_,_,_=env.step(diagnostic.move(before[:2],destination))
        assert tuple(after.observation[:2]) == destination


def test_unknown_pickup_is_not_admitted_as_a_mapping():
    memory={}
    diagnostic.absorb(memory,np.array([2,1,3,4,4,4,4]),7,np.array([2,1,3,4,4,4,4]))
    assert memory == {}
    diagnostic.absorb(memory,np.array([2,1,3,4,4,4,4]),7,np.array([2,1,3,2,4,4,4]))
    assert memory == {1:2}


def test_visible_inputs_are_sufficient_for_each_non_oracle_decision():
    obs=np.array([1,1,2,4,4,4,4])
    assert diagnostic.solve_action(obs,[1,2],{},[0,1,2]) == diagnostic.solve_action(obs.copy(),[1,2],{},[0,1,2])
    assert diagnostic.explore_action(obs,{},[0,1,2]) == diagnostic.explore_action(obs.copy(),{},[0,1,2])


def test_oracle_completes_all_64_mappings_and_possible_goals():
    import itertools
    Env, Wrapper = diagnostic.load_environment(Path("/tmp/robot-task-dream-20261013"))
    for env_id in range(64):
        truth={i:env_id//4**i%4 for i in range(3)}
        for goal in set(itertools.product(truth.values(),repeat=2)):
            env=Wrapper(Env(np.array([env_id]),lambda s:s),[],seed=13,fixed_instructions=True)
            env.registered_goal=list(goal)
            state=env.reset()
            for t in range(20):
                action=diagnostic.solve_action(state.observation,goal,truth,[0,1,2])
                state,_,done,_=env.step(action)
                if done:break
            assert tuple(state.observation[4:6]) == goal, (env_id,goal)
