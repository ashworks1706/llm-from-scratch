# greedy makes the locally best choice and never revisits it. it is correct only when that choice is provably safe.
# the proof pattern is an exchange argument: any optimal solution can be rewritten to use the greedy first choice.
# first try a counterexample: if a tiny input breaks the rule, it is not greedy, it is dp.
# most greedy solutions are a sort (O(n log n)) plus one linear pass, so name both costs.
# coin change with arbitrary denominations is NOT greedy (coins 1, 3, 4 and amount 6); that one is dp, file 20.
# explain: for jump game, why is tracking the farthest reachable index enough and no search is needed?
# explain: activity selection sorts by end time. give a counterexample for sorting by start time and one for shortest duration.


def can_jump(nums: list[int]) -> bool:
    # nums[i] is the max jump length from i. true when the last index is reachable. O(n), O(1) space.
    # invariant: farthest is the max reachable index so far; if i > farthest you are stuck. single item is true.
    pass


def jump_game_min_jumps(nums: list[int]) -> int:
    # minimum jumps to reach the last index, assuming it is reachable. bfs levels in disguise:
    # track the end of the current level and the farthest next level. one item gives 0.
    pass


def gas_station(gas: list[int], cost: list[int]) -> int:
    # circular route; start index that completes the loop, or -1. if sum(gas) < sum(cost) it is impossible.
    # otherwise reset the candidate start whenever the running tank goes negative. O(n), answer is unique when it exists.
    pass


def partition_labels(text: str) -> list[int]:
    # split into as many parts as possible so each letter appears in only one part; return part sizes.
    # record each letter's last index, extend the current part end to cover every letter seen. O(n).
    pass


def task_scheduler(tasks: list[str], cooldown: int) -> int:
    # least intervals to run all tasks with at least cooldown idle slots between equal tasks.
    # formula: max(len(tasks), (max_count - 1) * (cooldown + 1) + number_of_tasks_with_max_count). cooldown 0 gives len(tasks).
    pass


if __name__ == '__main__':
    # brute force oracle for can_jump: dfs with memo over reachable indices; compare on 200 random short lists.
    # assert gas_station([1,2,3,4,5], [3,4,5,1,2]) == 3 and gas_station([2,3,4], [3,4,3]) == -1.
    pass
