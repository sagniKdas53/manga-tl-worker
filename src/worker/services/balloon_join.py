"""B3b: join the groups one balloon's 0.35-character budget leaves apart.

Inside one detector balloon, proximity grouping uses the same tight budget as everywhere else, so
a balloon whose columns sit 0.8-2 characters apart comes back as two or three regions, each
translated on its own (ja/sample27, 25, 258, 242), and so does one with a hole where the OCR missed
a column (sample153, 136). Raising the budget for the whole balloon would also join two balloons
YOLO fused into one blob.

So the join works on pairs of groups that grouping already made. Two groups are joined when their
pieces, regrouped on their own at the bigger budget, make one group: the owner veto, the waist veto
and the orientation vote all judge the pair, and the bigger budget never reaches a piece outside it.
A wall stroke between the two groups (a balloon outline, see `bubble_geometry.gap_wall`) keeps them
apart even then: ja/sample24's two bracket balloons look like one staggered balloon by geometry.

Trial regroupings run on copies of the pieces, because the owner veto records its decision on the
pieces it judges.
"""

from collections.abc import Callable
from dataclasses import replace
from itertools import combinations

from worker.services.fragment_grouping import GroupingConfig, GroupingContext, group_fragments

# (group a, group b, regions) -> True when a wall separates them.
Wall = Callable[[list[int], list[int], list], bool]


def balloon_join(config: GroupingConfig, context: GroupingContext, budget: float, wall: Wall | None, max_lines: int):
    """A ``GroupingContext.group_join`` callable for one balloon. ``budget`` is in characters;
    no joined group holds more than ``max_lines`` pieces."""
    trial_config = replace(config, threshold_ratio=budget)
    trial_context = replace(context, group_join=None)

    def join(groups: list[list[int]], regions: list) -> list[list[int]]:
        return join_balloon_groups(groups, regions, trial_config, trial_context, wall, max_lines)

    return join


def join_balloon_groups(
    groups: list[list[int]],
    regions: list,
    trial_config: GroupingConfig,
    trial_context: GroupingContext,
    wall: Wall | None,
    max_lines: int,
) -> list[list[int]]:
    """Join pairs of ``groups`` until no pair joins. The joined group takes the first one's place."""
    groups = [list(group) for group in groups]
    joined_any = False
    changed = True
    while changed:
        changed = False
        for a, b in combinations(range(len(groups)), 2):
            pair = groups[a] + groups[b]
            if len(pair) > max_lines:
                continue
            pieces = [dict(regions[index]) for index in pair]
            if len(group_fragments(pieces, trial_config, trial_context)) != 1:
                continue
            if wall is not None and wall(groups[a], groups[b], regions):
                continue
            groups[a] = sorted(pair)
            del groups[b]
            joined_any = changed = True
            break
    if joined_any and trial_context.owner_veto is not None:
        # Record the joined groups' owner decisions on the real pieces; the trials judged copies.
        for group in groups:
            if len(group) > 1:
                trial_context.owner_veto(group, regions)
    return groups
