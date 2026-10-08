"""Act map generation (the game's StandardActMap), ported from r33hab/sts2
`RunMapGenerator.GenerateActMap`.

A map is a 7-column grid. Row 0 holds the act's Ancient, rows 1..boss_row-1
are rooms, and the boss sits alone on `boss_row` (16 for Act 1, 15 for the
Hive, 14 for Glory). Seven random paths are walked up from row 1 (no
crossing edges), then point types are dealt: row 1 is all Monsters, the
treasure row (boss_row - 7) all Treasure, the last row before the boss all
Rest Sites, and the rest are dealt Rest / Shop / Elite / Unknown counts with
the game's placement rules (no Rest or Elite below row 6, no Rest in the top
three rows, no Elite/Rest/Treasure/Shop next to the same type, no two
siblings of a type). Duplicate path segments are pruned and the grid is
centred, spread and straightened.

The RNG is a Python stream, so maps match the game's in shape and rules but
not bit for bit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

WIDTH = 7
PATHS = 7
START_COL = WIDTH // 2
SHOP_COUNT = 3

MONSTER = "Monster"
ELITE = "Elite"
REST = "RestSite"
SHOP = "Shop"
TREASURE = "Treasure"
UNKNOWN = "Unknown"
BOSS = "Boss"
ANCIENT = "Ancient"
_NONE = ""

# r33hab RunConstants.ActRoomCounts: (weak encounters, rooms); boss row = rooms + 1.
ACT_ROOMS = {"overgrowth": (3, 15), "underdocks": (3, 15), "hive": (2, 14), "glory": (2, 13)}

Coord = tuple[int, int]


@dataclass
class MapNode:
    col: int
    row: int
    kind: str = _NONE
    children: list[Coord] = field(default_factory=list)
    parents: list[Coord] = field(default_factory=list)
    can_be_modified: bool = True

    @property
    def coord(self) -> Coord:
        return (self.col, self.row)


@dataclass
class ActMap:
    act: str
    boss_row: int
    nodes: dict[Coord, MapNode]
    second_boss: bool = False

    def node(self, coord: Coord) -> MapNode:
        return self.nodes[coord]

    @property
    def start(self) -> Coord:
        return (START_COL, 0)

    def children_of(self, coord: Coord) -> list[MapNode]:
        return sorted((self.nodes[c] for c in self.nodes[coord].children), key=lambda n: (n.row, n.col))


def boss_row_for(act: str) -> int:
    return ACT_ROOMS[act][1] + 1


def elite_count(ascension: int) -> int:
    # Swarming Elites (A1) multiplies by 1.6: round(5 * 1.6) = 8.
    return 8 if ascension >= 1 else 5


def _gaussian_int(rng: random.Random, mean: int, std: int, lo: int, hi: int) -> int:
    while True:
        d = 1.0 - rng.random()
        n = 1.0 - rng.random()
        sample = math.sqrt(-2.0 * math.log(d)) * math.sin(2.0 * math.pi * n)
        v = int(round(mean + std * sample))
        if lo <= v <= hi:
            return v


def _point_counts(act: str, rng: random.Random) -> tuple[int, int]:
    """ActModel.GetMapPointTypes: (rest sites, unknowns)."""

    if act == "hive":
        rest = _gaussian_int(rng, 6, 1, 6, 7)
    elif act == "glory":
        rest = rng.randrange(5, 7)
    else:
        rest = _gaussian_int(rng, 7, 1, 6, 7)
    unknown = _gaussian_int(rng, 12, 1, 10, 14)
    return rest, unknown - 1 if act in ("hive", "glory") else unknown


class _Gen:
    def __init__(self, act: str, ascension: int, rng: random.Random) -> None:
        self.act = act
        self.asc = ascension
        self.rng = rng
        self.boss_row = boss_row_for(act)
        self.nodes: dict[Coord, MapNode] = {}

    # -- graph helpers ---------------------------------------------------------

    def get(self, col: int, row: int) -> MapNode:
        n = self.nodes.get((col, row))
        if n is None:
            n = MapNode(col, row)
            self.nodes[(col, row)] = n
        return n

    def add_edge(self, parent: Coord, child: Coord) -> None:
        p = self.get(*parent)
        c = self.get(*child)
        if child not in p.children:
            p.children.append(child)
        if parent not in c.parents:
            c.parents.append(parent)

    def remove_edge(self, parent: Coord, child: Coord) -> None:
        if parent in self.nodes and child in self.nodes[parent].children:
            self.nodes[parent].children.remove(child)
        if child in self.nodes and parent in self.nodes[child].parents:
            self.nodes[child].parents.remove(parent)

    def is_grid_row(self, row: int) -> bool:
        return 0 < row < self.boss_row

    def stable_shuffle(self, items: list[MapNode]) -> list[MapNode]:
        items = sorted(items, key=lambda n: (n.col, n.row))
        self.rng.shuffle(items)
        return items

    # -- generation --------------------------------------------------------------

    def generate(self) -> ActMap:
        self.get(START_COL, 0)
        self.get(START_COL, self.boss_row)
        rest_count, unknown_count = _point_counts(self.act, self.rng)
        starts: list[Coord] = []
        for path in range(PATHS):
            col = self.rng.randrange(WIDTH)
            if path == 1:
                while (col, 1) in starts:
                    col = self.rng.randrange(WIDTH)
            cur = self.get(col, 1)
            if (col, 1) not in starts:
                starts.append((col, 1))
            self._walk(cur)
        for s in sorted(starts):
            self.add_edge((START_COL, 0), s)
        for n in sorted((n for n in self.nodes.values() if n.row == self.boss_row - 1), key=lambda n: n.col):
            self.add_edge(n.coord, (START_COL, self.boss_row))
        self._assign_types(rest_count, unknown_count)
        for _ in range(3):
            self._prune_duplicates()
            if not self._repair(rest_count, unknown_count):
                break
        self._center()
        self._spread()
        self._straighten()
        self.get(START_COL, 0).kind = ANCIENT
        self.get(START_COL, self.boss_row).kind = BOSS
        return ActMap(act=self.act, boss_row=self.boss_row, nodes=self.nodes)

    def _walk(self, cur: MapNode) -> None:
        while cur.row < self.boss_row - 1:
            deltas = [-1, 0, 1]
            self.rng.shuffle(deltas)
            target = (cur.col, cur.row + 1)
            for d in deltas:
                col = min(max(cur.col + d, 0), WIDTH - 1)
                if not self._crosses(cur, col):
                    target = (col, cur.row + 1)
                    break
            self.add_edge(cur.coord, target)
            cur = self.nodes[target]

    def _crosses(self, cur: MapNode, target_col: int) -> bool:
        delta = target_col - cur.col
        sibling = self.nodes.get((target_col, cur.row))
        if delta == 0 or sibling is None:
            return False
        return any(c[0] - sibling.col == -delta for c in sibling.children)

    # -- point types ---------------------------------------------------------

    def _assign_types(self, rest_count: int, unknown_count: int) -> None:
        treasure_row = self.boss_row - 7
        rest_row = self.boss_row - 1
        for n in self.nodes.values():
            n.kind = (MONSTER if n.row == 1 else TREASURE if n.row == treasure_row
                      else REST if n.row == rest_row else BOSS if n.row == self.boss_row else _NONE)
            n.can_be_modified = n.row not in (1, treasure_row, rest_row)
        queue = ([REST] * rest_count + [SHOP] * SHOP_COUNT + [ELITE] * elite_count(self.asc)
                 + [UNKNOWN] * unknown_count)
        for _ in range(3):
            if not queue:
                break
            cands = [n for n in self.nodes.values()
                     if n.kind == _NONE and 1 < n.row < rest_row and n.row != treasure_row]
            for n in self.stable_shuffle(cands):
                if not queue:
                    break
                n.kind = self._next_valid(queue, n)
        for n in self.nodes.values():
            if n.kind == _NONE and n.row > 0:
                n.kind = MONSTER

    def _next_valid(self, queue: list[str], n: MapNode) -> str:
        for _ in range(len(queue)):
            kind = queue.pop(0)
            if self._valid(kind, n):
                return kind
            queue.append(kind)
        return _NONE

    def _valid(self, kind: str, n: MapNode) -> bool:
        if n.row < 6 and kind in (REST, ELITE):
            return False
        if n.row >= self.boss_row - 3 and kind == REST:
            return False
        if kind in (ELITE, REST, TREASURE, SHOP):
            if any(self.nodes[c].kind == kind for c in n.parents + n.children):
                return False
        if kind in (REST, MONSTER, UNKNOWN, ELITE, SHOP):
            for p in n.parents:
                for c in self.nodes[p].children:
                    if c != n.coord and self.nodes[c].kind == kind:
                        return False
        return True

    def _repair(self, rest_count: int, unknown_count: int) -> bool:
        repaired = False
        for kind, target in ((SHOP, SHOP_COUNT), (ELITE, elite_count(self.asc)), (REST, rest_count),
                             (UNKNOWN, unknown_count)):
            missing = target - sum(1 for n in self.nodes.values() if n.kind == kind)
            if missing <= 0:
                continue
            cands = [n for n in self.nodes.values() if n.kind == MONSTER and n.can_be_modified]
            for n in self.stable_shuffle(cands):
                if missing == 0:
                    break
                if self._valid(kind, n):
                    n.kind = kind
                    missing -= 1
                    repaired = True
        return repaired

    # -- pruning duplicate segments ----------------------------------------------

    def _prune_duplicates(self) -> None:
        for _ in range(51):
            groups = self._matching_segments()
            if not self._prune_paths(groups):
                return

    def _matching_segments(self) -> list[list[list[MapNode]]]:
        """Groups of same-shaped segments (same ends, same room types) of 3+ points.

        Walks every path from each valid segment start instead of re-slicing every
        full path (FindAllPaths + AddSegmentsToDictionary); the segments found are
        the same, only discovered in a different order."""

        segments: dict[str, list[list[MapNode]]] = {}
        starts = sorted((n for n in self.nodes.values() if n.kind != BOSS
                         and (len(n.children) > 1 or n.row == 0)), key=lambda n: (n.row, n.col))
        for start in starts:
            stack: list[list[MapNode]] = [[start]]
            while stack:
                path = stack.pop()
                end = path[-1]
                if len(path) >= 3 and len(end.parents) >= 2:
                    key = self._segment_key(path)
                    existing = segments.get(key)
                    if existing is None:
                        segments[key] = [path]
                    elif not any(self._overlap(e, path) for e in existing):
                        existing.append(path)
                if end.kind == BOSS:
                    continue
                for c in reversed(end.children):
                    stack.append(path + [self.nodes[c]])
        return [segs for _, segs in sorted(segments.items()) if len(segs) > 1]

    @staticmethod
    def _segment_key(seg: list[MapNode]) -> str:
        s, e = seg[0], seg[-1]
        prefix = f"{s.row}-{e.col},{e.row}-" if s.row == 0 else f"{s.col},{s.row}-{e.col},{e.row}-"
        return prefix + ",".join(n.kind for n in seg)

    @staticmethod
    def _overlap(a: list[MapNode], b: list[MapNode]) -> bool:
        if len(a) < 3 or len(b) < 3:
            return False
        return any(a[i] is b[i] for i in range(1, min(len(a), len(b)) - 1))

    def _prune_paths(self, groups: list[list[list[MapNode]]]) -> bool:
        for group in groups:
            self.rng.shuffle(group)
            if self._prune_all_but_last(group):
                return True
            if self._break_relationship(group):
                return True
        return False

    def _prune_all_but_last(self, matches: list[list[MapNode]]) -> int:
        pruned = 0
        for seg in matches:
            if pruned == len(matches) - 1:
                return pruned
            if self._prune_segment(seg):
                pruned += 1
        return pruned

    def _in_map(self, n: MapNode) -> bool:
        return n.kind == BOSS or n.row == 0 or self.nodes.get(n.coord) is n

    def _prune_segment(self, seg: list[MapNode]) -> bool:
        pruned = False
        for i in range(len(seg) - 1):
            n = seg[i]
            if not self._in_map(n):
                return True
            if len(n.children) > 1 or len(n.parents) > 1 or any(
                    len(self.nodes[p].children) == 1 and self.is_grid_row(p[1]) for p in n.parents):
                continue
            if any(len(x.children) > 1 and len(x.parents) == 1 for x in seg[i:]):
                continue
            if len(seg[-1].parents) == 1:
                return False
            seg_coords = {x.coord for x in seg}
            if not any(len(self.nodes[c].parents) == 1 for c in n.children if c not in seg_coords):
                self._remove_point(n)
                pruned = True
        return pruned

    def _remove_point(self, n: MapNode) -> None:
        self.nodes.pop(n.coord, None)
        for c in list(n.children):
            self.remove_edge(n.coord, c)
        for p in list(n.parents):
            self.remove_edge(p, n.coord)

    def _break_relationship(self, matches: list[list[MapNode]]) -> bool:
        for seg in matches:
            broken = False
            for i in range(len(seg) - 1):
                n = seg[i]
                if len(n.children) < 2:
                    continue
                child = seg[i + 1]
                if len(child.parents) != 1:
                    self.remove_edge(n.coord, child.coord)
                    broken = True
            if broken:
                return True
        return False

    # -- layout passes ---------------------------------------------------------

    def _col_empty(self, col: int) -> bool:
        return not any(self.is_grid_row(n.row) and n.col == col for n in self.nodes.values())

    def _center(self) -> None:
        left = self._col_empty(0) and self._col_empty(1)
        right = self._col_empty(WIDTH - 1) and self._col_empty(WIDTH - 2)
        delta = -1 if left and not right else 1 if right and not left else 0
        if delta == 0:
            return
        grid = [n for n in self.nodes.values() if self.is_grid_row(n.row)]
        grid.sort(key=lambda n: ((-n.col if delta > 0 else n.col), n.row))
        for n in grid:
            self._move(n, n.col + delta)

    def _spread(self) -> None:
        for row in range(1, self.boss_row):
            row_nodes = sorted((n for n in self.nodes.values() if n.row == row), key=lambda n: n.col)
            changed = True
            while changed:
                changed = False
                for n in row_nodes:
                    allowed = set(range(WIDTH))
                    for c in n.parents + n.children:
                        allowed &= set(range(max(0, c[0] - 1), min(WIDTH - 1, c[0] + 1) + 1))
                    best, best_gap = n.col, self._gap(n.col, row_nodes, n)
                    for col in sorted(allowed):
                        if col == n.col or (col, row) in self.nodes:
                            continue
                        gap = self._gap(col, row_nodes, n)
                        if gap > best_gap:
                            best, best_gap = col, gap
                    if best != n.col:
                        self._move(n, best)
                        changed = True

    @staticmethod
    def _gap(col: int, row_nodes: list[MapNode], cur: MapNode) -> int:
        gaps = [abs(col - n.col) for n in row_nodes if n is not cur]
        return min(gaps) if gaps else 1 << 30

    def _straighten(self) -> None:
        for row in range(1, self.boss_row):
            for col in range(WIDTH):
                n = self.nodes.get((col, row))
                if n is None or len(n.parents) != 1 or len(n.children) != 1:
                    continue
                p, c = n.parents[0], n.children[0]
                if n.col < c[0] and n.col < p[0] and col < WIDTH - 1 and (col + 1, row) not in self.nodes:
                    self._move(n, col + 1)
                elif n.col > c[0] and n.col > p[0] and col > 0 and (col - 1, row) not in self.nodes:
                    self._move(n, col - 1)

    def _move(self, n: MapNode, new_col: int) -> None:
        if new_col < 0 or new_col >= WIDTH or new_col == n.col or (new_col, n.row) in self.nodes:
            return
        old = n.coord
        del self.nodes[old]
        n.col = new_col
        new = n.coord
        self.nodes[new] = n
        for p in n.parents:
            if p in self.nodes:
                lst = self.nodes[p].children
                lst[lst.index(old)] = new
        for c in n.children:
            if c in self.nodes:
                lst = self.nodes[c].parents
                lst[lst.index(old)] = new


def generate_act_map(act: str, ascension: int, rng: random.Random, *, second_boss: bool = False) -> ActMap:
    m = _Gen(act, ascension, rng).generate()
    if second_boss:
        boss = m.nodes[(START_COL, m.boss_row)]
        second = MapNode(START_COL, m.boss_row + 1, kind=BOSS, can_be_modified=False)
        m.nodes[second.coord] = second
        boss.children.append(second.coord)
        second.parents.append(boss.coord)
        m.second_boss = True
    return m


__all__ = [
    "ACT_ROOMS", "ANCIENT", "ActMap", "BOSS", "ELITE", "MONSTER", "MapNode", "REST", "SHOP", "TREASURE",
    "UNKNOWN", "WIDTH", "boss_row_for", "elite_count", "generate_act_map",
]
