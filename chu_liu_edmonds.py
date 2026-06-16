"""
chu_liu_edmonds.py
==================
Chu-Liu/Edmonds algorithm for the minimum spanning arborescence of a
directed graph.  This is the inference algorithm used by the MST dependency
parser (Section 1) and re-used in the attention-based parser (Section 2).

Provided function
-----------------
  cle_min(scores_dict, n)
      Minimum spanning arborescence rooted at node 0.

Usage example
-------------
  from chu_liu_edmonds import cle_min

  # 4 nodes: 0 = ROOT, 1-3 = words
  scores = {
      (0,1): 1, (0,2): 2, (0,3): 3,
      (1,2): 1, (2,3): 1, (3,1): 1,
      (1,3): 4, (2,1): 3, (3,2): 3,
  }
  tree = cle_min(scores, 4)
  # tree == {1: 0, 2: 1, 3: 2}  (child -> parent)
"""


def cle_min(scores_dict, n):
    """
    Chu-Liu/Edmonds algorithm for the MINIMUM spanning arborescence.

    Parameters
    ----------
    scores_dict : dict  {(u, v): float}
        Directed edge weights.  Node 0 is the root.  Self-loops are ignored.
    n : int
        Total number of nodes  (0 = ROOT, 1 ... n-1 = words).

    Returns
    -------
    dict  {child: parent}
        The minimum-cost arborescence rooted at node 0.
        Every non-root node appears exactly once as a key.
    """
    def _cle(scores, nodes, root):
        # Step 1: for every non-root node pick the cheapest incoming edge.
        min_in = {}
        for v in nodes:
            if v == root:
                continue
            best = min(
                ((u, w) for (u, vv), w in scores.items() if vv == v and u != v),
                key=lambda x: x[1],
                default=None,
            )
            if best is None:
                return {}
            min_in[v] = best[0]

        # Step 2: detect a cycle in the chosen edges.
        def find_cycle():
            for start in nodes:
                if start == root:
                    continue
                visited, path = set(), []
                v = start
                while v not in visited and v != root:
                    visited.add(v)
                    path.append(v)
                    v = min_in.get(v, root)
                if v in visited and v != root:
                    return path[path.index(v):]
            return None

        cycle = find_cycle()
        if cycle is None:
            return min_in

        # Step 3: contract the cycle into a single super-node.
        cycle_set = set(cycle)
        cid = cycle[0]
        in_cost = {v: scores[(min_in[v], v)] for v in cycle}

        new_scores = {}
        for (u, v), w in scores.items():
            nu = cid if u in cycle_set else u
            nv = cid if v in cycle_set else v
            if nu == nv:
                continue
            if nv == cid:
                adj = w - in_cost[v]
                key = (nu, nv)
                if key not in new_scores or new_scores[key][0] > adj:
                    new_scores[key] = (adj, v)
            else:
                key = (nu, nv)
                if key not in new_scores or new_scores[key][0] > w:
                    new_scores[key] = (w, None)

        flat = {k: val[0] for k, val in new_scores.items()}
        new_nodes = (nodes - cycle_set) | {cid}

        # Step 4: recurse on the contracted graph.
        sub = _cle(flat, new_nodes, root)

        # Step 5: expand — restore all cycle edges except the one superseded
        # by the incoming edge selected in the contracted sub-problem.
        broken = None
        for (u, v), (w, cv) in new_scores.items():
            if v == cid and sub.get(cid) == u:
                broken = cv
                break

        result = {v: u for v, u in sub.items() if v != cid}
        for v in cycle:
            if v != broken:
                result[v] = min_in[v]
        if broken is not None:
            result[broken] = sub[cid]
        return result

    return _cle(scores_dict, set(range(n)), 0)
