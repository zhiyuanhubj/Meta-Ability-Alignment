###############################################################################
# Abduction – Reverse Rule-Graph Search Generator
#
# Implements the "Abduction" task genration. Each puzzle requires constructing and 
# validating a minimal backward proof tree (reverse rule-graph) given premises and goals.
#
# Core hyper-parameters (per difficulty level):
#  • chain_depth (d_l): maximum length of backward inference chains.
#  • num_goals (g_l): number of sink goals to explain.
#  • distractor_count (h_l): count of extra hyper-edges sharing symbols.
#  • cycle_prob (γ): probability of injecting cyclic rule dependencies.
#
# Difficulty mapping (Table 3, supplementary):
#    Level 1 → ⟨d_l=2–3, g_l=1, h_l=3–5, γ=0.10⟩
#    Level 2 → ⟨d_l=3–4, g_l=2, h_l=5–7, γ=0.15⟩
#    Level 3 → ⟨d_l=4–5, g_l=2, h_l=7–9, γ=0.20⟩
#    Level 4 → ⟨d_l=5–6, g_l=3, h_l=8–10, γ=0.25⟩
#    Level 5 → ⟨d_l=6–7, g_l=3, h_l=10–12, γ=0.30⟩
#
# This script produces JSONL files `abduction_dataset/1.jsonl` … `abduction_dataset/5.jsonl`,
# each containing N verified sample with fields:
#    { problem_id, premises, known_atoms, goals,
#      reachable_goals, unreachable_goals }
###############################################################################

import json
import random
import concurrent.futures
from typing import List, Dict, Tuple, Set, Any

# --- Atom and expression utilities -----------------------------------------
LIMITED_LETTERS = "ABCDEFGHIJKLMNOP"  # pool for random atom names

def _make_random_atom(length: int = 1) -> str:
    """Generate a random atom of given length."""
    return ''.join(random.choice(LIMITED_LETTERS) for _ in range(length))


def tokenize_expr(expr_str: str) -> List[str]:
    """Split a logical expression into tokens: parentheses and uppercase words."""
    tokens, i = [], 0
    while i < len(expr_str):
        ch = expr_str[i]
        if ch.isspace():
            i += 1; continue
        if ch in '()':
            tokens.append(ch); i += 1; continue
        if ch.isalpha():
            start = i
            while i < len(expr_str) and expr_str[i].isalpha():
                i += 1
            tokens.append(expr_str[start:i].upper())
        else:
            raise ValueError(f"Invalid char '{ch}' in {expr_str}")
    return tokens


# --- Parsing and evaluation ------------------------------------------------
# (Reuses parse_expression & build_eval_func from your main code)

from __main__ import parse_expression, build_eval_func, parse_premise, extract_atoms_from_premise

# --- Advanced subchain generation -----------------------------------------
def _random_expr(atoms: List[str], max_depth: int) -> str:
    """Build a random Boolean sub-expression of depth ≤ max_depth."""
    if max_depth <= 1 or random.random() < 0.4:
        a = random.choice(atoms)
        return f"{a}" if random.random() < 0.5 else f"(NOT {a})"
    left = _random_expr(atoms, max_depth-1)
    right = _random_expr(atoms, max_depth-1)
    op = random.choice(["AND", "OR"])
    return f"({left} {op} {right})"

def _random_complex_premise(atoms: List[str], max_depth: int) -> str:
    """Generate a random Horn clause '(body) => head'."""
    body = _random_expr(atoms, max_depth)
    head = random.choice(atoms)
    return f"({body}) => {head}"

def _generate_subchain(target: str,
                       depth: int,
                       cycle_prob: float,
                       existing: List[str]
                      ) -> Tuple[List[str], List[str]]:
    """
    Generate a backward chain ending at `target`. May introduce cycles.
    Returns (premises, used_atoms).
    """
    premises, used, chain_syms = [], [], []
    curr = target
    for _ in range(depth):
        if chain_syms and random.random() < cycle_prob:
            prev = random.choice(chain_syms)
        else:
            prev = _make_random_atom()
            chain_syms.append(prev)
        pool = existing + used + chain_syms + [curr]
        expr = _random_expr(pool, max_depth=2)
        premises.append(f"({expr}) => {curr}")
        for tok in tokenize_expr(expr):
            if tok.isalpha() and tok not in used:
                used.append(tok)
        curr = prev
    return premises, used


def generate_abduction_problem(
    problem_id: int,
    num_goals: int,
    reachable_k: int,
    chain_depth: int,
    distractors: int,
    cycle_prob: float
) -> Dict[str, Any]:
    """
    Assemble an abduction puzzle with given params.
    """
    goals = [_make_random_atom() for _ in range(num_goals)]
    random.shuffle(goals)
    reachable = goals[:reachable_k]
    unreachable = goals[reachable_k:]
    premises, known_atoms = [], []
    # reachable chains
    for g in reachable:
        sub, used = _generate_subchain(g, chain_depth, cycle_prob, [])
        premises += sub; known_atoms += used
    # unreachable chains (drop some links)
    for g in unreachable:
        sub, used = _generate_subchain(g, chain_depth, cycle_prob, [])
        drop = random.randint(1, len(sub)-1)
        premises += sub[:-drop]; known_atoms += used[:-drop]
    # add distractors
    pool = list(set(known_atoms + goals))
    for _ in range(distractors):
        premises.append(_random_complex_premise(pool, max_depth=2))
    # dedupe
    premises = list(dict.fromkeys(premises))
    known_atoms = list(dict.fromkeys(known_atoms))
    return {
        "problem_id": f"ABD_{problem_id}",
        "premises": premises,
        "known_atoms": known_atoms,
        "goals": goals,
        "reachable_goals": reachable,
        "unreachable_goals": unreachable
    }


def check_consistency(puzzle: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Verify puzzle via brute-force over all valuations (N ≤ 20).
    """
    # collect atoms
    atoms = set(puzzle["known_atoms"]) | set(puzzle["goals"])
    for pm in puzzle["premises"]:
        atoms |= extract_atoms_from_premise(pm)
    atoms = sorted(atoms)
    if len(atoms) > 20:
        return False, f"Too many atoms ({len(atoms)})"
    # compile premise funcs
    funcs = []
    try:
        for pm in puzzle["premises"]:
            funcs.append(parse_premise(pm))
    except ValueError as e:
        return False, f"Parse error: {e}"
    # enumerate
    sats = []
    for mask in range(1 << len(atoms)):
        vmap = {a: bool((mask >> i) & 1) for i, a in enumerate(atoms)}
        if not all(vmap[a] for a in puzzle["known_atoms"]): continue
        if all(f(vmap) for f in funcs): sats.append(vmap)
    if not sats:
        return False, "No model satisfies premises"
    # check goals
    for rg in puzzle["reachable_goals"]:
        if any(not m[rg] for m in sats):
            return False, f"Reachable goal {rg} fails"
    for ug in puzzle["unreachable_goals"]:
        if all(m[ug] for m in sats):
            return False, f"Unreachable goal {ug} true always"
    return True, "OK"


# --- Difficulty parameters -------------------------------------------------
DIFFICULTY_PARAMS = {
    1: dict(chain_depth=(2,3), num_goals=(1,1), distractors=(3,5), cycle_prob=0.10),
    2: dict(chain_depth=(3,4), num_goals=(2,2), distractors=(5,7), cycle_prob=0.15),
    3: dict(chain_depth=(4,5), num_goals=(2,2), distractors=(7,9), cycle_prob=0.20),
    4: dict(chain_depth=(5,6), num_goals=(3,3), distractors=(8,10),cycle_prob=0.25),
    5: dict(chain_depth=(6,7), num_goals=(3,3), distractors=(10,12),cycle_prob=0.30),
}


def generate_abduction_dataset(
    n_per_level: int = 100,
    max_workers: int = 4
) -> None:
    """
    Produce JSONL files `abduction_dataset/1.jsonl` … `abduction_dataset/5.jsonl`.
    """
    import os
    os.makedirs("abduction_dataset", exist_ok=True)

    for lvl in range(1, 6):
        params = DIFFICULTY_PARAMS[lvl]
        out_file = f"abduction_dataset/{lvl}.jsonl"
        count = 0
        with open(out_file, 'w', encoding='utf-8') as fw, \
             concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exec:
            futures = []
            while count < n_per_level:
                for _ in range(max_workers):
                    cd = random.randint(*params['chain_depth'])
                    kg = random.randint(*params['num_goals'])
                    dist = random.randint(*params['distractors'])
                    futures.append(
                        exec.submit(
                            generate_abduction_problem,
                            count, kg, 1, cd, dist, params['cycle_prob']
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    puzzle = fut.result()
                    ok, _ = check_consistency(puzzle)
                    if ok:
                        json.dump(puzzle, fw, ensure_ascii=False)
                        fw.write("\n")
                        count += 1
                        print(f"[L{lvl}] {count}/{n_per_level}")
                        if count >= n_per_level:
                            break
                futures.clear()
        print(f"Finished Level {lvl}: {n_per_level} puzzles → {out_file}")


if __name__ == '__main__':
    generate_abduction_dataset(n_per_level=100, max_workers=4)