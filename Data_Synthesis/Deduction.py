###############################################################################
# Deduction – Propositional Satisfiability Task Generator
#
# Implements the Deduction track from
# “Beyond ‘Aha!’: Toward Systematic Meta-Abilities Alignment in Large
# Reasoning Models.”  Each puzzle asks whether a single Boolean assignment
# ─────────── Paper-aligned hyper-parameters ───────────
# • nℓ  (→ n_vars)      – number of propositional variables.
# • fℓ  (sampled 3–6)   – number of independent formulas per puzzle.
# • dℓ  (→ max_depth)   – maximum parenthesis nesting depth.
#
# Difficulty levels (Table 1, supplementary):
#   Level 1 → ⟨4,  ≈3–4, 1⟩
#   Level 2 → ⟨6,  ≈3–5, 2⟩
#   Level 3 → ⟨8,  ≈4–5, 3⟩
#   Level 4 → ⟨10, ≈4–6, 4⟩
#   Level 5 → ⟨12, ≈4–6, 5⟩
#
# The public release supports only the *nested-formula* mode stated in the supplementary materials;  
# A brute-force 2ⁿ search is used; swap `_search` with a DPLL
# implementation (Algorithm 1 in the paper) if you need larger-n experiments.
#
# Intended uses
# • RL reward  – +2 if all formulas evaluate True under the model’s assignment,
#                −2 otherwise (see §3.2 reward schedule).
###############################################################################
from __future__ import annotations
import os
import json
import random
from typing import List, Dict, Optional

###############################################################################
# Utility helpers
###############################################################################
def write_jsonl(path: str, rows):
    """Write a list of dicts to *path* in JSON Lines format."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def init_seed(seed: int = 42):
    """Deterministic behaviour across runs."""
    random.seed(seed)

###############################################################################
# Sampler
###############################################################################
class NestedLogicPuzzleSampler:
    """
    Generate propositional-satisfiability puzzles comprising nested formulas.

    Parameters
    ----------
    difficulty : int
        Curriculum level 1‒5, mapping to (nℓ, dℓ) as defined in the paper.
    seed : int
        RNG seed for reproducibility.
    """

    DIFFICULTY = {
        1: dict(n_vars=4,  max_depth=1),
        2: dict(n_vars=6,  max_depth=2),
        3: dict(n_vars=8,  max_depth=3),
        4: dict(n_vars=10, max_depth=4),
        5: dict(n_vars=12, max_depth=5),
    }

    def __init__(self, difficulty: int = 3, seed: int = 42):
        if difficulty not in self.DIFFICULTY:
            raise ValueError("difficulty must be 1–5")

        cfg = self.DIFFICULTY[difficulty]
        self.n_vars    = cfg["n_vars"]     # == nℓ
        self.max_depth = cfg["max_depth"]  # == dℓ
        self.rng       = random.Random(seed)
        self.var_list  = [chr(ord('A') + i) for i in range(self.n_vars)]

    # ─────────────────────────────────────────────────────────────────────────
    # Formula generation
    # ─────────────────────────────────────────────────────────────────────────
    def _random_subformula(self, depth: int = 0) -> str:
        """
        Recursively create one nested propositional clause.

        Leaves (depth == dℓ) return a literal ‘A’ or ‘¬A’.
        Internal nodes:
        - 20 % chance produce a unary ‘¬( … )’
        - Otherwise produce a binary op from {→, ∧, ∨, ↔, ⊕}
        """
        if depth >= self.max_depth:
            var = self.rng.choice(self.var_list)
            return f"¬{var}" if self.rng.random() < 0.4 else var

        # 1-ary NOT with 20 % probability
        if self.rng.random() < 0.2:
            return f"¬({self._random_subformula(depth + 1)})"

        # Otherwise pick a binary operator
        op    = self.rng.choice(["→", "∧", "∨", "↔", "⊕"])
        left  = self._random_subformula(depth + 1)
        right = self._random_subformula(depth + 1)
        return f"({left} {op} {right})"

    def _random_formula_set(self) -> List[str]:
        """Sample fℓ ∈ {3,…,6} independent formulas."""
        f_l = self.rng.randint(3, 6)
        return [self._random_subformula(0) for _ in range(f_l)]

    # ─────────────────────────────────────────────────────────────────────────
    # Parsing & evaluation
    # ─────────────────────────────────────────────────────────────────────────
    def _tokenize(self, expr: str) -> List[str]:
        """Split parentheses, operators, and atoms into tokens."""
        expr = expr.replace("(", " ( ").replace(")", " ) ")
        return [tok for tok in expr.split() if tok]

    def _parse_expr(self, tokens: List[str], pos: int = 0):
        """
        Recursive-descent parser → AST.
        Grammar (informal):
            expr ::= '(' expr binop expr ')'   |
                     '(' expr ')'              |
                     '¬' expr                  |
                     literal
        """
        def rec(p: int):
            tok = tokens[p]
            if tok == '(':
                # Parenthesised sub-expression
                p += 1
                left, p = rec(p)
                if tokens[p] == ')':          # single-child "( A )" case
                    return left, p + 1
                op = tokens[p]; p += 1        # binary operator
                right, p = rec(p)
                if tokens[p] != ')':
                    raise ValueError("Missing ')'")
                return ("BIN", op, left, right), p + 1

            # Handle leading NOT
            if tok.startswith('¬'):
                if tok == '¬':
                    sub, p2 = rec(p + 1)
                    return ("NOT", sub), p2
                return ("NOT", ("VAR", tok[1:])), p + 1

            # Literal
            return ("VAR", tok), p + 1

        return rec(pos)

    def _eval_ast(self, node, assign: Dict[str, bool]) -> bool:
        """Evaluate AST under the given truth assignment."""
        typ = node[0]
        if typ == "VAR":  # leaf
            return assign[node[1]]
        if typ == "NOT":
            return not self._eval_ast(node[1], assign)

        # Binary op
        _, op, l, r = node
        a, b = self._eval_ast(l, assign), self._eval_ast(r, assign)
        return {
            "∧": a and b,
            "∨": a or b,
            "→": (not a) or b,
            "↔": a == b,
            "⊕": a != b,
        }[op]

    def _eval_expr(self, expr: str, assign: Dict[str, bool]) -> bool:
        """Convenience wrapper: string → tokens → AST → truth value."""
        ast, _ = self._parse_expr(self._tokenize(expr))
        return self._eval_ast(ast, assign)

    # ─────────────────────────────────────────────────────────────────────────
    # Brute-force search
    # ─────────────────────────────────────────────────────────────────────────
    def _satisfies(self, formulas: List[str], assign: Dict[str, bool]) -> bool:
        """True iff *assign* makes every formula evaluate to True."""
        return all(self._eval_expr(f, assign) for f in formulas)

    def _search(self, formulas: List[str]):
        """
        Generator over all satisfying assignments using 2ⁿ enumeration.

        Replace this with a DPLL-style solver (Alg. 1 in the paper)
        for larger-scale experiments.
        """
        n = len(self.var_list)
        for bits in range(1 << n):
            assign = {v: bool(bits & (1 << i)) for i, v in enumerate(self.var_list)}
            if self._satisfies(formulas, assign):
                yield assign

    # ─────────────────────────────────────────────────────────────────────────
    # Public sampling API
    # ─────────────────────────────────────────────────────────────────────────
    def sample_valid(self, k: int):
        """Return *k* puzzles that have ≥1 satisfying assignment."""
        puzzles = []
        while len(puzzles) < k:
            fms = self._random_formula_set()
            sols = list(self._search(fms))
            if sols:
                puzzles.append((fms, sols[0]))
        return puzzles

    def sample_unique(self, k: int):
        """Return *k* puzzles that have exactly **one** satisfying assignment."""
        puzzles = []
        while len(puzzles) < k:
            fms = self._random_formula_set()
            sols = list(self._search(fms))
            if len(sols) == 1:
                puzzles.append((fms, sols[0]))
        return puzzles

###############################################################################
# Puzzle ⇄ text formatter  (updated to match required I/O format)
###############################################################################
class PuzzleFormatter:
    """Human-readable prompt / solution for each puzzle spec."""

    def __init__(self,
                 formulas: List[str],
                 assignment: Optional[Dict[str, bool]]):
        self.formulas   = formulas
        self.assignment = assignment

    # ---------- prompt ----------
    def puzzle_text(self) -> str:
        """Return the puzzle exactly as 'Below are…' + dashed list."""
        lines = ["Below are some nested formulas:"]
        lines += [f"  - {f}" for f in self.formulas]
        lines.append("Please list the truth value of each variable")
        # Final '\n' to mimic your sample
        return "\n".join(lines) + "\n"

    # ---------- solution ----------
    def solution_text(self) -> str:
        """
        Enumerate each variable on its own line:
            (1) A is True
            (2) B is False
            …
        If no assignment exists, return the original UNSAT message.
        """
        if not self.assignment:
            return "No satisfying assignment exists."

        # Sort variables alphabetically for deterministic order.
        vars_sorted = sorted(self.assignment.keys())
        numbered = [
            f"({i}) {var} is {'True' if self.assignment[var] else 'False'}"
            for i, var in enumerate(vars_sorted, 1)
        ]
        return "\n".join(numbered) + "\n"

###############################################################################
# Batch generator
###############################################################################

def generate_dataset(
    outdir          : str  = "deduction_data",
    num_samples     : int  = 1000,
    unique_solution : bool = True,
    seed            : int  = 42,
):
    """
    Create 5 curriculum files:
        outdir/d1.jsonl   … outdir/d5.jsonl
    Each contains *num_samples* satisfiable puzzles in the format you specified.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)

    for lvl in range(1, 6):                          # difficulty 1 → 5
        sampler = NestedLogicPuzzleSampler(difficulty=lvl, seed=rng.randint(0, 1000000))
        sample_fn = sampler.sample_unique if unique_solution else sampler.sample_valid
        puzzles   = sample_fn(num_samples)

        # pack rows
        rows = []
        for idx, (formulas, assignment) in enumerate(puzzles):
            fmt = PuzzleFormatter(formulas, assignment)
            rows.append({
                "index"        : idx,
                "puzzle_text"  : fmt.puzzle_text(),
                "solution_text": fmt.solution_text()
            })

        outpath = os.path.join(outdir, f"d{lvl}.jsonl")
        write_jsonl(outpath, rows)
        print(f"[d{lvl}] wrote {len(rows):,} puzzles ➜ {outpath}")

###############################################################################
# CLI entry point – generates all five files with 1 000 samples each
###############################################################################
if __name__ == "__main__":
    init_seed(42)

    generate_dataset(
        outdir          = "deduction_data",
        num_samples     = 1000,   # exactly 1 000 per level
        unique_solution = True,    # keep True if you want a single answer per puzzle
        seed            = 42,
    )
