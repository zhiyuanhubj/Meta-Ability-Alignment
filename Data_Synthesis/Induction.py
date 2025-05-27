# Induction – masked-sequence completion Generator
#
# This script implements the *Induction* track (“masked-sequence completion”)
# from *Beyond ‘Aha!’: Toward Systematic Meta-Abilities Alignment in Large
# Reasoning Models*.  Each sample presents a numeric sequence following an
# implicit k-step rule and asks the solver to predict the final element “?”.
#
# ───────────  Curriculum mapping  ───────────
# Task Difficulty Level → k-step pattern
#   1         → 6-step cyclic rule   (was level 6 in the legacy code)
#   2         → 7-step cyclic rule
#   3         → 8-step cyclic rule
#   4         → 9-step cyclic rule
#   5         → 10-step cyclic rule
#
# The generator now focuses exclusively on the k-step cyclic rule
# family used for Induction Levels 1–5 (k = 6 … 10).
###############################################################################
from __future__ import annotations
import random, json
from typing import List, Dict, Any

###############################################################################
# Helper
###############################################################################
def save_jsonl(rows: List[Dict[str, Any]], path: str):
    """Save *rows* to *path* in JSON Lines format (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

###############################################################################
# Main generator
###############################################################################
class SequencePuzzleGenerator:
    """
    Rand-seeded generator for Induction task.

    Parameters
    ----------
    seed : int | None
        Random seed.  Leave `None` for nondeterministic behaviour.
    """

    # ────────── curriculum table ──────────
    CURRICULUM = {lvl: 5 + lvl for lvl in range(1, 6)}   # 1→6 steps … 5→10

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    # ------------------------------ public API ------------------------------
    def generate_puzzles(self, num: int, level: int) -> List[Dict[str, Any]]:
        """
        Generate *num* sequence-extrapolation puzzles for the given *level*.

        Returns a list of dicts, each ready for JSONL dumping.
        """
        puzzles = []
        for idx in range(1, num + 1):
            puzzle = self._multi_step_random(level)
            puzzles.append({
                "problem_id"      : f"{idx}",
                "difficulty"      : level,
                "puzzle_text"     : puzzle["question"],
                "solution_text"   : self._to_numeric(puzzle["answer"]),
                "complete_sequence": puzzle["complete_sequence"],
            })
        return puzzles

    # ------------------------------ core pattern ----------------------------
    def _multi_step_random(self, level: int) -> Dict[str, Any]:
        """
        k-step cyclic rule where k = CURRICULUM[level] (6 … 10).

        Operations are sampled from {+a, −b, ×c} and repeat every k steps.
        The final element is replaced by '?'.
        """
        steps = self.CURRICULUM[level]           # 6,7,8,9,10
        length = self.rng.randint(7, 11)         # visible length
        cur = self.rng.randint(1, 5)
        seq = [cur]

        # Build k distinct operations
        ops = []
        for _ in range(steps):
            op_type = self.rng.choice(("add", "sub", "mul"))
            if op_type == "add":
                val = self.rng.randint(1, 4)
                ops.append(lambda x, v=val: x + v)
            elif op_type == "sub":
                val = self.rng.randint(1, 4)
                ops.append(lambda x, v=val: x - v)
            else:  # mul
                val = self.rng.randint(2, 4)
                ops.append(lambda x, v=val: x * v)

        # Fill the sequence
        for i in range(1, length):
            cur = ops[(i - 1) % steps](seq[-1])
            seq.append(cur)

        # Prepare puzzle view
        complete_seq = seq[:]
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = "?"

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            "question"          : question,
            "answer"            : seq[-1],
            "complete_sequence" : complete_seq,
        }

    # ------------------------------ misc util -------------------------------
    @staticmethod
    def _to_numeric(ans: int | str) -> int:
        """
        Convert *ans* to the numeric format expected by downstream code:
        - integers stay unchanged;
        - strings become Σ ord(c).
        """
        if isinstance(ans, int):
            return ans
        return sum(ord(c) for c in ans)

###############################################################################
# CLI utility – generate 1 000 puzzles per level (L1-L5)
###############################################################################
if __name__ == "__main__":
    gen      = SequencePuzzleGenerator(seed=800)
    out_root = "induction_data"

    for lvl in range(1, 6):                            # L1 … L5
        puzzles = gen.generate_puzzles(num=1000, level=lvl)
        save_jsonl(puzzles, f"{out_root}/L{lvl}.jsonl")
        print(f"[L{lvl}] saved {len(puzzles):,} puzzles → {out_root}/L{lvl}.jsonl")
