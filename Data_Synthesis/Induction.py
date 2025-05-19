import random
import string
import json

class VariedPuzzleGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def generate_sequence_puzzle(self, level):
        """
        Randomly pick one sequence-generation pattern based on difficulty level.
        """
        patterns = {
            1: [self._seq_arithmetic, self._seq_geometric, self._seq_alternating_add_sub],
            2: [self._seq_two_step_add, self._seq_alt_mul_div, self._seq_missing_term],
            3: [lambda lvl: self._multi_step_random(lvl, steps=3)],
            4: [lambda lvl: self._multi_step_random(lvl, steps=4)],
            5: [lambda lvl: self._multi_step_random(lvl, steps=5)],
            6: [lambda lvl: self._multi_step_random(lvl, steps=6)],
            7: [lambda lvl: self._multi_step_random(lvl, steps=7)]

        }
        pattern_func = self.rng.choice(patterns[level])
        return pattern_func(level)

    def _seq_arithmetic(self, level):
        """Simple arithmetic progression: last item replaced by '?'."""
        length = self.rng.randint(5, 8)
        start = self.rng.randint(1, 5)
        step = self.rng.randint(1, 4)
        seq = [start + i * step for i in range(length)]  # all numeric
        complete_seq = seq[:]  # keep a copy to show the full numeric array

        # puzzle version: last item is '?'
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],              # correct numeric answer
            'complete_sequence': complete_seq
        }

    def _seq_geometric(self, level):
        """Simple geometric progression: last item replaced by '?'."""
        length = self.rng.randint(5, 8)
        start = self.rng.randint(1, 3)
        ratio = self.rng.randint(2, 4)
        seq = [start * (ratio ** i) for i in range(length)]
        complete_seq = seq[:]

        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],
            'complete_sequence': complete_seq
        }

    def _seq_alternating_add_sub(self, level):
        """
        Alternate add/sub, e.g. start, start+a, next - b, next + a, ...
        Replace the last item with '?'.
        """
        length = self.rng.randint(5, 7)
        start = self.rng.randint(1, 10)
        add_val = self.rng.randint(1, 3)
        sub_val = self.rng.randint(1, 3)

        seq = [start]
        for i in range(1, length):
            if i % 2 == 0:  # even index
                seq.append(seq[-1] + add_val)
            else:           # odd index
                seq.append(seq[-1] - sub_val)

        complete_seq = seq[:]
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],
            'complete_sequence': complete_seq
        }

    def _seq_two_step_add(self, level):
        """
        Pattern: repeating (+a, +b).
        Replace last item with '?'.
        """
        length = self.rng.randint(6, 9)
        start = self.rng.randint(1, 5)
        a = self.rng.randint(1, 4)
        b = self.rng.randint(1, 4)

        seq = [start]
        for i in range(1, length):
            if i % 2 == 0:
                seq.append(seq[-1] + a)
            else:
                seq.append(seq[-1] + b)

        complete_seq = seq[:]
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],
            'complete_sequence': complete_seq
        }

    def _seq_alt_mul_div(self, level):
        """
        Alternate multiply and integer-divide.
        Replace last item with '?'.
        """
        length = self.rng.randint(6, 9)
        cur = self.rng.randint(2, 5)
        seq = [cur]

        mul = self.rng.randint(2, 4)
        div = self.rng.randint(2, 4)

        for i in range(1, length):
            if i % 2 == 0:
                cur = cur * mul
            else:
                cur = max(1, cur // div)
            seq.append(cur)

        complete_seq = seq[:]
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],
            'complete_sequence': complete_seq
        }

    def _seq_missing_term(self, level):
        """
        Randomly remove one item in the *middle*, replacing it with '?'. 
        So the last item is still numeric. 
        We record the complete sequence in 'complete_sequence' 
        and the puzzle version in the question.
        """
        length = self.rng.randint(6, 8)
        start = self.rng.randint(1, 5)
        step = self.rng.randint(1, 4)
        seq_original = [start + i * step for i in range(length)]

        idx = self.rng.randint(1, length - 2)
        ans = seq_original[idx]

        puzzle_seq = seq_original[:]
        puzzle_seq[idx] = '?'

        question = (
            f"Given the following sequence with a missing term,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': ans,
            'complete_sequence': seq_original
        }

    def _multi_step_random(self, level, steps):
        """
        Multi-step repeating pattern, e.g. step1= +2, step2= -3, step3= *2, ...
        We'll fill a sequence of length 7~11, then replace last with '?'.
        """
        length = self.rng.randint(7, 11)
        cur = self.rng.randint(1, 5)
        seq = [cur]

        # build 'steps' different operations
        ops = []
        for _ in range(steps):
            op_type = self.rng.choice(['add','sub','mul'])
            if op_type == 'add':
                val = self.rng.randint(1,4)
                ops.append(lambda x,v=val: x + v)
            elif op_type == 'sub':
                val = self.rng.randint(1,4)
                ops.append(lambda x,v=val: x - v)
            else:
                val = self.rng.randint(2,4)
                ops.append(lambda x,v=val: x * v)

        for i in range(1, length):
            current_op = ops[(i - 1) % steps]
            cur = current_op(seq[-1])
            seq.append(cur)

        complete_seq = seq[:]
        puzzle_seq = [str(x) for x in seq]
        puzzle_seq[-1] = '?'

        question = (
            f"Given the following sequence,\n{puzzle_seq}\n"
            "What is the value at the question mark?"
        )
        return {
            'question': question,
            'answer': seq[-1],
            'complete_sequence': complete_seq
        }

    # -----------------------------
    # 2) Text puzzle generators
    # -----------------------------
    def generate_text_puzzle(self, level):
        """
        For 'text' type, we either do a Caesar cipher or a Substitution cipher.
        """
        if level <= 2:
            return self._text_caesar(level)
        else:
            return self._text_substitution(level)

    def _text_caesar(self, level):
        shift = self.rng.randint(1,5)
        plain = ''.join(self.rng.choices(string.ascii_uppercase, k=6))
        cipher = ''.join(chr((ord(c)-65+shift) % 26 + 65) for c in plain)

        question = (
            f"Caesar cipher with shift {shift}.\n"
            f"Encrypted text: {cipher}\n"
            "Decrypt to find the original 6-letter string."
        )
        return {
            'question': question,
            'answer': plain,
            'complete_sequence': None  # not a sequence puzzle
        }

    def _text_substitution(self, level):
        alpha = list(string.ascii_uppercase)
        perm = alpha[:]
        self.rng.shuffle(perm)
        mapping = dict(zip(alpha, perm))

        plain = ''.join(self.rng.choices(alpha, k=8))
        cipher = ''.join(mapping[c] for c in plain)

        question = (
            "Substitution cipher:\n"
            f"Encrypted text: {cipher}\n"
            "Decrypt to find the original 8-letter string."
        )
        return {
            'question': question,
            'answer': plain,
            'complete_sequence': None
        }

    def generate_logic_puzzle(self, level):
        """Generate a logic puzzle (placeholder example)."""
        return self._logic_formula(level)

    def _logic_formula(self, level):
        vars_ = [chr(65+i) for i in range(level)]
        expr = vars_[0]
        for v in vars_[1:]:
            expr = f"({expr} AND {v})"
        question = (
            "Determine if the following formula is a tautology, contradiction or satisfiable:\n"
            f"{expr}"
        )
        return {
            'question': question,
            'answer': 'TAUTOLOGY',
            'complete_sequence': None
        }

    def generate_puzzles(self, n, types, levels):
        """
        Generate `n` puzzles for each difficulty in `levels`,
        distributed among the puzzle `types`.
        """
        puzzles = []
        for level in levels:
            for p_type in types:
                # e.g. each type gets n/len(types) puzzles per level
                count_for_this_type = n // len(types)
                for i in range(count_for_this_type):
                    if p_type == 'sequence':
                        base = self.generate_sequence_puzzle(level)
                    elif p_type == 'text':
                        base = self.generate_text_puzzle(level)
                    else:
                        base = self.generate_logic_puzzle(level)

                    puzzles.append({
                        "problem_id": f"{level}-{p_type[:3]}-{i+1}",
                        "difficulty": level,
                        "puzzle_text": base['question'],
                        "solution_text": self._to_numeric(base['answer']),  # store numeric
                        "ability": p_type,
                        # new: store the full numeric sequence, if any
                        "complete_sequence": base.get('complete_sequence')
                    })
        return puzzles

    def save_jsonl(self, puzzles, filename):
        """Save the puzzles in JSONL format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for p in puzzles:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')

    def _to_numeric(self, ans):
        """
        If 'ans' is int, just return it.
        Otherwise, convert to string and sum ord() for each char.
        """
        if isinstance(ans, int):
            return ans
        return sum(ord(c) for c in str(ans))


if __name__ == '__main__':
    # Example usage:
    gen = VariedPuzzleGenerator(seed=800)
    puzzle_types = ['sequence']
    for level in range(6, 8):
        puzzles = gen.generate_puzzles(n=2000, types=puzzle_types, levels=[level])
        gen.save_jsonl(puzzles, f'new_squence_puzzles_level{level}.jsonl')
