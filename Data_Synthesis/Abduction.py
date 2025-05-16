import json
import random
import concurrent.futures


def tokenize_expr(expr_str):

    tokens = []
    i = 0
    while i < len(expr_str):
        ch = expr_str[i]

        if ch.isspace():
            i += 1
            continue

        if ch in ('(', ')'):
            tokens.append(ch)
            i += 1
        else:

            if ch.isalpha():
                start = i
                while i < len(expr_str) and expr_str[i].isalpha():
                    i += 1
                word = expr_str[start:i]
                word = word.upper()
                tokens.append(word)
            else:
                raise ValueError(f"Invalid character '{ch}' in expression: {expr_str}")
    return tokens



class TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        if self.pos < len(self.tokens):
            current = self.tokens[self.pos]
            self.pos += 1
            return current
        return None

    def expect(self, token):
        if self.peek() == token:
            self.advance()
        else:
            raise ValueError(f"Expecting '{token}', but got '{self.peek()}'")


def parse_expression(tokens):
    stream = TokenStream(tokens)
    node = parse_or_expr(stream)

    if stream.peek() is not None:
        raise ValueError(f"Extra tokens after valid expression: {stream.peek()}")
    return node

def parse_or_expr(stream):
    """
    OR_EXPR -> AND_EXPR { "OR" AND_EXPR }
    """
    left_node = parse_and_expr(stream)

    while True:
        if stream.peek() == "OR":
            op = stream.advance()  # 消耗 OR
            right_node = parse_and_expr(stream)
            left_node = ("OR", left_node, right_node)
        else:
            break

    return left_node

def parse_and_expr(stream):
    """
    AND_EXPR -> NOT_EXPR { "AND" NOT_EXPR }
    """
    left_node = parse_not_expr(stream)

    while True:
        if stream.peek() == "AND":
            op = stream.advance()  
            right_node = parse_not_expr(stream)
            left_node = ("AND", left_node, right_node)
        else:
            break

    return left_node

def parse_not_expr(stream):
    """
    NOT_EXPR -> { "NOT" } FACTOR
    """
    not_count = 0
    while stream.peek() == "NOT":
        stream.advance()
        not_count += 1

    factor_node = parse_factor(stream)

    if not_count % 2 == 1:
        return ("NOT", factor_node)
    else:
        return factor_node

def parse_factor(stream):
    """
    FACTOR -> IDENT | "(" EXPR ")"
    IDENT -> 大写字母串
    """
    token = stream.peek()
    if token == "(":
        # Consume '('
        stream.advance()
        subexpr = parse_or_expr(stream)
        # Expect ')'
        stream.expect(")")
        return subexpr
    else:
        # IDENT
        if token is None:
            raise ValueError("Unexpected end of tokens while expecting an IDENT or '('.")
        if not token.isalpha():
            raise ValueError(f"Expecting IDENT or '(', but got '{token}'.")
        # Consume the ident
        ident = stream.advance()
        return ("ATOM", ident)


def build_eval_func(ast_node):
    """
    把 parse_xxx 得到的语法树 (tuple 结构)，转成一个
    `lambda valuation: bool` 的可执行函数。
    - valuation: dict, 原子 -> bool
    """

    ntype = ast_node[0]

    if ntype == "ATOM":
        # ast_node = ("ATOM", "A")
        atom_name = ast_node[1]
        return lambda v: v[atom_name]

    elif ntype == "NOT":
        # ast_node = ("NOT", child)
        child = ast_node[1]
        child_func = build_eval_func(child)
        return lambda v: not child_func(v)

    elif ntype == "AND":
        # ast_node = ("AND", left, right)
        left_child = ast_node[1]
        right_child = ast_node[2]
        left_func = build_eval_func(left_child)
        right_func = build_eval_func(right_child)
        return lambda v: left_func(v) and right_func(v)

    elif ntype == "OR":
        # ast_node = ("OR", left, right)
        left_child = ast_node[1]
        right_child = ast_node[2]
        left_func = build_eval_func(left_child)
        right_func = build_eval_func(right_child)
        return lambda v: left_func(v) or right_func(v)

    else:
        raise ValueError(f"Unknown AST node type: {ntype}")


def parse_premise(premise_str):

    if "=>" not in premise_str:
        raise ValueError(f"Missing '=>' in premise: {premise_str}")
    left_side, right_side = premise_str.split("=>", 1)
    left_side = left_side.strip()
    right_side = right_side.strip()

    left_tokens = tokenize_expr(left_side)
    left_ast = parse_expression(left_tokens)
    left_func = build_eval_func(left_ast)

    right_tokens = tokenize_expr(right_side)
    right_ast = parse_expression(right_tokens)
    right_func = build_eval_func(right_ast)

    def premise_func(valuation):
        return (not left_func(valuation)) or right_func(valuation)
    return premise_func

def extract_atoms_from_premise(premise_str):

    if "=>" not in premise_str:
        return []
    left_side, right_side = premise_str.split("=>", 1)
    tokens_left = tokenize_expr(left_side)
    tokens_right = tokenize_expr(right_side)
    all_toks = tokens_left + tokens_right
    # 过滤掉保留字
    ignore_set = {"AND", "OR", "NOT", "(", ")"}
    atoms = [t for t in all_toks if t not in ignore_set]
    return list(set(atoms))



def check_puzzle_consistency_and_goals(puzzle):
    premises = puzzle["premises"]
    known_atoms = puzzle["known_atoms"]
    goals = puzzle["goals"]
    reachable_goals = puzzle["reachable_goals"]
    unreachable_goals = puzzle["unreachable_goals"]

    all_atoms_set = set()
    for pm in premises:
        pm_atoms = extract_atoms_from_premise(pm)
        all_atoms_set.update(pm_atoms)

    all_atoms_set.update(known_atoms)
    all_atoms_set.update(goals)
    all_atoms_list = sorted(list(all_atoms_set))
    N = len(all_atoms_list)
    if N > 20:
        return (False, f"Too many atoms ({N}), brute force not feasible.")

    premise_funcs = []
    try:
        for pm in premises:
            f = parse_premise(pm)
            premise_funcs.append(f)
    except ValueError as e:
        return (False, f"Parsing error: {e}")

    satisfiers = []
    for mask in range(1 << N):
        valuation = {}
        for i, atom in enumerate(all_atoms_list):
            bit = (mask >> i) & 1
            valuation[atom] = (bit == 1)

        if any(not valuation[ka] for ka in known_atoms):
            continue

        if all(func(valuation) for func in premise_funcs):
            satisfiers.append(valuation)

    if not satisfiers:
        return (False, "No valuation satisfies all premises => puzzle is inconsistent.")

    for rg in reachable_goals:
        for val in satisfiers:
            if not val[rg]:
                return (False, f"Reachable goal '{rg}' is false in some model => not truly reachable.")

    for ug in unreachable_goals:
        if all(val[ug] for val in satisfiers):
            return (False, f"Unreachable goal '{ug}' is true in all models => it is actually reachable.")

    return (True, "Puzzle passed checks.")



LIMITED_LETTERS = "ABCDEFGHIJKLMNO"

def _make_random_atom(length=None):
    if length is None:
        length = random.randint(1, 2)
    return "".join(random.choice(LIMITED_LETTERS) for _ in range(length))

def _random_expr(available_atoms, max_depth=2):
    """
    不做太深的生成，仅做示例
    """
    if max_depth <= 1 or random.random() < 0.4:
        atom = random.choice(available_atoms)
        if random.random() < 0.5:
            return atom
        else:
            return f"(NOT {atom})"
    else:
        left = _random_expr(available_atoms, max_depth-1)
        right = _random_expr(available_atoms, max_depth-1)
        op = random.choice(["AND", "OR"])
        return f"({left} {op} {right})"

def _random_complex_premise(available_atoms, max_depth=2):
    left_expr = _random_expr(available_atoms, max_depth)
    conclusion = random.choice(available_atoms)
    return f"({left_expr}) => {conclusion}"

def _generate_advanced_subchain(target, depth=3, cycle_probability=0.2, existing_symbols=None):
    if existing_symbols is None:
        existing_symbols = []

    premises = []
    used_atoms = []
    mid_symbols = []
    current_symbol = target

    for _ in range(depth):
        if mid_symbols and random.random() < cycle_probability:
            chosen_symbol = random.choice(mid_symbols)
        else:
            chosen_symbol = _make_random_atom()
            mid_symbols.append(chosen_symbol)

        candidate_atoms = existing_symbols + used_atoms + mid_symbols
        if current_symbol not in candidate_atoms:
            candidate_atoms.append(current_symbol)

        left_expr = _random_expr(candidate_atoms, max_depth=2)
        premise = f"({left_expr}) => {current_symbol}"
        premises.append(premise)

        tokens = tokenize_expr(left_expr)
        for t in tokens:
            if t not in ("AND", "OR", "NOT", "(", ")") and t not in used_atoms:
                used_atoms.append(t)

        current_symbol = chosen_symbol

    return premises, used_atoms, mid_symbols

def generate_advanced_logic_problem(
    problem_id=0,
    num_goals=3,
    reachable_count=1,
    chain_depth=2,
    distractor_count=3,
    cycle_probability=0.2
):
    goals = [_make_random_atom() for _ in range(num_goals)]
    random.shuffle(goals)

    reachable_goals = goals[:reachable_count]
    unreachable_goals = goals[reachable_count:]

    premises = []
    known_atoms = set()
    all_mid_symbols = set()

    for rg in reachable_goals:
        sub_prem, used_atoms, mid_syms = _generate_advanced_subchain(
            rg, chain_depth, cycle_probability
        )
        premises.extend(sub_prem)
        known_atoms.update(used_atoms)
        all_mid_symbols.update(mid_syms)

    for ug in unreachable_goals:
        sub_prem, used_atoms, mid_syms = _generate_advanced_subchain(
            ug, chain_depth, cycle_probability
        )
        keep_count = max(0, len(sub_prem) - random.randint(1,2))
        partial_premises = sub_prem[:keep_count]
        premises.extend(partial_premises)
        drop_count = min(len(used_atoms), random.randint(1,2))
        used_atoms_kept = used_atoms[:-drop_count] if drop_count > 0 else used_atoms
        known_atoms.update(used_atoms_kept)
        all_mid_symbols.update(mid_syms)

    all_known_list = list(known_atoms.union(all_mid_symbols))
    for _ in range(distractor_count):
        if random.random() < 0.3:
            new_atom = _make_random_atom()
            all_known_list.append(new_atom)
        premise = _random_complex_premise(all_known_list, max_depth=2)
        premises.append(premise)

    premises = list(set(premises))
    known_atoms = list(set(known_atoms))

    puzzle = {
        "problem_id": f"ADV_{problem_id}",
        "premises": premises,
        "known_atoms": known_atoms,
        "goals": goals,
        "reachable_goals": reachable_goals,
        "unreachable_goals": unreachable_goals,
    }
    return puzzle

def generate_valid_puzzle(max_tries=10000, **kwargs):
    """
    多次随机生成, 并检验 puzzle. 若成功返回 puzzle, 否则返回 None
    """
    for attempt in range(max_tries):
        puzzle = generate_advanced_logic_problem(**kwargs)
        ok, explanation = check_puzzle_consistency_and_goals(puzzle)
        if ok:
            puzzle["explanation"] = explanation
            return puzzle
    return None



def generate_puzzles_for_difficulty_parallel(difficulty, n=5):

    if difficulty == 1:
        chain_depth_range = (2, 3)
        num_goals_range = (1, 2)
        distractor_count_range = (3, 5)
        cycle_prob_range = (0.0, 0.2)
    elif difficulty == 2:
        chain_depth_range = (3, 4)
        num_goals_range = (2, 3)
        distractor_count_range = (5, 7)
        cycle_prob_range = (0.1, 0.25)
    elif difficulty == 3:
        chain_depth_range = (4, 5)
        num_goals_range = (2, 3)
        distractor_count_range = (7, 9)
        cycle_prob_range = (0.15, 0.3)
    elif difficulty == 4:
        chain_depth_range = (5, 6)
        num_goals_range = (3, 4)
        distractor_count_range = (8, 10)
        cycle_prob_range = (0.2, 0.35)
    else:  # difficulty == 5
        chain_depth_range = (6, 7)
        num_goals_range = (3, 4)
        distractor_count_range = (10, 12)
        cycle_prob_range = (0.25, 0.4)



    filename = f"v3_puzzles_difficulty_{difficulty}.jsonl"
    count_generated = 0
    with open(filename, "w", encoding="utf-8") as f, \
         concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:

        futures = []
        while count_generated < n:
            for _ in range(4):
                chain_depth = random.randint(*chain_depth_range)
                puzzle_future = executor.submit(
                    generate_valid_puzzle,
                    problem_id=count_generated,
                    num_goals=3,
                    reachable_count=1,
                    chain_depth=chain_depth,
                    distractor_count=random.randint(*distractor_count_range),
                    cycle_probability=0.2
                )
                futures.append(puzzle_future)

            for future in concurrent.futures.as_completed(futures):
                puzzle = future.result()
                if puzzle is not None:
                    puzzle["difficulty"] = difficulty
                    f.write(json.dumps(puzzle, ensure_ascii=False) + "\n")
                    count_generated += 1
                    print(f"Generated puzzle {count_generated}/{n}")
                    if count_generated >= n:
                        break
            futures.clear()

    print(f"== Finished difficulty {difficulty}, total {count_generated} puzzles saved to {filename}.")


if __name__ == "__main__":
    for diff in range(1,6):
        generate_puzzles_for_difficulty_parallel(diff, n=100)

