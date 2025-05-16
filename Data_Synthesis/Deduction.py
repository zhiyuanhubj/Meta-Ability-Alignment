
###############################################################################
# Advanced Logic Puzzle Sampler
#   - Supports multiple "modes" of puzzle generation for complexity:
#       1) nested formulas with various operators
#       2) random CNF (k-SAT style)
#       3) cardinal constraints
###############################################################################
class AdvancedLogicPuzzleSampler:
    """
    A sampler that can generate harder logic puzzles:
      - nested formulas (multi-depth, multiple operators)
      - CNF-based k-SAT
      - cardinal constraints
    And tries to find a satisfying assignment (if any) via brute force (only feasible for small n).
    """

    def __init__(
        self,
        seed: int = 42,
        n_vars: int = 6,
        max_depth: int = 3,
        puzzle_mode: str = "nested",
        max_clauses: int = 8,
        clause_size: int = 3,
        cardinal_k: int = 2,
        knk_num_people: int = 3
    ):
        """
        Args:
            seed: random seed
            n_vars: number of boolean variables (for formula-based modes)
            max_depth: max depth of nested formula
            puzzle_mode: "nested"
            max_clauses: used if puzzle_mode="cnf"
            clause_size: how many literals per clause if puzzle_mode="cnf"
            cardinal_k: 'exactly cardinal_k variables must be true' if puzzle_mode="cardinal"
        """
        self.rng = random.Random(seed)
        self.n_vars = n_vars
        self.max_depth = max_depth
        self.puzzle_mode = puzzle_mode
        self.max_clauses = max_clauses
        self.clause_size = clause_size
        self.cardinal_k = cardinal_k
        self.knk_num_people = knk_num_people

        # We'll store variables as strings "A", "B", "C", ...
        # If n_vars > 26, you could adapt to double letters, etc.
        self.var_list = [chr(ord('A') + i) for i in range(n_vars)]

    ##############################
    # FORMULA GENERATION LOGIC
    ##############################
    def _random_subformula(self, depth: int = 0) -> str:
        """
        Recursively generate a nested formula with random operators and possible negations.
        Operators included: →, ∧, ∨, ↔, ⊕
        """
        if depth >= self.max_depth:
            # Return a single variable or negation
            var = self.rng.choice(self.var_list)
            if self.rng.random() < 0.4:
                return f"¬{var}"
            else:
                return var
        else:
            # Decide if unary or binary
            use_unary = (self.rng.random() < 0.2)  # 20% chance
            if use_unary:
                sub = self._random_subformula(depth + 1)
                return f"¬({sub})"
            else:
                # binary operator
                op = self.rng.choice(["→", "∧", "∨", "↔", "⊕"])
                left = self._random_subformula(depth + 1)
                right = self._random_subformula(depth + 1)
                return f"({left} {op} {right})"

    def _random_nested_formula(self) -> str:
        return self._random_subformula(depth=0)

    def _random_cnf_formula(self) -> str:
        """
        Generate a CNF formula with 'max_clauses' clauses, each containing 'clause_size' literals.
        Example: (A ∨ ¬B ∨ C) ∧ (¬A ∨ B ∨ D) ...
        """
        clauses = []
        for _ in range(self.max_clauses):
            chosen_vars = self.rng.sample(self.var_list, k=min(self.clause_size, self.n_vars))
            literals = []
            for v in chosen_vars:
                if self.rng.random() < 0.5:
                    literals.append(v)
                else:
                    literals.append(f"¬{v}")
            clause_str = "(" + " ∨ ".join(literals) + ")"
            clauses.append(clause_str)
        formula = " ∧ ".join(clauses)
        return formula

    def _random_cardinality_constraint(self) -> str:
        """
        For demonstration, say "EXACTLY K of [all variables]"
        """
        return f"EXACTLY {self.cardinal_k} of {self.var_list}"


    def generate_puzzle_spec(self) -> Dict:
        """
        Depending on self.puzzle_mode, generate a puzzle with either:
          - nested formulas
          - CNF formula
          - cardinal constraint
        """
        if self.puzzle_mode == "nested":
            n_formulas = self.rng.randint(3, 6)
            formulas = [self._random_nested_formula() for _ in range(n_formulas)]
            puzzle_spec = {
                "mode": "nested",
                "formulas": formulas
            }
        elif self.puzzle_mode == "cnf":
            formula = self._random_cnf_formula()
            puzzle_spec = {
                "mode": "cnf",
                "formula": formula
            }
        elif self.puzzle_mode == "cardinal":
            constraint = self._random_cardinality_constraint()
            n_formulas = self.rng.randint(1, 3)
            base_formulas = [self._random_nested_formula() for _ in range(n_formulas)]
            puzzle_spec = {
                "mode": "cardinal",
                "constraint": constraint,
                "formulas": base_formulas
            }
        else:
            raise ValueError(f"Unknown puzzle_mode: {self.puzzle_mode}")

        return puzzle_spec

    ##############################
    # EVALUATION / SOLVING
    ##############################
    def _eval_bool_expr(self, expr: str, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate a formula that may contain: ¬, ∧, ∨, →, ↔, ⊕, parentheses, variables like 'A'.
        """
        tokens = self._tokenize(expr)
        ast, _ = self._parse_expr(tokens, 0)
        val = self._eval_ast(ast, assignment)
        return val

    def _eval_ast(self, node, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate an AST node with the given assignment.
        AST node forms:
          ("VAR", varname)
          ("NOT", subnode)
          ("BINOP", op, left_node, right_node)
        """
        nodetype = node[0]
        if nodetype == "VAR":
            return assignment[node[1]]
        elif nodetype == "NOT":
            return not self._eval_ast(node[1], assignment)
        elif nodetype == "BINOP":
            op = node[1]
            left_val = self._eval_ast(node[2], assignment)
            right_val = self._eval_ast(node[3], assignment)
            if op == "∧":
                return left_val and right_val
            elif op == "∨":
                return left_val or right_val
            elif op == "→":
                return (not left_val) or right_val
            elif op == "↔":
                return (left_val == right_val)
            elif op == "⊕":
                return (left_val != right_val)
            else:
                raise ValueError(f"Unknown binary operator: {op}")
        else:
            raise ValueError(f"Unknown AST node type: {nodetype}")

    def _tokenize(self, expr: str) -> List[str]:
        """
        Split expression into tokens (parentheses, operators, variable names, etc.).
        """
        expr = expr.replace("(", " ( ").replace(")", " ) ")
        raw_tokens = expr.split()
        tokens = [t.strip() for t in raw_tokens if t.strip()]
        return tokens

    def _parse_expr(self, tokens: List[str], pos: int = 0):
        """
        Convert tokens into an AST (recursive).
        Return (ast, new_pos).
        """
        def parse_rec(pos: int):
            if tokens[pos] == '(':
                pos += 1  # skip '('
                left_node, pos = parse_rec(pos)
                if tokens[pos] == ')':
                    # only one subexpression in parentheses
                    pos += 1
                    return left_node, pos
                else:
                    # must be a binary operator
                    op = tokens[pos]
                    pos += 1
                    right_node, pos = parse_rec(pos)
                    if tokens[pos] != ')':
                        raise ValueError("Missing closing parenthesis in subexpr.")
                    pos += 1
                    return ("BINOP", op, left_node, right_node), pos
            else:
                # could be '¬', '¬A', or a var
                token = tokens[pos]
                if token.startswith('¬') and len(token) == 1:
                    # just '¬'
                    pos += 1
                    sub_expr, pos = parse_rec(pos)
                    return ("NOT", sub_expr), pos
                elif token.startswith('¬') and len(token) > 1:
                    # e.g. '¬A'
                    varname = token[1:]
                    pos += 1
                    return ("NOT", ("VAR", varname)), pos
                else:
                    # normal variable
                    pos += 1
                    return ("VAR", token), pos

        ast, newpos = parse_rec(pos)
        return ast, newpos

    def _satisfies_all(self, puzzle_spec: Dict, assignment: Dict[str, bool]) -> bool:
        """
        Check if the assignment satisfies the puzzle (depending on puzzle mode).
        """
        mode = puzzle_spec["mode"]
        if mode == "nested":
            for fm in puzzle_spec["formulas"]:
                if not self._eval_bool_expr(fm, assignment):
                    return False
            return True
        elif mode == "cnf":
            formula = puzzle_spec["formula"]
            return self._eval_bool_expr(formula, assignment)
        elif mode == "cardinal":
            # check base_formulas
            for fm in puzzle_spec["formulas"]:
                if not self._eval_bool_expr(fm, assignment):
                    return False
            # EXACTLY K constraint
            count_true = sum(assignment[v] for v in self.var_list)
            return (count_true == self.cardinal_k)
        else:
            raise ValueError(f"Unknown puzzle_mode: {mode}")

    def _find_satisfying_assignment(self, puzzle_spec: Dict) -> Optional[Dict[str, bool]]:
        """
        Brute force check for *one* satisfying assignment. Return it or None.
        """
        mode = puzzle_spec["mode"]
        if mode == "knights_knaves":
            n_people = puzzle_spec["num_people"]
            var_list = [f"P{i}" for i in range(n_people)]
        else:
            var_list = self.var_list

        n = len(var_list)
        for bits in range(1 << n):
            assign = {}
            for i in range(n):
                assign[var_list[i]] = bool(bits & (1 << i))
            if self._satisfies_all(puzzle_spec, assign):
                return assign
        return None

    def _find_all_solutions(self, puzzle_spec: Dict) -> List[Dict[str, bool]]:
        """
        Brute force check all 2^n assignments. Return *all* solutions.
        If n is large, this is slow. But for small n, it's feasible.
        """
        mode = puzzle_spec["mode"]
        if mode == "knights_knaves":
            n_people = puzzle_spec["num_people"]
            var_list = [f"P{i}" for i in range(n_people)]
        else:
            var_list = self.var_list

        solutions = []
        n = len(var_list)
        for bits in range(1 << n):
            assign = {}
            for i in range(n):
                assign[var_list[i]] = bool(bits & (1 << i))
            if self._satisfies_all(puzzle_spec, assign):
                solutions.append(assign)
        return solutions

    ############################################################################
    # SAMPLE FUNCTIONS
    ############################################################################
    def sample_valid_puzzles(self, n_problems: int) -> List[Dict]:
        """
        Generate puzzle specs that have *at least one* satisfying assignment.
        (No uniqueness guarantee.)
        """
        puzzles = []
        while len(puzzles) < n_problems:
            spec = self.generate_puzzle_spec()
            assignment = self._find_satisfying_assignment(spec)
            if assignment is not None:
                puzzles.append({
                    "puzzle_spec": spec,
                    "valid": True,
                    "assignment": assignment
                })
        return puzzles

    def sample_invalid_puzzles(self, n_problems: int) -> List[Dict]:
        """
        Generate puzzle specs that are guaranteed unsatisfiable, by injecting contradictions.
        """
        puzzles = []
        while len(puzzles) < n_problems:
            spec = self.generate_puzzle_spec()
            # Force contradiction:
            if spec["mode"] in ["nested", "cnf"]:
                var = self.rng.choice(self.var_list)
                contradiction = f"({var} ∧ ¬{var})"
                if spec["mode"] == "nested":
                    spec["formulas"].append(contradiction)
                else:  # cnf
                    spec["formula"] = "(" + spec["formula"] + ") ∧ " + f"({var} ∧ ¬{var})"
            elif spec["mode"] == "cardinal":
                spec["constraint"] = f"EXACTLY {self.n_vars + 1} of {self.var_list}"
            elif spec["mode"] == "knights_knaves":
                spec["statements"] = ["P0 says: P0 is knight", "P0 says: P0 is knave"]
                spec["num_people"] = 1

            assignment = self._find_satisfying_assignment(spec)
            if assignment is None:
                puzzles.append({
                    "puzzle_spec": spec,
                    "valid": False,
                    "assignment": None
                })
        return puzzles

    def sample_unique_solution_puzzles(self, n_problems: int) -> List[Dict]:
        """
        Generate puzzle specs that have *exactly one* satisfying assignment.
        We'll use _find_all_solutions and only keep those with len(...) == 1.
        """
        puzzles = []
        while len(puzzles) < n_problems:
            spec = self.generate_puzzle_spec()
            all_solutions = self._find_all_solutions(spec)
            if len(all_solutions) == 1:
                # exactly one unique solution
                puzzles.append({
                    "puzzle_spec": spec,
                    "valid": True,
                    "assignment": all_solutions[0]  # the unique solution
                })
        return puzzles

###############################################################################
# Puzzle Formatter
###############################################################################
class AdvancedLogicPuzzleFormatter:
    """
    Convert puzzle specs into a textual puzzle statement + solution.
    """

    def __init__(self, puzzle_spec: Dict, assignment: Optional[Dict[str, bool]]):
        self.puzzle_spec = puzzle_spec
        self.assignment = assignment

    def format_puzzle(self) -> Dict:
        puzzle_text = ""
        solution_text = ""

        mode = self.puzzle_spec["mode"]
        if mode == "nested":
            puzzle_text = "Below are some nested formulas:\n"
            for fm in self.puzzle_spec["formulas"]:
                puzzle_text += f"  - {fm}\n"
            puzzle_text += "Please list the truth value of each variable\n"

            if self.assignment:
                assign_str = ", ".join(f"{k}={v}" for k,v in self.assignment.items())
                pairs = [s.strip() for s in assign_str.split(',')]
                result = ""
                for i, pair in enumerate(pairs, 1):
                    key, value = pair.split("=")
                    result += f"({i}) {key} is {value}\n"
                solution_text = f"{result}"
            else:
                solution_text = "No satisfying assignment exists."

        elif mode == "cnf":
            puzzle_text = f"Consider the CNF formula:\n  {self.puzzle_spec['formula']}\nDecide if it is satisfiable.\n"
            if self.assignment:
                assign_str = ", ".join(f"{k}={v}" for k,v in self.assignment.items())
                solution_text = f"Satisfiable. One solution is: {assign_str}"
            else:
                solution_text = "Unsatisfiable."

        elif mode == "cardinal":
            puzzle_text = "We have the following base formulas:\n"
            for fm in self.puzzle_spec["formulas"]:
                puzzle_text += f"  - {fm}\n"
            puzzle_text += f"Also, there's a cardinality constraint: {self.puzzle_spec['constraint']}\n"
            puzzle_text += "Decide if these can all be satisfied.\n"
            if self.assignment:
                assign_str = ", ".join(f"{k}={v}" for k,v in self.assignment.items())
                solution_text = f"Satisfiable. One solution is: {assign_str}"
            else:
                solution_text = "Unsatisfiable."

        else:
            puzzle_text = "Unknown puzzle mode."
            solution_text = "No solution."

        return {
            "puzzle_text": puzzle_text,
            "solution_text": solution_text
        }

###############################################################################
# Example usage: Generate data and write to JSONL
###############################################################################
def generate_data(
    mode: str = "nested",
    n_vars: int = 6,
    max_depth: int = 3,
    max_clauses: int = 8,
    clause_size: int = 3,
    cardinal_k: int = 2,
    knk_num_people: int = 3,
    num_valid: int = 2,
    num_invalid: int = 2,
    seed: int = 42,
    outdir: str = "advanced_logic_data",
    unique_solution: bool = False
):
    """
    Generate a mixture of valid + invalid puzzles in a chosen mode, 
    then save them as two JSONL files: {mode}_valid.jsonl and {mode}_invalid.jsonl

    If unique_solution=True, then "valid" puzzles are forced to have exactly one solution
    instead of "at least one".
    """
    sampler = AdvancedLogicPuzzleSampler(
        seed=seed,
        n_vars=n_vars,
        max_depth=max_depth,
        puzzle_mode=mode,
        max_clauses=max_clauses,
        clause_size=clause_size,
        cardinal_k=cardinal_k,
        knk_num_people=knk_num_people
    )

    # 1) Get valid puzzles (either at least one solution or exactly one solution)
    if unique_solution:
        valid_puzzles = sampler.sample_unique_solution_puzzles(num_valid)
    else:
        valid_puzzles = sampler.sample_valid_puzzles(num_valid)

    # 2) Get invalid puzzles (unsatisfiable)
    invalid_puzzles = sampler.sample_invalid_puzzles(num_invalid)

    # 3) Format and save
    valid_data = []
    for i, vp in enumerate(valid_puzzles):
        formatter = AdvancedLogicPuzzleFormatter(vp["puzzle_spec"], vp["assignment"])
        formatted = formatter.format_puzzle()
        item = {
            "index": i,
            "mode": mode,
            "valid": True,
            "puzzle_text": formatted["puzzle_text"],
            "solution_text": formatted["solution_text"],
            "puzzle_spec": vp["puzzle_spec"]
        }
        valid_data.append(item)

    invalid_data = []
    for i, ip in enumerate(invalid_puzzles):
        formatter = AdvancedLogicPuzzleFormatter(ip["puzzle_spec"], None)
        formatted = formatter.format_puzzle()
        item = {
            "index": i,
            "mode": mode,
            "valid": False,
            "puzzle_text": formatted["puzzle_text"],
            "solution_text": formatted["solution_text"],
            "puzzle_spec": ip["puzzle_spec"]
        }
        invalid_data.append(item)

    os.makedirs(outdir, exist_ok=True)
    write_jsonl(os.path.join(outdir, f"{mode}_valid.jsonl"), valid_data)
    write_jsonl(os.path.join(outdir, f"{mode}_invalid.jsonl"), invalid_data)
    print(f"[{mode}] Wrote {len(valid_data)} valid puzzles and {len(invalid_data)} invalid puzzles to {outdir}/")


###############################################################################
# MAIN (DEMO)
###############################################################################
if __name__ == "__main__":
    init_seed(42)

    # Example 1: Generate "nested" formula puzzles, with at least one solution
    # generate_data(
    #     mode="nested",
    #     n_vars=10,
    #     max_depth=3,
    #     num_valid=5,
    #     num_invalid=2,
    #     outdir="demo_data",
    #     unique_solution=False  # means "at least one solution"
    # )

    # Example 2: Generate "nested" formula puzzles, but now enforce EXACTLY ONE solution
    generate_data(
        mode="nested",
        n_vars=10,
        max_depth=4,
        num_valid=100,
        num_invalid=2,
        outdir="n_vars_10_max_depth_4",
        unique_solution=True
    )

