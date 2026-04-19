# 03 · Expression DSL, Parser, and AST

The system uses a small **DSL** for alpha factor expressions, written
in a style borrowed from Qlib / WorldQuant Alpha-101. The DSL is
intentionally simple so the LLM can generate it reliably.

## The DSL at a glance

Expressions are built from four kinds of atoms, combined with
operators and parenthesized function calls.

| Atom | Example |
|---|---|
| Data variable | `$close`, `$volume`, `$net_foreign_val` |
| Numeric literal | `5`, `0.1`, `1e-8`, `-0.5` |
| Function call | `RANK($close)`, `TS_MEAN($return, 5)` |
| Sub-expression | `(A + B) / C` |

### Binary operators (in precedence order, low → high)

| Precedence | Operators | Semantics |
|---|---|---|
| 1 (lowest) | `?:` | Conditional (right-assoc): `cond ? a : b` |
| 2 | <code>&#124;&#124;</code>, <code>&#124;</code> | Logical OR |
| 3 | `&&`, `&` | Logical AND |
| 4 | `>`, `<`, `>=`, `<=`, `==`, `!=` | Comparison |
| 5 | `+`, `-` | Additive |
| 6 | `*`, `/` | Multiplicative |
| 7 (highest) | Unary `+`, `-` | Sign |

The single-char logical ops (`|`, `&`) work because `AND` / `OR` in
`function_lib` cast to bool before bitwise-and/or-ing.

### Unsupported characters

`check_for_invalid_operators` in `expression_manager/expr_parser.py`
rejects any combination of `= ! & | ^ ~ \` ` @ # % ; { } [ ] " ' \\`
that isn't one of the valid DSL operators. Parentheses must be
balanced.

## Two parsers for two purposes

The project maintains **two independent pyparsing grammars**:

| File | Output | Used by |
|---|---|---|
| `expression_manager/expr_parser.py` | Executable Python **code string** | `backtest/factor_executor.py` (runs the expression) |
| `expression_manager/factor_ast.py` | Typed `Node` **AST** | `training/factor_tool.py`, reward similarity |

Why two? Because at runtime we want to **execute** an expression against
`daily_pv.h5`; at reward time we want to **compare AST structures** of
two expressions to compute similarity.

### 1. The execution parser — `expr_parser.parse_expression`

Takes a DSL string and returns an equivalent Python string that
references `numpy`, `pandas`, and every function name exposed in
`expression_manager/function_lib.py`.

Key transformations:

- Arithmetic between two Series becomes an explicit function call:
  `A + B` → `ADD(A, B)`, `A - B` → `SUBTRACT(A, B)`, etc. (see
  `parse_arith_op`). This guarantees proper MultiIndex alignment and
  NaN propagation across the `(datetime, instrument)` grid.
- Arithmetic where either operand is a literal number is left inline:
  `A + 1e-8` → `A+1e-8`.
- `cond ? a : b` → `pd.Series(np.where(cond, a, b), index=cond.index)`
  (see `parse_conditional_expression`).
- Logical `&& / &` → `AND(...)`, `|| / |` → `OR(...)`.
- Variable names like `$close` are stripped of `$` and then
  word-boundary-replaced to handle Python keyword collisions — e.g.
  `$return` → `col_return` (see `factor_executor.execute_expression`,
  lines 128-158).
- `TRUE/FALSE/NULL/NAN` aliases map to Python equivalents.

Example:

```
Input : RANK(DELTA($open, 1) - DELTA($open, 1)) / (1e-8 + 1)
Output: RANK(SUBTRACT(DELTA(open,1),DELTA(open,1)))/(1e-8+1)
```

The parser uses `pyparsing.ParserElement.enablePackrat()` for
performance on deeply nested expressions, and raises `ParseException`
on malformed input.

### 2. The AST parser — `factor_ast.parse_expression`

Builds a typed tree of `Node` subclasses:

```python path=F:\projects\AlphaAgentEvo\expression_manager\factor_ast.py start=17
@dataclass
class Node: ...
@dataclass
class VarNode(Node): name: str
@dataclass
class NumberNode(Node): value: float
@dataclass
class FunctionNode(Node):
    name: str
    args: List[Node]
@dataclass
class BinaryOpNode(Node):
    op: str
    left: Node
    right: Node
@dataclass
class UnaryOpNode(Node):
    op: str
    operand: Node
@dataclass
class ConditionalNode(Node):
    condition: Node
    true_expr: Node
    false_expr: Node
```

Key AST helpers (all take string expressions):

| Function | Purpose |
|---|---|
| `parse_expression(s) → Node` | Build the AST |
| `count_all_nodes(s) → int` | Total node count (used in similarity) |
| `count_free_args(s) → int` | Number of `NumberNode` instances |
| `count_unique_vars(s) → int` | Unique `$`-prefixed variables |
| `find_largest_common_subtree(root1, root2)` | Core similarity primitive; respects commutative ops (`+ * & && &#124; &#124;&#124; == !=`) |
| `compare_expressions(s1, s2) → SubtreeMatch` | Wraps `find_largest_common_subtree` |
| `match_alphazoo(prop, factor_df)` | Find closest match against a library |

### AST structural similarity

The reward function uses this formula (paper eq. 4):

```
sim(f_i, f_j) = |AST(f_i) ∩ AST(f_j)| / max(|AST(f_i)|, |AST(f_j)|)
```

Implemented in `training/factor_tool.py::_ast_similarity`:

```python path=F:\projects\AlphaAgentEvo\training\factor_tool.py start=54
def _ast_similarity(expr_a: str, expr_b: str) -> float:
    ast_a = parse_ast(expr_a)
    ast_b = parse_ast(expr_b)
    match = find_largest_common_subtree(ast_a, ast_b)
    size_a = count_all_nodes(expr_a)
    size_b = count_all_nodes(expr_b)
    if match is None or max(size_a, size_b) == 0:
        return 0.0
    return match.size / max(size_a, size_b)
```

Similarity is used in two places:
- **R_cons** (consistency): counts successful factors that are
  "similar-but-not-identical" to the seed (`H_LOW=0.1 < sim < H_HIGH=0.9`).
- **R_expl** (exploration): rewards novelty; a factor's reward scales
  with `1 - max_sim_to_predecessors`.

See [`06_reward.md`](./06_reward.md) for the full formulation.

## Pitfalls / tips

- The **execution parser's** `Optional("$")` means plain identifiers
  like `close` also parse, but the executor only has columns for `$`
  variables. Stick to `$close`, `$volume`, etc. for clarity and safety.
- `TS_CORR(series1, np.ndarray, p)` has a special branch that uses the
  array as the "fixed" second operand (useful with `SEQUENCE(p)` for
  regression-style windows). See `TS_CORR` and `TS_COVARIANCE` in
  `function_lib.py`.
- `DELTA(x, 1)` = `x - DELAY(x, 1)` = `diff` per instrument.
- `TS_MEAN(...)` uses `min_periods=1`, so warm-up days don't produce
  NaNs — factor values exist from day 1 of each instrument. This makes
  period masking well-behaved.
- `INDUSTRY_NEUTRALIZE(signal, $industry)` is not a TS operator; it
  removes per-sector mean inside each `datetime` group.
- Paper seeds use some **unimplemented** operators (`ZIGZAG_TOP`,
  `BARSLAST`, `WR`, `CCI`, `BBI`, `R_SQUARE`, `RESI`). Any expression
  using them will fail at runtime. The reward correctly reports
  `success=False` in that case.
