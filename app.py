from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import List, Dict, Optional, Tuple, Any

from flask import Flask, render_template, request

app = Flask(__name__)

# =========================
# Tokenización / Parsing
# =========================

@dataclass
class Tok:
    t: str                 # VAR, CONST, NOT, AND, OR, IMP, IFF, XNOR, XOR, NAND, NOR, NIMP, LP, RP
    v: Optional[str] = None  # VAR: nombre (p,q,r) | CONST: 'true'/'false'

# Precedencias de operadores (más alto = se evalúa antes)
# NOT > AND/NAND > OR/NOR/XOR/XNOR > IMP/NIMP > IFF
PRE = {
    "NOT": 5,
    "AND": 4,
    "NAND": 4,
    "OR": 3,
    "NOR": 3,
    "XOR": 3,
    "XNOR": 3,
    "IMP": 2,
    "NIMP": 2,
    "IFF": 1,
}
RIGHT_ASSOC = {"NOT", "IMP", "NIMP"}  # unary NOT y las implicativas se asocian a la derecha

_KW_OPS = {
    "not": "NOT",
    "and": "AND",
    "or": "OR",
    "imp": "IMP",
    "iff": "IFF",
    "xnor": "XNOR",
    "xor": "XOR",
    "nand": "NAND",
    "nor": "NOR",
    "nimp": "NIMP",
    "true": "CONST",
    "false": "CONST",
}

def tokenize(formula: str) -> List[Tok]:
    """Convierte una cadena en tokens.

    Entradas soportadas:
      - variables: p, q, r
      - operadores textuales: not, and, or, imp, iff, nand, nor, xor, xnor, nimp
      - constantes: true, false
      - también soporta símbolos: ¬/!, ∧/& , ∨/| , →/-> , ↔/<->
    """
    f = (formula or "")
    # Normalizar símbolos
    f = (f.replace("¬", "!")
           .replace("∧", "&")
           .replace("∨", "|")
           .replace("→", "->")
           .replace("↔", "<->")
           .replace("⨀", "xnor"))
    out: List[Tok] = []
    i = 0
    while i < len(f):
        c = f[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha():
            # leer identificador completo (para operadores textuales)
            j = i
            while j < len(f) and (f[j].isalnum() or f[j] == "_"):
                j += 1
            ident = f[i:j].lower()
            if ident in _KW_OPS:
                tt = _KW_OPS[ident]
                if tt == "CONST":
                    out.append(Tok("CONST", ident))
                else:
                    out.append(Tok(tt))
                i = j
            else:
                # variables: solo aceptamos p, q, r (una letra)
                if len(ident) == 1:
                    out.append(Tok("VAR", ident))
                    i = j
                else:
                    raise ValueError(f"Identificador no reconocido: {ident!r}. Usa p,q,r u operadores: nand, nor, xor, xnor, nimp, not, and, or, imp, iff.")
        elif c == "(":
            out.append(Tok("LP")); i += 1
        elif c == ")":
            out.append(Tok("RP")); i += 1
        elif c == "!":
            out.append(Tok("NOT")); i += 1
        elif c == "&":
            out.append(Tok("AND")); i += 1
        elif c == "|":
            out.append(Tok("OR")); i += 1
        elif c == "-" and i+1 < len(f) and f[i+1] == ">":
            out.append(Tok("IMP")); i += 2
        elif c == "<" and f[i:i+3] == "<->":
            out.append(Tok("IFF")); i += 3
        else:
            raise ValueError(f"Símbolo no reconocido cerca de: {c!r}")
    return out

def to_rpn(tokens: List[Tok]) -> List[Tok]:
    out: List[Tok] = []
    st: List[Tok] = []
    def is_op(t): return t.t in PRE
    for t in tokens:
        if t.t in {"VAR", "CONST"}:
            out.append(t)
        elif t.t == "LP":
            st.append(t)
        elif t.t == "RP":
            while st and st[-1].t != "LP":
                out.append(st.pop())
            if not st:
                raise ValueError("Paréntesis no balanceados")
            st.pop()  # saca LP
        elif is_op(t):
            while st:
                top = st[-1]
                if top.t in PRE:
                    le = PRE[t.t] <= PRE[top.t]
                    lt = PRE[t.t]  < PRE[top.t]
                    if ((t.t not in RIGHT_ASSOC and le)
                        or (t.t in RIGHT_ASSOC and lt)):
                        out.append(st.pop()); continue
                break
            st.append(t)
    while st:
        top = st.pop()
        if top.t in {"LP", "RP"}:
            raise ValueError("Paréntesis no balanceados")
        out.append(top)
    return out

# =========================
# AST y evaluación
# =========================

@dataclass
class Node:
    kind: str                    # VAR, CONST, NOT, AND, OR, IMP, IFF, XNOR, XOR, NAND, NOR, NIMP
    val: Optional[str] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None

def rpn_to_ast(rpn: List[Tok]) -> Node:
    st: List[Node] = []
    for t in rpn:
        if t.t == "VAR":
            st.append(Node("VAR", t.v))
        elif t.t == "CONST":
            st.append(Node("CONST", t.v))
        elif t.t == "NOT":
            a = st.pop()
            st.append(Node("NOT", left=a))
        elif t.t in {"AND", "OR", "IMP", "IFF", "XNOR", "XOR", "NAND", "NOR", "NIMP"}:
            b = st.pop(); a = st.pop()
            st.append(Node(t.t, left=a, right=b))
        else:
            raise ValueError("Token inesperado en RPN")
    if len(st) != 1:
        raise ValueError("Expresión inválida")
    return st[0]

def eval_ast(node: Node, ctx: Dict[str, bool]) -> bool:
    k = node.kind
    if k == "VAR":
        return bool(ctx[node.val])
    if k == "CONST":
        return True if node.val == "true" else False
    if k == "NOT":
        return not eval_ast(node.left, ctx)
    if k == "AND":
        return eval_ast(node.left, ctx) and eval_ast(node.right, ctx)
    if k == "OR":
        return eval_ast(node.left, ctx) or eval_ast(node.right, ctx)
    if k == "NAND":
        return not (eval_ast(node.left, ctx) and eval_ast(node.right, ctx))
    if k == "NOR":
        return not (eval_ast(node.left, ctx) or eval_ast(node.right, ctx))
    if k == "XOR":
        return eval_ast(node.left, ctx) != eval_ast(node.right, ctx)
    if k == "NIMP":
        # no es cierto que (p -> q)  ==  p y no q
        return eval_ast(node.left, ctx) and (not eval_ast(node.right, ctx))
    if k == "IMP":
        return (not eval_ast(node.left, ctx)) or eval_ast(node.right, ctx)
    if k in {"IFF", "XNOR"}:
        return eval_ast(node.left, ctx) == eval_ast(node.right, ctx)
    raise ValueError("Nodo inválido")

def vars_from_tokens(tokens: List[Tok]) -> List[str]:
    return sorted({t.v for t in tokens if t.t == "VAR"})


def operators_from_tokens(tokens: List[Tok]) -> List[str]:
    """Devuelve operadores usados (únicos) preservando orden de aparición."""
    seen = set()
    ops: List[str] = []
    order_map = {
        "NOT": "not",
        "AND": "and",
        "OR": "or",
        "IMP": "imp",
        "IFF": "iff",
        "XNOR": "xnor",
        "XOR": "xor",
        "NAND": "nand",
        "NOR": "nor",
        "NIMP": "nimp",
    }
    for t in tokens:
        if t.t in order_map and t.t not in seen:
            seen.add(t.t)
            ops.append(order_map[t.t])
    return ops


def ast_to_text(node: Node) -> str:
    """Imprime la fórmula usando operadores textuales (not/and/or/imp/iff/nand/nor/xor/xnor/nimp)."""
    prec = {
        "VAR": 6,
        "CONST": 6,
        "NOT": 5,
        "AND": 4,
        "NAND": 4,
        "OR": 3,
        "NOR": 3,
        "XOR": 3,
        "XNOR": 3,
        "IMP": 2,
        "NIMP": 2,
        "IFF": 1,
    }

    def _p(n: Node, parent_kind: Optional[str] = None) -> str:
        k = n.kind
        if k == "VAR":
            return n.val
        if k == "CONST":
            return n.val
        if k == "NOT":
            inner = _p(n.left, "NOT")
            # paréntesis si el hijo es binario
            if n.left.kind in {"AND", "OR", "IMP", "IFF", "XNOR", "XOR", "NAND", "NOR", "NIMP"}:
                inner = f"({inner})"
            s = f"not {inner}"
            if parent_kind and prec[k] < prec[parent_kind]:
                return f"({s})"
            return s

        op_map = {
            "AND": "and",
            "OR": "or",
            "IMP": "imp",
            "IFF": "iff",
            "XNOR": "xnor",
            "XOR": "xor",
            "NAND": "nand",
            "NOR": "nor",
            "NIMP": "nimp",
        }
        if k in op_map:
            a = _p(n.left, k)
            b = _p(n.right, k)
            s = f"{a} {op_map[k]} {b}"
            if parent_kind and prec[k] < prec[parent_kind]:
                return f"({s})"
            return s

        return "?"

    return _p(node)


def node_key(n: Node) -> Any:
    """Clave hashable para comparar nodos estructuralmente."""
    if n.kind in {"VAR", "CONST"}:
        return (n.kind, n.val)
    if n.kind == "NOT":
        return ("NOT", node_key(n.left))
    return (n.kind, node_key(n.left), node_key(n.right))


def desugar_to_base(n: Node) -> Node:
    """Convierte NAND/NOR/XOR/XNOR/NIMP a operadores base (NOT/AND/OR/IMP/IFF)."""
    k = n.kind
    if k in {"VAR", "CONST"}:
        return n
    if k == "NOT":
        return Node("NOT", left=desugar_to_base(n.left))

    a = desugar_to_base(n.left) if n.left else None
    b = desugar_to_base(n.right) if n.right else None

    if k == "AND":
        return Node("AND", left=a, right=b)
    if k == "OR":
        return Node("OR", left=a, right=b)
    if k == "IMP":
        return Node("IMP", left=a, right=b)
    if k in {"IFF", "XNOR"}:
        return Node("IFF", left=a, right=b)

    if k == "NAND":
        return Node("NOT", left=Node("AND", left=a, right=b))
    if k == "NOR":
        return Node("NOT", left=Node("OR", left=a, right=b))
    if k == "NIMP":
        # ¬(p→q) == p ∧ ¬q
        return Node("AND", left=a, right=Node("NOT", left=b))
    if k == "XOR":
        # (p ∨ q) ∧ ¬(p ∧ q)
        return Node(
            "AND",
            left=Node("OR", left=a, right=b),
            right=Node("NOT", left=Node("AND", left=a, right=b)),
        )

    # fallback
    return Node(k, left=a, right=b)


def simplify_with_laws(root: Node, max_passes: int = 50) -> Tuple[Node, List[Dict[str, str]]]:
    """Aplica SOLO las leyes de tus apuntes y devuelve (AST_simplificado, pasos)."""
    steps: List[Dict[str, str]] = []

    def add_step(ley: str, before: Node, after: Node):
        steps.append({
            "ley": ley,
            "antes": ast_to_text(before),
            "despues": ast_to_text(after),
        })

    def is_true(n: Node) -> bool:
        return n.kind == "CONST" and n.val == "true"

    def is_false(n: Node) -> bool:
        return n.kind == "CONST" and n.val == "false"

    def flatten(kind: str, n: Node) -> List[Node]:
        """Aplana por asociatividad para AND/OR."""
        if n.kind != kind:
            return [n]
        return flatten(kind, n.left) + flatten(kind, n.right)

    def rebuild(kind: str, items: List[Node]) -> Node:
        assert items
        cur = items[0]
        for it in items[1:]:
            cur = Node(kind, left=cur, right=it)
        return cur

    def simplify_node(n: Node) -> Node:
        # post-order: primero hijos
        if n.kind == "NOT":
            child0 = simplify_node(n.left)
            n0 = Node("NOT", left=child0)
            # De Morgan
            if child0.kind in {"AND", "OR"}:
                before = n0
                a = child0.left
                b = child0.right
                if child0.kind == "AND":
                    after = Node("OR", left=Node("NOT", left=a), right=Node("NOT", left=b))
                else:
                    after = Node("AND", left=Node("NOT", left=a), right=Node("NOT", left=b))
                add_step("Ley de De Morgan", before, after)
                return simplify_node(after)
            return n0

        if n.kind in {"AND", "OR", "IMP", "IFF"}:
            a = simplify_node(n.left)
            b = simplify_node(n.right)
            n0 = Node(n.kind, left=a, right=b)

            # IMP / IFF se mantienen (las leyes dadas se enfocan en AND/OR/NOT),
            # pero sí simplificamos identidades si llegan constantes.
            if n.kind in {"IMP", "IFF"}:
                return n0

            # Asociativa + Idempotencia + Identidad (para AND/OR)
            items = flatten(n.kind, n0)
            before_flat = n0
            # registrar asociativa si cambió la forma
            if node_key(before_flat) != node_key(rebuild(n.kind, items)):
                add_step("Ley asociativa", before_flat, rebuild(n.kind, items))

            # Identidad / Anulación por constantes
            if n.kind == "AND":
                if any(is_false(x) for x in items):
                    after = Node("CONST", "false")
                    add_step("Ley de identidad", before_flat, after)
                    return after
                # quitar true
                items2 = [x for x in items if not is_true(x)]
                if len(items2) != len(items):
                    after = rebuild("AND", items2) if items2 else Node("CONST", "true")
                    add_step("Ley de identidad", before_flat, after)
                    return simplify_node(after)
                items = items2
            else:  # OR
                if any(is_true(x) for x in items):
                    after = Node("CONST", "true")
                    add_step("Ley de identidad", before_flat, after)
                    return after
                # quitar false
                items2 = [x for x in items if not is_false(x)]
                if len(items2) != len(items):
                    after = rebuild("OR", items2) if items2 else Node("CONST", "false")
                    add_step("Ley de identidad", before_flat, after)
                    return simplify_node(after)
                items = items2

            # Idempotencia: quitar duplicados
            seen = set()
            uniq: List[Node] = []
            for x in items:
                kx = node_key(x)
                if kx not in seen:
                    seen.add(kx)
                    uniq.append(x)
            if len(uniq) != len(items):
                after = rebuild(n.kind, uniq) if uniq else (Node("CONST", "true") if n.kind == "AND" else Node("CONST", "false"))
                add_step("Ley de idempotencia", before_flat, after)
                return simplify_node(after)
            items = uniq

            # Absorción parcial
            if n.kind == "AND":
                # p and (p or q) = p
                item_keys = {node_key(x) for x in items}
                new_items = []
                changed = False
                for x in items:
                    if x.kind == "OR":
                        or_items = flatten("OR", x)
                        if any(node_key(y) in item_keys for y in or_items):
                            changed = True
                            continue  # se absorbe
                    new_items.append(x)
                if changed:
                    after = rebuild("AND", new_items) if new_items else Node("CONST", "true")
                    add_step("Ley de absorción (parcial)", before_flat, after)
                    return simplify_node(after)
            else:  # OR
                # p or (p and q) = p
                item_keys = {node_key(x) for x in items}
                new_items = []
                changed = False
                for x in items:
                    if x.kind == "AND":
                        and_items = flatten("AND", x)
                        if any(node_key(y) in item_keys for y in and_items):
                            changed = True
                            continue
                    new_items.append(x)
                if changed:
                    after = rebuild("OR", new_items) if new_items else Node("CONST", "false")
                    add_step("Ley de absorción (parcial)", before_flat, after)
                    return simplify_node(after)

            # Absorción completa
            if n.kind == "OR":
                # p or (not p and q) = p or q
                item_map = {node_key(x): x for x in items}
                new_items = items[:]
                changed = False
                for x in items:
                    if x.kind == "AND":
                        and_items = flatten("AND", x)
                        # buscar forma (not p) dentro del AND, y p fuera
                        for y in and_items:
                            if y.kind == "NOT":
                                p = y.left
                                if node_key(p) in item_map:
                                    # q = AND sin not p
                                    q_items = [z for z in and_items if node_key(z) != node_key(y)]
                                    if q_items:
                                        q = rebuild("AND", q_items) if len(q_items) > 1 else q_items[0]
                                        # reemplazar (not p and q) por q
                                        new_items = [w for w in new_items if node_key(w) != node_key(x)]
                                        if node_key(q) not in {node_key(w) for w in new_items}:
                                            new_items.append(q)
                                        changed = True
                    if changed:
                        break
                if changed:
                    after = rebuild("OR", new_items)
                    add_step("Ley de absorción (completa)", before_flat, after)
                    return simplify_node(after)

            if n.kind == "AND":
                # p and (not p or q) = p and q
                item_map = {node_key(x): x for x in items}
                new_items = items[:]
                changed = False
                for x in items:
                    if x.kind == "OR":
                        or_items = flatten("OR", x)
                        for y in or_items:
                            if y.kind == "NOT":
                                p = y.left
                                if node_key(p) in item_map:
                                    q_items = [z for z in or_items if node_key(z) != node_key(y)]
                                    if q_items:
                                        q = rebuild("OR", q_items) if len(q_items) > 1 else q_items[0]
                                        new_items = [w for w in new_items if node_key(w) != node_key(x)]
                                        if node_key(q) not in {node_key(w) for w in new_items}:
                                            new_items.append(q)
                                        changed = True
                    if changed:
                        break
                if changed:
                    after = rebuild("AND", new_items)
                    add_step("Ley de absorción (completa)", before_flat, after)
                    return simplify_node(after)

            # si queda solo un elemento
            if len(items) == 1:
                return items[0]
            return rebuild(n.kind, items)

        # otros nodos (extendidos) se desazucaran antes
        return n

    cur = root
    for _ in range(max_passes):
        nxt = simplify_node(cur)
        if node_key(nxt) == node_key(cur):
            break
        cur = nxt
    return cur, steps


def to_nand_only(n: Node) -> Node:
    """Convierte una fórmula base (NOT/AND/OR/IMP/IFF/VAR/CONST) a solo NAND."""
    # NOT a = nand(a,a)
    # AND(a,b) = nand(nand(a,b), nand(a,b))
    # OR(a,b) = nand(nand(a,a), nand(b,b))
    # IMP(a,b) = OR(NOT a, b)
    # IFF(a,b) = AND(IMP(a,b), IMP(b,a))

    if n.kind in {"VAR", "CONST"}:
        return n
    if n.kind == "NOT":
        a = to_nand_only(n.left)
        return Node("NAND", left=a, right=a)
    if n.kind == "AND":
        a = to_nand_only(n.left); b = to_nand_only(n.right)
        t = Node("NAND", left=a, right=b)
        return Node("NAND", left=t, right=t)
    if n.kind == "OR":
        a = to_nand_only(n.left); b = to_nand_only(n.right)
        na = Node("NAND", left=a, right=a)
        nb = Node("NAND", left=b, right=b)
        return Node("NAND", left=na, right=nb)
    if n.kind == "IMP":
        a = to_nand_only(Node("NOT", left=n.left))
        b = to_nand_only(n.right)
        return to_nand_only(Node("OR", left=a, right=b))
    if n.kind == "IFF":
        imp1 = Node("IMP", left=n.left, right=n.right)
        imp2 = Node("IMP", left=n.right, right=n.left)
        return to_nand_only(Node("AND", left=imp1, right=imp2))
    # por si llega algo raro
    return to_nand_only(desugar_to_base(n))

# =========================
# Tabla de verdad
# =========================

def truth_table(ast: Node, vars_ord: List[str]) -> Tuple[List[Tuple[Dict[str,bool], bool]], str]:
    filas: List[Tuple[Dict[str,bool], bool]] = []
    T = F = 0
    for bits in itertools.product([False, True], repeat=len(vars_ord)):
        ctx = {v: bits[i] for i, v in enumerate(vars_ord)}
        val = eval_ast(ast, ctx)
        filas.append((ctx, val))
        if val: T += 1
        else:    F += 1
    diag = "Tautología" if T == len(filas) else ("Contradicción" if F == len(filas) else "Contingencia")
    return filas, diag

# =========================
# Lenguaje natural mejorado
# =========================

PREC = {"VAR":5, "NOT":4, "AND":3, "OR":2, "XNOR":2, "IMP":1, "IFF":1}

def _sanitize_label(s: str) -> str:
    s = (s or "").strip()
    return " ".join(s.split()).lower()

def _sentence_case(s: str) -> str:
    s = " ".join((s or "").split()).strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s

def verbal(ast_root: Node, raw_labels: Dict[str, str]) -> str:
    labels = {k: (_sanitize_label(raw_labels.get(k)) or k) for k in ("p", "q", "r")}

    def wrap(child: Node, parent_op: str) -> str:
        txt = _phrase(child)
        # (solo semántico, no imprime paréntesis, pero permite anidar bien)
        return txt

    def _phrase(n: Node) -> str:
        k = n.kind
        if k == "VAR":
            return labels.get(n.val, n.val)

        if k == "CONST":
            return "verdadero" if n.val == "true" else "falso"

        if k == "NOT":
            inside = _phrase(n.left)
            return inside if inside.startswith("no ") else f"no {inside}"

        if k == "AND":
            a = wrap(n.left, "AND"); b = wrap(n.right, "AND")
            return f"{a} y {b}"

        if k == "OR":
            a = wrap(n.left, "OR"); b = wrap(n.right, "OR")
            return f"{a} o {b}"

        if k == "IMP":
            ant = wrap(n.left, "IMP"); cons = wrap(n.right, "IMP")
            return f"si {ant}, entonces {cons}"

        if k in ("IFF", "XNOR"):
            a = wrap(n.left, k); b = wrap(n.right, k)
            return f"{a} si y solo si {b}"

        if k == "NAND":
            a = wrap(n.left, "NAND"); b = wrap(n.right, "NAND")
            return f"no es cierto que {a} y {b}"

        if k == "NOR":
            a = wrap(n.left, "NOR"); b = wrap(n.right, "NOR")
            return f"no es cierto que {a} o {b}"

        if k == "XOR":
            a = wrap(n.left, "XOR"); b = wrap(n.right, "XOR")
            return f"{a} o {b}, pero no ambas"

        if k == "NIMP":
            a = wrap(n.left, "NIMP"); b = wrap(n.right, "NIMP")
            return f"{a}, pero no {b}"

        return "?"

    return _sentence_case(_phrase(ast_root))

# =========================
# Utilidades
# =========================

def pretty_formula(formula: str) -> str:
    return (formula or ""
            ).replace("<->", "↔"
            ).replace("->", "→"
            ).replace("&", "∧"
            ).replace("|", "∨"
            ).replace("xnor", "⨀"
            ).replace("!", "¬")

# =========================
# Rutas
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    data = {"p_text": "", "q_text": "", "r_text": "", "formula": ""}
    resultado = None
    error = None

    if request.method == "POST":
        data["p_text"] = request.form.get("p_text", "").strip()
        data["q_text"] = request.form.get("q_text", "").strip()
        data["r_text"] = request.form.get("r_text", "").strip()
        data["formula"] = request.form.get("formula", "").strip()

        try:
            if not data["formula"]:
                raise ValueError("Escribe una fórmula proposicional.")

            tokens = tokenize(data["formula"])
            used = vars_from_tokens(tokens)

            # Solo p, q, r
            if any(v not in {"p", "q", "r"} for v in used):
                raise ValueError("Solo se permiten variables p, q y r.")
            if not used:
                used = ["p"]

            rpn = to_rpn(tokens)
            ast = rpn_to_ast(rpn)

            # 1) Operadores detectados (textuales)
            ops_used = operators_from_tokens(tokens)

            # 2) Simplificación aplicando SOLO leyes vistas en clase
            base = desugar_to_base(ast)
            simpl, pasos = simplify_with_laws(base)
            leyes_aplicadas: List[str] = []
            seen_leyes = set()
            for p in pasos:
                if p["ley"] not in seen_leyes:
                    seen_leyes.add(p["ley"])
                    leyes_aplicadas.append(p["ley"])

            # 3) Equivalencia usando solo NAND (útil para el tema de construcción de operadores)
            nand_only = to_nand_only(simpl)

            filas, diag = truth_table(ast, used)
            ln = verbal(ast, {
                "p": data["p_text"] or "p",
                "q": data["q_text"] or "q",
                "r": data["r_text"] or "r",
            })

            resultado = {
                "formula_pretty": pretty_formula(data["formula"]),
                "formula_text": ast_to_text(ast),
                "operators": ops_used,
                "filas": filas,
                "diag": diag,
                "vars_ord": used,
                "ln": ln,
                "simplified_text": ast_to_text(simpl),
                "laws": leyes_aplicadas,
                "steps": pasos,
                "nand_text": ast_to_text(nand_only),
            }
        except Exception as e:
            error = str(e)

    return render_template("index.html", data=data, resultado=resultado, error=error)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
