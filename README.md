# ABP Lógica (Flask) — Tabla de verdad + Lenguaje natural + Simplificación (leyes) + NAND-only

Integrantes de proyecto:
- David Andrés Cuadrado
- Marlon Steven Espinosa Prada
- Rusbell Oveymar Endes Cerón

Aplicación web en **Python (Flask)** para trabajar con lógica proposicional usando **máximo 3 variables (p, q, r)**.

Permite:
- Validar una fórmula proposicional.
- Generar **tabla de verdad**.
- Dar una **lectura en lenguaje natural** (usando los significados que escribas para p, q, r).
- **Detectar operadores** usados en la fórmula (incluye `nand`, `nor`, `xor`, `nimp`).
- Simplificar la expresión aplicando **solo las leyes vistas en clase** (sin inventar reglas).
- Mostrar una **equivalencia usando solo NAND** (“construcción de operadores”).

---

## ✅ Características

- Variables soportadas: `p`, `q`, `r`
- Constantes soportadas: `true`, `false`
- Operadores textuales soportados:
  - `not`  (negación)
  - `and`  (conjunción)
  - `or`   (disyunción)
  - `imp`  (implicación)
  - `iff`  (bicondicional)
  - `xnor` (equivalencia)
  - `xor`  (o exclusivo)
  - `nand` (no-y)
  - `nor`  (no-o)
  - `nimp` (no implicación: “p pero no q”)

También acepta símbolos:
- `¬` o `!`  → `not`
- `∧` o `&`  → `and`
- `∨` o `|`  → `or`
- `→` o `->` → `imp`
- `↔` o `<->` → `iff`

---

## 🧠 Interpretación de operadores (lectura natural)

| Operador | Lectura |
|---|---|
| `not p` | “no p” |
| `p and q` | “p y q” |
| `p or q` | “p o q” |
| `p imp q` | “si p, entonces q” |
| `p iff q` / `p xnor q` | “p si y solo si q” |
| `p nand q` | “no es cierto que p y q” (¬(p∧q)) |
| `p nor q` | “no es cierto que p o q” (¬(p∨q)) |
| `p xor q` | “p o q, pero no ambas” |
| `p nimp q` | “p, pero no q” (p ∧ ¬q) |

---

## ⚙️ Precedencia (orden de evaluación)

De mayor a menor prioridad:

1. `not`
2. `and`, `nand`
3. `or`, `nor`, `xor`, `xnor`
4. `imp`, `nimp`
5. `iff`

> Recomendación: usa **paréntesis** para evitar dudas.

---

## 📦 Requisitos

- Python 3.10+ (recomendado)
- Flask (incluido en `requirements.txt`)

---

## 🚀 Instalación y ejecución

### 1) Crear entorno virtual (opcional pero recomendado)
Windows (PowerShell)

python -m venv .venv

.\.venv\Scripts\activate

2) Instalar dependencias

pip install -r requirements.txt

3) Ejecutar

python app.py



🧪 Cómo usar (paso a paso)

En Significado de p/q/r, escribe algo tipo:

p = “hoy llueve”

q = “hace sol”

r = “tengo licencia”

En Fórmula proposicional, escribe tu expresión usando operadores textuales.

Presiona Generar.

Vas a ver:

Proposición en lenguaje natural

Operadores detectados

Resultado final simplificado + leyes aplicadas (y tabla de pasos)

Equivalencia con solo nand

Tabla de verdad (V/F)

✅ Ejemplos listos para copiar/pegar
A) Probar los operadores nuevos

NAND

(p nand q)


NOR

(p nor q)


XOR

(p xor q)


NIMP (p pero no q)

(p nimp q)

B) Probar simplificación con leyes

Idempotencia

(p and p)


De Morgan

not (p and q)


Absorción parcial

p and (p or q)


Absorción parcial (otra forma)

p or (p and q)


Absorción completa

p or (not p and q)


Absorción completa (otra forma)

p and (not p or q)

C) Mega prueba (usa todo)
(((p nand q) or (p nor r) or (q xor r)) and (p nimp q) and not (p and r))
