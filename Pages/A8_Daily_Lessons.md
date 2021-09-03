# Daily Lessons

# Day 1: 
- **Math**: [Geometric Sequences](https://www.mathsisfun.com/algebra/sequences-sums-geometric.html)
- **Python**: List (`enumarate`), Dict (`keys()`, `values()`,`items()`)
- **LeetCode**: [50. Pow(x, n)](https://leetcode.com/problems/powx-n/): using Recursive with `Pow(x,n) = Pow(x,n//2)`, rmb for n = odd and even cases
# Day 2: 
- **Math**: [Modular Arithmetic](https://brilliant.org/wiki/modular-arithmetic/)
  - **Congruence** `a ≡ b (mod n)` For a positive integer n, the integers *a and b are congruent mod n* if their remainders when divided by n are the same.
    - For example: 52≡24(mod7): 52 and 24 are congruent (mod 7) because (52 mod 7) = 3 and (24 mod 7) = 3.
  - **Properties of multiplication** in Modular Arithmetic:
  - `(a mod n) mod n = a mod n`: This is obvious because a mod n ∈ [0,𝑛−1] and so the second modmod cannot have an effect.
  - `(A^2) mod C = (A * A) mod C = ((A mod C) * (A mod C)) mod C`
- **LeetCode**: [Fast Exponentiation](https://youtu.be/-3Lt-EwR_Hw)
# Day 3:
- **Python**: Nested List Comprehension `[[item if not item.isspace() else -1 for item in row] for row in board]` to build 2D matrix
