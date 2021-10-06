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
- **Python**: 
  - Nested List Comprehension `[[item if not item.isspace() else -1 for item in row] for row in board]` to build 2D matrix
  - String Formatting with Padding 0: For example, convert integer 2 to "02" `f"{month:02d}"`
  - Math's Ceil & Floor: `math.ceil()`, `math.floor()`
- **Math**: 
  - `Modular Multiplicative Inverse (MMI)`: **MMI(a, b) = x** s.t `a*x ≡ 1 (mod n)`
    - For example: a = 3, m = 11 => x = 4 as (3*4) mod 11 = 1
  - `Euclidean Algorithm` to find GCD of A & B & `Extended Euclidean Algorithm` to find **MMI(A, B)**
# Day 4:
- **LeetCode**: `Best Time to Buy and Sell Stock` (Keep track on the buying price, compare to the next days), `Climbing Stairs` (At T(n): first step = 1, remaining steps = T(n-1) or first step = 2, remaing steps = T(n-2). This recurrence relationship is similar to Fibonacci number)

# Day 5: 
- **LeetCode**: `3 Sum`, `Longest Palindromic Substring` and `Container With Most Water`
# Day 6: 
- **LeetCode**: `Number of Islands`, `Design Circular Queue`
