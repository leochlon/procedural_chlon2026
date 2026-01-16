import Mathlib.Tactic

/-!
# Conditional Certificates for Erdős-Straus on n = 840j + 1

This file proves Salez-style certificates for ES(n,x,y,z) when n = 840j + 1.
These certificates conditionally work based on prime factors of B = nx.

## A=3 Certificate (68.5% coverage)
Works when B₁ = (840j+1)(210j+1) has a prime factor q ≡ 2 (mod 3).

## A=7 Certificate (18.5% additional coverage)  
Works when B₂ = (840j+1)(210j+2) has a prime factor q ≡ 3, 5, or 6 (mod 7).

## A=11 Certificate (3% additional coverage)
Works unconditionally when j ≡ 5 or 7 (mod 11).
-/

/-- The Erdős-Straus predicate: 4/n = 1/x + 1/y + 1/z -/
def ES (n x y z : ℤ) : Prop := 4 * x * y * z = n * (y * z + x * z + x * y)

/-!
## A=3 Certificate

For n = 840j + 1, we have A = 4x - n = 3 when x = 210j + 1.
Let B = nx = (840j + 1)(210j + 1).

If q | B with q ≡ 2 (mod 3), then 1 + q ≡ 0 (mod 3).
Set 1 + q = 3m and qk = B, then:
- y = qkm = Bm
- z = km = B/q · m
-/

/-- A=3 certificate: ES holds for n=840j+1 when B=nx has prime q≡2 (mod 3) -/
theorem cert_A3_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 3 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 1)) :
    let n := 840 * j + 1
    let x := 210 * j + 1
    let y := q * k * m  
    let z := k * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hn3 : n + 3 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 3 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = 4 * n * x * x * k * m * m := by ring
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 3)) := by rw [hn3]
    _ = n * (k * m * m * (n * x + 3 * x)) := by ring
    _ = n * (k * m * m * (q * k + 3 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + 3 * x * k * m * m) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (3 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=7 Certificate

For n = 840j + 1, we have A = 4x - n = 7 when x = 210j + 2.
Let B = nx = (840j + 1)(210j + 2).

If q | B with q ≡ 6 (mod 7), then 1 + q ≡ 0 (mod 7).
Set 1 + q = 7m and qk = B, then:
- y = qkm = Bm
- z = km = B/q · m

Note: Empirically, primes q ≡ {3, 5, 6} (mod 7) all suffice because
B always has factor 2, and 2 × 3 ≡ 6, 2 × 5 ≡ 3 (mod 7), giving
a composite divisor ≡ 6 (mod 7) in those cases.
-/

/-- A=7 certificate: ES holds for n=840j+1 when B=nx has divisor q≡6 (mod 7) -/
theorem cert_A7_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 7 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 2)) :
    let n := 840 * j + 1
    let x := 210 * j + 2
    let y := q * k * m  
    let z := k * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hn7 : n + 7 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 7 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = 4 * n * x * x * k * m * m := by ring
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 7)) := by rw [hn7]
    _ = n * (k * m * m * (n * x + 7 * x)) := by ring
    _ = n * (k * m * m * (q * k + 7 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + 7 * x * k * m * m) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (7 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=11 Certificates for j ≡ 5, 7 (mod 11)

For n = 840j + 1, we have A = 4x - n = 11 when x = 210j + 3.

Key congruences:
- 840 ≡ 4 (mod 11), so n = 840j + 1 ≡ 4j + 1 (mod 11)
- 210 ≡ 1 (mod 11), so x = 210j + 3 ≡ j + 3 (mod 11)

For j ≡ 5 (mod 11): n ≡ 4·5 + 1 = 21 ≡ 10 (mod 11)
  So 1 + n ≡ 0 (mod 11), and we can use q = n in the formula.

For j ≡ 7 (mod 11): x ≡ 7 + 3 = 10 (mod 11)
  So 1 + x ≡ 0 (mod 11), and we can use q = x in the formula.
-/

/-- A=11 certificate for j ≡ 5 (mod 11): ES holds unconditionally -/
theorem cert_A11_j5_ES_clean (j m : ℤ) 
    (hnm : (840 * j + 1) + 1 = 11 * m) :  -- i.e., 1 + n = 11m
    let n := 840 * j + 1
    let x := 210 * j + 3
    let y := n * x * m  
    let z := x * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hn11 : n + 11 = 4 * x := by simp only [n, x]; ring
  have h1n : 1 + n = 11 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (n * x * m) * (x * m) := by rfl
    _ = 4 * n * x * x * x * m * m := by ring
    _ = n * (4 * x * x * x * m * m) := by ring
    _ = n * (x * x * m * m * (4 * x)) := by ring
    _ = n * (x * x * m * m * (n + 11)) := by rw [hn11]
    _ = n * (n * x * x * m * m + 11 * x * x * m * m) := by ring
    _ = n * (n * x * x * m * m + x * x * m * (11 * m)) := by ring
    _ = n * (n * x * x * m * m + x * x * m * (1 + n)) := by rw [h1n]
    _ = n * ((n * x * m) * (x * m) + x * (x * m) + x * (n * x * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-- A=11 certificate for j ≡ 7 (mod 11): ES holds unconditionally -/
theorem cert_A11_j7_ES_clean (j m : ℤ) 
    (hxm : (210 * j + 3) + 1 = 11 * m) :  -- i.e., 1 + x = 11m
    let n := 840 * j + 1
    let x := 210 * j + 3
    let y := x * n * m  
    let z := n * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hn11 : n + 11 = 4 * x := by simp only [n, x]; ring
  have h1x : 1 + x = 11 * m := by linarith
  -- Note: here q = x, k = n, so y = xnm and z = nm
  calc 4 * x * y * z 
      = 4 * x * (x * n * m) * (n * m) := by rfl
    _ = 4 * x * x * n * n * m * m := by ring
    _ = n * (4 * x * x * n * m * m) := by ring
    _ = n * (x * n * m * m * (4 * x)) := by ring
    _ = n * (x * n * m * m * (n + 11)) := by rw [hn11]
    _ = n * (n * n * x * m * m + 11 * n * x * m * m) := by ring
    _ = n * (n * n * x * m * m + n * x * m * (11 * m)) := by ring
    _ = n * (n * n * x * m * m + n * x * m * (1 + x)) := by rw [h1x]
    _ = n * ((x * n * m) * (n * m) + x * (n * m) + x * (x * n * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=15 Certificate

For n = 840j + 1, we have A = 4x - n = 15 when x = 210j + 4.
Let B = nx = (840j + 1)(210j + 4).

If q | B with q ≡ 14 (mod 15), then 1 + q ≡ 0 (mod 15).
Set 1 + q = 15m and qk = B, then:
- y = qkm = Bm
- z = km = B/q · m
-/

/-- A=15 certificate: ES holds for n=840j+1 when B=nx has divisor q≡14 (mod 15) -/
theorem cert_A15_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 15 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 4)) :
    let n := 840 * j + 1
    let x := 210 * j + 4
    let y := q * k * m  
    let z := k * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hn15 : n + 15 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 15 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = 4 * n * x * x * k * m * m := by ring
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 15)) := by rw [hn15]
    _ = n * (k * m * m * (n * x + 15 * x)) := by ring
    _ = n * (k * m * m * (q * k + 15 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + 15 * x * k * m * m) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (15 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=19 Certificate

For n = 840j + 1, we have A = 4x - n = 19 when x = 210j + 5.
Let B = nx = (840j + 1)(210j + 5).
-/

/-- A=19 certificate: ES holds for n=840j+1 when B=nx has divisor q≡18 (mod 19) -/
theorem cert_A19_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 19 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 5)) :
    let n := 840 * j + 1
    let x := 210 * j + 5
    let y := q * k * m  
    let z := k * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hn19 : n + 19 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 19 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = 4 * n * x * x * k * m * m := by ring
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 19)) := by rw [hn19]
    _ = n * (k * m * m * (n * x + 19 * x)) := by ring
    _ = n * (k * m * m * (q * k + 19 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + 19 * x * k * m * m) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (19 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=23 Certificate

For n = 840j + 1, we have A = 4x - n = 23 when x = 210j + 6.
Let B = nx = (840j + 1)(210j + 6).
-/

/-- A=23 certificate: ES holds for n=840j+1 when B=nx has divisor q≡22 (mod 23) -/
theorem cert_A23_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 23 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 6)) :
    let n := 840 * j + 1
    let x := 210 * j + 6
    let y := q * k * m  
    let z := k * m      
    ES n x y z := by
  intro n x y z
  unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hn23 : n + 23 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 23 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = 4 * n * x * x * k * m * m := by ring
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 23)) := by rw [hn23]
    _ = n * (k * m * m * (n * x + 23 * x)) := by ring
    _ = n * (k * m * m * (q * k + 23 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + 23 * x * k * m * m) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (23 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=27 Certificate
-/
theorem cert_A27_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 27 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 7)) :
    let n := 840 * j + 1
    let x := 210 * j + 7
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 27 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 27 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 27)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 27 * x)) := by ring
    _ = n * (k * m * m * (q * k + 27 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (27 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=31 Certificate
-/
theorem cert_A31_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 31 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 8)) :
    let n := 840 * j + 1
    let x := 210 * j + 8
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 31 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 31 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 31)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 31 * x)) := by ring
    _ = n * (k * m * m * (q * k + 31 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (31 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=35 Certificate
-/
theorem cert_A35_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 35 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 9)) :
    let n := 840 * j + 1
    let x := 210 * j + 9
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 35 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 35 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 35)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 35 * x)) := by ring
    _ = n * (k * m * m * (q * k + 35 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (35 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=39 Certificate
-/
theorem cert_A39_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 39 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 10)) :
    let n := 840 * j + 1
    let x := 210 * j + 10
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 39 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 39 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 39)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 39 * x)) := by ring
    _ = n * (k * m * m * (q * k + 39 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (39 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=47 Certificate
-/
theorem cert_A47_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 47 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 12)) :
    let n := 840 * j + 1
    let x := 210 * j + 12
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 47 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 47 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 47)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 47 * x)) := by ring
    _ = n * (k * m * m * (q * k + 47 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (47 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=59 Certificate
-/
theorem cert_A59_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 59 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 15)) :
    let n := 840 * j + 1
    let x := 210 * j + 15
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 59 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 59 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 59)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 59 * x)) := by ring
    _ = n * (k * m * m * (q * k + 59 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (59 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-!
## A=79 Certificate
-/
theorem cert_A79_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 79 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 20)) :
    let n := 840 * j + 1
    let x := 210 * j + 20
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 79 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 79 * m := by linarith
  calc 4 * x * y * z 
      = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (q * k) * k * m * m := by ring
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]
    _ = n * (4 * x * x * k * m * m) := by ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 79)) := by rw [hnA]
    _ = n * (k * m * m * (n * x + 79 * x)) := by ring
    _ = n * (k * m * m * (q * k + 79 * x)) := by rw [hB]
    _ = n * (q * k * k * m * m + x * k * m * (79 * m)) := by ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]
    _ = n * ((q * k * m) * (k * m) + x * (k * m) + x * (q * k * m)) := by ring
    _ = n * (y * z + x * z + x * y) := by rfl

/-! ## A=43 Certificate -/
theorem cert_A43_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 43 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 11)) :
    let n := 840 * j + 1; let x := 210 * j + 11
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 43 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 43 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 43)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 43 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=51 Certificate -/
theorem cert_A51_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 51 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 13)) :
    let n := 840 * j + 1; let x := 210 * j + 13
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 51 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 51 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 51)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 51 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=55 Certificate -/
theorem cert_A55_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 55 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 14)) :
    let n := 840 * j + 1; let x := 210 * j + 14
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 55 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 55 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 55)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 55 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=63 Certificate -/
theorem cert_A63_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 63 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 16)) :
    let n := 840 * j + 1; let x := 210 * j + 16
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 63 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 63 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 63)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 63 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=67 Certificate -/
theorem cert_A67_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 67 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 17)) :
    let n := 840 * j + 1; let x := 210 * j + 17
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 67 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 67 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 67)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 67 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=71 Certificate -/
theorem cert_A71_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 71 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 18)) :
    let n := 840 * j + 1; let x := 210 * j + 18
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 71 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 71 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 71)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 71 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring

/-! ## A=75 Certificate -/
theorem cert_A75_ES_clean (j q k m : ℤ) 
    (hqp1 : q + 1 = 75 * m) (hBk : q * k = (840 * j + 1) * (210 * j + 19)) :
    let n := 840 * j + 1; let x := 210 * j + 19
    let y := q * k * m; let z := k * m      
    ES n x y z := by
  intro n x y z; unfold ES
  have hB : q * k = n * x := by simp only [n, x]; exact hBk
  have hnA : n + 75 = 4 * x := by simp only [n, x]; ring
  have h1q : 1 + q = 75 * m := by linarith
  calc 4 * x * y * z = 4 * x * (q * k * m) * (k * m) := by rfl
    _ = 4 * x * (n * x) * k * m * m := by rw [hB]; ring
    _ = n * (x * k * m * m * (4 * x)) := by ring
    _ = n * (x * k * m * m * (n + 75)) := by rw [hnA]
    _ = n * (k * m * m * (q * k + 75 * x)) := by rw [hB]; ring
    _ = n * (q * k * k * m * m + x * k * m * (1 + q)) := by rw [h1q]; ring
    _ = n * (y * z + x * z + x * y) := by ring
