/-
  CRTFilters.lean - Coverage theorems using Chinese Remainder Theorem
  
  Combines certificates from SalezCertificates.lean to prove coverage
  for residue classes modulo 60 = lcm(4, 3, 5).
  
  Coverage: 36/60 = 60%
  Based on: Salez (arXiv:1406.6307v1)
-/

import Mathlib.Tactic
import Mathlib.Data.Nat.GCD.Basic

/-! ## ES Definition (matching SalezCertificates.lean) -/

def ES (n x y z : Nat) : Prop := 4 * x * y * z = n * (x * y + x * z + y * z)

/-! ## Covered Residue Sets -/

/-- Residues mod 60 covered by our three certificates -/
def CoveredSet60 : List Nat :=
  [0, 2, 4, 5, 8, 10, 11, 12, 14, 15, 16, 17, 20, 23, 24, 25, 26, 28, 29, 30,
   32, 35, 36, 38, 40, 41, 44, 45, 47, 48, 50, 52, 53, 55, 56, 59]

/-- Escape residues mod 60 (not covered by our certificates) -/
def EscapeSet60 : List Nat :=
  [1, 3, 6, 7, 9, 13, 18, 19, 21, 22, 27, 31, 33, 34, 37, 39, 42, 43, 46, 49, 51, 54, 57, 58]

/-! ## Coverage Proofs -/

/-- 60 = lcm(4, 3, 5) -/
theorem lcm_4_3_5 : Nat.lcm (Nat.lcm 4 3) 5 = 60 := by native_decide

/-- 4, 3, 5 are pairwise coprime -/
theorem coprime_4_3 : Nat.Coprime 4 3 := by native_decide
theorem coprime_4_5 : Nat.Coprime 4 5 := by native_decide
theorem coprime_3_5 : Nat.Coprime 3 5 := by native_decide

/-! ## Individual Coverage Theorems -/

/-- n ≡ 0 (mod 4) implies ES has solution -/
theorem coverage_mod4_r0 (n : Nat) (hn : n > 0) (hmod : n % 4 = 0) :
    ∃ x y z, ES n x y z := by
  use n / 2, n, n
  unfold ES
  have hdiv : 4 ∣ n := Nat.dvd_of_mod_eq_zero hmod
  obtain ⟨k, hk⟩ := hdiv
  have hk_pos : k > 0 := by omega
  have hdiv2 : (4 * k) / 2 = 2 * k := by omega
  calc 4 * (n / 2) * n * n 
      = 4 * ((4 * k) / 2) * (4 * k) * (4 * k) := by rw [hk]
    _ = 4 * (2 * k) * (4 * k) * (4 * k) := by rw [hdiv2]
    _ = 128 * k * k * k := by ring
    _ = (4 * k) * ((2 * k) * (4 * k) + (2 * k) * (4 * k) + (4 * k) * (4 * k)) := by ring
    _ = (4 * k) * (((4 * k) / 2) * (4 * k) + ((4 * k) / 2) * (4 * k) + (4 * k) * (4 * k)) := by rw [hdiv2]
    _ = n * (n / 2 * n + n / 2 * n + n * n) := by rw [hk]

/-- n ≡ 2 (mod 3) implies ES has solution -/
theorem coverage_mod3_r2 (n : Nat) (hn : n > 0) (hmod : n % 3 = 2) :
    ∃ x y z, ES n x y z := by
  -- For n ≡ 2 (mod 3), n+1 is divisible by 3
  -- Let t = (n+1)/3, then n = 3t - 1
  -- Use x = t, y = n, z = t*n
  have hdiv : 3 ∣ (n + 1) := by
    have h : (n + 1) % 3 = 0 := by omega
    exact Nat.dvd_of_mod_eq_zero h
  obtain ⟨t, ht⟩ := hdiv
  have ht_pos : t > 0 := by omega
  have hn_eq : n = 3 * t - 1 := by omega
  use t, n, t * n
  unfold ES
  have h1 : 3 * t ≥ 1 := by omega
  subst hn_eq
  zify [h1]
  ring

/-- n ≡ 0 (mod 5) implies ES has solution -/
theorem coverage_mod5_r0 (n : Nat) (hn : n > 0) (hmod : n % 5 = 0) :
    ∃ x y z, ES n x y z := by
  use 2 * n / 5, n, 2 * n
  unfold ES
  have hdiv : 5 ∣ n := Nat.dvd_of_mod_eq_zero hmod
  obtain ⟨k, hk⟩ := hdiv
  have hk_pos : k > 0 := by omega
  have hdiv5 : 2 * (5 * k) / 5 = 2 * k := by omega
  calc 4 * (2 * n / 5) * n * (2 * n)
      = 4 * (2 * (5 * k) / 5) * (5 * k) * (2 * (5 * k)) := by rw [hk]
    _ = 4 * (2 * k) * (5 * k) * (10 * k) := by rw [hdiv5]; ring
    _ = (5 * k) * ((2 * k) * (5 * k) + (2 * k) * (10 * k) + (5 * k) * (10 * k)) := by ring
    _ = (5 * k) * ((2 * (5 * k) / 5) * (5 * k) + (2 * (5 * k) / 5) * (2 * (5 * k)) + (5 * k) * (2 * (5 * k))) := by rw [hdiv5]; ring
    _ = n * (2 * n / 5 * n + 2 * n / 5 * (2 * n) + n * (2 * n)) := by rw [hk]

/-! ## Main Coverage Theorem -/

/-- Main theorem: For any n > 0, if n % 60 is in CoveredSet60, then ES has a solution.
    This covers 36/60 = 60% of all residue classes. -/
theorem coverage_mod_60 (n : Nat) (hn : n > 0)
    (h : n % 60 = 0 ∨ n % 60 = 2 ∨ n % 60 = 4 ∨ n % 60 = 5 ∨ n % 60 = 8 ∨ 
         n % 60 = 10 ∨ n % 60 = 11 ∨ n % 60 = 12 ∨ n % 60 = 14 ∨ n % 60 = 15 ∨
         n % 60 = 16 ∨ n % 60 = 17 ∨ n % 60 = 20 ∨ n % 60 = 23 ∨ n % 60 = 24 ∨
         n % 60 = 25 ∨ n % 60 = 26 ∨ n % 60 = 28 ∨ n % 60 = 29 ∨ n % 60 = 30 ∨
         n % 60 = 32 ∨ n % 60 = 35 ∨ n % 60 = 36 ∨ n % 60 = 38 ∨ n % 60 = 40 ∨
         n % 60 = 41 ∨ n % 60 = 44 ∨ n % 60 = 45 ∨ n % 60 = 47 ∨ n % 60 = 48 ∨
         n % 60 = 50 ∨ n % 60 = 52 ∨ n % 60 = 53 ∨ n % 60 = 55 ∨ n % 60 = 56 ∨
         n % 60 = 59) :
    ∃ x y z, ES n x y z := by
  rcases h with h0 | h2 | h4 | h5 | h8 | h10 | h11 | h12 | h14 | h15 |
               h16 | h17 | h20 | h23 | h24 | h25 | h26 | h28 | h29 | h30 |
               h32 | h35 | h36 | h38 | h40 | h41 | h44 | h45 | h47 | h48 |
               h50 | h52 | h53 | h55 | h56 | h59
  -- mod 4 = 0 cases: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56
  all_goals first
    | exact coverage_mod4_r0 n hn (by omega)
    | exact coverage_mod3_r2 n hn (by omega)
    | exact coverage_mod5_r0 n hn (by omega)

/-! ## Coverage Statistics -/

/-- Number of covered residues mod 60 -/
theorem covered_count : CoveredSet60.length = 36 := by native_decide

/-- Number of escape residues mod 60 -/
theorem escape_count : EscapeSet60.length = 24 := by native_decide

/-- Coverage percentage: 36/60 = 60% -/
theorem coverage_percentage : 36 * 100 / 60 = 60 := by native_decide


/-! ## Additional Coverage Theorems -/

/-- n ≡ 0 (mod 3) implies ES has solution -/
theorem coverage_mod3_r0 (n : Nat) (hn : n > 0) (hmod : n % 3 = 0) :
    ∃ x y z, ES n x y z := by
  use n / 3, n + 1, n * (n + 1)
  unfold ES
  have hdiv : 3 ∣ n := Nat.dvd_of_mod_eq_zero hmod
  obtain ⟨k, hk⟩ := hdiv
  have hk_pos : k > 0 := by omega
  have hdivk : (3 * k) / 3 = k := by omega
  calc 4 * (n / 3) * (n + 1) * (n * (n + 1))
      = 4 * ((3 * k) / 3) * (3 * k + 1) * (3 * k * (3 * k + 1)) := by rw [hk]
    _ = 4 * k * (3 * k + 1) * (3 * k * (3 * k + 1)) := by rw [hdivk]
    _ = (3 * k) * (k * (3 * k + 1) + k * (3 * k * (3 * k + 1)) + (3 * k + 1) * (3 * k * (3 * k + 1))) := by ring
    _ = (3 * k) * (((3 * k) / 3) * (3 * k + 1) + ((3 * k) / 3) * (3 * k * (3 * k + 1)) + (3 * k + 1) * (3 * k * (3 * k + 1))) := by rw [hdivk]
    _ = n * (n / 3 * (n + 1) + n / 3 * (n * (n + 1)) + (n + 1) * (n * (n + 1))) := by rw [hk]

/-- n ≡ 0 (mod 7) implies ES has solution -/
theorem coverage_mod7_r0 (n : Nat) (hn : n > 0) (hmod : n % 7 = 0) :
    ∃ x y z, ES n x y z := by
  use 2 * n / 7, 2 * n + 1, 2 * n * (2 * n + 1)
  unfold ES
  have hdiv : 7 ∣ n := Nat.dvd_of_mod_eq_zero hmod
  obtain ⟨k, hk⟩ := hdiv
  have hk_pos : k > 0 := by omega
  have hdiv7 : 2 * (7 * k) / 7 = 2 * k := by omega
  have h2n : 2 * (7 * k) = 14 * k := by ring
  calc 4 * (2 * n / 7) * (2 * n + 1) * (2 * n * (2 * n + 1))
      = 4 * (2 * (7 * k) / 7) * (2 * (7 * k) + 1) * (2 * (7 * k) * (2 * (7 * k) + 1)) := by rw [hk]
    _ = 4 * (2 * k) * (14 * k + 1) * (14 * k * (14 * k + 1)) := by rw [hdiv7, h2n]
    _ = (7 * k) * ((2 * k) * (14 * k + 1) + (2 * k) * (14 * k * (14 * k + 1)) + (14 * k + 1) * (14 * k * (14 * k + 1))) := by ring
    _ = (7 * k) * ((2 * (7 * k) / 7) * (14 * k + 1) + (2 * (7 * k) / 7) * (14 * k * (14 * k + 1)) + (14 * k + 1) * (14 * k * (14 * k + 1))) := by rw [hdiv7]
    _ = (7 * k) * ((2 * (7 * k) / 7) * (2 * (7 * k) + 1) + (2 * (7 * k) / 7) * (2 * (7 * k) * (2 * (7 * k) + 1)) + (2 * (7 * k) + 1) * (2 * (7 * k) * (2 * (7 * k) + 1))) := by rw [h2n]
    _ = n * (2 * n / 7 * (2 * n + 1) + 2 * n / 7 * (2 * n * (2 * n + 1)) + (2 * n + 1) * (2 * n * (2 * n + 1))) := by rw [hk]

/-! ## Extended Coverage Statistics for M = 420 -/

/-- 420 = lcm(3, 4, 5, 7) -/
theorem lcm_3_4_5_7 : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 7 = 420 := by native_decide

/-- With 5 certificates, we cover 348/420 = 82.86% of residue classes mod 420.
    Certificates: mod4 r=0, mod3 r=2, mod5 r=0, mod3 r=0, mod7 r=0 -/
theorem coverage_mod_420_summary : 348 * 1000 / 420 = 828 := by native_decide
