/-
  Erdős–Straus Conjecture: Gap Certificates

  Gap residues mod 420:
  {1, 73, 97, 121, 169, 193, 241, 289, 313, 337, 361, 409}.

  This file contains fully formalized (division-free) certificates for:
  - r = 73, 97, 193, 241, 313, 337, 409

  and odd-k (k = 2m+1) certificates (t=3, d=2) for:
  - r = 1, 121, 169, 289, 361

  Remaining work: even-k cases for r = 1, 121, 169, 289, 361.
-/

import Mathlib.Tactic

namespace GapCertificates

-- ES equation: 4/n = 1/x + 1/y + 1/z  ⟺  4xyz = n(xy + xz + yz)
def ES (n x y z : ℕ) : Prop := 4 * x * y * z = n * (x * y + x * z + y * z)

/-!
## Division-free 2-unit-fraction certificate lemma

Let `t = 4x - n`. If we can find `d` and positive integers `y,z` such that

  n + t = 4x
  n*x + d = t*y
  n*x*y = d*z

then `ES n x y z` holds.
-/
lemma es_of_two_uf (n x y z t d : ℕ)
    (hx : n + t = 4 * x)
    (hy : n * x + d = t * y)
    (hz : n * x * y = d * z) : ES n x y z := by
  unfold ES
  calc
    4 * x * y * z = (4 * x) * y * z := by ring
    _ = (n + t) * y * z := by simp [hx.symm]
    _ = n * y * z + t * y * z := by ring
    _ = n * y * z + (n * x + d) * z := by
      have : t * y * z = (n * x + d) * z := by
        calc
          t * y * z = (t * y) * z := by ring
          _ = (n * x + d) * z := by simp [hy.symm]
      simp [this]
    _ = n * y * z + n * x * z + d * z := by ring
    _ = n * y * z + n * x * z + n * x * y := by simp [hz.symm]
    _ = n * (x * y + x * z + y * z) := by ring

-- ============================================================
-- Fully proven gap residues
-- ============================================================

-- r=97: single certificate (t=3, d=5) for all k
theorem gap_cert_97 (k : ℕ) :
    ∃ x y z, ES (420 * k + 97) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 420 * k + 97
  let x : ℕ := 105 * k + 25
  let y : ℕ := 14700 * k ^ 2 + 6895 * k + 810
  -- since x = 5*(21k+5), avoid division by defining z as n*(x/5)*y
  let z : ℕ := n * (21 * k + 5) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 5 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 5 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 5 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

-- r=337: single certificate (t=3, d=5) for all k
theorem gap_cert_337 (k : ℕ) :
    ∃ x y z, ES (420 * k + 337) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 420 * k + 337
  let x : ℕ := 105 * k + 85
  let y : ℕ := 14700 * k ^ 2 + 23695 * k + 9550
  let z : ℕ := n * (21 * k + 17) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 5 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 5 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 5 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

-- r=73: parity split (even k: t=7,d=10; odd k: t=3,d=2)
theorem gap_cert_73_even (m : ℕ) :
    ∃ x y z, ES (840 * m + 73) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 73
  let x : ℕ := 210 * m + 20
  let y : ℕ := 25200 * m ^ 2 + 4590 * m + 210
  -- x = 10*(21m+2)
  let z : ℕ := n * (21 * m + 2) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 7 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 10 = 7 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 10 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 7 10 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_73_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 493) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 493
  let x : ℕ := 210 * m + 124
  let y : ℕ := 58800 * m ^ 2 + 69230 * m + 20378
  -- x = 2*(105m+62)
  let z : ℕ := n * (105 * m + 62) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_73 (k : ℕ) :
    ∃ x y z, ES (420 * k + 73) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk | hk
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (m + m) + 73 = 840 * m + 73 := by ring
    simpa [hn] using (gap_cert_73_even m)
  · rcases hk with ⟨m, rfl⟩
    -- k = m+m+1
    have hn : 420 * (2 * m + 1) + 73 = 840 * m + 493 := by ring
    simpa [hn] using (gap_cert_73_odd m)

-- r=193: parity split (even k: t=7,d=10; odd k: t=3,d=2)
theorem gap_cert_193_even (m : ℕ) :
    ∃ x y z, ES (840 * m + 193) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 193
  let x : ℕ := 210 * m + 50
  let y : ℕ := 25200 * m ^ 2 + 11790 * m + 1380
  -- x = 10*(21m+5)
  let z : ℕ := n * (21 * m + 5) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 7 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 10 = 7 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 10 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 7 10 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_193_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 613) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 613
  let x : ℕ := 210 * m + 154
  let y : ℕ := 58800 * m ^ 2 + 86030 * m + 31468
  -- x = 2*(105m+77)
  let z : ℕ := n * (105 * m + 77) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_193 (k : ℕ) :
    ∃ x y z, ES (420 * k + 193) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk | hk
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (m + m) + 193 = 840 * m + 193 := by ring
    simpa [hn] using (gap_cert_193_even m)
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 193 = 840 * m + 613 := by ring
    simpa [hn] using (gap_cert_193_odd m)

-- r=313: parity split (even k: t=7,d=20; odd k: t=3,d=2)
theorem gap_cert_313_even (m : ℕ) :
    ∃ x y z, ES (840 * m + 313) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 313
  let x : ℕ := 210 * m + 80
  let y : ℕ := 25200 * m ^ 2 + 18990 * m + 3580
  -- x = 10*(21m+8) and y is even, so z = n*(x/10)*(y/2)
  let y₂ : ℕ := 12600 * m ^ 2 + 9495 * m + 1790
  let z : ℕ := n * (21 * m + 8) * y₂
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 7 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 20 = 7 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 20 * z := by simp [n, x, y, y₂, z]; ring
    exact es_of_two_uf n x y z 7 20 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y₂, n]; nlinarith

theorem gap_cert_313_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 733) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 733
  let x : ℕ := 210 * m + 184
  let y : ℕ := 58800 * m ^ 2 + 102830 * m + 44958
  -- x = 2*(105m+92)
  let z : ℕ := n * (105 * m + 92) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_313 (k : ℕ) :
    ∃ x y z, ES (420 * k + 313) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk | hk
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (m + m) + 313 = 840 * m + 313 := by ring
    simpa [hn] using (gap_cert_313_even m)
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 313 = 840 * m + 733 := by ring
    simpa [hn] using (gap_cert_313_odd m)

-- r=241: single explicit certificate for all k
theorem gap_cert_241 (k : ℕ) :
    ∃ x y z, ES (420 * k + 241) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 420 * k + 241
  let x : ℕ := 105 * k + 63
  let y : ℕ := 2 * n * x
  let z : ℕ := 2 * n * (5 * k + 3)
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · simp [ES, n, x, y, z]; ring
  · dsimp [x]; nlinarith
  · dsimp [y, n, x]; nlinarith
  · dsimp [z, n]; nlinarith

-- r=409: parity split with a finite mod-11 split in the even case
theorem gap_cert_409_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 829) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 829
  let x : ℕ := 210 * m + 208
  let y : ℕ := 58800 * m ^ 2 + 116270 * m + 57478
  -- x = 2*(105m+104)
  let z : ℕ := n * (105 * m + 104) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_409_even (m : ℕ) :
    ∃ x y z, ES (840 * m + 409) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let r : ℕ := m % 11
  have hr : r < 11 := by simpa [r] using Nat.mod_lt m (by decide : 0 < 11)
  let j : ℕ := m / 11
  have hm : m = 11 * j + r := by
    -- Nat.div_add_mod gives: j*11 + r = m
    simpa [j, r, Nat.mul_comm, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      (Nat.div_add_mod m 11).symm
  -- Reduce to a statement with `m = 11*j + r` and then split on r.
  have h :
      ∃ x y z, ES (840 * (11 * j + r) + 409) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
    interval_cases r
    · -- r=0: (t=11, d=21)
      let n : ℕ := 840 * (11 * j + 0) + 409
      let x : ℕ := 2310 * j + 105
      let y : ℕ := 1940400 * j ^ 2 + 174090 * j + 3906
      let z : ℕ := n * (110 * j + 5) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 21 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 21 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 21 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
    · -- r=1: (t=11, d=35)
      let n : ℕ := 840 * (11 * j + 1) + 409
      let x : ℕ := 2310 * j + 315
      let y : ℕ := 1940400 * j ^ 2 + 526890 * j + 35770
      let z : ℕ := n * (66 * j + 9) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 35 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 35 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 35 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
    · -- r=2: (t=11, d=63) with y = 3*y₃
      let n : ℕ := 840 * (11 * j + 2) + 409
      let x : ℕ := 2310 * j + 525
      let y₃ : ℕ := 646800 * j ^ 2 + 293230 * j + 33236
      let y : ℕ := 3 * y₃
      let z : ℕ := n * (110 * j + 25) * y₃
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 63 = 11 * y := by simp [n, x, y, y₃]; ring
        have hz : n * x * y = 63 * z := by simp [n, x, y, y₃, z]; ring
        exact es_of_two_uf n x y z 11 63 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y, y₃]; nlinarith
      · dsimp [z, y₃, n]; nlinarith
    · -- r=3: (t=11, d=105) with y = 5*y₅
      let n : ℕ := 840 * (11 * j + 3) + 409
      let x : ℕ := 2310 * j + 735
      let y₅ : ℕ := 388080 * j ^ 2 + 246498 * j + 39144
      let y : ℕ := 5 * y₅
      let z : ℕ := n * (110 * j + 35) * y₅
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 105 = 11 * y := by simp [n, x, y, y₅]; ring
        have hz : n * x * y = 105 * z := by simp [n, x, y, y₅, z]; ring
        exact es_of_two_uf n x y z 11 105 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y, y₅]; nlinarith
      · dsimp [z, y₅, n]; nlinarith
    · -- r=4: (t=11, d=7)
      let n : ℕ := 840 * (11 * j + 4) + 409
      let x : ℕ := 2310 * j + 945
      let y : ℕ := 1940400 * j ^ 2 + 1585290 * j + 323792
      let z : ℕ := n * (330 * j + 135) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 7 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 7 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 7 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
    · -- r=5: (t=3, d=11) with n = 11*n₁₁
      let n : ℕ := 840 * (11 * j + 5) + 409
      let n₁₁ : ℕ := 840 * j + 419
      let x : ℕ := 2310 * j + 1153
      let y : ℕ := 7114800 * j ^ 2 + 7100170 * j + 1771396
      let z : ℕ := n₁₁ * x * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 3 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 11 = 3 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 11 * z := by simp [n, n₁₁, x, y, z]; ring
        exact es_of_two_uf n x y z 3 11 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, n₁₁, x, y, n]; nlinarith
    · -- r=6: (t=11, d=7)
      let n : ℕ := 840 * (11 * j + 6) + 409
      let x : ℕ := 2310 * j + 1365
      let y : ℕ := 1940400 * j ^ 2 + 2290890 * j + 676172
      let z : ℕ := n * (330 * j + 195) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 7 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 7 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 7 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
    · -- r=7: (t=3, d=11) with x = 11*x₁₁
      let n : ℕ := 840 * (11 * j + 7) + 409
      let x : ℕ := 2310 * j + 1573
      let x₁₁ : ℕ := 210 * j + 143
      let y : ℕ := 7114800 * j ^ 2 + 9687370 * j + 3297536
      let z : ℕ := n * x₁₁ * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 3 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 11 = 3 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 11 * z := by simp [n, x, x₁₁, y, z]; ring
        exact es_of_two_uf n x y z 3 11 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, x₁₁, y, n]; nlinarith
    · -- r=8: (t=11, d=63) with y = 3*y₃
      let n : ℕ := 840 * (11 * j + 8) + 409
      let x : ℕ := 2310 * j + 1785
      let y₃ : ℕ := 646800 * j ^ 2 + 998830 * j + 385616
      let y : ℕ := 3 * y₃
      let z : ℕ := n * (110 * j + 85) * y₃
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 63 = 11 * y := by simp [n, x, y, y₃]; ring
        have hz : n * x * y = 63 * z := by simp [n, x, y, y₃, z]; ring
        exact es_of_two_uf n x y z 11 63 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y, y₃]; nlinarith
      · dsimp [z, y₃, n]; nlinarith
    · -- r=9: (t=11, d=35)
      let n : ℕ := 840 * (11 * j + 9) + 409
      let x : ℕ := 2310 * j + 1995
      let y : ℕ := 1940400 * j ^ 2 + 3349290 * j + 1445290
      let z : ℕ := n * (66 * j + 57) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 35 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 35 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 35 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
    · -- r=10: (t=11, d=21)
      let n : ℕ := 840 * (11 * j + 10) + 409
      let x : ℕ := 2310 * j + 2205
      let y : ℕ := 1940400 * j ^ 2 + 3702090 * j + 1765806
      let z : ℕ := n * (110 * j + 105) * y
      refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
      · have hx : n + 11 = 4 * x := by simp [n, x]; ring
        have hy : n * x + 21 = 11 * y := by simp [n, x, y]; ring
        have hz : n * x * y = 21 * z := by simp [n, x, y, z]; ring
        exact es_of_two_uf n x y z 11 21 hx hy hz
      · dsimp [x]; nlinarith
      · dsimp [y]; nlinarith
      · dsimp [z, y, n]; nlinarith
  simpa [hm] using h

theorem gap_cert_409 (k : ℕ) :
    ∃ x y z, ES (420 * k + 409) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk | hk
  · rcases hk with ⟨m, rfl⟩
    -- k = m+m
    have hn : 420 * (m + m) + 409 = 840 * m + 409 := by ring
    simpa [hn] using (gap_cert_409_even m)
  · rcases hk with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 409 = 840 * m + 829 := by ring
    simpa [hn] using (gap_cert_409_odd m)

-- ============================================================
-- Odd-k certificates for the remaining gap residues
-- ============================================================

theorem gap_cert_1_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 421) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 421
  let x : ℕ := 210 * m + 106
  let y : ℕ := 58800 * m ^ 2 + 59150 * m + 14876
  let z : ℕ := n * (105 * m + 53) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_121_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 541) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 541
  let x : ℕ := 210 * m + 136
  let y : ℕ := 58800 * m ^ 2 + 75950 * m + 24526
  let z : ℕ := n * (105 * m + 68) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_169_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 589) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 589
  let x : ℕ := 210 * m + 148
  let y : ℕ := 58800 * m ^ 2 + 82670 * m + 29058
  let z : ℕ := n * (105 * m + 74) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_289_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 709) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 709
  let x : ℕ := 210 * m + 178
  let y : ℕ := 58800 * m ^ 2 + 99470 * m + 42068
  let z : ℕ := n * (105 * m + 89) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

theorem gap_cert_361_odd (m : ℕ) :
    ∃ x y z, ES (840 * m + 781) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  let n : ℕ := 840 * m + 781
  let x : ℕ := 210 * m + 196
  let y : ℕ := 58800 * m ^ 2 + 109550 * m + 51026
  let z : ℕ := n * (105 * m + 98) * y
  refine ⟨x, y, z, ?_, ?_, ?_, ?_⟩
  · have hx : n + 3 = 4 * x := by simp [n, x]; ring
    have hy : n * x + 2 = 3 * y := by simp [n, x, y]; ring
    have hz : n * x * y = 2 * z := by simp [n, x, y, z]; ring
    exact es_of_two_uf n x y z 3 2 hx hy hz
  · dsimp [x]; nlinarith
  · dsimp [y]; nlinarith
  · dsimp [z, y, n]; nlinarith

-- ============================================================
-- Remaining even-k cases (placeholders)
-- ============================================================

theorem gap_cert_1 (k : ℕ) (hk : k > 0) :
    ∃ x y z, ES (420 * k + 1) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk' | hk'
  · -- even k (hard)
    -- n = 840m + 1 : open/unfinished
    sorry
  · rcases hk' with ⟨m, rfl⟩
    -- k = m+m+1
    have hn : 420 * (2 * m + 1) + 1 = 840 * m + 421 := by ring
    simpa [hn] using (gap_cert_1_odd m)

theorem gap_cert_121 (k : ℕ) :
    ∃ x y z, ES (420 * k + 121) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk' | hk'
  · -- even k (hard)
    sorry
  · rcases hk' with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 121 = 840 * m + 541 := by ring
    simpa [hn] using (gap_cert_121_odd m)

theorem gap_cert_169 (k : ℕ) :
    ∃ x y z, ES (420 * k + 169) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk' | hk'
  · -- even k (hard)
    sorry
  · rcases hk' with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 169 = 840 * m + 589 := by ring
    simpa [hn] using (gap_cert_169_odd m)

theorem gap_cert_289 (k : ℕ) :
    ∃ x y z, ES (420 * k + 289) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk' | hk'
  · -- even k (hard)
    sorry
  · rcases hk' with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 289 = 840 * m + 709 := by ring
    simpa [hn] using (gap_cert_289_odd m)

theorem gap_cert_361 (k : ℕ) :
    ∃ x y z, ES (420 * k + 361) x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 := by
  rcases Nat.even_or_odd k with hk' | hk'
  · -- even k (hard)
    sorry
  · rcases hk' with ⟨m, rfl⟩
    have hn : 420 * (2 * m + 1) + 361 = 840 * m + 781 := by ring
    simpa [hn] using (gap_cert_361_odd m)

end GapCertificates
