/-
  BoundedSearch.lean - Provably complete bounded search for Erdős-Straus conjecture
  With Mathlib for tactic support
-/

import Mathlib.Tactic

/-! ## Main Definitions -/

def ES (n x y z : Nat) : Prop := 4 * x * y * z = n * (x * y + x * z + y * z)
def OrderedES (n x y z : Nat) : Prop := ES n x y z ∧ x ≤ y ∧ y ≤ z
def HasES (n : Nat) : Prop := ∃ x y z, ES n x y z
def HasOrderedES (n : Nat) : Prop := ∃ x y z, OrderedES n x y z

/-- Positive ordered solution (excludes trivial (0,0,0)) -/
def PositiveOrderedES (n x y z : Nat) : Prop := 
  ES n x y z ∧ x ≤ y ∧ y ≤ z ∧ x > 0 ∧ y > 0 ∧ z > 0

def HasPositiveOrderedES (n : Nat) : Prop := ∃ x y z, PositiveOrderedES n x y z

/-! ## Symmetry Lemmas -/

theorem ES_swap_xy (n x y z : Nat) : ES n x y z ↔ ES n y x z := by
  unfold ES; constructor <;> (intro h; ring_nf at h ⊢; omega)

theorem ES_swap_yz (n x y z : Nat) : ES n x y z ↔ ES n x z y := by
  unfold ES; constructor <;> (intro h; ring_nf at h ⊢; omega)

theorem ES_swap_xz (n x y z : Nat) : ES n x y z ↔ ES n z y x := by
  unfold ES; constructor <;> (intro h; ring_nf at h ⊢; omega)

/-! ## WLOG Lemma -/

theorem ES_wlog (n x y z : Nat) (h : ES n x y z) : HasOrderedES n := by
  by_cases hxy : x ≤ y <;> by_cases hyz : y ≤ z <;> by_cases hxz : x ≤ z
  · exact ⟨x, y, z, h, hxy, hyz⟩
  · omega
  · exact ⟨x, z, y, (ES_swap_yz n x y z).mp h, hxz, Nat.le_of_lt (Nat.lt_of_not_le hyz)⟩
  · have hzx : z < x := Nat.lt_of_not_le hxz
    have h' := (ES_swap_xz n x y z).mp h
    have h'' := (ES_swap_yz n z y x).mp h'
    exact ⟨z, x, y, h'', Nat.le_of_lt hzx, hxy⟩
  · exact ⟨y, x, z, (ES_swap_xy n x y z).mp h, Nat.le_of_lt (Nat.lt_of_not_le hxy), hxz⟩
  · have hzx : z < x := Nat.lt_of_not_le hxz
    have h' := (ES_swap_xy n x y z).mp h
    have h'' := (ES_swap_yz n y x z).mp h'
    exact ⟨y, z, x, h'', hyz, Nat.le_of_lt hzx⟩
  · omega
  · have h' := (ES_swap_xz n x y z).mp h
    exact ⟨z, y, x, h', Nat.le_of_lt (Nat.lt_of_not_le hyz), Nat.le_of_lt (Nat.lt_of_not_le hxy)⟩

/-- WLOG with positivity preservation -/
theorem ES_wlog_positive (n x y z : Nat) (h : ES n x y z) 
    (hx : x > 0) (hy : y > 0) (hz : z > 0) : HasPositiveOrderedES n := by
  by_cases hxy : x ≤ y <;> by_cases hyz : y ≤ z <;> by_cases hxz : x ≤ z
  · exact ⟨x, y, z, h, hxy, hyz, hx, hy, hz⟩
  · omega
  · exact ⟨x, z, y, (ES_swap_yz n x y z).mp h, hxz, Nat.le_of_lt (Nat.lt_of_not_le hyz), hx, hz, hy⟩
  · have hzx : z < x := Nat.lt_of_not_le hxz
    have h' := (ES_swap_xz n x y z).mp h
    have h'' := (ES_swap_yz n z y x).mp h'
    exact ⟨z, x, y, h'', Nat.le_of_lt hzx, hxy, hz, hx, hy⟩
  · exact ⟨y, x, z, (ES_swap_xy n x y z).mp h, Nat.le_of_lt (Nat.lt_of_not_le hxy), hxz, hy, hx, hz⟩
  · have hzx : z < x := Nat.lt_of_not_le hxz
    have h' := (ES_swap_xy n x y z).mp h
    have h'' := (ES_swap_yz n y x z).mp h'
    exact ⟨y, z, x, h'', hyz, Nat.le_of_lt hzx, hy, hz, hx⟩
  · omega
  · have h' := (ES_swap_xz n x y z).mp h
    exact ⟨z, y, x, h', Nat.le_of_lt (Nat.lt_of_not_le hyz), Nat.le_of_lt (Nat.lt_of_not_le hxy), hz, hy, hx⟩

/-! ## X-Bounds for Ordered Solutions -/

theorem x_lower_bound (n x y z : Nat) (hES : OrderedES n x y z) 
    (hx : x > 0) (hy : y > 0) (hz : z > 0) : n < 4 * x := by
  obtain ⟨hes, _, _⟩ := hES
  unfold ES at hes
  by_contra h_neg; push_neg at h_neg
  have h1 : 4 * x * y * z = n * x * y + n * x * z + n * y * z := by ring_nf at hes ⊢; omega
  have h2 : n * y * z ≥ 4 * x * y * z := by
    have := Nat.mul_le_mul_right (y * z) h_neg; ring_nf at this ⊢; omega
  have h7 : n * x * y + n * x * z = 0 := by omega
  have hn : n > 0 := by
    by_contra hnn; push_neg at hnn
    have hn0 : n = 0 := Nat.le_zero.mp hnn
    rw [hn0] at h1; simp at h1
    have hxyz : x * y * z > 0 := Nat.mul_pos (Nat.mul_pos hx hy) hz
    omega
  have h8 : n * x * y > 0 := Nat.mul_pos (Nat.mul_pos hn hx) hy
  omega

theorem x_upper_bound (n x y z : Nat) (hES : OrderedES n x y z)
    (_hx : x > 0) (hy : y > 0) (hz : z > 0) : 4 * x ≤ 3 * n := by
  obtain ⟨hes, hxy, hyz⟩ := hES
  unfold ES at hes
  have h1 : 4 * x * y * z = n * x * y + n * x * z + n * y * z := by ring_nf at hes ⊢; omega
  have hxz : x ≤ z := Nat.le_trans hxy hyz
  have hxy_yz : x * y ≤ y * z := by
    calc x * y ≤ z * y := Nat.mul_le_mul_right y hxz
      _ = y * z := by ring
  have hxz_yz : x * z ≤ y * z := Nat.mul_le_mul_right z hxy
  have h2 : x * y + x * z + y * z ≤ 3 * y * z := by
    calc x * y + x * z + y * z 
        ≤ y * z + y * z + y * z := by omega
      _ = 3 * y * z := by ring
  have h3 : 4 * x * y * z ≤ 3 * n * y * z := by
    have h3' : n * (x * y + x * z + y * z) ≤ n * (3 * y * z) := Nat.mul_le_mul_left n h2
    have heq : n * (3 * y * z) = 3 * n * y * z := by ring
    calc 4 * x * y * z = n * (x * y + x * z + y * z) := hes
      _ ≤ n * (3 * y * z) := h3'
      _ = 3 * n * y * z := heq
  have hyz_pos : y * z > 0 := Nat.mul_pos hy hz
  have h4 : 4 * x * (y * z) ≤ 3 * n * (y * z) := by
    have he1 : 4 * x * y * z = 4 * x * (y * z) := by ring
    have he2 : 3 * n * y * z = 3 * n * (y * z) := by ring
    omega
  exact Nat.le_of_mul_le_mul_right h4 hyz_pos

/-! ## Y-Bounds for Ordered Solutions -/

theorem ES_sum_relation (n x y z : Nat) (h : ES n x y z) (ha : 4 * x > n) :
    (4 * x - n) * y * z = n * x * (y + z) := by
  unfold ES at h
  have h_le : n ≤ 4 * x := Nat.le_of_lt ha
  have h1 : 4 * x * y * z = n * x * y + n * x * z + n * y * z := by ring_nf at h ⊢; omega
  have h2 : n * y * z ≤ 4 * x * y * z := by
    have := Nat.mul_le_mul_right (y * z) h_le; ring_nf at this ⊢; omega
  have h3 : 4 * x * y * z - n * y * z = n * x * y + n * x * z := by omega
  have h4 : (4 * x - n) * y * z = 4 * x * y * z - n * y * z := by
    have hsub : (4 * x - n) * (y * z) = 4 * x * (y * z) - n * (y * z) := Nat.sub_mul (4 * x) n (y * z)
    have h4a : 4 * x * (y * z) = 4 * x * y * z := by ring
    have h4b : n * (y * z) = n * y * z := by ring
    rw [h4a, h4b] at hsub
    have h4c : (4 * x - n) * y * z = (4 * x - n) * (y * z) := by ring
    rw [h4c, hsub]
  have h5 : n * x * (y + z) = n * x * y + n * x * z := by ring
  omega

theorem y_lower_bound (n x y z : Nat) (hES : OrderedES n x y z)
    (ha : 4 * x > n) (hz : z > 0) : y * (4 * x - n) ≥ n * x := by
  obtain ⟨hes, _, _⟩ := hES
  have hsum := ES_sum_relation n x y z hes ha
  have h1 : (4 * x - n) * y * z ≥ n * x * z := by
    rw [hsum]
    have hexp : n * x * (y + z) = n * x * y + n * x * z := by ring
    omega
  have h2 : (4 * x - n) * y * z = y * (4 * x - n) * z := by ring
  rw [h2] at h1
  exact Nat.le_of_mul_le_mul_right h1 hz

theorem y_upper_bound (n x y z : Nat) (hES : OrderedES n x y z)
    (ha : 4 * x > n) (hz : z > 0) : y * (4 * x - n) ≤ 2 * n * x := by
  obtain ⟨hes, _, hyz⟩ := hES
  have hsum := ES_sum_relation n x y z hes ha
  have h_yz_le : y + z ≤ 2 * z := by omega
  have h1 : (4 * x - n) * y * z ≤ 2 * n * x * z := by
    rw [hsum]
    calc n * x * (y + z) ≤ n * x * (2 * z) := Nat.mul_le_mul_left (n * x) h_yz_le
      _ = 2 * n * x * z := by ring
  have h2 : (4 * x - n) * y * z = y * (4 * x - n) * z := by ring
  rw [h2] at h1
  exact Nat.le_of_mul_le_mul_right h1 hz

/-! ## Z-Formula -/

theorem z_formula (n x y z : Nat) (hES : ES n x y z)
    (ha : 4 * x > n) (hb : (4 * x - n) * y > n * x) :
    z * ((4 * x - n) * y - n * x) = n * x * y := by
  unfold ES at hES
  have h_le1 : n ≤ 4 * x := Nat.le_of_lt ha
  have h_le2 : n * x ≤ (4 * x - n) * y := Nat.le_of_lt hb
  have hexp : 4 * x * y * z = n * x * y + n * x * z + n * y * z := by ring_nf at hES ⊢; omega
  suffices h : z * ((4 * x - n) * y - n * x) + n * x * z + n * y * z = 
               n * x * y + n * x * z + n * y * z by
    exact Nat.add_right_cancel (Nat.add_right_cancel h)
  rw [← hexp]
  have h_sub1 : (4 * x - n) * y = 4 * x * y - n * y := Nat.sub_mul (4 * x) n y
  have h_nxz_le : z * (n * x) ≤ z * ((4 * x - n) * y) := Nat.mul_le_mul_left z h_le2
  have h1 : z * ((4 * x - n) * y - n * x) = z * ((4 * x - n) * y) - z * (n * x) := 
    Nat.mul_sub z ((4 * x - n) * y) (n * x)
  have h2 : z * ((4 * x - n) * y) = z * (4 * x * y - n * y) := by rw [h_sub1]
  have h3 : z * (4 * x * y - n * y) = z * (4 * x * y) - z * (n * y) := Nat.mul_sub z _ _
  have h4 : z * ((4 * x - n) * y) = z * (4 * x * y) - z * (n * y) := by rw [h2, h3]
  have h6 : z * ((4 * x - n) * y - n * x) = z * (4 * x * y) - z * (n * y) - z * (n * x) := by
    rw [h1, h4]
  calc z * ((4 * x - n) * y - n * x) + n * x * z + n * y * z
      = (z * (4 * x * y) - z * (n * y) - z * (n * x)) + n * x * z + n * y * z := by rw [h6]
    _ = (4 * x * y * z - n * y * z - n * x * z) + n * x * z + n * y * z := by ring_nf
    _ = 4 * x * y * z := by omega

/-! ## Decidable ES Checker -/

def isValidSolution (n x y z : Nat) : Bool :=
  x > 0 && y > 0 && z > 0 && x ≤ y && y ≤ z && 4 * x * y * z = n * (x * y + x * z + y * z)

theorem isValidSolution_implies_ES (n x y z : Nat) (h : isValidSolution n x y z = true) : ES n x y z := by
  unfold isValidSolution at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  unfold ES; exact h.2

theorem orderedES_isValid (n x y z : Nat) (hES : OrderedES n x y z)
    (hx : x > 0) (hy : y > 0) (hz : z > 0) : isValidSolution n x y z = true := by
  unfold isValidSolution
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  obtain ⟨hes, hxy, hyz⟩ := hES
  unfold ES at hes
  exact ⟨⟨⟨⟨⟨hx, hy⟩, hz⟩, hxy⟩, hyz⟩, hes⟩

/-! ## List-Based Bounded Search -/

def computeZ (n x y : Nat) : Option Nat :=
  if 4 * x > n then
    let denom := (4 * x - n) * y - n * x
    if denom > 0 then
      let numer := n * x * y
      if numer % denom = 0 then some (numer / denom)
      else none
    else none
  else none

def tryTriple (n x y : Nat) : Option (Nat × Nat × Nat) :=
  match computeZ n x y with
  | some z => if isValidSolution n x y z then some (x, y, z) else none
  | none => none

theorem tryTriple_valid (n x y a b c : Nat) (h : tryTriple n x y = some (a, b, c)) : 
    isValidSolution n a b c = true := by
  unfold tryTriple at h
  split at h
  · split at h
    · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h; assumption
    · simp at h
  · simp at h

def candidatePairs (n : Nat) : List (Nat × Nat) :=
  let x_min := (n + 3) / 4
  let x_max := 3 * n / 4
  (List.range' x_min (x_max - x_min + 1)).flatMap fun x =>
    if 4 * x > n then
      let y_min := x
      let y_max := 2 * n * x / (4 * x - n) + 1
      (List.range' y_min (y_max - y_min + 1)).map fun y => (x, y)
    else []

def boundedSearch (n : Nat) : Option (Nat × Nat × Nat) :=
  if n ≤ 1 then none
  else (candidatePairs n).findSome? fun (x, y) => tryTriple n x y

/-! ## Helper Lemma for findSome? -/

theorem findSome?_some {α β : Type*} (l : List α) (f : α → Option β) (y : β) 
    (h : l.findSome? f = some y) : ∃ x ∈ l, f x = some y := by
  induction l with
  | nil => simp at h
  | cons a as ih =>
    simp only [List.findSome?] at h
    cases hfa : f a with
    | none => simp only [hfa] at h; have ⟨x, hx, hfx⟩ := ih h; exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩
    | some b => simp only [hfa] at h; cases h; exact ⟨a, List.mem_cons_self, hfa⟩

/-- If an element in the list produces some value, findSome? returns some value -/
theorem findSome?_isSome {α β : Type*} (l : List α) (f : α → Option β) 
    (a : α) (b : β) (ha : a ∈ l) (hf : f a = some b) : (l.findSome? f).isSome := by
  induction l with
  | nil => simp at ha
  | cons hd tl ih =>
    simp only [List.findSome?]
    cases hfhd : f hd with
    | some val => simp
    | none =>
      rcases List.mem_cons.mp ha with rfl | htl
      · simp [hf] at hfhd
      · exact ih htl

/-! ## Soundness -/

theorem boundedSearch_sound (n x y z : Nat) (h : boundedSearch n = some (x, y, z)) : ES n x y z := by
  unfold boundedSearch at h
  split at h
  · simp at h
  · have ⟨⟨px, py⟩, _, htry⟩ := findSome?_some _ _ _ h
    exact isValidSolution_implies_ES n x y z (tryTriple_valid n px py x y z htry)

/-! ## Completeness Helper Lemmas -/

/-- For a valid ordered ES solution with positive z, the denominator (4x-n)y - nx is positive -/
theorem denom_pos (n x y z : Nat) (hES : ES n x y z) 
    (hn : n > 0) (hx : x > 0) (hy : y > 0) (_hz : z > 0) (ha : 4 * x > n) :
    (4 * x - n) * y > n * x := by
  unfold ES at hES
  have h_le : n ≤ 4 * x := Nat.le_of_lt ha
  have h1 : 4 * x * y * z = n * x * y + n * x * z + n * y * z := by ring_nf at hES ⊢; omega
  have hsum := ES_sum_relation n x y z hES ha
  by_contra h_neg
  push_neg at h_neg
  have h3 : (4 * x - n) * y * z ≤ n * x * z := by
    have := Nat.mul_le_mul_right z h_neg
    calc (4 * x - n) * y * z = (4 * x - n) * y * z := rfl
      _ ≤ n * x * z := by ring_nf at this ⊢; omega
  have h4 : n * x * (y + z) ≤ n * x * z := by rw [← hsum]; exact h3
  have h5 : n * x * y + n * x * z ≤ n * x * z := by
    have heq : n * x * (y + z) = n * x * y + n * x * z := by ring
    rw [heq] at h4; exact h4
  have h6 : n * x * y > 0 := Nat.mul_pos (Nat.mul_pos hn hx) hy
  omega

/-- The computed z equals the actual z from an ES solution -/
theorem computeZ_correct (n x y z : Nat) (hES : ES n x y z)
    (hn : n > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : 4 * x > n) :
    computeZ n x y = some z := by
  unfold computeZ
  simp only [ha, ↓reduceIte]
  have hdenom_pos := denom_pos n x y z hES hn hx hy hz ha
  have hdenom : (4 * x - n) * y - n * x > 0 := by omega
  simp only [hdenom, ↓reduceIte]
  have hzf := z_formula n x y z hES ha hdenom_pos
  -- z * denom = nxy, so nxy % denom = 0
  have hdiv : n * x * y % ((4 * x - n) * y - n * x) = 0 := by
    have hdvd : (4 * x - n) * y - n * x ∣ n * x * y := by
      use z
      have : ((4 * x - n) * y - n * x) * z = z * ((4 * x - n) * y - n * x) := Nat.mul_comm _ _
      rw [this, hzf]
    exact Nat.mod_eq_zero_of_dvd hdvd
  simp only [hdiv, ↓reduceIte]
  -- And nxy / denom = z
  congr 1
  have hd_pos : (4 * x - n) * y - n * x > 0 := hdenom
  -- z * denom = nxy, so nxy / denom = z
  have heq : n * x * y = ((4 * x - n) * y - n * x) * z := by
    have : ((4 * x - n) * y - n * x) * z = z * ((4 * x - n) * y - n * x) := Nat.mul_comm _ _
    rw [this, hzf]
  exact Nat.div_eq_of_eq_mul_right hd_pos heq

/-- tryTriple finds the correct solution -/
theorem tryTriple_correct (n x y z : Nat) (hES : OrderedES n x y z)
    (hn : n > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : 4 * x > n) :
    tryTriple n x y = some (x, y, z) := by
  unfold tryTriple
  have hcz := computeZ_correct n x y z hES.1 hn hx hy hz ha
  simp only [hcz]
  have hvalid := orderedES_isValid n x y z hES hx hy hz
  simp only [hvalid, ↓reduceIte]

/-! ## Helper for List.range' membership -/

theorem mem_range'_iff (s len a : Nat) : a ∈ List.range' s len ↔ s ≤ a ∧ a < s + len := by
  simp only [List.mem_range']
  constructor
  · intro ⟨k, hk, ha⟩
    constructor
    · omega
    · omega
  · intro ⟨hs, ha⟩
    refine ⟨a - s, ?_, ?_⟩
    · omega
    · omega

/-! ## Completeness -/

/-- Completeness: the search finds a solution when a positive ordered solution exists. -/
theorem boundedSearch_complete (n : Nat) (hn : n > 1) 
    (x y z : Nat) (hord : OrderedES n x y z) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
    ∃ xyz, boundedSearch n = some xyz ∧ ES n xyz.1 xyz.2.1 xyz.2.2 := by
  by_cases hsearch : (boundedSearch n).isSome
  · obtain ⟨xyz, hxyz⟩ := Option.isSome_iff_exists.mp hsearch
    exact ⟨xyz, hxyz, boundedSearch_sound n xyz.1 xyz.2.1 xyz.2.2 hxyz⟩
  · simp only [Option.isSome_iff_exists, not_exists] at hsearch
    exfalso
    have hn0 : n > 0 := Nat.lt_trans Nat.zero_lt_one hn
    -- The bounds theorems ensure the solution is in range
    have hxlb : n < 4 * x := x_lower_bound n x y z hord hx hy hz
    have hxub : 4 * x ≤ 3 * n := x_upper_bound n x y z hord hx hy hz
    have ha : 4 * x > n := hxlb
    -- tryTriple finds the solution
    have htry : tryTriple n x y = some (x, y, z) := tryTriple_correct n x y z hord hn0 hx hy hz ha
    -- The pair (x, y) is in candidatePairs
    have hylb := y_lower_bound n x y z hord ha hz
    have hyub := y_upper_bound n x y z hord ha hz
    -- boundedSearch returns some
    unfold boundedSearch at hsearch
    simp only [Nat.not_le.mpr hn, ↓reduceIte] at hsearch
    -- Show (x, y) is in candidatePairs
    have hdiff_pos : 4 * x - n > 0 := by omega
    have hy_ub : y ≤ 2 * n * x / (4 * x - n) + 1 := by
      have h1 : y * (4 * x - n) ≤ 2 * n * x := hyub
      have h2 : y ≤ 2 * n * x / (4 * x - n) := Nat.le_div_iff_mul_le hdiff_pos |>.mpr h1
      omega
    -- x bounds for range membership
    have hx_lb : (n + 3) / 4 ≤ x := by omega
    have hx_ub : x ≤ 3 * n / 4 := by omega
    have hx_in_range : x ∈ List.range' ((n + 3) / 4) (3 * n / 4 - (n + 3) / 4 + 1) := by
      rw [mem_range'_iff]
      constructor
      · exact hx_lb
      · omega
    have hy_in_range : y ∈ List.range' x (2 * n * x / (4 * x - n) + 1 - x + 1) := by
      rw [mem_range'_iff]
      constructor
      · exact hord.2.1
      · omega
    have hxy_in : (x, y) ∈ candidatePairs n := by
      unfold candidatePairs
      simp only [List.mem_flatMap]
      refine ⟨x, hx_in_range, ?_⟩
      simp only [ha, ↓reduceIte, List.mem_map]
      exact ⟨y, hy_in_range, rfl⟩
    -- findSome? should find something since (x,y) ∈ candidatePairs and tryTriple returns some
    have hfind : ((candidatePairs n).findSome? fun (x, y) => tryTriple n x y).isSome := 
      findSome?_isSome _ _ (x, y) (x, y, z) hxy_in htry
    -- But hsearch says it returns none for all - contradiction
    simp only [Option.isSome_iff_exists] at hfind
    obtain ⟨result, hresult⟩ := hfind
    exact hsearch result hresult

/-- Alternative completeness using HasPositiveOrderedES -/
theorem boundedSearch_complete' (n : Nat) (h : HasPositiveOrderedES n) (hn : n > 1) : 
    ∃ xyz, boundedSearch n = some xyz ∧ ES n xyz.1 xyz.2.1 xyz.2.2 := by
  obtain ⟨x, y, z, hes, hxy, hyz, hx, hy, hz⟩ := h
  exact boundedSearch_complete n hn x y z ⟨hes, hxy, hyz⟩ hx hy hz

/-! ## Main Completeness -/

theorem main_completeness (n : Nat) (h : HasES n) : HasOrderedES n := by
  obtain ⟨x, y, z, hes⟩ := h
  exact ES_wlog n x y z hes

theorem main_completeness_positive (n : Nat) (x y z : Nat) (h : ES n x y z) 
    (hx : x > 0) (hy : y > 0) (hz : z > 0) : HasPositiveOrderedES n := by
  exact ES_wlog_positive n x y z h hx hy hz
