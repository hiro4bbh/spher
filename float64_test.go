package spher

import "fmt"
import "math"
import "testing"

func eq64Normal(x, y float64) bool {
	xIsNaN, yIsNaN := math.IsNaN(x), math.IsNaN(y)
	if xIsNaN || yIsNaN {
		if xIsNaN && yIsNaN {
			return true
		}
		return false
	}
	return Round64(x, FLOAT64_NORMAL_PRECISION) == Round64(y, FLOAT64_NORMAL_PRECISION)
}

func eqVector64Normal(x, y Vector64) bool {
	return Cmp64(x.Clone().Round(FLOAT64_NORMAL_PRECISION), y.Clone().Round(FLOAT64_NORMAL_PRECISION)) == 0
}

func eqMatrix64Normal(A, B *Matrix64) bool {
	if !((A.Nrows() == B.Nrows()) && (A.Ncols() == B.Ncols())) {
		return false
	}
	return eqVector64Normal(A.Elems(), B.Elems())
}

func TestRound64(t *testing.T) {
	test := func(expected float64, x float64, n int) {
		if got := Round64(x, n); !eq64Normal(expected, got) {
			t.Errorf("Round64(%f, %d): expected %f, but got %f", x, n, expected, got)
		}
	}
	test(1.0, 1.105, 0)
	test(2.0, 1.605, 0)
	test(math.NaN(), math.NaN(), 0)
	test(1.1, 1.1105, 1)
	test(1.2, 1.1605, 1)
	test(math.NaN(), math.NaN(), 1)
	test(0.0, 1.05, -1)
	test(10.0, 6.05, -1)
	test(math.NaN(), math.NaN(), -1)
}

func TestCmp64(t *testing.T) {
	test := func(expected int, x, y Vector64) {
		if got := Cmp64(x, y); expected != got {
			t.Errorf("Cmp64(%#v, %#v): expected %d, but got %d", x, y, expected, got)
		}
	}
	test(0, Vector64{1.0, 2.0, 3.0}, Vector64{1.0, 2.0, 3.0})
	test(1, Vector64{1.0, 3.0, 3.0}, Vector64{1.0, 2.0, 3.0})
	test(-1, Vector64{1.0, 1.0, 3.0}, Vector64{1.0, 2.0, 3.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, math.NaN(), 3.0}, Vector64{1.0, 2.0, 3.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, 2.0, 3.0}, Vector64{1.0, math.NaN(), 3.0})
	test(1, Vector64{1.0, 2.0, 3.0}, Vector64{1.0, 2.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, math.NaN(), 3.0}, Vector64{1.0, 2.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, 2.0, 3.0}, Vector64{1.0, math.NaN()})
	test(-1, Vector64{1.0, 2.0}, Vector64{1.0, 2.0, 3.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, math.NaN()}, Vector64{1.0, 2.0, 3.0})
	test(FLOAT64_NAN_BITS, Vector64{1.0, 2.0}, Vector64{1.0, math.NaN(), 3.0})
}

func TestDot64(t *testing.T) {
	test := func(expected float64, x, y Vector64) {
		if got := Dot64(x, y); (math.IsNaN(expected) && !math.IsNaN(got)) || (!math.IsNaN(expected) && !eq64Normal(expected, got)) {
			t.Errorf("Dot64(%#v, %#v): expected %f, but got %f", x, y, expected, got)
		}
	}
	test(10.0, Vector64{1.0, 2.0, 3.0}, Vector64{3.0, 2.0, 1.0})
	test(0.0, Vector64{}, Vector64{})
	test(math.NaN(), Vector64{1.0, 2.0, 3.0}, Vector64{})
}

func TestL2norm64(t *testing.T) {
	test := func(expected float64, x Vector64) {
		if got := L2norm64(x); (math.IsNaN(expected) && !math.IsNaN(got)) || (!math.IsNaN(expected) && !eq64Normal(expected, got)) {
			t.Errorf("L2norm64(%#v): expected %f, but got %f", x, expected, got)
		}
	}
	test(5.0, Vector64{3.0, 4.0})
	test(math.NaN(), Vector64{})
}

func TestVector64Clone(t *testing.T) {
	x := Vector64{1.0, 2.0, 3.0}
	cloneX := x.Clone()
	x[0] = 4.0
	if cloneX[0] != 1.0 {
		t.Errorf("%#v.Clone(): A clone of Vector64 must not be affected by changes of the original", x)
	}
	cloneX[1] = 5.0
	if x[1] != 2.0 {
		t.Errorf("%#v.Clone(): The original Vector64 must not be affected by its clones", x)
	}
}

func TestVector64Fill(t *testing.T) {
	test := func(expected Vector64, x Vector64, y float64) {
		if got := x.Fill(y); !eqVector64Normal(expected, got) {
			t.Errorf("%#v.Fill(%f): expected %f, but got %f", x, y, expected, got)
		}
	}
	test(Vector64{1.0, 1.0, 1.0}, Vector64{1.0, 2.0, 3.0}, 1.0)
	test(Vector64{}, Vector64{}, 1.0)
}

func TestVector64Madd(t *testing.T) {
	test := func(expected Vector64, x Vector64, alpha float64, y Vector64) {
		if got := x.Clone().Madd(alpha, y); !eqVector64Normal(expected, got) {
			t.Errorf("%#v.Madd(%f, %#v): expected %#v, but got %#v", x, alpha, y, expected, got)
		}
	}
	test(Vector64{7.0, 6.0, 5.0}, Vector64{1.0, 2.0, 3.0}, 2.0, Vector64{3.0, 2.0, 1.0})
	test(Vector64{1.0, 2.0, 3.0}, Vector64{1.0, 2.0, 3.0}, 0.0, Vector64{3.0, 2.0, 1.0})
	test(Vector64{-5.0, -2.0, 1.0}, Vector64{1.0, 2.0, 3.0}, -2.0, Vector64{3.0, 2.0, 1.0})
	test(Vector64{}, Vector64{}, 2.0, Vector64{})
	test(Vector64{}, Vector64{}, 0.0, Vector64{})
	test(Vector64{}, Vector64{}, -2.0, Vector64{})
}

func TestVector64Mul(t *testing.T) {
	test := func(expected Vector64, x Vector64, y float64) {
		if got := x.Clone().Mul(y); !eqVector64Normal(expected, got) {
			t.Errorf("%#v.Mul(%f): expected %#v, but got %#v", x, y, expected, got)
		}
	}
	test(Vector64{2.0, 4.0, 6.0}, Vector64{1.0, 2.0, 3.0}, 2.0)
	test(Vector64{}, Vector64{}, 2.0)
}

func TestVector64Normalize(t *testing.T) {
	test := func(expected Vector64, x Vector64) {
		if got := x.Clone().Normalize(); !eqVector64Normal(expected, got) {
			t.Errorf("%#v.Normalize(): expected %#v, but got %#v", x, expected, got)
		}
	}
	test(Vector64{3.0 / 5.0, 4.0 / 5.0}, Vector64{3.0, 4.0})
	test(Vector64{}, Vector64{})
}

func TestAugmentedSparseMatrix64(t *testing.T) {
	A := NewCSRMatrix64FromRowMap(4, 5, map[int]map[int]float64{
		0: map[int]float64{2: 1.0},
		2: map[int]float64{1: 2.0, 3: 3.0},
	})
	x := make(Vector64, A.Nrows()+A.Ncols())
	x[0] = 2.0
	x[A.Nrows()+2] = 3.0
	y := make(Vector64, A.Nrows()+A.Ncols())
	y.Apply(AugmentSparseMatrix64(A), x)
	expected, got := Vector64{3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0}, y
	if !eqVector64Normal(expected, got) {
		t.Errorf("Vector64.Apply(AugmentSparseMatrix64(%#v), %#v): expected %#v, but got %#v", A, x, expected, got)
	}
	x.Fill(0.0)
	x[A.Ncols()] = 2.0
	x[3] = 3.0
	y.Apply(AugmentSparseMatrix64(A.T()), x)
	expected, got = Vector64{0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0}, y
	if !eqVector64Normal(expected, got) {
		t.Errorf("Vector64.Apply(AugmentSparseMatrix64(%#v.T()), %#v): expected %#v, but got %#v", A, x, expected, got)
	}
}

func TestSymmetrizedMatrix64(t *testing.T) {
	A := NewCSRMatrix64FromRowMap(4, 5, map[int]map[int]float64{
		0: map[int]float64{2: 1.0},
		2: map[int]float64{1: 2.0, 3: 3.0},
	})
	denseA := NewMatrix64FromSparseMatrix64(A)
	tdenseAdenseA := denseA.T().(*Matrix64).Compose(denseA)
	tAA := &SymmetrizedMatrix64{A}
	dense_tAA := NewMatrix64FromSparseMatrix64(tAA)
	if !eqMatrix64Normal(tdenseAdenseA, dense_tAA) {
		t.Fatalf("unexpected t(dense(A))dense(A) != dense(t(A)A): t(dense(A))dense(A)=%#v,\ndense(t(A)*A)=%#v", tdenseAdenseA, dense_tAA)
	}
	z1, z2 := make(Vector64, tAA.Nrows()), make(Vector64, tAA.Nrows())
	for i := 0; i < A.Nrows(); i++ {
		for j := 0; j < A.Ncols(); j++ {
			y, x := make(Vector64, tAA.Nrows()), make(Vector64, tAA.Ncols())
			y[i], x[j] = 1.0, 1.0
			if expected, got := Dot64(y, z1.Apply(tdenseAdenseA, x)), Dot64(y, z2.Apply(dense_tAA, x)); expected != got {
				t.Errorf("(t(A)A)[%d,%d]: expected %#v, but got %#v", i, j, expected, got)
			}
		}
	}
}

func TestCSRMatrix64(t *testing.T) {
	A := NewCSRMatrix64FromRowMap(4, 5, map[int]map[int]float64{
		0: map[int]float64{2: 1.0},
		2: map[int]float64{1: 2.0, 3: 3.0},
	})
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%s", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%s\", *CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%v", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%v\", *CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%#v", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%#v\", *CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%s", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%s\", CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%v", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%v\", CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.CSRMatrix64(nrows:4, ncols:5)", fmt.Sprintf("%#v", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%#v\", CSRMatrix64): expected %#v, but got %#v", expected, got)
	}
	testApply := func(expected Vector64, A SparseMatrix64) {
		got := make(Vector64, A.Nrows()*A.Ncols())
		for j := 0; j < A.Ncols(); j++ {
			v := make(Vector64, A.Ncols())
			v[j] = 1.0
			got[j*A.Nrows():(j+1)*A.Nrows()].Apply(A, v)
		}
		if !eqVector64Normal(expected, got) {
			t.Errorf("NewCSRMatrix64FromRowMap(%#v).Apply64(_, _): expected %#v, but got %#v", A, expected, got)
		}
	}
	testApply(Vector64{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 2.0, 0.0,
		1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 3.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
	}, A)
	testApply(Vector64{
		0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 2.0, 0.0, 3.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
	}, A.T())
	testApply(Vector64{}, NewCSRMatrix64FromRowMap(0, 0, map[int]map[int]float64{}))
	testApply(Vector64{}, NewCSRMatrix64FromRowMap(0, 0, map[int]map[int]float64{}).T())
}

func TestMatrix64(t *testing.T) {
	A := NewMatrix64(4, 5)
	for i := 0; i < A.Nrows(); i++ {
		for j := 0; j < A.Ncols(); j++ {
			A.Elems()[i+j*A.Nrows()] = float64(i + j*A.Nrows())
		}
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%s", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%s\", *Matrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%v", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%v\", *Matrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%#v", A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%#v\", *Matrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%s", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%s\", Matrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%v", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%v\", Matrix64): expected %#v, but got %#v", expected, got)
	}
	if expected, got := "spher.Matrix64(nrows:4, ncols:5)", fmt.Sprintf("%#v", *A); expected != got {
		t.Errorf("fmt.Sprintf(\"%%#v\", Matrix64): expected %#v, but got %#v", expected, got)
	}
	ty, tx := make(Vector64, A.Nrows()), make(Vector64, A.Ncols())
	for i := 0; i < A.Nrows(); i++ {
		for j := 0; j < A.Ncols(); j++ {
			y, x := make(Vector64, A.Nrows()), make(Vector64, A.Ncols())
			y[i], x[j] = 1.0, 1.0
			ty.Apply(A, x)
			if expected, got := float64(i + j*A.Nrows()), Dot64(y, ty); !eq64Normal(expected, got) {
				t.Errorf("%#v.Dot(z.Apply(%#v, %#v)): expected %#v, but got %#v", y, A, x, expected, got)
			}
		}
	}
	tA := A.T().(*Matrix64)
	if tA.Nrows() != A.Ncols() {
		t.Errorf("unexpected tA.Nrows() != A.Ncols()")
	}
	if tA.Ncols() != A.Nrows() {
		t.Errorf("unexpected tA.Ncols() != A.Nrows()")
	}
	for i := 0; i < tA.Nrows(); i++ {
		for j := 0; j < tA.Ncols(); j++ {
			y, x := make(Vector64, tA.Nrows()), make(Vector64, tA.Ncols())
			y[i], x[j] = 1.0, 1.0
			tx.Apply(tA, x)
			if expected, got := float64(j + i*A.Nrows()), Dot64(y, tx); !eq64Normal(expected, got) {
				t.Errorf("%#v.Dot(z.Apply(%#v, %#v)): expected %#v, but got %#v", y, tA, x, expected, got)
			}
		}
	}
	C := tA.Compose(A)
	if C.Nrows() != tA.Nrows() {
		t.Errorf("unexpected C.Nrows() != tA.Nrows()")
	}
	if C.Ncols() != A.Ncols() {
		t.Errorf("unexpected C.Ncols() != A.Ncols()")
	}
	for i := 0; i < C.Nrows(); i++ {
		for j := 0; j < C.Ncols(); j++ {
			y, x := make(Vector64, C.Nrows()), make(Vector64, C.Ncols())
			y[i], x[j] = 1.0, 1.0
			tx.Apply(tA, x)
			if expected, got := Dot64(A.Elems()[i*A.Nrows():(i+1)*A.Nrows()], A.Elems()[j*A.Nrows():(j+1)*A.Nrows()]), C.Elems()[i+j*C.Ncols()]; !eq64Normal(expected, got) {
				t.Errorf("%#v.Dot(z.Apply(%#v, %#v)): expected %#v, but got %#v", y, C, x, expected, got)
			}
		}
	}
}

func TestNewMatrixI64(t *testing.T) {
	I5 := NewMatrix64I(5)
	if expected, got := I5.Nrows(), 5; expected != got {
		t.Errorf("I5.Nrows(): expected #v, but got %#v", expected, got)
	}
	if expected, got := I5.Ncols(), 5; expected != got {
		t.Errorf("I5.Ncols(): expected #v, but got %#v", expected, got)
	}
	for i := 0; i < I5.Nrows(); i++ {
		for j := 0; j < I5.Ncols(); j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if got := I5.Elems()[i+j*I5.Ncols()]; !eq64Normal(expected, got) {
				t.Errorf("I5[%d,%d]: expected %#v, but got %#v", i, j, expected, got)
			}
		}
	}
	I0 := NewMatrix64I(0)
	if expected, got := I0.Nrows(), 0; expected != got {
		t.Errorf("I0.Nrows(): expected #v, but got %#v", expected, got)
	}
	if expected, got := I0.Ncols(), 0; expected != got {
		t.Errorf("I0.Ncols(): expected #v, but got %#v", expected, got)
	}
}
