package spher

import "fmt"
import "math"
import "sort"

// This can be get as follows: math.Float64Bits(math.NaN()).
const FLOAT64_NAN_BITS = 0x7ff8000000000001

// Round float64 x to the specified number of decimal places.
func Round64(x float64, n int) float64 {
	if math.IsNaN(x) {
		return x
	}
	shift := math.Pow(10, float64(n))
	return math.Floor(x*shift+0.5) / shift
}

// Vector64 is a dense float64 vector.
// Unless specified, methods return self.
type Vector64 []float64

// Compares Vector64 x and y.
// If either of x or y contains float64 NaN, returns FLOAT64_NAN_BITS.
// Returns +1 if x is greater than y, -1 if x is smaller than y, otherwise 0.
func Cmp64(x, y Vector64) int {
	lmin := len(x)
	if lmin > len(y) {
		lmin = len(y)
	}
	i := 0
	s := 0
	for ; i < lmin; i++ {
		if math.IsNaN(x[i]) || math.IsNaN(y[i]) {
			return FLOAT64_NAN_BITS
		}
		if s == 0 {
			if x[i] < y[i] {
				s = -1
			} else if x[i] > y[i] {
				s = +1
			}
		}
	}
	for ix := i; ix < len(x); ix++ {
		if math.IsNaN(x[i]) {
			return FLOAT64_NAN_BITS
		}
	}
	for iy := i; iy < len(y); iy++ {
		if math.IsNaN(y[i]) {
			return FLOAT64_NAN_BITS
		}
	}
	if s != 0 {
		return s
	}
	if len(x) < len(y) {
		return -1
	} else if len(x) > len(y) {
		return +1
	}
	return 0
}

// Returns the dot product of Vector64 x and y.
func Dot64(x, y Vector64) float64 {
	if len(x) != len(y) {
		return math.NaN()
	}
	s := float64(0)
	for i := 0; i < len(x); i++ {
		s += x[i] * y[i]
	}
	return s
}

// Returns L2-norm of Vector64 x.
func L2Norm64(x Vector64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	return math.Pow(Dot64(x, x), 0.5)
}

// Apply SparseMatrix64 A to Vector64 x.
func (y Vector64) Apply(A SparseMatrix64, x Vector64) Vector64 {
	A.Apply64(y, x)
	return y
}

// Returns a clone of Vector64 x.
// A clone is not affected by changes on x, and x is not affected by its clones.
func (x Vector64) Clone() Vector64 {
	cloneX := make(Vector64, len(x))
	copy(cloneX, x)
	return cloneX
}

// Fills self with float64 y.
func (x Vector64) Fill(y float64) Vector64 {
	for i := 0; i < len(x); i++ {
		x[i] = y
	}
	return x
}

// Multiply Vector64 x by scalar y.
func (x Vector64) MulS(y float64) Vector64 {
	for i := 0; i < len(x); i++ {
		x[i] *= y
	}
	return x
}

// Normalize Vector64 x.
func (x Vector64) Normalize() Vector64 {
	x.MulS(1.0 / L2Norm64(x))
	return x
}

// Round the elements of Vector64 x to the specified number of decimal places.
func (x Vector64) Round(n int) Vector64 {
	for i := 0; i < len(x); i++ {
		x[i] = Round64(x[i], n)
	}
	return x
}

// SparseMatrix64 is an interface for a float64 sparse matrix.
type SparseMatrix64 interface {
	// Applies self to Vector64 x, and stores the result to Vector64 y.
	// If any error happens, fill Vector64 y with float64 NaN.
	Apply64(y, x Vector64)
	// Returns the number of columns.
	Ncols() int
	// Returns the number of rows.
	Nrows() int
	// Returns a transposed self.
	// The transpose states must not be affected by each other.
	T() SparseMatrix64
}

// AugmentedSparseMatrix64 is a type for augmented SparseMatrix64.
type AugmentedSparseMatrix64 struct {
	a SparseMatrix64
}

// Returns augmented matrix of SparseMatrix64 A.
// Augumented matrix of A is defined as (t(O, A), t(t(A), O)).
func AugmentSparseMatrix64(A SparseMatrix64) *AugmentedSparseMatrix64 {
	return &AugmentedSparseMatrix64{a: A}
}

// Applies self to Vector64 x, and stores the result to Vector64 y.
// If any error happens, fill Vector64 y with float64 NaN.
func (A *AugmentedSparseMatrix64) Apply64(y, x Vector64) {
	if !((len(y) == A.Nrows()) && (len(x) == A.Ncols())) {
		y.Fill(math.NaN())
		return
	}
	y.Fill(0.0)
	// (t(O, A), t(t(A), O))t(t(x1), t(x2)) = t(t(Ax2), t(t(A)x1)).
	y[0:A.a.Nrows()].Apply(A.a, x[A.a.Nrows():(A.a.Nrows()+A.a.Ncols())])
	y[A.a.Nrows():(A.a.Nrows()+A.a.Ncols())].Apply(A.a.T(), x[0:A.a.Nrows()])
}

// Returns the number of columns.
func (A *AugmentedSparseMatrix64) Ncols() int {
	return A.a.Nrows() + A.a.Ncols()
}

// Returns the number of rows.
func (A *AugmentedSparseMatrix64) Nrows() int {
	return A.a.Nrows() + A.a.Ncols()
}

// Returns a transposed self.
// The transpose states are not affected by each other.
func (A *AugmentedSparseMatrix64) T() SparseMatrix64 {
	return &AugmentedSparseMatrix64{a: A.a.T()}
}

// CSRMatrix64 is a float64 Compressed Sparse Row (CSR) Matrix.
type CSRMatrix64 struct {
	nrows, ncols int
	a            []float64
	ia, ja       []int
	transposed   bool
}

// Returns a new *CSRMatrix64 created from row map m.
func NewCSRMatrix64FromRowMap(nrows, ncols int, m map[int]map[int]float64) *CSRMatrix64 {
	if !((nrows >= 0) && (ncols >= 0)) {
		return nil
	}
	n := 0
	for i, mrow := range m {
		if !((0 <= i) && (i < nrows)) {
			return nil
		}
		n += len(mrow)
		for j, _ := range mrow {
			if !((0 <= j) && (j < ncols)) {
				return nil
			}
		}
	}
	a := make([]float64, n)
	ia := make([]int, nrows+1)
	ia[len(ia)-1] = n
	ja := make([]int, n)
	nuseds := 0
	for i := 0; i < nrows; i++ {
		mrow := m[i]
		aStart, aEnd := nuseds, nuseds+len(mrow)
		ia[i] = aStart
		for j, _ := range mrow {
			ja[nuseds] = j
			nuseds++
		}
		sort.Ints(ja[aStart:aEnd])
		for ai := aStart; ai < aEnd; ai++ {
			a[ai] = mrow[ja[ai]]
		}
	}
	return &CSRMatrix64{
		nrows:      nrows,
		ncols:      ncols,
		a:          a,
		ia:         ia,
		ja:         ja,
		transposed: false,
	}
}

// For interface GoStringer in package fmt.
// This accepts struct itself and struct pointer, and returns the same string representation.
// Hence, the string representation is slightly different from one used in package fmt.
func (A CSRMatrix64) GoString() string {
	return fmt.Sprintf("%T(nrows:%d, ncols:%d)", A, A.Nrows(), A.Ncols())
}

// For interface Stringer in package fmt.
// This is equivalent to method GoString.
func (A CSRMatrix64) String() string {
	return A.GoString()
}

// Applies self to Vector64 x, and stores the result to Vector64 y.
// If any error happens, fill Vector64 y with float64 NaN.
func (A *CSRMatrix64) Apply64(y, x Vector64) {
	if !((len(y) == A.Nrows()) && (len(x) == A.Ncols())) {
		y.Fill(math.NaN())
		return
	}
	y.Fill(0.0)
	if A.transposed {
		for i := 0; i < A.nrows; i++ {
			for ai := A.ia[i]; ai < A.ia[i+1]; ai++ {
				y[A.ja[ai]] += A.a[ai] * x[i]
			}
		}
	} else {
		for i := 0; i < A.nrows; i++ {
			for ai := A.ia[i]; ai < A.ia[i+1]; ai++ {
				y[i] += A.a[ai] * x[A.ja[ai]]
			}
		}
	}
}

// Returns the number of columns.
func (A *CSRMatrix64) Ncols() int {
	if A.transposed {
		return A.nrows
	} else {
		return A.ncols
	}
}

// Returns the number of rows.
func (A *CSRMatrix64) Nrows() int {
	if A.transposed {
		return A.ncols
	} else {
		return A.nrows
	}
}

// Returns a transposed self.
// The transpose states are not affected by each other.
func (A *CSRMatrix64) T() SparseMatrix64 {
	return &CSRMatrix64{
		nrows:      A.nrows,
		ncols:      A.ncols,
		a:          A.a,
		ia:         A.ia,
		ja:         A.ja,
		transposed: !A.transposed,
	}
}
