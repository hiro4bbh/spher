package spher

import "math"

// Round float64 x to the specified number of decimal places.
func Round64(x float64, n int) float64 {
	shift := math.Pow(10, float64(n))
	return math.Floor(x*shift+0.5) / shift
}

// Vector64 is a dense float64 vector.
// Unless specified, methods return self.
type Vector64 []float64

// Compares Vector64 x and y.
// Returns +1 if x is greater than y, -1 if x is smaller than y, otherwise 0.
func Cmp64(x, y Vector64) int {
	if len(x) < len(y) {
		return -1
	} else if len(x) > len(y) {
		return +1
	}
	for i := 0; i < len(x); i++ {
		if x[i] < y[i] {
			return -1
		} else if x[i] > y[i] {
			return +1
		}
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
