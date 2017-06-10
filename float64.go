package spher

import "math"

// Vector64 is a dense float64 vector.
type Vector64 []float64

// Returns the dot product of Vector64 x and y.
func Dot64(x, y Vector64) float64 {
	if len(x) != len(y) {
		return math.NaN()
	}
	s := float64(0)
	for i := 0; i < len(x); i++ {
		s += x[i]*y[i]
	}
	return s
}

// Returns L2-norm of Vector64 x.
func L2Norm64(x Vector64) float64 {
	return math.Pow(Dot64(x, x), 0.5)
}

// Multiply Vector64 x by scalar y.
func (x Vector64) MulS(y float64) {
	for i := 0; i < len(x); i++ {
		x[i] *= y
	}
}

// Normalize Vector64 x.
func (x Vector64) Normalize() {
	x.MulS(1.0/L2Norm64(x))
}
