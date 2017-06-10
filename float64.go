package spher

import "math"

// Returns true if x is NaN64, otherwise returns false.
func IsNaN64(x float64) bool {
	return math.IsNaN(x)
}

// Returns float64 NaN.
func NaN64() float64 {
	return math.NaN()
}

// Vector64 is a dense float64 vector.
type Vector64 []float64

// Returns the dot product of x and y.
func Dot64(x, y Vector64) float64 {
	if len(x) != len(y) {
		return NaN64()
	}
	s := float64(0)
	for i := 0; i < len(x); i++ {
		s += x[i]*y[i]
	}
	return s
}
