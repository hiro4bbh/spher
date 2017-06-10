package spher

import "math"
import "testing"

func TestDot64(t *testing.T) {
	x1 := Vector64{1.0, 2.0, 3.0, 4.0, 5.0}
	y1 := Vector64{5.0, 4.0, 3.0, 2.0, 1.0}
	if expected, got := 35.0, Dot64(x1, y1); expected != got {
		t.Errorf("Dot64(x1, y1): expected %f, but got %f", expected, got)
	}
	x2 := Vector64{}
	y2 := Vector64{}
	if expected, got := 0.0, Dot64(x2, y2); expected != got {
		t.Errorf("Dot64(x2, y2): expected %f, but got %f", expected, got)
	}
	if got := Dot64(x1, y2); !math.IsNaN(got) {
		t.Errorf("Dot64(x1, y2): expected NaN, but got %f", got)
	}
}

func TestVector64Normalize(t *testing.T) {
	// Also tests L2Norm64 and Vector64.MulS implicitly.
	x1 := Vector64{3.0, 4.0}
	x1.Normalize()
	if expected, got := 1.0, L2Norm64(x1); expected != got {
		t.Errorf("L2Norm64(x1.Normalize()): expected %f, but got %f", expected, got)
	}
}
