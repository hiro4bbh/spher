package spher

import "testing"

func TestDot64(t *testing.T) {
	x1 := Vector64{1.0, 2.0, 3.0, 4.0, 5.0}
	y1 := Vector64{5.0, 4.0, 3.0, 2.0, 1.0}
	if expected, got := 35.0, Dot64(x1, y1); expected != got {
		t.Errorf("Dot(x1, y1): expected %f, but got %f", expected, got)
	}
	x2 := Vector64{}
	y2 := Vector64{}
	if expected, got := 0.0, Dot64(x2, y2); expected != got {
		t.Errorf("Dot(x2, y2): expected %f, but got %f", expected, got)
	}
	if got := Dot64(x1, y2); !IsNaN64(got) {
		t.Errorf("Dot(x1, y2): expected NaN64, but got %f", got)
	}
}
