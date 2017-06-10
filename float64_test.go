package spher

import "math"
import "testing"

const FLOAT64_NORMAL_PRECISION = 12

func eq64Normal(x, y float64) bool {
	return Round64(x, FLOAT64_NORMAL_PRECISION) == Round64(y, FLOAT64_NORMAL_PRECISION)
}

func eqVector64Normal(x, y Vector64) bool {
	return Cmp64(x.Clone().Round(FLOAT64_NORMAL_PRECISION), x.Clone().Round(FLOAT64_NORMAL_PRECISION)) == 0
}

func TestRound64(t *testing.T) {
	test := func(expected float64, x float64, n int) {
		if got := Round64(x, n); expected != got {
			t.Errorf("Round64(%f, %d): expected %f, but got %f", x, n, expected, got)
		}
	}
	test(1.0, 1.105, 0)
	test(2.0, 1.605, 0)
	test(1.1, 1.1105, 1)
	test(1.2, 1.1605, 1)
	test(0.0, 1.05, -1)
	test(10.0, 6.05, -1)
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
	test(1, Vector64{1.0, 2.0, 3.0}, Vector64{1.0, 2.0})
	test(-1, Vector64{1.0, 2.0}, Vector64{1.0, 2.0, 3.0})
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

func TestL2Norm64(t *testing.T) {
	test := func(expected float64, x Vector64) {
		if got := L2Norm64(x); (math.IsNaN(expected) && !math.IsNaN(got)) || (!math.IsNaN(expected) && !eq64Normal(expected, got)) {
			t.Errorf("L2Norm64(%#v): expected %f, but got %f", x, expected, got)
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

func TestVector64MulS(t *testing.T) {
	test := func(expected Vector64, x Vector64, y float64) {
		if got := x.Clone().MulS(y); !eqVector64Normal(expected, got) {
			t.Errorf("%#v.MulS(%f): expected %#v, but got %#v", x, y, expected, got)
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
