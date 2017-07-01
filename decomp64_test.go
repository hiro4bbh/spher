package spher

import "fmt"
import "log"
import "math"
import "os"
import "strings"
import "testing"

func TestDiagonalizeSymmetricTridiagonal64(t *testing.T) {
	test := func(diagT, sdiagT Vector64) {
		errmsgs := []string{}
		diagTorig, sdiagTorig := diagT.Clone(), sdiagT.Clone()
		T := NewSymmetricTridiagonalMatrix64(diagT, sdiagT)
		Q := NewMatrix64I(len(diagT))
		DiagonalizeSymmetricTridiagonal64(diagT, sdiagT, Q)
		tQ := Q.T().(*Matrix64)
		tQQ, QtQ := tQ.Compose(Q), Q.Compose(tQ)
		I := NewMatrix64I(len(diagT))
		if !eqMatrix64Normal(tQQ, I) {
			errmsgs = append(errmsgs, fmt.Sprintf("t(Q)Q != I"))
		}
		if !eqMatrix64Normal(QtQ, I) {
			errmsgs = append(errmsgs, fmt.Sprintf("Qt(Q) != I"))
		}
		for i := 0; i < len(sdiagT); i++ {
			if !eq64Normal(sdiagT[i], 0.0) {
				errmsgs = append(errmsgs, fmt.Sprintf("sdiagT[%d] != 0.0", i))
			}
		}
		TQ := T.Compose(Q)
		diff := make(Vector64, len(diagT))
		for j := 0; j < len(diagT); j++ {
			x := make(Vector64, len(diagT))
			x.Madd(diagT[j], Q.SlicedColumns(j, 1).Elems())
			diff.Madd(1.0, x).Madd(-1.0, TQ.SlicedColumns(j, 1).Elems())
			relerr := L2norm64(diff) / math.Abs(diagT[j])
			if !eq64Normal(relerr, 0.0) {
				errmsgs = append(errmsgs, fmt.Sprintf("(TQ)[*,%d] != diagT[%d]Q[*,%d]: ||(TQ)[*,%d] - diagT[%d]Q[*,%d]||_2=%g*|diagT[%d]=%g|", j, j, j, j, j, j, relerr, j, diagT[j]))
			}
		}
		if len(errmsgs) > 0 {
			t.Errorf("DiagonalizeSymmetricTridiagonal64(%#v, %#v): unexpected %d error(s):\ndiagT=%#v\nsdiagT=%#v\n%s", diagTorig, sdiagTorig, len(errmsgs), diagT, sdiagT, strings.Join(errmsgs, "\n"))
		}
	}
	test(Vector64{1.0, 1.0}, Vector64{2.0})
	test(Vector64{1.0, -3.0}, Vector64{2.0})
	test(Vector64{1, 2e+05}, Vector64{3})
	test(Vector64{2e+05, 1}, Vector64{3})
	test(Vector64{1.0, 2.0, 3.0}, Vector64{4.0, 5.0})
	test(Vector64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, Vector64{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0})
	test(Vector64{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0}, Vector64{-11.0, 12.0, -13.0, 14.0, -15.0, 16.0, -17.0, 18.0, -19.0})
}

func TestTridiagonalizeSymmetricMatrix64(t *testing.T) {
	makeArrow := func(diag, z Vector64) *Matrix64 {
		A := NewMatrix64(len(diag), len(diag))
		for i := 0; i < len(diag); i++ {
			A.Elems()[i+i*A.Nrows()] = diag[i]
			if i < len(z) {
				A.Elems()[(A.Nrows()-1)+i*A.Nrows()] = z[i]
				A.Elems()[i+(A.Nrows()-1)*A.Nrows()] = z[i]
			}
		}
		return A
	}
	test := func(A *Matrix64) {
		Aclone := A.Clone()
		Q := NewMatrix64I(A.Nrows())
		if err := TridiagonalizeSymmetricMatrix64(Aclone, Q); err != nil {
			t.Fatalf("TridiagonalizeSymmetricMatrix64(*Matrix64(%d, %d, %#v), I_n): unexpected error: %s", A.Nrows(), A.Ncols(), A.Elems(), err)
		}
		errmsgs := []string{}
		I_n := NewMatrix64I(A.Nrows())
		tQQ := Q.T().(*Matrix64).Compose(Q)
		if !eqMatrix64Normal(tQQ, I_n) {
			errmsgs = append(errmsgs, fmt.Sprintf("t(Q)Q should be I_n"))
		}
		QtQ := Q.Compose(Q.T().(*Matrix64))
		if !eqMatrix64Normal(QtQ, I_n) {
			errmsgs = append(errmsgs, fmt.Sprintf("Qt(Q) should be I_n"))
		}
		for i := 0; i < Aclone.Nrows(); i++ {
			for j := 0; j < Aclone.Nrows(); j++ {
				if !((j-1 <= i) && (i <= j+1)) {
					if expected, got := 0.0, Aclone.Elems()[i+j*A.Nrows()]; !eq64Normal(expected, got) {
						errmsgs = append(errmsgs, fmt.Sprintf("A[%d,%d] should be %g, but got %g", i, j, expected, got))
					}
				}
			}
		}
		for i := 0; i < Aclone.Nrows()-1; i++ {
			Aclone.Elems()[i+(i+1)*Aclone.Nrows()] = Aclone.Elems()[(i+1)+i*Aclone.Nrows()]
		}
		AQ, QT := A.Compose(Q), Q.Compose(Aclone)
		if !eqMatrix64Normal(AQ, QT) {
			errmsgs = append(errmsgs, fmt.Sprintf("unexpected AQ != QT"))
		}
		if len(errmsgs) > 0 {
			t.Errorf("TridiagonalizeSymmetricMatrix64(*Matrix64(%d, %d, %#v), I_n): unexpected %d error(s):\n%s", A.Nrows(), A.Ncols(), A.Elems(), len(errmsgs), strings.Join(errmsgs, "\n"))
		}
	}
	test(makeArrow(Vector64{1.0, 2.0, 3.0}, Vector64{4.0, 4.0}))
	test(makeArrow(Vector64{1.0, 2.0, 3.0, 4.0}, Vector64{5.0, 5.0, 5.0}))
	test(makeArrow(Vector64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, Vector64{10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0}))
	test(makeArrow(Vector64{1.0, 2.0, -3.0, 4.0, 5.0, 6.0, -7.0, 8.0, 9.0}, Vector64{0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0, 17.0}))
	A1 := NewMatrix64(4, 4)
	A1.Elems()[0+0*A1.Nrows()] = 1.0
	A1.Elems()[1+1*A1.Nrows()] = 1.0
	A1.Elems()[2+2*A1.Nrows()] = 1.0
	A1.Elems()[3+3*A1.Nrows()] = 1.0
	A1.Elems()[2+1*A1.Nrows()] = 1e-9
	A1.Elems()[1+2*A1.Nrows()] = 1e-9
	test(A1)
	A2 := NewMatrix64(4, 4)
	A2.Elems()[0+0*A2.Nrows()] = 1.0
	A2.Elems()[1+1*A2.Nrows()] = 1.0
	A2.Elems()[2+2*A2.Nrows()] = 1.0
	A2.Elems()[3+3*A2.Nrows()] = 1.0
	A2.Elems()[2+1*A2.Nrows()] = -1e-9
	A2.Elems()[1+2*A2.Nrows()] = -1e-9
	test(A2)
	A3 := NewMatrix64(4, 4)
	A3.Elems()[0+0*A3.Nrows()] = 1.0
	A3.Elems()[1+1*A3.Nrows()] = 1.0
	A3.Elems()[2+2*A3.Nrows()] = 1.0
	A3.Elems()[3+3*A3.Nrows()] = 1.0
	A3.Elems()[2+1*A3.Nrows()] = -1e-9
	A3.Elems()[1+2*A3.Nrows()] = -1e-9
	A3.Elems()[3+1*A3.Nrows()] = 1e+2
	A3.Elems()[1+3*A3.Nrows()] = 1e+2
	test(A3)
}

func TestSVD64(t *testing.T) {
	test := func(A SparseMatrix64, k, p int) (Vector64, int) {
		var logger *log.Logger
		if testing.Verbose() {
			logger = log.New(os.Stdout, "TestSVD64: ", log.LstdFlags)
		}
		svdA, err := NewSVD64(A, k, p, logger)
		if err != nil {
			t.Fatalf("NewSVD64(%#v, %d, %d): unexpected error: %s", A, k, p, err)
		}
		errmsgs := []string{}
		x := make(Vector64, A.Nrows())
		y := make(Vector64, A.Nrows())
		for j := 0; j < k; j++ {
			copy(y, svdA.U.Elems()[j*A.Nrows():(j+1)*A.Nrows()])
			y.Mul(svdA.Sigma[j])
			x.Apply(A, svdA.V.Elems()[j*A.Ncols():(j+1)*A.Ncols()])
			x.Madd(-1, y).Round(FLOAT64_NORMAL_PRECISION)
			if L2norm64(x) != 0.0 {
				errmsgs = append(errmsgs, fmt.Sprintf("Av[*,%d] != sigma[%d]U[*,%d]: delta=%#v", j, j, j, x.Madd(-1, y).Round(FLOAT64_NORMAL_PRECISION)))
			}
		}
		Ik := NewMatrix64I(k)
		tVV := svdA.V.T().(*Matrix64).Compose(svdA.V)
		if !eqMatrix64Normal(tVV, Ik) {
			errmsgs = append(errmsgs, fmt.Sprintf("t(V)V != I_k"))
		}
		diagT_ := NewMatrix64(k, k)
		for i := 0; i < k; i++ {
			if !eq64Normal(svdA.Sigma[i], 0.0) {
				diagT_.Elems()[i+i*diagT_.Nrows()] = 1.0
			}
		}
		tUU := svdA.U.T().(*Matrix64).Compose(svdA.U)
		if !eqMatrix64Normal(tUU, diagT_) {
			diagT_diag_str := ""
			for i := 0; i < k; i++ {
				if i > 0 {
					diagT_diag_str += ","
				}
				diagT_diag_str += fmt.Sprintf("%g", diagT_.Elems()[i+i*diagT_.Nrows()])
			}
			errmsgs = append(errmsgs, fmt.Sprintf("t(U)U != diag(%s)", diagT_diag_str))
		}
		if len(errmsgs) > 0 {
			t.Errorf("NewSVD64(%#v, %d, %d): unexpected %d error(s):\nsigma=%#v\n%s", A, k, p, len(errmsgs), svdA.Sigma, strings.Join(errmsgs, "\n"))
			return nil, len(errmsgs)
		}
		return svdA.Sigma, 0
	}
	testAll := func(A SparseMatrix64) {
		kmax := A.Nrows()
		if kmax > A.Ncols() {
			kmax = A.Ncols()
		}
		sigmaKmax, nerrors := test(A, kmax, 0)
		if nerrors > 0 {
			return
		}
		for k := 1; k < kmax; k++ {
			p := 0
			for {
				sigmaK, nerrors := test(A, k, p)
				if eqVector64Normal(sigmaK, sigmaKmax[0:k]) {
					break
				}
				if nerrors == 0 {
					// Try a larger nuisance dimension p.
					if k+(p+1) <= A.Nrows() {
						p++
						continue
					}
					t.Errorf("NewSVD(%#v, %d, %d): expected sigma=%#v, but got sigma=%#v", A, k, p, sigmaKmax[0:k], sigmaK)
				}
				break
			}
		}
	}
	A1 := NewCSRMatrix64FromRowMap(10, 11, map[int]map[int]float64{
		0: map[int]float64{0: 1.0},
		1: map[int]float64{1: 2.0},
		2: map[int]float64{2: 3.0},
		3: map[int]float64{3: 4.0},
		4: map[int]float64{4: 5.0},
		5: map[int]float64{5: 6.0},
		6: map[int]float64{6: 7.0},
		7: map[int]float64{7: 8.0},
		8: map[int]float64{8: 9.0},
		9: map[int]float64{9: 8.0},
	})
	testAll(A1)
	A2 := NewCSRMatrix64FromRowMap(5, 4, map[int]map[int]float64{
		0: map[int]float64{0: 1.0},
		1: map[int]float64{0: 1.0, 1: 2.0},
		2: map[int]float64{0: 1.0, 1: 2.0, 2: 3.0},
		3: map[int]float64{0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
		4: map[int]float64{0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
	})
	testAll(A2)
	A3 := NewCSRMatrix64FromRowMap(5, 4, map[int]map[int]float64{
		0: map[int]float64{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
		1: map[int]float64{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
		2: map[int]float64{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
		3: map[int]float64{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
		4: map[int]float64{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
	})
	testAll(A3)
	A4 := NewCSRMatrix64FromRowMap(5, 4, map[int]map[int]float64{
		0: map[int]float64{1: 0.0, 2: 0.0, 3: 0.0},
		1: map[int]float64{1: 2.0, 2: 0.0, 3: 0.0},
		2: map[int]float64{1: 2.0, 2: 3.0, 3: 0.0},
		3: map[int]float64{1: 2.0, 2: 3.0, 3: 0.0},
		4: map[int]float64{1: 2.0, 2: 3.0, 3: 4.0},
	})
	testAll(A4)
	if t.Failed() {
		t.Fatalf("Already failed in some smaller test-cases.")
	}
	// Tridiagonal Toeplitz: diag(a1_n) + sdiag(b1_n) + t(sdiag(b1_n))
	//   eigenvalues: a + 2b*cos(k\pi/(n+1)), k = 1, ...., n
	// Here, a = 1.0, b = 2.0, then eigenvalues are in 1 + 2*2(-1, 1) = (-3, 5).
	A5map := map[int]map[int]float64{}
	for i := 0; i < 1000; i++ {
		A5map[i] = map[int]float64{i: 1.0}
		if i+1 < 1000 {
			A5map[i][i+1] = 2.0
		}
		if i-1 >= 0 {
			A5map[i][i-1] = 2.0
		}
	}
	A5 := NewCSRMatrix64FromRowMap(1000, 1000, A5map)
	test(A5, 100, 0)
	A6map := map[int]map[int]float64{}
	for i := 0; i < 10000; i++ {
		A6map[i] = map[int]float64{i: 1.0}
		if i+1 < 10000 {
			A6map[i][i+1] = 2.0
		}
		if i-1 >= 0 {
			A6map[i][i-1] = 6.0
		}
	}
	A6 := NewCSRMatrix64FromRowMap(10000, 10000, A6map)
	test(A6, 100, 0)
}
