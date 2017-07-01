package spher

import "bufio"
import "encoding/binary"
import "fmt"
import "io"
import "io/ioutil"
import "log"
import "math"
import "os"
import "sort"

// Apply Lanczos steps on A from the start-th column to the end-th column.
// This function assumes that A is symmetric, and that this function is used for Thick-Restart Lanczos algorithm.
// If start is 0, then the application starts with the fixed initial vector e_1, otherwise use w as the initial vector.
func ApplyLanczosSteps(A SparseMatrix64, V *Matrix64, diagT, sdiagT Vector64, w Vector64, start, end int) error {
	if !((V.Nrows() == A.Nrows()) && (V.Ncols() >= end)) {
		return fmt.Errorf("V.Nrows() must be A.Nrows(), and V.Ncols() must be greater than end")
	}
	if !(len(diagT) >= end) {
		return fmt.Errorf("diagT must have at least end elements")
	}
	if !(len(sdiagT) >= end-1) {
		return fmt.Errorf("sdiagT must have at least end - 1 elements")
	}
	if !(len(w) == A.Nrows()) {
		return fmt.Errorf("w must have A.Nrows() elements")
	}
	if !((0 <= start) && (start < end)) {
		return fmt.Errorf("start must be greater than or equal to 0, and less than end")
	}
	if start == 0 {
		// V[:,0] = e_0
		v0 := V.SlicedColumns(0, 1).Elems()
		v0.Fill(0.0)
		v0[0] = 1.0
		// w = AV[:,0]
		w.Apply(A, v0)
		// T[0,0] = t(V[:,0])w
		diagT[0] = Dot64(v0, w)
		// w -= diagT[0]V[:,0]
		w.Madd(-diagT[0], v0)
	} else {
		// For thick restart:
		//   V[:,start] = w/||w||_2
		vstart_1, vstart := V.SlicedColumns(start-1, 1).Elems(), V.SlicedColumns(start, 1).Elems()
		vstart.Mcopy(1.0/L2norm64(w), w)
		// w = AV[:,start]
		w.Apply(A, vstart)
		// T[start,start] = t(V[:,start])w
		diagT[start] = Dot64(vstart, w)
		// w -= \sum_{l=1}^{start} T[l,start]V[:,l]
		for l := 0; l < start; l++ {
			w.Madd(-sdiagT[l], V.SlicedColumns(l, 1).Elems())
		}
		// w -= T[start,start]v[:,start]
		w.Madd(-diagT[start], vstart)
		// Conservative full orthogonalization due to numerical errors.
		for l := 0; l < start-1; l++ {
			vl := V.SlicedColumns(l, 1).Elems()
			w.Madd(-Dot64(vl, w), vl)
		}
		s_start_1, s_start := Dot64(vstart_1, w), Dot64(vstart, w)
		w.Madd(-s_start_1, vstart_1).Madd(-s_start, vstart)
		diagT[start] += s_start
		sdiagT[start-1] += s_start_1
	}
	for j := start + 1; j < end; j++ {
		// T[j,j-1] = ||w||_2
		vj_1, vj := V.SlicedColumns(j-1, 1).Elems(), V.SlicedColumns(j, 1).Elems()
		sdiagT[j-1] = L2norm64(w)
		if Round64(sdiagT[j-1], FLOAT64_NORMAL_PRECISION) == 0.0 {
			// w was already in a invariant space (numerically).
			// Select a unit vector orthogonal to V[:,0], ..., V[:,j-1].
			// e_1, ..., e_n spans the domain of A, so search from these.
			for k := 0; k < A.Ncols(); k++ {
				vj.Fill(0.0)
				vj[k] = 1.0
				for l := 0; l < j; l++ {
					vl := V.SlicedColumns(l, 1).Elems()
					vj.Madd(-Dot64(vl, vj), vl)
				}
				if vjnorm := L2norm64(vj); Round64(vjnorm, FLOAT64_NORMAL_PRECISION) > 0.0 {
					vj.Mul(1.0 / vjnorm)
					break
				}
			}
			sdiagT[j-1] = 0.0
		} else {
			// V[:,j] = w/T[j,j-1]
			vj.Mcopy(1.0/sdiagT[j-1], w)
		}
		// w = AV[:,j]
		w.Apply(A, vj)
		// Classical Gram-Schimidt orthogonalization.
		//   T[j,j] = t(V[:,j])w
		//   w = w - T[j-1,j]V[:,j-1] - T[j,j]V[:,j]
		diagT[j] = Dot64(vj, w)
		w.Madd(-sdiagT[j-1], vj_1).Madd(-diagT[j], vj)
		// Conservative full orthogonalization due to numerical errors.
		for l := 0; l < j-1; l++ {
			vl := V.SlicedColumns(l, 1).Elems()
			w.Madd(-Dot64(vl, w), vl)
		}
		sj_1, sj := Dot64(vj_1, w), Dot64(vj, w)
		w.Madd(-sj_1, vj_1).Madd(-sj, vj)
		diagT[j] += sj
		sdiagT[j-1] += sj_1
	}
	return nil
}

// Calculate the Givens rotation such that (x, y) |-> (sqrt(x**2 + y**2), 0).
// The used Givens rotation matrix is (t(c, -s), t(s, c)).
func CalculateGivensRotation(x, y float64) (c, s float64) {
	// Try carefully:
	//   r := math.Hypot(x, y)
	//   c, s := x/r, -y/r
	if math.Abs(x) < math.Abs(y) {
		ratio := -x / y
		s = 1.0 / math.Sqrt(1.0+ratio*ratio)
		c = ratio * s
	} else {
		ratio := -y / x
		c = 1.0 / math.Sqrt(1.0+ratio*ratio)
		s = ratio * c
	}
	return
}

// Diagonalize symmetric tridiagonal matrix T constructed from diagT and sdiagT with bulge-chasing implicitly-shifted QR.
// diagT is the diagonal elements of T, and sdiagT is the super-/sub-diagonal elements of T.
func DiagonalizeSymmetricTridiagonal64(diagT, sdiagT Vector64, Q *Matrix64) error {
	if len(diagT) < 2 {
		return fmt.Errorf("len(diagT) must be >= 2, otherwise there is no need to apply QR steps")
	}
	if len(diagT) != len(sdiagT)+1 {
		return fmt.Errorf("diagT must be len(sdiagT) + 1")
	}
	if Q.Nrows() != Q.Ncols() {
		return fmt.Errorf("Q must be square Matrix64")
	}
	if len(diagT) != Q.Nrows() {
		return fmt.Errorf("Q.Nrows() and Q.Ncols() must be len(diagT)")
	}
	for p := 0; p < len(diagT)-1; p++ {
		for {
			// Find the first zero super-diagonal element.
			q := p
			for ; q < len(diagT)-1; q++ {
				s := math.Abs(diagT[q]) + math.Abs(diagT[q+1])
				if s == s+math.Abs(sdiagT[q]) {
					// sdiagT[q] is enough small.
					sdiagT[q] = 0.0
					break
				}
			}
			// If T[p,p+1] = T[p+1,p] = 0, break in order to start from p + 1.
			if p == q {
				break
			}
			// Diagonalize T[p:q,p:q] with Wilkinson shift.
			// Wilkinson shift is the eigenvalue of T[q-1:q,q-1:q] close to T[q,q].
			// See also LAPACK subroutine dlaev2 for high-accurate implementation.
			alpha, beta, gamma := diagT[q-1], diagT[q], sdiagT[q-1]
			// Eigenvalues are (alpha + beta)/2 +/- hypot((alpha - beta), 2.0gamma)/2.
			alphaPlusBeta, alphaMinusBeta, twoGamma := alpha+beta, alpha-beta, 2.0*gamma
			D := math.Hypot(alphaMinusBeta, twoGamma)
			// The absolute value of eigen1 is larger than or equal to the one of eigen2.
			eigen1 := 0.0
			// If alpha + beta is negative, then the smaller root is the larger eigenvalue.
			if alphaPlusBeta < 0.0 {
				eigen1 = (alphaPlusBeta - D) / 2.0
			} else {
				eigen1 = (alphaPlusBeta + D) / 2.0
			}
			// Calculate the other eigenvalue carefully.
			eigen2 := (math.Max(alpha, beta)/eigen1)*math.Min(alpha, beta) - (gamma/eigen1)*gamma
			shift := eigen1
			if math.Abs(eigen1-beta) > math.Abs(eigen2-beta) {
				shift = eigen1
			}
			if p == q-1 {
				// Solve the trailing 2-by-2-diagonalization directly.
				//      / c   s \
				// S := |       |: eigenvectors in the column order of eigen1 and eigen2.
				//      \ -s  c /
				// Here, (c, s) = (2gamma, beta - alpha + D).
				c, s := twoGamma, 0.0
				// Choose the numerically stable one.
				if alphaMinusBeta < 0.0 {
					s = alphaMinusBeta - D
				} else {
					s = alphaMinusBeta + D
				}
				// Normalize (c, s) carefully:
				//   (c, s) |-> (c/r, s/r), r = sqrt(c**2 + s**2)
				if math.Abs(c) < math.Abs(s) {
					ratio := -c / s
					s = 1.0 / math.Sqrt(1.0+ratio*ratio)
					c = ratio * s
				} else {
					ratio := -s / c
					c = 1.0 / math.Sqrt(1.0+ratio*ratio)
					s = ratio * c
				}
				// If the absolute value of alpha is larger than or equal to the one of beta, then flip two eigenvectors.
				if math.Signbit(alphaPlusBeta) == math.Signbit(alphaMinusBeta) {
					c, s = -s, c
				}
				// Apply the eigenvector matrix to Q: Q |-> QS.
				for i := 0; i < Q.Nrows(); i++ {
					q1, q2 := Q.Elems()[i+(q-1)*Q.Nrows()], Q.Elems()[i+q*Q.Nrows()]
					//Q.Elems()[i+(q-1)*Q.Nrows()], Q.Elems()[i+q*Q.Nrows()] = s1*q1+s3*q2, s2*q1+s4*q2
					Q.Elems()[i+(q-1)*Q.Nrows()], Q.Elems()[i+q*Q.Nrows()] = c*q1+s*q2, -s*q1+c*q2
				}
				diagT[q-1], diagT[q] = eigen1, eigen2
				sdiagT[q-1] = 0
				break
			}
			// Set the rotated diagonal and super-diagonal as one at the p- and p+1-th row.
			x, z := diagT[p]-shift, sdiagT[p]
			bulge := 0.0
			for k := p; k < q; k++ {
				// Calculate the Givens rotation: (x, z) |-> (sqrt(x**2 + z**2), 0).
				c, s := CalculateGivensRotation(x, z)
				// Apply the Givens rotation to T:
				// / t1  t2 \    / T[k,k]    T[k,k+1]   \/ c   s \
				// |        | := |                      ||       |
				// \ t3  t4 /    \ T[k+1,k]  T[k+1,k+1] /\ -s  c /
				t1, t2 := c*diagT[k]-s*sdiagT[k], s*diagT[k]+c*sdiagT[k]
				t3, t4 := c*sdiagT[k]-s*diagT[k+1], s*sdiagT[k]+c*diagT[k+1]
				// / T[k,k]    T[k,k+1]   \   / c  -s \/ t1  t2 \
				// |                      | = |       ||        |
				// \ T[k+1,k]  T[k+1,k+1] /   \ s   c /\ t3  t4 /
				if k > 0 {
					// Update T[k-1,k] = T[k,k-1]
					sdiagT[k-1] = c*sdiagT[k-1] - s*bulge
				}
				diagT[k], sdiagT[k] = c*t1-s*t3, s*t1+c*t3
				diagT[k+1] = s*t2 + c*t4
				if k+1 < q {
					// Calculate the bulge appears upon T[k+1,k+2] before T[k+1,k+2] = T[k+2,k+1] is updated.
					bulge = -s * sdiagT[k+1]
					// Update T[k+1,k+2] = T[k+2,k+1]
					sdiagT[k+1] *= c
				}
				// Apply the Givens rotation to Q: Q |-> QR.
				for i := 0; i < Q.Nrows(); i++ {
					q1, q2 := Q.Elems()[i+k*Q.Nrows()], Q.Elems()[i+(k+1)*Q.Nrows()]
					Q.Elems()[i+k*Q.Nrows()], Q.Elems()[i+(k+1)*Q.Nrows()] = c*q1-s*q2, s*q1+c*q2
				}
				x, z = sdiagT[k], bulge
			}
		}
	}
	return nil
}

// Tridiagonalize A with iterative Householder transformations such as A = Q T t(Q).
// The size of Q must be enough to store all Householder reflections in Q.
// A and Q are overwritten with the results.
// This function assumes that A is symmetric.
func TridiagonalizeSymmetricMatrix64(A, Q *Matrix64) error {
	if !((A.Ncols() <= Q.Nrows()) && (A.Ncols() <= Q.Ncols())) {
		return fmt.Errorf("size of Q must be enough large")
	}
	n := A.Nrows()
	p := make(Vector64, n-1)
	for k := 0; k < n-2; k++ {
		// Calculate Householder transformations I - \beta vt(v) such that A[k+1:n,k] |-> (||A[k+1:n,k]||_2, 0, ..., 0):
		//   /       ***                                  \
		//   | T[1:k-1,1:k-1]                             |
		//   |       ***       T[k-1,k]                   |
		//   |       T[k,k-1]  T[k,k]      A[k,k+1:n]     |
		//   |                 A[k+1:n,k]        ***      |
		//   \                      *      A[k+1:n,k+1:n] /
		// Reference:
		//   Golub, G. H., and Van Loan, C. F. Matrix computations 4th Edition. JHU Press, 2012, pp. 234-236.
		// v := A[k+1:n,k] - ||A[k+1:n,k]||_2e_1
		beta, v := 0.0, A.SlicedColumns(k, 1).Elems()[k+1:]
		vnorm := L2norm64(v)
		sigma := L2norm64(v[1:])
		if v[0]+sigma == v[0] {
			if v[0] < 0.0 {
				beta = 2.0 / (vnorm * vnorm)
			}
		} else {
			v[0] -= vnorm
			beta = 2.0 / (sigma*sigma + v[0]*v[0])
		}
		// Update A[k+1:n,k+1:n]
		pk := p[:n-(k+1)]
		for i := k + 1; i < n; i++ {
			s := 0.0
			for j := k + 1; j < n; j++ {
				s += A.Elems()[i+j*n] * v[j-(k+1)]
			}
			pk[i-(k+1)] = beta * s
		}
		pk.Madd(-beta*Dot64(pk, v)/2.0, v)
		for i := k + 1; i < n; i++ {
			for j := k + 1; j < n; j++ {
				A.Elems()[i+j*n] -= v[i-(k+1)]*pk[j-(k+1)] + pk[i-(k+1)]*v[j-(k+1)]
			}
		}
		// Update Q as QH_k, H_k := I_n - \beta vt(v)
		for i := 0; i < n; i++ {
			s := 0.0
			for j := k + 1; j < n; j++ {
				s += Q.Elems()[i+j*n] * v[j-(k+1)]
			}
			s *= beta
			for j := k + 1; j < n; j++ {
				Q.Elems()[i+j*n] -= s * v[j-(k+1)]
			}
		}
		// Update A[k+1:n,k] and A[k,k+1:n].
		A.Elems()[(k+1)+k*n] = vnorm
		A.Elems()[k+(k+1)*n] = vnorm
		for i := k + 2; i < n; i++ {
			A.Elems()[i+k*n] = 0.0
			A.Elems()[k+i*n] = 0.0
		}
	}
	return nil
}

// EVD64 stores the result of Eigen-Value Decomposition (EVD) of SparseMatrix64 A.
type EVD64 struct {
	A      SparseMatrix64
	Lambda Vector64
	V      *Matrix64
}

// Applies EVD on SparseMatrix64 A, and returns EVD64 containing k eigen values and corresponding eigenvectors.
// This function assumes that A is symmetric.
// If logger is nil, then use new Logger to ioutil.Discard.
//
// The applied algorithm is based on Thick-Restart Lanczos algorithm [Wu, Kesheng, and Horst Simon. "Thick-restart Lanczos method for large symmetric eigenvalue problems." SIAM Journal on Matrix Analysis and Applications, vol. 22, no. 2, pp. 602-616, 2000.].
// p is nuisance dimension used in Krylov subspace. If p is not positive, p is automatically selected.
func NewSymmetricEVD64(A SparseMatrix64, k, p int, logger *log.Logger) (*EVD64, error) {
	if logger == nil {
		logger = log.New(ioutil.Discard, "", log.LstdFlags)
	}
	// p: nuisance dimension
	if p > 0 {
		if k+p > A.Nrows() {
			return nil, fmt.Errorf("p must be less than A.Nrows() - k")
		}
	} else {
		// Select nuisance dimension automatically.
		p = k
		if p < 2 {
			p = 2
		}
		if k+p >= A.Nrows() {
			p = A.Nrows() - k
		}
	}
	k0 := k
	m := k + p
	// The matrix of eigen vectors
	V, Vold := NewMatrix64(A.Nrows(), m), NewMatrix64(A.Nrows(), m)
	// The vector of diagonal (and super-/sub-diagonal) elements of symmetric tridiagonal T.
	diagT := make(Vector64, m)
	sdiagT := make(Vector64, m-1)
	// The residual vector of Lanczos steps.
	r := make(Vector64, A.Nrows())
	// Repeat the implicit restart until T is diagonal.
	for niters := 1; ; niters++ {
		// Apply the initial/additional Lanczos steps.
		start, end := k, m
		if niters == 1 {
			start = 0
			logger.Printf("niters=%d: applying the initial Lanczos steps ...\n", niters)
		} else {
			logger.Printf("niters=%d: applying additional Lanczos steps ...\n", niters)
		}
		if err := ApplyLanczosSteps(A, V, diagT, sdiagT, r, start, end); err != nil {
			return nil, err
		}
		tVwStr := ""
		for j := 0; j < end; j++ {
			if j == start {
				tVwStr += " ::: "
			} else if j > 0 {
				tVwStr += " "
			}
			tVwStr += fmt.Sprintf("%.5g", Dot64(r, V.SlicedColumns(j, 1).Elems()))
		}
		logger.Printf("t(V)w=[%s] (updated: %d - %d)\n", tVwStr, start, end)
		// Diagonalize the symmetric tridiagonal T.
		Q := NewMatrix64I(m)
		if (niters > 1) && (k >= 2) {
			// Recovering tridiagonal form destroyed by thick-restart process with Householder reflection.
			A := NewMatrix64(m, m)
			for i := 0; i < m; i++ {
				A.Elems()[i+i*A.Nrows()] = diagT[i]
			}
			for i := 0; i < k; i++ {
				A.Elems()[k+i*A.Nrows()] = sdiagT[i]
				A.Elems()[i+k*A.Nrows()] = sdiagT[i]
			}
			for i := k; i < m-1; i++ {
				A.Elems()[i+(i+1)*A.Nrows()] = sdiagT[i]
				A.Elems()[(i+1)+i*A.Nrows()] = sdiagT[i]
			}
			if err := TridiagonalizeSymmetricMatrix64(A, Q); err != nil {
				return nil, err
			}
			for i := 0; i < m; i++ {
				diagT[i] = A.Elems()[i+i*A.Nrows()]
			}
			for i := 0; i < m-1; i++ {
				sdiagT[i] = A.Elems()[i+(i+1)*A.Nrows()]
			}
		}
		logger.Printf("niters=%d: diagonalizing the symmetric tridiagonal T ...\n", niters)
		DiagonalizeSymmetricTridiagonal64(diagT, sdiagT, Q)
		logger.Printf("niters=%d: ||r||_2=%g", niters, L2norm64(r))
		// Sort the eigenvalues and corresponding eigenvectors in the decreasing order of eigenvalues.
		logger.Printf("niters=%d: sorting the eigenvalues and corresponding eigenvectors in the decreasing order of eigenvalues ...\n", niters)
		sort.Stable(sort.Reverse(&Matrix64WithColumnValues{Q, diagT}))
		// Apply Q to V
		logger.Printf("niters=%d: appling Q to V ...\n", niters)
		V, Vold = Vold, V
		V.MatMul(Vold, Q)
		// Check the convergence:
		//   AV_m = V_m T_m + r_mt(e_m), T_m = Q_m Lambda_m t(Q_m)
		//   => AV_mQ_m = V_mQ_m Lambda_m + r_mQ_m[m,:]
		// Thus, the last term of the right-hand side is the residual term.
		logger.Printf("niters=%d: checking the convergence ...\n", niters)
		nconvs := 0
		residualsStr := ""
		rnorm := L2norm64(r)
		for j := 0; j < k0; j++ {
			residual := rnorm * Q.Elems()[(m-1)+j*Q.Nrows()]
			if j > 0 {
				residualsStr += " "
			}
			if Round64(math.Abs(residual), FLOAT64_NORMAL_PRECISION) == 0.0 {
				nconvs++
				residualsStr += "_"
				residual = 0.0
			} else {
				residualsStr += fmt.Sprintf("%g", residual)
			}
			// Update the multiplicative part of the residual vector
			if j < m-1 {
				sdiagT[j] = residual
			}
		}
		logger.Printf("niters=%d: %d vector(s) converged: residuals=[%s]\n", niters, nconvs, residualsStr)
		logger.Printf("niters=%d: diagT[0:nconvs]=%#v\n", niters, diagT[0:nconvs])
		if nconvs == k0 {
			break
		}
		// Avoid possible stagnation and accelerate the convergence.
		kprev := k
		if nconvs <= (m-k)/2 {
			k = k0 + nconvs
		} else {
			k = k0 + (m-k)/2
		}
		if k > m-2 {
			k = m - 2
		}
		if k < kprev {
			k = kprev
		}
		for j := kprev; j < k; j++ {
			residual := rnorm * Q.Elems()[(m-1)+j*Q.Nrows()]
			if Round64(math.Abs(residual), FLOAT64_NORMAL_PRECISION) == 0.0 {
				residual = 0.0
			}
			// Update the multiplicative part of the residual vector
			if j < m-1 {
				sdiagT[j] = residual
			}
		}
		logger.Printf("niters=%d: updated k from %d to %d for avoiding stagnation", niters, kprev, k)
	}
	return &EVD64{
		A:      A,
		Lambda: diagT[0:k0],
		V:      V.SlicedColumns(0, k0),
	}, nil
}

// Read the serialized EVD64 from reader.
func ReadEVD64(reader io.Reader) (*EVD64, error) {
	evd := &EVD64{}
	var info [2]uint64
	if err := binary.Read(reader, binary.LittleEndian, &info); err != nil {
		return nil, err
	}
	n, k := int(info[0]), int(info[1])
	evd.Lambda = make(Vector64, k)
	evd.V = NewMatrix64(n, k)
	if err := binary.Read(reader, binary.LittleEndian, evd.Lambda); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, evd.V.Elems()); err != nil {
		return nil, err
	}
	return evd, nil
}

// Open the file and read the serialized EVD64.
func OpenEVD64(name string) (*EVD64, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return ReadEVD64(bufio.NewReader(file))
}

// Write the serialized EVD64 to writer.
func (evd *EVD64) Write(writer io.Writer) error {
	if err := binary.Write(writer, binary.LittleEndian, [2]uint64{uint64(evd.A.Ncols()), uint64(len(evd.Lambda))}); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, evd.Lambda); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, evd.V.Elems()); err != nil {
		return err
	}
	return nil
}

// Open the file and write the serialized EVD64.
func (evd *EVD64) WriteToFile(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	err = evd.Write(writer)
	writer.Flush()
	return err
}

// SVD64 stores the result of Singular Value Decomposition (SVD) of SparseMatrix64 A.
type SVD64 struct {
	A     SparseMatrix64
	Sigma Vector64
	// The matrix of left-singular vectors
	U *Matrix64
	// The matrix of right-singular vectors
	V *Matrix64
}

// Applies SVD on SparseMatrix64 A, and returns SVD64 containing k singular values and corresponding left-/right-singular vectors.
// Apply EVD on t(A)A, then get singular values and right-singular vectors.
// Next, calculate left-singular vectors from right-singular vectors.
// The left-singular vectors corresponding to singular value 0 are zero vectors.
// See NewSymmetricEVD64 for treatment of argument logger.
func NewSVD64(A SparseMatrix64, k, p int, logger *log.Logger) (*SVD64, error) {
	mA, nA := A.Nrows(), A.Ncols()
	if k < 0 {
		return nil, nil
	} else if k == 0 {
		return &SVD64{
			A:     A,
			Sigma: Vector64{},
			U:     NewMatrix64(mA, k),
			V:     NewMatrix64(nA, k),
		}, nil
	}
	evd, err := NewSymmetricEVD64(Symmetrize64(A), k, p, logger)
	if err != nil {
		return nil, err
	}
	// Returns the result.
	sigma := evd.Lambda[0:k]
	for i := 0; i < k; i++ {
		sigma[i] = math.Pow(Round64(sigma[i], FLOAT64_NORMAL_PRECISION), 0.5)
	}
	V := evd.V.SlicedColumns(0, k)
	U := NewMatrix64(mA, k)
	for i := 0; i < k; i++ {
		if sigma[i] != 0.0 {
			U.SlicedColumns(i, 1).Elems().Apply(A, V.SlicedColumns(i, 1).Elems()).Mul(1.0 / sigma[i])
		}
	}
	return &SVD64{
		A:     A,
		Sigma: sigma,
		U:     U,
		V:     V,
	}, nil
}

// Read the serialized SVD64 from reader.
func ReadSVD64(reader io.Reader) (*SVD64, error) {
	svd := &SVD64{}
	var info [3]uint64
	if err := binary.Read(reader, binary.LittleEndian, &info); err != nil {
		return nil, err
	}
	nrows, ncols, k := int(info[0]), int(info[1]), int(info[2])
	svd.Sigma = make(Vector64, k)
	svd.U = NewMatrix64(nrows, k)
	svd.V = NewMatrix64(ncols, k)
	if err := binary.Read(reader, binary.LittleEndian, svd.Sigma); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, svd.U.Elems()); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, svd.V.Elems()); err != nil {
		return nil, err
	}
	return svd, nil
}

// Open the file, and read the serialized SVD64.
func OpenSVD64(name string) (*SVD64, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return ReadSVD64(bufio.NewReader(file))
}

// Write the serialized SVD64 to writer.
func (svd *SVD64) Write(writer io.Writer) error {
	if err := binary.Write(writer, binary.LittleEndian, [3]uint64{uint64(svd.A.Nrows()), uint64(svd.A.Ncols()), uint64(len(svd.Sigma))}); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, svd.Sigma); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, svd.U.Elems()); err != nil {
		return err
	}
	if err := binary.Write(writer, binary.LittleEndian, svd.V.Elems()); err != nil {
		return err
	}
	return nil
}

// Open the file, and write the serialized SVD64.
func (svd *SVD64) WriteToFile(name string) error {
	file, err := os.Create(name)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	err = svd.Write(writer)
	writer.Flush()
	return err
}
