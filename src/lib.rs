#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))] use nalgebra::ComplexField;
use nalgebra::{SMatrix, Vector2, Matrix3, Matrix3x1, Const, Vector3};
use types::{SolverParameters, SQPSolution, OmegaNullspaceMethod};

use crate::types::NearestRotationMethod;

pub mod types;
mod sqpnp;

pub const SQRT3: f32 = 1.732050807568877293527446341505872367_f32;

#[allow(non_snake_case)]
pub struct PnpSolver<'input> {
    projections: &'input [Vector2<f32>],
    points: &'input [Vector3<f32>],
    weights: &'input [f32],
    parameters: SolverParameters,

    omega: SMatrix<f32, 9, 9>,

    s: SMatrix<f32, 9, 1>,
    U: SMatrix<f32, 9, 9>,
    P: SMatrix<f32, 3, 9>,
    point_mean: Vector3<f32>,

    num_null_vectors: i32,
    solutions: [SQPSolution; 18],
    num_solutions: usize,
    nearest_rotation_matrix: fn(&SMatrix<f32, 9, 1>, &mut SMatrix<f32, 9, 1>),
}

impl<'input> PnpSolver<'input> {
    pub fn omega(&self) -> &SMatrix<f32, 9, 9> {
        &self.omega
    }
    pub fn eigen_vectors(&self) -> &SMatrix<f32, 9, 9> {
        &self.U
    }
    pub fn eigen_values(&self) -> &SMatrix<f32, 9, 1> {
        &self.s
    }
    pub fn null_space_dimension(&self) -> i32 {
        self.num_null_vectors
    }
    pub fn number_of_solutions(&self) -> usize {
        self.num_solutions
    }
    pub fn solution_ptr(&self, index: usize) -> Option<&SQPSolution> {
        if index >= self.num_solutions {
            None
        } else {
            Some(&self.solutions[index])
        }
    }

    /// Return average reprojection errors
    ///
    /// Panics if `errors.len() < self.number_of_solutions()`.
    pub fn average_squared_projection_errors(&self, errors: &mut [f32]) {
        assert!(errors.len() >= self.num_solutions);
        for i in 0..self.num_solutions {
            errors[i] = self.average_squared_projection_error(i)
        }
    }

    /// Constructor (initializes Omega, P and U, s, i.e. the decomposition of Omega)
    #[allow(non_snake_case)]
    pub fn new(
        points: &'input [Vector3<f32>],
        projections: &'input [Vector2<f32>],
        weights: Option<&'input [f32]>,
        parameters: SolverParameters,
    ) -> Option<Self> {
        let n = points.len();
        if n != projections.len() || n < 3 {
            return None;
        }

        let weights = weights.unwrap_or_else(|| &[]);
        if weights.len() != 0 && weights.len() != n {
            return None;
        }

        let mut num_null_vectors = -1;
        let mut omega = SMatrix::<f32, 9, 9>::zeros();
        let mut sum_wx @ mut sum_wy @ mut sum_wx2_plus_wy2 @ mut sum_w = 0.0;

        let mut sum_wX @ mut sum_wY @ mut sum_wZ = 0.0;

        let mut QA = SMatrix::<f32, 3, 9>::zeros();

        for i in 0..n {
            let w = *weights.get(i).unwrap_or(&1.0);

            if w == 0.0 {
                continue;
            }

            let proj = &projections[i];
            let wx = proj[0] * w;
            let wy = proj[1] * w;
            let wsq_norm_m = w*proj.norm_squared();
            sum_wx += wx;
            sum_wy += wy;
            sum_wx2_plus_wy2 += wsq_norm_m;
            sum_w += w;

            let pt = &points[i];
            let X = pt[0];
            let Y = pt[1];
            let Z = pt[2];
            let wX = w * X;
            let wY = w * Y;
            let wZ = w * Z;
            sum_wX += wX;
            sum_wY += wY;
            sum_wZ += wZ;

            // Accumulate Omega by kronecker( Qi, Mi*Mi' ) = A'*Qi*Ai. NOTE: Skipping block (3:5, 3:5) because it's same as (0:2, 0:2)
            let X2 = X*X;
            let XY = X*Y;
            let XZ = X*Z;
            let Y2 = Y*Y;
            let YZ = Y*Z;
            let Z2 = Z*Z;

            // a. Block (0:2, 0:2) populated by Mi*Mi'. NOTE: Only upper triangle
            omega[(0, 0)] += w*X2;
            omega[(0, 1)] += w*XY;
            omega[(0, 2)] += w*XZ;
            omega[(1, 1)] += w*Y2;
            omega[(1, 2)] += w*YZ;
            omega[(2, 2)] += w*Z2;

            // b. Block (0:2, 6:8) populated by -x*Mi*Mi'. NOTE: Only upper triangle
            omega[(0, 6)] -= wx*X2; omega[(0, 7)] -= wx*XY; omega[(0, 8)] -= wx*XZ;
                                    omega[(1, 7)] -= wx*Y2; omega[(1, 8)] -= wx*YZ;
                                                            omega[(2, 8)] -= wx*Z2;

            // c. Block (3:5, 6:8) populated by -y*Mi*Mi'. NOTE: Only upper triangle
            omega[(3, 6)] -= wy*X2; omega[(3, 7)] -= wy*XY; omega[(3, 8)] -= wy*XZ;
                                    omega[(4, 7)] -= wy*Y2; omega[(4, 8)] -= wy*YZ;
                                                            omega[(5, 8)] -= wy*Z2;

            // d. Block (6:8, 6:8) populated by (x^2+y^2)*Mi*Mi'. NOTE: Only upper triangle
            omega[(6, 6)] += wsq_norm_m*X2; omega[(6, 7)] += wsq_norm_m*XY; omega[(6, 8)] += wsq_norm_m*XZ;
                                            omega[(7, 7)] += wsq_norm_m*Y2; omega[(7, 8)] += wsq_norm_m*YZ;
                                                                            omega[(8, 8)] += wsq_norm_m*Z2;

            // Accumulating Qi*Ai in QA.
            // Note that certain pairs of elements are equal, so we save some operations by filling them outside the loop
            QA[(0, 0)] += wX; QA[(0, 1)] += wY; QA[(0, 2)] += wZ;   QA[(0, 6)] -= wx*X; QA[(0, 7)] -= wx*Y; QA[(0, 8)] -= wx*Z;
            //QA[(1, 3)] += wX; QA[(1, 4)] += wY; QA[(1, 5)] += wZ;
                                                                    QA[(1, 6)] -= wy*X; QA[(1, 7)] -= wy*Y; QA[(1, 8)] -= wy*Z;

            //QA[(2, 0)] -= wx*X; QA[(2, 1)] -= wx*Y; QA[(2, 2)] -= wx*Z;  QA[(2, 3)] -= wy*X; QA[(2, 4)] -= wy*Y; QA[(2, 5)] -= wy*Z;
            QA[(2, 6)] += wsq_norm_m*X; QA[(2, 7)] += wsq_norm_m*Y; QA[(2, 8)] += wsq_norm_m*Z;
        }


        // Complete QA
        QA[(1, 3)] = QA[(0, 0)]; QA[(1, 4)] = QA[(0, 1)]; QA[(1, 5)] = QA[(0, 2)];
        QA[(2, 0)] = QA[(0, 6)]; QA[(2, 1)] = QA[(0, 7)]; QA[(2, 2)] = QA[(0, 8)];
        QA[(2, 3)] = QA[(1, 6)]; QA[(2, 4)] = QA[(1, 7)]; QA[(2, 5)] = QA[(1, 8)];

        // Fill-in lower triangles of off-diagonal blocks (0:2, 6:8), (3:5, 6:8) and (6:8, 6:8)
        omega[(1, 6)] = omega[(0, 7)]; omega[(2, 6)] = omega[(0, 8)]; omega[(2, 7)] = omega[(1, 8)];
        omega[(4, 6)] = omega[(3, 7)]; omega[(5, 6)] = omega[(3, 8)]; omega[(5, 7)] = omega[(4, 8)];
        omega[(7, 6)] = omega[(6, 7)]; omega[(8, 6)] = omega[(6, 8)]; omega[(8, 7)] = omega[(7, 8)];

        // Fill-in upper triangle of block (3:5, 3:5)
        omega[(3, 3)] = omega[(0, 0)]; omega[(3, 4)] = omega[(0, 1)]; omega[(3, 5)] = omega[(0, 2)];
        omega[(4, 4)] = omega[(1, 1)]; omega[(4, 5)] = omega[(1, 2)];
        omega[(5, 5)] = omega[(2, 2)];

        // Fill lower triangle of Omega; elements (7, 6), (8, 6) & (8, 7) have already been assigned above
        omega[(1, 0)] = omega[(0, 1)];
        omega[(2, 0)] = omega[(0, 2)]; omega[(2, 1)] = omega[(1, 2)];
        omega[(3, 0)] = omega[(0, 3)]; omega[(3, 1)] = omega[(1, 3)]; omega[(3, 2)] = omega[(2, 3)];
        omega[(4, 0)] = omega[(0, 4)]; omega[(4, 1)] = omega[(1, 4)]; omega[(4, 2)] = omega[(2, 4)]; omega[(4, 3)] = omega[(3, 4)];
        omega[(5, 0)] = omega[(0, 5)]; omega[(5, 1)] = omega[(1, 5)]; omega[(5, 2)] = omega[(2, 5)]; omega[(5, 3)] = omega[(3, 5)]; omega[(5, 4)] = omega[(4, 5)];
        omega[(6, 0)] = omega[(0, 6)]; omega[(6, 1)] = omega[(1, 6)]; omega[(6, 2)] = omega[(2, 6)]; omega[(6, 3)] = omega[(3, 6)]; omega[(6, 4)] = omega[(4, 6)]; omega[(6, 5)] = omega[(5, 6)];
        omega[(7, 0)] = omega[(0, 7)]; omega[(7, 1)] = omega[(1, 7)]; omega[(7, 2)] = omega[(2, 7)]; omega[(7, 3)] = omega[(3, 7)]; omega[(7, 4)] = omega[(4, 7)]; omega[(7, 5)] = omega[(5, 7)];
        omega[(8, 0)] = omega[(0, 8)]; omega[(8, 1)] = omega[(1, 8)]; omega[(8, 2)] = omega[(2, 8)]; omega[(8, 3)] = omega[(3, 8)]; omega[(8, 4)] = omega[(4, 8)]; omega[(8, 5)] = omega[(5, 8)];


        // Q = Sum( wi*Qi ) = Sum( [ wi, 0, -wi*xi; 0, 1, -wi*yi; -wi*xi, -wi*yi, wi*(xi^2 + yi^2)] )
        let Q = Matrix3::new(
            sum_w,     0.0,       -sum_wx,
            0.0,       sum_w,     -sum_wy,
            -sum_wx,   -sum_wy,   sum_wx2_plus_wy2
        );
      
        // Qinv = inv( Q ) = inv( Sum( Qi) )
        // let Qinv = Q.try_inverse().unwrap();
        let mut Qinv = SMatrix::<f32, 3, 3>::zeros();
        invert_symmetric_3x3(Q, &mut Qinv);

        // Compute P = -inv( Sum(wi*Qi) ) * Sum( wi*Qi*Ai ) = -Qinv * QA
        let P = -Qinv * QA;
        // Complete Omega (i.e., Omega = Sum(A'*Qi*A') + Sum(Qi*Ai)'*P = Sum(A'*Qi*A') + Sum(Qi*Ai)'*inv(Sum(Qi))*Sum( Qi*Ai) 
        omega +=  QA.transpose()*P;

        // Finally, decompose Omega with the chosen method
        let U;
        let s;
        match parameters.omega_nullspace_method {
            // OmegaNullspaceMethod::Rrqr => {
                // Rank revealing QR nullspace computation with full pivoting.
                // This is slightly less accurate compared to SVD but x2 faster
                // Eigen::FullPivHouseholderQR<Eigen::Matrix<double, 9, 9> > rrqr(Omega_);
                // U_ = rrqr.matrixQ();
                //
                // Eigen::Matrix<double, 9, 9> R = rrqr.matrixQR().template triangularView<Eigen::Upper>();
                // s_ = R.diagonal().array().abs();
            // }
            OmegaNullspaceMethod::Cprrqr => {
                // Rank revealing QR nullspace computation with column pivoting.
                // This is potentially less accurate compared to RRQR but faster

                // Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 9> > cprrqr(Omega_);
                // U_ = cprrqr.householderQ();
                //
                // Eigen::Matrix<double, 9, 9> R = cprrqr.matrixR().template triangularView<Eigen::Upper>();
                // s_ = R.diagonal().array().abs();

                let cprrqr = omega.col_piv_qr();
                U = cprrqr.q();
                s = cprrqr.unpack_r().diagonal().abs();
            }
            OmegaNullspaceMethod::Svd => {
                // SVD-based nullspace computation. This is the most accurate but slowest option
                // Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(Omega_, Eigen::ComputeFullU);
                let svd = omega.svd(true, false);
                U = svd.u.unwrap();
                s = svd.singular_values;
            }
        }

        // Find dimension of null space; the check guards against overly large rank_tolerance
        while 7 - num_null_vectors >= 0 && s[(7 - num_null_vectors) as usize] < parameters.rank_tolerance {
            num_null_vectors += 1;
        }

        // Dimension of null space of Omega must be <= 6
        if num_null_vectors > 6 {
            return None;
        }
        num_null_vectors += 1;

        // 3D point weighted mean (for quick cheirality checks)
        let inv_sum_w = 1.0 / sum_w;
        let point_mean = Matrix3x1::new(sum_wX*inv_sum_w, sum_wY*inv_sum_w, sum_wZ*inv_sum_w);

        // Assign nearest rotation method
        let nearest_rotation_matrix = match parameters.nearest_rotation_method {
            NearestRotationMethod::Foam => nearest_rotation_matrix_foam,
            NearestRotationMethod::Svd => nearest_rotation_matrix_svd,
        };

        Some(Self {
            projections,
            points,
            weights,
            parameters,
            omega,
            s,
            U,
            P,
            point_mean,
            num_null_vectors,
            solutions: Default::default(),
            num_solutions: 0,
            nearest_rotation_matrix,
        })
    }
}

impl PnpSolver<'_> {
    fn average_squared_projection_error(&self, index: usize) -> f32 {
        let mut avg = 0.0;
        let r = &self.solutions[index].r_hat;
        let t = &self.solutions[index].t;

        for i in 0..self.points.len() {
            let m = &self.points[i];
            let x = r[0]*m[0] + r[1]*m[1] + r[2]*m[2] + t[0];
            let y = r[3]*m[0] + r[4]*m[1] + r[5]*m[2] + t[1];
            let inv_z = 1.0 / ( r[6]*m[0] + r[7]*m[1] + r[8]*m[2] + t[2] );

            let m = &self.projections[i];
            let dx = x*inv_z - m[0];
            let dy = y*inv_z - m[1];
            avg += dx*dx + dy*dy;
        }

        return avg / self.points.len() as f32;
    }

    /// Test cheirality on the mean point for a given solution
    fn test_positive_depth(&self, solution: &SQPSolution) -> bool {
        let r = &solution.r_hat;
        let t = &solution.t;
        let m = &self.point_mean;
        return r[6]*m[0] + r[7]*m[1] + r[8]*m[2] + t[2] > 0.;
    }

    /// Test cheirality on the majority of points for a given solution
    fn test_positive_majority_depths(&self, solution: &SQPSolution) -> bool {
        let r = &solution.r_hat;
        let t = &solution.t;
        let mut npos = 0;
        let mut nneg = 0;

        for i in 0..self.points.len() {
            if *self.weights.get(i).unwrap_or(&1.0) == 0.0 { continue; }
            let m = &self.points[i];
            if r[6]*m[0] + r[7]*m[1] + r[8]*m[2] + t[2] > 0. {
                npos += 1;
            } else {
                nneg += 1;
            }
        }

        return npos >= nneg;
    }
}


/// Determinant of 3x3 matrix stored as a 9x1 vector in *row-major* order
fn determinant_9x1(r: &SMatrix<f32, 9, 1>) -> f32 {
    (r[0]*r[4]*r[8] + r[1]*r[5]*r[6] + r[2]*r[3]*r[7]) - (r[6]*r[4]*r[2] + r[7]*r[5]*r[0] + r[8]*r[3]*r[1])
}


/// Invert a 3x3 symmetric matrix (using low triangle values only)
#[allow(non_snake_case)]
fn invert_symmetric_3x3(
    Q: SMatrix<f32, 3, 3>,
    Qinv: &mut SMatrix<f32, 3, 3>,
) -> bool {
    let det_threshold = 1e-10;
    // 1. Get the elements of the matrix
    let (a, b, c, d, e, f);
    a = Q[(0, 0)];
    b = Q[(1, 0)]; d = Q[(1, 1)];
    c = Q[(2, 0)]; e = Q[(2, 1)]; f = Q[(2, 2)];

    // 2. Determinant
    let (t2, t4, t7, t9, t12);
    t2 = e*e;
    t4 = a*d;
    t7 = b*b;
    t9 = b*c;
    t12 = c*c;
    let det = -t4*f+a*t2+t7*f-2.0*t9*e+t12*d;
  
    // TODO nalgebra doesn't have complete orthogonal decomposition
    // if det.abs() < det_threshold { *Qinv=Q.completeOrthogonalDecomposition().pseudoInverse(); return false; } // fall back to pseudoinverse
    if det.abs() < det_threshold { *Qinv=Q.pseudo_inverse(1e-10).unwrap(); return false; } // fall back to pseudoinverse

    // 3. Inverse
    let (t15, t20, t24, t30);
    t15 = 1.0/det;
    t20 = (-b*f+c*e)*t15;
    t24 = (b*e-c*d)*t15;
    t30 = (a*e-t9)*t15;
    Qinv[(0, 0)] = (-d*f+t2)*t15;
    Qinv[(0, 1)] = -t20;
    Qinv[(1, 0)] = -t20;
    Qinv[(0, 2)] = -t24;
    Qinv[(2, 0)] = -t24;
    Qinv[(1, 1)] = -(a*f-t12)*t15;
    Qinv[(1, 2)] = t30;
    Qinv[(2, 1)] = t30;
    Qinv[(2, 2)] = -(t4-t7)*t15;

    true
}

/// Simple SVD - based nearest rotation matrix. Argument should be a *row-major* matrix representation.
/// Returns a row-major vector representation of the nearest rotation matrix.
#[allow(non_snake_case)]
fn nearest_rotation_matrix_svd(e: &SMatrix<f32, 9, 1>, r: &mut SMatrix<f32, 9, 1>) {
    let E = e.reshape_generic(Const::<3>, Const::<3>);
    let svd = E.svd(true, true);
    let detUV = svd.u.unwrap().determinant() * svd.v_t.unwrap().determinant();
    // so we return back a row-major vector representation of the orthogonal matrix
    let R = svd.u.unwrap() * Matrix3::from_partial_diagonal(&[1., 1., detUV]) * svd.v_t.unwrap().transpose();
    *r = R.reshape_generic(Const, Const);
}


/// Faster nearest rotation computation based on FOAM. See M. Lourakis: "An Efficient Solution to Absolute Orientation", ICPR 2016
/// and M. Lourakis, G. Terzakis: "Efficient Absolute Orientation Revisited", IROS 2018.
///
/// Solve the nearest orthogonal approximation problem
/// i.e., given B, find R minimizing ||R-B||_F
/// 
/// The computation borrows from Markley's FOAM algorithm
/// "Attitude Determination Using Vector Observations: A Fast Optimal Matrix Algorithm", J. Astronaut. Sci. 1993.
/// 
///  Copyright (C) 2019 Manolis Lourakis (lourakis **at** ics forth gr)
///  Institute of Computer Science, Foundation for Research & Technology - Hellas
///  Heraklion, Crete, Greece.
/// 
#[allow(non_snake_case)]
fn nearest_rotation_matrix_foam(e: &SMatrix<f32, 9, 1>, r: &mut SMatrix<f32, 9, 1>) {
    let mut i;
    let B = e;
    let (mut l, mut lprev, detB, Bsq, adjBsq);
    let mut adjB = [0.0; 9];

    // det(B)
    detB=(B[0]*B[4]*B[8] - B[0]*B[5]*B[7] - B[1]*B[3]*B[8]) + (B[2]*B[3]*B[7] + B[1]*B[6]*B[5] - B[2]*B[6]*B[4]);
    if detB.abs() < 1E-04 { // singular, let SVD handle it
        nearest_rotation_matrix_svd(e, r);
        return;
    }

    // B's adjoint
    adjB[0]=B[4]*B[8] - B[5]*B[7]; adjB[1]=B[2]*B[7] - B[1]*B[8]; adjB[2]=B[1]*B[5] - B[2]*B[4];
    adjB[3]=B[5]*B[6] - B[3]*B[8]; adjB[4]=B[0]*B[8] - B[2]*B[6]; adjB[5]=B[2]*B[3] - B[0]*B[5];
    adjB[6]=B[3]*B[7] - B[4]*B[6]; adjB[7]=B[1]*B[6] - B[0]*B[7]; adjB[8]=B[0]*B[4] - B[1]*B[3];

    // ||B||^2, ||adj(B)||^2
    Bsq=(B[0]*B[0]+B[1]*B[1]+B[2]*B[2]) + (B[3]*B[3]+B[4]*B[4]+B[5]*B[5]) + (B[6]*B[6]+B[7]*B[7]+B[8]*B[8]);
    adjBsq=(adjB[0]*adjB[0]+adjB[1]*adjB[1]+adjB[2]*adjB[2]) + (adjB[3]*adjB[3]+adjB[4]*adjB[4]+adjB[5]*adjB[5]) + (adjB[6]*adjB[6]+adjB[7]*adjB[7]+adjB[8]*adjB[8]);

    // compute l_max with Newton-Raphson from FOAM's characteristic polynomial, i.e. eq.(23) - (26)
    l=0.5*(Bsq + 3.0); // 1/2*(trace(B*B') + trace(eye(3)))
    if detB<0.0 { l=-l; } // trB & detB have opposite signs!
    // for(i=15, lprev=0.0; fabs(l-lprev)>1E-12*fabs(lprev) && i>0; --i){
    i = 15;
    lprev = 0.0;
    while (l-lprev).abs() > 1E-12*lprev.abs() && i>0 {
        let (tmp, p, pp);

        tmp=l*l-Bsq;
        p=tmp*tmp - 8.0*l*detB - 4.0*adjBsq;
        pp=8.0*(0.5*tmp*l - detB);

        lprev=l;
        l-=p/pp;

        i -= 1;
    }

    // the rotation matrix equals ((l^2 + Bsq)*B + 2*l*adj(B') - 2*B*B'*B) / (l*(l*l-Bsq) - 2*det(B)), i.e. eq.(14) using (18), (19)
    {
        // compute (l^2 + Bsq)*B
        let mut tmp = [0.0; 9];
        let mut BBt = [0.0; 9];
        let mut denom;
        let a=l*l + Bsq;

        // BBt=B*B'
        BBt[0]=B[0]*B[0] + B[1]*B[1] + B[2]*B[2];
        BBt[1]=B[0]*B[3] + B[1]*B[4] + B[2]*B[5];
        BBt[2]=B[0]*B[6] + B[1]*B[7] + B[2]*B[8];

        BBt[3]=BBt[1];
        BBt[4]=B[3]*B[3] + B[4]*B[4] + B[5]*B[5];
        BBt[5]=B[3]*B[6] + B[4]*B[7] + B[5]*B[8];

        BBt[6]=BBt[2];
        BBt[7]=BBt[5];
        BBt[8]=B[6]*B[6] + B[7]*B[7] + B[8]*B[8];

        // tmp=BBt*B
        tmp[0]=BBt[0]*B[0] + BBt[1]*B[3] + BBt[2]*B[6];
        tmp[1]=BBt[0]*B[1] + BBt[1]*B[4] + BBt[2]*B[7];
        tmp[2]=BBt[0]*B[2] + BBt[1]*B[5] + BBt[2]*B[8];

        tmp[3]=BBt[3]*B[0] + BBt[4]*B[3] + BBt[5]*B[6];
        tmp[4]=BBt[3]*B[1] + BBt[4]*B[4] + BBt[5]*B[7];
        tmp[5]=BBt[3]*B[2] + BBt[4]*B[5] + BBt[5]*B[8];

        tmp[6]=BBt[6]*B[0] + BBt[7]*B[3] + BBt[8]*B[6];
        tmp[7]=BBt[6]*B[1] + BBt[7]*B[4] + BBt[8]*B[7];
        tmp[8]=BBt[6]*B[2] + BBt[7]*B[5] + BBt[8]*B[8];

        // compute R as (a*B + 2*(l*adj(B)' - tmp))*denom; note that adj(B')=adj(B)'
        denom=l*(l*l-Bsq) - 2.0*detB;
        denom=1.0/denom;
        r[0]=(a*B[0] + 2.0*(l*adjB[0] - tmp[0]))*denom;
        r[1]=(a*B[1] + 2.0*(l*adjB[3] - tmp[1]))*denom;
        r[2]=(a*B[2] + 2.0*(l*adjB[6] - tmp[2]))*denom;

        r[3]=(a*B[3] + 2.0*(l*adjB[1] - tmp[3]))*denom;
        r[4]=(a*B[4] + 2.0*(l*adjB[4] - tmp[4]))*denom;
        r[5]=(a*B[5] + 2.0*(l*adjB[7] - tmp[5]))*denom;

        r[6]=(a*B[6] + 2.0*(l*adjB[2] - tmp[6]))*denom;
        r[7]=(a*B[7] + 2.0*(l*adjB[5] - tmp[7]))*denom;
        r[8]=(a*B[8] + 2.0*(l*adjB[8] - tmp[8]))*denom;
    }

    //double R[9];
    //r=Eigen::Map<Eigen::Matrix<double, 9, 1>>(R);
}


/// Produce a distance from being orthogonal for a random 3x3 matrix
/// Matrix is provided as a vector
fn orthogonality_error(a: &SMatrix<f32, 9, 1>) -> f32 {
    let sq_norm_a1 = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    let sq_norm_a2 = a[3]*a[3] + a[4]*a[4] + a[5]*a[5];
    let sq_norm_a3 = a[6]*a[6] + a[7]*a[7] + a[8]*a[8];
    let dot_a1a2 = a[0]*a[3] + a[1]*a[4] + a[2]*a[5];
    let dot_a1a3 = a[0]*a[6] + a[1]*a[7] + a[2]*a[8];
    let dot_a2a3 = a[3]*a[6] + a[4]*a[7] + a[5]*a[8];

    return ((sq_norm_a1 - 1.)*(sq_norm_a1 - 1.) + (sq_norm_a2 - 1.)*(sq_norm_a2 - 1.)) + ((sq_norm_a3 - 1.)*(sq_norm_a3 - 1.) +
    2.*( dot_a1a2*dot_a1a2 + dot_a1a3*dot_a1a3 + dot_a2a3*dot_a2a3) );
}
