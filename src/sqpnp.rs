use nalgebra::SMatrix;
#[cfg(not(feature = "std"))] use nalgebra::ComplexField;

use crate::{PnpSolver, types::SQPSolution, SQRT3, orthogonality_error, determinant_9x1, invert_symmetric_3x3};


impl PnpSolver<'_> {
    fn handle_solution(&mut self, solution: &mut SQPSolution, min_sq_error: &mut f32) {
        let cheirok = self.test_positive_depth( solution ) || self.test_positive_majority_depths ( solution ); // check the majority if the check with centroid fails
        if cheirok {
            solution.sq_error = ( self.omega * solution.r_hat ).dot( &solution.r_hat );
            if ( *min_sq_error - solution.sq_error ).abs() > self.parameters.equal_squared_errors_diff {
                if *min_sq_error > solution.sq_error
                {
                    *min_sq_error = solution.sq_error;
                    self.solutions[0] = *solution;
                    self.num_solutions = 1;
                }
            } else { // look for a solution that's almost equal to this
                let mut found = false;
                for i in 0..self.num_solutions {
                    if ( self.solutions[i].r_hat - solution.r_hat ).norm_squared() < self.parameters.equal_vectors_squared_diff
                    {
                        if self.solutions[i].sq_error > solution.sq_error
                        {
                            self.solutions[i] = *solution;
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    self.solutions[self.num_solutions] = *solution;
                    self.num_solutions += 1;
                }
                if *min_sq_error > solution.sq_error { *min_sq_error = solution.sq_error; }
            }
        }
    }

    /// Solve the PnP 
    pub fn solve(&mut self) -> bool {
        let mut min_sq_error = f32::MAX;
        let num_eigen_points = if self.num_null_vectors > 0 { self.num_null_vectors as usize } else { 1 };
        // clear solutions
        self.num_solutions = 0;

        // for (int i = 9 - num_eigen_points; i < 9; i++) 
        for i in 9-num_eigen_points .. 9 {
            // NOTE: No need to scale by sqrt(3) here, but better be there for other computations (i.e., orthogonality test)
            let e = SQRT3 * self.U.column(i);
            let orthogonality_sq_error = orthogonality_error(&e);
            // Find nearest rotation vector
            let mut solution = [SQPSolution::default(); 2];

            // Avoid SQP if e is orthogonal
            if orthogonality_sq_error < self.parameters.orthogonality_squared_error_threshold {
                solution[0].r_hat = determinant_9x1(&e) * e;
                solution[0].t = self.P*solution[0].r_hat;
                solution[0].num_iterations = 0;

                self.handle_solution( &mut solution[0], &mut min_sq_error );
            } else {
                (self.nearest_rotation_matrix)( &e, &mut solution[0].r );
                solution[0] = self.run_sqp( &solution[0].r );
                solution[0].t = self.P*solution[0].r_hat;
                self.handle_solution( &mut solution[0] , &mut min_sq_error );

                (self.nearest_rotation_matrix)( &-e, &mut solution[1].r );
                solution[1] = self.run_sqp( &solution[1].r );
                solution[1].t = self.P*solution[1].r_hat;
                self.handle_solution( &mut solution[1] , &mut min_sq_error );
            }
        }

        let mut index;
        let mut c = 1;
        while {index = 9 - num_eigen_points - c; index} > 0 && min_sq_error > 3. * self.s[index] {
            let e = self.U.column(index).into_owned();
            let mut solution = [SQPSolution::default(); 2];

            (self.nearest_rotation_matrix)( &e, &mut solution[0].r);
            solution[0] = self.run_sqp( &solution[0].r );
            solution[0].t = self.P*solution[0].r_hat;
            self.handle_solution( &mut solution[0], &mut min_sq_error );

            (self.nearest_rotation_matrix)( &-e, &mut solution[1].r);
            solution[1] = self.run_sqp( &solution[1].r );
            solution[1].t = self.P*solution[1].r_hat;
            self.handle_solution( &mut solution[1], &mut min_sq_error );

            c += 1;
        }

        true
    }

    fn run_sqp(&mut self, r0: &SMatrix<f32, 9, 1>) -> SQPSolution {
        let mut r = *r0;

        let mut delta_squared_norm = f32::MAX;
        let mut delta = SMatrix::<f32, 9, 1>::zeros();
        let mut step = 0;

        while delta_squared_norm > self.parameters.sqp_squared_tolerance && step < self.parameters.sqp_max_iteration {
            step += 1;
            self.solve_sqpsystem(&r, &mut delta);
            r += delta;
            delta_squared_norm = delta.norm_squared();
        }

        let mut solution = SQPSolution::default();
        solution.num_iterations = step;
        solution.r = r;
        // clear the estimate and/or flip the matrix sign if necessary
        let mut det_r = determinant_9x1(&solution.r);
        if det_r < 0.0 {
            solution.r = -r;
            det_r = -det_r;
        }
        if det_r > self.parameters.sqp_det_threshold {
            (self.nearest_rotation_matrix)( &solution.r, &mut solution.r_hat );
        } else {
            solution.r_hat = solution.r;
        }

        return solution;
    }

    #[allow(non_snake_case)]
    fn solve_sqpsystem(&mut self, r: &SMatrix<f32, 9, 1>, delta: &mut SMatrix<f32, 9, 1>) {
        let sqnorm_r1 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        let sqnorm_r2 = r[3]*r[3] + r[4]*r[4] + r[5]*r[5];
        let sqnorm_r3 = r[6]*r[6] + r[7]*r[7] + r[8]*r[8];
        let dot_r1r2 = r[0]*r[3] + r[1]*r[4] + r[2]*r[5];
        let dot_r1r3 = r[0]*r[6] + r[1]*r[7] + r[2]*r[8];
        let dot_r2r3 = r[3]*r[6] + r[4]*r[7] + r[5]*r[8];

        // Obtain 6D normal (H) and 3D null space of the constraint Jacobian-J at the estimate (r)
        // NOTE: This is done via Gram-Schmidt orthogonalization
        let mut N = SMatrix::<f32, 9, 3>::zeros();  // Null space of J
        let mut H = SMatrix::<f32, 9, 6>::zeros();  // Row space of J
        let mut JH = SMatrix::<f32, 6, 6>::zeros(); // The lower triangular matrix J*Q

        row_and_null_space(r, &mut H, &mut N, &mut JH, None);

        // Great, now if delta = H*x + N*y, we first compute x by solving:
        // 
        //              (J*H)*x = g
        //
        // where g is the constraint vector g = [   1 - norm(r1)^2;
        // 					     	   1 - norm(r2)^2;
        //					     	   1 - norm(r3)^2;
        //					           -r1'*r2; 
        //						   -r2'*r3; 
        //						   -r1'*r3 ];
        // Eigen::Matrix<double, 6, 1> g; 
        // g[0] = 1 - sqnorm_r1; g[1] = 1 - sqnorm_r2; g[2] = 1 - sqnorm_r3; g[3] = -dot_r1r2; g[4] = -dot_r2r3; g[5] = -dot_r1r3;
        let g = SMatrix::<f32, 6, 1>::new(1. - sqnorm_r1, 1. - sqnorm_r2, 1. - sqnorm_r3, -dot_r1r2, -dot_r2r3, -dot_r1r3);

        let mut x = SMatrix::<f32, 6, 1>::zeros();
        x[0] = g[0] / JH[(0, 0)];
        x[1] = g[1] / JH[(1, 1)];
        x[2] = g[2] / JH[(2, 2)];
        x[3] = ( g[3] - JH[(3, 0)]*x[0] - JH[(3, 1)]*x[1] ) / JH[(3, 3)];
        x[4] = ( g[4] - JH[(4, 1)]*x[1] - JH[(4, 2)]*x[2] - JH[(4, 3)]*x[3] ) / JH[(4, 4)];
        x[5] = ( g[5] - JH[(5, 0)]*x[0] - JH[(5, 2)]*x[2] - JH[(5, 3)]*x[3] - JH[(5, 4)]*x[4] ) / JH[(5, 5)];

        // Now obtain the component of delta in the row space of E as delta_h = H*x and assign straight into delta
        *delta = H * x;

        // Then, solve for y from W*y = ksi, where matrix W and vector ksi are :
        //
        // W = N'*Omega*N and ksi = -N'*Omega*( r + delta_h );
        let nt_omega = N.transpose() * self.omega ;
        let W = nt_omega * N;
        let rhs = -(nt_omega * ( *delta + r ));

        // solve with LDLt and if it fails, use inverse
        let mut y = SMatrix::zeros();
        let mut Winv = SMatrix::zeros();
        if axb_solve_ldlt_3x3(&W, &rhs, &mut y) != 0 {
            invert_symmetric_3x3(W, &mut Winv);
            y = Winv * rhs;
        }

        // Finally, accumulate delta with component in tangent space (delta_n)
        *delta += N*y;
    }
}


/// Compute the 3D null space (N) and 6D normal space (H) of the constraint Jacobian at a 9D vector r 
/// (r is not necessarily a rotation but it must represent a rank-3 matrix)
/// NOTE: K is lower-triangular, so upper triangle may contain trash (is not filled by the function)...
#[allow(non_snake_case)]
fn row_and_null_space(
    r: &SMatrix<f32, 9, 1>, 
    H: &mut SMatrix<f32, 9, 6>, // Row space 
    N: &mut SMatrix<f32, 9, 3>, // Null space
    K: &mut SMatrix<f32, 6, 6>,  // J*Q (J - Jacobian of constraints)
    norm_threshold: Option<f32>, // Used to discard columns of Pn when finding null space
) { // threshold for column vector norm (of Pn)
    let norm_threshold = norm_threshold.unwrap_or(0.1);
    // Applying Gram-Schmidt orthogonalization on the Jacobian. 
    // The steps are fixed here to take advantage of the sparse form of the matrix
    //
    *H = SMatrix::zeros();

    // 1. q1
    let norm_r1 = f32::sqrt( r[0]*r[0] + r[1]*r[1] + r[2]*r[2] );
    let inv_norm_r1 = if norm_r1 > 1e-5 { 1.0 / norm_r1 } else { 0.0 };
    H[(0, 0)] = r[0] * inv_norm_r1; H[(1, 0)] = r[1] * inv_norm_r1; H[(2, 0)] = r[2] * inv_norm_r1;
    K[(0, 0)] = 2.*norm_r1;

    // 2. q2 
    let norm_r2 = f32::sqrt( r[3]*r[3] + r[4]*r[4] + r[5]*r[5] );
    let inv_norm_r2 = 1.0 / norm_r2;
    H[(3, 1)] = r[3]*inv_norm_r2; H[(4, 1)] = r[4]*inv_norm_r2; H[(5, 1)] = r[5]*inv_norm_r2;
    K[(1, 0)] = 0.; K[(1, 1)] = 2.*norm_r2;

    // 3. q3 = (r3'*q2)*q2 - (r3'*q1)*q1 ; q3 = q3/norm(q3)
    let norm_r3 = f32::sqrt( r[6]*r[6] + r[7]*r[7] + r[8]*r[8] );
    let inv_norm_r3 = 1.0 / norm_r3;
    H[(6, 2)] = r[6]*inv_norm_r3; H[(7, 2)] = r[7]*inv_norm_r3; H[(8, 2)] = r[8]*inv_norm_r3;
    K[(2, 0)] = 0.; K[(2, 1)] = 0.; K[(2, 2)] = 2.*norm_r3;

    // 4. q4
    let dot_j4q1 = r[3]*H[(0, 0)] + r[4]*H[(1, 0)] + r[5]*H[(2, 0)];
    let dot_j4q2 = r[0]*H[(3, 1)] + r[1]*H[(4, 1)] + r[2]*H[(5, 1)];

    H[(0, 3)] = r[3] - dot_j4q1*H[(0, 0)]; H[(1, 3)] = r[4] - dot_j4q1*H[(1, 0)]; H[(2, 3)] = r[5] - dot_j4q1*H[(2, 0)];
    H[(3, 3)] = r[0] - dot_j4q2*H[(3, 1)]; H[(4, 3)] = r[1] - dot_j4q2*H[(4, 1)]; H[(5, 3)] = r[2] - dot_j4q2*H[(5, 1)];
    let inv_norm_j4 = 1.0 / f32::sqrt( H[(0, 3)]*H[(0, 3)] + H[(1, 3)]*H[(1, 3)] + H[(2, 3)]*H[(2, 3)] + 
        H[(3, 3)]*H[(3, 3)] + H[(4, 3)]*H[(4, 3)] + H[(5, 3)]*H[(5, 3)] );

    H[(0, 3)] *= inv_norm_j4; H[(1, 3)] *= inv_norm_j4; H[(2, 3)] *= inv_norm_j4;
    H[(3, 3)] *= inv_norm_j4; H[(4, 3)] *= inv_norm_j4; H[(5, 3)] *= inv_norm_j4;

    K[(3, 0)] = r[3]*H[(0, 0)] + r[4]*H[(1, 0)] + r[5]*H[(2, 0)]; K[(3, 1)] = r[0]*H[(3, 1)] + r[1]*H[(4, 1)] + r[2]*H[(5, 1)]; 
    K[(3, 2)] = 0.; K[(3, 3)] = r[3]*H[(0, 3)] + r[4]*H[(1, 3)] + r[5]*H[(2, 3)]  +  r[0]*H[(3, 3)] + r[1]*H[(4, 3)] + r[2]*H[(5, 3)];

    // 5. q5
    let dot_j5q2 = r[6]*H[(3, 1)] + r[7]*H[(4, 1)] + r[8]*H[(5, 1)];
    let dot_j5q3 = r[3]*H[(6, 2)] + r[4]*H[(7, 2)] + r[5]*H[(8, 2)];
    let dot_j5q4 = r[6]*H[(3, 3)] + r[7]*H[(4, 3)] + r[8]*H[(5, 3)];

    H[(0, 4)] = -dot_j5q4*H[(0, 3)];                            H[(1, 4)] = -dot_j5q4*H[(1, 3)];                            H[(2, 4)] = -dot_j5q4*H[(2, 3)];
    H[(3, 4)] = r[6] - dot_j5q2*H[(3, 1)] - dot_j5q4*H[(3, 3)]; H[(4, 4)] = r[7] - dot_j5q2*H[(4, 1)] - dot_j5q4*H[(4, 3)]; H[(5, 4)] = r[8] - dot_j5q2*H[(5, 1)] - dot_j5q4*H[(5, 3)];
    H[(6, 4)] = r[3] - dot_j5q3*H[(6, 2)]; H[(7, 4)] = r[4] - dot_j5q3*H[(7, 2)]; H[(8, 4)] = r[5] - dot_j5q3*H[(8, 2)];

    let norm = H.column(4).norm();
    *&mut H.column_mut(4) /= norm;

    K[(4, 0)] = 0.; K[(4, 1)] = r[6]*H[(3, 1)] + r[7]*H[(4, 1)] + r[8]*H[(5, 1)]; K[(4, 2)] = r[3]*H[(6, 2)] + r[4]*H[(7, 2)] + r[5]*H[(8, 2)];
    K[(4, 3)] = r[6]*H[(3, 3)] + r[7]*H[(4, 3)] + r[8]*H[(5, 3)]; 
    K[(4, 4)] = r[6]*H[(3, 4)] + r[7]*H[(4, 4)] + r[8]*H[(5, 4)]  +  r[3]*H[(6, 4)] + r[4]*H[(7, 4)] + r[5]*H[(8, 4)]; 

    // 4. q6
    let dot_j6q1 = r[6]*H[(0, 0)] + r[7]*H[(1, 0)] + r[8]*H[(2, 0)];
    let dot_j6q3 = r[0]*H[(6, 2)] + r[1]*H[(7, 2)] + r[2]*H[(8, 2)]; 
    let dot_j6q4 = r[6]*H[(0, 3)] + r[7]*H[(1, 3)] + r[8]*H[(2, 3)]; 
    let dot_j6q5 = r[0]*H[(6, 4)] + r[1]*H[(7, 4)] + r[2]*H[(8, 4)]  +  r[6]*H[(0, 4)] + r[7]*H[(1, 4)] + r[8]*H[(2, 4)];

    H[(0, 5)] = r[6] - dot_j6q1*H[(0, 0)] - dot_j6q4*H[(0, 3)] - dot_j6q5*H[(0, 4)]; 
    H[(1, 5)] = r[7] - dot_j6q1*H[(1, 0)] - dot_j6q4*H[(1, 3)] - dot_j6q5*H[(1, 4)]; 
    H[(2, 5)] = r[8] - dot_j6q1*H[(2, 0)] - dot_j6q4*H[(2, 3)] - dot_j6q5*H[(2, 4)];

    H[(3, 5)] = -dot_j6q5*H[(3, 4)] - dot_j6q4*H[(3, 3)]; 
    H[(4, 5)] = -dot_j6q5*H[(4, 4)] - dot_j6q4*H[(4, 3)]; 
    H[(5, 5)] = -dot_j6q5*H[(5, 4)] - dot_j6q4*H[(5, 3)];

    H[(6, 5)] = r[0] - dot_j6q3*H[(6, 2)] - dot_j6q5*H[(6, 4)]; 
    H[(7, 5)] = r[1] - dot_j6q3*H[(7, 2)] - dot_j6q5*H[(7, 4)]; 
    H[(8, 5)] = r[2] - dot_j6q3*H[(8, 2)] - dot_j6q5*H[(8, 4)];

    let norm = H.column(5).norm();
    *&mut H.column_mut(5) *= 1.0 / norm;

    K[(5, 0)] = r[6]*H[(0, 0)] + r[7]*H[(1, 0)] + r[8]*H[(2, 0)]; K[(5, 1)] = 0.; K[(5, 2)] = r[0]*H[(6, 2)] + r[1]*H[(7, 2)] + r[2]*H[(8, 2)];
    K[(5, 3)] = r[6]*H[(0, 3)] + r[7]*H[(1, 3)] + r[8]*H[(2, 3)]; K[(5, 4)] = r[6]*H[(0, 4)] + r[7]*H[(1, 4)] + r[8]*H[(2, 4)] +   r[0]*H[(6, 4)] + r[1]*H[(7, 4)] + r[2]*H[(8, 4)];
    K[(5, 5)] = r[6]*H[(0, 5)] + r[7]*H[(1, 5)] + r[8]*H[(2, 5)] + r[0]*H[(6, 5)] + r[1]*H[(7, 5)] + r[2]*H[(8, 5)];

    // Great! Now H is an orthogonalized, sparse basis of the Jacobian row space and K is filled.
    //
    // Now get a projector onto the null space of H:
    let Pn = SMatrix::<f32, 9, 9>::identity() - ( *H*H.transpose() ); 

    // Now we need to pick 3 columns of P with non-zero norm (> 0.3) and some angle between them (> 0.3).
    //
    // Find the 3 columns of Pn with largest norms
    let mut index1 = usize::MAX;
    let mut index2 = usize::MAX;
    let mut index3 = usize::MAX;
    let mut max_norm1 = f32::MIN;
    let mut min_dot12 = f32::MAX;
    let mut min_dot1323 = f32::MAX;


    let mut col_norms = [0.0; 9];
    for i in 0..9 {
        col_norms[i] = Pn.column(i).norm();
        if col_norms[i] >= norm_threshold {
            if max_norm1 < col_norms[i] {
                max_norm1 = col_norms[i];
                index1 = i;
            }
        }
    }
    let v1 = Pn.column(index1);
    N.set_column(0, &(v1 * ( 1.0 / max_norm1 )));
    col_norms[index1] = -1.0; // mark to avoid use in subsequent loops

    for i in 0..9 {
        //if i == index1 { continue; }
        if col_norms[i] >= norm_threshold {
            let cos_v1_x_col = f32::abs(Pn.column(i).dot(&v1) / col_norms[i]);

            if cos_v1_x_col <= min_dot12 {
                index2 = i;
                min_dot12 = cos_v1_x_col;
            }
        }
    }
    let v2 = Pn.column(index2);
    N.set_column(1, &(v2 - v2.dot( &N.column(0) ) * N.column(0)));
    let norm = N.column(1).norm();
    *&mut N.column_mut(1) *= 1.0 / norm;
    col_norms[index2] = -1.0; // mark to avoid use in loop below

    for i in 0..9 {
        //if i == index2 || i == index1 { continue; }
        if col_norms[i] >= norm_threshold {
            let inv_norm = 1.0 / col_norms[i];
            let cos_v1_x_col = f32::abs(Pn.column(i).dot(&v1) * inv_norm);
            let cos_v2_x_col = f32::abs(Pn.column(i).dot(&v2) * inv_norm);

            if cos_v1_x_col + cos_v2_x_col <= min_dot1323 {
                index3 = i;
                min_dot1323 = cos_v2_x_col + cos_v2_x_col;
            }
        }
    }

    // Now orthogonalize the remaining 2 vectors v2, v3 into N
    let v3 = Pn.column(index3);

    N.set_column(2, &(v3 - ( v3.dot( &N.column(1) ) * N.column(1) ) - ( v3.dot( &N.column(0) ) * N.column(0) )));
    let norm = N.column(2).norm();
    *&mut N.column_mut(2) *= 1.0 / norm;
}

// Solve A*x=b for 3x3 SPD A.
// The solution involves computing a lower triangular sqrt-free Cholesky factor
// A=L*D*L' (L has ones on its diagonal, D is diagonal).
//
// Only the lower triangular part of A is accessed.
//
// The function returns 0 if successful, non-zero otherwise
//
// see http://euler.nmt.edu/~brian/ldlt.html
//
#[allow(non_snake_case)]
fn axb_solve_ldlt_3x3(A: &SMatrix<f32, 3, 3>, b: &SMatrix<f32, 3, 1>, x: &mut SMatrix<f32, 3, 1>) -> i32 {
    let mut L = [0.0; 3 * 3];
    let mut v = [0.0; 2];

    // D is stored in L's diagonal, i.e. L[0], L[4], L[8]
    // its elements should be positive
    v[0] = A[(0, 0)];
    L[0] = A[(0, 0)];
    if v[0] < 1E-10 { return 1; }
    v[1] = 1.0 / v[0];
    L[3] = A[(1, 0)] * v[1];
    L[6] = A[(2, 0)] * v[1];
    // L[1] = L[2] = 0.0;

    v[0] = L[3] * L[0];
    v[1] = A[(1, 1)] - L[3] * v[0];
    L[4] = v[1];
    if v[1] < 1E-10 { return 2; }
    L[7] = (A[(2, 1)] - L[6] * v[0]) / v[1];
    // L[5] = 0.0;

    v[0] = L[6] * L[0];
    v[1] = L[7] * L[4];
    L[8] = A[(2, 2)] - L[6] * v[0] - L[7] * v[1];
    if L[8] < 1E-10 { return 3; }

    // Forward solve L*x = b
    x[0] = b[0];
    x[1] = b[1] - L[3] * x[0];
    x[2] = b[2] - L[6] * x[0] - L[7] * x[1];

    // Backward solve D*L'*x = y
    x[2] = x[2] / L[8];
    x[1] = x[1] / L[4] - L[7] * x[2];
    x[0] = x[0] / L[0] - L[3] * x[1] - L[6] * x[2];

    0
}
