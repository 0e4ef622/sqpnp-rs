use nalgebra::{SMatrix, Vector3};

pub enum OmegaNullspaceMethod {
    // Rrqr,
    Cprrqr,
    Svd,
}

pub enum NearestRotationMethod {
    Foam,
    Svd,
}

pub struct SolverParameters {
    pub rank_tolerance: f32,
    pub sqp_squared_tolerance: f32,
    pub sqp_det_threshold: f32,
    pub sqp_max_iteration: i32,
    pub omega_nullspace_method: OmegaNullspaceMethod,
    pub nearest_rotation_method: NearestRotationMethod,
    pub orthogonality_squared_error_threshold: f32,
    pub equal_vectors_squared_diff: f32,
    pub equal_squared_errors_diff: f32,
    pub point_variance_threshold: f32,
}

impl SolverParameters {
    pub const DEFAULT_RANK_TOLERANCE: f32 = 1e-7;
    pub const DEFAULT_SQP_SQUARED_TOLERANCE: f32 = 1e-10;
    pub const DEFAULT_SQP_DET_THRESHOLD: f32 = 1.001;
    pub const DEFAULT_SQP_MAX_ITERATION: i32 = 15;
    pub const DEFAULT_OMEGA_NULLSPACE_METHOD: OmegaNullspaceMethod = OmegaNullspaceMethod::Cprrqr; // Originally Rrqr which isn't implemented
    pub const DEFAULT_NEAREST_ROTATION_METHOD: NearestRotationMethod = NearestRotationMethod::Foam;
    pub const DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD: f32 = 1e-8;
    pub const DEFAULT_EQUAL_VECTORS_SQUARED_DIFF: f32 = 1e-10;
    pub const DEFAULT_EQUAL_SQUARED_ERRORS_DIFF: f32 = 1e-6;
    pub const DEFAULT_POINT_VARIANCE_THRESHOLD: f32 = 1e-5;
}

impl Default for SolverParameters {
    fn default() -> Self {
        Self {
            rank_tolerance: Self::DEFAULT_RANK_TOLERANCE,
            sqp_squared_tolerance: Self::DEFAULT_SQP_SQUARED_TOLERANCE,
            sqp_det_threshold: Self::DEFAULT_SQP_DET_THRESHOLD,
            sqp_max_iteration: Self::DEFAULT_SQP_MAX_ITERATION,
            omega_nullspace_method: Self::DEFAULT_OMEGA_NULLSPACE_METHOD,
            nearest_rotation_method: Self::DEFAULT_NEAREST_ROTATION_METHOD,
            orthogonality_squared_error_threshold: Self::DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD,
            equal_vectors_squared_diff: Self::DEFAULT_EQUAL_VECTORS_SQUARED_DIFF,
            equal_squared_errors_diff: Self::DEFAULT_EQUAL_SQUARED_ERRORS_DIFF,
            point_variance_threshold: Self::DEFAULT_POINT_VARIANCE_THRESHOLD,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SQPSolution {
    /// Actual matrix upon convergence
    pub r: SMatrix<f32, 9, 1>,
    /// "Clean" (nearest) rotation matrix
    pub r_hat: SMatrix<f32, 9, 1>,
    pub t: Vector3<f32>,
    pub num_iterations: i32,
    pub sq_error: f32,
}

impl SQPSolution {
    #[cfg(feature = "std")]
    pub fn print(&self) {
        use nalgebra::Const;
        println!("r_hat: {:.8}", self.r_hat.reshape_generic(Const::<3>, Const::<3>).transpose());
        println!("t: {:.8}", self.t);
        println!("squared error: {:.5e}", self.sq_error);
        println!("number of SQP iterations: {}", self.num_iterations);
    }
}
