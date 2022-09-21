//! convert module has the trait for conversion of Inputs ConstructTensor.

use crate::tensor::ort_input_tensor::OrtInputTensor;
use crate::{memory::MemoryInfo, OrtError};
use onnxruntime_sys::OrtAllocator;
use std::fmt::Debug;

/// The Input type for Rust onnxruntime Session::run
/// Many types can construct a
pub trait ConstructTensor {
    /// Constuct an OrtTensor Input using the `MemoryInfo` and a raw pointer to the `OrtAllocator`.
    fn construct(
        self,
        memory_info: &MemoryInfo,
        allocator: *mut OrtAllocator,
    ) -> Result<OrtInputTensor<Self>, OrtError>
    where
        Self: Sized + Debug;
}
