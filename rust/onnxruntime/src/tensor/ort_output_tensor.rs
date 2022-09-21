//! Module containing tensor with memory owned by the ONNX Runtime

use std::{convert::TryFrom, fmt::Debug};

use ndarray::ArrayView;
use tracing::debug;

use onnxruntime_sys as sys;

use crate::{error::status_to_result, g_ort, OrtError, Result, TypeToTensorElementDataType};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
#[derive(Debug)]
pub struct OrtOutputTensor {
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct OrtOwnedTensorExtractor {
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    pub(crate) shape: Vec<usize>,
}

impl OrtOwnedTensorExtractor {
    pub(crate) fn new(shape: Vec<usize>) -> OrtOwnedTensorExtractor {
        OrtOwnedTensorExtractor {
            tensor_ptr: std::ptr::null_mut(),
            shape,
        }
    }

    pub(crate) fn extract(self) -> Result<OrtOutputTensor> {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.

        assert_ne!(self.tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(self.tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        (is_tensor == 1)
            .then_some(())
            .ok_or(OrtError::IsTensorCheck)?;

        Ok(OrtOutputTensor {
            tensor_ptr: self.tensor_ptr,
            shape: self.shape,
        })
    }
}

impl Drop for OrtOutputTensor {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping OrtOwnedTensor.");
        unsafe { g_ort().ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = std::ptr::null_mut();
    }
}

/// An Ouput tensor with the ptr and the item that will copy from the ptr.
#[derive(Debug)]
pub struct WithOutputTensor<T> {
    #[allow(dead_code)]
    pub(crate) tensor: OrtOutputTensor,
    item: T,
}

impl<T> std::ops::Deref for WithOutputTensor<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<'a, T> TryFrom<OrtOutputTensor> for WithOutputTensor<ArrayView<'a, T, ndarray::IxDyn>>
where
    T: TypeToTensorElementDataType,
{
    type Error = OrtError;

    fn try_from(value: OrtOutputTensor) -> Result<Self> {
        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr.cast::<*mut std::ffi::c_void>();
        let status = unsafe {
            g_ort().GetTensorMutableData.unwrap()(value.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let array_view =
            unsafe { ArrayView::from_shape_ptr(ndarray::IxDyn(&value.shape), output_array_ptr) };

        Ok(WithOutputTensor {
            tensor: value,
            item: array_view,
        })
    }
}

impl<T> TryFrom<OrtOutputTensor> for WithOutputTensor<&[T]>
where
    T: TypeToTensorElementDataType,
{
    type Error = OrtError;

    fn try_from(value: OrtOutputTensor) -> Result<Self> {
        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr.cast::<*mut std::ffi::c_void>();
        let status = unsafe {
            g_ort().GetTensorMutableData.unwrap()(value.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let length = value.shape.iter().fold(1, |acc, el| acc * el);

        let v = unsafe { std::slice::from_raw_parts(output_array_ptr, length) };

        Ok(WithOutputTensor {
            tensor: value,
            item: v,
        })
    }
}
