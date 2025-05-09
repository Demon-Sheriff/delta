// BSD 3-Clause License
//
// Copyright (c) 2025, BlackPortal ○
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataError {
    #[error("CSV error: {0}")]
    Csv(#[from] CsvError),
}

#[derive(Error, Debug)]
pub enum CsvError {
    #[error("Failed to open file: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("CSV file is empty")]
    EmptyFile,

    #[error("CSV must have at least one feature and one target column")]
    InsufficientColumns,

    #[error("Inconsistent column count: row {row} has {actual} columns, expected {expected}")]
    InconsistentColumns { row: usize, actual: usize, expected: usize },

    #[error("Invalid numeric value '{value}' at row {row}: {source}")]
    InvalidNumeric { value: String, row: usize, source: std::num::ParseFloatError },

    #[error("Invalid target value '{value}' at row {row}: {source}")]
    InvalidTarget { value: String, row: usize, source: std::num::ParseFloatError },

    #[error("Failed to shape data into array: {0}")]
    ArrayShape(#[from] ndarray::ShapeError),

    #[error("Failed to parse CSV: {0}")]
    CsvParse(#[from] csv::Error),
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Preprocessing error: {0}")]
    Preprocessing(#[from] PreprocessingError),

    #[error("Optimizer error: {0}")]
    Optimizer(#[from] OptimizerError),

    #[error("Loss error: {0}")]
    Loss(#[from] LossError),
}

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("Input array is empty")]
    EmptyInput,

    #[error("Input array has no features (zero columns)")]
    NoFeatures,

    #[error("Not fitted: call fit_transform first")]
    NotFitted,

    #[error("Dimension mismatch: expected {expected} features, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid numeric value (NaN or Inf) detected")]
    InvalidNumericValue,

    #[error("Invalid parameter provided (e.g., k out of range)")]
    InvalidParameter,

    #[error("Array operation failed: {0}")]
    ArrayOperation(#[from] ndarray::ShapeError),
}

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("Input array is empty")]
    EmptyInput,

    #[error("Zero samples provided")]
    ZeroSamples,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid numeric value (NaN or Inf) detected")]
    InvalidNumericValue,

    #[error("Numerical instability in gradient computation")]
    NumericalInstability,
}

#[derive(Error, Debug)]
pub enum LossError {
    #[error("Input arrays are empty")]
    EmptyInput,

    #[error("Dimension mismatch: expected {expected} elements, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid numeric value (NaN or Inf) detected")]
    InvalidNumericValue,

    #[error("Prediction values must be in [0.0, 1.0]")]
    InvalidPredictionRange,

    #[error("Actual values must be 0.0 or 1.0")]
    InvalidActualValue,
}
