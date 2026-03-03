// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature};
use datafusion_functions_nested::array_has::ArrayHas;
use std::any::Any;
use std::sync::Arc;

/// Spark-compatible `array_contains` function.
///
/// Delegates to DataFusion's [`ArrayHas`] for the core comparison, then applies
/// Spark's three-valued NULL semantics: when the needle is not found and the
/// array contains NULL elements, the result is `NULL` instead of `false`.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_contains>
#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct SparkArrayContains {
    inner: ArrayHas,
}

impl SparkArrayContains {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ScalarUDFImpl for SparkArrayContains {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_contains"
    }

    fn signature(&self) -> &Signature {
        self.inner.signature()
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Save a reference to the haystack before passing args to ArrayHas
        let haystack = args.args[0].clone();

        // Delegate to ArrayHas for the core comparison
        let result = self.inner.invoke_with_args(args)?;

        // Apply Spark's three-valued NULL semantics:
        // Where ArrayHas returned false, if the haystack row had NULL elements,
        // the result should be NULL instead.
        apply_spark_null_semantics(result, &haystack)
    }
}

/// Post-process the result from `ArrayHas` to apply Spark NULL semantics.
///
/// For each row where the result is `false`, check if the corresponding
/// haystack array row contains any NULL elements. If so, change the result
/// to `NULL` (three-valued logic: the comparison is "unknown").
fn apply_spark_null_semantics(
    result: ColumnarValue,
    haystack: &ColumnarValue,
) -> Result<ColumnarValue> {
    match result {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))) => {
            // Scalar false: check if the single haystack list has any nulls
            if haystack_has_nulls(haystack)? {
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))
            } else {
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))))
            }
        }
        ColumnarValue::Array(ref arr) => {
            let bool_arr = arr.as_boolean();
            // Only do work if there are any false values to potentially flip
            if bool_arr.false_count() == 0 {
                return Ok(result);
            }
            let num_rows = bool_arr.len();
            let haystack_arr = haystack.to_array(num_rows)?;
            // If the haystack's inner values have no nulls at all,
            // no false result can flip to NULL — skip post-processing.
            if haystack_values_null_count(&haystack_arr) == 0 {
                return Ok(result);
            }
            fixup_array_result(bool_arr, &haystack_arr)
        }
        // true or null: pass through unchanged
        other => Ok(other),
    }
}

/// Check if a haystack contains any NULL elements.
fn haystack_has_nulls(haystack: &ColumnarValue) -> Result<bool> {
    match haystack {
        ColumnarValue::Scalar(scalar) => {
            let arr = scalar.to_array_of_size(1)?;
            Ok(haystack_values_null_count(&arr) > 0)
        }
        ColumnarValue::Array(arr) => Ok(haystack_values_null_count(arr) > 0),
    }
}

/// O(1) check: total null count in the haystack's flat inner values array.
fn haystack_values_null_count(arr: &ArrayRef) -> usize {
    match arr.data_type() {
        DataType::List(_) => arr.as_list::<i32>().values().null_count(),
        DataType::LargeList(_) => arr.as_list::<i64>().values().null_count(),
        DataType::FixedSizeList(_, _) => arr.as_fixed_size_list().values().null_count(),
        _ => 0,
    }
}

/// For array results, flip `false` → `NULL` where the haystack row has NULL elements.
///
/// Uses bitmap operations instead of per-row branching:
/// 1. Compute a `row_has_nulls` bitmap from the haystack's offsets + values null buffer
/// 2. Combine: `new_validity = old_validity AND (values OR NOT row_has_nulls)`
fn fixup_array_result(
    bool_arr: &BooleanArray,
    haystack_arr: &ArrayRef,
) -> Result<ColumnarValue> {
    let len = bool_arr.len();

    let row_has_nulls = compute_row_has_nulls(haystack_arr, len);

    let values = bool_arr.values();
    let old_validity = match bool_arr.nulls() {
        Some(nulls) => nulls.inner().clone(),
        None => BooleanBuffer::new_set(len),
    };
    let new_validity = &old_validity & &(values | &!&row_has_nulls);

    Ok(ColumnarValue::Array(Arc::new(BooleanArray::new(
        values.clone(),
        Some(NullBuffer::new(new_validity)),
    ))))
}

/// Compute a bitmap where bit `i` is set if row `i`'s sub-array contains any NULL elements.
fn compute_row_has_nulls(haystack_arr: &ArrayRef, len: usize) -> BooleanBuffer {
    match haystack_arr.data_type() {
        DataType::List(_) => {
            compute_row_has_nulls_for_list(haystack_arr.as_list::<i32>(), len)
        }
        DataType::LargeList(_) => {
            compute_row_has_nulls_for_list(haystack_arr.as_list::<i64>(), len)
        }
        DataType::FixedSizeList(_, _) => {
            let list = haystack_arr.as_fixed_size_list();
            let value_length = list.value_length() as usize;
            match list.values().nulls() {
                Some(nulls) => {
                    let null_bits = nulls.inner();
                    let mut builder = BooleanBufferBuilder::new(len);
                    for i in 0..len {
                        if list.is_null(i) {
                            builder.append(false);
                        } else {
                            let start = i * value_length;
                            let valid =
                                null_bits.slice(start, value_length).count_set_bits();
                            builder.append(valid < value_length);
                        }
                    }
                    builder.finish()
                }
                None => BooleanBuffer::new_unset(len),
            }
        }
        _ => BooleanBuffer::new_unset(len),
    }
}

/// Compute row_has_nulls for GenericListArray (List and LargeList).
fn compute_row_has_nulls_for_list<O: arrow::array::OffsetSizeTrait>(
    list: &arrow::array::GenericListArray<O>,
    len: usize,
) -> BooleanBuffer {
    match list.values().nulls() {
        Some(nulls) => {
            let null_bits = nulls.inner();
            let offsets: Vec<usize> =
                list.offsets().iter().map(|o| o.as_usize()).collect();
            let mut builder = BooleanBufferBuilder::new(len);
            for i in 0..len {
                if list.is_null(i) {
                    builder.append(false);
                } else {
                    let start = offsets[i];
                    let row_len = offsets[i + 1] - start;
                    let valid = null_bits.slice(start, row_len).count_set_bits();
                    builder.append(valid < row_len);
                }
            }
            builder.finish()
        }
        None => BooleanBuffer::new_unset(len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(haystack: ColumnarValue, needle: ColumnarValue) -> Result<ColumnarValue> {
        let haystack_field = Arc::new(Field::new_list(
            "haystack",
            Field::new("item", DataType::Int32, true),
            true,
        ));
        let needle_field = Arc::new(Field::new("needle", DataType::Int32, true));
        let return_field = Arc::new(Field::new("return", DataType::Boolean, true));

        SparkArrayContains::default().invoke_with_args(ScalarFunctionArgs {
            args: vec![haystack, needle],
            arg_fields: vec![haystack_field, needle_field],
            number_rows: 1,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn unwrap_boolean(result: ColumnarValue) -> Option<bool> {
        match result {
            ColumnarValue::Scalar(ScalarValue::Boolean(v)) => v,
            other => panic!("Expected Boolean scalar, got: {other:?}"),
        }
    }

    fn make_list_scalar(values: &[Option<i32>]) -> ColumnarValue {
        let arr = Int32Array::from(values.to_vec());
        let list = ListArray::new(
            Arc::new(Field::new("item", DataType::Int32, true)),
            OffsetBuffer::from_lengths([values.len()]),
            Arc::new(arr),
            None,
        );
        ColumnarValue::Scalar(ScalarValue::List(Arc::new(list)))
    }

    #[test]
    fn test_found() {
        let haystack = make_list_scalar(&[Some(1), Some(2), Some(3)]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(2)));
        assert_eq!(
            unwrap_boolean(invoke(haystack, needle).unwrap()),
            Some(true)
        );
    }

    #[test]
    fn test_not_found() {
        let haystack = make_list_scalar(&[Some(1), Some(2), Some(3)]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(4)));
        assert_eq!(
            unwrap_boolean(invoke(haystack, needle).unwrap()),
            Some(false)
        );
    }

    #[test]
    fn test_not_found_with_nulls_returns_null() {
        // Key Spark behavior: array has NULLs and element not found → NULL
        let haystack = make_list_scalar(&[Some(1), None, Some(3)]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(2)));
        assert_eq!(unwrap_boolean(invoke(haystack, needle).unwrap()), None);
    }

    #[test]
    fn test_found_with_nulls_returns_true() {
        // Even with NULLs, if element is found → true
        let haystack = make_list_scalar(&[Some(1), None, Some(3)]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));
        assert_eq!(
            unwrap_boolean(invoke(haystack, needle).unwrap()),
            Some(true)
        );
    }

    #[test]
    fn test_null_needle() {
        let haystack = make_list_scalar(&[Some(1), Some(2), Some(3)]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(None));
        assert_eq!(unwrap_boolean(invoke(haystack, needle).unwrap()), None);
    }

    #[test]
    fn test_empty_array() {
        let haystack = make_list_scalar(&[]);
        let needle = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));
        assert_eq!(
            unwrap_boolean(invoke(haystack, needle).unwrap()),
            Some(false)
        );
    }
}
