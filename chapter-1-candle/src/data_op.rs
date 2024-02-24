//! # 数据操作
//!
//!    - [x] 1.1 运算符
//!    - [x] 1.2 广播机制
//!    - [x] 1.3 索引和切片
//!
//! 数据是模型的基础。在这一章中，我们将学习如何使用 `candle_core` 库进行数据操作。
//!

/// # 入门
///
/// 本节将介绍如何使用 `candle_core` 库进行数据操作。
///
/// [官方提供的pytorch用户速查表](https://huggingface.github.io/candle/guide/cheatsheet.html)
///
pub struct DataOp;

impl DataOp {
    /// # 运算符
    ///
    /// ## 二元运算符
    ///
    /// ```
    /// # use candle_core::{Device, Result, Tensor};
    /// # fn main() -> Result<()> {
    /// let x = Tensor::new(vec![1f32, 2f32, 4f32, 8f32], &Device::Cpu)?.reshape(&[2, 2])?;
    /// let y = Tensor::new(vec![2f32, 2f32, 2f32, 2f32], &Device::Cpu)?.reshape(&[2, 2])?;
    /// let add_result = &x + &y; // [[3, 4], [6, 10]]
    /// # assert_eq!(add_result?.to_vec2::<f32>()?, vec![vec![3f32, 4f32], vec![6f32, 10f32]]);
    /// let sub_result = &x - &y; // [[-1, 0], [2, 6]]
    /// # assert_eq!(sub_result?.to_vec2::<f32>()?, vec![vec![-1f32, 0f32], vec![2f32, 6f32]]);
    /// let mul_result = &x * &y; // [[2, 4], [8, 16]]
    /// # assert_eq!(mul_result?.to_vec2::<f32>()?, vec![vec![2f32, 4f32], vec![8f32, 16f32]]);
    /// let div_result = &x / &y; // [[0.5, 1], [2, 4]]
    /// # assert_eq!(div_result?.to_vec2::<f32>()?, vec![vec![0.5f32, 1f32], vec![2f32, 4f32]]);
    /// let pow_result = &x.pow(&y)?; // [[1, 4], [16, 64]]
    /// # assert_eq!(pow_result.to_vec2::<f32>()?, vec![vec![1f32, 4f32], vec![16f32, 64f32]]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    pub fn op() {
        unimplemented!()
    }

    /// # 广播机制
    ///
    /// ## 二元运算符
    ///
    /// ```
    /// # use candle_core::{Device, Result, Tensor};
    /// # fn main() -> Result<()> {
    /// let x = Tensor::new(vec![1f32, 2f32, 4f32, 8f32], &Device::Cpu)?.reshape(&[4, 1])?; // [[1], [2], [4], [8]]
    /// let y = Tensor::new(vec![1f32, 2f32], &Device::Cpu)?.reshape(&[1, 2])?; // [[1, 2]]
    /// let add_result = &x.broadcast_add(&y)?; // [[2, 3], [3, 4], [5, 6], [9, 10]]
    /// # assert_eq!(add_result.to_vec2::<f32>()?, vec![vec![2f32, 3f32], vec![3f32, 4f32], vec![5f32, 6f32], vec![9f32, 10f32]]);
    /// let sub_result = &x.broadcast_sub(&y)?; // [[0, -1], [1, 0], [3, 2], [7, 6]]
    /// # assert_eq!(sub_result.to_vec2::<f32>()?, vec![vec![0f32, -1f32], vec![1f32, 0f32], vec![3f32, 2f32], vec![7f32, 6f32]]);
    /// let mul_result = &x.broadcast_mul(&y)?; // [[1, 2], [2, 4], [4, 8], [8, 16]]
    /// # assert_eq!(mul_result.to_vec2::<f32>()?, vec![vec![1f32, 2f32], vec![2f32, 4f32], vec![4f32, 8f32], vec![8f32, 16f32]]);
    /// let div_result = &x.broadcast_div(&y)?; // [[1, 0.5], [2, 1], [4, 2], [8, 4]]
    /// # assert_eq!(div_result.to_vec2::<f32>()?, vec![vec![1f32, 0.5], vec![2f32, 1f32], vec![4f32, 2f32], vec![8f32, 4f32]]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn broadcast() {
        unimplemented!()
    }

    /// # 索引和切片
    /// 
    /// ```
    /// # use candle_core::{Device, Result, Tensor};
    /// use candle_core::IndexOp; // 导入索引操作
    /// # fn main() -> Result<()> {
    /// let x = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &Device::Cpu)?.reshape(&[2, 2])?;
    /// let result = x.i(1)?; // [3, 4]
    /// # assert_eq!(result.to_vec1::<f32>()?, vec![3f32, 4f32]);
    /// let result = x.i(0..1)?; // [[1, 2]]
    /// # assert_eq!(result.to_vec2::<f32>()?, vec![vec![1f32, 2f32]]);
    /// # Ok(())
    /// # }
    /// ```
    /// or
    /// 
    /// ```
    /// # use candle_core::{Device, Result, Tensor};
    /// use candle_core::IndexOp; // 导入索引操作
    /// # fn main() -> Result<()> {
    /// let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?.reshape(&[4, 4, 1])?; // [[[0], [1], [2], [3]], [[4], [5], [6], [7]], [[8], [9], [10], [11]], [[12], [13], [14], [15]]]
    /// 
    /// let x_1_result = x.i(1)?; // [[4], [5], [6], [7]]
    /// # assert_eq!(x_1_result.to_vec2::<f32>()?, vec![vec![4f32], vec![5f32], vec![6f32], vec![7f32]]);
    /// 
    /// let x_1_2_result = x.i((1, 2))?; // [6]
    /// # assert_eq!(x_1_2_result.to_vec1::<f32>()?, vec![6f32]);
    /// 
    /// let x_1_2_0_result = x.i((1, 2, 0))?; // 6
    /// # assert_eq!(x_1_2_0_result.to_scalar::<f32>()?, 6f32);
    /// 
    /// let x_1_y_all_result = x.i((0..4, ..1, 0))?; // [[0], [4], [8], [12]]
    /// # assert_eq!(x_1_y_all_result.to_vec2::<f32>()?, vec![vec![0f32], vec![4f32], vec![8f32], vec![12f32]]);
    /// # Ok(())
    /// # }
    /// ```
    /// 
    pub fn index_slice() {
        unimplemented!()
    }
}


#[cfg(test)]
mod tests {
    // use candle_core::D;

    use candle_core::{Device, Tensor, Result, IndexOp};

    // use super::*;

    #[test]
    fn add_works() -> Result<()> {
        // let a = Tensor::arange(0f32, 16f32, &Device::Cpu)?;
        // println!("a: {:?}", a.to_vec0::<f32>()?);

        let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?.reshape(&[4, 4, 1])?;
        println!("x: {:?}", &x.to_vec3::<f32>()?);

        // 获取一列
        let result = x.i((1..4, 1.., 0))?;
        println!("result: {:?}", &result.to_vec2::<f32>()?);

        Ok(())
    }

}
