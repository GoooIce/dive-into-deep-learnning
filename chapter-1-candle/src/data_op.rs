//! # 数据操作 
//! 
//!    - [x] 1.1 运算符
//!    - [x] 1.2 广播机制
//!    - [x] 1.3 索引和切片
//! 
//! 数据是模型的基础。在这一章中，我们将学习如何使用 `candle_core` 库进行数据操作。
//! 
use candle_core::{ Result, Tensor};

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
    /// let add_result = &x + &y; // [[3f32, 4f32], [6f32, 10f32]]
    /// # assert_eq!(add_result?.to_vec2::<f32>()?, vec![vec![6f32, 8f32], vec![10f32, 12f32]]);
    /// let sub_result = &x - &y; // [[-4f32, -4f32], [-4f32, -4f32]]
    /// # assert_eq!(sub_result?.to_vec2::<f32>()?, vec![vec![-4f32, -4f32], vec![-4f32, -4f32]]);
    /// let mul_result = &x * &y; // [[5f32, 12f32], [21f32, 32f32]]
    /// # assert_eq!(mul_result?.to_vec2::<f32>()?, vec![vec![5f32, 12f32], vec![21f32, 32f32]]);
    /// let div_result = &x / &y; // [[0.2f32, 0.33333334f32], [0.42857143f32, 0.5f32]]
    /// # assert_eq!(div_result?.to_vec2::<f32>()?, vec![vec![0.2f32, 0.33333334f32], vec![0.42857143f32, 0.5f32]]);
    /// let pow_result = &x.pow(&y)?; // [[1f32, 64f32], [2187f32, 65536f32]]
    /// # assert_eq!(pow_result.to_vec2::<f32>()?, vec![vec![1f32, 64f32], vec![2187f32, 65536f32]]);
    /// # Ok(())
    /// # }
    /// ```
    /// 
    pub fn add() { unimplemented!() }

    /// # 广播机制
    pub fn broadcast_add(left: Tensor, right: Tensor) -> Result<Tensor> {
        left.broadcast_add(&right)
    }

    /// # 索引和切片
    pub fn index_slice() -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, IndexOp};

    use super::*;

    #[test]
    fn add_broadcast_works() -> Result<()> {
        let d = Device::Cpu;

        // 使用加号运算符
        // 向量加法报错
        let left = Tensor::new(vec![1f32, 2f32, 3f32], &d)?;
        let right = Tensor::new(vec![1f32,2.,3.], &d)?;
        let result = &left.pow(&right)?;
        println!("{:?}", result);

        // assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn index_slice_works() -> Result<()> {
        let d = Device::Cpu;

        // 索引
        let x = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &d)?.reshape(&[2, 2])?;
        let result = x.i(1)?;

        assert_eq!(result.to_vec1::<f32>()?, vec![3f32, 4f32]);

        // 切片
        let x = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &d)?.reshape(&[2, 2])?;
        let result = x.i(0..1)?;

        assert_eq!(result.to_vec2::<f32>()?, vec![vec![1f32, 2f32]]);

        Ok(())
    }
}
