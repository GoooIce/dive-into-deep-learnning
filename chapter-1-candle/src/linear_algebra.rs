//! # 线性代数
//!
//! ## 张量
//!
//! 张量是一个多维数组，它是一个标量、向量、矩阵的泛化。
//!
//! ### 标量
//!
//! 标量是一个单一的数值，它没有方向和大小。
//! ```
//! let scalar = Tensor.new(1.0, &Device::Cpu)?.to_scalar::<f32>()?;
//! ```
//!
//! ### 向量
//!
//! 向量是一个一维数组，它有方向和大小。
//! ```
//! let vector = Tensor::new(vec![1f32, 2f32, 3f32], &Device::Cpu)?;
//! println!("{:?}", vector.shape()); // [3]
//! ```
//!
//! ### 矩阵
//!
//! 矩阵是一个多维数组，它有行和列。
//! ```
//! let matrix = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &Device::Cpu)?.reshape(&[2, 2])?;
//! println!("{:?}", &matrix.shape()); // [2, 2]
//! let transposed = &matrix.t()?;
//! println!("{:?}", transposed.to_vec2::<f32>()?);  // [[1, 3], [2, 4]]
//! ```
//!
//! ### 张量
//!
//! 张量是一个多维数组，它是一个标量、向量、矩阵的泛化。当我们处理图像、音频、文本等数据时，我们通常会构建具有更多轴的数据结构。
//!
//! 比如，一张彩色图像通常有三个轴：高度、宽度和颜色通道。
//! ```
//! let tensor = Tensor::arange(0f32, 24f32, 1f32, &Device::Cpu)?.reshape(&[2, 3, 4])?;
//! println!("{:?}", tensor.shape()); // [2, 3, 4]
//! ```
//!
//! ## 运算
//!
//! [DataOp]
//!
//! ## 降维
//!
//! ## 点积
//!
//! ## 矩阵 向量积
//!
//! ## 矩阵乘法
//!
//! ## 范数
//!
//! ## 矩阵分解

/// # 线性代数
pub struct LinearAlgebra;

impl LinearAlgebra {
    /// # 降维
    /// 
    /// ```
    /// # use candle_core::{Device, Result, Tensor, NdArray};
    /// # fn main() -> Result<()> {
    /// let x = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape(&[3, 4])?; // [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    /// let sum = &x.sum(0)?; // [12, 15, 18, 21]
    /// # assert_eq!(sum.shape().dims(), [4]);
    /// let sum_1 = &x.sum_all()?; // [66]
    /// # assert_eq!(sum_1.shape().elem_count(), 1);
    /// let mean = &x.mean(0)?; // [4, 5, 6, 7]
    /// # assert_eq!(mean.to_vec1::<f32>()?, vec![4f32, 5f32, 6f32, 7f32]);
    /// let mean_1 = &x.mean_all()?; // [5.5]
    /// let el = Tensor::new(&[x.shape().elem_count() as f32], &x.device())?;
    /// let mean_2 = sum_1.broadcast_div(&el)?; // [5.5]
    /// # assert_eq!(mean_1.to_scalar::<f32>()?, 5.5f32);
    /// # assert_eq!(mean_2.to_vec1::<f32>()?, vec![5.5f32]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn dimensionality_reduction() {
        // Empty implementation
    }
}

#[cfg(test)]
mod tests {

    use candle_core::{Device, Result, Tensor};
    // use super::*;

    #[test]
    fn add_works() -> Result<()> {
        let d = Device::cuda_if_available(0)?;

        let matrix = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &Device::Cpu)?.reshape(&[2, 2])?;
        println!("{:?}", &matrix.shape()); // [2, 2]
        let transposed = &matrix.t()?;
        println!("{:?}", transposed.to_vec2::<f32>()?); // [[1, 3], [2, 4]]

        let x = Tensor::arange(0f32, 12f32, &d)?.reshape(&[3, 4])?; // [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        println!("{:?}", &x); // [3, 4]
        let sum = &x.sum_keepdim(0)?;
        let sum_1 = &x.sum_all()?;
        let mean_1 = &x.mean_all()?;

        println!("{:?}", sum);
        println!("{:?}", sum_1);

        let el = Tensor::new(&[x.shape().elem_count() as f32], &d)?;
        println!("{:?}", el);

        let mean_2 = sum_1.broadcast_div(&el)?;
        println!("{:?}", mean_1.to_scalar::<f32>());
        println!("{:?}", mean_2.to_scalar::<f32>());



        // assert_eq!(sum.to_vec2::<f32>()?, vec![12f32, 15f32, 18f32, 21f32]);
        // assert_eq!(sum_1.to_vec1::<f32>()?, vec![12f32, 15f32, 18f32, 21f32]);

        Ok(())
    }
}
