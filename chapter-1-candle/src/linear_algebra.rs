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
//! let scalar = Tensor.new(1.0, &Device::CPU)?.to_scalar::<f32>()?;
//! ```
//!
//! ### 向量
//!
//! 向量是一个一维数组，它有方向和大小。
//! ```
//! let vector = Tensor::new(vec![1f32, 2f32, 3f32], &Device::CPU)?;
//! println!("{:?}", vector.shape()); // [3]
//! ```
//!
//! ### 矩阵
//! 
//! 矩阵是一个多维数组，它有行和列。
//! ```
//! let matrix = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &Device::CPU)?.reshape(&[2, 2])?;
//! println!("{:?}", matrix.shape()); // [2, 2]
//! ```
//! 
//! ### 张量
//!
//! 张量是一个多维数组，它是一个标量、向量、矩阵的泛化。当我们处理图像、音频、文本等数据时，我们通常会构建具有更多轴的数据结构。
//!
//! 比如，一张彩色图像通常有三个轴：高度、宽度和颜色通道。
//! ```
//! let tensor = Tensor::arange(0f32, 24f32, 1f32, &Device::CPU)?.reshape(&[2, 3, 4])?;
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
}

#[cfg(test)]
mod tests {
    
    use candle_core::{Device, Result, Tensor};
    // use super::*;

    #[test]
    fn add_works() -> Result<()> {
        let d = Device::cuda_if_available(0)?;

        let x = Tensor::arange(0f32, 4f32, &d)?;

        let sum = x.sum(0)?;

        assert_eq!(sum.to_scalar::<f32>()?, 6f32);

        Ok(())
    }

}
