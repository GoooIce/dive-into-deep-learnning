//! # 数据操作 
//! 
//!    - [x] 1.1 运算符
//!    - [x] 1.2 广播机制
//!    - [x] 1.3 索引和切片
//! 
//! 
//! 
use candle_core::{ Result, Tensor};

/// # 数据操作
pub struct DataOp;

impl DataOp {
    /// # 运算符
    pub fn add(left: Tensor, right: Tensor) -> Result<Tensor> {
        left + right
    }

    /// # 广播机制
    pub fn broadcast_add(left: Tensor, right: Tensor) -> Result<Tensor> {
        left.broadcast_add(&right)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, IndexOp};

    use super::*;

    #[test]
    fn add_works() -> Result<()> {
        // let d = Device::gpu(0);
        let d = Device::Cpu;

        // 标量加法
        let left = Tensor::new(2f32, &d)?;
        let right = Tensor::new(2f32, &d)?;
        let result = DataOp::add(left, right)?;

        assert_eq!(result.to_scalar::<f32>()?, 4f32);

        // 向量加法
        let left = Tensor::new(vec![1f32, 2f32, 3f32], &d)?;
        let right = Tensor::new(vec![4f32, 5f32, 6f32], &d)?;
        let result = DataOp::add(left, right)?;

        assert_eq!(result.to_vec1::<f32>()?, vec![5f32, 7f32, 9f32]);

        // 张量加法
        let left = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &d)?.reshape(&[2, 2])?;
        let right = Tensor::new(vec![5f32, 6f32, 7f32, 8f32], &d)?.reshape(&[2, 2])?;
        let result = DataOp::add(left, right)?;

        assert_eq!(
            result.to_vec2::<f32>()?,
            vec![vec![6f32, 8f32], vec![10f32, 12f32]]
        );

        Ok(())
    }

    #[test]
    fn add_broadcast_works() -> Result<()> {
        let d = Device::Cpu;

        // 使用加号运算符
        // 向量加法报错
        let left = Tensor::new(vec![1f32, 2f32, 3f32], &d)?;
        let right = Tensor::new(vec![1f32, 2f32], &d)?;
        let result = DataOp::add(left, right);

        assert!(result.is_err());

        // 向量加法
        let left = Tensor::new(vec![1f32, 2f32, 3f32], &d)?;
        let right = Tensor::new(1f32, &d)?;
        let result = DataOp::broadcast_add(left, right)?;

        assert_eq!(result.to_vec1::<f32>()?, vec![2f32, 3f32, 4f32]);

        // 矩阵加法
        let left = Tensor::new(vec![1f32, 2f32, 3f32, 4f32], &d)?.reshape(&[2, 2])?;
        let right = Tensor::new(vec![1f32, 2f32], &d)?;
        let result = DataOp::broadcast_add(left, right)?;

        assert_eq!(
            result.to_vec2::<f32>()?,
            vec![vec![2f32, 4f32], vec![4f32, 6f32]]
        );

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
