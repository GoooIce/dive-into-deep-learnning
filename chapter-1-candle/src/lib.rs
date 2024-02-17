//! # 第一章：Candle
//! 
//! ## 安装
//! 
//! ```bash
//! cargo new myapp
//! cd myapp
//! cargo add --git https://github.com/huggingface/candle.git candle-core
//! ```
//! [更详细的安装文档](https://huggingface.github.io/candle/guide/installation.html)
//!
//! 1. 数据操作：[data_op]
//! 
//! 2. 线性代数：[linear_algebra]
//! 
//! 3. 微积分：[calculus]
//!
extern crate candle_core;
use candle_core::{Device, Result, Tensor};

pub mod data_op;
pub mod linear_algebra;
pub mod calculus;
