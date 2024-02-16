//! # 第二章：hf-hub
//!
//! 1. 安装：
//! 
//! ```shell
//! cargo add hf-hub
//! ```
//! 
//! 2. sync & async
//! 
//! 3. proxy
//!

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
