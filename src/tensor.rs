use std::{slice, sync::Arc, vec};
use crate::model::Llama;
use std::ops::{Add, Mul};
use std::iter::repeat;

// implementations for tensor multiple and addition
fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result_shape = vec![];
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);

    for i in 0..max_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return None; // Incompatible shapes
        }
        result_shape.push(dim1.max(dim2));
    }

    result_shape.reverse();
    Some(result_shape)
}
impl Tensor<f32> {
    pub fn get_length(&self) -> usize{
        self.length
    }
    pub fn add(&self, other: &Self) -> Self {
        if let Some(broadcast_shape) = broadcast_shapes(self.shape(), other.shape()) {
            let mut result_data = vec![0.0; broadcast_shape.iter().product()];
            let mut result = Tensor::from_data(result_data, &broadcast_shape);

            for i in 0..result.length {
                let self_index = self.map_index(i, &broadcast_shape);
                let other_index = other.map_index(i, &broadcast_shape);
                unsafe {
                    *result.data_mut().get_unchecked_mut(i) = *self.data().get_unchecked(self_index) + *other.data().get_unchecked(other_index);
                }
            }

            result
        } else {
            panic!("Incompatible shapes for addition: {:?} and {:?}", self.shape(), other.shape());
        }
    }

    fn map_index(&self, index: usize, broadcast_shape: &[usize]) -> usize {
        let mut result_index = 0;
        let mut stride = 1;
        for (dim, &shape_dim) in self.shape().iter().rev().zip(broadcast_shape.iter().rev()) {
            let index_dim = index / stride % shape_dim;
            if *dim == 1 {
                result_index += 0;
            } else {
                result_index += index_dim * stride;
            }
            stride *= shape_dim;
        }
        result_index
    }
}

impl Tensor<f32> {
    pub fn mul(&self, other: &Self) -> Tensor<f32> {
        let _lhs = self.data();
        let _rhs = other.data();
        let mut result_vector: Vec<f32> = vec![0.0; self.shape().iter().product()];
        for(index, &value) in _lhs.iter().enumerate() {
            // 逐元素计算数组乘法
            result_vector.push(value*_rhs[index]);
        }

        Tensor::from_data(result_vector, &self.shape())
    }
}

// get the last dimension vector in the tensor
impl<T: Copy> Tensor<T> {
    // 获取高维张量中所有最后一维的向量
    pub fn all_last_dim_vectors(&self) -> Vec<Vec<T>> {
        if self.shape.is_empty() {
            panic!("Tensor has no dimensions");
        }

        // 获取最后一维的大小
        let last_dim_size = *self.shape.last().unwrap();

        // 计算前面所有维度的大小
        let mut stride = last_dim_size;
        for &dim in self.shape.iter().rev().skip(1) {
            stride *= dim;
        }

        // 提取所有最后一维的向量
        let mut result = Vec::new();
        for i in 0..(self.length / last_dim_size) {
            let start = self.offset + i * last_dim_size;
            let end = start + last_dim_size;
            result.push(self.data[start..end].to_vec());
        }

        result
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    // add unsafe method for replace element in tensor
    pub unsafe fn replace_unchecked(&mut self, index: usize, value: T) {
        if index >= self.length {
            panic!("Index out of bounds: {} >= {}", index, self.length);
        }

        // 获取可变指针并修改值
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        *ptr.add(index) = value;
    }
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub fn from_data(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice()),
            shape: shape.clone(),
            offset: 0,
            length,
        }
    }

    // data_mut means data MUTABLE slice. NOT multiple.
    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }


}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();

        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}