use candle_core::{Device, Tensor};

pub fn rand_norm_vector(shape: Vec<usize>, device: Option<Device>) -> Tensor {
    let d = device.unwrap_or(Device::Cpu);
    Tensor::randn::<Vec<usize>, f32>(0f32, 1.0, shape.clone(), &d).expect(
        format!("couldn't create vector of shape {:?} on device {:?}", shape, d).as_str()
    )
}