use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, Activation, layer_norm, LayerNormConfig, VarBuilder};

#[derive(Clone, Debug)]
pub struct GatedLinearUnit {
    w_4: Linear,
    w_5: Linear,    
}

impl GatedLinearUnit {

    pub fn new(d_in: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        Ok(GatedLinearUnit {
            w_4: linear(d_in, d_model, vb.pp("w4_"))?,
            w_5: linear(d_in, d_model, vb.pp("w5_"))?,
        })
    }

    pub fn forward(&self, x: &candle_core::Tensor) -> Result<Tensor> {
        let a = Activation::Sigmoid.forward(&self.w_4.forward(x)?)?; 
        let b = self.w_5.forward(x)?;
        a.mul(&b)
    }
}


#[derive(Clone, Debug)]
pub struct GatedResidualNetwork {
    w_2: Linear,
    w_3: Option<Linear>, // no bias

    w_1: Linear,

    gate: GatedLinearUnit,
    norm: candle_nn::LayerNorm,
}

impl GatedResidualNetwork {
    pub fn new(d_model: usize, a_in: usize, c_in: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let w_3 = match c_in {
            Some(cc_in) => Some(linear(cc_in, d_model, vb.pp("w3_"))?),
            None => None,
        };
        Ok(Self {
            w_2: linear(a_in, d_model, vb.pp("w2_"))?,
            w_3: w_3,
            w_1: linear(d_model, d_model, vb.pp("w1_"))?,
            gate: GatedLinearUnit::new(d_model, a_in, vb.pp("glu_"))?,
            norm: layer_norm(a_in, LayerNormConfig::default(), vb.pp("norm_") )?,
         })
    }

    pub fn forward(&self, a: &Tensor, c: Option<&Tensor>) -> Result<Tensor> {
        let mut x = self.w_2.forward(a)?;
        if c.is_some() {
            x = x.add(&self.w_3.as_ref().expect("`c` can only be provided if `c_in` non-None at instantiation").forward(&c.expect("msg"))?)?
        }
        let n2 = Activation::Elu(1.0).forward(&x)?;
        let n1 = self.w_1.forward(&n2)?;
        self.norm.forward(
            &a.add(&self.gate.forward(&n1)?)?
        )
    }
}
