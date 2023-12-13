use candle_core::{Result, Tensor};
use candle_nn::{Module, Activation, VarBuilder};

use crate::grn::GatedResidualNetwork;


#[derive(Clone, Debug)]
pub struct VariableSelectionNetwork {
    input_grns: Vec<GatedResidualNetwork>,
    selection_grn: GatedResidualNetwork,
}

impl VariableSelectionNetwork {
    pub fn new(m_x: usize, x_in: usize, d_model: usize, c_in: usize, vb: VarBuilder) -> Result<Self> {
        let input_grns = (1..m_x).map::<GatedResidualNetwork, _>(
            |i| GatedResidualNetwork::new(
                d_model, x_in, None, vb.pp(format!("grn_{}", i)),
            ).expect("msg")
        ).collect();

        Ok(VariableSelectionNetwork { input_grns, selection_grn: GatedResidualNetwork::new(
            d_model, x_in, Some(c_in), vb.pp(""),
        )? })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Map each of i'th element in x to its associated input_grns, and combine the resultant vectors.
        let inputs_x = Tensor::stack(
            &(1..self.input_grns.len()).map::<Tensor, _>(
                |i| self.input_grns[i].forward(&x, None).expect("msg"), // [i, :]
            ).collect::<Vec<_>>(), 1)?;

        let variable_selection = Activation::Sigmoid.forward(
            &self.selection_grn.forward(&x, Some(&c))?
        )?;

        Ok((inputs_x.mul(&variable_selection)?).cumsum(0)?) // TODO fix dim.
    }
}
        