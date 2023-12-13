use candle_core::{Result, DType, Device};
use candle_nn::{VarBuilder, VarMap};
use selection_network::VariableSelectionNetwork;

mod grn;
mod util;
mod selection_network;


fn main() -> Result<()> {
    let vb = VarBuilder::from_varmap(
        &VarMap::new(),
        DType::F32,
        &Device::Cpu
    );
    let vsn = VariableSelectionNetwork::new(7, 3, 5,  4, vb.pp("vsn_"));
    
    let x = util::rand_norm_vector(vec![7, 3], None);
    let c = util::rand_norm_vector(vec![7, 4], None);

    match vsn {
        Ok(v) => {
            match v.forward(&x, &c) {
                Ok(r) => println!("{:#?}", r),
                Err(e) => eprint!("{:#?}", e),
            }
        },
        Err(e) => eprintln!("{}", e)
    }

    Ok(())
}
