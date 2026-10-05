//! max(0, x) for every element, over Poulpy's CKKS.
//!
//! The polynomial is the one the OpenFHE, FIDESlib and DESILO answers to this
//! specification evaluate: the MILP-optimised polynomial of order 16 from the
//! ReLUFunction component of fairmath/polycircuit, the winning entry of the
//! FHERMA ReLU challenge's depth-constrained track. Its seventeen
//! coefficients are below, unchanged, so what the board compares is the
//! library and not the mathematics.
//!
//! They are given in the monomial basis, which is how the component states
//! them; the library takes either basis and folds the leading -54 in as a
//! coefficient rather than by repeated subtraction, which is what the
//! challenge's depth budget forced on the original and this specification
//! does not.
use poulpy_ckks::api::CKKSPolynomialEvaluationOps;
use poulpy_ckks::layouts::CKKSPlaintextOwned;
use poulpy_ckks::polynomial::{BSGSPolynomial, Basis, EncodeBSGS, Polynomial};
use poulpy_ckks::power_basis::{PowerBasis, PowerBasisGen};
use poulpy_ckks::test_suite::helpers::{alloc_ct, upload_pt};
use poulpy_ckks::CoeffsMeta;
use poulpy_core::layouts::Base2K;
use poulpy_hal::api::ScratchOwnedBorrow;

use crate::envelope::{Backend, Ct, Env, Pt, PtOut};
use crate::fherma::{Inputs, Outputs, Point, Tensor};

/// Coefficients of x^0 .. x^16, as published.
const POLYNOMIAL: [f64; 17] = [
    0.0323949878919212, 0.500001412106499, 2.13483086933591,
    -4.78160418051218e-05, -13.9205486530553, 0.00061641818435605,
    70.0957556465309, -0.00388040016141976, -213.087053403128,
    0.0129145434432087, 385.924971250905, -0.0230082806472531,
    -407.029727261512, 0.0206280915579812, 230.348436664049,
    -0.00728306945833855, -54.0,
];

pub struct State {
    poly: BSGSPolynomial<CKKSPlaintextOwned<Backend>>,
}

/// Public material, built once per point and never measured.
pub fn init(_p: &Point, env: &mut Env) -> State {
    let poly = Polynomial::new(Basis::Monomial, POLYNOMIAL.to_vec()).with_interval(-1.0f64, 1.0);
    let meta = CoeffsMeta::from_delta_budget(env.params.prec_meta.log_delta, 8);
    let host = poly
        .encode_bsgs(&env.host_module, Base2K(env.params.base2k as u32), meta)
        .expect("encode bsgs");
    State { poly: host.map_baby_steps_ref(|pt| upload_pt(&env.module, pt)) }
}

pub fn encoding(env: &mut Env, inp: &Inputs) -> Vec<Pt> {
    vec![env.encode_reals(&inp.xs.data)]
}

pub fn run(state: &mut State, env: &mut Env, cts: Vec<Ct>) -> Vec<Ct> {
    let x = cts.into_iter().next().expect("one ciphertext");
    let mut basis = PowerBasis::new(Basis::Chebyshev, x);
    basis
        .populate(
            POLYNOMIAL.len() - 1,
            state.poly.log_split(),
            state.poly.parity(),
            &env.module,
            &env.tensor_key,
            &mut env.scratch.borrow(),
        )
        .expect("power basis");
    let mut out = alloc_ct(&env.params, &env.module, env.params.k);
    env.module
        .ckks_eval_poly_real_const_coeffs_from_power_basis(
            &mut out,
            &state.poly,
            &basis,
            &env.tensor_key,
            &mut env.scratch.borrow(),
        )
        .expect("evaluate");
    vec![out]
}

pub fn decoding(p: &Point, env: &mut Env, pts: Vec<PtOut>) -> Outputs {
    let values = env.decode_reals(&pts[0], p.N as usize);
    Outputs { r: Tensor { shape: vec![p.N as i64], data: values } }
}

pub fn free(state: State) {
    drop(state);
}
