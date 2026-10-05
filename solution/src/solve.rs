//! max(0, x) for every element, over Poulpy's CKKS.
//!
//! The polynomial is the one the OpenFHE, FIDESlib and DESILO answers to this
//! specification evaluate: the MILP-optimised polynomial of order 16 from the
//! ReLUFunction component of fairmath/polycircuit, the winning entry of the
//! FHERMA ReLU challenge's depth-constrained track. Its seventeen
//! coefficients are below, unchanged, so what the board compares is the
//! library and not the mathematics.
//!
//! The component states them in the monomial basis; they are converted here
//! to the Chebyshev basis the library's evaluator works in, exactly. The
//! leading -54 is a coefficient like any other rather than something folded
//! in by repeated subtraction, which is what the challenge's depth budget
//! forced on the original and this specification does not.
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

/// The MILP polynomial, in the Chebyshev basis.
///
/// The component states it in the monomial basis, as x^0 .. x^16 with a
/// leading -54; this is the same polynomial written in the basis the
/// library's evaluator builds its power basis in, converted exactly — the
/// two agree to 9e-14 across [-1, 1]. Stating it in the other basis and
/// evaluating it in this one is not a conversion the library performs.
const POLYNOMIAL: [f64; 17] = [
    0.3192235998041225, 0.4999859402748887, 0.21038635064247718,
    -1.3080336695987431e-05, -0.04063791756501771, -1.1226404519249844e-05,
    0.016424135051813456, -8.83456151853052e-06, -0.008373177365811165,
    -6.149820007635184e-06, 0.004770552720991594, -3.6739515665634156e-06,
    -0.0028370111998732885, -1.6316940700166904e-06, 0.0017515181474669177,
    -4.4452328236929626e-07, -0.00164794921875,
];


pub struct State {
    poly: BSGSPolynomial<CKKSPlaintextOwned<Backend>>,
}

/// Public material, built once per point and never measured.
pub fn init(_p: &Point, env: &mut Env) -> State {
    let poly = Polynomial::new(Basis::Chebyshev, POLYNOMIAL.to_vec()).with_interval(-1.0f64, 1.0);
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
