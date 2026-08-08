// M3 ladder rung 2: the Morpheus graph through tract-onnx, fixed shapes.
use std::time::Instant;
use tract_onnx::prelude::*;

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let rank = ((q * n as f64).ceil() as usize).max(1).min(n);
    sorted[rank - 1]
}

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("onnx dir");
    let iters: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(200);
    for &batch in &[1usize, 4, 8] {
        let path = format!("{dir}/policy_wdl_b{batch}.onnx");
        let t0 = Instant::now();
        let model = tract_onnx::onnx()
            .model_for_path(&path)?
            .with_input_fact(0, f32::fact([batch, 49, 21, 21]).into())?
            .into_optimized()?
            .into_runnable()?;
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut seed: u32 = 12345;
        let n = batch * 49 * 21 * 21;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push(((seed >> 8) as f32 / 16777216.0) * 2.0 - 1.0);
        }
        let x: Tensor = tract_ndarray::Array4::from_shape_vec((batch, 49, 21, 21), data)?.into();
        for _ in 0..10 { model.run(tvec!(x.clone().into()))?; }
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let out = model.run(tvec!(x.clone().into()))?;
            std::hint::black_box(&out);
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "batch {batch} compile_ms {compile_ms:.1} p50 {:.3} p99 {:.3} min {:.3}",
            percentile(&times, 0.50), percentile(&times, 0.99), times[0]
        );
    }
    Ok(())
}
