// Which candle op costs what, at batch 1 and 4, on the Morpheus shapes.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn bench<F: FnMut() -> candle_core::Result<Tensor>>(label: &str, iters: usize, mut f: F) {
    for _ in 0..5 { f().unwrap(); }
    let t = Instant::now();
    for _ in 0..iters { std::hint::black_box(f().unwrap()); }
    println!("{label}: {:.4} ms/call", t.elapsed().as_secs_f64() * 1e3 / iters as f64);
}

fn main() -> candle_core::Result<()> {
    let d = Device::Cpu;
    for &b in &[1usize, 4] {
        let x64 = Tensor::rand(0f32, 1f32, (b, 64, 21, 21), &d)?;
        let x128 = Tensor::rand(0f32, 1f32, (b, 128, 21, 21), &d)?;
        let pw_e = Tensor::rand(0f32, 1f32, (128, 64, 1, 1), &d)?;
        let pw_p = Tensor::rand(0f32, 1f32, (64, 128, 1, 1), &d)?;
        let dw = Tensor::rand(0f32, 1f32, (128, 1, 3, 3), &d)?;
        let stem_w = Tensor::rand(0f32, 1f32, (64, 49, 3, 3), &d)?;
        let xin = Tensor::rand(0f32, 1f32, (b, 49, 21, 21), &d)?;
        println!("--- batch {b} ---");
        bench("stem conv3x3 49->64", 200, || xin.conv2d(&stem_w, 1, 1, 1, 1));
        bench("pw_expand 64->128", 500, || x64.conv2d(&pw_e, 0, 1, 1, 1));
        bench("pw_project 128->64", 500, || x128.conv2d(&pw_p, 0, 1, 1, 1));
        bench("dw 3x3 groups=128 d=1", 200, || x128.conv2d(&dw, 1, 1, 1, 128));
        bench("dw 3x3 groups=128 d=4", 200, || x128.conv2d(&dw, 4, 1, 4, 128));
        bench("clamp(relu6) 128ch", 500, || x128.clamp(0f32, 6f32));
    }
    Ok(())
}
