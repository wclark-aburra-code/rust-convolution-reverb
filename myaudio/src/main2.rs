extern crate hound;


use std::f32::consts::PI;
use hound::{SampleFormat, WavSpec, WavWriter};

fn generate_sine(filename: &str, frequency: f32, duration: u32) {
    let header = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(filename, header).expect("Failed to created WAV writer");
    let num_samples = duration * header.sample_rate;
    let signal_amplitude = 16384f32;
    for n in 0..num_samples {
        let t: f32 = n as f32 / header.sample_rate as f32;
        let x = signal_amplitude * (t * frequency * 2.0 * PI).sin();
        writer.write_sample(x as i16).unwrap();
    }
}
fn main() {
    generate_sine("test.wav", 1000f32, 5);
}
