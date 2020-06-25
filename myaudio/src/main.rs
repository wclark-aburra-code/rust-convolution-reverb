extern crate num;
extern crate rustfft;
use num::complex::Complex;
use num::Num;
use rustfft::FFTplanner;
use hound::{WavReader, WavWriter};
use std::f32::consts::PI;
use std::cmp::Ordering;

struct SignalBuffer<T: Num> {
    used: Option<usize>,
    new: Option<usize>,
    end: Option<usize>,
    zero_value: T,
    contents: Vec<T>
}

fn build_signal_buffer<T: Num>(length: usize, zero_element: T) -> SignalBuffer<T> where T: Clone {
    SignalBuffer { 
        used: None,
        new: None,
        end: None,
        zero_value: zero_element.clone(),  // very useful construct -- carry over to C++ version. test C++ version's performance with Options and zero_value  
        contents: vec![zero_element; length]
     }
}

impl SignalBuffer<num::Complex<f32>> {
    fn insert_real(&mut self, new_real_element: f32) {
        self.end = match self.end  {
            None =>  { self.contents[0].re = new_real_element; self.contents[0].im = 0.0; Some(0) }, // maybe set the imaginary part to zero: self.contents[0].im = 0... we want to guarantee this for general use of the struct, but it's a waste of performance if our implementation doesn't need it
            Some(end_val) => { self.contents[end_val+1].re = new_real_element; self.contents[end_val+1].im = 0.0; Some(end_val+1) }
        };
    }    
}

impl<T> SignalBuffer<T> where T: num::Num  {
    fn set_end(&mut self, end_idx: usize) {  // only for use after "contents" have been directly loaded via FFT process
        self.end = Some(end_idx);
    }
    fn insert(&mut self, new_element: T) {
        self.end = match self.end  {
            None =>  { self.contents[0] = new_element; Some(0) },
            Some(end_val) => { self.contents[end_val+1] = new_element; Some(end_val+1) }
        };
    }
    fn reset(&mut self) {
        self.used = None;
        self.new = self.end;
    }
    fn clear(&mut self) {
        self.used = None;
        self.new = None;
        self.end = None;
    }
    fn next(&mut self) -> T where T:Copy  {
        match self.new.cmp(&self.used) {
            Ordering::Greater =>  match self.used {
                None =>  { self.used = Some(0); self.contents[0] },
                Some(idx) => { self.used = Some(idx+1); self.contents[idx+1] }
            },
            _ => match self.used.cmp(&self.end) {  // this is redundant, we need to check this comparison in ABOVE match statement
                Ordering::Less => self.zero_value,
                _ => self.zero_value    // refactor function to return Option type, and return None here
                                        // make sure this applies properly to the overlap_buffer case, which should be all zeroes in the first loop iteration
            }
        }
    }    
}

fn hanning_multiplier(i: f32, block_size: f32) -> f32 {     // Hanning window to minimize spectral leakage; without this, we just get noise
    0.5 * (1.0 - (2.0*PI*i/(block_size-1.0)).cos())
}

fn padded(input: &[f32], total: usize) -> Vec<f32> {
    let num_zeroes = total - input.len();
    let zeroes = vec![0f32; num_zeroes];
    [&input[..], &zeroes[..]].concat()
}

fn wave_vector(signal_filename: &str) -> Vec<f32> {
    let mut signal_reader = WavReader::open(signal_filename).expect("Failed to open WAV file"); // this assumes the wave is mono
    signal_reader.samples::<i16>()
        .map(|x| (x.unwrap() as f32))
        .collect::<Vec<_>>()
}

fn convolve(dry_signal: Vec<f32>, ir_signal: Vec<f32>) -> (Vec<f32>, f32) {     // terminology: a "dry signal" is unprocessed. an "impulse response," the ir_signal, is the finite signal with which we convolve the dry signal, to yield a processed or "wet" signal
    let mut max = 0.0;
    for sample in &dry_signal { // use mutable iterator instead, to avoid confusion around borrowing and dereferencing?
        if sample > &max {
            max = *sample;
        }
    }
    let ir_length = ir_signal.len();
    println!("ir_length {}", ir_length);
    let mut total_size: usize = 262144; // this should start way lower -- some IRs will be short, s.a. a quick moving-average LPF, or differentiator HPF. only reverb/delay IRs will be especially long
    println!("total_size {}", total_size);
    while total_size < ir_length { 
        total_size *= 2;
    }
    let window_size = total_size - ir_length + 1;
    println!("window_size {}", window_size);
    let output_length = total_size; // same, just not mutable
    println!("output_length {}", output_length);
    let overlap_length = output_length - window_size;
    println!("overlap_length {}", overlap_length);
    let window_size_as_float = window_size as f32;
    let mut ir_wav_arr = padded(&ir_signal, output_length).iter().map(|x| Complex::new(*x as f32, 0f32)).collect::<Vec<_>>();

    // take FFT of impulse response, save that to a vector for use as the convolution kernel
    let mut ir_planner = FFTplanner::new(false);   // "false" argument -- this is a forward FFT
    let ir_fft = ir_planner.plan_fft(output_length);
    let mut ir_out: Vec<num::Complex<f32>> = vec![Complex::new(0.0,0.0); output_length];
    ir_fft.process(&mut ir_wav_arr, &mut ir_out);

    let mut num_sections = ((dry_signal.len() as f32) / window_size_as_float).round() as usize; // is this a floor as expected? OBOB? test...
    let total_size = num_sections*window_size;
    let input_vec: Vec<f32> = match dry_signal.len().cmp(&total_size) {
        Ordering::Equal => dry_signal.clone(),  // highly unlikely
        _ => { num_sections+=1; padded(&dry_signal, num_sections*window_size) } // num_sections incremented by second line of block
    };
    let dry_sig_length = dry_signal.len();
    println!("sig_length {}", dry_sig_length);
    let mut results: Vec<f32> = Vec::with_capacity(dry_sig_length + ir_wav_arr.len() - 1); // with_capacity is the same as using reserve()

    let mut section_buffer: SignalBuffer<num::Complex<f32>> = build_signal_buffer(output_length, Complex::new(0f32, 0f32));
    let mut stored_overlap_buffer: SignalBuffer<f32>  = build_signal_buffer(overlap_length, 0.0);
    let mut next_overlap_buffer: SignalBuffer<f32>  = build_signal_buffer(overlap_length, 0.0);
    let mut section_freq_buffer: SignalBuffer<num::Complex<f32>>  = build_signal_buffer(output_length, Complex::new(0f32, 0f32));
    let mut output_buffer: SignalBuffer<f32>  = build_signal_buffer(window_size, 0.0);

    let mut forward_planner = FFTplanner::new(false);   // "false" argument -- this is a forward FFT
    let forward_fft = forward_planner.plan_fft(output_length);

    let mut in_inverse: Vec<num::Complex<f32>> = vec![Complex::new(0.0,0.0); output_length];
    let mut out_inverse: Vec<num::Complex<f32>> = vec![Complex::new(0.0,0.0); output_length];
    let mut inverse_planner = FFTplanner::new(true);    // "true" argument -- this is an inverse FFT
    let inverse_fft = inverse_planner.plan_fft(output_length);    

    let mut i_float: f32;
    let mut complex_product;
    let mut max_new = 0.0;
    for j in 0..num_sections {
        println!("convolving section {}", j);
        i_float = 0.0;
        let offset = j*window_size;
        for i in offset..(offset+window_size) {
            section_buffer.insert_real(hanning_multiplier(i_float, window_size_as_float) * input_vec[i as usize]);
            i_float += 1.0;
        }
        for _ in window_size..output_length {
            section_buffer.insert_real(0.0);    // padding, for buffer
        }
        forward_fft.process(&mut section_buffer.contents, &mut section_freq_buffer.contents);
        section_freq_buffer.set_end(output_length/2);
        section_freq_buffer.reset();

        for (idx, ir_sample) in ir_out.iter().enumerate() {
            complex_product = section_freq_buffer.next()*ir_sample;
            in_inverse[idx] = complex_product;
        }
        inverse_fft.process(&mut in_inverse, &mut out_inverse);
        stored_overlap_buffer.reset();
        next_overlap_buffer.clear();
        for k in 0..window_size {
            output_buffer.insert(out_inverse[k as usize].re + stored_overlap_buffer.next());
        }
        for k in window_size..output_length {
            match k.cmp(&overlap_length) {
                Ordering::Less => next_overlap_buffer.insert(out_inverse[k as usize].re + stored_overlap_buffer.next()),
                _ => next_overlap_buffer.insert(out_inverse[k as usize].re)
            };
        }
        stored_overlap_buffer.clear();
        next_overlap_buffer.reset();
        for _ in 0..overlap_length {
            stored_overlap_buffer.insert(next_overlap_buffer.next());
        }
        stored_overlap_buffer.reset();
        output_buffer.reset();
        for _ in 0..window_size {
            let next_output_sample = output_buffer.next();
            results.push(next_output_sample);
            if next_output_sample < max_new {
                max_new = next_output_sample;
            }
        }
        section_buffer.clear();
        section_freq_buffer.clear();
        output_buffer.clear();
    }
    let normalization_factor = max/max_new;
    println!("NORM {}", normalization_factor);
    (results, normalization_factor)
}

fn write_normalized_signal(result_signal: Vec<f32>, normalization_factor: f32, write_filename: &str) -> () {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int
    };
    let mut writer = WavWriter::create(write_filename, spec).unwrap();
    for sample in result_signal {
        writer.write_sample((sample*normalization_factor) as i16).unwrap();
    }
}

fn main() {
    let dry_signal = wave_vector("H.wav");
    let ir_signal = wave_vector("spaceEchoIR.wav");
    let (wet_signal, normalization_factor) = convolve(dry_signal, ir_signal);
    write_normalized_signal(wet_signal, normalization_factor, "correct0.wav");
}
