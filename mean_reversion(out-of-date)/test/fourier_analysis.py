import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

def analyze_file(file_path):
    print(f"Processing {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Sort by timestamp just in case
    df = df.sort_values('timestamp')
    
    # Extract time and signal
    t_raw = df['timestamp'].values
    signal_raw = df['power'].values
    
    # Normalize time to start at 0
    t_start = t_raw[0]
    t_raw = t_raw - t_start
    
    # Interpolate to uniform sampling rate
    # Calculate mean sampling interval
    dt_mean = np.mean(np.diff(t_raw))
    print(f"Mean sampling interval: {dt_mean:.4f} seconds")
    
    # Define uniform time grid (resampling at 1 second or similar)
    # Let's use a step slightly smaller than the mean to avoid losing detail, 
    # but given the data looks like ~6s, maybe 1s is good.
    dt_uniform = 1.0 
    t_uniform = np.arange(0, t_raw[-1], dt_uniform)
    
    # Linear interpolation
    f_interp = interp1d(t_raw, signal_raw, kind='linear', fill_value="extrapolate")
    signal_uniform = f_interp(t_uniform)
    
    # Perform FFT
    n = len(signal_uniform)
    fft_result = np.fft.rfft(signal_uniform)
    freqs = np.fft.rfftfreq(n, d=dt_uniform)
    
    # Magnitude for picking dominant frequencies
    magnitude = np.abs(fft_result) / n
    
    # Ignore DC component (0 Hz) for dominant frequency selection if desired, 
    # but we need it for reconstruction.
    # Let's pick top K dominant frequencies excluding DC (index 0)
    top_k = 10
    
    # Indices of top frequencies (excluding DC)
    # argsort is ascending, so take last k
    top_indices = np.argsort(magnitude[1:])[-top_k:] + 1
    # Sort indices to be in frequency order
    top_indices = np.sort(top_indices)
    
    print("Dominant Frequencies (Hz):")
    for idx in top_indices:
        print(f"  {freqs[idx]:.6f} Hz (Period: {1/freqs[idx]:.2f} s)")
        
    # Reconstruct signal using ONLY these dominant frequencies + DC
    # We can do this by zeroing out other bins and doing IRFFT
    fft_filtered = np.zeros_like(fft_result)
    fft_filtered[0] = fft_result[0] # Keep DC
    fft_filtered[top_indices] = fft_result[top_indices]
    
    fitted_signal = np.fft.irfft(fft_filtered, n=n)
    
    # Separate Sine and Cosine components for the top frequencies
    # FFT value X[k] = A_k - i B_k
    # component is (2/N) * (Real * cos + Imag * sin) ? 
    # Actually, standard definition:
    # X[k] = sum(x[n] * exp(-i 2pi k n / N))
    # x[n] = (1/N) sum(X[k] * exp(i 2pi k n / N))
    #      = (1/N) [ X[0] + sum_{k=1}^{N/2} (X[k] e^{...} + X[-k] e^{...}) ]
    # For real signal, X[-k] = conj(X[k]).
    # Term k: X[k] (cos(...) + i sin(...))
    # Let X[k] = a + ib. 
    # Term k + Term -k = (a+ib)(cos + i sin) + (a-ib)(cos - i sin)
    #                  = a cos + i a sin + i b cos - b sin + a cos - i a sin - i b cos - b sin
    #                  = 2a cos - 2b sin
    # So the component is 2 * Real(X[k])/N * cos(wt) - 2 * Imag(X[k])/N * sin(wt)
    
    components = {}
    
    # DC Component
    dc_component = np.full_like(t_uniform, fft_result[0].real / n)
    
    for idx in top_indices:
        freq = freqs[idx]
        omega = 2 * np.pi * freq
        
        # Coefficients
        a_k = fft_result[idx].real
        b_k = fft_result[idx].imag
        
        # Scale by 2/N because we are taking one side of the spectrum
        scale = 2.0 / n
        
        # Cosine wave: (2 * Real / N) * cos(wt)
        cos_wave = scale * a_k * np.cos(omega * t_uniform)
        
        # Sine wave: (-2 * Imag / N) * sin(wt) (Note the negative sign from expansion above)
        sin_wave = scale * (-b_k) * np.sin(omega * t_uniform)
        
        components[freq] = (cos_wave, sin_wave)

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # 1. Original vs Fitted
    plt.subplot(3, 1, 1)
    plt.plot(t_uniform, signal_uniform, label='Original (Interpolated)', alpha=0.5)
    plt.plot(t_uniform, fitted_signal, label=f'Fitted (Top {top_k} Freqs)', linewidth=2)
    plt.plot(t_uniform, dc_component, label='DC Component', linestyle='--')
    plt.title('Original vs Fitted Signal')
    plt.legend()
    plt.grid(True)
    
    # 2. Components (Top 3 for visibility)
    plt.subplot(3, 1, 2)
    plot_count = 0
    for freq in top_indices[:3]: # Just plot first few dominant ones to avoid clutter
        f_hz = freqs[freq]
        cos_w, sin_w = components[f_hz]
        plt.plot(t_uniform, cos_w, label=f'{f_hz:.4f}Hz Cos')
        plt.plot(t_uniform, sin_w, linestyle=':', label=f'{f_hz:.4f}Hz Sin')
        plot_count += 1
    plt.title('Separated Sine/Cosine Waves (Top 3 Dominant)')
    plt.legend()
    plt.grid(True)

    # 3. Frequency Spectrum
    plt.subplot(3, 1, 3)
    plt.plot(freqs, magnitude)
    plt.scatter(freqs[top_indices], magnitude[top_indices], color='red', zorder=5)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 0.1) # Zoom in on lower frequencies usually relevant for appliances
    plt.grid(True)
    
    output_img = os.path.join(os.path.dirname(__file__), 'fourier_result.png')
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

    # Save results to CSV
    results_df = pd.DataFrame({
        'time': t_uniform,
        'original_interpolated': signal_uniform,
        'fitted_signal': fitted_signal,
        'dc_component': dc_component
    })

    # Add components to DataFrame
    for idx in top_indices:
        freq = freqs[idx]
        cos_w, sin_w = components[freq]
        results_df[f'freq_{freq:.6f}_cos'] = cos_w
        results_df[f'freq_{freq:.6f}_sin'] = sin_w

    output_csv = os.path.join(os.path.dirname(__file__), 'fourier_fitting_data.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Fitted data and components saved to {output_csv}")

if __name__ == "__main__":
    # Path to one of the data files
    data_dir = r"f:\B__ProfessionProject\NILM\Clasp\mean_reversion\project\washing_machine\data"
    # Pick the first file or a specific one
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if files:
        target_file = os.path.join(data_dir, files[0])
        analyze_file(target_file)
    else:
        print("No CSV files found in data directory.")
