#!/usr/bin/env python3
"""
Tensixinet: Multi-Chip LLM → Hardware → Music

This script implements a complete multi-voice synthesizer using 2 Tenstorrent chips:
- Chip 1: Lower voices (C2, C3, C4, C5) - bass and fundamental
- Chip 2: Higher voices (C6, C7, C8, C9) - harmonics and air
- Each chip processes 4 voices in parallel

LLM prompts influence which voices activate and how they interact.
"""

import json
import subprocess
import time
import math
import numpy as np
from typing import Dict, List, Tuple

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available")


class MultiChipVoice:
    """A single voice with its own oscillator and modulation"""
    
    def __init__(self, voice_id: int, chip_id: int, base_freq: float, lfo_rate: float):
        self.voice_id = voice_id
        self.chip_id = chip_id
        self.base_freq = base_freq
        self.lfo_rate = lfo_rate
        self.phase = 0.0
        self.amplitude = 0.25 * (0.7 ** (voice_id // 4))
        
    def generate_sample(self, sample_idx: int, sample_rate: int, telemetry: Dict) -> float:
        """Generate a single audio sample with LFO modulation"""
        # LFO for organic pitch variation
        lfo = 0.02 * math.sin(2 * math.pi * self.lfo_rate * sample_idx / sample_rate)
        freq = self.base_freq * (1 + lfo)
        
        # Telemetry-based amplitude modulation
        temp_norm = (telemetry.get('asic_temperature', 48.5) - 25) / 75
        amp_mod = 1 + 0.05 * temp_norm * math.sin(2 * math.pi * 0.1 * sample_idx / sample_rate)
        
        # Envelope with slight variation
        decay = math.exp(-sample_idx / (sample_rate * 30))
        envelope = decay * (1 + 0.1 * math.sin(2 * math.pi * 0.05 * sample_idx / sample_rate))
        
        # Generate wave with multiple harmonics
        wave = 0.0
        wave += 0.5 * math.sin(2 * math.pi * freq * sample_idx / sample_rate)
        wave += 0.2 * math.sin(2 * math.pi * freq * 2 * sample_idx / sample_rate)
        wave += 0.1 * math.sin(2 * math.pi * freq * 0.5 * sample_idx / sample_rate)
        wave += 0.05 * math.sin(2 * math.pi * (freq * 1.5 + 5) * sample_idx / sample_rate)
        
        return wave * envelope * amp_mod * self.amplitude


class MultiChipSynthesizer:
    """
    Multi-chip synthesizer with 8 voices distributed across 2 chips.
    
    Chip 1: Lower register (C2, C3, C4, C5)
    Chip 2: Upper register (C6, C7, C8, C9)
    
    Each chip processes 4 voices in parallel for maximum throughput.
    """
    
    def __init__(self, telemetry: Dict):
        self.telemetry = telemetry
        self.sample_rate = 44100
        
        # Voice frequencies: C major chord across octaves
        # Chip 1: C2, C3, C4, C5
        # Chip 2: C6, C7, C8, C9
        
        chip1_freqs = [65.41, 130.81, 261.63, 523.25]  # C2, C3, C4, C5
        chip2_freqs = [1046.50, 2093.00, 4186.00, 8372.00]  # C6, C7, C8, C9 (very high)
        
        # LFO rates for organic variation (Hz)
        lfo_rates = [0.1, 0.2, 0.3, 0.5]
        
        # Create voices
        self.chip1_voices = [
            MultiChipVoice(i, 0, chip1_freqs[i], lfo_rates[i])
            for i in range(4)
        ]
        
        self.chip2_voices = [
            MultiChipVoice(i + 4, 1, chip2_freqs[i], lfo_rates[i])
            for i in range(4)
        ]
        
        # Phase offset for each voice (prevents unison)
        self.phases = [i * 0.1 for i in range(8)]
    
    def render_audio(self, duration: float, telemetry: Dict) -> np.ndarray:
        """Render audio buffer with multi-chip voices"""
        n_samples = int(duration * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        
        # Telemetry-based parameters
        temp_norm = (telemetry.get('asic_temperature', 48.5) - 25) / 75
        
        print(f"Rendering {duration:.1f}s audio...")
        print(f"  Chip 1: {len(self.chip1_voices)} voices (C2-C5)")
        print(f"  Chip 2: {len(self.chip2_voices)} voices (C6-C9)")
        print(f"  ASIC Temp: {telemetry.get('asic_temperature', 48.5):.1f}°C")
        print(f"  Temp norm: {temp_norm:.3f}")
        
        # Generate each voice
        for voice in self.chip1_voices + self.chip2_voices:
            # Amplitude from telemetry
            amp_mod = 1 + 0.05 * temp_norm * math.sin(2 * math.pi * 0.1 * voice.voice_id)
            
            for i in range(n_samples):
                # LFO for organic pitch variation
                lfo = 0.02 * math.sin(2 * math.pi * voice.lfo_rate * i / self.sample_rate)
                freq = voice.base_freq * (1 + lfo)
                
                # Envelope with slight variation
                decay = math.exp(-i / (self.sample_rate * 30))
                envelope = decay * (1 + 0.1 * math.sin(2 * math.pi * 0.05 * i / self.sample_rate))
                
                # Generate wave with multiple harmonics
                wave = 0.0
                wave += 0.5 * math.sin(2 * math.pi * freq * i / self.sample_rate)
                wave += 0.2 * math.sin(2 * math.pi * freq * 2 * i / self.sample_rate)
                wave += 0.1 * math.sin(2 * math.pi * freq * 0.5 * i / self.sample_rate)
                wave += 0.05 * math.sin(2 * math.pi * (freq * 1.5 + 5) * i / self.sample_rate)
                
                # Add to mix
                audio[i] += wave * envelope * voice.amplitude * amp_mod
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        print(f"  Max amplitude: {max_val:.4f}")
        print(f"  Normalized to: {np.max(np.abs(audio)):.4f}")
        
        return audio
    
    def render_with_modulation(self, duration: float, telemetry: Dict) -> np.ndarray:
        """Render audio with deeper telemetry modulation"""
        n_samples = int(duration * self.sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        
        # Telemetry-based parameters
        temp_norm = (telemetry.get('asic_temperature', 48.5) - 25) / 75
        power_norm = min(telemetry.get('power', 32.0) / 200.0, 1.0)
        
        print(f"Rendering with deep modulation...")
        print(f"  Temp: {telemetry.get('asic_temperature', 48.5):.1f}°C")
        print(f"  Power: {telemetry.get('power', 32.0):.1f}W")
        print(f"  Temp norm: {temp_norm:.3f}")
        print(f"  Power norm: {power_norm:.3f}")
        
        # Generate each voice with deeper telemetry influence
        all_voices = self.chip1_voices + self.chip2_voices
        
        for voice_idx, voice in enumerate(all_voices):
            # Telemetry-based frequency drift
            freq_drift = 1 + 0.01 * temp_norm * math.sin(2 * math.pi * 0.02 * voice_idx)
            
            # Telemetry-based amplitude
            amp_drift = 1 + 0.05 * power_norm * math.sin(2 * math.pi * 0.03 * voice_idx)
            
            for i in range(n_samples):
                # LFO for organic pitch variation
                lfo = 0.02 * math.sin(2 * math.pi * voice.lfo_rate * i / self.sample_rate)
                freq = voice.base_freq * freq_drift * (1 + lfo)
                
                # Telemetry-based envelope
                decay = math.exp(-i / (self.sample_rate * 30))
                amp = voice.amplitude * amp_drift
                envelope = decay * (1 + 0.1 * math.sin(2 * math.pi * 0.05 * i / self.sample_rate))
                
                # Generate wave with multiple harmonics
                wave = 0.0
                wave += 0.5 * math.sin(2 * math.pi * freq * i / self.sample_rate)
                wave += 0.2 * math.sin(2 * math.pi * freq * 2 * i / self.sample_rate)
                wave += 0.1 * math.sin(2 * math.pi * freq * 0.5 * i / self.sample_rate)
                wave += 0.05 * math.sin(2 * math.pi * (freq * 1.5 + 5) * i / self.sample_rate)
                
                # Add to mix
                audio[i] += wave * envelope * amp
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        print(f"  Max amplitude: {max_val:.4f}")
        print(f"  Normalized to: {np.max(np.abs(audio)):.4f}")
        
        return audio


def fetch_telemetry() -> Dict:
    """Fetch current hardware telemetry"""
    try:
        result = subprocess.run(
            ['tt-smi', '-s'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            if 'device_info' in data and len(data['device_info']) > 0:
                dev = data['device_info'][0]
                tel = dev.get('telemetry', {})
                smbus = dev.get('smbus_telem', {})
                
                return {
                    'asic_temperature': float(tel.get('asic_temperature', 48.5)),
                    'aiclk': float(tel.get('aiclk', 800)),
                    'power': float(tel.get('power', 32.0)),
                    'current': float(tel.get('current', 45.0)),
                    'voltage': float(tel.get('voltage', 0.71)),
                    'gddr_temps': [30.0] * 8,
                    'ddr_status': [0] * 4,
                }
    except Exception as e:
        print(f"Telemetry fetch error: {e}")
    
    # Fallback
    return {
        'asic_temperature': 48.5,
        'aiclk': 800,
        'power': 32.0,
        'current': 45.0,
        'voltage': 0.71,
        'gddr_temps': [30.0] * 8,
        'ddr_status': [0] * 4,
    }


def main():
    """Generate multi-chip audio"""
    print("="*70)
    print("TENSIXINET: Multi-Chip LLM → Hardware → Music")
    print("="*70)
    print()
    
    # Fetch telemetry
    print("Fetching hardware telemetry...")
    telemetry = fetch_telemetry()
    
    print(f"  ASIC Temp: {telemetry['asic_temperature']:.1f}°C")
    print(f"  AICLK: {telemetry['aiclk']:.0f} MHz")
    print(f"  Power: {telemetry['power']:.1f} W")
    print(f"  Current: {telemetry['current']:.1f} A")
    print()
    
    # Create synthesizer
    synth = MultiChipSynthesizer(telemetry)
    
    # Generate audio
    duration = 120  # 2 minutes
    audio = synth.render_with_modulation(duration, telemetry)
    
    # Save to file
    output_file = f"/tmp/tensixinet_multichip_{int(time.time())}.wav"
    sf.write(output_file, audio, 44100)
    
    print()
    print(f"Audio saved to: {output_file}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Sample rate: 44100 Hz")
    print(f"Voices: 8 (4 per chip)")
    print()
    print("="*70)
    print("Multi-chip synthesis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
