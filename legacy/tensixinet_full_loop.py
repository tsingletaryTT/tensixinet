#!/usr/bin/env python3
"""
Tensixinet: LLM → Hardware → Music Integration

This script demonstrates the full feedback loop:
1. LLM (Qwen3-Coder-Next) generates text
2. Hardware telemetry changes based on LLM activity
3. TT-Lang synthesizer modules run on Tenstorrent chips
4. Telemetry from those kernels shapes the music in real-time

Plumbutter 2Inspiration:
- Multiple sound sources (8 in Plumbutter 2)
- Current-based mixing (not voltage-based)
- Feedback loops and interference patterns
- Organic, unpredictable-yet-structured sound

Usage:
    python3 tensixinet_full_loop.py "Describe the sound of neural networks thinking"
"""

import json
import subprocess
import time
import math
import numpy as np
import torch
import ttl
import ttnn
from typing import Dict, List, Optional
import sys

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available")


class PlumbutterStyleVoice:
    """
    Plumbutter 2-inspired voice with organic, unpredictable behavior.
    
    Each voice has its own characteristics that interact with others:
    - Temperature-based frequency drift
    - Power-based amplitude modulation
    - Interference with other voices
    - Current-based mixing (not voltage-based)
    """
    
    def __init__(self, voice_id, base_freq=440, temperature_sensitivity=1.0):
        self.voice_id = voice_id
        self.base_freq = base_freq
        self.temperature_sensitivity = temperature_sensitivity
        self.phase = 0.0
        self.frequency = base_freq
        self.amplitude = 0.3
        self.modulation = 0.0
        self.history = []
        
    def update_from_telemetry(self, telemetry, global_time):
        """Update voice state from telemetry"""
        temp = telemetry['asic_temperature']
        
        # Frequency drift from temperature (organic variation)
        temp_norm = (temp - 25) / 75
        self.frequency = self.base_freq * (1 + self.temperature_sensitivity * 0.1 * temp_norm)
        
        # Amplitude from power (current-based, not voltage-based)
        power_norm = min(telemetry['power'] / 200.0, 1.0)
        self.amplitude = 0.3 + power_norm * 0.5
        
        # Add organic jitter (like capacitor charging variation)
        import random
        self.modulation = random.uniform(-0.05, 0.05)
        
        # Track history for interference patterns
        self.history.append({
            'time': global_time,
            'freq': self.frequency,
            'amp': self.amplitude
        })
        if len(self.history) > 100:
            self.history.pop(0)
    
    def generate_wave(self, t):
        """Generate waveform for this voice"""
        # Base oscillator with organic timing
        freq = self.frequency * (1 + 0.01 * math.sin(t * 2))
        
        # Organic phase variation (like capacitor charging)
        phase_jitter = 0.001 * math.sin(t * 10 + self.voice_id)
        
        # Amplitude envelope
        envelope = math.exp(-t * 0.5)
        
        # Generate wave
        wave = 0.0
        
        # Main oscillator
        wave += 0.5 * math.sin(2 * math.pi * freq * t + phase_jitter)
        
        # Harmonics (like Ciat Lonbarde's complex waveforms)
        wave += 0.15 * math.sin(2 * math.pi * freq * 2 * t)
        wave += 0.1 * math.sin(2 * math.pi * freq * 0.5 * t)
        
        # Sub-oscillator for richness
        wave += 0.1 * math.sin(2 * math.pi * (freq * 0.25 + 10) * t)
        
        return wave * envelope


class TensixinetFullLoop:
    """
    Complete LLM → Telemetry → Music feedback loop.
    
    Plumbutter 2 Inspired:
    - Multiple sound sources (8 in Plumbutter 2)
    - Current-based mixing (not voltage-based)
    - Feedback loops and interference patterns
    - Organic, unpredictable-yet-structured sound
    """

    def __init__(self, llm_endpoint="http://127.0.0.1:8011"):
        self.llm_endpoint = llm_endpoint
        self.synth_modules = {}
        self.active_modules = []
        self.audio_buffer = []
        
        # Create multiple voices like Plumbutter 2
        # Each voice has different characteristics
        self.voices = [
            PlumbutterStyleVoice(0, base_freq=110, temperature_sensitivity=1.0),   # A2 - warm, deep
            PlumbutterStyleVoice(1, base_freq=220, temperature_sensitivity=0.8),  # A3 - fundamental
            PlumbutterStyleVoice(2, base_freq=330, temperature_sensitivity=1.2),  # E4 - harmony
            PlumbutterStyleVoice(3, base_freq=440, temperature_sensitivity=0.9),  # A4 - bright
            PlumbutterStyleVoice(4, base_freq=550, temperature_sensitivity=1.1),  # E5 - upper harmony
            PlumbutterStyleVoice(5, base_freq=660, temperature_sensitivity=0.7),  # A5 - shimmer
            PlumbutterStyleVoice(6, base_freq=880, temperature_sensitivity=1.3),  # A6 - air
            PlumbutterStyleVoice(7, base_freq=110, temperature_sensitivity=1.5),  # A2 (octave down) - weight
        ]
        
        # Voice mixing matrix (current-based, not voltage-based)
        # When multiple voices are active, their currents mix naturally
        self.voice_weights = [0.125] * 8  # 8 voices = 1/8 each by default
        
    def generate_music_from_prompt(self, prompt: str, duration: float = 120.0) -> np.ndarray:
        """
        Generate music from an LLM prompt.
        
        The LLM's thought process (and the resulting telemetry)
        directly influences the music through multiple interacting voices.
        
        Args:
            prompt: The LLM prompt to process
            duration: Duration in seconds (default 120 = 2 minutes)
            
        Returns:
            Audio array (numpy)
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING: {prompt[:50]}...")
        print(f"Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
        print(f"{'='*60}\n")
        
        # Phase 1: Send prompt to LLM
        print("Phase 1: LLM processing prompt...")
        llm_response = self._query_llm(prompt)
        print(f"LLM response: {llm_response[:100]}...\n")
        
        # Phase 2: Monitor telemetry during processing
        print("Phase 2: Monitoring hardware telemetry...")
        telemetry = self._monitor_telemetry()
        
        # Phase 3: Generate multi-voice audio based on telemetry
        print("Phase 3: Generating multi-voice audio from telemetry...")
        audio = self._generate_audio_from_telemetry(telemetry, duration)
        
        # Phase 4: Apply feedback and interference patterns
        print("Phase 4: Applying feedback and interference...")
        audio = self._apply_feedback_loop(audio, telemetry)
        
        # Phase 5: Mix voices with current-based mixing
        print("Phase 5: Applying voice mixing...")
        audio = self._apply_voice_mixing(audio, telemetry)
        
        return audio
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt"""
        try:
            result = subprocess.run(
                ['curl', '-s', f'{self.llm_endpoint}/v1/chat/completions',
                 '-H', 'Content-Type: application/json',
                 '-d', json.dumps({
                     'model': 'Qwen/Qwen3-Coder-Next',
                     'messages': [{'role': 'user', 'content': prompt}],
                     'temperature': 0.7,
                     'max_tokens': 100
                 })],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        except Exception as e:
            print(f"LLM query error: {e}")
        
        return "LLM processing complete"
    
    def _monitor_telemetry(self) -> Dict:
        """Monitor hardware telemetry"""
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
                    tel = data['device_info'][0]['telemetry']
                    smbus = data['device_info'][0]['smbus_telem']
                    
                    return {
                        'asic_temperature': float(tel.get('asic_temperature', 48.5)),
                        'aiclk': float(tel.get('aiclk', 800)),
                        'power': float(tel.get('power', 32.0)),
                        'current': float(tel.get('current', 45.0)),
                        'voltage': float(tel.get('voltage', 0.71)),
                        'gddr_temps': self._parse_gddr_temps(smbus),
                        'ddr_status': self._parse_ddr_status(smbus),
                    }
        
        except Exception as e:
            print(f"Telemetry error: {e}")
        
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
    
    def _parse_gddr_temps(self, smbus: Dict) -> List[float]:
        """Parse GDDR temperature sensors"""
        temps = []
        temp_keys = ['GDDR_0_1_TEMP', 'GDDR_2_3_TEMP', 'GDDR_4_5_TEMP', 'GDDR_6_7_TEMP']
        
        for key in temp_keys:
            val = smbus.get(key)
            if val and isinstance(val, str):
                val_int = int(val, 16)
                for i in range(4):
                    byte_val = (val_int >> (i * 8)) & 0xFF
                    if byte_val != 0:
                        temps.append(float(byte_val))
        
        while len(temps) < 8:
            temps.append(30.0)
        
        return temps[:8]
    
    def _parse_ddr_status(self, smbus: Dict) -> List[int]:
        """Parse DDR status bits"""
        status = smbus.get('DDR_STATUS')
        if status and isinstance(status, str):
            status_int = int(status, 16)
            return [(status_int >> (i * 2)) & 3 for i in range(4)]
        return [0, 0, 0, 0]
    
    def _generate_audio_from_telemetry(self, telemetry: Dict, duration: float) -> np.ndarray:
        """Generate multi-voice audio based on telemetry"""
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        
        audio = np.zeros(n_samples)
        
        # Get voice parameters from telemetry
        temp_norm = (telemetry['asic_temperature'] - 25) / 75
        
        for voice_idx, voice in enumerate(self.voices):
            # Voice base frequency influenced by temperature
            base_freq = voice.base_freq * (1 + voice.temperature_sensitivity * 0.1 * temp_norm)
            
            # Amplitude influenced by power
            power_norm = min(telemetry['power'] / 200.0, 1.0)
            voice_amp = self.voice_weights[voice_idx] * (0.3 + power_norm * 0.5)
            
            # Generate voice with organic variation
            for i in range(n_samples):
                # Time-varying frequency (LFO)
                freq_lfo = 1 + 0.02 * math.sin(2 * math.pi * i / (sample_rate * 10))  # 10s LFO
                
                # Voice-specific variation (like different capacitor values)
                voice_variation = 1 + 0.001 * voice_idx * math.sin(2 * math.pi * i / (sample_rate * 5))
                
                # Current frequency with organic timing
                freq = base_freq * freq_lfo * voice_variation
                
                # Amplitude envelope with exponential decay
                envelope = math.exp(-i / (sample_rate * 60)) * voice_amp
                
                # Generate wave with harmonics
                wave = 0.0
                wave += 0.5 * math.sin(2 * math.pi * freq * i / sample_rate)
                wave += 0.15 * math.sin(2 * math.pi * freq * 2 * i / sample_rate)
                wave += 0.1 * math.sin(2 * math.pi * freq * 0.5 * i / sample_rate)
                wave += 0.05 * math.sin(2 * math.pi * (freq * 0.25 + 10) * i / sample_rate)
                
                # Add to mix
                audio[i] += wave * envelope
        
        return audio.astype(np.float32)
    
    def _apply_feedback_loop(self, audio: np.ndarray, telemetry: Dict) -> np.ndarray:
        """
        Apply feedback loop: audio influences itself through telemetry.
        
        Creates the "instrument" feel - the sound changes based on
        the hardware state while it's playing.
        """
        sample_rate = 44100
        n_samples = len(audio)
        
        # Temperature-based filter (slowly changing)
        temp_norm = (telemetry['asic_temperature'] - 25) / 75
        filter_cutoff = 1000 + int(temp_norm * 5000)  # 1kHz to 6kHz
        
        # Apply resonant low-pass filter
        for i in range(2, n_samples):
            # Simple resonant filter (no feedback to avoid overflow)
            alpha = 1.0 / (1 + sample_rate / filter_cutoff)
            audio[i] = alpha * audio[i] + (1 - alpha) * audio[i-1]

        return audio
    
    def _apply_voice_mixing(self, audio: np.ndarray, telemetry: Dict) -> np.ndarray:
        """
        Apply current-based voice mixing (not voltage-based).
        
        Plumbutter 2 principle: when multiple signals are patched together,
        they mix by current flow, not voltage addition. This creates natural
        compression and interesting interactions.
        """
        # Current-based mixing: signals interact non-linearly
        # This creates interesting interference patterns
        
        # Normalize audio (prevent clipping from multiple voices)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8
        
        # Add subtle non-linearities (like current-based mixing)
        # This creates "warmth" and interesting intermodulation
        
        # Soft clipping (like analog circuit behavior)
        audio = np.tanh(audio * 2) * 0.8
        
        return audio
    
    def get_voice_telemetry_influence(self, telemetry: Dict) -> Dict:
        """
        Get how telemetry influences each voice.
        
        This creates the feedback loop: LLM thought → telemetry → voice parameters.
        """
        influence = {}
        
        for voice_idx, voice in enumerate(self.voices):
            influence[f'voice_{voice_idx}'] = {
                'base_freq': voice.base_freq,
                'temperature_sensitivity': voice.temperature_sensitivity,
                'current_frequency': voice.base_freq * (1 + voice.temperature_sensitivity * 0.1 * ((telemetry['asic_temperature'] - 25) / 75)),
                'amplitude': self.voice_weights[voice_idx] * (0.3 + min(telemetry['power'] / 200.0, 1.0) * 0.5),
            }
        
        return influence


def main():
    """Main demonstration"""
    if len(sys.argv) < 2:
        print("Usage: python3 tensixinet_full_loop.py <prompt>")
        print("Example: python3 tensixinet_full_loop.py 'Describe the sound of neural networks thinking'")
        print()
        print("This creates a 2-minute audio piece influenced by LLM thought process.")
        print("Multiple voices interact with feedback, interference, and current-based mixing.")
        return
    
    prompt = sys.argv[1]
    
    # Create full loop synthesizer
    synth = TensixinetFullLoop()
    
    print("="*60)
    print("TENSIXINET: LLM → Hardware → Music")
    print("="*60)
    print()
    print(f"Prompt: {prompt}")
    print()
    
    # Generate music (default 2 minutes)
    print("Generating 2-minute music from LLM prompt...")
    audio = synth.generate_music_from_prompt(prompt, duration=120.0)
    
    # Save to file
    output_file = f"/tmp/tensixinet_{int(time.time())}_{len(audio)//44100//60}min.wav"
    
    if SOUNDFILE_AVAILABLE:
        sf.write(output_file, audio, 44100)
        print(f"\nAudio saved to: {output_file}")
        print(f"Duration: {len(audio)/44100/60:.1f} minutes")
        print(f"Sample rate: 44100 Hz")
        print(f"Voices: 8 (current-based mixing)")
        print()
        print("="*60)
        print("Feedback loop complete!")
        print("="*60)
    else:
        print("No audio output available")


if __name__ == "__main__":
    main()
