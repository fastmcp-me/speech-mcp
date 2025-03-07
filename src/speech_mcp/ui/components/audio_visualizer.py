"""
Audio visualizer component for the Speech UI.

This module provides a PyQt widget for visualizing audio levels and waveforms.
"""

import time
import math
import logging
import random
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QLinearGradient

# Setup logging
logger = logging.getLogger(__name__)

class AudioVisualizer(QWidget):
    """
    Widget for visualizing audio levels and waveforms.
    """
    def __init__(self, parent=None, mode="user", width_factor=1.0):
        """
        Initialize the audio visualizer.
        
        Args:
            parent: Parent widget
            mode: "user" or "agent" to determine color scheme
            width_factor: Factor to adjust the width of bars (1.0 = full width, 0.5 = half width)
        """
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.audio_levels = [0.0] * 50  # Store recent audio levels
        self.setStyleSheet("background-color: #1e1e1e;")
        self.mode = mode
        self.width_factor = width_factor
        self.active = False  # Track if visualizer is active
        
        # Set colors based on mode
        if self.mode == "user":
            self.bar_color = QColor(0, 200, 255, 180)  # Blue for user
            self.glow_color = QColor(0, 120, 255, 80)  # Softer blue glow
        else:
            self.bar_color = QColor(0, 255, 100, 200)  # Brighter green for agent
            self.glow_color = QColor(0, 220, 100, 100)  # Stronger green glow
            
        # Inactive colors (grey)
        self.inactive_bar_color = QColor(100, 100, 100, 120)  # Grey for inactive
        self.inactive_glow_color = QColor(80, 80, 80, 60)  # Softer grey glow
        
        # Add a smoothing factor to make the visualization less jumpy
        self.smoothing_factor = 0.3
        self.last_level = 0.0
        
        # Timer for animation
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(30)  # Update at ~30fps
        
        # Animation time for dynamic effects
        self.animation_time = 0.0
        
        # Pre-recorded animation patterns for agent mode
        if self.mode == "agent":
            self.initialize_prerecorded_patterns()
            self.current_pattern = "wave"  # Default pattern
            
            # For agent mode, we need to ensure continuous updates
            # This is a backup timer in case the main animation timer stops
            self.agent_update_timer = QTimer(self)
            self.agent_update_timer.timeout.connect(self.continuous_update)
            self.agent_update_timer.start(100)  # Backup timer at 10fps
    
    def continuous_update(self):
        """Ensure continuous updates for agent visualization"""
        if self.mode == "agent" and self.active:
            self.update()  # Force a repaint
    
    def initialize_prerecorded_patterns(self):
        """Initialize pre-recorded animation patterns for agent visualization"""
        # Create different animation patterns
        self.patterns = {
            "wave": self.generate_wave_pattern(),
            "pulse": self.generate_pulse_pattern(),
            "bounce": self.generate_bounce_pattern()
        }
        self.pattern_index = 0
    
    def generate_wave_pattern(self):
        """Generate a smooth wave pattern"""
        pattern = []
        # Create a smooth sine wave pattern
        steps = 40
        for i in range(steps):
            # Sine wave with varying amplitude
            angle = (i / steps) * (2 * math.pi)
            level = 0.3 + 0.4 * math.sin(angle)
            pattern.append(level)
        return pattern
    
    def generate_pulse_pattern(self):
        """Generate a pulsing pattern"""
        pattern = []
        # Create a pulsing pattern
        steps = 30
        for i in range(steps):
            # Pulse wave (higher in middle)
            position = i / steps
            if position < 0.5:
                level = 0.3 + 1.2 * position  # Rising
            else:
                level = 0.3 + 1.2 * (1.0 - position)  # Falling
            pattern.append(level * 0.7)  # Scale to appropriate range
        return pattern
    
    def generate_bounce_pattern(self):
        """Generate a bouncing pattern"""
        pattern = []
        # Create a bouncing pattern
        steps = 25
        for i in range(steps):
            # Bouncing effect
            position = i / steps
            level = 0.7 - 0.6 * abs(math.sin(position * math.pi))
            pattern.append(level)
        return pattern
    
    def set_active(self, active):
        """Set the visualizer as active or inactive."""
        self.active = active
        if active and self.mode == "agent":
            # Reset pattern index when activating
            self.pattern_index = 0
            # Randomly select a pattern
            patterns = list(self.patterns.keys())
            self.current_pattern = random.choice(patterns)
        self.update()
    
    def update_level(self, level):
        """Update with a new audio level."""
        if self.mode == "agent" and hasattr(self, 'patterns'):
            # For agent mode, we use pre-recorded patterns instead of audio input
            # This is only called to trigger animation updates
            self.animation_time += 0.1
            return
            
        # For user mode, we still use the audio input
        # Apply smoothing to avoid abrupt changes
        smoothed_level = (level * (1.0 - self.smoothing_factor)) + (self.last_level * self.smoothing_factor)
        self.last_level = smoothed_level
        
        # For center-rising visualization, we just need to update the current level
        # We'll shift all values in paintEvent
        self.audio_levels.pop(0)
        self.audio_levels.append(smoothed_level)
        
        # Update animation time
        self.animation_time += 0.1
    
    def paintEvent(self, event):
        """Draw the audio visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(event.rect(), QColor(30, 30, 30))
        
        # Draw waveform
        width = self.width()
        height = self.height()
        mid_height = height / 2
        
        # Choose colors based on active state
        if self.active:
            bar_color = self.bar_color
            glow_color = self.glow_color
        else:
            bar_color = self.inactive_bar_color
            glow_color = self.inactive_glow_color
        
        # Set pen for waveform
        pen = QPen(bar_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw the center-rising waveform
        # We'll draw bars at different positions with heights based on audio levels
        bar_count = 40  # Number of bars to draw
        bar_width = (width / bar_count) * self.width_factor
        bar_spacing = 2  # Pixels between bars
        
        # Calculate animation phase for dynamic effects
        # Use system time to ensure continuous animation even if update calls are irregular
        phase = time.time() % (2 * math.pi)
        
        # Special handling for agent mode with pre-recorded patterns
        if self.mode == "agent" and hasattr(self, 'patterns') and self.active:
            pattern = self.patterns[self.current_pattern]
            pattern_length = len(pattern)
            
            # Use time-based animation instead of incrementing an index
            # This ensures smooth animation even if paintEvent calls are delayed
            time_index = int((time.time() * 10) % pattern_length)
            
            # Draw bars using the pre-recorded pattern
            for i in range(bar_count):
                # Calculate pattern index with wrapping
                pattern_idx = (time_index + i) % pattern_length
                base_level = pattern[pattern_idx]
                
                # Add subtle wave effect to make visualization more dynamic
                wave_effect = 0.05 * math.sin(phase + i * 0.2)
                level = max(0.0, min(1.0, base_level + wave_effect))
                
                # Calculate bar height based on level
                bar_height = level * mid_height * 0.95
                
                # Calculate x position (centered)
                x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
                x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
                
                # Draw the bar (right side)
                if i < bar_count / 2:
                    # Draw glow effect first (larger, more transparent)
                    glow_rect = QRectF(
                        x - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect, QColor(
                        glow_color.red(), 
                        glow_color.green(), 
                        glow_color.blue(), 
                        80 - i * 2
                    ))
                    
                    # Draw the main bar
                    rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                    painter.fillRect(rect, QColor(
                        bar_color.red(), 
                        bar_color.green(), 
                        bar_color.blue(), 
                        180 - i * 3
                    ))
                
                # Draw the mirrored bar (left side)
                if i < bar_count / 2:
                    # Draw glow effect first
                    glow_rect_mirror = QRectF(
                        x_mirror - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect_mirror, QColor(
                        glow_color.red(), 
                        glow_color.green(), 
                        glow_color.blue(), 
                        80 - i * 2
                    ))
                    
                    # Draw the main bar
                    rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                    painter.fillRect(rect_mirror, QColor(
                        bar_color.red(), 
                        bar_color.green(), 
                        bar_color.blue(), 
                        180 - i * 3
                    ))
                    
            # Request another update to keep the animation going
            self.update()
        else:
            # Original code for user mode or inactive agent mode
            for i in range(bar_count):
                # Calculate the position in the audio_levels array
                # Center bars use the most recent values
                if i < len(self.audio_levels):
                    # For bars in the middle, use the most recent levels
                    level_idx = len(self.audio_levels) - 1 - i
                    if level_idx >= 0:
                        level = self.audio_levels[level_idx]
                    else:
                        level = 0.0
                else:
                    level = 0.0
                    
                # If inactive, flatten the visualization
                if not self.active:
                    level = level * 0.2  # Reduce height significantly when inactive
                    
                # Add subtle wave effect to make visualization more dynamic
                wave_effect = 0.05 * math.sin(phase + i * 0.2)
                level = max(0.0, min(1.0, level + wave_effect))
                
                # Calculate bar height based on level
                bar_height = level * mid_height * 0.95
                
                # Calculate x position (centered)
                x = (width / 2) + (i * bar_width / 2) - (bar_width / 2)
                x_mirror = (width / 2) - (i * bar_width / 2) - (bar_width / 2)
                
                # Draw the bar (right side)
                if i < bar_count / 2:
                    # Draw glow effect first (larger, more transparent)
                    glow_rect = QRectF(
                        x - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect, QColor(
                        glow_color.red(), 
                        glow_color.green(), 
                        glow_color.blue(), 
                        80 - i * 2
                    ))
                    
                    # Draw the main bar
                    rect = QRectF(x, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                    painter.fillRect(rect, QColor(
                        bar_color.red(), 
                        bar_color.green(), 
                        bar_color.blue(), 
                        180 - i * 3
                    ))
                
                # Draw the mirrored bar (left side)
                if i < bar_count / 2:
                    # Draw glow effect first
                    glow_rect_mirror = QRectF(
                        x_mirror - bar_width * 0.2, 
                        mid_height - bar_height * 1.1, 
                        (bar_width - bar_spacing) * 1.4, 
                        bar_height * 2.2
                    )
                    painter.fillRect(glow_rect_mirror, QColor(
                        glow_color.red(), 
                        glow_color.green(), 
                        glow_color.blue(), 
                        80 - i * 2
                    ))
                    
                    # Draw the main bar
                    rect_mirror = QRectF(x_mirror, mid_height - bar_height, bar_width - bar_spacing, bar_height * 2)
                    painter.fillRect(rect_mirror, QColor(
                        bar_color.red(), 
                        bar_color.green(), 
                        bar_color.blue(), 
                        180 - i * 3
                    ))
        
        # Draw a thin center line with a gradient
        gradient = QLinearGradient(0, mid_height, width, mid_height)
        gradient.setColorAt(0, QColor(100, 100, 100, 0))
        gradient.setColorAt(0.5, QColor(100, 100, 100, 100))
        gradient.setColorAt(1, QColor(100, 100, 100, 0))
        
        painter.setPen(QPen(gradient, 1))
        painter.drawLine(0, int(mid_height), width, int(mid_height))