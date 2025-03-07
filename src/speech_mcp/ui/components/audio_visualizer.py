"""
Audio visualizer component for the Speech UI.

This module provides a PyQt widget for visualizing audio levels and waveforms.
"""

import time
import math
import logging
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
    
    def set_active(self, active):
        """Set the visualizer as active or inactive."""
        self.active = active
        self.update()
    
    def update_level(self, level):
        """Update with a new audio level."""
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
        phase = self.animation_time % (2 * math.pi)
        
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