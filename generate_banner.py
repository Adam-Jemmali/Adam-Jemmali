#!/usr/bin/env python3
"""
Phase 1: GitHub Profile Banner Generator
Generates animated dithered portrait banner with info panel
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from typing import List, Tuple

class BannerGenerator:
    def __init__(self, photo_path: str, username: str = "Adam-Jemmali"):
        self.username = username
        self.width = 1180
        self.height = 610
        self.photo = self.load_image(photo_path)
        self.dither_size = (395, 205)

        # Color palette
        self.colors = {
            'dark_portrait': '#A78BFA',
            'light_portrait': '#7C3AED',
            'dark_chrome': '#22D3EE',
            'light_chrome': '#0891B2',
            'accent': '#10B981',
            'background': '#0A101F',
        }

    def load_image(self, path: str) -> Image.Image:
        """Load image from file"""
        img = Image.open(path)
        return img

    def crop_portrait(self) -> Image.Image:
        """Crop head + shoulders with improved silhouette detection"""
        img = self.photo.convert('RGB')
        w, h = img.size

        # Convert to HSV for better color-based segmentation
        img_hsv = img.convert('HSV')
        h_arr, s_arr, v_arr = [np.array(c, dtype=np.float32) for c in img_hsv.split()]

        # Convert RGB to LAB-like lightness for better person detection
        img_rgb = np.array(img, dtype=np.float32)
        R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        # Lightness channel (perceptual brightness)
        lightness = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0

        # Skin detection heuristic: moderate lightness, not too extreme
        # Avoid pure blue backgrounds (low R and B relative to G)
        bg_likelihood = np.zeros_like(lightness)

        # Very bright pixels = likely background light
        bg_likelihood += (lightness > 0.85).astype(float)

        # Very dark pixels = likely background shadow
        bg_likelihood += (lightness < 0.15).astype(float)

        # Strong blue (high B relative to R) = likely blue background
        bg_likelihood += ((B > R + 20) & (lightness > 0.3)).astype(float)

        # Moderate lightness (skin-like) = foreground
        fg_mask = (lightness > 0.2) & (lightness < 0.85) & (bg_likelihood == 0)

        # Morphological cleanup: remove noise, fill holes
        fg_mask = ndimage.binary_opening(fg_mask, iterations=2)
        fg_mask = ndimage.binary_fill_holes(fg_mask)
        fg_mask = ndimage.binary_closing(fg_mask, iterations=3)

        # Find the largest connected component (the person)
        labeled, num = ndimage.label(fg_mask)
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            largest_label = np.argmax(sizes[1:]) + 1
            fg_mask = labeled == largest_label

        # Find bounding box with generous padding
        coords = np.where(fg_mask)
        if len(coords[0]) > 100:  # Ensure we have a substantial foreground
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # Generous padding: 25% above, 10% below sides
            height_fg = y_max - y_min
            width_fg = x_max - x_min

            y_min = max(0, int(y_min - height_fg * 0.25))
            y_max = min(h, int(y_max + height_fg * 0.15))
            x_min = max(0, int(x_min - width_fg * 0.15))
            x_max = min(w, int(x_max + width_fg * 0.15))

            img = img.crop((x_min, y_min, x_max, y_max))

        # Resize to dither size
        return img.resize(self.dither_size, Image.Resampling.LANCZOS)

    def apply_contrast(self, img: Image.Image) -> Image.Image:
        """Apply 1.3x contrast with autocontrast"""
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
        return img

    def floyd_steinberg_dither(self, img: Image.Image) -> np.ndarray:
        """1-bit Floyd-Steinberg dithering with serpentine order"""
        img_array = np.array(img.convert('L'), dtype=np.float32) / 255.0
        h, w = img_array.shape

        output = np.zeros((h, w), dtype=np.uint8)

        for y in range(h):
            if y % 2 == 0:  # Left to right
                x_range = range(w)
            else:  # Right to left (serpentine)
                x_range = range(w - 1, -1, -1)

            for x in x_range:
                old_val = img_array[y, x]
                new_val = 1.0 if old_val > 0.5 else 0.0
                output[y, x] = int(new_val * 255)

                error = old_val - new_val

                # Distribute error
                if y % 2 == 0:  # LTR
                    if x + 1 < w:
                        img_array[y, x + 1] += error * 7 / 16
                    if y + 1 < h:
                        if x - 1 >= 0:
                            img_array[y + 1, x - 1] += error * 3 / 16
                        img_array[y + 1, x] += error * 5 / 16
                        if x + 1 < w:
                            img_array[y + 1, x + 1] += error * 1 / 16
                else:  # RTL
                    if x - 1 >= 0:
                        img_array[y, x - 1] += error * 7 / 16
                    if y + 1 < h:
                        if x + 1 < w:
                            img_array[y + 1, x + 1] += error * 3 / 16
                        img_array[y + 1, x] += error * 5 / 16
                        if x - 1 >= 0:
                            img_array[y + 1, x - 1] += error * 1 / 16

        return output

    def dither_to_dots(self, dither: np.ndarray, dot_size: int = 3, skip: int = 1) -> List[Tuple[float, float]]:
        """Convert dithered bitmap to dot coordinates"""
        h, w = dither.shape
        dots = []

        for y in range(0, h, skip):
            for x in range(0, w, skip):
                if dither[y, x] > 128:
                    dots.append((x * dot_size, y * dot_size))

        return dots

    def generate_info_panel(self) -> str:
        """Generate info panel text - bold, centered, multi-line for full visibility"""
        lines = []
        y = 120

        # Name
        lines.append(f'    <text x="{self.width // 2}" y="{y}" text-anchor="middle" font-size="16" class="banner-text" font-family="monospace">Name: Adam Jemmali</text>')
        y += 35

        # Role
        lines.append(f'    <text x="{self.width // 2}" y="{y}" text-anchor="middle" font-size="14" class="banner-text" font-family="monospace">CV | Agentic AI | Sports EdTech</text>')
        y += 32

        # Stack - split into multiple lines
        stack_line1 = "Next.js, Tailwind, FastAPI, AWS, Supabase, Drizzle, Sentry, Stripe"
        stack_line2 = "Claude, Postgres FTS, Markdown, GitHub Actions, PostHog, Resend, Gemini, Make.com, n8n"

        lines.append(f'    <text x="{self.width // 2}" y="{y}" text-anchor="middle" font-size="12" class="banner-text" font-family="monospace">Stack: {stack_line1}</text>')
        y += 28
        lines.append(f'    <text x="{self.width // 2}" y="{y}" text-anchor="middle" font-size="12" class="banner-text" font-family="monospace">{stack_line2}</text>')

        return '\n'.join(lines)

    def generate_dot_paths(self, dots: List[Tuple[float, float]]) -> str:
        """Generate SVG circle elements for dots - full rectangle coverage"""
        dot_elements = []
        for i, (x, y) in enumerate(dots):
            dot_elements.append(f'      <circle cx="{x}" cy="{y}" r="1.5" class="dot"/>')

        return '\n'.join(dot_elements)

    def generate_svg(self, dots: List[Tuple[float, float]]) -> str:
        """Generate banner SVG"""
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}">
  <defs>
    <style>
      @media (prefers-color-scheme: dark) {{
        :root {{ background: {self.colors['background']}; }}
      }}
      .dot {{
        fill: {self.colors['dark_portrait']};
        shape-rendering: crispEdges;
      }}
      .banner-text {{
        fill: #FFFFFF;
        font-weight: 900;
        text-shadow: 0 0 4px rgba(0,0,0,0.8);
        animation: pulse-text 2s ease-in-out infinite;
      }}
      @keyframes pulse-text {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
      }}
    </style>
  </defs>

  <!-- Background - Purple base for dither effect -->
  <rect width="{self.width}" height="{self.height}" fill="{self.colors['light_portrait']}"/>

  <!-- Portrait (left side) -->
  <g id="visual-map">

    <!-- Dithered portrait dots -->
    <g id="portrait-dots">
{self.generate_dot_paths(dots)}
    </g>

    <!-- Info Panel (inside/below portrait) -->
{self.generate_info_panel()}
  </g>

  <!-- Full-width Handle Pill -->
  <rect x="0" y="555" width="{self.width}" height="55" fill="{self.colors['accent']}" rx="20"/>
  <text x="{self.width // 2}" y="590" text-anchor="middle" font-size="14" fill="#000" font-family="monospace" font-weight="bold">@{self.username}</text>
</svg>'''

        return svg

    def generate(self) -> Tuple[str, str]:
        """Generate both dark and light SVG banners"""
        # Process image
        portrait = self.crop_portrait()
        portrait = self.apply_contrast(portrait)

        # Dither
        dither = self.floyd_steinberg_dither(portrait)
        dots = self.dither_to_dots(dither, skip=1)

        # Generate SVG
        svg = self.generate_svg(dots)

        return svg, svg

def main():
    photo_path = "pfp.png"

    print("Loading image from: pfp.png")

    generator = BannerGenerator(photo_path, username="Adam-Jemmali")
    dark_svg, light_svg = generator.generate()

    # Save SVGs
    with open("dark-banner.svg", "w") as f:
        f.write(dark_svg)
    print("[OK] Generated: dark-banner.svg")

    with open("light-banner.svg", "w") as f:
        f.write(light_svg)
    print("[OK] Generated: light-banner.svg")

    print("\n[SUCCESS] Banner SVGs created successfully!")
    print("[INFO] Files: dark-banner.svg, light-banner.svg")

if __name__ == "__main__":
    main()
