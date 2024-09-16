import librosa
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import logging
import threading
import pygame
import os
import tempfile
from PIL import Image, ImageDraw, ImageTk, ImageFilter
from functools import lru_cache
import random
import colorsys

class RedirectText:
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.insert(tk.END, string)
        self.text_space.see(tk.END)

    def flush(self):
        pass


class MusicVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Music Visualizer")

        self.setup_gui()
        self.setup_logging()

        pygame.mixer.init()
        self.playing = False
        self.start_time = 0
        self.end_time = 0
        self.y = None
        self.sr = None
        self.temp_video = None

        # Load flower images
        self.MAX_FLOWERS = 25
        self.flower_images = self.load_flower_images()
        self.flower_particles = []
        self.background_color = self.get_pastel_color(0)  # Initial background color
        self.background_flower_particles = []  # Separate list for background flowers
        self.foreground_flower_particles = []  # Separate list for foreground flowers

    def setup_gui(self):
        # Input file selection
        tk.Label(self.master, text="Input MP3 File:").grid(
            row=0, column=0, sticky="e"
        )
        self.input_entry = tk.Entry(self.master, width=50)
        self.input_entry.grid(row=0, column=1)
        tk.Button(
            self.master, text="Browse", command=self.browse_input_file
        ).grid(row=0, column=2)

        # Output file selection
        tk.Label(self.master, text="Output MP4 File:").grid(
            row=1, column=0, sticky="e"
        )
        self.output_entry = tk.Entry(self.master, width=50)
        self.output_entry.grid(row=1, column=1)
        tk.Button(
            self.master, text="Browse", command=self.browse_output_file
        ).grid(row=1, column=2)

        # Time selection
        tk.Label(self.master, text="Start Time (s):").grid(
            row=2, column=0, sticky="e"
        )
        self.start_time_entry = tk.Entry(self.master, width=10)
        self.start_time_entry.grid(row=2, column=1, sticky="w")

        tk.Label(self.master, text="End Time (s):").grid(
            row=3, column=0, sticky="e"
        )
        self.end_time_entry = tk.Entry(self.master, width=10)
        self.end_time_entry.grid(row=3, column=1, sticky="w")

        # Preview button
        self.preview_button = tk.Button(
            self.master, text="Preview", command=self.preview_audio
        )
        self.preview_button.grid(row=4, column=1)

        # Add option to choose visualization type
        tk.Label(self.master, text="Visualization Type:").grid(row=4, column=0, sticky="e")
        self.viz_type = tk.StringVar(value="Abstract")
        self.viz_type_menu = ttk.Combobox(self.master, textvariable=self.viz_type, values=["Abstract", "Flowers"])
        self.viz_type_menu.grid(row=4, column=1, sticky="w")

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.master, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5
        )

        # Preview frame
        self.preview_frame = tk.Frame(
            self.master, width=500, height=500, bg="black"
        )
        self.preview_frame.grid(row=9, column=0, columnspan=3, padx=5, pady=5)
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack()

        # Progress label
        self.progress_label = tk.Label(self.master, text="")
        self.progress_label.grid(row=6, column=0, columnspan=3)

        # Start processing button
        self.start_button = tk.Button(
            self.master, text="Create Video", command=self.start_processing
        )
        self.start_button.grid(row=7, column=1)

        # Log display
        self.log_display = scrolledtext.ScrolledText(self.master, height=10)
        self.log_display.grid(
            row=8, column=0, columnspan=3, sticky="nsew", padx=5, pady=5
        )

        self.master.grid_rowconfigure(8, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

    def load_flower_images(self):
        flower_images = []
        flower_dir = "new_flowers"
        for filename in os.listdir(flower_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(flower_dir, filename)
                img = Image.open(img_path).convert("RGBA")
                # Resize the image to be larger (e.g., 100x100 pixels)
                img = img.resize((100, 100), Image.LANCZOS)
                flower_images.append(img)
        return flower_images
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.log_redirect = RedirectText(self.log_display)
        logging.getLogger().addHandler(logging.StreamHandler(self.log_redirect))

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("MP3 files", "*.mp3")]
        )
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, filename)
        if filename:
            self.load_audio(filename)
        logging.info(f"Selected input file: {filename}")

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")]
        )
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, filename)
        logging.info(f"Selected output file: {filename}")

    def load_audio(self, filename):
        logging.info(f"Loading audio file: {filename}")
        self.y, self.sr = librosa.load(filename)
        duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.end_time_entry.delete(0, tk.END)
        self.end_time_entry.insert(0, str(duration))
        logging.info(f"Audio loaded. Duration: {duration:.2f} seconds")

    def preview_audio(self):
        if self.playing:
            pygame.mixer.music.stop()
            self.preview_button.config(text="Preview")
            self.playing = False
        else:
            try:
                start_time = float(self.start_time_entry.get())
                end_time = float(self.end_time_entry.get())
                if (
                    start_time >= end_time
                    or start_time < 0
                    or end_time > librosa.get_duration(y=self.y, sr=self.sr)
                ):
                    raise ValueError("Invalid time range")

                # Create a temporary file with the selected portion
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                )
                temp_filename = temp_file.name
                temp_file.close()

                y_selected = self.y[
                    int(start_time * self.sr) : int(end_time * self.sr)
                ]
                librosa.output.write_wav(temp_filename, y_selected, self.sr)

                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                self.preview_button.config(text="Stop")
                self.playing = True

                # Schedule cleanup of temporary file
                self.master.after(
                    int((end_time - start_time) * 1000),
                    self.stop_preview,
                    temp_filename,
                )
            except Exception as e:
                logging.error(f"Error in audio preview: {str(e)}")
                messagebox.showerror(
                    "Error", f"Error in audio preview: {str(e)}"
                )

    def stop_preview(self, temp_filename):
        pygame.mixer.music.stop()
        self.preview_button.config(text="Preview")
        self.playing = False
        try:
            os.remove(temp_filename)
        except Exception as e:
            logging.error(f"Error removing temporary file: {str(e)}")

    def extract_music_features(self, y, sr):
        logging.info("Extracting music features")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return onset_env, chroma

    def generate_abstract_image(
        self, chroma_frame, onset_strength, spectral_contrast, mfcc, t
    ):
        width, height = 500, 500

        if self.viz_type.get() == "Abstract":
            image = Image.new("RGB", (width, height), color="black")
            draw = ImageDraw.Draw(image)

            # Helper function to ensure valid numeric values
            def valid_value(val, default=0.0):
                try:
                    return float(val) if val not in [None, ""] else default
                except (ValueError, TypeError):
                    return default

            # Create a static mandala-style abstract image
            mandala_image = self.generate_mandala_image(width, height)

            # Function to handle ring drawing with overflow behavior (but no reflection)
            def draw_wave_ring(
                center_x,
                center_y,
                base_radius,
                color,
                wave_amplitude,
                wave_frequency,
                music_feature,
                mask,
            ):
                modulated_radius = base_radius + wave_amplitude * np.sin(
                    2 * np.pi * wave_frequency * t * music_feature
                )
                ring_outer = max(0, modulated_radius + 5)  # Outer ring
                ring_inner = max(0, modulated_radius - 5)  # Inner ring

                # Clip the outer/inner ring to boundaries
                draw.ellipse(
                    [
                        (center_x - ring_outer, center_y - ring_outer),
                        (center_x + ring_outer, center_y + ring_outer),
                    ],
                    outline=color,
                    width=2,
                )
                draw.ellipse(
                    [
                        (center_x - ring_inner, center_y - ring_inner),
                        (center_x + ring_inner, center_y + ring_inner),
                    ],
                    outline=color,
                    width=2,
                )

                mask.ellipse(
                    [
                        (center_x - ring_outer, center_y - ring_outer),
                        (center_x + ring_outer, center_y + ring_outer),
                    ],
                    fill=255,
                )
                mask.ellipse(
                    [
                        (center_x - ring_inner, center_y - ring_inner),
                        (center_x + ring_inner, center_y + ring_inner),
                    ],
                    fill=0,
                )

            # Create a mask image for the annulus
            mask = Image.new("L", (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)

            # Validate music feature inputs and assign default values if needed
            onset_strength = valid_value(onset_strength, 0.1)
            chroma_val = valid_value(np.mean(chroma_frame), 0.1)
            mfcc_val = valid_value(np.mean(mfcc), 0.1)

            # Multiple rings with different parameters
            base_radius = onset_strength * (width / 4) + 50
            wave_amplitude = 15
            wave_frequency = 0.5  # Frequency of the waves
            draw_wave_ring(
                width / 2,
                height / 2,
                base_radius,
                "cyan",
                wave_amplitude,
                wave_frequency,
                onset_strength,
                mask_draw,
            )

            base_radius = chroma_val * (width / 5) + 40
            draw_wave_ring(
                width / 2,
                height / 2,
                base_radius,
                "magenta",
                wave_amplitude,
                wave_frequency,
                chroma_val,
                mask_draw,
            )

            base_radius = mfcc_val * (width / 8) + 30
            draw_wave_ring(
                width / 2,
                height / 2,
                base_radius,
                "yellow",
                wave_amplitude,
                wave_frequency,
                mfcc_val,
                mask_draw,
            )

            # Apply the mask to reveal parts of the mandala in the rings
            abstract_revealed = Image.composite(mandala_image, image, mask)
            return np.array(abstract_revealed)

        elif self.viz_type.get() == "Flowers":
            # Update background color smoothly
            target_color = self.get_pastel_color(np.mean(chroma_frame))
            self.background_color = self.interpolate_color(self.background_color, target_color, 0.05)
            
            image = Image.new("RGBA", (width, height), self.background_color + (255,))  # Add alpha channel
            draw = ImageDraw.Draw(image)  # Create draw object for the image

            # Add new flowers with a certain probability
            if random.random() < onset_strength * 0.5:
                self.add_new_flower_particle(width, height, is_background=random.random() < 0.3)  # 30% chance of background

            # Additional flowers on percussion beats
            if self.is_percussion_beat_detected(onset_strength):
                self.throw_beat_flower(width, height)  # Throw extra flower for beat
            
            # Update both background and foreground particles
            self.update_flower_particles(t)

            # Draw background flowers first
            for particle in self.background_flower_particles:
                self.draw_flower_particle(image, particle)

            # Draw foreground flowers on top
            for particle in self.foreground_flower_particles:
                self.draw_flower_particle(image, particle)

            # Remove flowers that have fallen off the screen for both background and foreground
            self.background_flower_particles = [p for p in self.background_flower_particles if p['y'] < height + 50]
            self.foreground_flower_particles = [p for p in self.foreground_flower_particles if p['y'] < height + 50]

            # Convert RGBA to RGB for compatibility
            image = image.convert("RGB")
            
        return np.array(image)
    
    def add_new_flower_particle(self, width, height, is_background=False):
        total_flowers = len(self.background_flower_particles) + len(self.foreground_flower_particles)
        
        if total_flowers >= self.MAX_FLOWERS:
            return  # Don't add more flowers if the limit is reached
        
        # Proceed to add a new flower if below limit
        flower = random.choice(self.flower_images)
        particle = {
            'image': flower,
            'x': random.randint(0, width),
            'y': height,
            'vy': random.uniform(-25, -15),
            'rotation': random.uniform(0, 360),
            'angular_velocity': random.uniform(-5, 5),
            'scale': random.uniform(0.5, 1.5),
            'is_background': is_background
        }
        #self.flower_particles.append(particle)
        if is_background:
            self.background_flower_particles.append(particle)  # Add to background list
        else:
            self.foreground_flower_particles.append(particle)  # Add to foreground list

    def throw_beat_flower(self, width, height):
        # Check if the number of flowers exceeds the MAX_FLOWERS limit
        total_flowers = len(self.background_flower_particles) + len(self.foreground_flower_particles)
        
        if total_flowers >= self.MAX_FLOWERS:
            return  # Don't add more flowers if the limit is reached

        # Add new flower if under the limit
        flower = random.choice(self.flower_images)
        particle = {
            'image': flower,
            'x': random.randint(0, width),
            'y': height,
            'vy': random.uniform(-30, -20),
            'rotation': random.uniform(0, 360),
            'angular_velocity': random.uniform(-5, 5),
            'scale': random.uniform(1.0, 2.0),
            'is_background': False  # Always a foreground particle
        }
        self.foreground_flower_particles.append(particle)

    def is_percussion_beat_detected(self, onset_strength):
        # Basic threshold to detect beats, tune this value as needed
        beat_threshold = 0.8  # You can adjust this threshold
        return onset_strength > beat_threshold

    def update_flower_particles(self, t):
        gravity = 0.5
        
        for particle in self.background_flower_particles:
            particle['vy'] += gravity
            particle['y'] += particle['vy']
            particle['x'] += random.uniform(-1, 1)  # Slight horizontal movement
            particle['rotation'] += particle['angular_velocity']

        for particle in self.foreground_flower_particles:
            particle['vy'] += gravity
            particle['y'] += particle['vy']
            particle['x'] += random.uniform(-1, 1)
            particle['rotation'] += particle['angular_velocity']

    def draw_flower_particle(self, image, particle):
        # Rotate and scale the flower image
        img = particle['image'].rotate(particle['rotation'], expand=True)
        new_size = (int(img.width * particle['scale']), int(img.height * particle['scale']))
        img = img.resize(new_size, Image.LANCZOS)

        # Get flower color (average color from the image)
        flower_color = img.convert('RGB').getpixel((img.width // 2, img.height // 2))
        glow_color = flower_color  # Set the glow color to match the flower

        # Create a glowing outline
        glow_size = (int(new_size[0] * 1.2), int(new_size[1] * 1.2))  # Slightly larger than the flower
        glow_image = Image.new('RGBA', glow_size, (0, 0, 0, 0))  # Transparent image for glow
        glow_img = img.resize(glow_size, Image.LANCZOS)
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=10))  # Apply blur to simulate glow

        # Paste the glow behind the original image
        glow_position = (int(particle['x'] - glow_img.width / 2), int(particle['y'] - glow_img.height / 2))
        glow_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))  # Create an overlay for the glow
        glow_overlay.paste(glow_img, glow_position, glow_img)
        image.alpha_composite(glow_overlay)

        # If the flower is in the background, apply a consistent blur
        if particle['is_background']:
            img = img.filter(ImageFilter.GaussianBlur(radius=3))  # Always blurred for background

        # Calculate position for the flower
        pos = (int(particle['x'] - img.width / 2), int(particle['y'] - img.height / 2))

        # Paste the flower onto the image with alpha support
        image.paste(img, pos, img)

        # Pulsing effect on the glow: Change opacity or size based on the onset strength
        glow_intensity = int(255 * (1 - particle['onset_strength']))  # Stronger onsets create brighter glows
        glow_img.putalpha(glow_intensity)  # Adjust transparency of the glow

    def get_pastel_color(self, hue):
        # Convert hue to pastel color
        r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.7)
        return (int(r * 255), int(g * 255), int(b * 255))

    def interpolate_color(self, color1, color2, t):
        # Smooth color transition
        return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))
    
    def get_background_color(self, chroma_frame, onset_strength):
        # Simple algorithm to determine background color
        # You can make this more sophisticated based on your needs
        hue = np.mean(chroma_frame) * 360
        saturation = 0.5
        value = 1 - (onset_strength / 2)  # Darker for stronger onsets
        return self.hsv_to_rgb(hue / 360, saturation, value)

    def resize_image(self, image, width, height):
        return image.resize((width, height), Image.LANCZOS)
    

    @lru_cache(maxsize=128)
    def generate_mandala_image(self, width, height):
        # Generate a mandala-like abstract image similar to what you uploaded
        mandala_image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(mandala_image)

        num_shapes = 10  # Number of layers of patterns
        center_x, center_y = width // 2, height // 2

        def complementary_color(base_color):
            # Returns a complementary color by rotating hue by 180 degrees
            r, g, b = base_color
            return (255 - r, 255 - g, 255 - b)

        colors = [
            (255, 87, 51),  # Red-Orange
            (80, 200, 120),  # Mint Green
            (137, 207, 240),  # Light Blue
            (255, 223, 186),  # Peach
            (0, 150, 136),  # Teal
        ]

        for i in range(num_shapes):
            radius = width // 4 - i * (width // (2 * num_shapes))
            color = colors[i % len(colors)]  # Rotate colors
            complementary = complementary_color(color)

            x0, y0 = center_x - radius, center_y - radius
            x1, y1 = center_x + radius, center_y + radius

            if x1 >= x0 and y1 >= y0:
                # Draw concentric circles with different geometric patterns
                draw.ellipse(
                    [(x0, y0), (x1, y1)],
                    outline=color,
                    width=2,
                )
            else:
                print(f"Skipping ellipse with invalid coordinates: ({x0}, {y0}), ({x1}, {y1})")

            # Adding inner petals or geometric patterns
            petal_radius = radius // 2
            for angle in range(
                0, 360, 30
            ):  # Drawing petals around the circle
                rad_angle = np.radians(angle)
                petal_x = center_x + int(petal_radius * np.cos(rad_angle))
                petal_y = center_y + int(petal_radius * np.sin(rad_angle))

                draw.polygon(
                    [
                        (center_x, center_y),
                        (petal_x, petal_y),
                        (
                            center_x
                            + int(
                                petal_radius
                                * 0.5
                                * np.cos(rad_angle + np.pi / 12)
                            ),
                            center_y
                            + int(
                                petal_radius
                                * 0.5
                                * np.sin(rad_angle + np.pi / 12)
                            ),
                        ),
                    ],
                    fill=complementary,
                )

        return mandala_image

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (int(v * 255), int(t * 255), int(p * 255))
        if i == 1:
            return (int(q * 255), int(v * 255), int(p * 255))
        if i == 2:
            return (int(p * 255), int(v * 255), int(t * 255))
        if i == 3:
            return (int(p * 255), int(q * 255), int(v * 255))
        if i == 4:
            return (int(t * 255), int(p * 255), int(v * 255))
        if i == 5:
            return (int(v * 255), int(p * 255), int(q * 255))

    def create_video(self, y, sr, output_file, start_time, end_time):
        logging.info(f"Starting video creation process")
        logging.info(f"Output video file: {output_file}")

        try:
            duration = end_time - start_time
            fps = 24
            n_frames = int(duration * fps)

            logging.info(f"Generating {n_frames} frames at {fps} fps")

            hop_length = sr // fps
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)

            frames = []
            for frame_number in range(n_frames):
                progress = int((frame_number / n_frames) * 100)
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Generating frame {frame_number}/{n_frames}")

                t = frame_number / fps
                index = min(frame_number, chroma.shape[1] - 1, len(onset_env) - 1, spectral_contrast.shape[1] - 1, mfcc.shape[1] - 1)

                frame = self.generate_abstract_image(
                    chroma[:, index],
                    onset_env[index],
                    spectral_contrast[:, index],
                    mfcc[:, index],
                    t,
                )

                frames.append(frame)

                if frame_number % 10 == 0:
                    self.update_preview(frame)

            logging.info("Creating video clip")
            clip = mpy.ImageSequenceClip(frames, fps=fps)

            logging.info("Adding audio to video")
            audio = mpy.AudioFileClip(self.input_entry.get()).subclip(start_time, end_time)
            final_clip = clip.set_audio(audio)

            logging.info(f"Writing output file: {output_file}")
            final_clip.write_videofile(output_file, fps=fps, logger=None)

            # Validate the output file
            if not os.path.exists(output_file):
                raise Exception("Output file was not created")

            if os.path.getsize(output_file) == 0:
                raise Exception("Output file is empty")

            logging.info("Video creation complete!")
            self.progress_label.config(text="Video creation complete!")
            messagebox.showinfo(
                "Process Complete", "The video has been created successfully!"
            )
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"An error occurred: {str(e)}\n\nPlease check the log for more details.",
            )

    def extract_music_features(self, y, sr):
        logging.info("Extracting music features")
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        duration = len(y) / sr
        n_chroma_frames = int(duration * 30)
        hop_length = max(1, len(y) // n_chroma_frames)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

        logging.info(
            f"Extracted features: onset_env shape: {onset_env.shape}, chroma shape: {chroma.shape}"
        )
        return onset_env, chroma

    def update_preview(self, frame):
        img = Image.fromarray(frame)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image=img)
        self.preview_label.config(image=img)
        self.preview_label.image = img

    def start_processing(self):
        input_file = self.input_entry.get()
        output_file = self.output_entry.get()
        if input_file and output_file:
            try:
                start_time = float(self.start_time_entry.get())
                end_time = float(self.end_time_entry.get())
                if (
                    start_time >= end_time
                    or start_time < 0
                    or end_time > librosa.get_duration(y=self.y, sr=self.sr)
                ):
                    raise ValueError("Invalid time range")

                logging.info("Starting processing")
                self.start_button.config(state=tk.DISABLED)
                self.progress_var.set(0)
                self.progress_label.config(text="Processing...")

                y_selected = self.y[
                    int(start_time * self.sr) : int(end_time * self.sr)
                ]

                thread = threading.Thread(
                    target=self.create_video,
                    args=(
                        y_selected,
                        self.sr,
                        output_file,
                        start_time,
                        end_time,
                    ),
                )
                thread.start()

                self.master.after(100, self.check_thread, thread)
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}", exc_info=True)
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            logging.warning("Input or output file not selected")
            messagebox.showerror(
                "Error", "Please select both input and output files."
            )

    def check_thread(self, thread):
        if thread.is_alive():
            self.master.after(100, self.check_thread, thread)
        else:
            self.start_button.config(state=tk.NORMAL)



if __name__ == "__main__":
    root = tk.Tk()
    app = MusicVisualizer(root)
    root.mainloop()