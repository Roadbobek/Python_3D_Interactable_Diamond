import pygame
import numpy as np
import math
import random
import os
import multiprocessing

# --- Initial Configuration ---
INITIAL_WINDOW_SIZE = 700 # Used as a base for scaling calculations AND max random window dimension
MIN_RANDOM_WINDOW_DIM = 10 # Minimum dimension for randomly sized windows
NUM_WINDOWS_TO_SPAWN = 48   # How many windows to create for organic coverage (Adjust as needed)

# --- Grid Configuration (REMOVED/COMMENTED OUT) ---
# GRID_ROWS_FOR_COVERAGE = 4
# GRID_COLS_FOR_COVERAGE = 5

BACKGROUND_COLOR = (30, 30, 30)  # Dark grey
DIAMOND_COLOR = (0, 200, 255)  # Bright cyan
EDGE_COLOR = (255, 255, 255)   # White
EDGE_THICKNESS = 2
FPS = 60

# Rotation Speed Settings
ROTATION_SPEED_X = 0.2
ROTATION_SPEED_Y = 0.35

# Sparkle Configuration
NUM_SPARKLES = 1000 # Base number of sparkles for a window of INITIAL_WINDOW_SIZE area
SCALE_SPARKLES_DENSITY = True
MIN_SPARKLES_WHEN_SCALING = 20
SMALL_WINDOW_AREA_THRESHOLD_FACTOR = 0.3
SMALL_WINDOW_SPARKLE_BOOST_FACTOR = 1.5

SPARKLE_COLORS = [
    (255, 255, 255),  # White
    (200, 200, 255),  # Light Blue/Purple
    (255, 255, 200)   # Light Yellow
]
SPARKLE_SPEED_RANGE = (0.2, 0.8)
SPARKLE_SIZE_RANGE = (1, 3)

# Perspective Projection Parameters
INITIAL_FOV = 400
CAMERA_DISTANCE = 7

# Overall scale for the final shape
SHAPE_SCALE = 2.5

# --- Ridged Octahedron Shape Parameters (Unit Size) ---
CORNER_Y_EXTENT = 1.0
CORNER_XZ_EXTENT = 1.0
RIDGE_PULL_FACTOR = 0.75

# --- Geometry (Computed once, shared by processes) ---
def create_ridged_octahedron_geometry():
    corners = [
        np.array([0, CORNER_Y_EXTENT, 0]), np.array([0, -CORNER_Y_EXTENT, 0]),
        np.array([CORNER_XZ_EXTENT, 0, 0]), np.array([-CORNER_XZ_EXTENT, 0, 0]),
        np.array([0, 0, CORNER_XZ_EXTENT]), np.array([0, 0, -CORNER_XZ_EXTENT])
    ]
    all_vertices = list(corners)
    ridge_vertex_map = {}
    octahedron_edges = [
        (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (4, 3), (3, 5), (5, 2)
    ]
    for v_idx1, v_idx2 in octahedron_edges:
        corner1, corner2 = corners[v_idx1], corners[v_idx2]
        mid_point = (corner1 + corner2) / 2.0
        ridge_vertex = mid_point * RIDGE_PULL_FACTOR
        edge_key = tuple(sorted((v_idx1, v_idx2)))
        if edge_key not in ridge_vertex_map:
            ridge_vertex_map[edge_key] = len(all_vertices)
            all_vertices.append(ridge_vertex)
    octahedron_original_faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 2, 5), (1, 5, 3), (1, 3, 4)
    ]
    final_faces = []
    for c_idx1, c_idx2, c_idx3 in octahedron_original_faces:
        m_idx12 = ridge_vertex_map[tuple(sorted((c_idx1, c_idx2)))]
        m_idx23 = ridge_vertex_map[tuple(sorted((c_idx2, c_idx3)))]
        m_idx31 = ridge_vertex_map[tuple(sorted((c_idx3, c_idx1)))]
        final_faces.extend([
            (c_idx1, m_idx12, m_idx31), (c_idx2, m_idx23, m_idx12),
            (c_idx3, m_idx31, m_idx23), (m_idx12, m_idx23, m_idx31)
        ])
    return [v * SHAPE_SCALE for v in all_vertices], final_faces

GEOMETRY_VERTICES, GEOMETRY_FACES = create_ridged_octahedron_geometry()


# --- Sparkle Class ---
class Sparkle:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self.x = random.randint(0, self.window_width)
        self.y = random.randint(0, self.window_height)
        self.size = random.randint(SPARKLE_SIZE_RANGE[0], SPARKLE_SIZE_RANGE[1])
        self.color = random.choice(SPARKLE_COLORS)
        self.speed_x = random.uniform(SPARKLE_SPEED_RANGE[0], SPARKLE_SPEED_RANGE[1]) * random.choice([-1, 1])
        self.speed_y = random.uniform(SPARKLE_SPEED_RANGE[0], SPARKLE_SPEED_RANGE[1]) * random.choice([-1, 1])

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        if self.x > self.window_width: self.x = 0
        elif self.x < 0: self.x = self.window_width
        if self.y > self.window_height: self.y = 0
        elif self.y < 0: self.y = self.window_height

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)

# --- Helper function to calculate target sparkle count ---
def calculate_target_sparkle_count(current_w, current_h,
                                   scale_density_flag, base_num_sparkles,
                                   initial_ref_w, initial_ref_h):
    if scale_density_flag:
        initial_area = float(initial_ref_w * initial_ref_h)
        if initial_area <= 0: initial_area = 1.0
        sparkle_density = base_num_sparkles / initial_area
        current_area = float(current_w * current_h)
        calculated_target = sparkle_density * current_area
        if current_area > 0 and current_area < (initial_area * SMALL_WINDOW_AREA_THRESHOLD_FACTOR):
            calculated_target *= SMALL_WINDOW_SPARKLE_BOOST_FACTOR
        return max(MIN_SPARKLES_WHEN_SCALING, int(calculated_target))
    else:
        return base_num_sparkles

# --- Helper function to adjust the size of the sparkles list ---
def adjust_sparkle_list_size(sparkles_list, target_count, current_w, current_h):
    current_len = len(sparkles_list)
    if current_len < target_count:
        for _ in range(target_count - current_len):
            sparkles_list.append(Sparkle(current_w, current_h))
    elif current_len > target_count:
        sparkles_list[:] = sparkles_list[:target_count]

# --- Rotation Matrices ---
def get_rotation_matrix_x(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def get_rotation_matrix_y(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# --- Main Application Logic for a Single Window ---
def run_single_window_app(window_width, window_height, window_x, window_y):
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    pygame.init()
    current_window_width = window_width
    current_window_height = window_height

    screen = pygame.display.set_mode((current_window_width, current_window_height), pygame.RESIZABLE)
    pygame.display.set_caption(f"Ridged Octahedron ({os.getpid()})")
    clock = pygame.time.Clock()

    angle_x, angle_y = random.uniform(0,360), random.uniform(0,360)

    sparkles = []
    initial_sparkle_target = calculate_target_sparkle_count(
        current_window_width, current_window_height,
        SCALE_SPARKLES_DENSITY, NUM_SPARKLES,
        INITIAL_WINDOW_SIZE, INITIAL_WINDOW_SIZE
    )
    adjust_sparkle_list_size(sparkles, initial_sparkle_target, current_window_width, current_window_height)

    running = True
    while running:
        old_win_w, old_win_h = current_window_width, current_window_height

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                current_window_width = max(200, new_w)
                current_window_height = max(200, new_h)
                screen = pygame.display.set_mode((current_window_width, current_window_height), pygame.RESIZABLE)

                if old_win_w > 0 and old_win_h > 0:
                    for sparkle in sparkles:
                        sparkle.x = sparkle.x * (current_window_width / old_win_w)
                        sparkle.y = sparkle.y * (current_window_height / old_win_h)
                        sparkle.x = max(0, min(sparkle.x, current_window_width))
                        sparkle.y = max(0, min(sparkle.y, current_window_height))
                        sparkle.window_width = current_window_width
                        sparkle.window_height = current_window_height

                if SCALE_SPARKLES_DENSITY:
                    target_s_count = calculate_target_sparkle_count(
                        current_window_width, current_window_height,
                        SCALE_SPARKLES_DENSITY, NUM_SPARKLES,
                        INITIAL_WINDOW_SIZE, INITIAL_WINDOW_SIZE
                    )
                    adjust_sparkle_list_size(sparkles, target_s_count, current_window_width, current_window_height)

        angle_x += ROTATION_SPEED_X
        angle_y += ROTATION_SPEED_Y

        rot_x = get_rotation_matrix_x(angle_x)
        rot_y = get_rotation_matrix_y(angle_y)
        rotation_matrix = np.dot(rot_y, rot_x)

        current_min_dim = min(current_window_width, current_window_height)
        fov_scale_factor = current_min_dim / INITIAL_WINDOW_SIZE if INITIAL_WINDOW_SIZE > 0 else 1.0
        dynamic_fov = max(50, INITIAL_FOV * fov_scale_factor)

        projected_points = []
        rotated_vertices_cache = []
        for vertex_3d in GEOMETRY_VERTICES:
            rotated_vertex = np.dot(rotation_matrix, vertex_3d)
            rotated_vertices_cache.append(rotated_vertex)
            z_val = rotated_vertex[2] + CAMERA_DISTANCE
            if z_val <= 0.1: z_val = 0.1
            scale_factor = dynamic_fov / z_val
            x_proj = rotated_vertex[0] * scale_factor + current_window_width / 2
            y_proj = -rotated_vertex[1] * scale_factor + current_window_height / 2
            projected_points.append((int(x_proj), int(y_proj)))

        faces_to_draw = []
        for i, face_indices in enumerate(GEOMETRY_FACES):
            v_rot = [rotated_vertices_cache[idx] for idx in face_indices]
            avg_z = (v_rot[0][2] + v_rot[1][2] + v_rot[2][2]) / 3.0
            faces_to_draw.append({'index': i, 'avg_z': avg_z, 'vertices_indices': face_indices})
        faces_to_draw.sort(key=lambda f: f['avg_z'], reverse=True)

        screen.fill(BACKGROUND_COLOR)
        for sparkle in sparkles:
            sparkle.update()
            sparkle.draw(screen)
        for face_data in faces_to_draw:
            point_list = [projected_points[idx] for idx in face_data['vertices_indices']]
            pygame.draw.polygon(screen, DIAMOND_COLOR, point_list)
            pygame.draw.lines(screen, EDGE_COLOR, True, point_list, EDGE_THICKNESS)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

# --- Launcher for Multiple Windows ---
def launch_multiple_windows():
    pygame.display.init()
    try:
        display_info = pygame.display.Info()
        primary_screen_width = display_info.current_w
        primary_screen_height = display_info.current_h
    except pygame.error:
        print("Warning: Could not get primary screen dimensions. Using defaults.")
        primary_screen_width = 1920 # Fallback
        primary_screen_height = 1080 # Fallback
    finally:
        pygame.display.quit()

    processes = []
    print(f"Attempting to launch {NUM_WINDOWS_TO_SPAWN} window(s) organically...")

    # Define how much of a window can be off-screen (e.g., 0.3 means up to 30%)
    max_offscreen_ratio = 0.3

    for i in range(NUM_WINDOWS_TO_SPAWN):
        # Random size for the square window
        rand_side = random.randint(MIN_RANDOM_WINDOW_DIM, INITIAL_WINDOW_SIZE)
        rand_w = rand_side
        rand_h = rand_side

        # Calculate spawn range for top-left (x, y) to allow partial off-screen
        # Minimum x: window can be off-screen to the left by max_offscreen_ratio * width
        min_x_spawn = int(-rand_w * max_offscreen_ratio)
        # Maximum x: window's left edge such that (1 - max_offscreen_ratio) of it is on screen if placed at screen edge
        max_x_spawn = int(primary_screen_width - rand_w * (1 - max_offscreen_ratio))

        min_y_spawn = int(-rand_h * max_offscreen_ratio)
        max_y_spawn = int(primary_screen_height - rand_h * (1 - max_offscreen_ratio))

        # Ensure spawn ranges are valid (max >= min)
        if max_x_spawn < min_x_spawn: # Window is wider than the allowed visible part of the screen
            rand_x = int((primary_screen_width - rand_w) / 2) # Center it
        else:
            rand_x = random.randint(min_x_spawn, max_x_spawn)

        if max_y_spawn < min_y_spawn: # Window is taller than the allowed visible part of the screen
            rand_y = int((primary_screen_height - rand_h) / 2) # Center it
        else:
            rand_y = random.randint(min_y_spawn, max_y_spawn)

        print(f"  Launching window {i+1}/{NUM_WINDOWS_TO_SPAWN}: Size=({rand_w}x{rand_h}) at ({rand_x},{rand_y})")

        p = multiprocessing.Process(target=run_single_window_app, args=(rand_w, rand_h, rand_x, rand_y))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    print("All windows closed.")

if __name__ == '__main__':
    # multiprocessing.freeze_support() # Usually only needed for frozen executables on Windows
    launch_multiple_windows()
