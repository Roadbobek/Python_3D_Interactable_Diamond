import pygame
import numpy as np
import math
import random

# --- Initial Configuration ---
INITIAL_WINDOW_SIZE = 700 # Used as a base for scaling calculations
BACKGROUND_COLOR = (30, 30, 30)  # Dark grey
DIAMOND_COLOR = (0, 200, 255)  # Bright cyan
EDGE_COLOR = (255, 255, 255)   # White
EDGE_THICKNESS = 2 # (Default: 2)
FPS = 60

# Rotation Speed Settings
ROTATION_SPEED_X = 0.2  # Degrees per frame for X-axis rotation (Default: 0.4)
ROTATION_SPEED_Y = 0.35  # Degrees per frame for Y-axis rotation (Default: 0.7)
# ROTATION_SPEED_Z = 0.3  # Optional: Degrees per frame for Z-axis rotation (if you enable Z rotation) (Default: 0.3)


# Sparkle Configuration
NUM_SPARKLES = 1000 # Base number of sparkles for the INITIAL_WINDOW_SIZE area (Default: 100)
SCALE_SPARKLES_DENSITY = True # SETTING: True to scale count with window, False for fixed NUM_SPARKLES

MIN_SPARKLES_WHEN_SCALING = 25      # New: Minimum number of sparkles if density scaling is ON
SMALL_WINDOW_AREA_THRESHOLD_FACTOR = 0.3 # New: If current area < 30% of initial area, it's "small"
SMALL_WINDOW_SPARKLE_BOOST_FACTOR = 1.5  # New: Boost sparkle count by 50% in "small" windows

SPARKLE_COLORS = [
    (255, 255, 255),  # White
    (200, 200, 255),  # Light Blue/Purple
    (255, 255, 200)   # Light Yellow
]
SPARKLE_SPEED_RANGE = (0.2, 0.8)
SPARKLE_SIZE_RANGE = (1, 3)

# Perspective Projection Parameters
INITIAL_FOV = 400  # Base FOV for the initial window size
CAMERA_DISTANCE = 7 # Might need adjustment based on final shape size

# Overall scale for the final shape
SHAPE_SCALE = 2.5 # This will scale the unit ridged octahedron (Default: 2.5)

# --- Ridged Octahedron Shape Parameters (Unit Size) ---
CORNER_Y_EXTENT = 1.0
CORNER_XZ_EXTENT = 1.0
RIDGE_PULL_FACTOR = 0.75


# --- Global Screen Dimensions (will be updated on resize) ---
SCREEN_WIDTH, SCREEN_HEIGHT = INITIAL_WINDOW_SIZE, INITIAL_WINDOW_SIZE


# --- Helper function to calculate target sparkle count ---
def calculate_target_sparkle_count(current_w, current_h,
                                   scale_density_flag, base_num_sparkles,
                                   initial_ref_w, initial_ref_h):
    if scale_density_flag:
        initial_area = float(initial_ref_w * initial_ref_h)
        if initial_area <= 0: initial_area = 1.0 # Avoid division by zero or issues with non-positive area

        sparkle_density_per_unit_area = base_num_sparkles / initial_area
        current_window_area = float(current_w * current_h)

        calculated_target_count = sparkle_density_per_unit_area * current_window_area

        # Apply boost for small windows
        small_window_threshold_area = initial_area * SMALL_WINDOW_AREA_THRESHOLD_FACTOR
        if current_window_area > 0 and current_window_area < small_window_threshold_area:
            calculated_target_count *= SMALL_WINDOW_SPARKLE_BOOST_FACTOR

        target_count = int(calculated_target_count)

        # Ensure a minimum number of sparkles when scaling is on
        final_target_count = max(MIN_SPARKLES_WHEN_SCALING, target_count)

        return final_target_count
    else:
        return base_num_sparkles

# --- Helper function to adjust the size of the sparkles list ---
def adjust_sparkle_list_size(sparkles_list, target_count):
    current_count = len(sparkles_list)
    if current_count < target_count:
        for _ in range(target_count - current_count):
            # New sparkles are initialized using global SCREEN_WIDTH/HEIGHT by Sparkle class
            sparkles_list.append(Sparkle())
    elif current_count > target_count:
        # Remove sparkles from the end
        sparkles_list[:] = sparkles_list[:target_count]


# --- Ridged Octahedron Geometry ---
def create_ridged_octahedron_geometry():
    # 1. Define the 6 main corner vertices of the base octahedron
    corners = [
        np.array([0, CORNER_Y_EXTENT, 0]),             # 0: Top
        np.array([0, -CORNER_Y_EXTENT, 0]),            # 1: Bottom
        np.array([CORNER_XZ_EXTENT, 0, 0]),            # 2: +X equatorial
        np.array([-CORNER_XZ_EXTENT, 0, 0]),           # 3: -X equatorial
        np.array([0, 0, CORNER_XZ_EXTENT]),            # 4: +Z equatorial
        np.array([0, 0, -CORNER_XZ_EXTENT])            # 5: -Z equatorial
    ]

    all_vertices = list(corners) # Start with corner vertices
    ridge_vertex_map = {} # To store (corner_idx1, corner_idx2) -> ridge_vertex_global_idx

    # 2. Define the 12 edges of the base octahedron and create ridge vertices
    octahedron_edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),  # Top pyramid edges
        (1, 2), (1, 3), (1, 4), (1, 5),  # Bottom pyramid edges (connecting to same equatorial)
        (2, 4), (4, 3), (3, 5), (5, 2)   # Equatorial belt edges
    ]

    for v_idx1, v_idx2 in octahedron_edges:
        corner1 = corners[v_idx1]
        corner2 = corners[v_idx2]
        mid_point = (corner1 + corner2) / 2.0
        ridge_vertex = mid_point * RIDGE_PULL_FACTOR

        edge_key = tuple(sorted((v_idx1, v_idx2)))
        if edge_key not in ridge_vertex_map:
            ridge_vertex_map[edge_key] = len(all_vertices)
            all_vertices.append(ridge_vertex)

    # 3. Define the 8 original faces of the octahedron
    octahedron_original_faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 2, 5), (1, 5, 3), (1, 3, 4)
    ]

    final_faces = []
    # 4. Subdivide each original octahedron face into 4 new faces
    for c_idx1, c_idx2, c_idx3 in octahedron_original_faces:
        m_idx12 = ridge_vertex_map[tuple(sorted((c_idx1, c_idx2)))]
        m_idx23 = ridge_vertex_map[tuple(sorted((c_idx2, c_idx3)))]
        m_idx31 = ridge_vertex_map[tuple(sorted((c_idx3, c_idx1)))]

        final_faces.append((c_idx1, m_idx12, m_idx31))
        final_faces.append((c_idx2, m_idx23, m_idx12))
        final_faces.append((c_idx3, m_idx31, m_idx23))
        final_faces.append((m_idx12, m_idx23, m_idx31))

    return all_vertices, final_faces

unscaled_vertices, unscaled_faces = create_ridged_octahedron_geometry()
diamond_vertices_orig = [v * SHAPE_SCALE for v in unscaled_vertices]
diamond_faces = unscaled_faces


# --- Sparkle Class ---
class Sparkle:
    def __init__(self):
        # Uses global SCREEN_WIDTH, SCREEN_HEIGHT for initial random placement
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        self.size = random.randint(SPARKLE_SIZE_RANGE[0], SPARKLE_SIZE_RANGE[1])
        self.color = random.choice(SPARKLE_COLORS)
        self.speed_x = random.uniform(SPARKLE_SPEED_RANGE[0], SPARKLE_SPEED_RANGE[1]) * random.choice([-1, 1])
        self.speed_y = random.uniform(SPARKLE_SPEED_RANGE[0], SPARKLE_SPEED_RANGE[1]) * random.choice([-1, 1])

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Wrap around screen edges using current global SCREEN_WIDTH, SCREEN_HEIGHT
        if self.x > SCREEN_WIDTH: self.x = 0
        elif self.x < 0: self.x = SCREEN_WIDTH
        if self.y > SCREEN_HEIGHT: self.y = 0
        elif self.y < 0: self.y = SCREEN_HEIGHT

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)

# --- Rotation Matrices ---
def get_rotation_matrix_x(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def get_rotation_matrix_y(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def get_rotation_matrix_z(angle_degrees):
    angle_rad = math.radians(angle_degrees)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# --- Main Application ---
def main():
    global SCREEN_WIDTH, SCREEN_HEIGHT

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Resizable Rotating Ridged Octahedron")
    clock = pygame.time.Clock()

    angle_x, angle_y = 0, 0
    # Removed local rotation_speed_x and rotation_speed_y variables here

    initial_screen_min_dim = min(SCREEN_WIDTH, SCREEN_HEIGHT)

    sparkles = []
    # Initial sparkle population based on the setting
    # At this point, SCREEN_WIDTH/HEIGHT are INITIAL_WINDOW_SIZE
    initial_sparkle_target_count = calculate_target_sparkle_count(
        SCREEN_WIDTH, SCREEN_HEIGHT,
        SCALE_SPARKLES_DENSITY, NUM_SPARKLES,
        INITIAL_WINDOW_SIZE, INITIAL_WINDOW_SIZE # Base density reference
    )
    for _ in range(initial_sparkle_target_count):
        sparkles.append(Sparkle())


    running = True
    while running:
        old_screen_width_for_scaling = SCREEN_WIDTH
        old_screen_height_for_scaling = SCREEN_HEIGHT

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                # Enforce minimum window size
                if new_w < 200: new_w = 200
                if new_h < 200: new_h = 200

                # Update global screen dimensions
                SCREEN_WIDTH, SCREEN_HEIGHT = new_w, new_h
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)

                # 1. Rescale positions of all existing sparkles first
                if old_screen_width_for_scaling > 0 and old_screen_height_for_scaling > 0:
                    for sparkle in sparkles:
                        sparkle.x = sparkle.x * (SCREEN_WIDTH / old_screen_width_for_scaling)
                        sparkle.y = sparkle.y * (SCREEN_HEIGHT / old_screen_height_for_scaling)
                        # Clamp them to new bounds
                        sparkle.x = max(0, min(sparkle.x, SCREEN_WIDTH))
                        sparkle.y = max(0, min(sparkle.y, SCREEN_HEIGHT))

                # 2. Adjust the number of sparkles if density scaling is enabled
                if SCALE_SPARKLES_DENSITY:
                    target_count = calculate_target_sparkle_count(
                        SCREEN_WIDTH, SCREEN_HEIGHT, # Current (new) dimensions
                        SCALE_SPARKLES_DENSITY, NUM_SPARKLES,
                        INITIAL_WINDOW_SIZE, INITIAL_WINDOW_SIZE # Base density reference
                    )
                    adjust_sparkle_list_size(sparkles, target_count)
                    # New sparkles added by adjust_sparkle_list_size are initialized
                    # using the new SCREEN_WIDTH/HEIGHT, so they are correctly placed.

        # Update rotation angles using the global settings
        angle_x += ROTATION_SPEED_X
        angle_y += ROTATION_SPEED_Y
        # angle_z += ROTATION_SPEED_Z # Uncomment if you want Z-axis rotation

        rot_x = get_rotation_matrix_x(angle_x)
        rot_y = get_rotation_matrix_y(angle_y)
        # rot_z = get_rotation_matrix_z(angle_z) # Uncomment for Z-axis rotation

        # Combine rotations
        rotation_matrix = np.dot(rot_y, rot_x)
        # rotation_matrix = np.dot(rot_z, rotation_matrix) # If using Z-axis rotation, apply it first or last depending on desired effect

        current_min_dim = min(SCREEN_WIDTH, SCREEN_HEIGHT)
        if initial_screen_min_dim > 0:
            fov_scale_factor = current_min_dim / initial_screen_min_dim
        else:
            fov_scale_factor = 1.0
        dynamic_fov = INITIAL_FOV * fov_scale_factor
        if dynamic_fov < 50: dynamic_fov = 50

        projected_points = []
        rotated_vertices = []
        for vertex_3d in diamond_vertices_orig:
            rotated_vertex = np.dot(rotation_matrix, vertex_3d)
            rotated_vertices.append(rotated_vertex)

            z_val = rotated_vertex[2] + CAMERA_DISTANCE
            if z_val <= 0.1: z_val = 0.1

            scale_factor = dynamic_fov / z_val
            x_proj = rotated_vertex[0] * scale_factor + SCREEN_WIDTH / 2
            y_proj = -rotated_vertex[1] * scale_factor + SCREEN_HEIGHT / 2
            projected_points.append((int(x_proj), int(y_proj)))

        faces_to_draw = []
        for i, face_indices in enumerate(diamond_faces):
            v_rot = [rotated_vertices[idx] for idx in face_indices]
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

if __name__ == '__main__':
    main()