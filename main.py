import numpy as np
from typing import List
import tkinter as tk

SCREEN_SIZE = 100
SCREEN_CENTER = SCREEN_SIZE // 2
CUBE_SCALE = 20
PIXEL_SIZE = 6
FONT = ('monospace', 4)
FPS = 100
cube_angles = [0, 0, 0]
FACE_LETTERS = ['#', '/', '+', '?', '^', '`']
FACES = {
    0: np.array([[-1,-1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]),
    1: np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]),
    2: np.array([[-1, 1, -1], [-1, -1, -1], [1, -1, -1], [1, 1, -1]]),
    3: np.array([[1, 1, -1], [1, 1, 1], [1, -1, 1], [1, -1, -1]]),
    4: np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1]]),
    5: np.array([[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]])
}

def get_empty_screen() -> List[str]:
    """
    :return: 2D list of empty screen (represents pixels)
    """
    return [[' '] * SCREEN_SIZE for _ in range(SCREEN_SIZE)]

def get_cube_vertices() -> np.ndarray:
    return np.array([[-1, -1, -1],
                         [1, -1, -1],
                         [1, 1, -1],
                         [-1, 1, -1],
                         [-1, -1, 1],
                         [1, -1, 1],
                         [1, 1, 1],
                         [-1, 1, 1]])

def find_minimal_z(vertices: np.ndarray) -> float:
    """
    :param vertices: 3D vertices of cube
    :return: minimal z value of cube
    """
    return np.min(vertices[:, 2])

def is_vertices_neighbors(vertex_1: np.array, vertex_2: np.array) -> bool:
    """
    :param vertex_1:
    :param vertex_2:
    :return: return True if the vertices has common edge, else return False
    """
    # calc distance between vertices
    return np.linalg.norm(vertex_1 - vertex_2) >= 1.95 and np.linalg.norm(vertex_1 - vertex_2) <= 2.05


def vertice_3D_to_2D(vertex_3D: np.array) -> np.array:
    """
    :param vertex_3D:
    :return: the 2D coordinates of the vertex
    """
    x, y = vertex_3D[0]*CUBE_SCALE + SCREEN_CENTER, vertex_3D[1]*CUBE_SCALE + SCREEN_CENTER
    return np.array([x, y])

def add_vertices_to_screen(screen: np.ndarray, cube_vertices: np.ndarray, face_letter: str) -> None:
    """
    :param screen: 2D array that represents the screen
    :param cube_vertices: vertices to add to the screen
    :param face_letter: The letter that shows the face
    :return: None
    """
    for vertice in cube_vertices:
        vertice_2D = vertice_3D_to_2D(vertice)
        screen[int(vertice_2D[1])][int(vertice_2D[0])] = face_letter

def add_edge(screen: np.ndarray, vertice_1: np.array, vertice_2: np.array, face_letter: str) -> None:
    """
    Adds an edge to the screen
    :param screen: 2D array that represents the screen
    :param vertice_1: 3D coordinates of the first vertex
    :param vertice_2: 3D coordinates of the second vertex
    :param face_letter: letter that represents the face
    :return: None
    """
    vertice_1_2D = vertice_3D_to_2D(vertice_1)
    vertice_2_2D = vertice_3D_to_2D(vertice_2)
    x1, y1 = vertice_1_2D[0], vertice_1_2D[1]
    x2, y2 = vertice_2_2D[0], vertice_2_2D[1]
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx ** 2 + dy ** 2)

    if x1 >= x2:
        x, y = x1, y1
        is_left_point_lower = True
    else:
        x, y = x2, y2
        is_left_point_lower = False
    for _ in range(int(distance)):
        if is_left_point_lower:
            x += dx / distance
            y += dy / distance
        else:
            x -= dx / distance
            y -= dy / distance
        screen[int(y)][int(x)] = face_letter

def add_all_edges_to_screen(screen: np.ndarray, face_vertices: np.ndarray, face_letter: str) -> None:
    """
    Adds all edges of the face to the screen
    :param screen: 2D array that represents the screen
    :param face_vertices: vertices of the face
    :param face_letter:
    :return: None
    """
    for i in range(len(face_vertices)):
        for j in range(i+1, len(face_vertices)):
            if is_vertices_neighbors(face_vertices[i], face_vertices[j]):
                add_edge(screen, face_vertices[i], face_vertices[j], face_letter)

def add_face(screen: np.ndarray, vertex_1: np.array, vertex_2: np.array, vertex_3: np.array, vertex_4: np.array, face_letter: str) -> None:
    """
    Fills up a face of the cube with the letter
    :param screen:
    :param vertex_1:
    :param vertex_2:
    :param vertex_3:
    :param vertex_4:
    :param face_letter:
    :return: None
    """
    v1, v2, v3, v4 = vertice_3D_to_2D(vertex_1), vertice_3D_to_2D(vertex_2), vertice_3D_to_2D(vertex_3), vertice_3D_to_2D(vertex_4)
    min_x = min(v1[0], v2[0], v3[0], v4[0])
    min_y = min(v1[1], v2[1], v3[1], v4[1])
    max_x = max(v1[0], v2[0], v3[0], v4[0])
    max_y = max(v1[1], v2[1], v3[1], v4[1])
    for x in range(int(min_x), int(max_x)+1):
        first_in_row, last_in_row = -1, -1
        for y in range(int(min_y), int(max_y)+1):
            if screen[y][x] == face_letter:
                first_in_row = (x, y)
                break
        for y in range(int(max_y), int(min_y)-1, -1):
            if screen[y][x] == face_letter:
                last_in_row = (x, y)
                break
        if first_in_row != -1 and last_in_row != -1:
            for y in range(first_in_row[1], last_in_row[1]+1):
                screen[y][x] = face_letter


def rotate_cube(cube_vertices: np.ndarray, x_angle: float, y_angle: float, z_angle: float) -> np.ndarray:
    """
    Rotates the cube by the given angles
    :param cube_vertices: angles to rotate the cube by
    :param x_angle:
    :param y_angle:
    :param z_angle:
    :return: new rotated vertices of the cube
    """
    x_angle = np.radians(x_angle)
    y_angle = np.radians(y_angle)
    z_angle = np.radians(z_angle)
    x_rotation_matrix = np.array([[1, 0, 0],
                                  [0, np.cos(x_angle), -np.sin(x_angle)],
                                  [0, np.sin(x_angle), np.cos(x_angle)]])
    y_rotation_matrix = np.array([[np.cos(y_angle), 0, np.sin(y_angle)],
                                  [0, 1, 0],
                                  [-np.sin(y_angle), 0, np.cos(y_angle)]])
    z_rotation_matrix = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                                  [np.sin(z_angle), np.cos(z_angle), 0],
                                  [0, 0, 1]])
    rotation_matrix = np.matmul(np.matmul(x_rotation_matrix, y_rotation_matrix), z_rotation_matrix)
    return np.matmul(cube_vertices, rotation_matrix)

def get_screen_data() -> np.ndarray:
    """
    Calculate the next screen frame
    :return: updated screen frame
    """
    screen: List[str] = get_empty_screen()

    cube_vertices = get_cube_vertices()
    rotated_cube_vertices = rotate_cube(cube_vertices, cube_angles[0], cube_angles[1], cube_angles[2])
    minimal_z = find_minimal_z(rotated_cube_vertices)

    for face_index in range(len(FACES)):
        face_letter = FACE_LETTERS[face_index]
        cube_vertices = FACES[face_index]
        cube_vertices = rotate_cube(cube_vertices, cube_angles[0], cube_angles[1], cube_angles[2])
        if minimal_z in cube_vertices[:, 2]: # show only the faces that are in front of the camera
            continue
        add_vertices_to_screen(screen, cube_vertices, face_letter)
        add_all_edges_to_screen(screen, cube_vertices, face_letter)
        add_face(screen, cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3], face_letter)

    return screen



def update_screen(screen: List[str], canvas) -> None:
    """
    Update the screen window with the given screen frame
    :param screen:
    :param canvas:
    :return: None
    """
    canvas.delete("all")  # Clear the canvas
    for row, screen_row in enumerate(screen):
        for col, letter in enumerate(screen_row):
            x1 = col * PIXEL_SIZE
            y1 = row * PIXEL_SIZE
            canvas.create_text(x1 + PIXEL_SIZE / 2, y1 + PIXEL_SIZE / 2, text=letter, font=FONT, fill='white')

def repeat_update(root, canvas) -> None:
    """
    Repeatedly updates the screen with the next frame
    :param root: tkinter root window
    :param canvas: tkinter canvas
    :return: None
    """
    cube_angles[0] += 1
    cube_angles[1] += 2
    cube_angles[2] += 3
    update_screen(get_screen_data(), canvas)
    root.after(int(1000 // FPS), repeat_update, root, canvas)

def create_window() -> None:
    """
    Creates the tkinter window
    :return: None
    """
    root = tk.Tk()
    root.title("Rotating Cube")

    # Create a canvas to display the letters
    canvas = tk.Canvas(root, width=SCREEN_SIZE * PIXEL_SIZE, height=SCREEN_SIZE * PIXEL_SIZE, bg='black')
    canvas.pack()

    update_screen(get_empty_screen(), canvas)
    repeat_update(root, canvas)
    root.mainloop()

if __name__ == '__main__':
    create_window()
