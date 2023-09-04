import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from IPython.display import display as ipy_display
from matplotlib.animation import FuncAnimation

from sportslabkit.types.detection import Detection


def display_tracking_animation(detections, ground_truth_positions, predictions=None, width=360, height=240):
    def update(
        frame,
        detections,
        ground_truth_positions,
        predictions=None,
        width=360,
        height=240,
    ):
        ax.clear()

        # Set the plot limits
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Set the background color to white
        ax.set_facecolor("white")

        # Plot ground truth positions as filled blue boxes
        for gt_pos in ground_truth_positions[frame]:
            x, y, w, h = gt_pos
            gt_box = patches.Rectangle((x, y), w, h, facecolor="blue", alpha=0.5)
            ax.add_patch(gt_box)

        # Plot detections as hollow green boxes
        for detection in detections[frame]:
            x, y, w, h = detection.box
            detection_box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="green", facecolor="none")
            ax.add_patch(detection_box)

        # Plot predictions as hollow red boxes
        if predictions is not None:
            for prediction in predictions[frame]:
                x, y, w, h = prediction
                prediction_box = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none")
                ax.add_patch(prediction_box)

    if isinstance(detections[0], Detection):
        detections = [[detection] for detection in detections]
    if not isinstance(ground_truth_positions[0][0], list):
        ground_truth_positions = [[gt] for gt in ground_truth_positions]
    if predictions is not None and not isinstance(predictions[0][0], list):
        predictions = [[detection] for detection in predictions]

    fig, ax = plt.subplots(1, figsize=(12, 6))
    ani = FuncAnimation(
        fig,
        update,
        frames=len(detections),
        fargs=(detections, ground_truth_positions, predictions, width, height),
        interval=200,
    )

    # Convert the animation to an HTML element and display it in the notebook
    html = HTML(ani.to_jshtml())
    ipy_display(html)

    # Close the figure to prevent it from being displayed again
    plt.close()


def simulate_moving_object(
    num_frames: int,
    x: int = 0,
    y: int = 125,
    vx: int = 15,
    vy: int = 0,
    box_size: int = 25,
    class_id: int = 1,
):
    # Initial box position (center coordinates) and velocity
    box_position = np.array([x, y])
    box_velocity = np.array([vx, vy])  # Pixels per frame

    detections = []
    ground_truth_positions = []

    for _ in range(num_frames):
        # Generate score
        score = random.uniform(0.9, 1.0)

        # Scale the noise by the inverse of the score
        noise_scaling_factor = 10 / score

        # Update box position with noisy_velocity
        box_position += box_velocity

        # Calculate the bounding box (x, y, w, h) based on the box_position
        box = [
            box_position[0] - box_size // 2,
            box_position[1] - box_size // 2,
            box_size,
            box_size,
        ]

        # Add Gaussian noise to the box, correlated with the score
        noisy_box_position = box_position + np.random.normal(0, noise_scaling_factor, 2)
        noisy_box = [
            int(noisy_box_position[0] - box_size // 2),
            int(noisy_box_position[1] - box_size // 2),
            box_size,
            box_size,
        ]

        det: Detection = Detection(box=noisy_box, score=score, class_id=class_id)
        detections.append(det)

        ground_truth_positions.append(box)

    return detections, ground_truth_positions


def simulate_moving_objects(
    num_objects: int,
    num_frames: int,
    width: int = 360,
    height: int = 240,
    box_size: int = 25,
    frame_drop_rate: float = 0.1,
):
    all_detections: list[list[Detection]] = [[] for _ in range(num_frames)]
    all_gt_positions: list[list[list[int]]] = [[] for _ in range(num_frames)]

    for obj in range(num_objects):
        # randomly sample the object x and y coordinates
        x = random.randint(0, width - box_size)
        y = random.randint(0, height - box_size)

        # randomly sample the vx, vy velocities but based on the object's position
        # e.g if the object is in the top left corner, it's more likely to move down and right
        vx = random.randint(-5, 5)
        vy = random.randint(-5, 5)
        if x < width / 2:
            vx = random.randint(0, 5)
        else:
            vx = random.randint(-5, 0)
        if y < height / 2:
            vy = random.randint(0, 5)
        else:
            vy = random.randint(-5, 0)

        # Simulate the moving object
        detections, ground_truth_positions = simulate_moving_object(num_frames, x, y, vx, vy, box_size, class_id=obj)

        for frame in range(num_frames):
            # Get current frame detections and ground truth_positions
            det = detections[frame]
            gt_pos = ground_truth_positions[frame]

            all_gt_positions[frame].append(gt_pos)

            # Occasionally drop frames
            if random.random() < frame_drop_rate:
                continue

            all_detections[frame].append(det)
    return all_detections, all_gt_positions


def generate_frames(ground_truth_positions, width=360, height=240, box_size=25):
    frames = []

    for gt_positions in ground_truth_positions:
        # Create an empty frame (3D NumPy array) filled with zeros (black color)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw boxes for ground truth positions with the specified color intensity
        for idx, (x, y, w, h) in enumerate(gt_positions):
            box_color = np.array(
                [
                    idx**2 * 30 % 256,
                    (idx**2 * 50 + 50) % 256,
                    (idx**2 * 20 + 100) % 256,
                ],
                dtype=np.uint8,
            )
            noise = np.random.randint(-20, 20, (box_size, box_size, 3), dtype=np.int16)
            noisy_box_color = np.clip(box_color + noise, 0, 255).astype(np.uint8)

            # if the box is out of bounds, make it smaller
            # then fill the rectangle with the noisy box color
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, width), min(y + h, height)

            if y2 - y1 > 0 and x2 - x1 > 0:
                frame[y1:y2, x1:x2] = noisy_box_color[: y2 - y1, : x2 - x1]

        frames.append(frame)

    return frames


def display_generated_frames(frames):
    def update_plot(frame_number, frames, ax):
        ax.clear()
        im = ax.imshow(frames[frame_number], origin="upper", animated=True)
        return [im]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create an animation using the frames
    ani = FuncAnimation(
        fig,
        update_plot,
        frames=len(frames),
        fargs=(frames, ax),
        interval=200,
        blit=True,
    )

    # Convert the animation to an HTML element and display it in the notebook
    html = HTML(ani.to_jshtml())
    ipy_display(html)

    # Close the figure to prevent it from being displayed again
    plt.close()
