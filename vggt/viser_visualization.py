import time
import argparse
import trimesh
import viser
import numpy as np
from pathlib import Path

def main():
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    # I have hardcoded your path here so you don't need to type it in the terminal.
    import argparse
    args = argparse.Namespace()
    
    # Your specific data path
    args.input = "/projects/standard/csci5561/shared/G11/my_datasets" 
    
    args.port = 8080
    args.no_share = False # Set to True if you want to block the public link

    # -------------------------------------------------------------------------
    # LOGIC
    # -------------------------------------------------------------------------
    dataset_path = Path(args.input)
    
    # Try to find point clouds. 
    # Since 'my_datasets' has subfolders (dishes, etc), we need to look deeper.
    # This logic aggregates ALL ply files from ALL sub-datasets into one list.
    ply_files = []
    if dataset_path.exists():
        # Look in direct outputs
        ply_files.extend(list((dataset_path / "outputs" / "pointclouds").glob("*.ply")))
        
        # Look in subfolders (e.g. dishes/outputs/pointclouds)
        for sub in dataset_path.iterdir():
            if sub.is_dir():
                sub_ply = sub / "outputs" / "pointclouds"
                if sub_ply.exists():
                    ply_files.extend(list(sub_ply.glob("*.ply")))
    
    if not ply_files:
        print(f"Error: No .ply files found in {dataset_path} or its subfolders.")
        print("Did you run the processing script (vggt_robust.py) first?")
        return

    print(f"---------------------------------------------------------")
    print(f"Found {len(ply_files)} point clouds.")
    print(f"Starting Viser Server...")
    print(f"---------------------------------------------------------")
    
    server = viser.ViserServer(port=args.port, share=not args.no_share)
    
    # Create display names: "DatasetName | filename.ply" for clarity
    file_options = {}
    for f in ply_files:
        dataset_name = f.parent.parent.parent.name 
        key = f"{dataset_name} | {f.name}"
        file_options[key] = f

    file_names = sorted(list(file_options.keys()))

    # State management for dynamic point cloud loading
    current_handle = None
    current_center = np.array([0.0, 0.0, 0.0])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        \"\"\"When new client connects, point camera at current object center.\"\"\"\n        client.camera.look_at = current_center

    def load_ply(name):
        \"\"\"Load and display selected point cloud file.\"\"\"\n        nonlocal current_handle, current_center
        path = file_options[name]
        
        print(f\"Loading: {name}...\")
        try:
            mesh = trimesh.load(path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return

        # Handle Scene objects (multiple geometries)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return

        # Extract points and colors from mesh
        points = mesh.vertices
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3]
        else:
            colors = np.ones_like(points) * 255

        # Remove previous point cloud if exists
        if current_handle is not None:
            current_handle.remove()

        # Add new point cloud to scene
        current_handle = server.scene.add_point_cloud(
            name="point_cloud",
            points=points,
            colors=colors,
            point_size=0.015,
        )
        
        # Update camera for all connected clients to focus on new object
        if len(points) > 0:
            center = np.mean(points, axis=0)
            current_center = center
            for client in server.get_clients().values():
                client.camera.look_at = center

    # Create GUI controls
    with server.gui.add_folder("Controls"):
        gui_file = server.gui.add_dropdown(
            "Select File",
            options=file_names,
            initial_value=file_names[-1]  # Default to highest quality filter
        )
        gui_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.015
        )

    @gui_file.on_update
    def _(_):
        \"\"\"Callback when dropdown selection changes.\"\"\"\n        load_ply(gui_file.value)

    @gui_size.on_update
    def _(_):
        \"\"\"Callback when point size slider changes.\"\"\"\n        if current_handle is not None:
            current_handle.point_size = gui_size.value

    # Load initial point cloud
    load_ply(file_names[-1])

    # Keep server running
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()